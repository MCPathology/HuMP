"""
HyperPath (MICCAI'25) baseline adapted to IBMIL-style 1D bag interface.

Reference: Huang et al., "HyperPath: Hyperbolic Prompt Learning for Few-Shot
WSI Classification", MICCAI 2025.

Differences from the official `hypermil.py` (faithful enough to be a fair
baseline, but compatible with this codebase):
  * Drops CoOp-style PromptLearner. Class text prototypes are computed once at
    init time via plain CONCH text encoder over a small list of hand-written
    prompts per class, then mean-pooled. (Prompt ensembling is preserved.)
  * Drops external region input `x2` and `vis_concepts`. Region features are
    obtained by attention pooling over patches. Slide features come from a
    second attention pooling over region features.
  * Final classification logits come from a Euclidean linear head on the
    slide feature; the hyperbolic text-prototype distance drives the
    auxiliary InfoNCE + entailment-cone losses, mirroring HyperPath's spirit.

Forward signature matches `HypABMIL` (the existing `hyp_a` model in this
codebase) so it can plug into `train_tcga.py` with minimal dispatch changes:
    forward(bag_feats, table_feats=None, bag_label=None)
        -> (bag_prediction, pred, attention, aux_loss)

Author: response-letter helper for HuMP R1Q7 / R2Q1.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.manifolds.lorentz import Lorentz

# CONCH is required for the official text prototypes. We import lazily so
# that the file is still importable on machines without CONCH installed
# (random prototypes will be used as a development fallback).
try:
    from conch.open_clip_custom import (
        create_model_from_pretrained,
        tokenize,
        get_tokenizer,
    )
    _CONCH_OK = True
except Exception as _e:                                # pragma: no cover
    _CONCH_OK = False
    _CONCH_IMPORT_ERROR = _e


# ============================================================================
#  Task-specific prompts.
#
#  Style note: we follow the OFFICIAL HyperPath prompt convention
#  (Huang et al., MICCAI'25, `models/prompts.py`):
#    * Each class is a *list of bare-term synonyms / abbreviations*.
#    * The only active template is `"CLASSNAME"` (no natural-language wrappers).
#    * The CONCH text encoder is robust to terse medical terms, and HyperPath's
#      paper shows the verbose templates ('an H&E image of CLASSNAME.', ...) do
#      not help over the bare-term protocol.
#  Tasks that exist in the official HyperPath repo (BRCA, BRCA-HER2, NSCLC,
#  LUAD-EGFR) reuse the official prompts verbatim. Tasks the official repo does
#  NOT cover (CCRCC, Camelyon16) are written here in the same style.
# ============================================================================

def _expand_prompts(class_synonym_lists, templates=("CLASSNAME",)):
    """Replicate HyperPath's expand step: for each class, cross-product its
    synonyms with the template list and collect the resulting strings.
    Returns [n_classes][n_prompts] list of strings."""
    out = []
    for cls in class_synonym_lists:
        flat = []
        for syn in cls:
            for tpl in templates:
                flat.append(tpl.replace("CLASSNAME", syn))
        out.append(flat)
    return out


# ----- NSCLC (LUAD vs LUSC) ---- VERBATIM from HyperPath nsclc_prompts() ----
NSCLC_PROMPTS = _expand_prompts(
    [
        # class 0: LUAD
        [
            "adenocarcinoma",
            "lung adenocarcinoma",
            "adenocarcinoma of the lung",
            "luad",
        ],
        # class 1: LUSC
        [
            "squamous cell carcinoma",
            "lung squamous cell carcinoma",
            "squamous cell carcinoma of the lung",
            "lusc",
        ],
    ]
)

# ----- CCRCC subtype (2-class, ccRCC vs non-ccRCC) -----
# Not in the official HyperPath repo; written in the same bare-term style.
CCRCC_SUBTYPE_PROMPTS = _expand_prompts(
    [
        # class 0: clear-cell RCC
        [
            "clear cell renal cell carcinoma",
            "clear cell rcc",
            "ccrcc",
            "renal cell carcinoma with clear cytoplasm",
            "vhl-associated clear cell renal carcinoma",
        ],
        # class 1: non-clear-cell RCC
        [
            "papillary renal cell carcinoma",
            "chromophobe renal cell carcinoma",
            "non-clear-cell renal cell carcinoma",
            "non-clear-cell rcc",
            "renal cell carcinoma with eosinophilic granular cytoplasm",
        ],
    ]
)

# ----- CCRCC staging (4-class, AJCC Stage I / II / III / IV) -----
# Not in the official HyperPath repo; written in the same bare-term style.
# Each class lists synonyms covering Stage name, T stage, and a descriptor.
CCRCC_STAGE_PROMPTS = _expand_prompts(
    [
        # class 0: Stage I  (T1, organ-confined, <= 7 cm)
        [
            "stage i renal cell carcinoma",
            "stage 1 renal cell carcinoma",
            "t1 renal cell carcinoma",
            "early-stage renal cell carcinoma",
            "organ-confined renal cell carcinoma",
            "low-stage renal cell carcinoma",
        ],
        # class 1: Stage II  (T2, organ-confined, > 7 cm)
        [
            "stage ii renal cell carcinoma",
            "stage 2 renal cell carcinoma",
            "t2 renal cell carcinoma",
            "large organ-confined renal cell carcinoma",
            "intermediate-stage renal cell carcinoma",
        ],
        # class 2: Stage III  (T3 / N1: perinephric, vein, regional LN)
        [
            "stage iii renal cell carcinoma",
            "stage 3 renal cell carcinoma",
            "t3 renal cell carcinoma",
            "locally advanced renal cell carcinoma",
            "renal cell carcinoma with perinephric extension",
            "renal cell carcinoma with renal vein invasion",
        ],
        # class 3: Stage IV  (T4 / M1: beyond Gerota's fascia, distant mets)
        [
            "stage iv renal cell carcinoma",
            "stage 4 renal cell carcinoma",
            "t4 renal cell carcinoma",
            "advanced renal cell carcinoma",
            "metastatic renal cell carcinoma",
            "high-stage renal cell carcinoma",
        ],
    ]
)

# Default CCRCC_PROMPTS = staging (this is what HuMP reports). Dispatch by
# n_classes is performed in get_prompts() below.
CCRCC_PROMPTS = CCRCC_STAGE_PROMPTS

# ----- Camelyon16 (normal vs lymph-node metastasis) -----
# Not in the official HyperPath repo; written in the same bare-term style.
C16_PROMPTS = _expand_prompts(
    [
        # class 0: normal lymph node
        [
            "normal lymph node",
            "benign lymph node",
            "non-metastatic lymph node",
            "reactive lymph node",
            "lymph node without tumor",
        ],
        # class 1: lymph node metastasis
        [
            "lymph node metastasis",
            "metastatic carcinoma in lymph node",
            "metastatic breast cancer",
            "lymph node with tumor cells",
            "metastatic adenocarcinoma",
        ],
    ]
)


def get_prompts(task: str, n_classes: int = None):
    """Map a task tag (case-insensitive) to its [class][prompt] list.

    For CCRCC, dispatches between subtype (2-class) and staging (4-class)
    prompts based on `n_classes`. Explicit task tags 'CCRCC_STAGE' /
    'CCRCC_SUBTYPE' override the auto-dispatch.
    """
    key = task.strip().upper()
    if key in ("NSCLC", "TCGA", "TCGA-LUNG", "TCGA-NSCLC"):
        return NSCLC_PROMPTS
    if key in ("CCRCC_STAGE", "CCRCC_STAGING"):
        return CCRCC_STAGE_PROMPTS
    if key in ("CCRCC_SUBTYPE", "CCRCC_SUB"):
        return CCRCC_SUBTYPE_PROMPTS
    if key in ("CCRCC", "CPTAC-CCRCC", "KIRC"):
        # auto-dispatch: 2 -> subtype, 4 -> staging
        if n_classes == 2:
            return CCRCC_SUBTYPE_PROMPTS
        if n_classes == 4:
            return CCRCC_STAGE_PROMPTS
        # fall back to staging (HuMP's default CCRCC task)
        return CCRCC_STAGE_PROMPTS
    if key in ("C16", "CAMELYON16", "CAMELYON"):
        return C16_PROMPTS
    raise ValueError(
        f"hyperpath.get_prompts: no prompt set for task='{task}'. "
        f"Add a new prompt list to hyperpath.py."
    )


# ============================================================================
#  Building blocks
# ============================================================================

class _AttnAgg(nn.Module):
    """ABMIL-style attention aggregator. [1, N, D] -> ([1, 1, D], [1, N])."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: [1, N, D]
        a = self.gate(x).squeeze(-1)                 # [1, N]
        a = a.softmax(dim=-1)
        out = (a.unsqueeze(-1) * x).sum(dim=1, keepdim=True)  # [1, 1, D]
        return out, a


def _oxy_angle_origin(x_hyp, y_hyp, eps=1e-6):
    """Angle at the origin formed by the rays through hyperbolic points x and y.

    Computed from the spatial (non-time) components of the Lorentz embedding,
    which is equivalent to the angle between the corresponding tangent vectors
    at the origin.
    """
    xs = x_hyp[..., 1:]
    ys = y_hyp[..., 1:]
    xn = xs.norm(dim=-1, keepdim=True).clamp_min(eps)
    yn = ys.norm(dim=-1, keepdim=True).clamp_min(eps)
    if xs.shape != ys.shape:
        # broadcast pairwise: returns [..., len(x), len(y)] not used here;
        # we only call with broadcastable shapes.
        cos = (xs * ys).sum(dim=-1, keepdim=True) / (xn * yn)
    else:
        cos = (xs * ys).sum(dim=-1, keepdim=True) / (xn * yn)
    cos = cos.clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.arccos(cos)


def _half_aperture(x_hyp, curv, K=0.1, eps=1e-6):
    """Half-aperture of the entailment cone rooted at x (Lorentz model).

    psi(x) = arcsin( 2K / (sqrt(curv) * |x_space|) ),
    clipped to a valid arcsin domain.
    """
    xs = x_hyp[..., 1:]
    xn = xs.norm(dim=-1, keepdim=True).clamp_min(eps)
    arg = (2.0 * K / (curv.sqrt() * xn)).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.arcsin(arg)


# ============================================================================
#  HyperPath baseline model
# ============================================================================

class HyperPathMIL(nn.Module):
    """1D-bag adapter of HyperPath. See file docstring for the differences
    from the official implementation."""

    def __init__(
        self,
        input_dim: int = 512,
        n_classes: int = 2,
        task: str = "NSCLC",
        curv_init: float = 1.0,
        cone_K: float = 0.1,
        # CONCH loading
        conch_ckpt: str = "hf_hub:MahmoodLab/conch",
        hf_token: str = None,
        # Loss weights (mirroring the official defaults)
        w_contrastive: float = 1.0,
        w_entailment: float = 10.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.cone_K = cone_K
        self.w_contrastive = w_contrastive
        self.w_entailment = w_entailment

        D = input_dim

        # Projection heads (Euclidean -> shared latent).
        self.patch_proj  = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.region_proj = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.slide_proj  = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        self.text_proj   = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))

        # Hierarchical attention aggregation.
        self.patch2region = _AttnAgg(D)
        self.region2slide = _AttnAgg(D)

        # Hyperbolic manifold + learnable curvature.
        self.manifold = Lorentz(k=curv_init)
        self.curv = nn.Parameter(torch.tensor(float(curv_init)).log(), requires_grad=True)
        self._curv_minmax = (math.log(curv_init / 10.0), math.log(curv_init * 10.0))

        # Per-level lift scaling (kept <= 1 at inference).
        self.patch_alpha  = nn.Parameter(torch.tensor(D ** -0.5).log())
        self.region_alpha = nn.Parameter(torch.tensor(D ** -0.5).log())
        self.slide_alpha  = nn.Parameter(torch.tensor(D ** -0.5).log())
        self.text_alpha   = nn.Parameter(torch.tensor(D ** -0.5).log())

        # InfoNCE temperature.
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.07).log())

        # Classification head on top of slide feature (Euclidean).
        self.classifier = nn.Linear(D, n_classes)

        # Build per-class text prototypes once at init via CONCH.
        prompts = get_prompts(task, n_classes=n_classes)
        assert len(prompts) == n_classes, (
            f"HyperPathMIL: task='{task}' resolved to {len(prompts)} classes of "
            f"prompts, but n_classes={n_classes}. Either pass an explicit task "
            f"tag (e.g. 'CCRCC_STAGE' for 4-class staging, 'CCRCC_SUBTYPE' for "
            f"2-class), or extend the prompt list in hyperpath.py."
        )
        text_proto = self._build_text_prototypes(prompts, conch_ckpt, hf_token)
        # Make trainable so the projection layer can fine-tune them.
        self.register_buffer("text_prototypes", text_proto, persistent=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_text_prototypes(self, prompts, conch_ckpt, hf_token):
        """Encode all class prompts with CONCH text encoder, mean-pool per class.

        `conch_ckpt` can be either:
          * a HuggingFace URI like 'hf_hub:MahmoodLab/conch'  (needs hf_token), or
          * a local filesystem path to a CONCH .bin / .pt checkpoint.
        Local paths are detected automatically (no 'hf_hub:' prefix and file exists).
        """
        import os
        if not _CONCH_OK:
            print(
                f"[HyperPathMIL][WARN] CONCH not importable ({_CONCH_IMPORT_ERROR}); "
                "using random text prototypes. Install CONCH for a faithful baseline."
            )
            return torch.randn(len(prompts), self.input_dim) * 0.02

        # ---- decide local vs HF ----
        is_local = (
            conch_ckpt is not None
            and not conch_ckpt.startswith("hf_hub:")
            and os.path.exists(conch_ckpt)
        )
        kwargs = {}
        if is_local:
            print(f"[HyperPathMIL] Loading CONCH from local checkpoint: {conch_ckpt}")
        else:
            print(f"[HyperPathMIL] Loading CONCH from HF: {conch_ckpt}")
            if hf_token is not None:
                kwargs["hf_auth_token"] = hf_token
        conch_model, _ = create_model_from_pretrained(
            "conch_ViT-B-16", conch_ckpt, **kwargs
        )
        conch_model = conch_model.cuda().eval()
        tokenizer = get_tokenizer()
        protos = []
        for cls_prompts in prompts:
            toks = tokenize(texts=cls_prompts, tokenizer=tokenizer).cuda()
            feats = conch_model.encode_text(toks).float()  # [Np, D]
            protos.append(feats.mean(dim=0))               # [D]
        out = torch.stack(protos, dim=0).detach().cpu()    # [n_classes, D]
        del conch_model
        torch.cuda.empty_cache()
        return out

    # ------------------------------------------------------------------
    #  Hyperbolic lift / drop helpers (mirror `HypABMIL.tohyp/toeuc`)
    # ------------------------------------------------------------------
    def _to_hyp(self, x, alpha):
        x = x * alpha.exp()
        ones = torch.ones_like(x[..., 0:1])
        x_ext = torch.cat([ones, x], dim=-1)
        return self.manifold.expmap0(x_ext)

    def _to_euc(self, x_hyp):
        return self.manifold.logmap0(x_hyp)[..., 1:]

    # ------------------------------------------------------------------
    #  Loss components
    # ------------------------------------------------------------------
    def _entail_loss(self, parent_hyp, child_hyp, alpha: float = 1.0):
        """Entailment-cone loss: child should sit inside parent's cone."""
        if parent_hyp.dim() == child_hyp.dim() - 1:
            parent_hyp = parent_hyp.unsqueeze(-2)
        ang = _oxy_angle_origin(parent_hyp, child_hyp)
        ape = _half_aperture(parent_hyp, self.curv.exp(), K=self.cone_K)
        penalty = torch.clamp(ang - alpha * ape, min=0.0)
        return penalty.mean()

    def _infonce_distance(self, q_hyp, pos_hyp, neg_hyp, scale):
        """Hyperbolic-distance-based InfoNCE between q (one anchor) and
        positive / negative text prototypes. Distance is computed on the
        tangent-space coordinates (Euclidean norm in R^D), which is a
        monotone surrogate for the Lorentz geodesic distance."""
        q = self._to_euc(q_hyp).view(1, -1)            # [1, D]
        p = self._to_euc(pos_hyp).view(-1, q.size(-1)) # [Np, D]
        n = self._to_euc(neg_hyp).view(-1, q.size(-1)) # [Nn, D]
        sim_pos = -torch.cdist(q, p)                   # [1, Np]
        sim_neg = -torch.cdist(q, n)                   # [1, Nn]
        logits = torch.cat([sim_pos, sim_neg], dim=-1) * scale
        # Target index 0 (first column is a positive). For multiple positives
        # we use the mean over positive columns.
        log_prob = F.log_softmax(logits, dim=-1)
        n_pos = sim_pos.size(-1)
        return -log_prob[:, :n_pos].mean()

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, bag_feats, table_feats=None, bag_label=None):
        """
        bag_feats : [N, D] flat patch features (D == input_dim, e.g. 512 for CONCH).
        table_feats : unused; kept for interface compatibility with `hyp_a`.
        bag_label : [1, n_classes] one-hot (training) or None (eval).

        Returns:
            bag_prediction : [1, n_classes]  ← logits from classification head
            pred           : same as bag_prediction (kept for tuple symmetry)
            attention      : [1, N] patch attention from patch->region pooling
            aux_loss       : scalar tensor (0 if no label / eval)
        """
        # Clamp curvature into the safe range each forward.
        self.curv.data = torch.clamp(self.curv.data, *self._curv_minmax)
        _curv = self.curv.exp()
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        # ------------- shape normalization -------------
        if bag_feats.dim() == 2:
            bag_feats = bag_feats.unsqueeze(0)         # [1, N, D]
        N = bag_feats.size(1)

        # ------------- Euclidean projections + hierarchical aggregation
        patch_e  = self.patch_proj(bag_feats)              # [1, N, D]
        region_e, attn_p2r = self.patch2region(patch_e)    # [1, 1, D], [1, N]
        region_e = self.region_proj(region_e)
        slide_e, _         = self.region2slide(region_e)   # [1, 1, D]
        slide_e  = self.slide_proj(slide_e)
        slide_e_flat = slide_e.squeeze(1)                  # [1, D]

        # ------------- lift to Lorentz
        patch_h  = self._to_hyp(patch_e,  self.patch_alpha)        # [1, N, D+1]
        region_h = self._to_hyp(region_e, self.region_alpha)       # [1, 1, D+1]
        slide_h  = self._to_hyp(slide_e,  self.slide_alpha)        # [1, 1, D+1]

        # ------------- text prototypes
        text_e = self.text_proj(self.text_prototypes.to(bag_feats.device))  # [C, D]
        text_h = self._to_hyp(text_e, self.text_alpha)                       # [C, D+1]

        # ------------- classification head (Euclidean slide feat)
        logits = self.classifier(slide_e_flat)             # [1, n_classes]

        # ------------- auxiliary HyperPath losses
        if bag_label is not None and self.training:
            label_bool = bag_label.squeeze(0).bool()
            if label_bool.numel() != self.n_classes:
                # treat any non-1-hot input as class-index
                idx = bag_label.long().view(-1)[0]
                label_bool = torch.zeros(self.n_classes, dtype=torch.bool,
                                         device=bag_feats.device)
                label_bool[idx] = True

            pos_text_h = text_h[label_bool]                # [n_pos, D+1]
            neg_text_h = text_h[~label_bool]               # [n_neg, D+1]
            # degenerate single-class fallback
            if neg_text_h.numel() == 0:
                neg_text_h = pos_text_h

            # (1) Contrastive: slide should be close to positive class text
            contrastive = self._infonce_distance(
                slide_h.view(1, 1, -1), pos_text_h, neg_text_h, _scale
            )

            # (2) Entailment cones along the hierarchy:
            #       text(pos) ⊃ slide ⊃ region ⊃ patch
            ent_loss  = self._entail_loss(pos_text_h, slide_h.view(1, -1))
            ent_loss += self._entail_loss(slide_h.view(1, -1), region_h.view(1, -1))
            ent_loss += self._entail_loss(region_h.view(1, -1), patch_h.view(N, -1))
            ent_loss = ent_loss / 3.0

            aux_loss = self.w_contrastive * contrastive + self.w_entailment * ent_loss
        else:
            aux_loss = torch.zeros((), device=bag_feats.device)

        # `attention` shape requirements in train_tcga: prior models return either
        # [1, N] (abmil) or [K, N] (acmil). [1, N] is compatible with the .max/.min/
        # .mean stats reporting in train(), so we expose attn_p2r directly.
        return logits, logits, attn_p2r, aux_loss

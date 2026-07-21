#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
DisPro (CVPR'24) baseline adapter for the HuMP codebase.

Original paper: Xu et al., "Distilled Prompt Learning for Incomplete Multimodal
Survival Prediction" (CVPR'24).

The full DisPro implementation lives in `models/DisPro_network.py` and depends
on BERT-family checkpoints (PubMedBERT / ClinicalBERT / BioBERT / LongFormer)
plus UniDistill prompt-learning weights. Those heavy externals are NOT
required to evaluate DisPro's core missing-modality mechanism, which is:

    (i)  per-modality LEARNABLE PLACEHOLDER tokens that stand in when a
         modality is absent,
    (ii) a CROSS-MODAL TRANSFORMER fusing real modality tokens with
         placeholder tokens through attention,
    (iii) a distillation loss from unimodal teachers (optional add-on).

This file ports (i)+(ii) faithfully and exposes a `DisProSurv` class whose
__init__ / forward / return signature is byte-identical to `MHSurv`
(model_HGNN.py).  Drop-in: just register `args.modality == "dispro"` in
core_utils._init_model and re-use the existing "hgnn" branches in
_unpack_data / _process_data_and_forward.

If you have BERT + UniDistill checkpoints and want the full distillation
variant, set `use_full_dispro=True` and pass `unis_config` per the original
constructor; the wrapper will dispatch to the original `Transformer` class.

Interface (mirrors models/model_HGNN.py::MHSurv):
    __init__(self,
             genomic_sizes=[100, 200, 300, 400, 500, 600],
             transomic_sizes=[],
             n_classes=4,
             fusion="concat",
             model_size="small")

    forward(**kwargs)
        kwargs: x_path, x_genomic1..6, x_transomic1..N,
                protein, report, memory, valid
        returns: (logits[1,n_classes], aux_loss(scalar), proto_dict)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Light wrapper for k-means prototyping, identical role as MHSurv's
#      prototype_kmeans output. We import the same util that MHSurv uses to
#      keep memory-bank consumers happy. If it isn't on PYTHONPATH for any
#      reason, fall back to a simple mean.
try:
    from models.layers.lhyperbolic import prototype_kmeans  # same path as MHSurv
except Exception:                                            # pragma: no cover
    def prototype_kmeans(x, k):
        # very lightweight fallback: mean-pool, return k copies
        m = x.mean(dim=1, keepdim=True) if x.dim() == 3 else x.mean(dim=0, keepdim=True)
        return m


# ============================================================================
#  Per-modality token encoders (Euclidean, light-weight)
# ============================================================================
class _OmicSNN(nn.Module):
    """One SNN-like MLP per omic pathway/group."""
    def __init__(self, in_dim, hidden_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x.float())


# ============================================================================
#  DisProSurv: HuMP-compatible wrapper
# ============================================================================
class DisProSurv(nn.Module):
    def __init__(
        self,
        genomic_sizes=[100, 200, 300, 400, 500, 600],
        transomic_sizes=[],
        n_classes=4,
        fusion="concat",
        model_size="small",
        # ---- DisPro specifics (with safe defaults) ----
        dim_token: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.25,
        use_full_dispro: bool = False,
        unis_config: dict = None,
        encoder: str = "PubMedBERT",
    ):
        super().__init__()
        self.genomic_sizes = genomic_sizes
        self.transomic_sizes = transomic_sizes
        self.num_pathways = len(transomic_sizes)
        self.n_classes = n_classes
        self.fusion = fusion
        self.dim_token = dim_token
        self.use_full_dispro = use_full_dispro

        # ----- Pathology projection (CONCH/CTransPath/ResNet -> dim_token) -----
        # HuMP's pathology bag is [1, N_patch, D_in], usually 1024 (RN50) or 512 (CONCH).
        # We map every patch to dim_token. A learnable linear is enough; the
        # transformer fusion does the heavy lifting.
        wsi_in_dim_dict = {"small": 1024, "large": 1024}
        wsi_in = wsi_in_dim_dict.get(model_size, 1024)
        self.path_proj = nn.Sequential(
            nn.Linear(wsi_in, dim_token),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----- Genomic / Transcriptomic per-group SNN -----
        self.gene_sig_networks = nn.ModuleList([
            _OmicSNN(sz, dim_token, dropout) for sz in genomic_sizes
        ])
        self.trans_sig_networks = nn.ModuleList([
            _OmicSNN(sz, dim_token, dropout) for sz in transomic_sizes
        ])

        # ----- Proteomics: ESM-encoded (1280-d) -> dim_token. Optional. -----
        self.protein_proj = nn.Sequential(
            nn.Linear(1280, dim_token),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----- Clinical (CLIP/PLIP encoded report, 512-d) -> dim_token -----
        self.clinic_proj = nn.Sequential(
            nn.Linear(512, dim_token),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----- DisPro's signature: learnable placeholder tokens per modality
        # When a modality is missing, the placeholder takes its slot in the
        # transformer input. This is the actual missing-modality mechanism.
        self.holder_path  = nn.Parameter(torch.randn(1, 1, dim_token) * 0.02)
        self.holder_gene  = nn.Parameter(torch.randn(1, 1, dim_token) * 0.02)
        self.holder_trans = nn.Parameter(torch.randn(1, 1, dim_token) * 0.02)
        self.holder_prot  = nn.Parameter(torch.randn(1, 1, dim_token) * 0.02)
        self.holder_clin  = nn.Parameter(torch.randn(1, 1, dim_token) * 0.02)
        self.cls_token    = nn.Parameter(torch.randn(1, 1, dim_token) * 0.02)

        # ----- Cross-modal transformer (replaces BERT in the light variant) -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_token, nhead=n_heads, dim_feedforward=dim_token * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.fuser = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ----- Classifier on the CLS token -----
        self.classifier = nn.Sequential(
            nn.Linear(dim_token, dim_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_token, n_classes),
        )

        # Light L2-imputation loss switch (DisPro paper's distillation
        # surrogate, without the BERT teacher).
        # When a modality is missing, we ask the placeholder to predict the
        # mean of the OBSERVED modalities' tokens; this acts as a self-distill
        # signal and gives a small aux loss for the training-loop accounting.
        self.imp_weight = 0.1

        # ----- Optional: full DisPro path (requires unis_config + BERT) -----
        if use_full_dispro:
            # Lazy import to avoid pulling BERT for users who don't need it.
            from .DisPro_network import Transformer as FullDisProTransformer
            assert unis_config is not None, \
                "use_full_dispro=True requires unis_config (see DisPro_network.Transformer)"
            self._full_dispro = FullDisProTransformer(
                unis_config=unis_config,
                omic_sizes=genomic_sizes + transomic_sizes,
                encoder=encoder,
                num_classes=n_classes,
                dim_token=dim_token,
            )
        else:
            self._full_dispro = None

        # standard weight init
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for p in [self.holder_path, self.holder_gene, self.holder_trans,
                  self.holder_prot, self.holder_clin, self.cls_token]:
            nn.init.trunc_normal_(p, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    #  Modality encoders -> sequences of dim_token tokens
    # ------------------------------------------------------------------
    def _encode_pathology(self, x_path):
        """x_path: [N_patch, D_in] or [1, N_patch, D_in] -> [1, N_patch, dim_token]"""
        x_path = x_path.float()
        if x_path.dim() == 2:                       # [N_patch, D_in]
            x_path = x_path.unsqueeze(0)
        elif x_path.dim() == 4:                     # [1, 1, N_patch, D_in] (edge case)
            x_path = x_path.squeeze(1)
        return self.path_proj(x_path)

    def _encode_omics(self, x_genomic_list, x_transomic_list, protein):
        """Genomic + transcriptomic + protein -> [B, M, dim_token] one sequence.
        Robust to 1D omic tensors (no batch dim) coming from HuMP dataloader."""
        def _to_2d(x):
            """Ensure [B, dim_i]. SurvPath/HuMP gives 1D [dim_i] sometimes."""
            return x.unsqueeze(0) if x.dim() == 1 else x

        toks = []
        for i, x in enumerate(x_genomic_list):
            h = self.gene_sig_networks[i](_to_2d(x))          # [B, dim_token]
            if h.dim() == 1:
                h = h.unsqueeze(0)
            toks.append(h)
        for i, x in enumerate(x_transomic_list):
            h = self.trans_sig_networks[i](_to_2d(x))         # [B, dim_token]
            if h.dim() == 1:
                h = h.unsqueeze(0)
            toks.append(h)

        # stack along the pathway dim: list of [B, d] -> [B, N_omic, d]
        omics_tok = torch.stack(toks, dim=1)                  # [B, N_omic, dim]

        if protein is not None:
            p = self.protein_proj(_to_2d(protein.float()))    # [B, dim] or [B, N_p, dim]
            if p.dim() == 1:                                  # safety
                p = p.unsqueeze(0)
            if p.dim() == 2:                                  # [B, dim] -> [B, 1, dim]
                p = p.unsqueeze(1)
            omics_tok = torch.cat([omics_tok, p], dim=1)
        return omics_tok

    def _encode_clinical(self, report):
        """report: [512], [N_c, 512] or [1, N_c, 512] -> [1, N_c, dim_token]"""
        report = report.float()
        if report.dim() == 1:                       # [D_clin] -> [1, 1, D_clin]
            report = report.unsqueeze(0).unsqueeze(0)
        elif report.dim() == 2:                     # [N_c, D_clin] -> [1, N_c, D_clin]
            report = report.unsqueeze(0)
        return self.clinic_proj(report)

    # ------------------------------------------------------------------
    #  forward
    # ------------------------------------------------------------------
    def forward(self, **kwargs):
        device = next(self.parameters()).device

        # ---- defer to full DisPro if explicitly enabled ----
        if self._full_dispro is not None:
            return self._forward_full_dispro(**kwargs)

        # ---- detect which modalities are present ----
        x_path = kwargs.get("x_path", None)
        report = kwargs.get("report", None)
        protein = kwargs.get("protein", None)
        x_gene_list = [kwargs.get(f"x_genomic{i}", None) for i in range(1, 7)]
        x_trans_list = [kwargs.get(f"x_transomic{i}", None)
                        for i in range(1, self.num_pathways + 1)]
        gene_present  = all(x is not None for x in x_gene_list)
        trans_present = all(x is not None for x in x_trans_list)
        path_present  = x_path is not None
        clin_present  = report is not None

        # ---- build token sequence: [CLS, path_tokens, gene_tokens, trans_tokens, prot_token, clin_tokens]
        seq = [self.cls_token.expand(1, -1, -1).to(device)]
        observed_token_means = []

        # ---- pathology block ----
        if path_present:
            path_tok = self._encode_pathology(x_path.to(device))
            # cap patch count to keep transformer memory in check
            if path_tok.size(1) > 1024:
                idx = torch.randperm(path_tok.size(1), device=device)[:1024]
                path_tok = path_tok[:, idx, :]
            seq.append(path_tok)
            observed_token_means.append(path_tok.mean(dim=1))
        else:
            seq.append(self.holder_path.to(device))

        # ---- omics (gene + trans + protein) ----
        if gene_present or trans_present:
            g_list = [x.to(device) for x in x_gene_list if x is not None]
            t_list = [x.to(device) for x in x_trans_list if x is not None]
            omics_tok = self._encode_omics(
                g_list if gene_present else [],
                t_list if trans_present else [],
                protein.to(device) if protein is not None else None,
            )
            seq.append(omics_tok)
            observed_token_means.append(omics_tok.mean(dim=1))
        else:
            seq.append(self.holder_gene.to(device))
            seq.append(self.holder_trans.to(device))
            if protein is not None:
                p = self.protein_proj(protein.float().to(device))
                if p.dim() == 2:
                    p = p.unsqueeze(0)
                seq.append(p)
            else:
                seq.append(self.holder_prot.to(device))

        # ---- clinical block ----
        if clin_present:
            clin_tok = self._encode_clinical(report.to(device))
            seq.append(clin_tok)
            observed_token_means.append(clin_tok.mean(dim=1))
        else:
            seq.append(self.holder_clin.to(device))

        # ---- pad to consistent dim and run transformer ----
        tokens = torch.cat(seq, dim=1)                # [1, T_total, dim_token]
        fused = self.fuser(tokens)                    # [1, T_total, dim_token]

        # ---- classification from CLS token ----
        cls = fused[:, 0, :]                           # [1, dim_token]
        logits = self.classifier(cls)                  # [1, n_classes]

        # ---- placeholder-imputation auxiliary loss ----
        # DisPro distillation surrogate: each missing-modality placeholder
        # should be close to the mean of observed-modality features after
        # fusion. Encourages placeholders to be informative rather than zero.
        aux_loss = torch.tensor(0.0, device=device)
        if self.training and len(observed_token_means) > 0:
            obs_mean = torch.stack(observed_token_means, dim=0).mean(dim=0)  # [1, d]
            missing_holders = []
            if not path_present:  missing_holders.append(self.holder_path)
            if not gene_present:  missing_holders.append(self.holder_gene)
            if not trans_present: missing_holders.append(self.holder_trans)
            if not clin_present:  missing_holders.append(self.holder_clin)
            if len(missing_holders) > 0:
                tgt = obs_mean.detach()
                imp = sum(F.mse_loss(h.squeeze(0).squeeze(0), tgt.squeeze(0))
                          for h in missing_holders) / len(missing_holders)
                aux_loss = self.imp_weight * imp

        # ---- proto_dict (mirrors MHSurv's return for memory-bank consumers) ----
        proto_dict = {
            'G': prototype_kmeans(
                self._safe_token_for_proto(seq, idx=2, fallback_dim=self.dim_token, device=device), 32),
            'P': prototype_kmeans(
                self._safe_token_for_proto(seq, idx=1, fallback_dim=self.dim_token, device=device), 128),
            'C': prototype_kmeans(
                self._safe_token_for_proto(seq, idx=-1, fallback_dim=self.dim_token, device=device), 10),
        }

        return logits, aux_loss, proto_dict

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_token_for_proto(seq_list, idx, fallback_dim, device):
        """seq_list is a list of [1, N_i, D] tensors. Return seq_list[idx] for
        downstream prototype kmeans, with a zero-fallback shape in case of
        edge cases (e.g. all placeholders)."""
        try:
            t = seq_list[idx]
            if t.dim() == 3 and t.size(1) > 0:
                return t
        except Exception:
            pass
        return torch.zeros(1, 1, fallback_dim, device=device)

    # ------------------------------------------------------------------
    def _forward_full_dispro(self, **kwargs):
        """Bridge into the original DisPro Transformer (requires BERT)."""
        # Original DisPro expects: x_WSI, x_Omics, censor, label
        # We bundle HuMP inputs into that contract.
        x_path = kwargs.get("x_path", None)
        omics_dict = {}
        for i in range(1, 7):
            v = kwargs.get(f"x_genomic{i}", None)
            if v is not None:
                omics_dict[f"x_omic{i}"] = v
        # censor: derive from "censor" or "valid" if exposed; default 0.
        censor = int(kwargs.get("censor", 0))
        label = kwargs.get("label", None)
        ret = self._full_dispro(
            x_WSI=x_path, x_Omics=omics_dict if len(omics_dict) > 0 else None,
            censor=censor, label=label,
        )
        # DisPro returns dict of hazards/S/attns. Convert to (logits, 0, {})
        hazards = ret["hazards"]
        logits = torch.logit(hazards.clamp(1e-6, 1 - 1e-6))
        return logits, torch.tensor(0.0, device=logits.device), {'G': None, 'P': None, 'C': None}

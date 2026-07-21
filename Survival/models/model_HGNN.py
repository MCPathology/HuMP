import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.cross_attention import FeedForward
from models.layers.lhyperbolic import SCFusion, SAFusion, LHypFusion, HypABMIL, LCoMLP, HypLinear,  hyperbolic_entailment_completion_strict, prototype_kmeans # hyperbolic_entailment_loss_pairwise,
from models.manifolds.lorentz import * 
from .util import initialize_weights
from .util import SNN_Block, HypSNNBlock
from .layers.layers import *
try:
    import dhg  # noqa: F401
    from dhg.nn import HGNNPConv  # noqa: F401
except ImportError:
    dhg = None
    HGNNPConv = None
from collections import defaultdict

def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

class AttentionMIL(nn.Module):
    """
    Attention-based MIL pooling (no classifier)
    Input:
        x: [N, D]  - instance-level features
    Output:
        bag_feature: [1, D]  - aggregated bag-level feature
        att_weights: [N, 1]  - attention weights for each instance
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionMIL, self).__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Linear(hidden_dim, 1)  # attention score

    def forward(self, x):
        # x: [N, D]
        x = x.squeeze(0)
        A = self.attention_V(x)          # [N, H]
        A = self.attention_U(A)          # [N, 1]
        A = torch.softmax(A, dim=0)      # normalize over instances
        bag_feature = torch.sum(A * x, dim=0, keepdim=True)  # [1, D]
        # print(bag_feature.shape)
        return bag_feature, A
    
def hyperbolic_entailment_loss_pairwise(
    x: torch.Tensor,  # [1, N, d]
    y: torch.Tensor,  # [1, M, d]
    manifold=None,
    curv: float = 1.0,
    min_radius: float = 0.1,
    eps: float = 1e-6,
    clamp_val: float = 1 - 1e-6,
    temperature: float = 0.07,
) -> torch.Tensor:
    """CLIP-style contrastive loss, supports N != M."""
    x = x.squeeze(0)  # [N, d]
    y = y.squeeze(0)  # [M, d]

    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    logits = x @ y.T / temperature  # [N, M]

    # soft targets: each x_i uniformly matches all y_j, and vice versa
    loss_x = -torch.mean(F.log_softmax(logits, dim=1))   # average over [N, M]
    loss_y = -torch.mean(F.log_softmax(logits.T, dim=1)) # average over [M, N]

    return (loss_x + loss_y) / 2

class H2Surv(nn.Module):
    def __init__(self, genomic_sizes=[100, 200, 300, 400, 500, 600], transomic_sizes=[], n_classes=4, fusion="concat", model_size="small",graph_type="HGNN"):
        super(H2Surv, self).__init__()
        self.graph_type = graph_type
        self.genomic_sizes = genomic_sizes
        self.transomic_sizes = transomic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.num_pathways = len(transomic_sizes)
        
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024,256,  256], "large": [1024, 1024, 1024, 256]},
            "protein": {"small": [1280,256,  256], "large": [1024, 1024, 1024, 256]},
        }
        
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        p_hidden = self.size_dict["protein"][model_size]
        
        fc_ffpe = []
        for idx in range(len(hidden) - 1):
            fc_ffpe.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            
            fc_ffpe.append(nn.ReLU6())
            fc_ffpe.append(nn.Dropout(0.25))
        self.ffpe_fc = nn.Sequential(*fc_ffpe)
        
        # Genomic Embedding Network
        self.k = 1.0
        self.init_per_gene_hyp(genomic_sizes)
        self.init_per_trans_hyp(transomic_sizes)
        self.init_per_protein_hyp(p_hidden, sim=True)
        self.gene_hyp = LCoMLP(Lorentz(k=self.k))
        self.omic = HypLinear(Lorentz(k=self.k),256,256)
        self.kappa_g = KGGenerator(256, 256, 256)
        self.kappa_p = KPGenerator(256, 256, 256)
        # multi-modal fusion
        self.attention_fusion = GeneFusion(
            embedding_dim=256,
            num_heads = 4,
            num_pathways = 6
        )
        kappa = torch.tensor([0.1,0.01])
        # self.init_hyp(kappa)
        
        # Classification Layer
        self.feed_forward = FeedForward(256, dropout=0.25)
        self.layer_norm = nn.LayerNorm(256)
        self.p_norm = nn.LayerNorm(256)
        # self.mil = HypABMIL(Lorentz(k=self.k), 256, 256)
        self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1],hidden[-1]//2), nn.ReLU6()]
            )
        
        self.classifier = nn.Linear(hidden[-1]//2, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]
        
        x_genomic = [kwargs["x_genomic%d" % i].to("cuda") for i in range(1, 7)]
        x_transomic = [kwargs["x_transomic%d" % i].to("cuda") for i in range(1, self.num_pathways+1)]
        
        if kwargs["protein"] is not None:
            is_protein = True
            protein = kwargs["protein"].to("cuda")
            # print(protein)
            protein_features = self.protein_networks(protein)
        else:
            is_protein = False
            protein_features = None
            
        # gene 
        genomic = [self.gene_sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_genomic)]
        transomic = [self.trans_sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_transomic)] 
        genomics_features = torch.cat((torch.stack(genomic), torch.stack(transomic)), dim=0).unsqueeze(0)# torch.stack(genomic).unsqueeze(0)
        c = self.kappa_g(genomics_features, 6, 331)
        c = max(0.1,c)
        self.gene_hyp.setk(c)
        genomics_features, dist = self.gene_hyp(genomics_features, 6, protein_features)
        # genomics_features = 
        # genomics_features = self.omic(genomics_features, 'euc')
        
        g_num = 6
        t_num = 331 
        pr_num = 100
        g_total = genomics_features.shape[1]
        
        token_cross = genomics_features
        token_cross = self.feed_forward(token_cross)
        token_cross = self.layer_norm(token_cross)
       
        gene_embed = token_cross[:, :g_total, :]
        gene_embed = torch.mean(gene_embed, dim=1)
        
        fusion = self.mm(gene_embed)  
        
        logits = self.classifier(fusion)  # [1, n_classes]
        
        return logits,  dist# torch.tensor([0.0]).to('cuda')#
        
    def init_hyp(self, kappa):
        self.manifolds = []
        for k in range(kappa.shape[0]):
            # print(kappa[k])
            self.manifolds.append(Lorentz(k=kappa[k]))
        self.hyp_fusion = []
        for manifold in self.manifolds:
            self.hyp_fusion.append(LHypFusion(manifold, dim=256).to('cuda'))
        
        self.gating = Gating(len(self.manifolds)+1).to('cuda')

    def init_per_gene_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.gene_sig_networks = nn.ModuleList(sig_networks)
        
    def init_per_gene_hyp(self, omic_sizes, k=3.0):
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = HypSNNBlock(Lorentz(k=k), in_dim=input_dim)
            sig_networks.append(fc_omic)
        self.gene_sig_networks = nn.ModuleList(sig_networks)
    

    def init_per_trans_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.trans_sig_networks = nn.ModuleList(sig_networks)

    def init_per_trans_hyp(self, omic_sizes, k=3.0):
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = HypSNNBlock(Lorentz(k=k), in_dim=input_dim)
            sig_networks.append(fc_omic)
        self.trans_sig_networks = nn.ModuleList(sig_networks)
    
    def init_per_protein_model(self, p_hidden):
        protein_networks = []
        for idx in range(2):
            protein_networks.append(nn.Linear(p_hidden[idx], p_hidden[idx + 1]))
            protein_networks.append(nn.ReLU6())
            protein_networks.append(nn.Dropout(0.25))
        self.protein_networks = nn.Sequential(*protein_networks)
        
    def init_per_protein_hyp(self, hidden, sim=True, k=0.1):
        if sim:
            protein_networks = [HypSNNBlock(Lorentz(k=k), in_dim=383)]
        else:
            protein_networks = [HypSNNBlock(Lorentz(k=k), in_dim=1280)]
        self.protein_networks = nn.Sequential(*protein_networks)
        
        
class MHSurv(nn.Module):
    def __init__(self, genomic_sizes=[100, 200, 300, 400, 500, 600], transomic_sizes=[], n_classes=4, fusion="concat", model_size="small",graph_type="HGNN"):
        super(MHSurv, self).__init__()
        self.graph_type = graph_type
        self.genomic_sizes = genomic_sizes
        self.transomic_sizes = transomic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.num_pathways = len(transomic_sizes)
        
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024,256,  256], "large": [1024, 1024, 1024, 256]},
            "protein": {"small": [1280,256,  256], "large": [1024, 1024, 1024, 256]},
        }
        
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        p_hidden = self.size_dict["protein"][model_size]
        
        fc_ffpe = []
        for idx in range(len(hidden) - 1):
            fc_ffpe.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            
            fc_ffpe.append(nn.ReLU6())
            fc_ffpe.append(nn.Dropout(0.25))
        self.ffpe_fc = nn.Sequential(*fc_ffpe)
        
        # Genomic Embedding Network
        self.k = 1.0
        self.manifold = Lorentz(k=self.k)
        self.init_per_gene_model(genomic_sizes)
        self.init_per_trans_model(transomic_sizes)
        self.init_per_protein_model(p_hidden)
        self.gene_agg = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256)
            )
        self.trans_agg = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256)
            )
        self.clinic_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256)
            )

        # multi-modal fusion
        # self.attention_fusion = SCFusion(dim=256) 
        self.attention_fusion = SAFusion(manifold=self.manifold, dim=256) # HierarchicalBiFusion(manifold=self.manifold, dim=256)# LHypFusion( 
        #     Lorentz(k=self.k),
        #     dim=256
        #  )
        # self.init_hyp(kappa)
        
        # Classification Layer
        self.feed_forward = FeedForward(256, dropout=0.25)
        self.layer_norm = nn.LayerNorm(256)
        self.p_norm = nn.LayerNorm(256)
        # self.mil = AttentionMIL(256, 256)
        self.mil = HypABMIL(Lorentz(k=self.k), 256, 256)
        self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1]*3,hidden[-1]), nn.ReLU6()]
            )
        
        self.classifier = nn.Linear(hidden[-1], self.n_classes)

        # ----------------------------------------------------------------
        # Per-modality running prototypes for missing-modality imputation.
        # Updated via EMA in Branch D (full modality, train phase) and
        # consumed by the n_missing>=1 chain-HGS branches so step-1 has a
        # meaningful 2nd anchor instead of zero-tensor fallback.
        # NOT a Parameter / buffer: we want a plain detached tensor that
        # survives across batches but is not part of state_dict autosave
        # logic (avoid shape-mismatch issues across cohorts).
        # ----------------------------------------------------------------
        self._global_prototype = {'P': None, 'G': None, 'C': None}
        self._proto_ema = 0.9
        # set externally from core_utils.py before each forward; '' = full
        self._test_missing_mode = ''

        self.apply(initialize_weights)

    # ------------------------------------------------------------------
    # Helper: return the running prototype for one modality, falling back
    # to the per-call `prototype` kwarg, finally to None.  The returned
    # tensor is detached (no grad), on whatever device it was stashed on.
    # ------------------------------------------------------------------
    def _proto_for(self, modality, fallback=None):
        gp = getattr(self, '_global_prototype', None) or {}
        p = gp.get(modality, None)
        if p is not None:
            return p.to('cuda')
        if fallback is not None and isinstance(fallback, torch.Tensor):
            return fallback.to('cuda')
        return None

    # ------------------------------------------------------------------
    # EMA update of running prototypes.  Called from Branch D after the
    # k-means prototypes for the current batch are computed.  We detach
    # and clone so the EMA tensor never holds the autograd graph.
    # ------------------------------------------------------------------
    def _update_prototype(self, key, new_proto):
        if new_proto is None:
            return
        new_p = new_proto.detach()
        gp = self._global_prototype
        cur = gp.get(key, None)
        if cur is None or cur.shape != new_p.shape:
            gp[key] = new_p.clone()
        else:
            ema = self._proto_ema
            gp[key] = (ema * cur + (1.0 - ema) * new_p).detach()

    # ------------------------------------------------------------------
    # Partial-training "skip" forward.  Called when at least one modality
    # is dropped during training and --train_skip_imputation is enabled.
    #
    # Design (Strategy A, modality dropout):
    #   * encoders run only for OBSERVED modalities
    #   * attention_fusion is invoked over the observed token cat (1 or 2
    #     modalities); missing modality is NOT replaced by anything
    #   * cross-modal entailment losses computed only for observed pairs
    #     -- this implements the paper-faithful design that hierarchy is
    #        only constrained when both endpoints are real
    #   * final mm head still consumes a 3-slot vector [G | P | C] with
    #     missing slots = zero, so the classifier head shape is invariant
    #   * EMA prototypes update only for observed modalities
    # ------------------------------------------------------------------
    def _forward_skip_imputation(self, kwargs, p_obs, o_obs, c_obs):
        g_loss = t_loss = torch.tensor(0.0, device='cuda')
        g2p_loss = p2c_loss = torch.tensor(0.0, device='cuda')

        pathology_features = None
        genomics_features = None
        clinic_features = None
        pathology_proto = genomics_proto = clinic_proto = None

        # ---- (1) Encoders run only for observed modalities ----
        if p_obs:
            pathology_features = self.ffpe_fc(kwargs['x_path']).to('cuda')
        if o_obs:
            # process_multiomics returns (token, g_loss, t_loss) and consumes
            # x_genomic1..6, x_transomic*, protein internally; safe because the
            # missing-mode knock-out already zeroed them if O was dropped.
            genomics_features, g_loss, t_loss = self.process_multiomics(kwargs)
        if c_obs:
            clinic_features = self.clinic_fc(kwargs['report']).unsqueeze(0)

        # ---- (2) Pairwise entailment losses (only when both endpoints exist) ----
        if p_obs and o_obs:
            g2p_loss = hyperbolic_entailment_loss_pairwise(
                pathology_features, genomics_features, self.manifold, self.k)
        if p_obs and c_obs:
            p2c_loss = hyperbolic_entailment_loss_pairwise(
                clinic_features, pathology_features, self.manifold, self.k)

        # ---- (3) Fusion over observed modalities ----
        slot_G = torch.zeros(1, 256, device='cuda')
        slot_P = torch.zeros(1, 256, device='cuda')
        slot_C = torch.zeros(1, 256, device='cuda')

        obs_tokens = []   # in canonical G→P→C order for stable splitting
        obs_keys = []
        if o_obs:
            obs_tokens.append(genomics_features); obs_keys.append('G')
        if p_obs:
            obs_tokens.append(pathology_features); obs_keys.append('P')
        if c_obs:
            obs_tokens.append(clinic_features); obs_keys.append('C')

        if len(obs_tokens) == 1:
            # Single-modal case: skip self-attention fusion, go straight to
            # feed_forward + layer_norm.  Equivalent to "no cross-modal mixing".
            token_cross = self.feed_forward(obs_tokens[0])
            token_cross = self.layer_norm(token_cross)
            pooled = token_cross.mean(dim=1)        # [1, 256]
            if obs_keys[0] == 'G':   slot_G = pooled
            elif obs_keys[0] == 'P': slot_P = pooled
            else:                    slot_C = pooled

        elif len(obs_tokens) == 2:
            # 2 observed modalities: SAFusion internally concatenates two
            # streams along token dim and self-attends -- symmetric in the
            # two inputs, so we can pass them positionally regardless of
            # which modalities they actually are.  We then split outputs
            # back by recorded token counts.
            token_cross = self.attention_fusion(obs_tokens[0], obs_tokens[1], None)
            token_cross = self.feed_forward(token_cross)
            token_cross = self.layer_norm(token_cross)
            offset = 0
            for tok, key in zip(obs_tokens, obs_keys):
                n = tok.shape[1]
                pooled = token_cross[:, offset:offset + n, :].mean(dim=1)
                if key == 'G':   slot_G = pooled
                elif key == 'P': slot_P = pooled
                else:            slot_C = pooled
                offset += n
        else:
            # Defensive: caller guarantees obs_count in {1, 2}; full modality
            # would not reach this method.
            raise RuntimeError("skip_imputation reached with obs_count={}".format(
                len(obs_tokens)))

        # ---- (4) Final classifier head ----
        token_mm = torch.cat((slot_G, slot_P, slot_C), dim=-1)
        fusion = self.mm(token_mm)
        logits = self.classifier(fusion)

        # ---- (5) Per-modality EMA prototype update (observed only) ----
        if self.training:
            if pathology_features is not None:
                pathology_proto = prototype_kmeans(pathology_features, 128)
                self._update_prototype('P', pathology_proto)
            if genomics_features is not None:
                genomics_proto = prototype_kmeans(genomics_features, 32)
                self._update_prototype('G', genomics_proto)
            if clinic_features is not None:
                clinic_proto = prototype_kmeans(clinic_features, 10)
                self._update_prototype('C', clinic_proto)

        aux_loss = (g_loss + t_loss) + (p2c_loss + g2p_loss)
        return logits, aux_loss, {
            'G': genomics_proto, 'P': pathology_proto, 'C': clinic_proto,
        }

    def forward(self, **kwargs):
        # if kwargs["x_path"] is None:
        #     proto = kwargs["prototype"].to('cuda')
        #     genomics_features, g_loss, t_loss = self.process_multiomics(kwargs)
        #     x_table = kwargs["report"]
        #     clinic_features = self.clinic_fc(x_table).unsqueeze(0)

        #     embed_dict = {'P':proto, 'G':genomics_features, 'C':clinic_features}
        #     pathology_features = proto# hyperbolic_entailment_completion_strict(embed_dict, 'P', self.manifold)
        #     genomics_proto = pathology_proto = clinic_proto = None
        #     g2p_loss = p2c_loss = torch.tensor(0.0, device="cuda")
            
        # elif kwargs["x_genomic1"] is None:
        #     proto = kwargs["prototype"].to('cuda')
        #     x_path = kwargs["x_path"]
        #     pathology_features = self.ffpe_fc(x_path).to("cuda")
            
        #     x_table = kwargs["report"]
        #     clinic_features = self.clinic_fc(x_table).unsqueeze(0)
        #     embed_dict = {'P': pathology_features, 'G': proto, 'C': clinic_features}
        #     genomics_features = proto# hyperbolic_entailment_completion_strict(embed_dict, 'G', self.manifold)
        #     g_loss = t_loss = torch.tensor(0.0, device="cuda")
        #     genomics_proto = pathology_proto = clinic_proto = None
        #     g2p_loss = p2c_loss = torch.tensor(0.0, device="cuda")

        # elif kwargs["report"] is None:
        #     proto = kwargs["prototype"].to('cuda')
        #     genomics_features, g_loss, t_loss = self.process_multiomics(kwargs)
        #     x_path = kwargs["x_path"]
        #     pathology_features = self.ffpe_fc(x_path).to("cuda")

        #     embed_dict = {'P': pathology_features, 'G': genomics_features, 'C': proto}
        #     clinic_features = proto# hyperbolic_entailment_completion_strict(embed_dict, 'C', self.manifold)
        #     genomics_proto = pathology_proto = clinic_proto = None
        #     g2p_loss = p2c_loss = torch.tensor(0.0, device="cuda")

        # ================================================================
        #   Missing-modality dispatch (single + double-missing supported)
        #
        #   `missing_mode` kwarg controls which modalities to drop at
        #   inference time, even if data is provided:
        #       ''      / None   -> full modality
        #       'P'              -> pathology missing
        #       'O'              -> omics missing (all of g/t/protein)
        #       'C'              -> clinical missing
        #       'PO' / 'OP'      -> pathology + omics missing (only C)
        #       'PC' / 'CP'      -> pathology + clinical missing (only O)
        #       'OC' / 'CO'      -> omics + clinical missing (only P)
        #
        #   When --missing_mode is not passed, behaviour falls back to
        #   detecting None inputs (original single-missing pathway).
        # ================================================================
        missing_mode = (kwargs.get('missing_mode', '') or '').upper()
        if 'P' in missing_mode:
            kwargs['x_path'] = None
        if 'O' in missing_mode:
            for _i in range(1, 7):
                kwargs['x_genomic%d' % _i] = None
            for _i in range(1, self.num_pathways + 1):
                kwargs['x_transomic%d' % _i] = None
            kwargs['protein'] = None
        if 'C' in missing_mode:
            kwargs['report'] = None

        p_obs = kwargs.get('x_path') is not None
        o_obs = kwargs.get('x_genomic1') is not None
        c_obs = kwargs.get('report') is not None
        n_missing = int(not p_obs) + int(not o_obs) + int(not c_obs)

        # initialise default losses / protos so all branches return them
        g_loss = t_loss = torch.tensor(0.0, device='cuda')
        g2p_loss = p2c_loss = torch.tensor(0.0, device='cuda')
        genomics_proto = pathology_proto = clinic_proto = None

        # =================================================================
        #   Partial-training SKIP path:
        #     when --train_skip_imputation is on and at least one modality is
        #     missing, we DO NOT impute it.  The missing modality is removed
        #     from fusion, from pairwise entailment losses, and from the
        #     final mm head (zero-vector slot).  Inference always sets
        #     skip_imputation=False so HGS imputation still runs there.
        # =================================================================
        skip_imp = bool(kwargs.get('skip_imputation', False))
        if skip_imp and n_missing >= 1 and n_missing <= 2:
            return self._forward_skip_imputation(kwargs, p_obs, o_obs, c_obs)

        memory = kwargs.get('memory', None)
        # Per-modality prototype sources, in priority order:
        #   (1) self._global_prototype[m]  – EMA tracked across batches
        #   (2) kwargs['prototype']        – per-call fallback (legacy)
        #   (3) None                       – HGS zero-tensor fallback (degraded)
        # We do NOT share one tensor across modalities anymore, since their
        # token counts (P:128, G:32, C:10) and semantic levels differ.
        kw_proto = kwargs.get('prototype', None)
        if kw_proto is not None and isinstance(kw_proto, torch.Tensor):
            kw_proto = kw_proto.to('cuda')

        p_proto = self._proto_for('P', fallback=kw_proto)
        g_proto = self._proto_for('G', fallback=kw_proto)
        c_proto = self._proto_for('C', fallback=kw_proto)

        # -----------------------------------------------------------------
        #   Branch A:  n_missing == 1   (single-modality completion via HGS)
        #   For each case we keep two real anchors and call HGS on the
        #   missing one: cone defined by parent + child.
        # -----------------------------------------------------------------
        if n_missing == 1 and not p_obs:
            # P missing, O+C observed
            genomics_features, g_loss, t_loss = self.process_multiomics(kwargs)
            clinic_features = self.clinic_fc(kwargs['report']).unsqueeze(0)
            embed_dict = {'P': p_proto, 'G': genomics_features, 'C': clinic_features}
            pathology_features = hyperbolic_entailment_completion_strict(
                embed_dict, 'P', self.manifold)

        elif n_missing == 1 and not o_obs:
            # O missing, P+C observed
            pathology_features = self.ffpe_fc(kwargs['x_path']).to('cuda')
            clinic_features = self.clinic_fc(kwargs['report']).unsqueeze(0)
            embed_dict = {'P': pathology_features, 'G': g_proto, 'C': clinic_features}
            genomics_features = hyperbolic_entailment_completion_strict(
                embed_dict, 'G', self.manifold)

        elif n_missing == 1 and not c_obs:
            # C missing, P+O observed
            genomics_features, g_loss, t_loss = self.process_multiomics(kwargs)
            pathology_features = self.ffpe_fc(kwargs['x_path']).to('cuda')
            embed_dict = {'P': pathology_features, 'G': genomics_features, 'C': c_proto}
            clinic_features = hyperbolic_entailment_completion_strict(
                embed_dict, 'C', self.manifold)

        # -----------------------------------------------------------------
        #   Branch B:  n_missing == 2   (chain HGS completion)
        #
        #   Step 1: ONE real anchor + per-modality proto fills the second
        #           cone slot.  Each missing slot uses ITS OWN proto (not a
        #           single shared tensor), so the cone geometry and norm
        #           hierarchy still reflect the modality's level.
        #   Step 2: 1 real + 1 step-1-imputed anchor; canonical 2-anchor HGS
        #           on the final missing modality.
        # -----------------------------------------------------------------
        elif n_missing == 2 and p_obs:
            # only P observed -> impute G first (anchors: P_real + C_proto),
            # then impute C using P (real) + G (imputed).
            pathology_features = self.ffpe_fc(kwargs['x_path']).to('cuda')
            # step 1: G is missing, C slot holds C's running prototype
            embed_dict = {'P': pathology_features, 'G': g_proto, 'C': c_proto}
            genomics_features = hyperbolic_entailment_completion_strict(
                embed_dict, 'G', self.manifold)
            # step 2: C is missing
            embed_dict = {'P': pathology_features, 'G': genomics_features, 'C': c_proto}
            clinic_features = hyperbolic_entailment_completion_strict(
                embed_dict, 'C', self.manifold)

        elif n_missing == 2 and o_obs:
            # only O observed -> impute P first (anchors: G_real + C_proto),
            # then impute C using P (imputed) + G (real).
            genomics_features, g_loss, t_loss = self.process_multiomics(kwargs)
            # step 1: P is missing, C slot holds C's running prototype
            embed_dict = {'P': p_proto, 'G': genomics_features, 'C': c_proto}
            pathology_features = hyperbolic_entailment_completion_strict(
                embed_dict, 'P', self.manifold)
            # step 2: C is missing
            embed_dict = {'P': pathology_features, 'G': genomics_features, 'C': c_proto}
            clinic_features = hyperbolic_entailment_completion_strict(
                embed_dict, 'C', self.manifold)

        elif n_missing == 2 and c_obs:
            # only C observed -> impute P first (anchors: G_proto + C_real),
            # then impute G using P (imputed) + C (real).
            clinic_features = self.clinic_fc(kwargs['report']).unsqueeze(0)
            # step 1: P is missing, G slot holds G's running prototype
            embed_dict = {'P': p_proto, 'G': g_proto, 'C': clinic_features}
            pathology_features = hyperbolic_entailment_completion_strict(
                embed_dict, 'P', self.manifold)
            # step 2: G is missing
            embed_dict = {'P': pathology_features, 'G': g_proto, 'C': clinic_features}
            genomics_features = hyperbolic_entailment_completion_strict(
                embed_dict, 'G', self.manifold)

        # -----------------------------------------------------------------
        #   Branch C:  triple missing -> error
        # -----------------------------------------------------------------
        elif n_missing >= 3:
            raise ValueError("MHSurv: all three modalities missing, no signal to impute from")

        # -----------------------------------------------------------------
        #   Branch D:  n_missing == 0   (full modality, unchanged)
        # -----------------------------------------------------------------
        else:
            x_path = kwargs["x_path"]
            x_genomic = [kwargs["x_genomic%d" % i].to("cuda") for i in range(1, 7)]
            x_transomic = [kwargs["x_transomic%d" % i].to("cuda") for i in range(1, self.num_pathways+1)]
            x_table = kwargs["report"]

            if kwargs["protein"] is not None:
                is_protein = True
                protein = kwargs["protein"].to("cuda")
                protein_features = self.protein_networks(protein)
            else:
                is_protein = False
                protein_features = None

            pathology_features = self.ffpe_fc(x_path).to("cuda")
            clinic_features = self.clinic_fc(x_table).unsqueeze(0)

            genomic = [self.gene_sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_genomic)]
            transomic = [self.trans_sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_transomic)]
            genomics = self.gene_agg(torch.stack(genomic)).unsqueeze(0)
            transomics = self.trans_agg(torch.stack(transomic)).unsqueeze(0)

            g_loss = hyperbolic_entailment_loss_pairwise(genomics, transomics, self.manifold, self.k)
            if is_protein:
                t_loss = hyperbolic_entailment_loss_pairwise(transomics, protein_features, self.manifold, self.k)
                genomics = torch.cat((genomics, transomics, protein_features), dim=-2)
            else:
                t_loss = torch.tensor(0.0, device='cuda')
                genomics = torch.cat((genomics, transomics), dim=-2)
            genomics_features = genomics

            g2p_loss = hyperbolic_entailment_loss_pairwise(pathology_features, genomics_features, self.manifold, self.k)
            p2c_loss = hyperbolic_entailment_loss_pairwise(clinic_features, pathology_features, self.manifold, self.k)

            pathology_proto = prototype_kmeans(pathology_features, 128)
            genomics_proto = prototype_kmeans(genomics_features, 32)
            clinic_proto = prototype_kmeans(clinic_features, 10)

            # ---- EMA-update running prototypes (only during training) ----
            # During eval we keep the last training-time prototype frozen so
            # the same imputation rule applies across all val cases.
            if self.training:
                self._update_prototype('P', pathology_proto)
                self._update_prototype('G', genomics_proto)
                self._update_prototype('C', clinic_proto)

            # ====================== Attention Map 保存 ======================
            # 只在完整三模态 + 推理时保存 (return_attn_maps 开关)
            # if kwargs.get("return_attn_maps", False):
            #     patient_id = kwargs.get("patient_id", None)
            #     save_root = kwargs.get("attn_save_root", "./heat_pt")
            #     if patient_id is None:
            #         print("[warn] return_attn_maps=True 但未提供 patient_id, 跳过保存")
            #     else:
            #         self._compute_and_save_modality_attn_maps(
            #             gene_tokens=genomics_features,      # [1, N_g, D]
            #             path_tokens=pathology_features,     # [1, N_patch, D]
            #             clinic_tokens=clinic_features,      # [1, N_c, D]
            #             patient_id=patient_id,
            #             save_root=save_root,
            #         )
            # ==============================================================

        p_total = pathology_features.shape[1]
        g_total = genomics_features.shape[1]
        token_cross = self.attention_fusion(genomics_features, pathology_features, clinic_features)
        token_cross = self.feed_forward(token_cross)
        token_cross = self.layer_norm(token_cross)

        gene_embed = token_cross[:, :g_total, :]
        path_embed = token_cross[:, g_total:g_total+p_total, :]
        clinic_embed = token_cross[:, p_total+g_total:, :]

        gene_embed = torch.mean(gene_embed, dim=1)
        path_embed = torch.mean(path_embed, dim=1)
        clinic_embed = torch.mean(clinic_embed, dim=1)
        token_mm = torch.cat((gene_embed, path_embed, clinic_embed), dim=-1)
        fusion = self.mm(token_mm)

        logits = self.classifier(fusion)

        # ====================== Clinic 列级梯度归因 ======================
        # 仅在完整三模态 + return_clinic_attr=True 时触发
        # is_full_branch = (
        #     kwargs.get("x_path", None) is not None
        #     and kwargs.get("x_genomic1", None) is not None
        #     and kwargs.get("report", None) is not None
        # )
        # if kwargs.get("return_clinic_attr", False) and is_full_branch:
        #     x_table_raw = kwargs.get("x_table_raw", None)
        #     patient_id = kwargs.get("patient_id", None)
        #     save_root = kwargs.get("attn_save_root", "./heat_pt")
        #     if x_table_raw is None or patient_id is None:
        #         print("[warn] return_clinic_attr=True 但缺 x_table_raw 或 patient_id, 跳过")
        #     elif not x_table_raw.requires_grad:
        #         print("[warn] x_table_raw.requires_grad=False, 无法做梯度归因, 跳过")
        #     else:
        #         self._compute_and_save_clinic_column_attr(
        #             x_table=x_table_raw,
        #             logits=logits,
        #             patient_id=patient_id,
        #             save_root=save_root,
        #         )
        # ==============================================================

        # ============= 分子三组 (genomics/transomics/protein) heatmap + token归因 =============
        # if kwargs.get("return_molecular_group_maps", False) and is_full_branch:
        #     patient_id = kwargs.get("patient_id", None)
        #     save_root = kwargs.get("attn_save_root", "./heat_pt")
        #     genomics_raw = kwargs.get("genomics_raw", None)
        #     transomics_raw = kwargs.get("transomics_raw", None)
        #     protein_raw = kwargs.get("protein_raw", None)
        #     if patient_id is not None and genomics_raw is not None:
        #         # 用 SNN 输出、agg 之前的 tokens 做 heatmap
        #         # genomic: list of [256] → stack → [1, 6, 256]
        #         # transomic: list of [256] → stack → [1, N_trans, 256]
        #         # 这样两组保留各自 pathway 的独立信息, heatmap 有区分度
        #         genomic_pre_agg = torch.stack(genomic).unsqueeze(0)     # [1, 6, 256]
        #         transomic_pre_agg = torch.stack(transomic).unsqueeze(0) # [1, N_trans, 256]
        #         self._compute_and_save_molecular_group_maps(
        #             genomics_tokens=genomic_pre_agg,
        #             transomics_tokens=transomic_pre_agg,
        #             protein_tokens=protein_features if is_protein else None,
        #             path_tokens=pathology_features,
        #             logits=logits,
        #             genomics_raw=genomics_raw,
        #             transomics_raw=transomics_raw,
        #             protein_raw=protein_raw,
        #             patient_id=patient_id,
        #             save_root=save_root,
        #         )
        #     else:
        #         print("[warn] return_molecular_group_maps=True 但缺 genomics_raw 或 patient_id")
        # ==============================================================

        return logits, (g_loss+t_loss)+(p2c_loss+g2p_loss), {'G': genomics_proto, 'P': pathology_proto, 'C': clinic_proto}


    def _compute_and_save_modality_attn_maps(self, gene_tokens, path_tokens, clinic_tokens,
                                            patient_id, save_root):
        """
        计算 gene / path / clinic 三模态对 WSI patches 的 cross-attention,
        每模态沿 modality-token 维做 mean 聚合, 得到 [N_patch] 向量,
        分别保存到 save_root/{gene,path,clinic}/{patient_id}.pt

        Args:
            gene_tokens   : [1, N_g, D]   (已包含 genomics + transomics + [protein])
            path_tokens   : [1, N_patch, D]   ← 作为 Key/Value,保留空间语义
            clinic_tokens : [1, N_c, D]
            patient_id    : str  (例如 'TCGA-XX-XXXX-01Z-00-DX1')
            save_root     : str
        """
        import math
        import os

        with torch.no_grad():
            K = path_tokens.squeeze(0)      # [N_patch, D]
            d_k = K.shape[-1]
            scale = math.sqrt(d_k)
            N_patch = K.shape[0]

            modality_map = {
                "gene":   gene_tokens.squeeze(0),    # [N_g, D]
                "path":   path_tokens.squeeze(0),    # [N_patch, D]  self-attention
                "clinic": clinic_tokens.squeeze(0),  # [N_c, D]
            }

            # 清理 patient_id 后缀 (兼容可能传入 .svs)
            pid = patient_id
            if pid.endswith(".svs"):
                pid = pid[:-4]

            for modality, Q in modality_map.items():
                # Q: [N_q, D], K: [N_patch, D]
                attn_scores = torch.matmul(Q, K.T) / scale          # [N_q, N_patch]
                attn_weights = F.softmax(attn_scores, dim=-1)        # 每个 query 在 patches 上归一化
                # mean over modality tokens -> [N_patch]
                per_patch = attn_weights.mean(dim=0)                 # [N_patch]

                save_dir = os.path.join(save_root, modality)
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"{pid}.pt")

                torch.save({
                    "patient_id": pid,
                    "modality": modality,
                    "cross_attn_scores": per_patch.detach().cpu(),   # [N_patch]
                    "num_patches": N_patch,
                }, out_path)
                print(f"[attn] saved {modality} -> {out_path}  shape={tuple(per_patch.shape)}")

    def _compute_and_save_clinic_column_attr(self, x_table, logits, patient_id, save_root):
        """
        对 clinic 输入 x_table 的每一列 (M 列 LLM embedding, 每列 512 维) 做
        Gradient x Input 归因, 目标为 risk = sum(sigmoid(logits)).

        Args:
            x_table    : [M, 512]  原始 clinic 输入 tensor, 必须 requires_grad=True
            logits     : [1, n_classes]  分类头输出
            patient_id : str
            save_root  : str  会在 save_root/clinic_col_attr/ 下保存
        """
        import os

        # risk: 正值 = 高风险. 用累积 hazard 作为目标, 方向直观.
        hazards = torch.sigmoid(logits)                      # [1, n_classes]
        risk = torch.sum(hazards)                            # scalar, ↑ = 高风险

        # 只对 x_table 求梯度
        grad = torch.autograd.grad(
            outputs=risk,
            inputs=x_table,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]                                                 # [M, 512]

        # Gradient x Input, 沿 512 维度求和 -> [M]
        col_attr = (x_table.detach() * grad.detach()).sum(dim=1)   # [M]

        pid = patient_id
        if pid.endswith(".svs"):
            pid = pid[:-4]

        save_dir = os.path.join(save_root, "clinic_col_attr")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{pid}.pt")

        torch.save({
            "patient_id": pid,
            "col_attr": col_attr.cpu(),                 # [M]  Gradient x Input
            "num_columns": int(col_attr.shape[0]),
            "risk": float(risk.detach().cpu().item()),
        }, out_path)
        print(f"[clinic-attr] saved -> {out_path}  M={col_attr.shape[0]}  risk={risk.item():.4f}")

    def _compute_and_save_molecular_group_maps(
        self, genomics_tokens, transomics_tokens, protein_tokens,
        path_tokens, logits,
        genomics_raw, transomics_raw, protein_raw,
        patient_id, save_root,
    ):
        """
        genomics / transomics / protein 三组分别:
          1) cross-attention heatmap on WSI patches → [N_patch]
          2) Gradient×Input token 重要性           → [M_group]

        Args:
            genomics_tokens   : [1, M_g, D]   aggregated genomics
            transomics_tokens : [1, M_t, D]   aggregated transomics
            protein_tokens    : [1, M_p, D] or None
            path_tokens       : [1, N_patch, D]
            logits            : [1, n_classes]
            genomics_raw      : list of [dim_i] tensors, requires_grad=True (6 个 pathway 原始输入)
            transomics_raw    : list of [dim_i] tensors, requires_grad=True
            protein_raw       : [1, D_prot] or None, requires_grad=True if not None
            patient_id        : str
            save_root         : str
        """
        import math
        import os

        pid = patient_id
        if pid.endswith(".svs"):
            pid = pid[:-4]

        K = path_tokens.squeeze(0).detach()     # [N_patch, D]
        d_k = K.shape[-1]
        scale = math.sqrt(d_k)
        N_patch = K.shape[0]

        # risk: 正值 = 高风险
        hazards = torch.sigmoid(logits)
        risk = torch.sum(hazards)
        risk_val = float(risk.detach().cpu().item())

        # 三组: (名称, tokens[1,M,D], raw_inputs)
        groups = [
            ("genomics",   genomics_tokens,   genomics_raw),
            ("transomics", transomics_tokens,  transomics_raw),
        ]
        if protein_tokens is not None and protein_raw is not None:
            groups.append(("protein", protein_tokens, protein_raw))

        # 收集所有需要求梯度的 raw inputs
        all_raw_inputs = []
        for _, _, raw in groups:
            if isinstance(raw, (list, tuple)):
                all_raw_inputs.extend(raw)
            else:
                all_raw_inputs.append(raw)

        # 一次性求所有梯度 (retain_graph=True 因为要对多组分别处理)
        grads = torch.autograd.grad(
            outputs=risk,
            inputs=all_raw_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        grad_idx = 0
        for group_name, tokens, raw in groups:
            # ---- 1. Cross-attention heatmap: Q=group tokens, K=WSI ----
            with torch.no_grad():
                Q = tokens.squeeze(0)                             # [M, D]
                attn = torch.matmul(Q, K.T) / scale               # [M, N_patch]
                attn = F.softmax(attn, dim=-1)
                per_patch = attn.mean(dim=0)                       # [N_patch]

            # ---- 2. 原始特征重要性: Gradient×Input → 每个基因/特征一个值 ----
            if isinstance(raw, (list, tuple)):
                n_raw = len(raw)
                raw_grads = grads[grad_idx:grad_idx + n_raw]
                grad_idx += n_raw
                per_gene_list = []
                pathway_imp_list = []
                for r, g in zip(raw, raw_grads):
                    if g is None:
                        per_gene_list.append(torch.zeros_like(r))
                        pathway_imp_list.append(torch.tensor(0.0))
                    else:
                        per_gene_list.append(r.detach() * g.detach())
                        pathway_imp_list.append((r.detach() * g.detach()).sum())
                token_importance = torch.cat(per_gene_list, dim=0)
                pathway_importance = torch.stack(pathway_imp_list)
            else:
                g = grads[grad_idx]
                grad_idx += 1

                # raw 可能是 [D], [N, D], 或 [1, N, D]
                r = raw.detach()
                if r.dim() == 3:
                    r = r.squeeze(0)        # [1, N, D] → [N, D]
                if g is not None and g.dim() == 3:
                    g = g.detach().squeeze(0)

                print(f"[mol-group] {group_name}: raw shape={tuple(raw.shape)}  "
                      f"grad shape={tuple(g.shape) if g is not None else 'None'}")

                if g is None:
                    if r.dim() == 1:
                        token_importance = torch.zeros_like(r)           # [D]
                    else:
                        token_importance = torch.zeros(r.shape[0], device=r.device)  # [N]
                    pathway_importance = torch.tensor([0.0])
                    print(f"[mol-group] {group_name}: grad is None, skipping attribution")
                else:
                    gd = g.detach() if not g.is_leaf else g
                    if r.dim() == 1:
                        # [D] * [D] → per-feature importance
                        token_importance = (r * gd)                       # [D]
                    else:
                        # [N, D] * [N, D] → sum over D → per-token importance [N]
                        token_importance = (r * gd).sum(dim=-1)           # [N]
                    pathway_importance = token_importance.sum().unsqueeze(0)  # [1]

            # ---- 保存 ----
            save_dir = os.path.join(save_root, group_name)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{pid}.pt")

            torch.save({
                "patient_id": pid,
                "group": group_name,
                "cross_attn_scores": per_patch.detach().cpu(),            # [N_patch]  WSI heatmap
                "gene_importance": token_importance.detach().cpu(),        # [sum(dim_i)]  每个基因
                "pathway_importance": pathway_importance.detach().cpu(),   # [n_pathways]  每个 pathway 汇总
                "num_patches": N_patch,
                "num_genes": int(token_importance.shape[0]),
                "num_pathways": int(pathway_importance.shape[0]),
                "risk": risk_val,
            }, out_path)
            print(f"[mol-group] {group_name} -> {out_path}  "
                  f"patches={N_patch}  genes={token_importance.shape[0]}  "
                  f"pathways={pathway_importance.shape[0]}")

    def _calculate_and_save_cross_attn_matrix(self, query_features, key_features, patient_id, save_dir):
        """
        Calculates the scaled dot-product attention score matrix between two feature sequences.

        Args:
            query_features (torch.Tensor): The first feature sequence (e.g., pathology).
                                           Shape: [1, N_q, d]
            key_features (torch.Tensor): The second feature sequence (e.g., genomics).
                                         Shape: [1, N_k, d]
            patient_id (str): The unique identifier for the patient.
            save_dir (str): The directory where the score file will be saved.
        """
        import math
        import os
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Squeeze the batch dimension (from [1, N, d] to [N, d])
        query = query_features.squeeze(0)
        key = key_features.squeeze(0)
        
        # Get the feature dimension for scaling
        d_k = query.shape[-1]
        
        # --- Calculate Scaled Dot-Product Attention Scores ---
        # (N_q, d) @ (d, N_k) -> (N_q, N_k)
        # This computes the similarity between every token in the query and every token in the key.
        attn_scores = torch.matmul(query, key.T) / math.sqrt(d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        # The 'attn_scores' tensor now has the shape [N_q, N_k], as requested.
        print(f'{patient_id}:{attn_weights.shape}')
        # --- Prepare data for saving ---
        # Detach from the computation graph and move to CPU
        scores_to_save = attn_weights.detach().cpu()
        
        # Create the dictionary
        data_to_save = {
            'patient_id': patient_id,
            'cross_attn_scores': scores_to_save
        }
        
        # Define the output filename and save the dictionary
        output_filename = os.path.join(save_dir, f"{patient_id.removesuffix('.svs')}.pt")
        torch.save(data_to_save, output_filename)
        
        # Optional: print a confirmation message
        # print(f"Saved attention score for patient {patient_id} to {output_filename}")
    def process_multiomics(self, kwargs):
        """
        处理基因组（genomic）、转录组（transomic）和蛋白组（protein）模态，
        返回聚合后的基因特征与对应的蕴含损失项。

        Args:
            kwargs (dict): forward() 传入的关键字参数

        Returns:
            genomics_features (torch.Tensor): 聚合后的特征 [1, N, D]
            g_loss (torch.Tensor): genomics→transomics 的蕴含损失
            t_loss (torch.Tensor): transomics→protein 的蕴含损失
        """
        # ----------- 1️⃣ 获取输入 -----------
        x_genomic = [kwargs.get(f"x_genomic{i}", None) for i in range(1, 7)]
        x_genomic = [x.to("cuda") for x in x_genomic if x is not None]

        x_transomic = [kwargs.get(f"x_transomic{i}", None) for i in range(1, self.num_pathways + 1)]
        x_transomic = [x.to("cuda") for x in x_transomic if x is not None]

        protein = kwargs.get("protein", None)
        is_protein = protein is not None

        # ----------- 2️⃣ 特征提取 -----------
        # 若模态缺失，则置零张量代替
        genomic = [self.gene_sig_networks[idx].forward(sig_feat.float()) 
                for idx, sig_feat in enumerate(x_genomic)]
        genomics = self.gene_agg(torch.stack(genomic)).unsqueeze(0)

        transomic = [self.trans_sig_networks[idx].forward(sig_feat.float()) 
                    for idx, sig_feat in enumerate(x_transomic)]
        transomics = self.trans_agg(torch.stack(transomic)).unsqueeze(0)


        if is_protein:
            protein_features = self.protein_networks(protein.to("cuda"))
        else:
            protein_features = None

        # ----------- 3️⃣ 蕴含损失计算 -----------
        g_loss = t_loss = torch.tensor(0.0, device="cuda")

        # ----------- 4️⃣ 模态拼接 -----------
        genomics_features = torch.cat(
            [x for x in [genomics, transomics, protein_features] if x is not None],
            dim=-2
        )

        return genomics_features, g_loss, t_loss

    def init_per_gene_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.gene_sig_networks = nn.ModuleList(sig_networks)
        
    def init_per_gene_hyp(self, omic_sizes, k=3.0):
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = HypSNNBlock(Lorentz(k=k), in_dim=input_dim)
            sig_networks.append(fc_omic)
        self.gene_sig_networks = nn.ModuleList(sig_networks)
    

    def init_per_trans_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.trans_sig_networks = nn.ModuleList(sig_networks)

    def init_per_trans_hyp(self, omic_sizes, k=3.0):
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = HypSNNBlock(Lorentz(k=k), in_dim=input_dim)
            sig_networks.append(fc_omic)
        self.trans_sig_networks = nn.ModuleList(sig_networks)
    
    def init_per_protein_model(self, p_hidden):
        protein_networks = []
        for idx in range(2):
            protein_networks.append(nn.Linear(p_hidden[idx], p_hidden[idx + 1]))
            protein_networks.append(nn.ReLU6())
            protein_networks.append(nn.Dropout(0.25))
        self.protein_networks = nn.Sequential(*protein_networks)
        
    def init_per_protein_hyp(self, hidden, k=0.1):
        protein_networks = [HypSNNBlock(Lorentz(k=k), in_dim=1280)]
        self.protein_networks = nn.Sequential(*protein_networks)

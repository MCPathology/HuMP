import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SURVIVAL_CODE = _REPO_ROOT / "Survival"
for _path in (_REPO_ROOT, _SURVIVAL_CODE):
    _path = str(_path)
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    try:
        from Models.manifolds.lorentz import Lorentz
        from Models.layers.lhyperbolic import *
    except ImportError:
        from models.manifolds.lorentz import Lorentz
        from models.layers.lhyperbolic import *
    try:
        from Models.DTFD.network import DimReduction
        from Models.DTFD.Attention import HypAttention_Gated, HypClassifier_1fc
    except ImportError:
        DimReduction = None
        HypAttention_Gated = None
        HypClassifier_1fc = None
except Exception as _models_import_error:
    # Pure Euclidean baselines (ABMIL / ConcatMIL) do not need the HuMP Models
    # package. Keep this file importable when only those baselines are used.
    Lorentz = None
    DimReduction = None
    HypAttention_Gated = None
    HypClassifier_1fc = None

    def hyperbolic_entailment_loss_pairwise(*args, **kwargs):
        raise RuntimeError(
            "Hyperbolic layers are unavailable because the HuMP hyperbolic package "
            f"could not be imported: {_models_import_error}"
        )

class Attention(nn.Module):
    def __init__(self, in_size, out_size, confounder_path=False, confounder_learn=False, \
        confounder_dim=128, confounder_merge='cat'):
        super(Attention, self).__init__()
        self.L = in_size
        self.D = in_size
        self.K = 1
        self.confounder_merge = confounder_merge
        assert confounder_merge in ['cat', 'add', 'sub']
        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, self.L),
        #     nn.ReLU(),
        # )

        # self.attention_1 = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.Tanh(),
            
        # )
        # self.attention_1 = nn.Identity()
        # self.attention_2 = nn.Linear(self.D, self.K)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier =  nn.Linear(self.L*self.K, out_size)
        self.confounder_path=None
        if confounder_path: 
            print('deconfounding')
            self.confounder_path = confounder_path
            conf_list = []
            for i in confounder_path:
                conf_list.append(torch.from_numpy(np.load(i)).view(-1,in_size).float())
            conf_tensor = torch.cat(conf_list, 0) 
            conf_tensor_dim = conf_tensor.shape[-1]
            if confounder_learn:
                self.confounder_feat = nn.Parameter(conf_tensor, requires_grad=True)
            else:
                self.register_buffer("confounder_feat",conf_tensor)
            joint_space_dim = confounder_dim
            dropout_v = 0.5
            # self.confounder_W_q = nn.Linear(in_size, joint_space_dim)
            # self.confounder_W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            self.W_q = nn.Linear(in_size, joint_space_dim)
            self.W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            if confounder_merge == 'cat':
                self.classifier =  nn.Linear(self.L*self.K+conf_tensor_dim, out_size)
            elif confounder_merge == 'add' or 'sub':
                self.classifier =  nn.Linear(self.L*self.K, out_size)
            self.dropout = nn.Dropout(dropout_v)

    def forward(self, x):
        # x = x.squeeze(0)

        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.feature_extractor_part2(H)  # NxL

        # A = self.attention_1(x)  
        # A = self.attention_2(A)  # NxK
        A = self.attention(x)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        # print('norm')
        # A = F.softmax(A/ torch.sqrt(torch.tensor(x.shape[1])), dim=1)  # For Vis

        M = torch.mm(A, x)  # KxL
        if self.confounder_path:
            device = M.device
            # bag_q = self.confounder_W_q(M)
            # conf_k = self.confounder_W_k(self.confounder_feat)
            bag_q = self.W_q(M)
            conf_k = self.W_k(self.confounder_feat)
            deconf_A = torch.mm(conf_k, bag_q.transpose(0, 1))
            deconf_A = F.softmax( deconf_A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
            conf_feats = torch.mm(deconf_A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
            if self.confounder_merge == 'cat':
                M = torch.cat((M,conf_feats),dim=1)
            elif self.confounder_merge == 'add':
                M = M + conf_feats
            elif self.confounder_merge == 'sub':
                M = M - conf_feats
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        if self.confounder_path:
            return Y_prob, M, deconf_A
        else:
            return Y_prob, M, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
        

class HypABMIL(nn.Module):
    def __init__(self, k=1.0, input_dim=768, hidden_dim=256, dropout=False,
                 n_classes=2, activation='softmax',
                 clinical_in_dim=512, gene_in_dim=512,
                 fusion_modalities='clinical',
                 missing_completion='hgs', hierarchy_weight=1.0,
                 cone_K=0.1, num_prototypes=8,
                 num_gene_pathways=331):
        """
        Hyperbolic MIL for the HuMP classification setting.

        The model explicitly follows the paper's cross-modal hierarchy
            molecular/gene (O) -> pathology (P) -> clinical (C).
        Clinical and molecular inputs are kept as distinct modalities instead
        of being merged into a generic table token, which lets the hierarchy
        loss and HGS missing-modality completion operate on the right anchors.
        """
        super(HypABMIL, self).__init__()
        if Lorentz is None:
            raise RuntimeError(
                "HypABMIL requires the HuMP Models package. Use --model concat_mil "
                "or restore Models/ before selecting --model hyp_a."
            )

        self.manifold = Lorentz(k=k)
        self.activation = activation
        self.device = 'cuda'
        self.hidden_dim = hidden_dim
        self.missing_completion = missing_completion
        self.hierarchy_weight = hierarchy_weight
        self.cone_K = cone_K
        self.num_prototypes = num_prototypes
        self.num_gene_pathways = num_gene_pathways
        self.modalities = set(fusion_modalities.split('+')) if fusion_modalities else set()
        self.use_clinical = 'clinical' in self.modalities
        self.use_gene = 'gene' in self.modalities
        assert missing_completion in ('placeholder', 'hgs', 'zero'), \
            f"unknown missing_completion={missing_completion}"

        self.h_soft = HypActivation(self.manifold, nn.Softmax())
        self.attention_a = nn.Sequential(*[
            HypLinear(self.manifold, hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.25)
        ])
        self.p_mlp = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ])
        self.clinical_mlp = nn.Sequential(*[
            nn.Linear(clinical_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ])
        self.gene_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gene_in_dim, hidden_dim),
                nn.ELU(),
                nn.AlphaDropout(p=0.25),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.AlphaDropout(p=0.25),
            )
            for _ in range(num_gene_pathways)
        ])
        # Backward-compatible alias for code that still introspects table_mlp.
        self.table_mlp = self.clinical_mlp

        self.attention_b = nn.Sequential(*[
            HypLinear(self.manifold, hidden_dim, hidden_dim),
            nn.ReLU(),
            HypLinear(self.manifold, hidden_dim, hidden_dim)
        ])
        self.attention_c = self.attention = nn.Sequential(*[
            HypLinear(self.manifold, hidden_dim, hidden_dim),
            nn.Tanh(),
            HypLinear(self.manifold, hidden_dim, 1)
        ])
        self.classifer = HypLinear(self.manifold, hidden_dim, n_classes)

        self.gene_placeholder = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.clinical_placeholder = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        # Backward-compatible alias for old checkpoints / logs.
        self.clinic_placeholder = self.clinical_placeholder

        self.register_buffer('gene_proto', torch.zeros(num_prototypes, 1, hidden_dim))
        self.register_buffer('gene_proto_count', torch.zeros(num_prototypes))
        self.register_buffer('clinical_proto', torch.zeros(num_prototypes, 1, hidden_dim))
        self.register_buffer('clinical_proto_count', torch.zeros(num_prototypes))
        # Backward-compatible aliases.
        self.clinic_proto = self.clinical_proto
        self.clinic_proto_count = self.clinical_proto_count

    def _normalize_tokens(self, tokens):
        if tokens is None:
            return None
        t = tokens.float()
        if t.numel() == 0:
            return None
        if t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 2:
            t = t.unsqueeze(0)
        elif t.dim() == 4 and t.size(0) == 1:
            t = t.squeeze(0)
        return t

    def _anchor(self, tokens):
        if tokens is None:
            return None
        return tokens.mean(dim=-2, keepdim=True)

    def _encode_gene_tokens(self, gene):
        if gene.size(1) < self.num_gene_pathways:
            pad = gene.new_zeros(gene.size(0), self.num_gene_pathways - gene.size(1), gene.size(2))
            gene = torch.cat([gene, pad], dim=1)
        elif gene.size(1) > self.num_gene_pathways:
            gene = gene[:, :self.num_gene_pathways]
        encoded = [
            self.gene_mlp[idx](gene[:, idx, :])
            for idx in range(self.num_gene_pathways)
        ]
        return torch.stack(encoded, dim=1)

    def _proto_state(self, name):
        if name == 'gene':
            return self.gene_placeholder, self.gene_proto, self.gene_proto_count
        return self.clinical_placeholder, self.clinical_proto, self.clinical_proto_count

    def _update_proto(self, name, tokens):
        if (not self.training) or tokens is None:
            return
        _, proto, count = self._proto_state(name)
        with torch.no_grad():
            anchor = self._anchor(tokens.detach()).squeeze(0)
            initialized = count > 0
            if not initialized.all():
                idx = int((~initialized).nonzero(as_tuple=False)[0].item())
                proto[idx].copy_(anchor)
                count[idx].fill_(1.0)
                return
            distances = torch.cdist(anchor.view(1, -1), proto.squeeze(1), p=2).squeeze(0)
            idx = int(torch.argmin(distances).item())
            n_old = count[idx].item()
            n_new = n_old + 1.0
            momentum = 1.0 / min(n_new, 100.0)
            proto[idx].mul_(1.0 - momentum).add_(anchor * momentum)
            count[idx].fill_(min(n_new, 10000.0))

    def _select_proto(self, name, query_anchor, device):
        placeholder, proto, count = self._proto_state(name)
        placeholder = placeholder.to(device)
        initialized = count > 0
        if not initialized.any():
            return placeholder
        proto_bank = proto[initialized].to(device)
        if query_anchor is None:
            idx = int(torch.argmax(count[initialized]).item())
            return proto_bank[idx:idx + 1]
        query = self._anchor(query_anchor) if query_anchor.dim() == 3 else query_anchor
        distances = torch.cdist(query.view(1, -1), proto_bank.squeeze(1), p=2).squeeze(0)
        idx = int(torch.argmin(distances).item())
        return proto_bank[idx:idx + 1]

    def _hgs_complete(self, name, anchors):
        """Complete a missing modality from observed hierarchy anchors.

        This is a deterministic, parameter-free HGS implementation: observed
        anchors define the feasible hierarchy region, and the modality prototype
        regularizes the final tangent representation. It deliberately avoids a
        trainable decoder so completion reuses the learned fusion geometry.
        """
        anchors = [self._anchor(a) if a.dim() == 3 else a for a in anchors if a is not None]
        device = anchors[0].device if anchors else self.gene_placeholder.device
        placeholder, _, _ = self._proto_state(name)
        placeholder = placeholder.to(device)

        if self.missing_completion == 'zero':
            return torch.zeros_like(placeholder)
        if self.missing_completion == 'placeholder' or not anchors:
            return placeholder

        anchor_mid = torch.stack(anchors, dim=0).mean(dim=0)
        proto = self._select_proto(name, anchor_mid, device)
        # Tangent-space geodesic-average surrogate used as the final HGS
        # refinement between the hierarchy-compatible candidate and the
        # nearest learned cluster prototype.
        return 0.5 * anchor_mid + 0.5 * proto

    def _curvature_value(self, device):
        k = getattr(self.manifold, 'k', torch.tensor(1.0, device=device))
        if not isinstance(k, torch.Tensor):
            k = torch.tensor(float(k), device=device)
        return k.abs().clamp_min(1e-6)

    def _entailment_loss(self, parent_euc, child_euc):
        if parent_euc is None or child_euc is None:
            device = parent_euc.device if parent_euc is not None else child_euc.device
            return torch.tensor(0.0, device=device)
        parent = self.tohyp(self._anchor(parent_euc) if parent_euc.dim() == 3 else parent_euc)
        child = self.tohyp(self._anchor(child_euc) if child_euc.dim() == 3 else child_euc)
        p = self.toeuc(parent)
        c = self.toeuc(child)
        p_norm = p.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        c_norm = c.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos = ((p * c).sum(dim=-1, keepdim=True) / (p_norm * c_norm)).clamp(-1 + 1e-6, 1 - 1e-6)
        angle = torch.arccos(cos)
        k = self._curvature_value(parent.device)
        radius = parent[..., 1:].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aperture = torch.arcsin((2.0 * self.cone_K / (torch.sqrt(k) * radius)).clamp(-1 + 1e-6, 1 - 1e-6))
        return torch.clamp(angle - aperture, min=0.0).mean()

    def forward(self, feats, clinical=None, gene=None):
        """
        Args:
            feats    : WSI patch features [N, D] or [1, N, D].
            clinical : clinical tokens [N_c, D] / [1, N_c, D] / None.
            gene     : molecular/gene tokens [N_o, D] / [1, N_o, D] / None.
        Returns:
            (Y_prob, logits, attention, hierarchy_loss)
        """
        device = feats.device if isinstance(feats, torch.Tensor) else self.device

        p_tok = self.p_mlp(feats)
        if p_tok.dim() == 2:
            p_tok = p_tok.unsqueeze(0)
        p_anchor = self._anchor(p_tok)

        gene_tok = None
        if self.use_gene:
            gene = self._normalize_tokens(gene)
            if gene is not None:
                gene_tok = self._encode_gene_tokens(gene)
                self._update_proto('gene', gene_tok)
            elif self.training:
                # Real missing molecular measurements in training are skipped,
                # so the sample is optimized with WSI + available clinical data.
                gene_tok = None
            else:
                # Molecular missing: constrained by observed pathology and, if
                # available, clinical patient-level context.
                clinical_anchor_for_gene = None
                if self.use_clinical:
                    c_norm = self._normalize_tokens(clinical)
                    if c_norm is not None:
                        c_tmp = self.clinical_mlp(c_norm)
                        clinical_anchor_for_gene = self._anchor(c_tmp)
                gene_tok = self._hgs_complete('gene', [p_anchor, clinical_anchor_for_gene])

        clinical_tok = None
        if self.use_clinical:
            clinical = self._normalize_tokens(clinical)
            if clinical is not None:
                clinical_tok = self.clinical_mlp(clinical)
                self._update_proto('clinical', clinical_tok)
            else:
                clinical_tok = self._hgs_complete('clinical', [gene_tok, p_anchor])

        tokens = []
        if gene_tok is not None:
            tokens.append(gene_tok)
        tokens.append(p_tok)
        if clinical_tok is not None:
            tokens.append(clinical_tok)
        x_concat = torch.cat(tokens, dim=-2)
        x_hyp = self.tohyp(x_concat)

        a = self.attention_a(x_hyp)
        a = a.view(-1, self.hidden_dim + 1)
        b = self.attention_b(a)
        A = self.attention(x_hyp.squeeze(0))
        A = torch.transpose(A[..., 1:], 1, 0)
        if self.activation == 'softmax':
            A = F.softmax(A, dim=-1)
        b = torch.mm(A, b[..., 1:])
        b = self.manifold.add_time(b)
        M = self.toeuc(b)

        M = self.tohyp(M)
        out = self.classifer(M)
        logits = self.toeuc(out)
        Y_prob = F.softmax(logits, dim=1)

        if self.training:
            loss_terms = []
            if gene_tok is not None:
                loss_terms.append(self._entailment_loss(gene_tok, p_anchor))
            if clinical_tok is not None:
                loss_terms.append(self._entailment_loss(p_anchor, clinical_tok))
            hierarchy_loss = self.hierarchy_weight * sum(loss_terms) / max(len(loss_terms), 1)
        else:
            hierarchy_loss = torch.tensor(0.0, device=device)

        return Y_prob, logits, A, hierarchy_loss

    def tohyp(self, x):
        x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
        x = self.manifold.expmap0(x)
        return x

    def toeuc(self, x):
        x = self.manifold.logmap0(x)[..., 1:]
        return x
class HypACMIL(nn.Module):
    def __init__(self, k, in_size, out_size, hidden=256, D=128, droprate=0, n_token=5, n_masked_patch=0, mask_drop=0.6):
        super(HypACMIL, self).__init__()
        if Lorentz is None or DimReduction is None:
            raise RuntimeError("HypACMIL requires the HuMP Models package.")
        self.manifold = Lorentz(k=k)
        self.dimreduction = DimReduction(in_size, hidden)
        self.attention = Attention_Gated(hidden, D, n_token)
        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(HypClassifier_1fc(hidden, out_size, droprate))
        self.n_masked_patch = n_masked_patch
        self.n_token = n_token
        self.Slide_classifier = HypClassifier_1fc(hidden, out_size, droprate)
        self.mask_drop = mask_drop


    def forward(self, x): ## x: N x L
        
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, x) ## K x L
        
        outputs = []
        for i, head in enumerate(self.classifier):
            tem = head(afeat[i])
            # print(tem.shape)
            outputs.append(tem)
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return torch.stack(outputs, dim=0), self.Slide_classifier(bag_feat), A_out.unsqueeze(0)

    
class HypClassifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0, k=1.0, confounder_path=False):
        super(HypClassifier_1fc, self).__init__()
        self.manifold = Lorentz(k=k)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

        
        self.fc = HypLinear(self.manifold, n_channels, n_classes)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.tohyp(x)
        pred = self.fc(x)
        pred = self.toeuc(pred)
        return pred
        
    def tohyp(self, x):
        
        x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
        x = self.manifold.expmap0(x)
        return x
    
    def toeuc(self, x):
        x = self.manifold.logmap0(x)[...,1:]
        return x        
class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x
        
class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN


        return A  ### K x N
        
class ACMIL(nn.Module):
    def __init__(self, in_size, out_size, hidden=256, D=128, droprate=0, n_token=5, n_masked_patch=0, mask_drop=0.6):
        super(ACMIL, self).__init__()
        if DimReduction is None:
            raise RuntimeError("ACMIL requires Models.DTFD.network.DimReduction.")
        self.dimreduction = DimReduction(in_size, hidden)
        self.attention = Attention_Gated(hidden, D, n_token)
        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(hidden, out_size, droprate))
        self.n_masked_patch = n_masked_patch
        self.n_token = n_token
        self.Slide_classifier = Classifier_1fc(hidden, out_size, droprate)
        self.mask_drop = mask_drop


    def forward(self, x, training=True): ## x: N x L
        # x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and training:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, x) ## K x L
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return torch.stack(outputs, dim=0), self.Slide_classifier(bag_feat), A_out.unsqueeze(0)

    def forward_feature(self, x, use_attention_mask=False): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return bag_feat


# ============================================================================
#  Concat-MIL: simple Euclidean multimodal baseline (WSI + clinical text)
#  Architecturally the Euclidean counterpart of HypABMIL with the entailment /
#  curvature components removed. Used as a fair simple-fusion baseline for
#  the R1Q7 multimodal classification table.
# ============================================================================

class ConcatMIL(nn.Module):
    """
    Late-fusion multimodal MIL baseline.

    Forward:
        feats : [N, input_dim]   - WSI patch bag (e.g. 512 from CONCH)
        table : [1, M, table_dim] or [M, table_dim] - clinical/text tokens
                                   (e.g. PLIP/CLIP-encoded, dim=512)
    Returns:
        bag_prediction : [1, n_classes]   - logits
        pred           : same as bag_prediction (tuple-shape compatibility)
        attention      : [1, N+M]         - attention over concatenated tokens
    """

    def __init__(self, input_dim=512, table_dim=512, hidden_dim=256,
                 n_classes=2, dropout=0.25):
        super().__init__()
        # patch encoder (Euclidean, mirrors HypABMIL.p_mlp)
        self.p_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # clinical/text encoder (Euclidean, mirrors HypABMIL.table_mlp)
        self.table_mlp = nn.Sequential(
            nn.Linear(table_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # gated attention over concatenated tokens
        self.attn_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attn_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attn_w = nn.Linear(hidden_dim, 1)
        # classifier
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def _encode_table_tokens(self, table):
        if table is None:
            return None
        if table.numel() == 0:
            return None
        if table.dim() == 3:
            t = table.squeeze(0)
        else:
            t = table
        t = self.table_mlp(t)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    def forward(self, feats, clinical=None, gene=None):
        # ---- normalize shapes ----
        # feats: [N, D]
        x = self.p_mlp(feats)                              # [N, H]

        table_tokens = []
        for table in (gene, clinical):
            t = self._encode_table_tokens(table)
            if t is not None:
                table_tokens.append(t)

        tokens = torch.cat([x] + table_tokens, dim=0) if table_tokens else x

        # ---- gated attention pool ----
        a_v = self.attn_V(tokens)                          # [N+M, H]
        a_u = self.attn_U(tokens)                          # [N+M, H]
        A = self.attn_w(a_v * a_u)                         # [N+M, 1]
        A = torch.transpose(A, 1, 0)                       # [1, N+M]
        A = F.softmax(A, dim=1)
        bag_feat = torch.mm(A, tokens)                     # [1, H]

        # ---- classifier ----
        logits = self.classifier(bag_feat)                 # [1, n_classes]
        return logits, logits, A


class SNNBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ELU(),
            nn.AlphaDropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)


class TriModalConcatMIL(nn.Module):
    """
    Three-modality late-concatenation baseline.

    Each modality is first reduced to one vector:
      WSI      -> gated-attention MIL pooling over patch tokens
      Gene     -> pathway-specific SNN encoders, then mean pooling
      Clinical -> shared MLP encoder, then mean pooling

    The three modality vectors are directly concatenated for classification.
    Missing table-side modalities are represented as zero vectors, so this
    baseline can run the same missing-modality protocol as HypABMIL.
    """

    def __init__(self, input_dim=512, table_dim=512, hidden_dim=256,
                 n_classes=2, num_gene_pathways=331, dropout=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gene_pathways = num_gene_pathways

        self.wsi_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attn_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attn_w = nn.Linear(hidden_dim, 1)

        self.gene_mlp = nn.ModuleList([
            nn.Sequential(
                SNNBlock(table_dim, hidden_dim, dropout=dropout),
                SNNBlock(hidden_dim, hidden_dim, dropout=dropout),
            )
            for _ in range(num_gene_pathways)
        ])
        self.clinical_mlp = nn.Sequential(
            nn.Linear(table_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def _normalize_tokens(self, tokens, device):
        if tokens is None or tokens.numel() == 0:
            return None
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        elif tokens.dim() == 4 and tokens.size(0) == 1:
            tokens = tokens.squeeze(0)
        return tokens.float().to(device)

    def _pool_wsi(self, feats):
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        x = self.wsi_mlp(feats.float())
        a_v = self.attn_V(x)
        a_u = self.attn_U(x)
        A = self.attn_w(a_v * a_u).transpose(2, 1)
        A = F.softmax(A, dim=2)
        wsi_embed = torch.bmm(A, x).squeeze(1)
        return wsi_embed, A.squeeze(0)

    def _pool_gene(self, gene, device):
        gene = self._normalize_tokens(gene, device)
        if gene is None:
            return torch.zeros(1, self.hidden_dim, device=device)
        if gene.size(1) < self.num_gene_pathways:
            pad = gene.new_zeros(gene.size(0), self.num_gene_pathways - gene.size(1), gene.size(2))
            gene = torch.cat([gene, pad], dim=1)
        elif gene.size(1) > self.num_gene_pathways:
            gene = gene[:, :self.num_gene_pathways]
        encoded = [
            self.gene_mlp[idx](gene[:, idx, :])
            for idx in range(self.num_gene_pathways)
        ]
        return torch.stack(encoded, dim=1).mean(dim=1)

    def _pool_clinical(self, clinical, device):
        clinical = self._normalize_tokens(clinical, device)
        if clinical is None:
            return torch.zeros(1, self.hidden_dim, device=device)
        return self.clinical_mlp(clinical).mean(dim=1)

    def forward(self, feats, clinical=None, gene=None):
        device = feats.device
        wsi_embed, A = self._pool_wsi(feats)
        gene_embed = self._pool_gene(gene, device)
        clinical_embed = self._pool_clinical(clinical, device)
        fused = torch.cat([wsi_embed, gene_embed, clinical_embed], dim=1)
        logits = self.classifier(fused)
        return logits, logits, A


class SurvPathMIL(nn.Module):
    """
    Classification adaptation of SurvPath for WSI + pathway-grouped gene input.

    The original SurvPath builds one pathway token per omic signature, projects
    WSI patches to the same latent dimension, applies multimodal attention over
    the concatenated pathway + WSI token sequence, then averages pathway and WSI
    branches before prediction. Here the final survival head is replaced by a
    classification head to match the current MIL benchmark.
    """

    def __init__(self, input_dim=512, gene_dim=512, hidden_dim=256,
                 n_classes=2, num_pathways=331, heads=1, dropout=0.1,
                 max_patches=4096):
        super().__init__()
        self.num_pathways = num_pathways
        self.hidden_dim = hidden_dim
        self.max_patches = max_patches

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        self.sig_networks = nn.ModuleList([
            nn.Sequential(
                SNNBlock(gene_dim, hidden_dim, dropout=0.25),
                SNNBlock(hidden_dim, hidden_dim, dropout=0.25),
            )
            for _ in range(num_pathways)
        ])

        self.modality_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.heads = heads
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.to_logits = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def _mm_attention(self, tokens, gene_len):
        x = self.attn_norm(tokens)
        b, n, d = x.shape
        h = self.heads
        head_dim = d // h
        qkv = self.to_qkv(x).view(b, n, 3, h, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * (head_dim ** -0.5)

        if gene_len > 0:
            q_gene = q[:, :, :gene_len, :]
            k_gene = k[:, :, :gene_len, :]
            v_gene = v[:, :, :gene_len, :]
            q_wsi = q[:, :, gene_len:, :]
            k_wsi = k[:, :, gene_len:, :]

            attn_gene_gene = torch.matmul(q_gene, k_gene.transpose(-2, -1))
            attn_gene_wsi = torch.matmul(q_gene, k_wsi.transpose(-2, -1))
            attn_gene_all = torch.cat([attn_gene_gene, attn_gene_wsi], dim=-1).softmax(dim=-1)
            attn_gene_all = self.attn_dropout(attn_gene_all)
            out_gene = torch.matmul(attn_gene_all, v)

            attn_wsi_gene = torch.matmul(q_wsi, k_gene.transpose(-2, -1)).softmax(dim=-1)
            attn_wsi_gene = self.attn_dropout(attn_wsi_gene)
            out_wsi = torch.matmul(attn_wsi_gene, v_gene)
            out = torch.cat([out_gene, out_wsi], dim=2)
            attn_scores = torch.cat(
                [
                    attn_gene_all.mean(dim=(1, 2)),
                    attn_wsi_gene.mean(dim=(1, 2)),
                ],
                dim=1,
            )
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, v)
            attn_scores = attn.mean(dim=(1, 2))

        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return out, attn_scores

    def _normalize_gene(self, gene, device):
        if gene is None or gene.numel() == 0:
            return None
        if gene.dim() == 2:
            gene = gene.unsqueeze(0)
        elif gene.dim() == 4 and gene.size(0) == 1:
            gene = gene.squeeze(0)
        gene = gene.float().to(device)
        if gene.size(1) < self.num_pathways:
            pad = gene.new_zeros(gene.size(0), self.num_pathways - gene.size(1), gene.size(2))
            gene = torch.cat([gene, pad], dim=1)
        elif gene.size(1) > self.num_pathways:
            gene = gene[:, :self.num_pathways]
        return gene

    def _encode_gene(self, gene):
        h_gene = [
            self.sig_networks[idx](gene[:, idx, :])
            for idx in range(self.num_pathways)
        ]
        return torch.stack(h_gene, dim=1)

    def forward(self, feats, gene=None):
        device = feats.device
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        if self.max_patches is not None and self.max_patches > 0 and feats.size(1) > self.max_patches:
            if self.training:
                idx = torch.randperm(feats.size(1), device=device)[:self.max_patches]
                idx, _ = torch.sort(idx)
            else:
                idx = torch.linspace(
                    0, feats.size(1) - 1, self.max_patches, device=device
                ).long()
            feats = feats.index_select(1, idx)
        wsi_tokens = self.wsi_projection_net(feats.float())
        wsi_tokens = wsi_tokens + self.modality_embed[:, 1:2, :]

        gene = self._normalize_gene(gene, device)
        if gene is not None:
            gene_tokens = self._encode_gene(gene)
            gene_tokens = gene_tokens + self.modality_embed[:, 0:1, :]
            tokens = torch.cat([gene_tokens, wsi_tokens], dim=1)
            gene_len = self.num_pathways
        else:
            tokens = wsi_tokens
            gene_len = 0

        attn_out, A = self._mm_attention(tokens, gene_len)
        tokens = self.attn_norm(tokens + attn_out)
        tokens = self.ffn_norm(tokens + self.feed_forward(tokens))

        if gene_len > 0:
            gene_embed = tokens[:, :gene_len, :].mean(dim=1)
            wsi_embed = tokens[:, gene_len:, :].mean(dim=1)
        else:
            wsi_embed = tokens.mean(dim=1)
            gene_embed = torch.zeros_like(wsi_embed)
        embedding = torch.cat([gene_embed, wsi_embed], dim=1)
        logits = self.to_logits(embedding)

        return logits, logits, A

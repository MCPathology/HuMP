"""
Adapted MIL baselines for the HuMP comparison framework:

  * WiKG  -- "Knowledge-graph MIL" (Yang et al., CVPR 2024).
             Original: https://github.com/aitrepreneur/WIKG
  * ILRA  -- "Exploring Low-Rank Property in MIL" (Xiang et al., ICLR 2023).
             Original paper: ICLR 2023.

Both are rewritten to fit the pure-WSI MIL trainer contract used in
`MIL/train_tcga.py`:

    Input
        feats : [N, D]              patch features (trainer flattens via .view(-1, feats_size))
    Output
        bag_prediction : [1, n_classes]    logits (used in CE loss)
        pred           : [1, n_classes]    == bag_prediction (kept for the 3-tuple convention)
        attention      : [1, N]            per-patch importance (logged as max/min/mean)

Selectable in train_tcga.py via `--model wikg` / `--model ilra`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
    _HAS_PYG = True
except Exception:                     # PyG not installed; we provide fallbacks below
    _HAS_PYG = False


# ---------------------------------------------------------------------------
# Small init helper (mirrors MIL/abmil.py style)
# ---------------------------------------------------------------------------
def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ===========================================================================
#                                 WiKG
# ===========================================================================
class WiKG(nn.Module):
    """Knowledge-graph MIL with top-k neighbour gated aggregation."""

    def __init__(self, dim_in=512, dim_hidden=512, topk=6, n_classes=2,
                 agg_type='bi-interaction', dropout=0.3, pool='attn'):
        super().__init__()

        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())

        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        # gated knowledge attention components
        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError(f"WiKG: agg_type={agg_type} not supported")

        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)

        # ---- readout ----------------------------------------------------
        self._pool_kind = pool
        if pool == "mean":
            if _HAS_PYG:
                self.readout = global_mean_pool
            else:
                self.readout = None      # use manual mean below
        elif pool == "max":
            if _HAS_PYG:
                self.readout = global_max_pool
            else:
                self.readout = None
        elif pool == "attn":
            # we always compute the attention weights manually so we can
            # surface them to the trainer (PyG's GlobalAttention hides them).
            self._att_net = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden // 2),
                nn.LeakyReLU(),
                nn.Linear(dim_hidden // 2, 1),
            )
            self.readout = None
        else:
            raise NotImplementedError(f"WiKG: pool={pool} not supported")

        self.apply(_init_weights)

    # ------------------------------------------------------------------
    def forward(self, feats):
        # ---- normalise input to [1, N, D] for the original WiKG ops ----
        if feats.dim() == 2:
            x = feats.unsqueeze(0)                   # [1, N, D]
        elif feats.dim() == 3:
            x = feats                                # [1, N, D] already
        else:
            raise ValueError(f"WiKG: bad input shape {feats.shape}")

        x = self._fc1(x)                             # [1, N, H]
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  # residual-style refinement

        e_h = self.W_head(x)                         # [1, N, H]
        e_t = self.W_tail(x)                         # [1, N, H]

        # ---- (1) construct neighbour graph via top-k attention ----
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)               # [1, N, N]
        N = e_h.size(1)
        k_eff = min(self.topk, N)
        topk_weight, topk_index = torch.topk(attn_logit, k=k_eff, dim=-1)     # [1, N, k]
        topk_index = topk_index.to(torch.long)
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)
        batch_indices = torch.arange(
            e_t.size(0), device=topk_index.device).view(-1, 1, 1)
        Nb_h = e_t[batch_indices, topk_index_expanded, :]                     # [1, N, k, H]

        topk_prob = F.softmax(topk_weight, dim=2)                             # [1, N, k]
        eh_r = (torch.mul(topk_prob.unsqueeze(-1), Nb_h)
                + torch.matmul((1 - topk_prob).unsqueeze(-1),
                               e_h.unsqueeze(2)))                             # [1, N, k, H]

        # ---- (2) gated knowledge attention ----
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, k_eff, -1)               # [1, N, k, H]
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)                 # [1, N, k]
        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)                 # [1, N, 1, k]
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)                      # [1, N, H]

        # ---- (3) message aggregation ----
        if self.agg_type == 'gcn':
            embedding = self.activation(self.linear(e_h + e_Nh))
        elif self.agg_type == 'sage':
            embedding = self.activation(self.linear(torch.cat([e_h, e_Nh], dim=2)))
        else:  # bi-interaction
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        h = self.message_dropout(embedding)                                    # [1, N, H]
        nodes = h.squeeze(0)                                                   # [N, H]

        # ---- (4) readout + record attention for logging ----
        if self._pool_kind == "attn":
            attn_logits = self._att_net(nodes)                                # [N, 1]
            attn = F.softmax(attn_logits, dim=0)                              # [N, 1]
            bag = (attn * nodes).sum(dim=0, keepdim=True)                     # [1, H]
            attention = attn.transpose(0, 1)                                  # [1, N]
        elif self._pool_kind == "mean":
            if _HAS_PYG:
                bag = self.readout(nodes, batch=None)
                bag = bag if bag.dim() == 2 else bag.unsqueeze(0)
            else:
                bag = nodes.mean(dim=0, keepdim=True)
            attention = torch.full((1, nodes.size(0)),
                                   1.0 / nodes.size(0), device=nodes.device)
        else:  # max
            if _HAS_PYG:
                bag = self.readout(nodes, batch=None)
                bag = bag if bag.dim() == 2 else bag.unsqueeze(0)
            else:
                bag = nodes.max(dim=0, keepdim=True).values
            attention = torch.full((1, nodes.size(0)),
                                   1.0 / nodes.size(0), device=nodes.device)

        bag = self.norm(bag)
        logits = self.fc(bag)                                                  # [1, n_classes]
        return logits, logits, attention


# ===========================================================================
#                                 ILRA
# ===========================================================================
class _MultiHeadAttention(nn.Module):
    """Multi-head attention block used by ILRA's GAB / NLP."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU()) if gated else None

    def forward(self, Q, K, return_attn=False):
        Q0 = Q
        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)

        A, attn_w = self.multihead_attn(Q, K, V, need_weights=return_attn)

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if self.gate is not None:
            O = O.mul(self.gate(Q0))
        if return_attn:
            return O, attn_w
        return O


class _GAB(nn.Module):
    """Geometric Attention Block (eq. 16 of the ILRA paper)."""

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.latent = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.latent)
        self.project_forward = _MultiHeadAttention(
            dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = _MultiHeadAttention(
            dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H = self.project_forward(latent_mat, X)
        X_hat = self.project_backward(X, H)
        return X_hat


class _NLP(nn.Module):
    """Non-Local Pooling: a single learnable global token attends over X.

    Exposes its attention weights so the trainer can log them.
    """

    def __init__(self, dim, num_heads, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = _MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        global_embedding = self.S.repeat(X.size(0), 1, 1)
        out, attn_w = self.mha(global_embedding, X, return_attn=True)
        return out, attn_w


class ILRA(nn.Module):
    """Exploring Low-Rank Property in MIL — adapted to HuMP trainer contract."""

    def __init__(self, num_layers=2, feat_dim=768, n_classes=2,
                 hidden_feat=256, num_heads=8, topk=2, ln=False):
        super().__init__()
        gab_blocks = []
        for idx in range(num_layers):
            gab_blocks.append(_GAB(
                dim_in=feat_dim if idx == 0 else hidden_feat,
                dim_out=hidden_feat,
                num_heads=num_heads,
                num_inds=topk,
                ln=ln,
            ))
        self.gab_blocks = nn.ModuleList(gab_blocks)

        self.pooling = _NLP(dim=hidden_feat, num_heads=num_heads, ln=ln)
        self.classifier = nn.Linear(hidden_feat, n_classes)

        self.apply(_init_weights)

    def forward(self, feats):
        # ---- normalise input to [1, N, D] ----
        if feats.dim() == 2:
            x = feats.unsqueeze(0)                       # [1, N, D]
        elif feats.dim() == 3:
            x = feats
        else:
            raise ValueError(f"ILRA: bad input shape {feats.shape}")

        for block in self.gab_blocks:
            x = block(x)                                 # [1, N, H]

        feat, attn = self.pooling(x)                     # feat: [1, 1, H]; attn: [B, 1, N] or None
        logits = self.classifier(feat).squeeze(1)        # [1, n_classes]

        # ---- attention surfaced for the trainer's logger ----
        if attn is not None:
            # nn.MultiheadAttention returns [B, Lq, Lk] = [1, 1, N]; squeeze to [1, N]
            attention = attn.squeeze(1)                  # [1, N]
        else:
            attention = torch.full(
                (1, x.size(1)), 1.0 / x.size(1), device=x.device)

        return logits, logits, attention


# ===========================================================================
# Self-check (run as a script to sanity-test shapes)
# ===========================================================================
if __name__ == "__main__":
    feats = torch.randn(2000, 1024).cuda()

    m_wikg = WiKG(dim_in=1024, dim_hidden=512, topk=6,
                  n_classes=2, agg_type='bi-interaction',
                  dropout=0.3, pool='attn').cuda()
    o = m_wikg(feats)
    print("WiKG :", [t.shape for t in o])    # [[1,2], [1,2], [1,2000]]

    m_ilra = ILRA(feat_dim=1024, n_classes=2,
                  hidden_feat=256, num_heads=8, topk=4).cuda()
    o = m_ilra(feats)
    print("ILRA :", [t.shape for t in o])    # [[1,2], [1,2], [1,2000]]

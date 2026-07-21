from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional
import torch.nn.init as init
from models.manifolds.lorentz import Lorentz
import math

def oxy_angle_pairwise(x, y, curv=1.0, eps=1e-8):
    """
    x: [N, d]
    y: [M, d]
    return: [N, M] angle matrix
    """

    N, d = x.shape
    M, _ = y.shape

    x_ = x.unsqueeze(1)               # [N, 1, d]
    y_ = y.unsqueeze(0)               # [1, M, d]

    # time components
    x_time = torch.sqrt(1 / curv + torch.sum(x_**2, dim=-1))   # [N, 1]
    y_time = torch.sqrt(1 / curv + torch.sum(y_**2, dim=-1))   # [1, M]

    # Lorentz inner product * curvature
    c_xy = curv * (torch.sum(x_ * y_, dim=-1) - x_time * y_time)   # [N, M]

    # components of acos()
    acos_numer = y_time + c_xy * x_time              # [N, M]
    acos_denom = torch.sqrt(torch.clamp(c_xy**2 - 1, min=eps))  # [N, M]

    norm_x = torch.norm(x_, dim=-1)   # [N, 1]

    acos_input = acos_numer / (norm_x * acos_denom + eps)
    acos_input = torch.clamp(acos_input, -1+eps, 1-eps)

    return torch.acos(acos_input)     # [N, M]


# -----------------------------
# 3. Meru half-aperture
# -----------------------------
def half_aperture(x, curv=1.0, min_radius=0.1, eps=1e-8):
    """
    x: [N, d]
    return: [N]
    """
    norm_x = torch.norm(x, dim=-1)            # [N]
    asin_input = 2 * min_radius / (norm_x * curv**0.5 + eps)
    asin_input = torch.clamp(asin_input, -1+eps, 1-eps)
    return torch.asin(asin_input)             # [N]

# def hyperbolic_entailment_loss_pairwise(
#     x: torch.Tensor,  # [1, N, d]
#     y: torch.Tensor,  # [1, M, d]
#     manifold,
#     curv: float = 1.0,
#     min_radius: float = 0.1,
#     eps: float = 1e-6,
#     clamp_val: float = 1 - 1e-6,
# ) -> torch.Tensor:
#     """Numerically stable hyperbolic entailment loss between all pairs (x_i, y_j)."""
#     x = x.squeeze(0)   # [N, d]
#     y = y.squeeze(0)   # [M, d]
#     x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
#     y = y / (y.norm(dim=-1, keepdim=True) + 1e-6)


#     x_h = manifold.expmap0(x)
#     y_h = manifold.expmap0(y)

#     # pairwise angle
#     angle = oxy_angle_pairwise(x_h, y_h, curv)   # [N, M]

#     # half aperture for each x_i, broadcast to [N, M]
#     psi = half_aperture(x_h, curv).unsqueeze(1)  # [N, 1]

#     # entailment: angle <= psi
#     loss = torch.clamp(angle - psi, min=0)

#     return loss.mean()

def hyperbolic_entailment_loss_pairwise(
    x: torch.Tensor,  # [1, N, d]
    y: torch.Tensor,  # [1, M, d]
    manifold,
    curv: float = 1.0,
    min_radius: float = 0.1,
    eps: float = 1e-6,
    clamp_val: float = 1 - 1e-6,
) -> torch.Tensor:
    """Numerically stable hyperbolic entailment loss between all pairs (x_i, y_j)."""
    manifold.expmap0(x)
    manifold.expmap0(y)
    x = x.squeeze(0)  # [N, d]
    y = y.squeeze(0)  # [M, d]

    # ---- half aperture (��(x)) ----
    norm_x = torch.norm(x, dim=-1).clamp_min(min_radius)
    asin_input = 2 * min_radius / (norm_x * curv**0.5 + eps)
    asin_input = torch.clamp(asin_input, min=-clamp_val, max=clamp_val)
    psi_x = torch.asin(asin_input)  # [N]
    psi_x = psi_x.unsqueeze(1)      # [N, 1]

    # ---- pairwise angle ��Oxy ----
    x_ = x.unsqueeze(1)  # [N, 1, d]
    y_ = y.unsqueeze(0)  # [1, M, d]

    # time components (avoid negative inside sqrt)
    x_time = torch.sqrt(torch.clamp(1 / curv + torch.sum(x_**2, dim=-1), min=eps))
    y_time = torch.sqrt(torch.clamp(1 / curv + torch.sum(y_**2, dim=-1), min=eps))

    # Lorentzian inner product �� curvature
    c_xyl = curv * (torch.sum(x_ * y_, dim=-1) - x_time * y_time)
    c_xyl = torch.clamp(c_xyl, min=-1/clamp_val, max=-eps)  # ensure |c_xyl|>1

    # arc-cosine input
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (torch.norm(x_, dim=-1) * acos_denom + eps)
    acos_input = torch.clamp(acos_input, min=-clamp_val, max=clamp_val)

    angle_xy = torch.acos(acos_input)  # [N, M]

    # ---- final loss ----
    loss = F.relu(angle_xy - psi_x)
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    return loss.mean()


def prototype_kmeans(features, K=10, max_iter=100):
    """
    在欧氏空间上进行K-means聚类，返回每个样本的K个原型中心。
    
    参数:
        features: [B, N, D] 欧氏空间特征
        K: 聚类数
        max_iter: 最大迭代次数
    返回:
        prototypes: [B, K, D] 欧氏空间下的聚类中心
    """
    B, N, D = features.shape
    prototypes = []

    for b in range(B):
        x = features[b]  # [N, D]
        # 随机初始化 K 个中心
        idx = torch.randint(0, N, (K,), device=x.device)
        centers = x[idx]

        for _ in range(max_iter):
            # 计算欧式距离 [N, K]
            dists = torch.cdist(x, centers, p=2)
            assign = dists.argmin(dim=1)  # [N]

            new_centers = []
            for k in range(K):
                mask = assign == k
                if mask.sum() == 0:
                    new_centers.append(centers[k])  # 保留旧中心
                else:
                    mean = x[mask].mean(dim=0)
                    new_centers.append(mean)
            new_centers = torch.stack(new_centers)
            
            # 检查收敛
            if torch.allclose(centers, new_centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        prototypes.append(centers)

    prototypes = torch.stack(prototypes)  # [B, K, D]
    return prototypes

def hyperbolic_entailment_completion_strict(
    embed_dict,                   # {"G": ..., "P": ..., "C": ...}, each [B,N,D] / [N,D] / [B,D]
    missing_modality: str,        # "G" | "P" | "C"
    manifold,                     # Lorentz manifold (passed for API compat)
    cone_angle: float = 0.35,     # radians, cone half-angle
    in_scale: float = 0.6,        # missing_modality == 'G' (inner / parent) radius scale
    out_scale: float = 1.2,       # missing_modality == 'C' (outer / child)  radius scale
    between_frac: float = 0.5,    # missing_modality == 'P' (between) linear-interp coefficient
    max_trials: int = 50,         # vectorised: we sample max_trials directions in ONE batched op
    avg_with_prototype: bool = True,   # paper Eq.(Avg): m_final = Avg(m_s, m_proto)
):
    """
    HuMP HGS completion — vectorised + paper-faithful refactor.

    Three cases (parent/child relation per the molecular→tissue→clinical hierarchy):
        missing G : impute "more general" (inside cone(P) ∩ cone(C), smaller norm)
        missing P : impute "between"      (intersection direction, norm between |G| and |C|)
        missing C : impute "more specific"(inside cone(G) ∩ cone(P), larger norm)

    Output: tensor of shape [B, N, D] matching the prototype's N (or the anchors' N
    if prototype has only 1 token).

    Compared to the previous implementation:
      * shapes (B, D) are inferred from inputs, not hard-coded.
      * device is inferred, not hard-coded to 'cuda'.
      * the rejection loop is fully vectorised: max_trials candidates per batch
        element sampled in one go, then first-valid picked per row.
      * the buggy `avg = (cand + cand) / 2` is replaced with the paper's
        m_final = (m_s + m_proto) / 2  (Avg-at-origin surrogate of geodesic mean).
      * dead `num_samples > 1` branch removed.
    """
    # ---------- 0. introspect shapes / device from inputs ----------
    proto = embed_dict[missing_modality]

    # pick any non-prototype anchor to infer (B, D, device).
    # We do NOT raise when proto is None — caller may legitimately not have
    # a prototype yet (e.g. early-epoch inference). We fall back to a zero
    # prototype later, which makes m_final = 0.5 * m_sampled (degenerate but
    # numerically safe).
    ref = None
    for k, v in embed_dict.items():
        if k == missing_modality:
            continue
        if v is not None and isinstance(v, torch.Tensor):
            ref = v
            break
    if ref is None and proto is not None:
        ref = proto
    if ref is None:
        # truly no signal — return an empty zero token so downstream concat
        # doesn't crash (extreme defensive case; should never happen in practice).
        return torch.zeros(1, 1, 256, device='cuda')

    device = ref.device
    D = ref.shape[-1]
    B = ref.shape[0] if ref.dim() == 3 else 1

    # target N: prefer prototype's N; otherwise inherit from a sensible anchor
    if proto is not None and proto.dim() == 3:
        target_n = proto.shape[1]
    elif proto is not None and proto.dim() == 2:
        target_n = proto.shape[0]
    else:
        # no proto -> default to 1 token; downstream attention_fusion handles N=1 fine
        target_n = 1

    cos_thresh = math.cos(cone_angle)

    # ---------- 1. utilities ----------
    def _to_BD(x):
        """Reduce any [B,N,D] / [N,D] / [B,D] tensor to [B, D] by mean over N."""
        if x.dim() == 3:
            return x.mean(dim=1)                          # [B, D]
        if x.dim() == 2:
            # ambiguous: [N,D] or [B,D]; if first dim matches B, treat as [B,D];
            # otherwise reduce over first dim.
            return x if x.shape[0] == B else x.mean(dim=0, keepdim=True)
        return x.view(B, D)

    def _norm_unit(u, eps=1e-8):
        n = u.norm(dim=-1, keepdim=True).clamp_min(eps)
        return u / n, n.squeeze(-1)

    def _vec_rejection_dir(u_a, u_b):
        """Vectorised rejection sampling of a unit direction inside both cones of
        u_a and u_b.  Returns [B, D] unit vector (falls back to bisector if no
        valid sample after max_trials)."""
        a_unit, _ = _norm_unit(u_a)
        b_unit, _ = _norm_unit(u_b)
        base = F.normalize(a_unit + b_unit, dim=-1)        # [B, D]

        # sample max_trials candidates per batch in one batched op
        noise = F.normalize(torch.randn(B, max_trials, D, device=device), dim=-1)
        alpha = torch.rand(B, max_trials, 1, device=device) * 0.25
        cand = F.normalize(
            (1.0 - alpha) * base.unsqueeze(1) + alpha * noise, dim=-1)   # [B, T, D]

        cos_a = (cand * a_unit.unsqueeze(1)).sum(dim=-1)    # [B, T]
        cos_b = (cand * b_unit.unsqueeze(1)).sum(dim=-1)
        ok = (cos_a >= cos_thresh) & (cos_b >= cos_thresh)  # [B, T]

        # first valid trial per batch; if no valid, argmax returns 0 (arbitrary)
        first_idx = ok.float().argmax(dim=1)                # [B]
        any_valid = ok.any(dim=1)                            # [B]
        chosen = cand[torch.arange(B, device=device), first_idx]    # [B, D]
        # fallback to bisector for rows with no valid sample
        return torch.where(any_valid.unsqueeze(-1), chosen, base)

    def _scaled(dir_unit, norm_BD):
        return dir_unit * norm_BD.unsqueeze(-1)             # [B, D]

    # ---------- 2. per-case sampling ----------
    # Helper: when an anchor is None (double-missing step 1 typically passes the
    # proto in place of a missing slot; if proto is also None, the slot ends up
    # as None).  We fall back to a zero tensor on the correct device so the
    # bisector / norm math degenerates gracefully instead of raising.
    def _zero_anchor():
        return torch.zeros(B, D, device=device)

    if missing_modality == "G":
        p_anchor = embed_dict.get("P", None)
        c_anchor = embed_dict.get("C", None)
        if p_anchor is None and c_anchor is None:
            # nothing to anchor on -> emit zero token, paper-Avg with proto will
            # still fold in the prototype below.
            cand_bd = _zero_anchor()
        else:
            if p_anchor is None:
                p_bd = _zero_anchor()
            else:
                p_bd = _to_BD(p_anchor)
            if c_anchor is None:
                c_bd = _zero_anchor()
            else:
                c_bd = _to_BD(c_anchor)
            dir_unit = _vec_rejection_dir(p_bd, c_bd)
            target_norm = torch.min(p_bd.norm(dim=-1), c_bd.norm(dim=-1)) * in_scale
            cand_bd = _scaled(dir_unit, target_norm)

    elif missing_modality == "P":
        g_anchor = embed_dict.get("G", None)
        c_anchor = embed_dict.get("C", None)
        if g_anchor is None and c_anchor is None:
            cand_bd = _zero_anchor()
        else:
            if g_anchor is None:
                g_bd = _zero_anchor()
            else:
                g_bd = _to_BD(g_anchor)
            if c_anchor is None:
                c_bd = _zero_anchor()
            else:
                c_bd = _to_BD(c_anchor)
            dir_unit = _vec_rejection_dir(g_bd, c_bd)
            target_norm = ((1.0 - between_frac) * g_bd.norm(dim=-1)
                           + between_frac * c_bd.norm(dim=-1))
            cand_bd = _scaled(dir_unit, target_norm)

    elif missing_modality == "C":
        g_anchor = embed_dict.get("G", None)
        p_anchor = embed_dict.get("P", None)
        if g_anchor is None and p_anchor is None:
            cand_bd = _zero_anchor()
        else:
            if g_anchor is None:
                g_bd = _zero_anchor()
            else:
                g_bd = _to_BD(g_anchor)
            if p_anchor is None:
                p_bd = _zero_anchor()
            else:
                p_bd = _to_BD(p_anchor)
            dir_unit = _vec_rejection_dir(g_bd, p_bd)
            target_norm = torch.max(g_bd.norm(dim=-1), p_bd.norm(dim=-1)) * out_scale
            cand_bd = _scaled(dir_unit, target_norm)

    else:
        raise ValueError("missing_modality must be one of 'G', 'P', 'C'")

    # ---------- 3. expand to [B, target_n, D] ----------
    cand_BND = cand_bd.unsqueeze(1).expand(-1, target_n, -1).contiguous()    # [B, N, D]

    # ---------- 4. paper Eq.(Avg):  m_final = Avg_G(m_s, m_proto) ----------
    # (Euclidean tangent-at-origin surrogate of the Lorentz geodesic mean —
    # exact for small radii and matches the reduction used elsewhere in HuMP.)
    if avg_with_prototype:
        # zero-tensor fallback when caller did not provide a prototype
        if proto is None:
            proto3 = torch.zeros(B, target_n, D, device=device, dtype=cand_BND.dtype)
        else:
            if proto.dim() == 2:
                proto3 = proto.unsqueeze(0)                   # [N,D] -> [1,N,D]
            elif proto.dim() == 3:
                proto3 = proto
            else:
                proto3 = proto.view(1, 1, -1).expand(B, target_n, D).contiguous()
            # broadcast if proto has B=1 but anchors have B>1
            if proto3.shape[0] != B:
                proto3 = proto3.expand(B, -1, -1)
            # broadcast N if proto has only 1 token
            if proto3.shape[1] != target_n:
                if proto3.shape[1] == 1:
                    proto3 = proto3.expand(-1, target_n, -1)
                else:
                    # fallback: average over proto's N to align
                    proto3 = proto3.mean(dim=1, keepdim=True).expand(-1, target_n, -1)
        out = 0.5 * (cand_BND + proto3)
    else:
        out = cand_BND

    return out

class HypActivation(nn.Module):
    """
    Hyperbolic Activation Layer

    Parameters:
        manifold (Manifold): The manifold to use for the activation.
        activation (function): The activation function.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, activation, manifold_out=None):
        super(HypActivation, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.activation = activation

    def forward(self, x):
        """Forward pass for hyperbolic activation."""
        x_space = x[...,1:]
        x_space = self.activation(x_space)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x

class HypNormalization(nn.Module):
    def __init__(self, manifold, manifold_out=None):
        super(HypNormalization, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out

    def forward(self, x):
        x_space = x[..., 1:]
        x_space = x_space / x_space.norm(dim=-1, keepdim=True)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x
        
class HypLayerNorm(nn.Module):
    def __init__(self, manifold, dim, manifold_out=None):
        super(HypLayerNorm, self).__init__()
        self.in_features = dim
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.layer = nn.LayerNorm(self.in_features)
        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, k=3.0):
        if k != 3.0:
            self.manifold = Lorentz(k=k)
        x_space = x[..., 1:]
        x_space = self.layer(x_space)
        x_time = ((x_space**2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x
        
class HypDropout(nn.Module):
    """
    Hyperbolic Dropout Layer

    Parameters:
        manifold (Manifold): The manifold to use for the dropout.
        dropout (float): The dropout probability.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, dropout, manifold_out=None):
        super(HypDropout, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=False):
        """Forward pass for hyperbolic dropout."""
        if training:
            x_space = x[..., 1:]
            x_space = self.dropout(x_space)
            x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
            if self.manifold_out is not None:
                x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypLinear(nn.Module):
    """
    Hyperbolic Linear Layer

    Parameters:
        manifold (Manifold): The manifold to use for the linear transformation.
        in_features (int): The size of each input sample.
        out_features (int): The size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        dropout (float, optional): The dropout probability. Default is 0.0.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, in_features, out_features, bias=True, dropout=0.0):
        super().__init__()
        self.in_features = in_features + 1  # +1 for time dimension
        self.out_features = out_features
        self.bias = bias
        self.manifold = manifold

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.dropout_rate = dropout
        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if self.bias:
            init.constant_(self.linear.bias, 0)

    def forward(self, x, x_manifold='hyp', k=3.0):
        """Forward pass for hyperbolic linear layer."""
        if x_manifold != 'hyp':
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            x = self.manifold.expmap0(x)

        x_space = self.linear(x)
        
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if x_manifold != 'hyp':
            x = self.manifold.logmap0(x)[...,1:]
        return x

class HypCoAttn(nn.Module):
    def __init__(self, manifold, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([math.sqrt(dim)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.manifold = manifold
        
    def forward(self, qs, ks, vs, output_attn=False):
        # negative squared distance (less than 0)
        att_weight = -self.manifold.cinner(qs, ks)  # [H, N, N]

        att_weight = att_weight / self.scale + self.bias  # [H, N, N]

        att_weight = nn.Softmax(dim=-1)(att_weight)  # [H, N, N]
        att_output = self.manifold.mid_point(vs, att_weight)  # [N, H, D]
        # att_output = att_output.transpose(0, 1)  # [N, H, D]
        # att_output = self.manifold.mid_point(att_output)
        return att_output


class HypAggregation(nn.Module):
    """
    Hyperbolic aggregation layer.
    """
    def __init__(self, manifold, in_features, dropout=0.25, use_att=True, local_agg=True):
        super(HypAggregation, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)
        self.bn = nn.LayerNorm(256)
    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x)[...,1:]
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.expmap(x, support_t)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
                support_t = torch.cat([torch.ones_like(support_t)[..., 0:1], support_t], dim=-1)
        else:
            support_t = torch.spmm(adj.float(), x_tangent)
            support_t = self.bn(support_t)
            support_t = torch.cat([torch.ones_like(support_t)[..., 0:1], support_t], dim=-1)
        output = self.manifold.expmap0(support_t)
        return output

class LConvAttn(nn.Module):
    def __init__(self, manifold, dim=256, kernel=3):
        super().__init__()
        self.manifold = manifold
        self.k =3.0
        self.conv1 = LorentzConv1d(manifold, dim, dim, kernel)
        #self.conv2 = LorentzConv1d(manifold, dim, dim, kernel)
        #self.conv3 = LorentzConv1d(manifold, dim, dim, kernel)
        self.global_attn = HypCoAttn(self.manifold, dim=dim)
        self.global_to_transomic = HypCoAttn(self.manifold, dim=dim)
        self.global_to_genomic = HypCoAttn(self.manifold, dim=dim)
        self.norm = HypNormalization(self.manifold)
        self.LN1 = HypLayerNorm(self.manifold, dim=dim)
        #self.LN2 = HypLayerNorm(self.manifold, dim=dim)
        #self.LN3 = HypLayerNorm(self.manifold, dim=dim)

    def forward(self, x, g_num=6):
        genomic = self.norm(self.tohyp(x[:,:g_num,:]))
        transomic = self.norm(self.tohyp(x[:,g_num:,:]))
        x = self.norm(self.tohyp(x))
        # conv_x = self.conv(x)
        conv_x = x
        attn_x = self.global_attn(conv_x, conv_x, conv_x)
        conv_x = conv_x.narrow(-1, 1, 256) + attn_x.narrow(-1, 1, 256)
        conv_x = self.LN1(self.manifold.add_time(conv_x))

        # attn_g = self.global_to_genomic(conv_x, genomic, genomic)
        # attn_t = self.global_to_transomic(conv_x, transomic, transomic)
        attn_g = self.global_to_genomic(genomic, conv_x, conv_x)
        attn_t = self.global_to_transomic(transomic, conv_x, conv_x)
        # out = self.manifold.mobius_add(x, attn_g)
        # out = self.manifold.mobius_add(out,attn_t)
        # out = self.LN2(out)
        out = x.narrow(-1, 1, 256) +  torch.cat((attn_g, attn_t), dim=-2).narrow(-1, 1, 256)
        out = self.LN1(self.manifold.add_time(out))
        '''x = self.norm(self.tohyp(x))
        
        x_1 = self.conv1(x,self.k)
        x_1 = self.manifold.activation(x_1, nn.ReLU())'''
        
        # x = x + x_1
        # x = self.manifold.add_time(x)
        # x = self.LN1(x,self.k)
        
        out = self.toeuc(x)
        
        return out

    def tohyp(self, x):
        
        x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
        x = self.manifold.expmap0(x)
        return x
    
    def toeuc(self, x):
        x = self.manifold.logmap0(x)[:,:,1:]
        return x
    
    def setk(self, k):
        manifold = Lorentz(k=k)
        
class MINE_1(nn.Module):
    def __init__(self, dim=256):
        super(MINE_1, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 1)
        self.activation = nn.ReLU6()

    def forward(self, x, y):
        joint = torch.cat([x, y], dim=1)
        marginal = torch.cat([x, torch.roll(y, 1, dims=0)], dim=1)
        t1 = self.fc2(self.activation(self.fc1(joint)))
        t2 = self.fc2(self.activation(self.fc1(marginal)))
        mi_lb = torch.mean(t1) - torch.log(torch.mean(torch.exp(t2)))
        return torch.exp(mi_lb)
    
class MINE_2(nn.Module):
    def __init__(self, dim=256):
        super(MINE_2, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 1)
        self.activation = nn.ReLU6()

    def forward(self, x, y):
        joint = torch.cat([x, y], dim=1)
        marginal = torch.cat([x, torch.roll(y, 1, dims=0)], dim=1)
        t1 = self.fc2(self.activation(self.fc1(joint)))
        t2 = self.fc2(self.activation(self.fc1(marginal)))
        mi_lb = torch.mean(t1) - torch.log(torch.mean(torch.exp(t2)))
        return -mi_lb

class LCoMLP(nn.Module):
    def __init__(self, manifold, dim=256, kernel=3):
        super().__init__()
        self.manifold = manifold
        self.k =3.0
        self.proj_g = HypLinear(self.manifold, dim, dim)
        self.proj_t = HypLinear(self.manifold, dim, dim)
        self.proj_p = HypLinear(self.manifold, dim, dim)
        
        self.proj_mg = HypLinear(self.manifold, dim, dim)
        self.proj_mt = HypLinear(self.manifold, dim, dim)
        self.proj_mp = HypLinear(self.manifold, dim, dim)
        
        self.norm = HypNormalization(self.manifold)
        
        self.loss1 = MINE_1(dim=256)
        self.loss2 = MINE_1(dim=256)
        self.loss3 = MINE_1(dim=256)
        
        self.loss4 = MINE_2(dim=256)
        self.loss5 = MINE_2(dim=256)
        
    def forward(self, x, g_num=6, protein=None):
        genomic = self.norm(self.tohyp(x[:,:g_num,:]))
        transomic = self.norm(self.tohyp(x[:,g_num:,:]))
        
        if protein is not None:
            protein = self.norm(self.tohyp(protein))
            g = self.proj_g(genomic)
            t = self.proj_t(transomic)
            p = self.proj_p(protein)
            
            mg = self.proj_mg(genomic)
            mt = self.proj_mt(transomic)
            mp = self.proj_mp(protein)
        
            g_out = self.toeuc(g)
            t_out = self.toeuc(t)
            p_out = self.toeuc(p)
            
            mg_out = self.toeuc(mg)
            mt_out = self.toeuc(mt)
            mp_out = self.toeuc(mp)
            
            
            total_loss = self.loss1(g_out,mg_out) + self.loss2(t_out,mt_out) + self.loss3(p_out,mp_out) + self.loss4(mp_out,mt_out) + self.loss5(mt_out,mg_out)
            total_loss = max(0.0,total_loss)
            out = torch.cat((g_out,t_out,p_out,mg_out,mt_out,mp_out),dim=-2)
        
            return out, total_loss
        else:
            g = self.proj_g(genomic)
            t = self.proj_p(transomic)
            mg = self.proj_mg(genomic)
            mt = self.proj_mt(transomic)
        
            g_out = self.toeuc(g)
            t_out = self.toeuc(t)
            mg_out = self.toeuc(mg)
            mt_out = self.toeuc(mt)
        
            total_loss = self.loss1(g_out,mg_out) + self.loss2(t_out,mt_out) + self.loss3(mg_out,mt_out)
            total_loss = max(0.0,total_loss)
            out = torch.cat((g_out,t_out,mg_out,mt_out),dim=-2)
        
            return out, total_loss

    def tohyp(self, x):
        
        x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
        x = self.manifold.expmap0(x)
        return x
    
    def toeuc(self, x):
        x = self.manifold.logmap0(x)[:,:,1:]
        return x
    
    def setk(self, k):
        self.manifold = Lorentz(k=k)
        for proj in [self.norm, self.proj_g, self.proj_p, self.proj_mg, self.proj_mt]:
            proj.manifold = self.manifold

class HypABMIL(nn.Module):

    def __init__(self, manifold, input_dim=256, hidden_dim=256, dropout=False, n_classes=4, activation='softmax'):
        """
        Attention Network with Sigmoid Gating (3 fc layers). Supports batching 
        args:
            input_dim (int): input feature dimension
            hidden_dim (int): hidden layer dimension
            dropout (bool): whether to use dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(HypABMIL, self).__init__()
        
        self.manifold = manifold
        self.activation = activation
        self.device = 'cuda'
        self.h_soft = HypActivation(self.manifold, nn.Softmax())
        self.attention_a = nn.Sequential(*[
            HypLinear(self.manifold, input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.25)
            ])
        self.attention_b = nn.Sequential(*[HypLinear(self.manifold, hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        HypLinear(self.manifold, hidden_dim, hidden_dim)
                                        ])
        self.attention_c = self.attention = nn.Sequential(*[
            HypLinear(self.manifold, input_dim, hidden_dim), # matrix V
            nn.Tanh(),
            HypLinear(self.manifold, hidden_dim, 1) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        ])

    def forward(self, x):
        """
        Forward pass 
        x List[(torch.Tensor)]: List of [patches x d] w/ len(x) = bs
        """
        x = self.tohyp(x)
        # gated attention 
        a = self.attention_a(x) # [1, N, 257]
        a = a.view(-1, 257)
        b = self.attention_b(a) # [N, 257]
        A = self.attention(x.squeeze(0)) # [N, 2]
        A = torch.transpose(A[...,1:], 1, 0)
          # N x n_classes
        # A = self.manifold.add_time(A)
        if self.activation == 'softmax':
            A = F.softmax(A,dim=-1)
        
        b = torch.mm(A, b[...,1:])
        # print(b.shape)
        b = self.manifold.add_time(b)
        out = self.toeuc(b)

        return out
    
    def tohyp(self, x):
        
        x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
        x = self.manifold.expmap0(x)
        return x
    
    def toeuc(self, x):
        x = self.manifold.logmap0(x)[...,1:]
        return x

    def setk(self, k):
        self.manifold = Lorentz(k=k)
        for layer in [self.h_soft,self.attention_a,self.attention_b,self.attention_c]:
            for module in layer.modules():
                if isinstance(module,HypLinear):
                    module.manifold = self.manifold

class LHypFusion(nn.Module):
    def __init__(self, manifold, dim):
        super().__init__()
        self.manifold = manifold
        
        self.co_attn_p2g = HypCoAttn(self.manifold, dim=dim)
        self.co_attn_p2t = HypCoAttn(self.manifold, dim=dim)
        
        self.co_attn_t2p = HypCoAttn(self.manifold, dim=dim)
        self.co_attn_g2p = HypCoAttn(self.manifold, dim=dim)
        
        
        self.norm = HypNormalization(self.manifold)
        self.LN1 = HypLayerNorm(self.manifold, dim=dim)
        self.LN2 = HypLayerNorm(self.manifold, dim=dim)
        self.LN3 = HypLayerNorm(self.manifold, dim=dim)
        
    def forward(self, x, g_num=6, p_num=4096):
        g = x[:,:g_num,:]
        p = x[:,g_num:g_num+p_num,:]

        g = self.norm(self.tohyp(g))
        p = self.norm(self.tohyp(p))
        
        p_x = p + self.co_attn_p2g(vs=g, ks=g, qs=p)
        p_x = self.LN1(p_x)
        
        g_x = g + self.co_attn_g2p(vs=p, ks=p, qs=g)
        g_x = self.LN2(g_x)

        output = torch.cat((g_x, p_x), dim=1)    
        output = self.toeuc(output)
        
        return output
    
    def tohyp(self, x):
        
        x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
        x = self.manifold.expmap0(x)
        return x
    
    def toeuc(self, x):
        x = self.manifold.logmap0(x)[:,:,1:]
        return x
        
class HierarchicalBiFusion(nn.Module):
    """
    A Hierarchical and Bidirectional Fusion model for three modalities:
    Genomics (g), Pathology/WSI (p), and Clinical (c).
    Fusion happens in Hyperbolic space.
    """
    def __init__(self, manifold, dim):
        super().__init__()
        self.manifold = manifold
        self.norm = HypNormalization(self.manifold)
        # === Attention Modules for Bidirectional Fusion ===

        # Stage 1: Genomics <--> Pathology
        # Bottom-up (g -> p): Use genomics to enhance pathology
        self.co_attn_g_to_p = HypCoAttn(self.manifold, dim=dim)
        # Top-down (p -> g): Use pathology to guide genomics
        self.co_attn_p_to_g = HypCoAttn(self.manifold, dim=dim)

        # Stage 2: Pathology <--> Clinical
        # Bottom-up (p -> c): Use pathology to enhance clinical
        self.co_attn_p_to_c = HypCoAttn(self.manifold, dim=dim)
        # Top-down (c -> p): Use clinical to guide pathology
        self.co_attn_c_to_p = HypCoAttn(self.manifold, dim=dim)
        
        # === Normalization Layers ===
        # Use separate LayerNorm for each fusion step to maintain stability
        self.LN_p1 = HypLayerNorm(self.manifold, dim=dim)
        self.LN_g1 = HypLayerNorm(self.manifold, dim=dim)
        self.LN_p2 = HypLayerNorm(self.manifold, dim=dim)
        self.LN_c1 = HypLayerNorm(self.manifold, dim=dim)
        
    def forward(self, g, p, c):
        """
        Input features for the three modalities.
        g: Genomics features [B, num_genes, D]
        p: Pathology features [B, num_patches, D]
        c: Clinical features  [B, num_clinical, D]
        """
        
        # --- 1. Project all features to Hyperbolic space ---
        g_hyp = self.norm(self.tohyp(g))
        p_hyp = self.norm(self.tohyp(p))
        c_hyp = self.norm(self.tohyp(c))

        # --- 2. Stage 1 Fusion: Genomics (g) <--> Pathology (p) ---
        
        # Bottom-up (Micro -> Macro): g -> p
        # Enhance pathology features with information from genomics
        p_hyp_g = p_hyp + self.co_attn_g_to_p(qs=p_hyp, ks=g_hyp, vs=g_hyp) + self.co_attn_c_to_p(qs=p_hyp, ks=c_hyp, vs=c_hyp)
        p_hyp_g = self.LN_p1(p_hyp_g)
        
        # Top-down (Macro -> Micro): p -> g
        # Guide genomics features using the context from pathology
        g_hyp_p = g_hyp + self.co_attn_p_to_g(qs=g_hyp, ks=p_hyp, vs=p_hyp) 
        g_hyp_p = self.LN_g1(g_hyp_p)

        # --- 3. Stage 2 Fusion: Pathology (p) <--> Clinical (c) ---
        # Note: We use the already enhanced feature p_hyp_g from the previous stage.
        
        # Bottom-up (Micro -> Macro): p -> c
        # Enhance clinical features with information from genomics-enhanced pathology
        c_hyp_p = c_hyp + self.co_attn_p_to_c(qs=c_hyp, ks=p_hyp_g, vs=p_hyp)
        c_hyp_p = self.LN_c1(c_hyp_p)

        # --- 4. Project final fused features back to Euclidean space ---
        # At this point, we have three mutually informed feature sets:
        # g_hyp_p: Genomics, informed by Pathology
        # p_hyp_gc: Pathology, informed by Genomics and Clinical
        # c_hyp_p: Clinical, informed by Pathology (which contains Genomics info)
        
        g_final = self.toeuc(g_hyp_p)
        p_final = self.toeuc(p_hyp_g)
        c_final = self.toeuc(c_hyp_p)

        # --- 5. Concatenate for final output ---
        output = torch.cat((g_final, p_final, c_final), dim=1)
        
        return output
    
    def tohyp(self, x):
        # Helper to project Euclidean -> Hyperbolic
        # Adds a dimension for the hyperbolic space coordinate
        x_h = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
        x_h = self.manifold.expmap0(x_h)
        return x_h
    
    def toeuc(self, x_h):
        # Helper to project Hyperbolic -> Euclidean
        # Removes the extra hyperbolic coordinate
        x = self.manifold.logmap0(x_h)[..., 1:]
        return x

class SAFusion(nn.Module):
    def __init__(self, manifold, dim):
        super().__init__()
        self.manifold = manifold
        
        self.attn = HypCoAttn(self.manifold, dim=dim)
        self.norm = HypNormalization(self.manifold)
        self.LN = HypLayerNorm(self.manifold, dim=dim)
        
    def forward(self, g, p, c=None):
        if c is not None:
            x_cat = torch.cat((g, p, c), dim=1)
        else:
            x_cat = torch.cat((g, p), dim=1)
        x_hyp = self.tohyp(x_cat)
        x_hyp = self.norm(x_hyp)

        attn_output = self.attn(qs=x_hyp, ks=x_hyp, vs=x_hyp)
        x_hyp = self.manifold.mobius_add(x_hyp, attn_output) 
        x_hyp = self.LN(x_hyp)

        output = self.toeuc(x_hyp)
        
        return output
    
    def tohyp(self, x):
        x_h = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
        x_h = self.manifold.expmap0(x_h)
        return x_h
    
    def toeuc(self, x_h):
        x = self.manifold.logmap0(x_h)[..., 1:]
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalLatent(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        # x shape: [B, N, D]
        return self.net(x)

class CausalWSIModulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 因为 z_g 和 z_c 拼接后是 2*dim
        self.gamma_net = nn.Linear(dim * 2, dim)
        self.beta_net = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, p, z_g_global, z_c_global):
        """
        p: [B, N_p, D] - 病理 patch 序列
        z_g_global: [B, D] - 基因全局向量
        z_c_global: [B, D] - 临床全局向量
        """
        # 1. 拼接因果原因 (Causes)
        z = torch.cat([z_g_global, z_c_global], dim=-1) # [B, 2*D]

        # 2. 生成调制参数 (基于全局背景)
        gamma = torch.sigmoid(self.gamma_net(z)).unsqueeze(1) # [B, 1, D]
        beta = self.beta_net(z).unsqueeze(1)                  # [B, 1, D]

        # 3. 对病理的每一个 patch 进行调制 (Broadcasting)
        # 含义：基因和临床特征决定了病理图像中哪些形态特征（channels）更具危险性
        p_hat = gamma * p + beta 
        return self.norm(p_hat), z

class SCFusion(nn.Module):
    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.dim = dim
        
        # 1. 潜变量提取 (处理序列)
        self.zg_net = CausalLatent(dim)
        self.zc_net = CausalLatent(dim)
        self.zp_net = CausalLatent(dim) # 增加对病理的初步映射

        # 2. 特征调制
        self.modulator = CausalWSIModulation(dim)
        
        # 3. 因果重构投影 (辅助任务)
        self.causal_reconstructor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # 4. 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1)
        )

    def forward(self, g, p, c):
        """
        g: [B, Ng, D], p: [B, Np, D], c: [B, Nc, D]
        """
        # --- Step 1: 序列映射与聚合 ---
        # 映射
        h_g = self.zg_net(g) # [B, Ng, D]
        h_p = self.zp_net(p) # [B, Np, D]
        h_c = self.zc_net(c) # [B, Nc, D]
        
        # 聚合 (Mean Pooling) 得到全局表示，用于构建因果关系
        # 即使 N 不同，聚合后都变成 [B, D]
        z_g_global = h_g.mean(dim=1) 
        z_c_global = h_c.mean(dim=1)
        z_p_global = h_p.mean(dim=1)

        # --- Step 2: 因果调制 (Modulation) ---
        # 用全局的 G 和 C 去调制病理的所有 patch
        p_hat_seq, z_combined = self.modulator(h_p, z_g_global, z_c_global)
        
        # 调制后的病理也聚合为全局向量
        p_hat_global = p_hat_seq.mean(dim=1) # [B, D]

        # --- Step 3: 因果约束 (G, C -> P) ---
        # 强制要求全局基因和临床特征能够重构出全局病理特征
        p_global_recon = self.causal_reconstructor(z_combined)
        loss_causal = F.mse_loss(p_global_recon, z_p_global.detach())

        # --- Step 4: 最终融合 (Final Fusion) ---
        # 拼接三个全局向量：基因、临床、受调制的病理
        final_feat = torch.cat([z_g_global, z_c_global, p_hat_global], dim=-1) # [B, 3*D]

        return final_feat, loss_causal


class SCMFusion(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.dim = dim
        
        # ---------------------------
        # 1. 特征编码器 (Encoders)
        # ---------------------------
        # 将各模态映射到统一的因果隐空间
        self.enc_g = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.enc_p = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.enc_c = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        
        # ---------------------------
        # 2. 结构方程模型 (Structural Equation: G + C -> P)
        # ---------------------------
        # 这是一个生成器，试图用基因和临床特征“画出”病理特征
        # 生物学假设：Phenotype = Function(Genotype, Environment)
        self.mechanism_P = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim), # 输出 P_expected
            nn.Tanh() # 假设特征由Tanh激活，这里保持一致
        )
        
        # ---------------------------
        # 3. 因果门控融合 (Causal Gated Fusion)
        # ---------------------------
        # 我们不仅要融合，还要根据因果链赋予权重
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, g, p, c=None):
        """
        Inputs:
            g: [B, dim] - Genomic features
            p: [B, dim] - Pathological features
            c: [B, dim] - Clinical features (optional)
        Returns:
            logits: 生存预测分数
            structural_loss: 结构因果一致性损失
        """
        # --- Step 1: Embedding ---
        h_g = self.enc_g(g)
        h_p = self.enc_p(p)
        
        if c is not None:
            h_c = self.enc_c(c)
        else:
            h_c = torch.zeros_like(h_g).to(g.device) # 缺失值处理

        # --- Step 2: Causal Mechanism Modeling (G, C -> P) ---
        # 拼接 G 和 C 作为原因
        cause_gc = torch.cat([h_g, h_c], dim=1)
        
        # 预测预期的病理特征 (P_expected)
        # 这代表了："根据该患者的基因突变和临床状态，他的病理图像理应长什么样"
        h_p_expected = self.mechanism_P(cause_gc)
        
        # 计算结构损失 (Structural Loss)
        # 强制网络学习 G/C 到 P 的映射关系
        loss_mechanism = F.mse_loss(h_p_expected, h_p)

        # --- Step 3: Orthogonal Decomposition (关键步骤) ---
        # 计算残差 (P_unique)
        # 这代表了："病理图像中特有的、无法被基因解释的异质性信息"
        # 例如：某些特异性的组织纹理或无法被已知基因捕获的微环境变化
        h_p_unique = h_p - h_p_expected
        
        # 可选：对残差进行正交化约束，确保 P_unique 不包含 G, C 的信息 (解纠缠)
        # loss_ortho = torch.mean(torch.sum(h_p_unique * h_p_expected, dim=1)**2)

        # --- Step 4: Final Fusion for Prognosis ---
        # 我们使用 G, C (原始原因) 和 P_unique (独立图像信息) 进行预测
        # 这样避免了 P 中包含的 G 信息被重复计算 (Double Counting)
        
        # 构造序列用于 Attention: [G, C, P_unique]
        # 注意这里用的是 h_p_unique 而不是原始 h_p
        seq = torch.stack([h_g, h_c, h_p_unique], dim=1)
        
        # Self-Attention 捕获三者对预后的动态贡献
        attn_out, _ = self.attention(query=seq, key=seq, value=seq)
        
        return attn_out, loss_mechanism

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.manifolds.lorentz import Lorentz

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


def half_aperture(x, curv=1.0, min_radius=0.1, eps=1e-8):
    """
    x: [N, d]
    return: [N]
    """
    norm_x = torch.norm(x, dim=-1)            # [N]
    asin_input = 2 * min_radius / (norm_x * curv**0.5 + eps)
    asin_input = torch.clamp(asin_input, -1+eps, 1-eps)
    return torch.asin(asin_input)             # [N]

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

    # ---- half aperture psi(x) ----
    norm_x = torch.norm(x, dim=-1).clamp_min(min_radius)
    asin_input = 2 * min_radius / (norm_x * curv**0.5 + eps)
    asin_input = torch.clamp(asin_input, min=-clamp_val, max=clamp_val)
    psi_x = torch.asin(asin_input)  # [N]
    psi_x = psi_x.unsqueeze(1)      # [N, 1]

    # ---- pairwise angle Oxy ----
    x_ = x.unsqueeze(1)  # [N, 1, d]
    y_ = y.unsqueeze(0)  # [1, M, d]

    # time components (avoid negative inside sqrt)
    x_time = torch.sqrt(torch.clamp(1 / curv + torch.sum(x_**2, dim=-1), min=eps))
    y_time = torch.sqrt(torch.clamp(1 / curv + torch.sum(y_**2, dim=-1), min=eps))

    # Lorentzian inner product scaled by curvature
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
    """K-means clustering in Euclidean space; returns K prototype centres per sample.

    Args:
        features (Tensor): [B, N, D] Euclidean features.
        K        (int):    number of clusters.
        max_iter (int):    maximum iterations.

    Returns:
        prototypes (Tensor): [B, K, D] cluster centres.
    """
    B, N, D = features.shape
    prototypes = []

    for b in range(B):
        x = features[b]  # [N, D]
        # randomly initialise K cluster centres
        idx = torch.randint(0, N, (K,), device=x.device)
        centers = x[idx]

        for _ in range(max_iter):
            # pairwise Euclidean distance [N, K]
            dists = torch.cdist(x, centers, p=2)
            assign = dists.argmin(dim=1)  # [N]

            new_centers = []
            for k in range(K):
                mask = assign == k
                if mask.sum() == 0:
                    new_centers.append(centers[k])  # keep old centre if cluster is empty
                else:
                    new_centers.append(x[mask].mean(dim=0))
            new_centers = torch.stack(new_centers)

            # check convergence
            if torch.allclose(centers, new_centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        prototypes.append(centers)

    prototypes = torch.stack(prototypes)  # [B, K, D]
    return prototypes

def hyperbolic_entailment_completion_strict(
    embed_dict,            # dict: {"G": [B,D] or None, "W": [B,D] or None, "C": [B,D] or None}
    missing_modality: str, # "G" | "W" | "C"
    manifold,              # Lorentz manifold instance with logmap0, expmap0, expmap, logmap
    cone_angle=0.35,       # radians: half-angle of cone
    in_scale=0.6,          # when sampling "inner" (more general), scale relative to min radius
    out_scale=1.2,         # when sampling "outer" (more specific), scale relative to max radius
    between_frac=0.5,      # fraction for between-position (0->near source,1->near target)
    num_samples=1,
    max_trials=50
):
    """Strict cone-intersection based entailment completion.

    Three missing-modality cases:
      - missing G: P and C jointly entail G -> sample G' inside cone(P) ∩ cone(C),
                   closer to origin (*more general*).
      - missing P: G entails P, P entails C -> sample P' with norm between |G| and |C|,
                   direction aligned with both.
      - missing C: G and P jointly entail C -> sample C' inside cone(G) ∩ cone(P),
                   farther from origin (*more specific*).

    Returns:
        Tensor of shape [B, target_n, D].

    Notes:
        Assumes batch size B=1 and feature dimension D=256.
        Cone membership is checked in the tangent space (Euclidean) via cosine threshold.
    """
    device = 'cuda'
    B, D = 1, 256  # assumed batch size and feature dimension

    def normalize_vec(u, eps=1e-8):
        norm = u.norm(dim=-1, keepdim=True).clamp(min=eps)
        return u / norm, norm.squeeze(-1)  # returns (unit_vec [B,D], norm [B])

    def in_cone(candidate_dir, base_dir, cos_threshold):
        # returns bool mask [B]: True if angle(candidate_dir, base_dir) <= cone_angle
        c_unit, _ = normalize_vec(candidate_dir)
        b_unit, _ = normalize_vec(base_dir)
        cos = (c_unit * b_unit).sum(dim=-1)
        return cos >= cos_threshold  # True if angle <= cone_angle

    # precompute cos threshold
    cos_thresh = math.cos(cone_angle)

    def sample_dir_intersection(u_a, u_b):
        """Sample a unit direction near the intersection of cone(u_a) and cone(u_b)
        via rejection sampling around the bisector direction."""
        a_unit, _ = normalize_vec(u_a)
        b_unit, _ = normalize_vec(u_b)
        base_dir = F.normalize(a_unit + b_unit, dim=-1)  # bisector [B, D]

        samples = []
        trials = 0
        while len(samples) < num_samples and trials < max_trials:
            trials += 1
            noise = F.normalize(torch.randn(B, D, device=device), dim=-1)
            alpha = torch.rand(B, 1, device=device) * 0.25
            cand = F.normalize((1 - alpha) * base_dir + alpha * noise, dim=-1)
            ok = in_cone(cand, u_a, cos_thresh) & in_cone(cand, u_b, cos_thresh)
            if ok.any():
                samples.append(cand)

        if len(samples) == 0:
            # fallback: use the bisector if rejection sampling fails
            return base_dir.unsqueeze(1).expand(-1, num_samples, -1) if num_samples > 1 else base_dir

        stacked = torch.stack(samples, dim=1)  # [B, ntrials, D]
        cos_a = torch.einsum('btd,bd->bt', stacked, F.normalize(u_a, dim=-1))
        cos_b = torch.einsum('btd,bd->bt', stacked, F.normalize(u_b, dim=-1))
        mask = (cos_a >= cos_thresh) & (cos_b >= cos_thresh)  # [B, ntrials]
        first_idx = mask.float().argmax(dim=1)
        selected = stacked[torch.arange(B, device=device), first_idx]  # [B, D]
        if num_samples == 1:
            return selected
        return selected.unsqueeze(1).expand(-1, num_samples, -1)

    def sample_in_intersection_outer(u_a, u_b):
        """Sample inside cone(u_a) ∩ cone(u_b) with norm > max(|u_a|, |u_b|) (more specific)."""
        cand_dir = sample_dir_intersection(u_a, u_b)
        target_norm = torch.max(u_a.norm(dim=-1), u_b.norm(dim=-1)) * out_scale  # [B]
        if num_samples == 1:
            return F.normalize(cand_dir, dim=-1) * target_norm.unsqueeze(-1)
        cand_unit = F.normalize(cand_dir.view(-1, D), dim=-1).view(B, num_samples, D)
        return cand_unit * target_norm.unsqueeze(1).unsqueeze(-1)

    def sample_in_intersection_inner(u_a, u_b):
        """Sample inside cone(u_a) ∩ cone(u_b) with norm < min(|u_a|, |u_b|) (more general)."""
        cand_dir = sample_dir_intersection(u_a, u_b)
        target_norm = torch.min(u_a.norm(dim=-1), u_b.norm(dim=-1)) * in_scale  # [B]
        if num_samples == 1:
            return F.normalize(cand_dir, dim=-1) * target_norm.unsqueeze(-1)
        cand_unit = F.normalize(cand_dir.view(-1, D), dim=-1).view(B, num_samples, D)
        return cand_unit * target_norm.unsqueeze(1).unsqueeze(-1)

    def sample_between(u_g, u_c):
        """Sample a direction aligned with both u_g and u_c, with norm interpolated between them."""
        dir_base = F.normalize(F.normalize(u_g, dim=-1) + F.normalize(u_c, dim=-1), dim=-1)
        selected_dir = dir_base
        for _ in range(max_trials):
            ok_g = F.cosine_similarity(selected_dir, F.normalize(u_g, dim=-1), dim=-1) >= cos_thresh
            ok_c = F.cosine_similarity(selected_dir, F.normalize(u_c, dim=-1), dim=-1) >= cos_thresh
            if (ok_g & ok_c).all():
                break
            noise = F.normalize(torch.randn(B, D, device=device), dim=-1)
            alpha = torch.rand(B, 1, device=device) * 0.2
            selected_dir = F.normalize((1 - alpha) * dir_base + alpha * noise, dim=-1)
        new_norm = (1 - between_frac) * u_g.norm(dim=-1) + between_frac * u_c.norm(dim=-1)
        if num_samples == 1:
            return selected_dir * new_norm.unsqueeze(-1)
        return selected_dir.unsqueeze(1).expand(-1, num_samples, -1) * new_norm.unsqueeze(1).unsqueeze(-1)

    # Look up the prototype for the missing modality to determine output sequence length.
    prototype = embed_dict[missing_modality]
    if prototype is None:
        raise ValueError(f"Prototype for '{missing_modality}' is required to set the output shape.")
    target_n = prototype.shape[0]  # target sequence length for the completed modality

    if missing_modality == "G":
        if embed_dict["P"] is None or embed_dict["C"] is None:
            raise ValueError("Need both P and C to complete G.")
        w = torch.mean(embed_dict["P"], dim=1)
        c = torch.mean(embed_dict["C"], dim=1)
        cand_tangent_single = sample_in_intersection_inner(w, c)  # [B, D]

    elif missing_modality == "P":
        if embed_dict["G"] is None or embed_dict["C"] is None:
            raise ValueError("Need both G and C to complete P.")
        g = torch.mean(embed_dict["G"], dim=1)
        c = torch.mean(embed_dict["C"], dim=1)
        cand_tangent_single = sample_between(g, c)  # [B, D]

    elif missing_modality == "C":
        if embed_dict["G"] is None or embed_dict["P"] is None:
            raise ValueError("Need both G and P to complete C.")
        g = torch.mean(embed_dict["G"], dim=1)
        w = torch.mean(embed_dict["P"], dim=1)
        cand_tangent_single = sample_in_intersection_outer(g, w)  # [B, D]

    else:
        raise ValueError("missing_modality must be 'G', 'P', or 'C'.")

    # Expand the single completed vector to target sequence length.
    return cand_tangent_single.unsqueeze(1).expand(-1, target_n, -1)

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
        return att_output

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

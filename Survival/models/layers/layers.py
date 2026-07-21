import torch
import torch.nn as nn
import torch.nn.functional as F


class KGGenerator(nn.Module):
    """Curvature scalar generator used by legacy H2Surv experiments."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x, *args, **kwargs):
        return self.net(x.mean(dim=1)).mean()


class KPGenerator(KGGenerator):
    pass


class GeneFusion(nn.Module):
    """Lightweight token self-attention used by legacy H2Surv code paths."""

    def __init__(self, embedding_dim=256, num_heads=4, num_pathways=6):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class Gating(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_inputs))

    def forward(self, xs):
        weights = F.softmax(self.logits, dim=0)
        return sum(w * x for w, x in zip(weights, xs))


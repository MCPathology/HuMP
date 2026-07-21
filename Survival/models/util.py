import torch
import torch.nn as nn


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SNN_Block(nn.Module):
    """Self-normalizing block used for pathway-level omics encoders."""

    def __init__(self, dim1, dim2, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class HypSNNBlock(nn.Module):
    """Euclidean pathway encoder with a Lorentz-compatible interface."""

    def __init__(self, manifold, in_dim, out_dim=256, dropout=0.25):
        super().__init__()
        self.manifold = manifold
        self.block = SNN_Block(in_dim, out_dim, dropout=dropout)

    def forward(self, x):
        return self.block(x)


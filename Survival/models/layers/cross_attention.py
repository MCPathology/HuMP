import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward block for modality-token fusion."""

    def __init__(self, dim, hidden_dim=None, dropout=0.25):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


import torch
import torch.nn.functional as F


class Lorentz:
    """Minimal Lorentz-manifold helper used by HuMP layers.

    The model code stores hyperbolic points as ``[time, spatial...]`` tensors.
    These utilities keep tensors on the Lorentz sheet while exposing the small
    operation set used by the fusion and HGS modules.
    """

    def __init__(self, k=1.0, eps=1e-8):
        self.k = k
        self.eps = eps

    def _k_like(self, x):
        if torch.is_tensor(self.k):
            return self.k.to(device=x.device, dtype=x.dtype)
        return torch.tensor(float(self.k), device=x.device, dtype=x.dtype)

    def add_time(self, x):
        k = self._k_like(x)
        spatial = x[..., 1:] if x.shape[-1] > 0 and x.shape[-1] != 256 else x
        time = torch.sqrt(torch.clamp((spatial * spatial).sum(dim=-1, keepdim=True) + k, min=self.eps))
        return torch.cat([time, spatial], dim=-1)

    def expmap0(self, u):
        spatial = u[..., 1:] if u.shape[-1] != 256 else u
        return self.add_time(spatial)

    def logmap0(self, x):
        if x.shape[-1] == 256:
            return self.add_time(x)
        return x

    def cinner(self, x, y):
        x_h = self.add_time(x) if x.shape[-1] == 256 else x
        y_h = self.add_time(y) if y.shape[-1] == 256 else y
        x_time, x_space = x_h[..., :1], x_h[..., 1:]
        y_time, y_space = y_h[..., :1], y_h[..., 1:]
        return torch.matmul(x_space, y_space.transpose(-1, -2)) - torch.matmul(x_time, y_time.transpose(-1, -2))

    def mid_point(self, x, weights=None):
        x_h = self.add_time(x) if x.shape[-1] == 256 else x
        if weights is None:
            spatial = x_h[..., 1:].mean(dim=-2, keepdim=True)
        else:
            spatial = torch.matmul(weights, x_h[..., 1:])
        return self.add_time(spatial)

    def mobius_add(self, x, y):
        x_h = self.add_time(x) if x.shape[-1] == 256 else x
        y_h = self.add_time(y) if y.shape[-1] == 256 else y
        return self.add_time(x_h[..., 1:] + y_h[..., 1:])

    def expmap(self, x, u):
        x_h = self.add_time(x) if x.shape[-1] == 256 else x
        return self.add_time(x_h[..., 1:] + u)

    def logmap(self, x, y):
        x_h = self.add_time(x) if x.shape[-1] == 256 else x
        y_h = self.add_time(y) if y.shape[-1] == 256 else y
        return y_h[..., 1:] - x_h[..., 1:]

    def activation(self, x, activation):
        x_h = self.add_time(x) if x.shape[-1] == 256 else x
        return self.add_time(activation(x_h[..., 1:]))

    def normalize(self, x):
        x_h = self.add_time(x) if x.shape[-1] == 256 else x
        spatial = F.normalize(x_h[..., 1:], dim=-1)
        return self.add_time(spatial)


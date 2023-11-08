import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True):
        super().__init__()
        self.scale = dim**-0.5
        self.weight = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        x = x / norm.clamp(min=1e-8)
        if self.weight is not None:
            x *= self.weight
        return x


class LayerNorm(nn.Module):
    """LayerNorm but with scale/bias individually toggleable"""

    def __init__(self, dim, scale=True, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim)) if scale else None
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-8)


def get_norm(name, dim):
    if name == "ln":
        return LayerNorm(dim)
    elif name == "ln_nobias":
        return LayerNorm(dim, bias=False)
    elif name == "rms":
        return RMSNorm(dim)
    elif name == "none":
        return nn.Identity()
    else:
        raise Exception("Unknown norm type '{name}'")

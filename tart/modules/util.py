import math

import torch
import torch.nn.functional as F
from torch import nn


def l2norm(x, dim=-1, scaled=False):
    ret = nn.functional.normalize(x, p=2, dim=dim)
    if scaled:
        ret *= ret.shape[dim] ** 0.5
    return ret


def time_shift(x):
    C = x.shape[-1]
    ts = nn.functional.pad(x, (0, 0, 1, -1))
    x = torch.cat([ts[:, :, : C // 2], x[:, :, C // 2 :]], dim=-1)
    return x


def scale_gradient(x, amount=0.1):
    return x * amount + (1 - amount) * x.detach()


class ScaleGradient(nn.Module):
    def __init__(self, frac):
        super().__init__()
        self.frac = frac

    def forward(self, x):
        return x * self.frac + (1 - self.frac) * x.detach()


class PadLeft(nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, x):
        return F.pad(x, (self.amount, 0))


class LearnedPadLeft(nn.Module):
    def __init__(self, dim, amount):
        super().__init__()
        self.amount = amount
        self.pad_emb = nn.Parameter(torch.zeros(1, amount, dim))

    def forward(self, x):
        return torch.cat([self.pad_emb.expand(x.shape[0], -1, -1), x], dim=-2)


class TimeShift(nn.Module):
    def __init__(self, dim, dim_frac=0.25):
        super().__init__()
        self.shift_dim = max(1, int(dim * dim_frac))
        self.pad = LearnedPadLeft(self.shift_dim, 1)

    def forward(self, x):
        shifted = self.pad(x[..., :-1, : self.shift_dim])
        return torch.cat([shifted, x[:, :, self.shift_dim :]], dim=-1)


class TimeShift2(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        with torch.no_grad():
            w = torch.ones(1, 1, hdim)
            for i in range(hdim):
                w[:, :, i] = i / hdim
            self.weight = nn.Parameter(w)

    def forward(self, x):
        shifted = F.pad(x, (0, 0, 1, -1))
        mix_w = self.weight
        x = x * mix_w + shifted * (1 - mix_w)
        return x


class DimExpand(nn.Module):
    def __init__(self, hdim, expand_to, norm_cls):
        super().__init__()
        self.expander = nn.Sequential(
            norm_cls(hdim),
            nn.Linear(hdim, expand_to - hdim),
        )

    def forward(self, x):
        return torch.cat([x, self.expander(x)], dim=-1)


class L2Norm(nn.Module):
    def forward(self, x):
        return l2norm(x)


class Affine(nn.Module):
    def __init__(self, hdim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hdim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(hdim))
        else:
            self.bias = None

    def forward(self, x):
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class LeftPaddedConv(nn.Module):
    def __init__(self, dim_in, dim_out, ksize, stride, pad, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, dim_out, ksize, stride, 0, bias=bias)
        self.pad_emb = nn.Parameter(torch.zeros(1, dim_in, pad))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([self.pad_emb.expand(x.shape[0], -1, -1), x], dim=-1)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class Reshape(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return x.reshape(*self.args, **self.kwargs)

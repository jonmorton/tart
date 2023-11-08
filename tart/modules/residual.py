import torch
from torch import nn

from .norm import get_norm
from .util import TimeShift


class Residual(nn.Module):
    def __init__(
        self,
        module,
        dim,
        norm_type="ln",
        prenorm=True,
        postnorm=False,
        scale=False,
        gate=False,
        time_shift=False,
    ):
        super().__init__()
        self.module = module
        self.norm = get_norm(norm_type, dim) if prenorm else nn.Identity()
        self.norm2 = get_norm(norm_type, dim) if postnorm else nn.Identity()
        self.gate = gate
        self.time_shift = time_shift
        self.scale = scale

        if gate:
            self.gate_fc = nn.Linear(dim, dim)
        if scale:
            self.scale_p = nn.Parameter(torch.full([1], 0.5))
        if time_shift:
            self.time_shifter = TimeShift(dim)

    def init_weights(self):
        if self.gate:
            nn.init.zeros_(self.gate_fc.weight)

    def forward(self, x):
        skip = x

        if self.time_shift:
            x = self.time_shifter(x)

        x = self.norm(x)

        if self.gate:
            g = torch.sigmoid(self.gate_fc(x))

        x = self.module(x)

        if self.gate:
            x = x * g

        x = self.norm2(x)

        if self.scale:
            x = x * self.scale_p

        return x + skip

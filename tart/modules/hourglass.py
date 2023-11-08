import nnt
import torch
from torch import nn

from .transformer import build_transformer
from .util import LearnedPadLeft


class Resampler(nn.Module):
    def downsample(self, x):
        raise NotImplementedError()

    def upsample(self, x, downsample_info=None):
        raise NotImplementedError()


class FixedResampler(Resampler):
    def __init__(self, shorten_factor, dim_hi, dim_lo):
        super().__init__()
        self.shorten_factor = shorten_factor
        self.shift = LearnedPadLeft(dim_hi, shorten_factor - 1)
        self.proj_down = nn.Linear(dim_hi * shorten_factor, dim_lo)
        self.proj_up = nn.Linear(dim_lo, dim_hi * shorten_factor)

        self.n_reg = 8
        self.reg = nn.Parameter(
            torch.randn((1, self.n_reg, self.dim_lo)) / (dim_lo**0.5)
        )

    def downsample(self, x):
        # x = self.shift(x)
        seq_len = x.shape[-2]
        pad_amt = nnt.round_to_multiple(seq_len, self.shorten_factor, up=True) - seq_len
        x = nn.functional.pad(x, (0, 0, 0, pad_amt), value=0.0)
        x = self.shift(x[..., : -(self.shorten_factor - 1), :])
        x = x.view(
            -1, x.shape[-2] // self.shorten_factor, x.shape[-1] * self.shorten_factor
        )
        x = self.proj_down(x)
        x = torch.cat([x, self.reg.expand(x.shape[0], -1, -1)], dim=-2)
        return x, pad_amt

    def upsample(self, x, downsample_pad_amount):
        x = x[..., self.n_reg :, :]
        seq_len = x.shape[-2]
        x = self.proj_up(x)
        x = x.view(
            -1, seq_len * self.shorten_factor, x.shape[-1] // self.shorten_factor
        )
        if downsample_pad_amount > 0:
            x = x[..., :-downsample_pad_amount, :]
        return x


class Level(nn.Module):
    def __init__(
        self,
        resampler: Resampler,
        pre_module,
        inner_module,
        post_module,
    ):
        super().__init__()
        self.pre_module = pre_module
        self.inner_module = inner_module
        self.post_module = post_module
        self.resampler = resampler
        self.resid_scale = nn.Parameter(torch.tensor([0.5]))
        self.i = 0

    def forward(self, x):
        x = self.pre_module(x)
        skip = x
        x, downsample_info = self.resampler.downsample(x)
        x = self.inner_module(x)
        x = self.resampler.upsample(x, downsample_info)
        x = skip + x * self.resid_scale
        x = self.post_module(x)
        return x


def build_hourglass(
    context_len,
    base_dim,
    dim_factor,
    shorten_factor=2,
    num_levels=2,
    pre_layers=2,
    post_layers=2,
    bottom_layers=4,
    window_factors=(1, 1, 1, 1),
):
    """
    base_dim: dim of highet (outer) level
    dim_factor: each subsequent level has dim multiplied by dim_factor
    shorten_factor: sequence reduction factor for each level
    window_factors: window_size = seq_len / window_factor (specified fore each level)
    """

    def build_trf(num_layers, dim, window_size=-1):
        return build_transformer(
            num_layers,
            dim,
            norm_type="rms",
            attn_rotary=True,
            attn_window_size=window_size,
            mlp_activation="geglu",
            time_shift=True,
        )

    def dim_for_level(lvl):
        return nnt.round_to_multiple(int(base_dim * dim_factor**lvl), 64)

    def window_size_for_level(lvl):
        ws = window_factors[lvl]
        if ws == 1:
            return -1
        return context_len // ws

    inner = build_trf(
        bottom_layers,
        dim_for_level(num_levels - 1),
        window_size_for_level(num_levels - 1),
    )

    for i in reversed(range(num_levels - 1)):
        hi_dim = dim_for_level(i)
        lo_dim = dim_for_level(i + 1)
        window_size = window_size_for_level(i)
        inner = Level(
            FixedResampler(shorten_factor, hi_dim, lo_dim),
            build_trf(pre_layers, hi_dim, window_size),
            inner,
            build_trf(post_layers, hi_dim, window_size),
        )

    return inner

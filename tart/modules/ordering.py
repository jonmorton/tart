import torch
from torch import nn


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_apply(x, cos, sin, scale=1.0):
    return x * cos * scale + _rotate_half(x) * sin * scale


def rope_gen(dim, seq_len, dtype, device):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
    )
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class RotaryEmbedding(nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_len):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            cos, sin = rope_gen(self.dim, seq_len, x.dtype, x.device)
            if x.dim() == 4:
                self._cos_cached = cos[None, :, None, :]
                self._sin_cached = sin[None, :, None, :]
            elif x.dim() == 3:
                r = x.shape[0] // seq_len
                self._cos_cached = cos[:, None, :].repeat(r, 1, 1)
                self._sin_cached = sin[:, None, :].repeat(r, 1, 1)

        return self._cos_cached, self._sin_cached

    def forward(self, q, k, seq_len):
        cos, sin = self._update_cos_sin_tables(k, seq_len)

        return (
            rope_apply(q, cos, sin, 1.0),
            rope_apply(k, cos, sin, -1.0),
        )


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(
            torch.ones(
                1,
            )
        )
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._emb_cached = None

    def _get_cached_emb(self, x):
        seq_len = x.shape[1]
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
            or self._emb_cached.device != x.device
            or self._emb_cached.dtype != x.dtype
        ):
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            sinu = torch.einsum("i , j -> i j", t, self.inv_freq)
            self._emb_cached = torch.cat((sinu.sin(), sinu.cos()), dim=-1)

        return self._emb_cached

    def forward(self, x):
        return x + self._get_cached_emb(x) * self.scale


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, n_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, seq_len, n_dim))

    def forward(self, x):
        return x + self.weight[:, : x.size(1), :]

    def init_weights(self):
        nn.init.zeros_(self.weight)

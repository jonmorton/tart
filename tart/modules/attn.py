import math

import torch.amp

try:
    from flash_attn.flash_attn_interface import flash_attn_func

    has_flash = True
except ImportError:
    has_flash = False

from torch import nn

from .norm import RMSNorm
from .ordering import RotaryEmbedding


def compute_causal_attn(q, k, v, window_size=-1, softmax_scale=-1):
    if has_flash:
        orig_dtype = q.dtype
        if window_size:
            window_size = (window_size, 0)
        else:
            window_size = (-1, -1)
        return flash_attn_func(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
        ).to(orig_dtype)
    else:
        return nn.functional.scaled_dot_product_attention(
            q, k, v, causal=True, scale=softmax_scale
        )


class CausalSelfAttention(nn.Module):
    def __init__(
        self, embed_dim, num_heads, rotary=False, qknorm=False, window_size=-1
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rotary = rotary
        self.qknorm = qknorm
        self.window_size = window_size

        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        if rotary:
            self.rope = RotaryEmbedding(self.head_dim)
        if qknorm:
            self.q_norm = RMSNorm(self.head_dim, scale=True)
            self.k_norm = RMSNorm(self.head_dim, scale=True)

    def init_weights(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.orthogonal_(self.Wq.weight)
        nn.init.zeros_(self.Wk.weight)
        nn.init.orthogonal_(self.Wv.weight)

    def forward(self, x):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        q = self.Wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.Wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.Wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        if self.rotary:
            q, k = self.rope(q, k, seq_len)
        if self.qknorm:
            q, k = self.q_norm(q), self.k_norm(k)

        softmax_scale = 1.0 if self.qknorm else 1.0 / math.sqrt(self.head_dim)
        output = compute_causal_attn(
            q,
            k,
            v,
            self.window_size,
            softmax_scale,
        ).view(batch_size, seq_len, -1)
        return self.out_proj(output)

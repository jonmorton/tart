import collections

from torch import nn

from .attn import CausalSelfAttention
from .mlp import MLP
from .residual import Residual


def build_transformer(
    num_layers: int,
    dim: int,
    norm_type: str = "ln",
    sigmoid_gated_residual: bool = False,
    time_shift: bool = False,
    attn_rotary: bool = False,
    attn_qknorm: bool = False,
    attn_window_size: int = -1,
    mlp_expand: int = 4,
    mlp_activation: str = "gelu",
):
    blocks = collections.OrderedDict()

    for i in range(num_layers):
        if i % 2 == 0:
            module = CausalSelfAttention(
                dim,
                num_heads=dim // 64,
                rotary=attn_rotary,
                qknorm=attn_qknorm,
                window_size=attn_window_size,
            )
            name = f"attn{i}"
        else:
            module = MLP(dim, expand=mlp_expand, activation=mlp_activation)
            name = f"mlp{i}"
        block = Residual(
            module,
            dim,
            norm_type,
            gate=sigmoid_gated_residual,
            time_shift=time_shift,
        )
        blocks[name] = block

    return nn.Sequential(blocks)

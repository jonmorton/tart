import nnt
from torch import nn

from .. import vocab
from ..config import ModelConfig
from ..modules import embed, norm, ordering, transformer

layout = nnt.Registry("layouts")


@layout
def gpt2(mc: ModelConfig):
    return nn.Sequential(
        # Input
        embed.TokenEmbedding(vocab.size, mc.embed_dim),
        ordering.LearnedPositionalEmbedding(mc.seq_len, mc.embed_dim),
        # Transformer
        transformer.build_transformer(mc.num_layers, mc.embed_dim),
        # Output
        norm.get_norm("ln", mc.embed_dim),
        nn.Linear(mc.embed_dim, vocab.size),
    )


@layout
def ibt(mc: ModelConfig):
    """
    Improved baseline transformer
    Better performance than GPT2 at similar compute, memory, and parameter count
    """
    return nn.Sequential(
        # Input
        embed.TokenEmbedding(vocab.size, mc.embed_dim),
        # Transformer
        transformer.build_transformer(
            mc.num_layers,
            mc.embed_dim,
            norm_type="rms",
            time_shift=True,
            attn_rotary=True,
            mlp_activation="geglu",
            attn_window_size=mc.seq_len // 2,
        ),
        # Output
        norm.get_norm("ln", mc.embed_dim),
        nn.Linear(mc.embed_dim, vocab.size, bias=False),
    )


@layout
def hourglass(mc: ModelConfig):
    """
    U-net style heirarchical transformer with skip connections.
    """
    from ..modules.hourglass import build_hourglass

    sf = 4
    hi_dim = mc.embed_dim
    return nn.Sequential(
        # Input
        embed.TokenEmbedding(vocab.size, hi_dim),
        # Transformer
        build_hourglass(
            mc.seq_len,
            hi_dim,
            dim_factor=1.8,
            shorten_factor=sf,
            num_levels=2,
            window_factors=(2, 1),
            bottom_layers=6,
            pre_layers=2,
            post_layers=4,
        ),
        # Output
        norm.get_norm("ln", hi_dim),
        nn.Linear(hi_dim, vocab.size, bias=False),
    )

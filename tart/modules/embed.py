import nnt
from torch import nn

from ..modules.util import l2norm


class TokenEmbedding(nn.Module):
    def __init__(self, num_embed, embed_dim, norm=False):
        super().__init__()
        self.norm = norm
        self.num_embeddings = num_embed
        self._real_nembed = nnt.round_to_multiple(num_embed, 8, up=True)
        self.embed = nn.Embedding(self._real_nembed, embed_dim)

    def forward(self, token_indices):
        embeds = self.embed(token_indices)
        if self.norm:
            embeds = l2norm(embeds, scaled=True)
        return embeds

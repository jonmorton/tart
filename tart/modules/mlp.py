from nnt import round_to_multiple
from torch import nn

from .activation import get_activation


class MLP(nn.Module):
    def __init__(
        self,
        dim,
        expand=4,
        out_dim=None,
        activation="gelu",
    ):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        # keep things parameter-equal
        if activation in ("geglu", "swiglu"):
            expand *= 4 / 3
            hdim_factor = 2
        else:
            hdim_factor = 1
        hdim = round_to_multiple(int(dim * expand), 64)
        self.fc_in = nn.Linear(dim, hdim)
        self.fc_out = nn.Linear(hdim // hdim_factor, out_dim)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.fc_out(x)
        return x

    def init_weights(self):
        nn.init.zeros_(self.fc_out.weight)

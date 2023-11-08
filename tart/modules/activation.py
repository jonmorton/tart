from torch import nn


def geglu(x):
    x, g = x.chunk(2, dim=-1)
    return x * nn.functional.gelu(g)


def swiglu(x):
    x, g = x.chunk(2, dim=-1)
    return x * nn.functional.silu(g)


class GEGLU(nn.Module):
    def forward(self, x):
        return geglu(x)


class SwiGLU(nn.Module):
    def forward(self, x):
        return swiglu(x)


class ReluSquared(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.relu(x, self.inplace).pow(2.0)


def get_activation(name, inplace=False):
    if name == "geglu":
        return GEGLU()
    elif name == "swiglu":
        return SwiGLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU(inplace)
    elif name == "none":
        return nn.Identity()
    elif name in ("relusq", "relusquared", "relu_squared", "squared_relu"):
        return ReluSquared(inplace)
    else:
        raise Exception(f"Uknown activation name '{name}'")

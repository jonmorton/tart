import torch
from torch import nn

from ..modules.norm import LayerNorm, RMSNorm


@torch.no_grad()
def init_module(module):
    if isinstance(module, (nn.Linear,)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv1d,)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.uniform_(module.weight, a=-1e-4, b=1e-4)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


@torch.no_grad()
def initialize_weights(model):
    for m in model.modules():
        init_module(m)
    for m in model.modules():
        if hasattr(m, "init_weights") and callable(m.init_weights):
            m.init_weights()


def get_wd_params(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            yield m.weight

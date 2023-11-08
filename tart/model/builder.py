from ..config import cfg
from .layouts import layout
from .weights import initialize_weights
from .wrapper import AutoregressiveModelWrapper


def build_model():
    layout_fn = layout.get(cfg.model.layout)
    model = layout_fn(cfg.model)
    initialize_weights(model)
    wrapper = AutoregressiveModelWrapper(model)
    return wrapper

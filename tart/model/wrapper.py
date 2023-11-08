import torch
from torch import nn

from ..config import cfg


class AutoregressiveModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.context_len = cfg.model.seq_len

    def forward(self, x, targets=None):
        pred = self.model(x)
        if targets is not None:
            loss = self.loss(pred, targets)
        else:
            loss = {}
        return pred, loss

    def loss(self, logits, targets):
        losses = {}
        vocab_size = logits.shape[-1]
        logits = logits - logits.max(-1, keepdims=True)[0]
        gt_onehot = torch.nn.functional.one_hot(targets, vocab_size)
        predicted_logits = (gt_onehot * logits).sum(dim=-1)
        exp_logits = torch.exp(logits)
        sum_exp_logits = torch.sum(exp_logits, dim=-1)
        losses["ce"] = (torch.log(sum_exp_logits) - predicted_logits).mean()

        if self.training and cfg.optim.logit_reg_weight > 0.0:
            losses["logit_reg"] = (
                torch.log(sum_exp_logits) ** 2
            ).mean() * cfg.optim.logit_reg_weight

        correct = predicted_logits == 0.0
        return losses, correct

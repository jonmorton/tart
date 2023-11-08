from typing import Optional, Union

import torch
import torch.nn.functional as F

from . import vocab
from .model.wrapper import AutoregressiveModelWrapper


@torch.no_grad()
def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.75,
    top_k: float = 0.0,
) -> torch.Tensor:
    logits = logits / temperature
    if top_k > 0:
        k = max(1, int(top_k * logits.size(-1)))
        v, _ = torch.topk(logits, min(k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float("Inf")
    if top_p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        non_nucleus = cum_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        non_nucleus[..., 1:] = non_nucleus[..., :-1].clone()
        non_nucleus[..., 0] = 0
        remove_indices = sorted_indices[non_nucleus]
        logits[remove_indices] = -float("Inf")

    probs = F.softmax(logits)
    token = torch.multinomial(probs, num_samples=1)
    return token


@torch.no_grad()
def generate(
    model: AutoregressiveModelWrapper,
    cond_seq: Union[torch.Tensor, list[int]],
    max_new_tokens,
    temperature=1.0,
    top_p: float = 0.75,
    top_k: int = 0,
    stop_at: Optional[list[int]] = None,
):
    if isinstance(cond_seq, list):
        cond_seq = torch.tensor(cond_seq, dtype=torch.long)
    seq = cond_seq.to(next(iter(model.parameters())).device)
    if stop_at is not None:
        stop_at = torch.tensor(stop_at, dtype=torch.long, device=seq.device)
    for _ in range(max_new_tokens):
        cond_seq = (
            seq if seq.size(-1) <= model.context_len else seq[-model.context_len :]
        )
        logits, _ = model(cond_seq.unsqueeze(0))
        pred_token = sample(logits[0, -1, :], temperature, top_p, top_k)
        seq = torch.cat((seq, pred_token), dim=0)
        if stop_at is not None and torch.isin(pred_token, stop_at).any(dim=-1):
            break

    return seq


@torch.no_grad()
def generate_text(
    model: AutoregressiveModelWrapper,
    cond: str = "",
    max_new_tokens=100,
    prepend_eod=False,
    stop_at=(vocab.EOD,),
    temperature=1.0,
    top_p: float = 0.75,
    top_k: int = 0,
):
    cond_idx = vocab.encode(cond)
    if prepend_eod:
        cond_idx = [vocab.EOD] + cond_idx

    tokens = generate(
        model, cond_idx, max_new_tokens, temperature, top_p, top_k, stop_at
    )
    tokens = tokens.flatten().cpu().tolist()

    return vocab.decode(tokens)

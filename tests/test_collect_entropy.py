import math
import torch


def _entropy(logits, legal):
    lse = logits.logsumexp(dim=-1)
    log_p = logits - lse.unsqueeze(-1)
    safe_p = torch.where(legal, log_p.exp(), torch.zeros_like(log_p))
    safe_log_p = torch.where(legal, log_p, torch.zeros_like(log_p))
    return -(safe_p * safe_log_p).sum(dim=-1)


def test_uniform_legal_entropy_is_log_n():
    legal = torch.zeros(2, 8, dtype=torch.bool)
    legal[0, :4] = True
    legal[1, :2] = True
    logits = torch.where(legal, torch.zeros(2, 8), torch.full((2, 8), float("-inf")))
    ent = _entropy(logits, legal)
    assert abs(ent[0].item() - math.log(4)) < 1e-5
    assert abs(ent[1].item() - math.log(2)) < 1e-5


def test_peaked_row_has_near_zero_entropy():
    legal = torch.ones(1, 4, dtype=torch.bool)
    logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]])
    ent = _entropy(logits, legal)
    assert ent.item() < 0.01

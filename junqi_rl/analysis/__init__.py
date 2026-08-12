"""Post-training evaluation, replay recording and checkpoint leagues.

Torch-dependent entry points are imported lazily so the league registry and
evaluation statistics remain usable in the lightweight core installation.
"""

from __future__ import annotations

from typing import Any

from .league import LeagueEntry, LeaguePool
from .protocol import EvaluationCounts, merge_evaluations, wilson_interval


def eval_vs_random(*args: Any, **kwargs: Any) -> Any:
    from .evaluate import eval_vs_random as implementation

    return implementation(*args, **kwargs)


def eval_head_to_head(*args: Any, **kwargs: Any) -> Any:
    from .evaluate import eval_head_to_head as implementation

    return implementation(*args, **kwargs)


def evaluate_paired_vs_random(*args: Any, **kwargs: Any) -> Any:
    from .random_eval import evaluate_paired_vs_random as implementation

    return implementation(*args, **kwargs)


def record_game_with_policy(*args: Any, **kwargs: Any) -> Any:
    from .record import record_game_with_policy as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "EvaluationCounts",
    "LeagueEntry",
    "LeaguePool",
    "eval_head_to_head",
    "eval_vs_random",
    "evaluate_paired_vs_random",
    "merge_evaluations",
    "record_game_with_policy",
    "wilson_interval",
]

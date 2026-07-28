"""Reproducible evaluation statistics shared by training and reports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationCounts:
    """Outcome counts with conservative requested-game denominators."""

    wins: int
    losses: int
    draws: int
    ongoing: int = 0

    def __post_init__(self) -> None:
        if min(self.wins, self.losses, self.draws, self.ongoing) < 0:
            raise ValueError("evaluation counts must be non-negative")

    @property
    def requested(self) -> int:
        return self.wins + self.losses + self.draws + self.ongoing

    @property
    def completed(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score(self) -> float:
        """Chess-style score over completed games: win=1, draw=0.5."""

        return (
            (self.wins + 0.5 * self.draws) / self.completed
            if self.completed
            else 0.0
        )

    def as_metrics(self, *, prefix: str = "eval") -> dict[str, float]:
        denom = max(1, self.requested)
        low, high = wilson_interval(self.wins, self.requested)
        return {
            f"{prefix}/wins": float(self.wins),
            f"{prefix}/losses": float(self.losses),
            f"{prefix}/draws": float(self.draws),
            f"{prefix}/ongoing": float(self.ongoing),
            f"{prefix}/win_rate": self.wins / denom,
            f"{prefix}/loss_rate": self.losses / denom,
            f"{prefix}/draw_rate": self.draws / denom,
            f"{prefix}/ongoing_rate": self.ongoing / denom,
            f"{prefix}/score_completed": self.score,
            f"{prefix}/win_rate_ci95_low": low,
            f"{prefix}/win_rate_ci95_high": high,
            f"{prefix}/num_games": float(self.completed),
            f"{prefix}/requested_games": float(self.requested),
        }


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial rate."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0:
        return 0.0, 1.0
    n = float(trials)
    p = successes / n
    z2 = z * z
    center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
        / (1.0 + z2 / n)
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def counts_from_metrics(
    metrics: Mapping[str, float],
    *,
    prefix: str = "eval",
) -> EvaluationCounts:
    """Read exact counts when present, with compatible rate fallback."""

    requested = int(metrics.get(f"{prefix}/requested_games", 0.0))
    completed = int(metrics.get(f"{prefix}/num_games", 0.0))
    ongoing = int(metrics.get(f"{prefix}/ongoing", requested - completed))
    wins = int(
        metrics.get(
            f"{prefix}/wins",
            round(metrics.get(f"{prefix}/win_rate", 0.0) * requested),
        )
    )
    losses = int(
        metrics.get(
            f"{prefix}/losses",
            round(metrics.get(f"{prefix}/loss_rate", 0.0) * requested),
        )
    )
    draws = int(
        metrics.get(
            f"{prefix}/draws",
            max(0, completed - wins - losses),
        )
    )
    return EvaluationCounts(wins=wins, losses=losses, draws=draws, ongoing=ongoing)


def merge_evaluations(
    *metrics: Mapping[str, float],
    prefix: str = "eval",
) -> dict[str, float]:
    """Merge evaluation shards using counts, never by averaging rates."""

    counts = [counts_from_metrics(item, prefix=prefix) for item in metrics]
    merged = EvaluationCounts(
        wins=sum(item.wins for item in counts),
        losses=sum(item.losses for item in counts),
        draws=sum(item.draws for item in counts),
        ongoing=sum(item.ongoing for item in counts),
    )
    out = merged.as_metrics(prefix=prefix)
    completed = max(1, merged.completed)
    requested = max(1, merged.requested)
    weighted_lengths = [
        (
            float(item.get(f"{prefix}/avg_game_len", 0.0)),
            counts[i].completed,
        )
        for i, item in enumerate(metrics)
    ]
    out[f"{prefix}/avg_game_len"] = (
        sum(value * weight for value, weight in weighted_lengths) / completed
    )
    out[f"{prefix}/avg_game_len_all"] = (
        sum(
            float(item.get(f"{prefix}/avg_game_len_all", 0.0))
            * counts[i].requested
            for i, item in enumerate(metrics)
        )
        / requested
    )
    return out


__all__ = [
    "EvaluationCounts",
    "counts_from_metrics",
    "merge_evaluations",
    "wilson_interval",
]

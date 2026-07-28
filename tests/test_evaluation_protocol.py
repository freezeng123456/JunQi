from __future__ import annotations

import json
from pathlib import Path

import pytest

from junqi_rl.analysis.league import LeaguePool
from junqi_rl.analysis.protocol import (
    EvaluationCounts,
    merge_evaluations,
    wilson_interval,
)


def test_wilson_interval_is_bounded_and_tightens() -> None:
    low_small, high_small = wilson_interval(6, 10)
    low_large, high_large = wilson_interval(60, 100)
    assert 0.0 <= low_small < 0.6 < high_small <= 1.0
    assert 0.0 <= low_large < 0.6 < high_large <= 1.0
    assert high_large - low_large < high_small - low_small


def test_merge_evaluations_uses_counts_not_mean_rates() -> None:
    first = EvaluationCounts(wins=8, losses=2, draws=0).as_metrics()
    second = EvaluationCounts(wins=1, losses=0, draws=0, ongoing=9).as_metrics()
    merged = merge_evaluations(first, second)
    assert merged["eval/wins"] == 9.0
    assert merged["eval/requested_games"] == 20.0
    assert merged["eval/win_rate"] == pytest.approx(0.45)


def _checkpoint(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_league_registry_roundtrip_sampling_and_elo(tmp_path: Path) -> None:
    registry = tmp_path / "league.json"
    pool = LeaguePool(registry, max_entries=3)
    _entries = [
        pool.register(_checkpoint(tmp_path / f"ckpt_{i}.pt", f"model-{i}".encode()), rollout=i)
        for i in range(4)
    ]
    assert len(pool.entries) == 3
    assert pool.entries[0].rollout == 0
    assert pool.entries[-1].rollout == 3

    sampled_a = pool.sample(seed=17, exclude_rollout=3)
    sampled_b = pool.sample(seed=17, exclude_rollout=3)
    assert sampled_a is not None and sampled_b is not None
    assert sampled_a.sha256 == sampled_b.sha256
    assert sampled_a.rollout != 3

    latest = pool.entries[-1]
    opponent = pool.entries[0]
    pool.record_match(
        latest.sha256,
        opponent.sha256,
        first_score=1.0,
        games=8,
    )
    assert latest.rating > 1000.0
    assert opponent.rating < 1000.0
    assert latest.games == opponent.games == 8

    reloaded = LeaguePool(registry, max_entries=3)
    assert [item.sha256 for item in reloaded.entries] == [
        item.sha256 for item in pool.entries
    ]
    raw = json.loads(registry.read_text(encoding="utf-8"))
    assert raw["version"] == 1

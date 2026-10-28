"""Actual-game opening diagnostics; never changes actions, RNG, or losses."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

POLICY_FIELDS = (
    "action_entropy", "action_entropy_ratio", "action_top1", "action_top2",
    "source_entropy", "source_entropy_ratio", "source_top1", "source_top2",
    "legal_actions", "movable_sources",
)


def policy_statistics(probs, legal, entropy):
    """Compact per-position statistics from the already computed probabilities.

    A single-choice support has zero entropy ratio. No CPU synchronization,
    extra inference, or random draws occur on this path.
    """
    import torch
    n = math.isqrt(probs.shape[-1])
    if n * n != probs.shape[-1] or probs.shape != legal.shape:
        raise ValueError("expected matching square action probabilities and masks")
    source = probs.reshape(-1, n, n).sum(-1)
    n_action = legal.sum(-1).float()
    n_source = legal.reshape(-1, n, n).any(-1).sum(-1).float()
    h_source = -(source * source.clamp_min(1e-30).log()).sum(-1)
    top_action = probs.topk(2, dim=-1).values
    top_source = source.topk(2, dim=-1).values
    return torch.stack((entropy, entropy / n_action.clamp_min(2).log(),
        top_action[:, 0], top_action.sum(-1), h_source,
        h_source / n_source.clamp_min(2).log(), top_source[:, 0],
        top_source.sum(-1), n_action, n_source), dim=-1).detach()


def distribution(values):
    counts = Counter(values)
    n = sum(counts.values())
    h = -sum((v/n) * math.log(v/n) for v in counts.values()) if n else 0.0
    ordered = counts.most_common()
    return dict(counts=dict(ordered), unique=len(counts), entropy_nats=h,
        effective_choices=math.exp(h) if n else 0.0,
        top1=ordered[0][1]/n if n else 0.0,
        top2=sum(v for _, v in ordered[:2])/n if n else 0.0)


def summarize_games(records):
    missing = sum(int(r.get("first_action", -1)) < 0 for r in records)
    valid = [r for r in records if int(r.get("first_action", -1)) >= 0]
    policies = [r["first_policy"] for r in valid if "first_policy" in r]
    eligible = [p for p in policies if p["movable_sources"] >= 4]
    actions = distribution(int(r["first_action"]) for r in valid)
    sources = distribution(int(r["first_action"]) // 129 for r in valid)
    return dict(games=len(records), missing_first_actions=missing,
        missing_first_policy=len(valid)-len(policies), actions=actions, sources=sources,
        policy_mean={k: sum(p[k] for p in policies)/len(policies)
            for k in POLICY_FIELDS} if policies else {},
        eligible_policy_games=len(eligible),
        near_deterministic_source_fraction=(sum(p["source_top1"] >= .95 and
            p["source_entropy_ratio"] <= .2 for p in eligible)/len(eligible)
            if eligible else None),
        # Screening flag only: no universal claim about optimal play.
        sampled_two_action_concentration=bool(len(valid) >= 1000 and actions["top2"] >= .90))


def write_milestone(save_dir, checkpoint, completed_games, milestone):
    root = Path(save_dir)
    with (root / "training_games.jsonl").open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != completed_games:
        raise ValueError("first-move record count does not match completed games")
    if len({(r["lineage_id"], r["game_id"]) for r in records}) != len(records):
        raise ValueError("duplicate completed-game record")
    result = dict(schema="junqi.actual-opening-milestone.v1", milestone=milestone,
        completed_games=completed_games, checkpoint=str(checkpoint),
        cumulative=summarize_games(records), recent_1000=summarize_games(records[-1000:]),
        recent_5000=summarize_games(records[-5000:]))
    path = root / f"first_move_{milestone:06d}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return result

"""First-decision distributions on fixed and freshly generated legal boards.

The probe never advances a training environment or changes its setup pool.
It evaluates full probabilities without sampling and restores training RNG.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch


@dataclass
class OpeningMonitorConfig:
    enabled: bool = False
    every_games: int = 1000
    boards_per_bank: int = 128
    batch_size: int = 64
    seed: int = 93_102_201
    min_movable_pieces: int = 4
    min_source_samples: int = 64
    source_mass_limit: float = 0.95
    source_entropy_limit: float = 0.20
    patience: int = 3

    def validate(self):
        for name in ("every_games", "boards_per_bank", "batch_size",
                     "min_source_samples", "patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"opening.{name} must be a positive integer")
        if self.min_movable_pieces < 2:
            raise ValueError("opening.min_movable_pieces must be >= 2")
        if self.min_source_samples > self.boards_per_bank * 4:
            raise ValueError("opening.min_source_samples exceeds a bank's decisions")
        if not 0 < self.source_mass_limit <= 1:
            raise ValueError("opening.source_mass_limit must be in (0, 1]")
        if not 0 <= self.source_entropy_limit <= 1:
            raise ValueError("opening.source_entropy_limit must be in [0, 1]")


def summarize_sources(probabilities, movable, cfg):
    """Condition each source's probability on that piece being movable.

    A mine at the same square therefore cannot dilute the collapse signal.
    Left/right destination variation does not count as selecting other pieces.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    legal = np.asarray(movable, dtype=bool)
    if p.ndim != 2 or p.shape != legal.shape:
        raise ValueError("source probabilities and mobility must have matching [B,S] shapes")
    if not np.isfinite(p).all() or (p < 0).any() or (p[~legal] != 0).any():
        raise ValueError("nonfinite, negative or illegal opening probability")
    if not np.allclose(p.sum(1), 1, atol=2e-5, rtol=0):
        raise ValueError("opening probabilities are not normalized")
    count = legal.sum(1)
    keep = count >= cfg.min_movable_pieces
    if not keep.any():
        raise ValueError("opening probe has no decisions with enough movable pieces")
    p, legal, count = p[keep], legal[keep], count[keep]
    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(1) / np.log(count)
    available = legal.sum(0)
    conditional = np.divide(p.sum(0), available, out=np.zeros(p.shape[1]), where=available > 0)
    supported = available >= cfg.min_source_samples
    if not supported.any():
        raise ValueError("opening probe has insufficient conditional source support")
    source = int(np.argmax(np.where(supported, conditional, -1)))
    mass = float(conditional[source])
    mean_entropy = float(entropy.mean())
    conditional_entropy = float(entropy[legal[:, source]].mean())
    return dict(decisions=int(len(probabilities)), eligible_decisions=int(len(p)),
        dominant_source=source, dominant_source_samples=int(available[source]),
        max_conditional_source_mass=mass, normalized_source_entropy=mean_entropy,
        dominant_source_conditional_entropy=conditional_entropy,
        mean_largest_source_probability=float(p.max(1).mean()),
        conditional_source_probability=conditional.tolist(),
        source_available_samples=available.tolist(),
        source_probability_mean=p.mean(0).tolist(),
        collapsed=bool(mass >= cfg.source_mass_limit and conditional_entropy <= cfg.source_entropy_limit))


@contextmanager
def preserve_probe_state(policy):
    py_state, np_state, training = random.getstate(), np.random.get_state(), policy.training
    device = next(policy.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            policy.eval()
            yield
    finally:
        policy.train(training)
        random.setstate(py_state)
        np.random.set_state(np_state)


class OpeningMonitor:
    def __init__(self, cfg, output_dir):
        cfg.validate()
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.checks = 0
        self.streak = 0
        self.next_games = 0
        self.world = None
        self.last_result = None

    def due(self, completed_games):
        return completed_games >= self.next_games

    def state_dict(self):
        return dict(version=2, config=asdict(self.cfg), checks=self.checks,
            streak=self.streak, next_games=self.next_games, last_result=self.last_result)

    def load_state_dict(self, state):
        if state["version"] not in (1, 2) or state["config"] != asdict(self.cfg):
            raise ValueError("opening-monitor resume contract mismatch")
        for name in ("checks", "streak", "next_games", "last_result"):
            setattr(self, name, state[name])
        if state["version"] == 1:
            # Preserve cadence/RNG seed sequence while starting the refined
            # conditional-entropy streak with observations under v2 only.
            self.streak = 0

    def _bank(self, policy, seed):
        from junqi_core.rules import Seat
        from junqi_core.setup import generate_random_setup
        from junqi_rl.gpu_rollout import GpuRollout
        device = next(policy.parameters()).device
        if device.type != "cuda":
            raise ValueError("live opening probe requires the native CUDA environment")
        if self.world is None:
            self.world = GpuRollout(self.cfg.batch_size,
                include_global=policy.uses_global_summary, identity_encoding="known_only",
                device_id=device.index if device.index is not None else torch.cuda.current_device())
        rng = random.Random(seed)
        entries = [(setup, int(seat)) for _ in range(self.cfg.boards_per_bank)
            for setup in [generate_random_setup(rng)] for seat in Seat]
        probabilities, movable = [], []
        for start in range(0, len(entries), self.cfg.batch_size):
            rows = entries[start:start+self.cfg.batch_size]
            n = len(rows)
            rows += [rows[-1]] * (self.cfg.batch_size - n)
            self.world.reset(setups=[setup for setup, _ in rows])
            actors = torch.tensor([seat for _, seat in rows], dtype=torch.int8, device=device)
            sp, gl = self.world.build_acting_seat_observation_torch(actors)
            mask = self.world.legal_mask_canonical_torch_device(actors)
            actions = mask.to(torch.int32).argmax(1).to(torch.int64)
            out = policy(sp, gl, mask, actions=actions)
            size = math.isqrt(mask.shape[-1])
            p = out["log_probs"].exp().reshape(len(rows), size, size).sum(-1)
            m = mask.reshape(len(rows), size, size).any(-1)
            probabilities.append(p[:n].cpu().numpy())
            movable.append(m[:n].cpu().numpy())
        result = summarize_sources(np.concatenate(probabilities), np.concatenate(movable), self.cfg)
        from junqi_core.board import FLAT_TO_COMPACT, index_to_pos
        for slot in range(30):
            x, y = index_to_pos(Seat.SOUTH, slot)
            if int(FLAT_TO_COMPACT[y * 17 + x]) == result["dominant_source"]:
                result.update(dominant_local_slot=slot,
                    dominant_local_row=slot // 5 + 1, dominant_local_column=slot % 5 + 1)
                break
        result["seed"] = seed
        return result

    def check(self, policy, *, completed_games, rollout):
        with preserve_probe_state(policy):
            fixed = self._bank(policy, self.cfg.seed)
            fresh = self._bank(policy, self.cfg.seed + 1_000_003 * (self.checks + 1))
        # Require both independent banks to identify the same dominant square.
        collapsed = (fixed["collapsed"] and fresh["collapsed"] and
            fixed["dominant_source"] == fresh["dominant_source"])
        self.streak = self.streak + 1 if collapsed else 0
        self.checks += 1
        self.next_games = (completed_games // self.cfg.every_games + 1) * self.cfg.every_games
        result = dict(schema="junqi.opening-monitor.v2", completed_games=completed_games,
            rollout=rollout, check_index=self.checks, fixed=fixed, fresh=fresh,
            collapse_streak=self.streak, status="collapse" if self.streak >= self.cfg.patience
                else ("warning" if collapsed else "healthy"))
        self.last_result = result
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "opening_progress.jsonl").open("a") as out:
            out.write(json.dumps(result, allow_nan=False) + "\n")
        return result

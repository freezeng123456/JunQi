"""Completed-game accounting and single-GPU collection continuation.

Terminal events are captured before auto-reset. No host sync is added to the
per-move path; only completed rows are downloaded at a rollout boundary.
"""
from __future__ import annotations

import copy
import random
import uuid
from collections import Counter

import numpy as np
import torch


def setup_pool_index(env, seed, pool_size):
    """Mirror reset_terminated_envs_kernel's uint64 pool selection."""
    mask = (1 << 64) - 1
    h = (env * 2654435761 + seed) & mask
    h ^= h >> 17
    h = (h * 0xbf58476d1ce4e5b9) & mask
    h ^= h >> 31
    return h % pool_size


class GameProgress:
    REASONS = {0: "team_victory", 1: "total_move_limit", 2: "no_combat_limit", 3: "other_draw"}

    def __init__(self, num_envs, device, *, lineage_id=None):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.lineage_id = lineage_id or uuid.uuid4().hex
        self.serial = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.initial_seed = torch.zeros_like(self.serial)
        self.pool_hashes = None
        self.initialized = False
        self.completed_games = 0
        self.environment_steps = 0
        self.total_game_moves = 0
        self.reasons = Counter()
        self.winners = Counter()

    def begin(self, world, steps, *, reset_seed):
        if self.pool_hashes is None:
            self.pool_hashes = world.setup_pool_hashes()
        if not self.initialized:
            self.initial_seed.fill_(reset_seed)
            self.initialized = True
        self.events = torch.zeros((steps, self.num_envs, 7), dtype=torch.int64, device=self.device)

    def record(self, world, result, fired, step):
        # Copy these views before reset overwrites the engine's scalar arrays.
        row = self.events[step]
        row[:, 0] = fired
        row[:, 1] = self.serial * self.num_envs + torch.arange(self.num_envs, device=self.device)
        row[:, 2] = world.move_counter_torch()
        row[:, 3] = result["winner_team"]
        draw = result["draw"]
        row[:, 4] = torch.where(draw, torch.where(
            world.move_counter_torch() >= world.state.max_num_moves, 1,
            torch.where(world.moves_since_last_combat_torch() >= 200, 2, 3)), 0)
        row[:, 5] = self.initial_seed
        row[:, 6] = self.environment_steps + (step + 1) * self.num_envs
        self.serial.add_(fired.to(torch.int64))

    def after_reset(self, world, fired, *, reset_seed):
        self.initial_seed.copy_(torch.where(fired, reset_seed, self.initial_seed))

    def finish(self, rollout):
        rows = self.events[self.events[:, :, 0].bool()].cpu().tolist()
        records = []
        for _, game_id, moves, winner, reason, initial_seed, env_steps in rows:
            label = self.REASONS[reason]
            pool_index = setup_pool_index(game_id % self.num_envs, initial_seed, len(self.pool_hashes))
            records.append(dict(lineage_id=self.lineage_id, game_id=game_id,
                moves=moves, winner_team=winner, termination_reason=label,
                initial_setup_sha256=self.pool_hashes[pool_index],
                setup_pool_index=pool_index, setup_reset_seed=initial_seed,
                tracked_environment_steps=env_steps, ending_rollout=rollout))
            self.reasons[label] += 1
            self.winners[str(winner)] += 1
            self.total_game_moves += moves
        self.completed_games += len(records)
        self.environment_steps += self.events.shape[0] * self.num_envs
        del self.events
        return records

    def summary(self):
        return dict(lineage_id=self.lineage_id, completed_games=self.completed_games,
            tracked_environment_steps=self.environment_steps,
            total_completed_game_moves=self.total_game_moves,
            mean_completed_game_moves=self.total_game_moves / max(1, self.completed_games),
            termination_reasons=dict(self.reasons), winners=dict(self.winners))

    def state_dict(self):
        return dict(version=2, num_envs=self.num_envs, lineage_id=self.lineage_id,
            serial=self.serial.cpu().clone(), initial_seed=self.initial_seed.cpu().clone(),
            initialized=self.initialized, completed_games=self.completed_games,
            environment_steps=self.environment_steps, total_game_moves=self.total_game_moves,
            reasons=dict(self.reasons), winners=dict(self.winners))

    def load_state_dict(self, state):
        if state["version"] != 2 or state["num_envs"] != self.num_envs:
            raise ValueError("game progress schema or environment count mismatch")
        for key in ("lineage_id", "initialized", "completed_games", "environment_steps", "total_game_moves"):
            setattr(self, key, state[key])
        self.serial.copy_(state["serial"])
        self.initial_seed.copy_(state["initial_seed"])
        self.pool_hashes = None
        self.reasons = Counter(state["reasons"])
        self.winners = Counter(state["winners"])
        if sum(self.reasons.values()) != self.completed_games or sum(self.winners.values()) != self.completed_games:
            raise ValueError("inconsistent completed game counters")


def capture_rng(minibatch_rng):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        minibatch=copy.deepcopy(minibatch_rng.bit_generator.state),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, minibatch_rng):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    minibatch_rng.bit_generator.state = state["minibatch"]
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def runtime_contract(cfg):
    from dataclasses import asdict
    return dict(env=asdict(cfg.env), net=asdict(cfg.net),
        torch_deterministic=cfg.torch_deterministic, dtype=cfg.ppo.dtype,
        random_opponent=cfg.random_opponent, reward_shaping=cfg.reward_shaping,
        fixed_setup_styles=cfg.fixed_setup_styles, mixed_setup=cfg.mixed_setup,
        mixed_own_team_styles=cfg.mixed_own_team_styles)


def capture_runtime(world, progress, minibatch_rng, cfg, legacy_environment_steps, *, saved_rng=None):
    return dict(version=1, contract=runtime_contract(cfg), world=world.state_dict(),
        games=progress.state_dict(), rng=capture_rng(minibatch_rng) if saved_rng is None else saved_rng,
        legacy_environment_steps=legacy_environment_steps,
        legacy_completed_games=None, boundary="after_optimizer_update")


def restore_runtime(state, world, progress, minibatch_rng, cfg):
    if state["version"] != 1 or state["contract"] != runtime_contract(cfg):
        raise ValueError("runtime contract mismatch; resume requires the same environment and input config")
    world.load_state_dict(state["world"])
    progress.load_state_dict(state["games"])
    restore_rng(state["rng"], minibatch_rng)
    return state["legacy_environment_steps"]

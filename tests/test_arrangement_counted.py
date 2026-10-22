"""Regression gates for dynamic setup attribution and resumable setup PPO."""
import copy
import hashlib
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from junqi_core.rules import PieceType
from junqi_core.setup import generate_random_lineup, validate_lineup
from junqi_rl.arrangement.buffer import ArrangementBuffer
from junqi_rl.arrangement.credits import credit_completed_games
from junqi_rl.arrangement.sampling import generate_arrangements
from junqi_rl.networks.arrangement_net import (
    ArrangementNet, ArrangementNetConfig, PIECE_TYPE_VALUE_TO_VOCAB_IDX,
    VOCAB_IDX_TO_PIECE_TYPE_VALUE,
)
from junqi_rl.training.game_progress import GameProgress
from junqi_rl.training.config import load_config
from argparse import Namespace
from pathlib import Path


def samples(seed):
    rng = random.Random(seed)
    raw = np.array([[p.value for p in generate_random_lineup(rng)] for _ in range(4)], dtype=np.int8)
    vocab = torch.tensor([[PIECE_TYPE_VALUE_TO_VOCAB_IDX[int(v)] for v in row] for row in raw])
    return raw.reshape(1, 120), F.one_hot(vocab, 13).float()


def add(buffer, arrangements, step):
    n = len(arrangements)
    buffer.add_arrangements(arrangements, torch.zeros(n, 30, 3), torch.zeros(n, 30),
        F.log_softmax(torch.zeros(n, 30, 13), -1), torch.arange(n) % 4, step)


def test_dynamic_pool_refresh_preserves_live_game_setup_and_resume():
    first, a = samples(44)
    second, b = samples(45)
    world = SimpleNamespace(setup_pool_revision=0, pool=first,
        state=SimpleNamespace(max_num_moves=4000), move_counter_torch=lambda: torch.tensor([83, 94]),
        moves_since_last_combat_torch=lambda: torch.zeros(2, dtype=torch.long))
    world.setup_pool_hashes = lambda: [hashlib.sha256(row.tobytes()).hexdigest() for row in world.pool]
    world.setup_pool_snapshot = lambda: world.pool.copy()
    progress = GameProgress(2, "cpu", lineage_id="fixture", track_arrangements=True)
    progress.begin(world, 1, reset_seed=71)
    result = dict(draw=torch.zeros(2, dtype=torch.bool), winner_team=torch.tensor([0, 1]))
    progress.record(world, result, torch.tensor([True, False]), 0)
    progress.after_reset(world, torch.tensor([True, False]), reset_seed=72)
    progress.finish(1)
    world.pool, world.setup_pool_revision = second, 1
    progress.begin(world, 1, reset_seed=73)
    progress.record(world, result, torch.tensor([True, False]), 0)
    progress.after_reset(world, torch.tensor([True, False]), reset_seed=74)
    rows = progress.finish(2)
    assert rows[0]["initial_setup_sha256"] == hashlib.sha256(first[0].tobytes()).hexdigest()
    resumed = GameProgress(2, "cpu", track_arrangements=True)
    resumed.load_state_dict(copy.deepcopy(progress.state_dict()))
    resumed.begin(world, 1, reset_seed=999)
    resumed.record(world, result, torch.tensor([True, True]), 0)
    rows = resumed.finish(3)
    assert [r["initial_setup_sha256"] for r in rows] == [
        hashlib.sha256(pool[0].tobytes()).hexdigest() for pool in (second, first)]
    np.testing.assert_array_equal(resumed.completed_setups, np.concatenate([second, first]))
    buf = ArrangementBuffer(storage_duration=40)
    add(buf, a, 0)
    add(buf, b, 8)
    assert credit_completed_games(buf, resumed.completed_setups, rows) == 8
    assert buf.counts.sum() == 8


def test_duplicate_terminal_rewards_survive_pool_refresh_and_buffer_resume():
    pool, arr = samples(7)
    buf = ArrangementBuffer(storage_duration=40)
    add(buf, arr, 0)
    repeated = np.repeat(pool, 3, axis=0)
    rows = [dict(winner_team=w) for w in (0, 1, 0)]
    assert credit_completed_games(buf, repeated, rows) == 12
    torch.testing.assert_close(buf.counts, torch.full((4, 1), 3.0))
    torch.testing.assert_close(buf.rewards, torch.tensor([[1., 0., 2.], [2., 0., 1.]] * 2) / 3)
    buf.filter(8)
    _, fresh = samples(8)
    add(buf, fresh, 8)
    assert buf.counts.sum() == 12
    restored = ArrangementBuffer(storage_duration=40)
    restored.load_state_dict(copy.deepcopy(buf.state_dict()))
    assert restored._lookup_index == buf._lookup_index
    for name in buf._sample_fields + buf._reward_fields:
        torch.testing.assert_close(getattr(restored, name), getattr(buf, name), rtol=0, atol=0)
    assert credit_completed_games(restored, pool, [dict(winner_team=-1)]) == 4
    assert restored.counts.sum() == 16


def test_exact_completion_mask_needs_no_random_fallback():
    torch.manual_seed(27)
    net = ArrangementNet(ArrangementNetConfig(depth=1, embed_dim=32, n_head=4))
    result = generate_arrangements(128, net, max_resample=0)
    assert not result.fallback_mask.any()
    assert result.stats["retry_rate"] == 0
    for row in result.samples.argmax(-1).tolist():
        lineup = [PieceType(VOCAB_IDX_TO_PIECE_TYPE_VALUE[v]) for v in row]
        assert validate_lineup(lineup).ok


def test_counted_arrangement_config_has_separate_facts_only_inputs():
    cfg = load_config(Namespace(config=str(Path(__file__).parents[1] / "configs/v100_facts_arrangement.yaml"), extra=[]))
    assert cfg.arr.enabled and not cfg.disable_arr_train
    assert cfg.net.identity_encoding == "known_only" and not cfg.net.use_global_summary
    assert not cfg.belief.enabled

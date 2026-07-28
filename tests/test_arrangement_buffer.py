"""Tests for junqi_rl.arrangement.buffer."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from junqi_rl.arrangement.buffer import (
    ArrangementBuffer,
    Batch,
    DEFAULT_CATEGORICAL_AGGREGATION,
    _arrangement_ids,
    _mark_most_recent_appearance,
)
from junqi_rl.networks.arrangement_net import (
    ARRANGEMENT_SIZE,
    N_PIECE_TYPE_WITH_NONE,
    N_VF_CAT_DEFAULT,
    PIECE_TYPE_VALUE_TO_VOCAB_IDX,
)
from junqi_core.rules import PieceType
from junqi_core.setup import generate_random_lineup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lineup_to_vocab(lineup) -> torch.Tensor:
    return torch.tensor(
        [PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value] for pt in lineup], dtype=torch.long,
    )


def _fake_batch(N: int, *, seed: int = 0, use_cat_vf: bool = True):
    """Make (arrangements, values, ents, log_probs, seat_idx) for N rows."""
    import random

    rng = random.Random(seed)
    vocabs = torch.stack([_lineup_to_vocab(generate_random_lineup(rng)) for _ in range(N)])
    arrangements = F.one_hot(vocabs, num_classes=N_PIECE_TYPE_WITH_NONE).float()
    if use_cat_vf:
        values = torch.randn(N, ARRANGEMENT_SIZE, N_VF_CAT_DEFAULT)
    else:
        values = torch.randn(N, ARRANGEMENT_SIZE)
    ents = torch.randn(N, ARRANGEMENT_SIZE).abs()
    log_probs = F.log_softmax(torch.randn(N, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE), dim=-1)
    seat_idx = torch.randint(0, 4, (N,), dtype=torch.long)
    return arrangements, values, ents, log_probs, seat_idx, vocabs


# ---------------------------------------------------------------------------
# arrangement_ids + mark_most_recent_appearance
# ---------------------------------------------------------------------------


def test_arrangement_ids_stable_and_distinct():
    torch.manual_seed(0)
    N = 16
    import random
    rng = random.Random(123)
    vocabs = torch.stack([_lineup_to_vocab(generate_random_lineup(rng)) for _ in range(N)])
    ids1 = _arrangement_ids(vocabs.to(torch.uint8))
    ids2 = _arrangement_ids(vocabs.to(torch.uint8))
    assert ids1 == ids2  # deterministic across invocations
    # Identical rows yield identical ids.
    dup = vocabs.clone()
    dup[-1] = vocabs[0]
    ids3 = _arrangement_ids(dup.to(torch.uint8))
    assert ids3[0] == ids3[-1]


def test_mark_most_recent_basic():
    values = [7, 7, 3, 5, 3, 3]
    ts = torch.tensor([1, 3, 2, 4, 5, 5], dtype=torch.long)
    mask = _mark_most_recent_appearance(values, ts).tolist()
    # 7 → index 1 (ts=3), 3 → index 4 or 5 (both ts=5, first-seen wins → 4),
    # 5 → index 3.
    assert mask.count(True) == 3
    assert mask[0] is False and mask[1] is True
    assert mask[3] is True
    # For value=3, index 4 should be selected (first with max ts among duplicates).
    assert mask[4] is True
    assert mask[5] is False


# ---------------------------------------------------------------------------
# add_arrangements
# ---------------------------------------------------------------------------


def test_add_arrangements_shapes_and_softmax():
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=True)
    N = 5
    arr, values, ents, log_probs, seat_idx, _ = _fake_batch(N, seed=1)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)
    # Stored values should have been softmaxed along the last dim.
    assert torch.allclose(
        buf.values.sum(dim=-1),
        torch.ones(N, ARRANGEMENT_SIZE), atol=1e-5,
    )
    assert buf.arrangements.shape == (N, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE)
    assert buf.seat_idx.shape == (N,)
    assert buf.ents.shape == (N, ARRANGEMENT_SIZE)
    assert buf.log_probs.shape == (N, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE)
    assert buf.counts.shape == (N, 1)
    assert buf.rewards.shape == (N, N_VF_CAT_DEFAULT)
    assert buf.ready_flags.shape == (N,)


def test_add_arrangements_dedup_keeps_newest():
    """Adding the SAME arrangement twice at different steps should keep only
    one row, with the larger step_added."""
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=True)
    arr, values, ents, log_probs, seat_idx, _ = _fake_batch(3, seed=2)
    # First add at step=0.
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)
    assert len(buf) == 3
    # Add the same arrangement #0 again at step=10 (one duplicate + two fresh).
    arr2, values2, ents2, log_probs2, seat_idx2, _ = _fake_batch(2, seed=3)
    arr_dup = torch.cat([arr[:1], arr2], dim=0)
    values_dup = torch.cat([
        torch.randn(1, ARRANGEMENT_SIZE, N_VF_CAT_DEFAULT), values2
    ], dim=0)
    ents_dup = torch.cat([torch.randn(1, ARRANGEMENT_SIZE).abs(), ents2], dim=0)
    logp_dup = torch.cat([
        F.log_softmax(torch.randn(1, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE), dim=-1),
        log_probs2,
    ], dim=0)
    seat_dup = torch.cat([seat_idx[:1], seat_idx2], dim=0)
    buf.add_arrangements(arr_dup, values_dup, ents_dup, logp_dup, seat_dup, step=10)
    # We had 3 + 3 = 6 rows before dedup; the duplicate row appears twice.
    # After dedup exactly 5 distinct ids should remain.
    assert len(buf) == 5
    # The max step_added should be 10 for the duplicated lineup.
    assert int(buf.step_added.max()) == 10


def test_add_arrangements_rejects_bad_dtypes():
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=True)
    arr, values, ents, log_probs, seat_idx, _ = _fake_batch(2, seed=4)
    # Wrong dtype for seat_idx.
    with pytest.raises(ValueError):
        buf.add_arrangements(arr, values, ents, log_probs, seat_idx.float(), step=0)
    # Wrong shape for values.
    with pytest.raises(ValueError):
        buf.add_arrangements(arr, values[..., :2], ents, log_probs, seat_idx, step=0)


# ---------------------------------------------------------------------------
# add_rewards — the core data-flow test.
# ---------------------------------------------------------------------------


def test_add_rewards_marks_row_ready_and_updates_mean():
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=False)
    arr, values, ents, log_probs, seat_idx, vocabs = _fake_batch(3, seed=5, use_cat_vf=False)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)

    # Env 0 uses row 0's arrangement, reward +1.
    env_arr = vocabs  # (3, 30) int64
    is_term = torch.tensor([True, False, True], dtype=torch.bool)
    rewards = torch.tensor([1.0, 0.0, -1.0])
    buf.add_rewards(env_arr, is_term, rewards)
    assert bool(buf.ready_flags[0])
    assert not bool(buf.ready_flags[1])  # its terminal flag was False
    assert bool(buf.ready_flags[2])
    assert float(buf.rewards[0]) == 1.0
    assert float(buf.rewards[2]) == -1.0

    # Second hit on row 0 with reward -1 → running mean = 0.0.
    is_term2 = torch.tensor([True, False, False], dtype=torch.bool)
    rewards2 = torch.tensor([-1.0, 0.0, 0.0])
    buf.add_rewards(env_arr, is_term2, rewards2)
    assert abs(float(buf.rewards[0])) < 1e-6
    assert int(buf.counts[0]) == 2


def test_add_rewards_cat_vf_onehot_conversion():
    """When use_cat_vf=True, scalar reward {-1,0,1} → one-hot bins (0,1,2)."""
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=True)
    arr, values, ents, log_probs, seat_idx, vocabs = _fake_batch(3, seed=6)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)

    is_term = torch.tensor([True, True, True], dtype=torch.bool)
    rewards = torch.tensor([1.0, 0.0, -1.0])
    buf.add_rewards(vocabs, is_term, rewards)
    # Row 0 reward is +1 → bin index 2 (win).
    assert torch.allclose(buf.rewards[0], torch.tensor([0.0, 0.0, 1.0]))
    # Row 1 reward 0 → bin 1 (draw).
    assert torch.allclose(buf.rewards[1], torch.tensor([0.0, 1.0, 0.0]))
    # Row 2 reward -1 → bin 0 (lose).
    assert torch.allclose(buf.rewards[2], torch.tensor([1.0, 0.0, 0.0]))


def test_add_rewards_unknown_arrangement_skipped():
    """An env whose arrangement isn't in the buffer should be ignored."""
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=False)
    arr, values, ents, log_probs, seat_idx, vocabs = _fake_batch(2, seed=7, use_cat_vf=False)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)

    import random
    stranger = _lineup_to_vocab(generate_random_lineup(random.Random(999)))
    env_arr = torch.stack([stranger, vocabs[0]])
    is_term = torch.tensor([True, True], dtype=torch.bool)
    rewards = torch.tensor([1.0, -1.0])
    buf.add_rewards(env_arr, is_term, rewards)
    # Only the second env should have updated the buffer.
    assert bool(buf.ready_flags[0])   # the one whose arrangement matches row 0
    assert not bool(buf.ready_flags[1])
    assert float(buf.rewards[0]) == -1.0


def test_add_rewards_before_add_arrangements_raises():
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=False)
    vocabs = torch.zeros(2, ARRANGEMENT_SIZE, dtype=torch.long)
    with pytest.raises(RuntimeError):
        buf.add_rewards(vocabs, torch.tensor([True, False]), torch.tensor([1.0, 0.0]))


# ---------------------------------------------------------------------------
# process_data — MC backup math
# ---------------------------------------------------------------------------


def test_process_data_scalar_vf_mc_target_equals_reward():
    """With td_lambda=gae_lambda=1 and td_target starting at reward,
    val_est[:, 0] should equal reward for ready rows (pure MC → backup to t=0)."""
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=False)
    arr, values, ents, log_probs, seat_idx, vocabs = _fake_batch(3, seed=8, use_cat_vf=False)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)
    rewards = torch.tensor([1.0, 0.0, -1.0])
    buf.add_rewards(vocabs, torch.ones(3, dtype=torch.bool), rewards)
    stats = buf.process_data(td_lambda=1.0, gae_lambda=1.0, reg_temp=0.0, reg_norm=10.0)
    assert "arr_buf/abs_adv_q90" in stats
    # Cross-check: for scalar VF, TD(1) backward from reward gives
    # val_est[:, 0] = rewards (telescoping V_t cancels).
    assert torch.allclose(buf.val_est[:, 0], rewards, atol=1e-5)


def test_process_data_cat_vf_scalar_adv_has_right_sign():
    """Advantage direction should correlate with reward for cat-VF setup:
    if reward was +1 (win bin) the aggregated adv should be positive at t=0
    once the random-init value is softmaxed to a near-uniform distribution."""
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=True)
    arr, values, ents, log_probs, seat_idx, vocabs = _fake_batch(3, seed=9)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)
    rewards = torch.tensor([1.0, 0.0, -1.0])
    buf.add_rewards(vocabs, torch.ones(3, dtype=torch.bool), rewards)
    buf.process_data(td_lambda=1.0, gae_lambda=1.0, reg_temp=0.0, reg_norm=10.0)
    # Advantages at t=0 scaled by categorical_aggregation should roughly
    # reflect: row with reward +1 → positive; reward -1 → negative.
    adv_t0 = buf.adv_est[:, 0]   # already scalar after aggregation
    assert float(adv_t0[0]) > float(adv_t0[2]), "reward-+1 row should have larger adv than reward-(-1) row"


def test_process_data_empty_returns_empty_dict():
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=True)
    arr, values, ents, log_probs, seat_idx, _ = _fake_batch(2, seed=10)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)
    # No rewards added → no ready rows.
    stats = buf.process_data(td_lambda=1.0, gae_lambda=1.0, reg_temp=0.02, reg_norm=10.0)
    assert stats == {}


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def test_sample_batches_cover_all_ready_rows():
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=False)
    N = 10
    arr, values, ents, log_probs, seat_idx, vocabs = _fake_batch(N, seed=11, use_cat_vf=False)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)
    buf.add_rewards(vocabs, torch.ones(N, dtype=torch.bool), torch.arange(N).float() * 0.1 - 0.5)
    buf.process_data(td_lambda=1.0, gae_lambda=1.0, reg_temp=0.02, reg_norm=10.0)

    seen = 0
    for batch in buf.sample(batch_size=3):
        assert isinstance(batch, Batch)
        assert batch.arrangements.size(0) <= 3
        assert batch.arrangements.size(1) == ARRANGEMENT_SIZE
        assert batch.seat_idx.ndim == 1
        seen += batch.arrangements.size(0)
    assert seen == N


def test_sample_before_add_raises():
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=False)
    with pytest.raises(RuntimeError):
        list(buf.sample(batch_size=4))


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_drops_expired_rows():
    buf = ArrangementBuffer(storage_duration=10, device="cpu", use_cat_vf=False)
    arr, values, ents, log_probs, seat_idx, vocabs = _fake_batch(5, seed=12, use_cat_vf=False)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)
    buf.filter(current_step=5)   # all rows still within duration=10
    assert len(buf) == 5
    buf.filter(current_step=11)  # all rows now expired
    assert len(buf) == 0
    assert buf.need_arrangements is True


def test_filter_forces_need_arrangements():
    buf = ArrangementBuffer(storage_duration=10, device="cpu", use_cat_vf=False)
    arr, values, ents, log_probs, seat_idx, _ = _fake_batch(3, seed=13, use_cat_vf=False)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)
    assert buf.need_arrangements is False
    buf.filter(current_step=5)
    # Even if rows survive, filter() resets need_arrangements=True.
    assert buf.need_arrangements is True


# ---------------------------------------------------------------------------
# Full lifecycle smoke
# ---------------------------------------------------------------------------


def test_full_lifecycle():
    """add_arrangements → add_rewards → process_data → sample → filter loop."""
    buf = ArrangementBuffer(storage_duration=50, device="cpu", use_cat_vf=True)
    for epoch in range(3):
        arr, values, ents, log_probs, seat_idx, vocabs = _fake_batch(
            8, seed=100 + epoch,
        )
        buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=epoch * 10)
        term = torch.ones(8, dtype=torch.bool)
        rewards = torch.tensor([1.0, 0.0, -1.0, 1.0, -1.0, 0.0, 1.0, -1.0])
        buf.add_rewards(vocabs, term, rewards)
        stats = buf.process_data(
            td_lambda=1.0, gae_lambda=1.0, reg_temp=0.02, reg_norm=10.0,
        )
        assert "arr_buf/n_ready" in stats
        # Consume all minibatches.
        total = sum(b.arrangements.size(0) for b in buf.sample(batch_size=4))
        assert total == int(stats["arr_buf/n_ready"])
        buf.filter(current_step=epoch * 10 + 10)


# ---------------------------------------------------------------------------
# Regression: memory-bounded add_arrangements (v17 OOM fix)
# ---------------------------------------------------------------------------


def test_add_arrangements_peak_memory_bounded():
    """Regression test for the v17 OOM at rollout 374.

    In the legacy implementation ``add_arrangements`` called
    ``cat(new, old)`` and then ``buffer[keep_mask]`` on the combined
    tensor, peaking at ~3× buffer size. With a 100k-row buffer this
    pushed the CUDA allocator over 14 GB on T4.

    The fix dedups the old rows *before* cat, so peak ≈ 1× buffer. We
    verify the semantic: after many adds with overlapping row IDs, the
    buffer size stays bounded (rows are deduped, not accumulated) when
    all new IDs collide with prior IDs.
    """
    buf = ArrangementBuffer(storage_duration=1000, device="cpu", use_cat_vf=False)

    # First epoch: add 8 distinct rows.
    arr, values, ents, log_probs, seat_idx, _ = _fake_batch(8, seed=7, use_cat_vf=False)
    buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=0)
    assert len(buf) == 8

    # Subsequent epochs: add the SAME 8 rows (identical hashes). Because
    # each new id collides with the prior one and the new row supersedes
    # the old (larger step_added), buffer size should stay at 8.
    for epoch in range(1, 10):
        buf.add_arrangements(arr, values, ents, log_probs, seat_idx, step=epoch)
        assert len(buf) == 8, (
            f"epoch {epoch}: buffer size drifted to {len(buf)}; "
            "dedup is leaking rows."
        )
        # Validate that step_added tracks the latest insertion.
        assert bool((buf.step_added == epoch).all()), (
            f"epoch {epoch}: step_added not updated to latest; "
            f"saw {buf.step_added.unique().tolist()}"
        )


def test_add_arrangements_mixed_new_and_existing():
    """Half-overlapping batches: new rows append, overlapping rows supersede."""
    buf = ArrangementBuffer(storage_duration=1000, device="cpu", use_cat_vf=False)

    # Epoch 0: 8 unique rows [A,B,C,D,E,F,G,H].
    arr0, v0, e0, lp0, s0, _ = _fake_batch(8, seed=21, use_cat_vf=False)
    buf.add_arrangements(arr0, v0, e0, lp0, s0, step=0)
    assert len(buf) == 8

    # Epoch 1: 4 overlapping rows [A,B,C,D] + 4 new rows [I,J,K,L].
    arr1, v1, e1, lp1, s1, _ = _fake_batch(4, seed=22, use_cat_vf=False)
    arr_mixed = torch.cat([arr0[:4], arr1], dim=0)
    v_mixed = torch.cat([v0[:4], v1], dim=0)
    e_mixed = torch.cat([e0[:4], e1], dim=0)
    lp_mixed = torch.cat([lp0[:4], lp1], dim=0)
    s_mixed = torch.cat([s0[:4], s1], dim=0)
    buf.add_arrangements(arr_mixed, v_mixed, e_mixed, lp_mixed, s_mixed, step=1)

    # Expected: 12 unique rows (8 original, 4 of them refreshed to step=1,
    # plus 4 new rows).
    assert len(buf) == 12, f"expected 12 unique rows, got {len(buf)}"
    # 4 rows should be at step=1 (the new batch + the 4 refreshed).
    assert int((buf.step_added == 1).sum()) == 8, (
        "4 new + 4 refreshed should all be at step=1"
    )
    # 4 rows should still be at step=0 (the un-overlapped [E,F,G,H]).
    assert int((buf.step_added == 0).sum()) == 4

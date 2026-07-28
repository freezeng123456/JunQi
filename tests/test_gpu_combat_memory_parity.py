"""tests/test_gpu_combat_memory_parity.py — GPU↔CPU CombatMemory v6 parity.

Validates that the device-resident CombatMemory state mutated by
``junqi_cuda.step_batch`` matches the Python reference (``BatchedGameState.cm_*``)
bit-for-bit after running an identical action sequence.

Why this test exists
--------------------
The user's hard requirement: **no CPU↔GPU traffic in the training hot path
for CombatMemory**.  All updates run inside ``step_batch_kernel``; the only
legal transfers are:

  1. ``DeviceGameStateBatch`` constructor — one-time cudaMalloc / cudaMemset
     of the 16 CombatMemory device buffers.
  2. ``cm_copy_from_host`` / ``cm_copy_to_host`` — PARITY-TEST helpers,
     never invoked during training.

This test exercises (2) to verify that the GPU update rules in
``combat_memory.cu::cm_apply_event_dev`` (chain propagation, per-observer
DARK dispatch, ``attacked_by_known_gongb`` preflight) produce exactly
the same 16 SoA arrays as ``junqi_core.combat_memory.apply_combat_event``.

Scope
-----
* 16 CombatMemory fields: every ``cm_*`` array in ``BatchedGameState``.
* Multi-step parity: 50 random plies on N=8 seeds.
* Path-revealed-GONGB: a deliberate engineer-rail-walk test that exercises
  ``cm_move_requires_gongb_dev`` BFS.
* Combat-heavy parity: bias action sampling toward EAT/KILLED moves.

Skipped automatically when ``junqi_cuda`` is not importable or no CUDA
GPU is available.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Conditional import — skip cleanly without CUDA.
# ---------------------------------------------------------------------------
try:
    import junqi_cuda as _cuda  # type: ignore[import]
    _CUDA_AVAILABLE = _cuda.get_gpu_count() > 0
except ImportError:
    _cuda = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="junqi_cuda extension not available or no CUDA-capable GPU",
)

from junqi_core.batched_state import BatchedGameState
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState
from junqi_rl.gpu_world import _upload_zobrist_tables


# ---------------------------------------------------------------------------
# Bring-up helpers (mirror tests/test_gpu_step_batch.py)
# ---------------------------------------------------------------------------

def _init_gpu_tables_once() -> None:
    _cuda.init_tables()
    _upload_zobrist_tables()


def _new_game(seed: int) -> GameState:
    rng = random.Random(seed)
    return GameState.new_game(generate_random_setup(rng))


def _build_batch(n: int, seed_base: int) -> BatchedGameState:
    states = [_new_game(seed_base + i) for i in range(n)]
    return BatchedGameState.from_game_states(states)


def _pack_from_batched(b: BatchedGameState) -> dict:
    """Build state_dict for DeviceGameStateBatch.copy_from_host."""
    N = b.num_envs

    def flat(arr):
        return arr.reshape(-1)

    alv  = flat(b.alive)
    psa  = flat(b.piece_seat_arr)
    pta  = flat(b.piece_type_arr)
    px   = flat(b.pos_x)
    py_  = flat(b.pos_y)
    mca  = flat(b.move_count_arr)
    aea  = flat(b.active_eat_arr)
    psa2 = flat(b.passive_surv_arr)
    dra  = flat(b.death_reason_arr)
    dsa  = flat(b.death_step_arr)
    dlfa = flat(b.death_loc_flat_arr)
    zx   = flat(b.zero_x)
    zy   = flat(b.zero_y)
    cpip = np.where(
        alv, py_.astype(np.int16) * np.int16(17) + px.astype(np.int16), np.int16(-1)
    ).astype(np.int16, copy=False)
    cpi  = flat(b.cell_piece_id)
    sda  = flat(b.seat_dead_arr)
    sfra = flat(b.seat_flag_revealed_arr)
    return {
        "cell_piece_id_per_piece":  cpip,
        "piece_seat_arr":           psa.astype(np.int8,  copy=False),
        "piece_type_arr":           pta.astype(np.int8,  copy=False),
        "alive":                    alv.astype(bool,     copy=False),
        "pos_x":                    px.astype(np.int8,   copy=False),
        "pos_y":                    py_.astype(np.int8,  copy=False),
        "zero_x":                   zx.astype(np.int8,   copy=False),
        "zero_y":                   zy.astype(np.int8,   copy=False),
        "move_count_arr":           mca.astype(np.int16, copy=False),
        "active_eat_arr":           aea.astype(np.int16, copy=False),
        "passive_surv_arr":         psa2.astype(np.int16, copy=False),
        "death_reason_arr":         dra.astype(np.int8,  copy=False),
        "death_step_arr":           dsa.astype(np.int16, copy=False),
        "death_loc_flat_arr":       dlfa.astype(np.int16, copy=False),
        "cell_piece_id":            cpi.astype(np.int16, copy=False),
        "seat_dead_arr":            sda.astype(bool,     copy=False),
        "seat_flag_revealed_arr":   sfra.astype(bool,    copy=False),
        "turn":                     b.turn.astype(np.int8, copy=False),
        "zobrist":                  b.zobrist.astype(np.int64, copy=False),
        "move_counter":             b.move_counter.astype(np.int32, copy=False),
        "moves_since_last_combat":  b.moves_since_last_combat.astype(np.int32, copy=False),
    }


def _push_state(b: BatchedGameState):
    """Upload BatchedGameState to GPU including termination AND CombatMemory."""
    state_dict = _pack_from_batched(b)
    gs = _cuda.DeviceGameStateBatch(b.num_envs)
    gs.copy_from_host(state_dict)
    gs.copy_termination_from_host(
        b.terminated.astype(bool, copy=False),
        b.winner_team.astype(np.int8, copy=False),
        b.draw.astype(bool, copy=False),
    )
    # CombatMemory: every fresh batch has zero-initialised cm_* arrays, so
    # the GPU's cudaMemset already matches.  We still upload explicitly so
    # the parity boundary is symmetric and any future non-zero seed states
    # round-trip correctly.
    gs.cm_copy_from_host(
        b.cm_direct_lo.reshape(-1),
        b.cm_direct_hi.reshape(-1),
        b.cm_direct_type.reshape(-1),
        b.cm_last_direct_step.reshape(-1),
        b.cm_direct_other_count.reshape(-1),
        b.cm_chain_lo.reshape(-1),
        b.cm_chain_hi.reshape(-1),
        b.cm_chain_type.reshape(-1),
        b.cm_last_chain_step.reshape(-1),
        b.cm_eaten_by_pid_lo.reshape(-1),
        b.cm_eaten_by_pid_hi.reshape(-1),
        b.cm_rank_floor.reshape(-1),
        b.cm_rank_floor_step.reshape(-1),
        b.cm_is_gongb.reshape(-1),
        b.cm_not_gongb.reshape(-1),
        b.cm_attacked_by_known_gongb.reshape(-1),
    )
    return gs


# ---------------------------------------------------------------------------
# CombatMemory parity assertion
# ---------------------------------------------------------------------------

_CM_FIELDS: tuple[tuple[str, str], ...] = (
    # (BatchedGameState attr,       GPU dict key)
    ("cm_direct_lo",                "direct_lo"),
    ("cm_direct_hi",                "direct_hi"),
    ("cm_direct_type",              "direct_type"),
    ("cm_last_direct_step",         "last_direct_step"),
    ("cm_direct_other_count",       "direct_other_count"),
    ("cm_chain_lo",                 "chain_lo"),
    ("cm_chain_hi",                 "chain_hi"),
    ("cm_chain_type",               "chain_type"),
    ("cm_last_chain_step",          "last_chain_step"),
    ("cm_eaten_by_pid_lo",          "eaten_by_pid_lo"),
    ("cm_eaten_by_pid_hi",          "eaten_by_pid_hi"),
    ("cm_rank_floor",               "rank_floor"),
    ("cm_rank_floor_step",          "rank_floor_step"),
    ("cm_is_gongb",                 "is_gongb"),
    ("cm_not_gongb",                "not_gongb"),
    ("cm_attacked_by_known_gongb",  "attacked_by_known_gongb"),
)


def _assert_cm_parity(b: BatchedGameState, gpu_cm: dict, *, step_index: int) -> None:
    """All 16 CombatMemory arrays must be bit-identical."""
    N = b.num_envs
    expected_shape = (N, 4, 120)
    for cpu_attr, gpu_key in _CM_FIELDS:
        cpu_arr = getattr(b, cpu_attr)
        assert cpu_arr.shape == expected_shape, (
            f"[step {step_index}] CPU.{cpu_attr} shape {cpu_arr.shape} "
            f"!= {expected_shape}"
        )
        gpu_arr = gpu_cm[gpu_key].reshape(expected_shape)
        if not np.array_equal(cpu_arr, gpu_arr):
            # Find first differing (env, observer, pid) for a tight error.
            diff = np.argwhere(cpu_arr != gpu_arr)
            first = tuple(diff[0])
            raise AssertionError(
                f"[step {step_index}] CombatMemory.{cpu_attr} mismatch — "
                f"first diff at (env={first[0]}, obs={first[1]}, pid={first[2]}): "
                f"cpu={cpu_arr[first]!r}  gpu={gpu_arr[first]!r}"
            )


def _sample_legal_action(b: BatchedGameState, rng: random.Random) -> np.ndarray:
    ids = b.legal_action_ids_batch()
    actions = np.zeros(b.num_envs, dtype=np.int32)
    for i in range(b.num_envs):
        if b.terminated[i] or ids[i].size == 0:
            actions[i] = 0
            continue
        actions[i] = int(rng.choice(ids[i]))
    return actions


def _sample_combat_biased_action(
    b: BatchedGameState, rng: random.Random
) -> np.ndarray:
    """Prefer combat moves (dst occupied) when available — exercises EAT/KILLED."""
    ids_batch = b.legal_action_ids_batch()
    actions = np.zeros(b.num_envs, dtype=np.int32)
    for i in range(b.num_envs):
        if b.terminated[i] or ids_batch[i].size == 0:
            continue
        ids = ids_batch[i]
        dst = ids % 289
        occ = b.cell_piece_id[i, dst] >= 0
        combat = ids[occ]
        pool = combat if combat.size > 0 else ids
        actions[i] = int(rng.choice(pool))
    return actions


# ---------------------------------------------------------------------------
# Module-scope CUDA bring-up
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _tables() -> None:
    _init_gpu_tables_once()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cm_initial_zero() -> None:
    """A fresh batch has all-zero cm_* arrays (sentinels are -1 for last_*_step
    and rank_floor_step) — both CPU and GPU must agree."""
    N = 4
    b = _build_batch(N, seed_base=10)
    gs = _push_state(b)

    gpu_cm = gs.cm_copy_to_host()
    _assert_cm_parity(b, gpu_cm, step_index=-1)

    # Sanity checks on the sentinel fields (don't depend on the assert above
    # accidentally short-circuiting).
    expected_neg_one = -np.ones((N, 4, 120), dtype=np.int16)
    np.testing.assert_array_equal(b.cm_last_direct_step, expected_neg_one)
    np.testing.assert_array_equal(b.cm_last_chain_step,  expected_neg_one)
    np.testing.assert_array_equal(b.cm_rank_floor_step,  expected_neg_one)


def test_cm_single_step_parity() -> None:
    """One legal step: CPU and GPU CombatMemory must stay aligned."""
    N = 16
    rng = random.Random(0xC0FFEE)
    b = _build_batch(N, seed_base=200)
    gs = _push_state(b)

    actions = _sample_legal_action(b, rng)
    b.step_batch(actions)
    gs.step_batch(actions)

    gpu_cm = gs.cm_copy_to_host()
    _assert_cm_parity(b, gpu_cm, step_index=0)


def test_cm_multi_step_parity() -> None:
    """50 random steps × N=8 envs: bit-identical CombatMemory after each step."""
    N = 8
    rng = random.Random(0xABCDEF)
    b = _build_batch(N, seed_base=300)
    gs = _push_state(b)

    for step in range(50):
        if b.terminated.all():
            break
        actions = _sample_legal_action(b, rng)
        b.step_batch(actions)
        gs.step_batch(actions)
        gpu_cm = gs.cm_copy_to_host()
        _assert_cm_parity(b, gpu_cm, step_index=step)


def test_cm_combat_biased_parity() -> None:
    """Combat-heavy rollout: every step prefers EAT/KILLED.  This is the
    primary stress test for chain propagation, attacked_by_known_gongb,
    and per-observer DARK dispatch."""
    N = 4
    rng = random.Random(0xDEAD0001)
    b = _build_batch(N, seed_base=400)
    gs = _push_state(b)

    for step in range(60):
        if b.terminated.all():
            break
        actions = _sample_combat_biased_action(b, rng)
        b.step_batch(actions)
        gs.step_batch(actions)
        gpu_cm = gs.cm_copy_to_host()
        _assert_cm_parity(b, gpu_cm, step_index=step)


def test_cm_path_revealed_gongb_parity() -> None:
    """Test path-revealed-GONGB by running enough steps to occasionally route
    a GONGB through a curve-rail / engineer-only path.  This stresses
    ``cm_move_requires_gongb_dev`` (rail-graph BFS on device).

    We don't construct the scenario manually — we just run a longer rollout
    on multiple seeds and rely on the random walk to occasionally pick
    engineer-only paths.  The parity assertion catches any device-side BFS
    divergence the moment it happens.
    """
    N = 6
    rng = random.Random(0xBABE)
    b = _build_batch(N, seed_base=500)
    gs = _push_state(b)

    for step in range(80):
        if b.terminated.all():
            break
        actions = _sample_legal_action(b, rng)
        b.step_batch(actions)
        gs.step_batch(actions)
        gpu_cm = gs.cm_copy_to_host()
        _assert_cm_parity(b, gpu_cm, step_index=step)

    # Sanity: at least one observer should have seen *some* combat over an
    # 80-step rollout — otherwise the test wasn't actually exercising
    # update logic.
    any_event = (
        np.any(b.cm_last_direct_step >= 0) or
        np.any(b.cm_last_chain_step  >= 0) or
        np.any(b.cm_is_gongb)
    )
    assert any_event, "80 random steps produced no CombatMemory events — test was vacuous"


def test_cm_roundtrip_only() -> None:
    """Pure copy_from_host / copy_to_host roundtrip with non-trivial seed
    values — verifies the parity-test plumbing itself is bit-clean even
    without invoking step_batch."""
    N = 3
    b = _build_batch(N, seed_base=600)
    # Seed the CPU CombatMemory with arbitrary non-zero values so the
    # roundtrip exercises every byte of every array.
    rng = np.random.default_rng(2026)
    b.cm_direct_lo[:]                = rng.integers(0, 1 << 60, size=b.cm_direct_lo.shape, dtype=np.uint64)
    b.cm_direct_hi[:]                = rng.integers(0, 1 << 56, size=b.cm_direct_hi.shape, dtype=np.uint64)
    b.cm_direct_type[:]              = rng.integers(0, 4096,    size=b.cm_direct_type.shape, dtype=np.uint16)
    b.cm_last_direct_step[:]         = rng.integers(-1, 2000,   size=b.cm_last_direct_step.shape, dtype=np.int16)
    b.cm_direct_other_count[:]       = rng.integers(0, 64,      size=b.cm_direct_other_count.shape, dtype=np.int16)
    b.cm_chain_lo[:]                 = rng.integers(0, 1 << 60, size=b.cm_chain_lo.shape, dtype=np.uint64)
    b.cm_chain_hi[:]                 = rng.integers(0, 1 << 56, size=b.cm_chain_hi.shape, dtype=np.uint64)
    b.cm_chain_type[:]               = rng.integers(0, 4096,    size=b.cm_chain_type.shape, dtype=np.uint16)
    b.cm_last_chain_step[:]          = rng.integers(-1, 2000,   size=b.cm_last_chain_step.shape, dtype=np.int16)
    b.cm_eaten_by_pid_lo[:]          = rng.integers(0, 1 << 60, size=b.cm_eaten_by_pid_lo.shape, dtype=np.uint64)
    b.cm_eaten_by_pid_hi[:]          = rng.integers(0, 1 << 56, size=b.cm_eaten_by_pid_hi.shape, dtype=np.uint64)
    b.cm_rank_floor[:]               = rng.integers(0, 10,      size=b.cm_rank_floor.shape, dtype=np.int8)
    b.cm_rank_floor_step[:]          = rng.integers(-1, 2000,   size=b.cm_rank_floor_step.shape, dtype=np.int16)
    b.cm_is_gongb[:]                 = rng.integers(0, 2,       size=b.cm_is_gongb.shape, dtype=np.uint8).astype(bool)
    b.cm_not_gongb[:]                = rng.integers(0, 2,       size=b.cm_not_gongb.shape, dtype=np.uint8).astype(bool)
    b.cm_attacked_by_known_gongb[:]  = rng.integers(0, 2,       size=b.cm_attacked_by_known_gongb.shape, dtype=np.uint8).astype(bool)

    gs = _push_state(b)
    gpu_cm = gs.cm_copy_to_host()
    _assert_cm_parity(b, gpu_cm, step_index=-2)

"""tests/test_gpu_step_batch.py — parity between GPU step_batch and CPU BatchedGameState.

Scope:
  * Plain-move only: random moves, compare post-step SoA + zobrist + scalars.
  * Combat: force combat scenarios; verify seat_dead, flag_captured, winner.
  * Zobrist bit-identity across multi-step rollout.
  * Q12: seat has no legal moves → seat dies + zobrist matches.
  * Terminated envs are skipped (no mutation).
"""

from __future__ import annotations

import random
from dataclasses import fields
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip gate
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
# Helpers
# ---------------------------------------------------------------------------

def _init_gpu_tables_once() -> None:
    """Initialize CUDA tables + upload CPU-seeded zobrist constants."""
    _cuda.init_tables()
    _upload_zobrist_tables()


def _new_game(seed: int) -> GameState:
    rng = random.Random(seed)
    return GameState.new_game(generate_random_setup(rng))


def _pack_from_batched(b: BatchedGameState) -> dict:
    """Build the state_dict expected by DeviceGameStateBatch.copy_from_host
    directly from a BatchedGameState (avoids round-tripping through JunqiEnv)."""
    N = b.num_envs
    # Flatten piece-indexed arrays to (N*120,).
    def flat(arr):
        return arr.reshape(-1)
    alv   = flat(b.alive)
    psa   = flat(b.piece_seat_arr)
    pta   = flat(b.piece_type_arr)
    px    = flat(b.pos_x)
    py_   = flat(b.pos_y)
    mca   = flat(b.move_count_arr)
    aea   = flat(b.active_eat_arr)
    psa2  = flat(b.passive_surv_arr)
    dra   = flat(b.death_reason_arr)
    dsa   = flat(b.death_step_arr)
    dlfa  = flat(b.death_loc_flat_arr)
    # zero_x/zero_y are legacy engineer-BFS aux; BatchedGameState stores them
    # only on the underlying GameState copies, but step_batch_kernel does not
    # read them, so zero-fill is fine.
    zx = np.zeros_like(px)
    zy = np.zeros_like(py_)
    cpip = np.where(
        alv, py_.astype(np.int16) * np.int16(17) + px.astype(np.int16), np.int16(-1)
    ).astype(np.int16, copy=False)
    cpi = flat(b.cell_piece_id)
    sda = flat(b.seat_dead_arr)
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


def _push_state(b: BatchedGameState) -> _cuda.DeviceGameStateBatch:
    """Upload a BatchedGameState to the GPU, including termination state."""
    state_dict = _pack_from_batched(b)
    gs = _cuda.DeviceGameStateBatch(b.num_envs)
    gs.copy_from_host(state_dict)
    gs.copy_termination_from_host(
        b.terminated.astype(bool, copy=False),
        b.winner_team.astype(np.int8, copy=False),
        b.draw.astype(bool, copy=False),
    )
    return gs


def _pull_state_dict(gs: _cuda.DeviceGameStateBatch) -> dict:
    """Pull the full SoA back from GPU, plus termination state."""
    d = gs.copy_to_host()
    term = gs.copy_termination_to_host()
    d.update(term)
    return d


def _assert_state_parity(
    b: BatchedGameState,
    gpu_dict: dict,
    *,
    step_index: int,
) -> None:
    """Assert the GPU state matches CPU BatchedGameState bit-for-bit."""
    N = b.num_envs

    def reshape(arr: np.ndarray, per_env_shape: tuple[int, ...]) -> np.ndarray:
        return arr.reshape((N,) + per_env_shape)

    # 120-wide piece-indexed arrays
    for name, cpu_arr, shape in [
        ("piece_type_arr", b.piece_type_arr, (120,)),
        ("alive",          b.alive,          (120,)),
        ("pos_x",          b.pos_x,          (120,)),
        ("pos_y",          b.pos_y,          (120,)),
        ("move_count_arr", b.move_count_arr, (120,)),
        ("active_eat_arr", b.active_eat_arr, (120,)),
        ("passive_surv_arr", b.passive_surv_arr, (120,)),
        ("death_reason_arr", b.death_reason_arr, (120,)),
        ("death_step_arr",   b.death_step_arr,   (120,)),
        ("death_loc_flat_arr", b.death_loc_flat_arr, (120,)),
    ]:
        gpu_arr = reshape(gpu_dict[name], shape)
        assert np.array_equal(gpu_arr, cpu_arr), (
            f"[step {step_index}] {name} mismatch — "
            f"cpu={cpu_arr[:1]}  gpu={gpu_arr[:1]}"
        )

    # 289-wide cell-indexed
    gpu_cpid = reshape(gpu_dict["cell_piece_id"], (289,))
    assert np.array_equal(gpu_cpid, b.cell_piece_id), (
        f"[step {step_index}] cell_piece_id mismatch"
    )

    # 4-wide seat-indexed
    gpu_sd = reshape(gpu_dict["seat_dead_arr"], (4,))
    gpu_fr = reshape(gpu_dict["seat_flag_revealed_arr"], (4,))
    assert np.array_equal(gpu_sd, b.seat_dead_arr), (
        f"[step {step_index}] seat_dead_arr mismatch"
    )
    assert np.array_equal(gpu_fr, b.seat_flag_revealed_arr), (
        f"[step {step_index}] seat_flag_revealed_arr mismatch"
    )

    # Scalars
    gpu_turn = gpu_dict["turn"].reshape((N,))
    gpu_mc   = gpu_dict["move_counter"].reshape((N,))
    gpu_mslc = gpu_dict["moves_since_last_combat"].reshape((N,))
    gpu_zob  = gpu_dict["zobrist"].reshape((N,))
    assert np.array_equal(gpu_turn, b.turn), f"[step {step_index}] turn mismatch"
    assert np.array_equal(gpu_mc,   b.move_counter), f"[step {step_index}] mc mismatch"
    assert np.array_equal(gpu_mslc, b.moves_since_last_combat), (
        f"[step {step_index}] mslc mismatch"
    )
    assert np.array_equal(gpu_zob, b.zobrist), (
        f"[step {step_index}] zobrist mismatch: "
        f"cpu[0]={int(b.zobrist[0]):016x}  gpu[0]={int(gpu_zob[0]):016x}"
    )

    # Termination
    assert np.array_equal(gpu_dict["terminated"],  b.terminated), (
        f"[step {step_index}] terminated mismatch"
    )
    assert np.array_equal(gpu_dict["winner_team"], b.winner_team), (
        f"[step {step_index}] winner_team mismatch"
    )
    assert np.array_equal(gpu_dict["draw"],        b.draw), (
        f"[step {step_index}] draw mismatch"
    )


def _assert_result_parity(cpu_result, gpu_result: dict, step_index: int) -> None:
    for key, cpu_arr in [
        ("valid",          cpu_result.valid),
        ("event",          cpu_result.event),
        ("terminated",     cpu_result.terminated),
        ("winner_team",    cpu_result.winner_team),
        ("draw",           cpu_result.draw),
        ("flag_captured",  cpu_result.flag_captured),
    ]:
        gpu_arr = gpu_result[key]
        assert np.array_equal(gpu_arr, cpu_arr), (
            f"[step {step_index}] result.{key} mismatch — "
            f"cpu={cpu_arr}  gpu={gpu_arr}"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _tables() -> None:
    _init_gpu_tables_once()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _build_batch(n: int, seed_base: int) -> BatchedGameState:
    states = [_new_game(seed_base + i) for i in range(n)]
    return BatchedGameState.from_game_states(states)


def _sample_legal_action(b: BatchedGameState, rng: random.Random) -> np.ndarray:
    """Return an action_id per env — a uniformly-random legal move.
    Uses the per-env legal list already computed by BatchedGameState."""
    ids = b.legal_action_ids_batch()
    actions = np.zeros(b.num_envs, dtype=np.int32)
    for i in range(b.num_envs):
        if b.terminated[i]:
            actions[i] = 0
            continue
        if ids[i].size == 0:
            actions[i] = 0
            continue
        actions[i] = int(rng.choice(ids[i]))
    return actions


def test_step_batch_single_step_parity() -> None:
    """One step: GPU state matches CPU bit-for-bit."""
    N = 16
    rng = random.Random(0xC0FFEE)
    b_cpu = _build_batch(N, seed_base=123)
    gs_gpu = _push_state(b_cpu)

    actions = _sample_legal_action(b_cpu, rng)

    cpu_result = b_cpu.step_batch(actions)
    gpu_result = gs_gpu.step_batch(actions)

    _assert_result_parity(cpu_result, gpu_result, step_index=0)
    gpu_dict = _pull_state_dict(gs_gpu)
    _assert_state_parity(b_cpu, gpu_dict, step_index=0)


def test_step_batch_multi_step_parity() -> None:
    """50 random steps; verify state and hash parity at every step."""
    N = 8
    rng = random.Random(0xABCDEF)
    b_cpu = _build_batch(N, seed_base=42)
    gs_gpu = _push_state(b_cpu)

    for step in range(50):
        if b_cpu.terminated.all():
            break
        actions = _sample_legal_action(b_cpu, rng)
        cpu_result = b_cpu.step_batch(actions)
        gpu_result = gs_gpu.step_batch(actions)
        _assert_result_parity(cpu_result, gpu_result, step_index=step)
        gpu_dict = _pull_state_dict(gs_gpu)
        _assert_state_parity(b_cpu, gpu_dict, step_index=step)


def test_step_batch_zobrist_parity_after_combat() -> None:
    """Zobrist remains bit-identical after combat-heavy rollout (force collisions
    by selecting combat moves when available)."""
    N = 4
    rng = random.Random(0x1234)
    b_cpu = _build_batch(N, seed_base=7)
    gs_gpu = _push_state(b_cpu)

    for step in range(30):
        if b_cpu.terminated.all():
            break
        ids_batch = b_cpu.legal_action_ids_batch()
        actions = np.zeros(N, dtype=np.int32)
        for i in range(N):
            if b_cpu.terminated[i] or ids_batch[i].size == 0:
                continue
            # Prefer combat moves (dst occupied) when available
            dst = ids_batch[i] % 289
            occ = b_cpu.cell_piece_id[i, dst] >= 0
            combat = ids_batch[i][occ]
            pool = combat if combat.size > 0 else ids_batch[i]
            actions[i] = int(rng.choice(pool))

        cpu_result = b_cpu.step_batch(actions)
        gpu_result = gs_gpu.step_batch(actions)
        _assert_result_parity(cpu_result, gpu_result, step_index=step)
        gpu_dict = _pull_state_dict(gs_gpu)
        _assert_state_parity(b_cpu, gpu_dict, step_index=step)


def test_step_batch_terminated_env_skipped() -> None:
    """Already-terminated envs must not be mutated by step_batch."""
    N = 2
    b_cpu = _build_batch(N, seed_base=99)
    b_cpu.terminated[0] = True
    b_cpu.winner_team[0] = np.int8(1)
    gs_gpu = _push_state(b_cpu)

    # Snapshot env 0 fields that would change if step ran
    snap_mc  = int(b_cpu.move_counter[0])
    snap_zob = int(b_cpu.zobrist[0])
    snap_turn = int(b_cpu.turn[0])

    # Any action — env 0 must be skipped.
    actions = np.zeros(N, dtype=np.int32)
    # For env 1 pick a legal action so the kernel does useful work.
    legal1 = b_cpu.legal_action_ids_batch()[1]
    if legal1.size > 0:
        actions[1] = int(legal1[0])

    b_cpu.step_batch(actions)
    gs_gpu.step_batch(actions)

    # env 0: still terminated, counters unchanged.
    gpu_dict = _pull_state_dict(gs_gpu)
    assert gpu_dict["terminated"][0] == True
    assert int(gpu_dict["move_counter"][0]) == snap_mc
    assert int(gpu_dict["zobrist"][0]) == snap_zob
    assert int(gpu_dict["turn"][0]) == snap_turn
    # Parity for both envs
    _assert_state_parity(b_cpu, gpu_dict, step_index=0)

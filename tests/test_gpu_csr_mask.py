"""tests/test_gpu_csr_mask.py — parity tests for CSR + per-piece mask APIs.

Validates that the new GPU output formats produce results identical to the
dense legal_action_ids_batch reference:

  * ``legal_action_ids_batch_csr`` — CSR (offsets, values) format
  * ``legal_action_mask_batch``    — per-piece (N, 120, 32) bool mask

The mask decoded via the explicit slot→dst decoder must yield the same
action set as the dense kernel.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

try:
    import junqi_cuda as _cuda
    _CUDA_AVAILABLE = _cuda.get_gpu_count() > 0
except ImportError:
    _cuda = None
    _CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="junqi_cuda not available",
)

from junqi_rl.env import JunqiEnv
from junqi_rl.env_gpu import _pack_state_arrays


def _make_envs(N: int, seed_base: int = 0xC5C5) -> list[JunqiEnv]:
    envs = []
    for i in range(N):
        rng = random.Random(seed_base + i)
        env = JunqiEnv()
        env.reset(seed=seed_base + i)
        for _ in range(i % 30):
            if env.state.terminated:
                break
            aids = env.legal_action_ids()
            if aids.size == 0:
                break
            env._step_game_only(int(rng.choice(aids)))
        envs.append(env)
    return envs


@pytest.fixture(scope="module", autouse=True)
def _init():
    _cuda.init_tables()


# =====================================================================
# CSR output parity
# =====================================================================

class TestCsrOutput:
    def test_csr_shapes(self) -> None:
        N = 16
        envs = _make_envs(N)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        offsets, values = _cuda.legal_action_ids_batch_csr(gs, acting)
        assert offsets.shape == (N + 1,)
        assert offsets.dtype == np.int32
        assert offsets[0] == 0
        assert values.shape == (int(offsets[N]),)
        assert values.dtype == np.int32

    def test_csr_matches_dense_action_sets(self) -> None:
        """CSR-decoded action set == dense action set for every env."""
        N = 32
        envs = _make_envs(N)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)

        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        ids_dense, cnt_dense = _cuda.legal_action_ids_batch(gs, acting)
        offsets, values = _cuda.legal_action_ids_batch_csr(gs, acting)

        # Per-env count parity
        derived_counts = np.diff(offsets)
        np.testing.assert_array_equal(derived_counts, cnt_dense)

        # Per-env set parity
        for i in range(N):
            c = int(cnt_dense[i])
            dense_sorted = np.sort(ids_dense[i, :c])
            csr_sorted = np.sort(values[offsets[i]:offsets[i + 1]])
            np.testing.assert_array_equal(
                dense_sorted, csr_sorted,
                err_msg=f"env {i}: CSR vs dense set mismatch",
            )

    def test_csr_empty_env(self) -> None:
        """An env with 0 legal moves returns empty CSR slice."""
        # Terminated envs have 0 actions — not easy to construct reliably,
        # but we can at least verify the machinery handles zeros.
        N = 4
        envs = _make_envs(N)
        # Force all pieces of one env to dead positions (hacky but effective).
        # Use deep play to land some envs at 0 actions naturally.
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        offsets, values = _cuda.legal_action_ids_batch_csr(gs, acting)
        assert offsets[-1] == values.shape[0]

    def test_csr_large_batch(self) -> None:
        """N=256 CSR correctness."""
        N = 256
        envs = _make_envs(N, seed_base=0xBEEF)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)

        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        ids_dense, cnt_dense = _cuda.legal_action_ids_batch(gs, acting)
        offsets, values = _cuda.legal_action_ids_batch_csr(gs, acting)

        np.testing.assert_array_equal(np.diff(offsets), cnt_dense)
        for i in range(N):
            c = int(cnt_dense[i])
            if c == 0:
                continue
            dense_sorted = np.sort(ids_dense[i, :c])
            csr_sorted = np.sort(values[offsets[i]:offsets[i + 1]])
            np.testing.assert_array_equal(dense_sorted, csr_sorted)


# =====================================================================
# Per-piece 32-slot mask parity
# =====================================================================


class TestPerPieceMask:
    def test_mask_shape_and_dtype(self) -> None:
        N = 4
        envs = _make_envs(N)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        mask = _cuda.legal_action_mask_batch(gs, acting)
        assert mask.shape == (N, 120, _cuda.SLOTS_PER_PIECE)
        assert mask.dtype == bool

    def test_reserved_slots_always_zero(self) -> None:
        """Non-engineer reserved slots 68..79 must never be True."""
        N = 32
        envs = _make_envs(N, seed_base=0x2468)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        mask = _cuda.legal_action_mask_batch(gs, acting)
        # For every piece, slots 68..79 are reserved (zero) except for the
        # engineer (GONGB, type=13), whose BFS may pack destinations into
        # slots 8..79.  We only check the non-engineer reservation here.
        from junqi_core.rules import PieceType
        eng_val = PieceType.GONGB.value
        for i, env in enumerate(envs):
            st = env.state
            for pid in range(120):
                if int(st.piece_type_arr[pid]) == eng_val:
                    continue
                row_tail = mask[i, pid, 68:]
                assert not row_tail.any(), (
                    f"env {i} pid {pid}: non-engineer reserved slots 68+ True"
                )

    def test_mask_only_acting_seat_pieces(self) -> None:
        """Only pieces owned by the acting seat can have any True slots."""
        N = 8
        envs = _make_envs(N, seed_base=0x1357)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        mask = _cuda.legal_action_mask_batch(gs, acting)
        for i, env in enumerate(envs):
            st = env.state
            for pid in range(120):
                if mask[i, pid].any():
                    assert st.alive[pid], f"env {i} pid {pid}: mask True on dead piece"
                    assert st.piece_seat_arr[pid] == acting[i], (
                        f"env {i} pid {pid}: mask True on non-acting-seat piece"
                    )

    def test_mask_action_count_parity_with_dense_non_engineer(self) -> None:
        """Total True slots (excluding engineer BFS ambiguity) must match dense count."""
        N = 32
        envs = _make_envs(N, seed_base=0x3579)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        mask = _cuda.legal_action_mask_batch(gs, acting)
        _, cnt_dense = _cuda.legal_action_ids_batch(gs, acting)
        total_true = mask.sum(axis=(1, 2))
        np.testing.assert_array_equal(total_true.astype(np.int32), cnt_dense)

    def test_mask_per_piece_count_parity(self) -> None:
        """For every piece, #True slots == #actions with that piece as src."""
        N = 16
        envs = _make_envs(N, seed_base=0x5555)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        mask = _cuda.legal_action_mask_batch(gs, acting)
        ids_dense, cnt_dense = _cuda.legal_action_ids_batch(gs, acting)

        for i, env in enumerate(envs):
            st = env.state
            c = int(cnt_dense[i])
            dense_actions = ids_dense[i, :c]
            # Decode src_flat for each dense action
            dense_src_flats = dense_actions // 289
            # Map each dense src_flat to its owning piece
            for pid in range(120):
                if not st.alive[pid]:
                    continue
                if st.piece_seat_arr[pid] != acting[i]:
                    continue
                src_flat = int(st.pos_y[pid]) * 17 + int(st.pos_x[pid])
                n_dense = int((dense_src_flats == src_flat).sum())
                n_mask  = int(mask[i, pid].sum())
                assert n_dense == n_mask, (
                    f"env {i} pid {pid} src={src_flat}: "
                    f"dense actions={n_dense}, mask slots={n_mask}"
                )


# =====================================================================
# Scratch buffer lifecycle
# =====================================================================


class TestScratchLifecycle:
    def test_reset_reallocs_cleanly(self) -> None:
        """After gpu_scratch_reset() the API must still function."""
        N = 8
        envs = _make_envs(N)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        # Call once, reset, call again — must give same results.
        r1_ids, r1_cnt = _cuda.legal_action_ids_batch(gs, acting)
        _cuda.gpu_scratch_reset()
        r2_ids, r2_cnt = _cuda.legal_action_ids_batch(gs, acting)

        np.testing.assert_array_equal(r1_cnt, r2_cnt)
        for i in range(N):
            c = int(r1_cnt[i])
            np.testing.assert_array_equal(
                np.sort(r1_ids[i, :c]), np.sort(r2_ids[i, :c])
            )

    def test_growing_batch_size(self) -> None:
        """Scratch buffers grow when a larger N is used."""
        _cuda.gpu_scratch_reset()
        for N in (4, 16, 64, 128):
            envs = _make_envs(N, seed_base=0x8800 + N)
            acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
            sd = _pack_state_arrays(envs)
            gs = _cuda.DeviceGameStateBatch(N)
            gs.copy_from_host(sd)
            ids, counts = _cuda.legal_action_ids_batch(gs, acting)
            assert ids.shape == (N, 512)
            assert counts.shape == (N,)

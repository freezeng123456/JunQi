"""Tests for junqi_rl.belief.inference.

``refresh_beliefs_neural`` needs a real GpuRollout to test end-to-end, so
we only test the pure-function helper here. The end-to-end test lives
with the integration smoke in P1.6.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from junqi_rl.belief.inference import (
    NUM_CELLS,
    N_SEATS,
    _get_enemy_mask,
    belief_logits_to_upload_shape,
)
from junqi_rl.belief.reveal_tracker import _SEAT_CELLS, _build_enemy_mask
from junqi_rl.networks.belief_net import N_BELIEF_TYPES


# ---------------------------------------------------------------------------
# _get_enemy_mask cache
# ---------------------------------------------------------------------------


def test_enemy_mask_tensor_shape_and_dtype():
    mask = _get_enemy_mask(torch.device("cpu"))
    assert mask.shape == (N_SEATS, NUM_CELLS)
    assert mask.dtype == torch.bool


def test_enemy_mask_tensor_matches_numpy_version():
    """The cached tensor must encode the same masks as the numpy source."""
    mask_t = _get_enemy_mask(torch.device("cpu"))
    for seat in range(4):
        expected = _build_enemy_mask(seat)
        got = mask_t[seat].numpy()
        assert np.array_equal(got, expected), f"seat {seat} mismatch"


def test_enemy_mask_tensor_each_row_has_50_true():
    """Each of the 4 rows corresponds to 2 enemy seats × 25 cells = 50."""
    mask = _get_enemy_mask(torch.device("cpu"))
    counts = mask.sum(dim=-1)
    assert torch.equal(counts, torch.full((N_SEATS,), 50))


# ---------------------------------------------------------------------------
# belief_logits_to_upload_shape
# ---------------------------------------------------------------------------


def test_upload_shape_basic():
    """(N, 4, 289, 12) probs → (N, 4, 12, 289) float32 upload layout."""
    N = 3
    probs = torch.rand(N, N_SEATS, NUM_CELLS, N_BELIEF_TYPES)
    # Normalise so each cell's 12-dim sums to 1 (like a real softmax).
    probs = probs / probs.sum(dim=-1, keepdim=True)

    up = belief_logits_to_upload_shape(probs)
    assert up.shape == (N, N_SEATS, N_BELIEF_TYPES, NUM_CELLS)
    assert up.dtype == torch.float32


def test_upload_shape_does_not_zero_non_enemy_cells_by_default():
    """BUG-O fix (2026-05-11): without an explicit enemy_mask argument,
    ``belief_logits_to_upload_shape`` must NOT zero non-enemy-territory
    cells.

    Rationale: the legacy behaviour zeroed cells based on the static
    initial-territory mask, which silently discarded BeliefNet outputs
    for any enemy piece that walked into the observer's own territory
    (~23% of alive enemies after ~500 random plies). The GPU obs kernel
    already gates ``d_belief`` reads on the per-cell piece_seat check,
    so leaving non-enemy cells' beliefs intact is harmless: those cells
    are never read.
    """
    N = 2
    probs = torch.rand(N, N_SEATS, NUM_CELLS, N_BELIEF_TYPES)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    up = belief_logits_to_upload_shape(probs)   # (N, 4, 12, 289), no mask

    # All cells should preserve the (transposed) probs — none zeroed.
    for env in range(N):
        for observer in range(N_SEATS):
            # Every cell's 12-way distribution must equal the input probs.
            expected = probs[env, observer]   # (289, 12)
            for c in range(NUM_CELLS):
                expected_dist = expected[c]   # (12,)
                got_dist = up[env, observer, :, c]
                assert torch.allclose(got_dist, expected_dist, atol=1e-6), (
                    f"env={env} observer={observer} cell={c} differs"
                )


def test_upload_shape_honours_explicit_enemy_mask():
    """When a caller explicitly passes an enemy_mask, non-enemy cells
    are zeroed. Backwards-compat guarantee for tests / custom callers."""
    N = 2
    probs = torch.rand(N, N_SEATS, NUM_CELLS, N_BELIEF_TYPES)
    probs = probs / probs.sum(dim=-1, keepdim=True)

    masks = np.stack([_build_enemy_mask(s) for s in range(N_SEATS)], axis=0)
    enemy_mask = torch.from_numpy(masks).to(torch.bool)

    up = belief_logits_to_upload_shape(probs, enemy_mask=enemy_mask)

    for env in range(N):
        for observer in range(N_SEATS):
            non_enemy_cells = (~enemy_mask[observer]).nonzero(as_tuple=True)[0]
            for c in non_enemy_cells:
                assert torch.all(up[env, observer, :, c] == 0), (
                    f"env={env} observer={observer} cell={c.item()} "
                    f"should be zeroed out under explicit enemy_mask"
                )


def test_upload_shape_enemy_cells_preserved():
    """Enemy cells must match the original probs (modulo transpose)."""
    N = 1
    probs = torch.rand(N, N_SEATS, NUM_CELLS, N_BELIEF_TYPES)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    up = belief_logits_to_upload_shape(probs)

    for observer in range(N_SEATS):
        enemy_mask = torch.from_numpy(_build_enemy_mask(observer))
        enemy_cells = enemy_mask.nonzero(as_tuple=True)[0]
        for c in enemy_cells:
            expected = probs[0, observer, c]           # (12,)
            got = up[0, observer, :, c]                # (12,)
            assert torch.allclose(expected, got, atol=1e-6)


def test_upload_shape_transpose_orientation():
    """Verify the transpose is (cell, type) → (type, cell) not the opposite."""
    # Construct probs where (env=0, seat=0, cell=c, type=t) = c * 100 + t.
    # This lets us spot-check individual entries after transpose.
    probs = torch.zeros(1, N_SEATS, NUM_CELLS, N_BELIEF_TYPES)
    for c in range(NUM_CELLS):
        for t in range(N_BELIEF_TYPES):
            probs[0, 0, c, t] = c * 100 + t

    up = belief_logits_to_upload_shape(probs)
    # For observer=0, enemies are seats 1 and 3 → cells in _SEAT_CELLS[1, 3].
    enemy_mask = torch.from_numpy(_build_enemy_mask(0))
    for c in range(NUM_CELLS):
        if not enemy_mask[c]:
            continue
        for t in range(N_BELIEF_TYPES):
            expected = c * 100 + t
            assert up[0, 0, t, c].item() == expected, (
                f"cell {c} type {t}: expected {expected}, got {up[0, 0, t, c].item()}"
            )


def test_upload_shape_rejects_bad_input():
    with pytest.raises(ValueError, match="probs must be"):
        belief_logits_to_upload_shape(torch.zeros(2, 4, 128, 12))   # wrong cell count
    with pytest.raises(ValueError, match="probs must be"):
        belief_logits_to_upload_shape(torch.zeros(2, 4, 289, 10))   # wrong type count
    with pytest.raises(ValueError, match="4 seats"):
        belief_logits_to_upload_shape(torch.zeros(2, 3, 289, 12))   # 3 seats


def test_upload_shape_dtype_coerces_to_float32():
    """Input in fp16/fp64 must come out as fp32 (GpuRollout.upload_beliefs
    contract)."""
    for dtype in (torch.float16, torch.float64):
        probs = torch.rand(1, 4, 289, 12, dtype=dtype)
        up = belief_logits_to_upload_shape(probs)
        assert up.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_upload_shape_on_cuda():
    probs = torch.rand(2, 4, 289, 12, device="cuda")
    up = belief_logits_to_upload_shape(probs)
    assert up.device.type == "cuda"
    assert up.shape == (2, 4, 12, 289)


# ---------------------------------------------------------------------------
# refresh_beliefs_neural — chunked forward parity
# ---------------------------------------------------------------------------


class _FakeRollout:
    """Tiny stand-in for GpuRollout that satisfies refresh_beliefs_neural's
    contract on CPU: builds a random (N, 4, C, H, W) obs tensor and captures
    the upload_beliefs argument so tests can compare against a reference run.
    """

    def __init__(self, num_envs: int, obs_channels: int | None = None, device: str = "cpu"):
        if obs_channels is None:
            from junqi_core.observation import OBS_CHANNELS
            obs_channels = OBS_CHANNELS
        self.num_envs = num_envs
        self._device = torch.device(device)
        # Use a small batch of reproducible random obs.
        torch.manual_seed(0)
        self._obs = torch.randn(
            num_envs, N_SEATS, obs_channels, 17, 17, device=self._device
        )
        self.last_upload: np.ndarray | None = None

    def build_all_seat_observations_torch(self):
        # (obs_spatial, obs_global); inference.py discards the global one.
        return self._obs, None

    def upload_beliefs(self, arr: np.ndarray) -> None:
        self.last_upload = arr.copy()


def _run_refresh_and_capture(num_envs: int, chunk_size: int) -> np.ndarray:
    """Build a deterministic BeliefNet + FakeRollout and capture the
    upload payload for a given chunk_size."""
    from junqi_rl.belief.inference import refresh_beliefs_neural
    from junqi_rl.networks.belief_net import BeliefNet, BeliefNetConfig

    torch.manual_seed(42)
    # Tiny config for fast CPU test (we still exercise the batching path).
    net = BeliefNet(BeliefNetConfig(
        n_encoder_layer=1, n_head=2, embed_dim=32, ff_factor=2,
        cnn_channels=32, cnn_layers=1,
    ))
    net.eval()
    ro = _FakeRollout(num_envs=num_envs, device="cpu")
    refresh_beliefs_neural(
        rollout=ro,
        belief_net=net,
        empty_cache=False,        # avoid sync cost / no-op on CPU anyway
        chunk_size=chunk_size,
    )
    assert ro.last_upload is not None
    return ro.last_upload


def _assert_chunked_probabilities_match(
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    """Compare probabilities across batch shapes with FP32-safe tolerances.

    CPU BLAS and CUDA kernels may select different matrix-multiplication
    implementations for different batch sizes. Their FP32 accumulation order
    can therefore differ by a few parts in 100,000 without changing model
    semantics.
    """
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.sum(axis=2), 1.0, rtol=5e-5, atol=5e-6)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-6)


def test_refresh_beliefs_neural_chunked_matches_unchunked():
    """Chunking must remain numerically equivalent to an unchunked forward."""
    N = 8
    upload_chunked = _run_refresh_and_capture(num_envs=N, chunk_size=8)
    upload_unchunked = _run_refresh_and_capture(num_envs=N, chunk_size=0)  # 0 → no chunking
    _assert_chunked_probabilities_match(upload_chunked, upload_unchunked)


def test_refresh_beliefs_neural_chunk_size_zero_is_unchunked():
    """chunk_size=0 and chunk_size>=4N should both run a single forward."""
    N = 4
    up0 = _run_refresh_and_capture(num_envs=N, chunk_size=0)
    up_full = _run_refresh_and_capture(num_envs=N, chunk_size=N * N_SEATS)
    _assert_chunked_probabilities_match(up0, up_full)


def test_refresh_beliefs_neural_odd_chunk_size():
    """Non-divisor chunk sizes (e.g. chunk=5 for 4N=32) must still produce
    the same output — last chunk just has fewer rows."""
    N = 8
    up_odd = _run_refresh_and_capture(num_envs=N, chunk_size=5)
    up_full = _run_refresh_and_capture(num_envs=N, chunk_size=0)
    _assert_chunked_probabilities_match(up_odd, up_full)

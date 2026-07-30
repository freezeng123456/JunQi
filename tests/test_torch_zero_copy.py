"""Verify Torch consumes CUDA observation pointers without a host copy."""

import numpy as np
import pytest
import torch

from junqi_core.observation import OBS_CHANNELS
from junqi_rl.gpu_rollout import GpuRollout

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA GPU not available",
    ),
]


def test_zero_copy_spatial():
    N = 4
    r = GpuRollout(num_envs=N)
    r.reset(seed_base=0)
    # Build obs on device once
    r.build_all_seat_observations()

    class _CudaView:
        def __init__(self, ptr, shape):
            self.__cuda_array_interface__ = {
                "shape": tuple(shape),
                "typestr": "<f4",
                "data": (int(ptr), False),
                "version": 2,
            }

    sp_view = _CudaView(r.obs.d_spatial_ptr, (N, 4, OBS_CHANNELS, 17, 17))
    gl_view = _CudaView(r.obs.d_global_ptr,  (N, 4, 28))

    sp_t = torch.as_tensor(sp_view, device="cuda")
    gl_t = torch.as_tensor(gl_view, device="cuda")

    assert sp_t.shape == (N, 4, OBS_CHANNELS, 17, 17)
    assert gl_t.shape == (N, 4, 28)
    assert sp_t.dtype == torch.float32
    assert sp_t.device.type == "cuda"

    # Parity with host copy path
    sp_host, gl_host = r.obs.copy_to_host()
    assert np.allclose(sp_t.cpu().numpy(), sp_host)
    assert np.allclose(gl_t.cpu().numpy(), gl_host)

    # Zero-copy: modify via torch, see via host copy
    sp_t[0, 0, 0, 0, 0] = 42.0
    torch.cuda.synchronize()
    sp_host2, _ = r.obs.copy_to_host()
    assert sp_host2[0, 0, 0, 0, 0] == 42.0, (
        "writing via torch tensor did not land in device buffer — not zero-copy!"
    )


def test_build_all_seat_observations_torch():
    """The new GpuRollout.build_all_seat_observations_torch helper matches
    the host-copy path and is on CUDA."""
    N = 8
    r = GpuRollout(num_envs=N)
    r.reset(seed_base=123)
    sp_t, gl_t = r.build_all_seat_observations_torch()
    assert sp_t.device.type == "cuda"
    assert gl_t.device.type == "cuda"
    assert sp_t.shape == (N, 4, OBS_CHANNELS, 17, 17)
    assert gl_t.shape == (N, 4, 28)
    sp_host, gl_host = r.obs.copy_to_host()
    assert np.allclose(sp_t.cpu().numpy(), sp_host)
    assert np.allclose(gl_t.cpu().numpy(), gl_host)

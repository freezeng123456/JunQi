"""Target-GPU gates for known-only observations and probability-free history."""
import random
import numpy as np
import pytest
import torch
cuda = pytest.importorskip('junqi_cuda')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
from junqi_core.observation import ObservationBuilder, CHANNEL_LAYOUT
from junqi_core.rules import ShowMode, Seat, PieceType
from junqi_core.state import Action
from junqi_rl.env import JunqiEnv
from junqi_rl.env_gpu import _pack_state_arrays
from junqi_rl.gpu_rollout import GpuRollout
from tests.test_gpu_obs_parity import _pack_combat_memory, _advance_env
from tests.test_feature_information_boundaries import position

@pytest.mark.parametrize('mode', list(ShowMode))
def test_known_only_cpu_cuda_parity_with_public_reveals(mode):
    cuda.init_tables()
    envs = []
    for i in range(8):
        env = JunqiEnv(show_mode=mode, include_global=False, identity_encoding='known_only')
        env.reset(seed=20261002+i)
        _advance_env(env, i*20, random.Random(i))
        envs.append(env)
    # Public mine identity and flag-reveal branch; original owner alone knows types.
    st = position(PieceType.SILING, PieceType.DILEI)
    st, _ = st.step(Action(Seat.SOUTH, (2, 7), (1, 7)))
    st.show_mode = mode
    envs[0]._state = st
    envs[1].state.combat_memory.is_gongb[:, 37] = True
    envs[2].state._set_flag_revealed(Seat.WEST)
    state = cuda.DeviceGameStateBatch(len(envs))
    state.copy_from_host(_pack_state_arrays(envs))
    state.cm_copy_from_host(*_pack_combat_memory(envs))
    cuda.upload_state_observer_seats(state, np.tile(np.arange(4, dtype=np.int8), (len(envs), 1)))
    out = cuda.DeviceObservationBatch(len(envs), False)
    cuda.build_observation_batch_resident(state, out, np.int8(mode.value), True)
    actual, gl = out.copy_to_host()
    builder = ObservationBuilder(include_global=False, identity_encoding='known_only')
    expected = np.stack([np.stack([builder.build(e.state, None, s).spatial.copy()
                                  for s in Seat]) for e in envs])
    mask = np.ones(317, dtype=bool)
    # Host upload has no move ring; native history equivalence tested below.
    mask[CHANNEL_LAYOUT['move_history']] = False
    np.testing.assert_allclose(actual[:, :, mask], expected[:, :, mask], rtol=0, atol=1e-6)
    assert gl.shape[-1] == 0 and not state.has_belief_buffers

def test_no_rule_kernels_buffers_or_history_even_after_legacy_use(monkeypatch):
    legacy = GpuRollout(2, include_global=False)
    legacy.reset(seed_base=800)
    old_history = legacy.create_rollout_history(32)
    def forbidden(*args, **kwargs):
        raise AssertionError('facts path touched a rule probability operation')
    for name in ('upload_state_beliefs', 'upload_belief_prior_table', 'init_all_beliefs',
                 'init_beliefs_for_reset_envs', 'update_beliefs_after_step', 'snapshot_pre_step_flags'):
        if hasattr(cuda, name): monkeypatch.setattr(cuda, name, forbidden)
    env = GpuRollout(2, include_global=False, identity_encoding='known_only', max_num_moves=8)
    env.reset(seed_base=800)
    history = env.create_rollout_history(32)
    assert old_history.history_bytes - history.history_bytes == 32*2*12*289*4
    assert not history._impl.store_beliefs
    spatial_rows, masks, seats = [], [], []
    for step in range(32):
        acting = env.turn_torch().clone()
        sp, gl = env.build_acting_seat_observation_torch(acting)
        legal = env.legal_mask_canonical_torch_device(acting).clone()
        spatial_rows.append(sp.clone()); masks.append(legal); seats.append(acting)
        history.snapshot(env.state, acting, step)
        result = env.step_device_torch(legal.to(torch.int8).argmax(-1).to(torch.int32), acting)
        env.update_beliefs_device(result, acting)
        env.reset_terminated_device(seed=9000+step)
        assert not env.state.has_belief_buffers
    indices = torch.arange(64, device='cuda', dtype=torch.int64).flip(0)
    actors = torch.stack(seats).flatten()[indices]
    sp, gl, mask = history.reconstruct(indices, actors, dtype=torch.float32)
    torch.testing.assert_close(sp, torch.stack(spatial_rows).flatten(0, 1)[indices], rtol=0, atol=0)
    torch.testing.assert_close(mask, torch.stack(masks).flatten(0, 1)[indices], rtol=0, atol=0)
    assert gl.shape == (64, 0)
    with pytest.raises(RuntimeError, match='no rule'):
        env.rule_beliefs_torch()

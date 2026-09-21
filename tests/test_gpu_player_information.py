"""Native stepping, observer rotation, and neural refresh of public flags."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cuda = pytest.importorskip("junqi_cuda")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from junqi_core.batched_state import BatchedGameState
from junqi_core.board import FLAT_TO_COMPACT
from junqi_core.info_model import BeliefTensor
from junqi_core.observation import CHANNEL_LAYOUT, ObservationBuilder
from junqi_core.rotation import world_to_canonical
from junqi_core.rules import Seat
from junqi_rl.belief.inference import refresh_beliefs_neural
from junqi_rl.gpu_rollout import GpuRollout
from tests.test_belief_refresh_constraints import ConstantNet
from tests.test_gpu_combat_memory_parity import _push_state
from tests.test_player_information_integration import commander_position


@pytest.mark.parametrize("owner", list(Seat))
def test_native_public_flag_matches_board_and_survives_neural_refresh(owner):
    state, action, flag_pos = commander_position(owner)
    world = GpuRollout(num_envs=1)
    world.state = _push_state(BatchedGameState.from_game_states([state]))
    beliefs = {seat: BeliefTensor.initial(state, seat) for seat in Seat}
    prior = np.zeros((1, 4, 12, 289), dtype=np.float32)
    for observer, belief in beliefs.items():
        for (x, y), vector in belief.probs.items():
            prior[0, observer.value, :, y * 17 + x] = vector
    world.upload_beliefs(prior)
    world.rule_beliefs_torch()
    sx, sy = world_to_canonical(*action.src, owner)
    dx, dy = world_to_canonical(*action.dst, owner)
    aid = int(FLAT_TO_COMPACT[sy * 17 + sx]) * 129 + int(FLAT_TO_COMPACT[dy * 17 + dx])
    acting = torch.tensor([owner.value], device="cuda", dtype=torch.int8)
    result = world.step_device_torch(torch.tensor([aid], device="cuda", dtype=torch.int32), acting)
    world.update_beliefs_device(result, acting)
    after, cpu_result = state.step(action)
    for belief in beliefs.values():
        belief.update(state, after, cpu_result)
    actual_sp, actual_gl = world.build_all_seat_observations()
    for observer in Seat:
        expected = ObservationBuilder().build(after, beliefs[observer], observer)
        np.testing.assert_allclose(actual_sp[0, observer.value], expected.spatial, atol=1e-6)
        np.testing.assert_allclose(actual_gl[0, observer.value], expected.global_, atol=1e-5)

    # A deliberately wrong, highly confident neural prediction cannot hide a flag.
    refresh_beliefs_neural(world, ConstantNet(11).cuda(), empty_cache=False)
    x, y = flag_pos
    assert np.all(world._beliefs[0, :, 0, y * 17 + x] == 1.0)
    for pos, piece in after.pieces.items():
        if piece.seat is owner and pos != flag_pos:
            assert np.all(world._beliefs[0, :, 0, pos[1] * 17 + pos[0]] == 0.0)

    # Native initialization also accepts a restored mid-game state with an open flag.
    world.state = _push_state(BatchedGameState.from_game_states([after]))
    world.upload_beliefs(np.zeros_like(prior))
    cuda.init_all_beliefs(world.state)
    assert torch.all(world.rule_beliefs_torch()[0, :, 0, y * 17 + x] == 1.0)
    world.reset(seed_base=71)
    sp, _ = world.build_all_seat_observations()
    assert not sp[:, :, CHANNEL_LAYOUT["flag_revealed"]].any()

# ruff: noqa: E402
"""Neural refresh may reweight uncertainty, never rewrite rule knowledge."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from junqi_core.info_model import TRACKED_TYPES, BeliefTensor
from junqi_core.observation import ObservationBuilder
from junqi_core.rules import PieceType, Seat
from junqi_core.state import Action
from junqi_rl.belief.inference import refresh_beliefs_neural
from tests.test_feature_information_boundaries import position


class CaptureRollout:
    num_envs = 1

    def __init__(self, public_mine=True):
        st = position(PieceType.SILING if public_mine else PieceType.PAIZH, PieceType.DILEI)
        beliefs = {s: BeliefTensor.initial(st, s) for s in Seat}
        if public_mine:
            nxt, result = st.step(Action(Seat.SOUTH, (2, 7), (1, 7)))
            for b in beliefs.values():
                b.update(st, nxt, result)
            st = nxt
        self.observations = [ObservationBuilder().build(st, beliefs[s], s).snapshot() for s in Seat]
        self.rules = torch.zeros(1, 4, 12, 289)
        for seat in Seat:
            for x, y in st.pieces:
                self.rules[0, seat.value, :, y * 17 + x] = torch.from_numpy(beliefs[seat].get((x, y)))
        self.uploaded = None

    def build_all_seat_observations_torch(self):
        return torch.from_numpy(np.stack([o.spatial for o in self.observations])[None]), None

    def rule_beliefs_torch(self):
        return self.rules

    def upload_beliefs(self, payload):
        self.uploaded = payload.copy()


class ConstantNet(torch.nn.Module):
    def __init__(self, favored=None):
        super().__init__()
        self.favored = favored

    def forward(self, x, seat_idx):
        logits = torch.zeros(len(x), 289, 12, device=x.device)
        if self.favored is not None:
            logits[..., self.favored] = 1000
        return {"logits": logits}


def test_refresh_preserves_public_mine_and_all_rule_exclusions():
    ro = CaptureRollout()
    refresh_beliefs_neural(ro, ConstantNet(), empty_cache=False)
    mine = TRACKED_TYPES.index(PieceType.DILEI)
    assert ro.uploaded[0, Seat.SOUTH.value, mine, 7*17+1] == 1
    rules = ro.rules.numpy()
    assert np.count_nonzero(ro.uploaded[rules == 0]) == 0
    known = (rules > 0).sum(axis=2) == 1
    np.testing.assert_array_equal(ro.uploaded.transpose(0, 1, 3, 2)[known],
                                  rules.transpose(0, 1, 3, 2)[known])
    # Empty and dead cells stay empty, not uniform distributions.
    assert np.count_nonzero(ro.uploaded.sum(axis=2)[rules.sum(axis=2) == 0]) == 0


def test_neural_certainty_does_not_become_a_new_hard_fact():
    ro = CaptureRollout(public_mine=False)
    target = 7 * 17 + 1
    prior = ro.rules[0, Seat.SOUTH.value, :, target].numpy()
    support = np.flatnonzero(prior > 0)
    assert len(support) > 1
    refresh_beliefs_neural(ro, ConstantNet(int(support[0])), empty_cache=False)
    assert ro.uploaded[0, 0, support[0], target] == 1
    refresh_beliefs_neural(ro, ConstantNet(), empty_cache=False)
    actual = ro.uploaded[0, 0, :, target]
    np.testing.assert_allclose(actual[support], 1 / len(support), rtol=2e-7)
    assert np.count_nonzero(actual) == len(support)


def test_zero_neural_mass_on_allowed_types_falls_back_to_rules():
    ro = CaptureRollout(public_mine=False)
    # Force all softmax mass onto flag even where the slot rules prohibit it.
    flag = TRACKED_TYPES.index(PieceType.JUNQI)
    refresh_beliefs_neural(ro, ConstantNet(flag), empty_cache=False)
    target = 7 * 17 + 1
    assert ro.rules[0, 0, flag, target] == 0
    np.testing.assert_array_equal(ro.uploaded[0, 0, :, target], ro.rules[0, 0, :, target].numpy())


def test_live_enemy_mask_uses_world_coordinates_for_every_observer():
    from junqi_core.observation import CHANNEL_LAYOUT
    from junqi_core.rotation import world_to_canonical
    from junqi_rl.belief.inference import _live_enemy_world_mask
    obs = torch.zeros(2, 4, 317, 17, 17)
    expected = torch.zeros(2, 4, 289, dtype=torch.bool)
    for seat in Seat:
        for env, (x, y) in enumerate(((1, 7), (7, 13))):
            cx, cy = world_to_canonical(x, y, seat)
            obs[env, seat.value, CHANNEL_LAYOUT['piece_left_side_enemy'], cy, cx] = 1
            expected[env, seat.value, y*17+x] = True
    torch.testing.assert_close(_live_enemy_world_mask(obs), expected)


def test_nonfinite_predictions_fall_back_without_losing_rules():
    from junqi_rl.belief.inference import constrain_neural_beliefs
    rules = torch.zeros(1, 4, 12, 289)
    rules[:, :, 3:5, 1] = 0.5
    probs = torch.full((1, 4, 289, 12), float('nan'))
    result = constrain_neural_beliefs(probs, rules, torch.ones(1, 4, 289, dtype=torch.bool))
    torch.testing.assert_close(result, rules)

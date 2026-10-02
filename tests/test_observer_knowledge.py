"""Observer history is factual, persistent, and independent of hidden labels."""
from dataclasses import fields, replace
import random

import numpy as np
import pytest

from junqi_core.identity import TRACKED_TYPES, TYPE_TO_INDEX
from junqi_core.knowledge import ObserverKnowledge
from junqi_core.rules import Event, PieceType, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import Action, GameState
from junqi_rl.belief.joint import build_joint_input
from junqi_rl.env import JunqiEnv
from tests.test_feature_information_boundaries import position, public_mine_sequence


def opening(seed=7):
    return GameState.new_game(generate_random_setup(random.Random(seed)),
                              show_mode=ShowMode.DARK)


def facts_env(**kwargs):
    return JunqiEnv(show_mode=ShowMode.DARK, include_global=False,
                    identity_encoding='known_only', **kwargs)


def assert_inputs_equal(left, right):
    for field in fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def test_identity_input_never_reads_hidden_opening_labels():
    state = opening()
    original = build_joint_input([ObserverKnowledge(state, Seat.SOUTH).snapshot()])
    hidden = (state.piece_seat_arr >= 0) & (state.piece_seat_arr != Seat.SOUTH)
    state.piece_type_arr[hidden] = -127  # Any hidden read would be an invalid enum.
    poisoned = build_joint_input([ObserverKnowledge(state, Seat.SOUTH).snapshot()])
    assert_inputs_equal(original, poisoned)
    assert original.piece_ids.shape == (1, 75)
    assert (original.known_types == -1).all()  # DARK also hides the teammate.
    assert original.type_domains.dtype == np.bool_
    np.testing.assert_array_equal(original.initial_inventory.sum(-1), [[0, 25, 25, 25]])


def test_same_public_combat_does_not_assign_average_rank_probabilities():
    inputs = []
    for kind in (PieceType.SHIZH, PieceType.JUNZH, PieceType.SILING):
        before = position(kind, PieceType.LVZH)
        knowledge = ObserverKnowledge(before, Seat.WEST)
        after, result = before.step(Action(Seat.SOUTH, (2, 7), (1, 7)))
        # Debug fields are never observer input, even if populated.
        result = replace(result, src_type_revealed=kind,
                         dst_type_revealed=PieceType.LVZH)
        knowledge.observe(before, after, result)
        snapshot = knowledge.snapshot()
        inputs.append(snapshot)
        pid = before.pieces[(2, 7)].piece_id
        assert snapshot.known_types[pid] == -1
        allowed = [TRACKED_TYPES[i] for i in np.flatnonzero(snapshot.type_domains[pid])]
        assert allowed == [PieceType.SILING, PieceType.JUNZH, PieceType.SHIZH]
        event = snapshot.events[0]
        assert event.allowed_type_pairs()[TYPE_TO_INDEX[kind], TYPE_TO_INDEX[PieceType.LVZH]]
    for snapshot in inputs[1:]:
        assert_inputs_equal(inputs[0], snapshot)


def test_public_commander_mine_and_engineer_facts_persist_after_death():
    state, actions = public_mine_sequence()
    knowledge = ObserverKnowledge(state, Seat.EAST)
    commander = state.pieces[(2, 7)].piece_id
    mine = state.pieces[(1, 7)].piece_id
    engineer = state.pieces[(1, 6)].piece_id
    for seat, src, dst in actions:
        after, result = state.step(Action(seat, src, dst))
        knowledge.observe(state, after, result)
        state = after
    assert not knowledge.alive[commander] and not knowledge.alive[mine]
    assert knowledge.known_types[commander] == TYPE_TO_INDEX[PieceType.SILING]
    assert knowledge.known_types[mine] == TYPE_TO_INDEX[PieceType.DILEI]
    assert knowledge.known_types[engineer] == TYPE_TO_INDEX[PieceType.GONGB]
    assert knowledge.events[-1].engineer_revealed
    assert len(knowledge.events) == 3


def test_unknown_dead_identity_remains_unknown_and_retained():
    before = position(PieceType.SHIZH, PieceType.LVZH)
    knowledge = ObserverKnowledge(before, Seat.EAST)
    victim = before.pieces[(1, 7)].piece_id
    after, result = before.step(Action(Seat.SOUTH, (2, 7), (1, 7)))
    knowledge.observe(before, after, result)
    assert not knowledge.alive[victim]
    assert knowledge.known_types[victim] == -1
    assert knowledge.events[-1].defender_pid == victim
    # An incomplete fixture cannot masquerade as a complete army.
    with pytest.raises(ValueError, match='complete standard opening'):
        build_joint_input([knowledge.snapshot()])


def test_revealed_engineer_identity_is_not_erased_by_death():
    before = position(PieceType.GONGB, PieceType.LVZH)
    engineer = before.pieces[(2, 7)].piece_id
    before.combat_memory.is_gongb[:, engineer] = True
    knowledge = ObserverKnowledge(before, Seat.EAST)
    after, result = before.step(Action(Seat.SOUTH, (2, 7), (1, 7)))
    knowledge.observe(before, after, result)
    snapshot = knowledge.snapshot()
    assert not snapshot.alive[engineer]
    assert snapshot.known_types[engineer] == TYPE_TO_INDEX[PieceType.GONGB]
    assert snapshot.type_domains[engineer].sum() == 1
    assert snapshot.events[-1].engineer_revealed


@pytest.mark.parametrize('seed', [7, 41, 20261002])
def test_real_history_constraints_keep_the_true_assignment_feasible(seed):
    state = opening(seed)
    knowledge = [ObserverKnowledge(state, s) for s in Seat]
    old = build_joint_input([k.snapshot() for k in knowledge])
    rng = np.random.default_rng(seed)
    saw_combat = False
    for _ in range(160):
        if state.terminated:
            break
        action_id = int(rng.choice(state.legal_action_ids(state.turn)))
        src, dst = divmod(action_id, 289)
        after, result = state.step(Action(state.turn, (src % 17, src // 17),
                                        (dst % 17, dst // 17)))
        for k in knowledge:
            k.observe(state, after, result)
        event = knowledge[0].events[-1]
        if event.has_combat:
            saw_combat = True
            attacker = TYPE_TO_INDEX[PieceType(int(state.piece_type_arr[event.attacker_pid]))]
            defender = TYPE_TO_INDEX[PieceType(int(state.piece_type_arr[event.defender_pid]))]
            assert event.allowed_type_pairs()[attacker, defender]
        state = after
        snapshots = [k.snapshot() for k in knowledge]
        inputs = build_joint_input(snapshots)
        for row, pids in enumerate(inputs.piece_ids):
            labels = [TYPE_TO_INDEX[PieceType(int(state.piece_type_arr[p]))] for p in pids]
            assert inputs.type_domains[row, np.arange(75), labels].all()
    assert saw_combat
    assert not inputs.alive.all()
    assert (inputs.cells[~inputs.alive] == -1).all()
    np.testing.assert_array_equal(inputs.initial_inventory, old.initial_inventory)
    assert old.public_history == ((), (), (), ())
    assert old.alive.all()  # The old snapshot was not mutated.
    with pytest.raises(ValueError):
        inputs.type_domains[0, 0, 0] = True


def test_recording_is_opt_in_and_reset_discards_previous_history():
    env = facts_env()
    env.reset(seed=1)
    assert env._knowledge is None
    with pytest.raises(RuntimeError, match='not enabled'):
        env.knowledge_snapshot(Seat.SOUTH)
    env = facts_env(record_public_history=True)
    env.reset(seed=1)
    env._step_game_only(int(env.legal_action_ids()[0]))
    assert len(env.knowledge_snapshot(Seat.SOUTH).events) == 1
    env.reset(seed=2)
    assert env.knowledge_snapshot(Seat.SOUTH).events == ()
    with pytest.raises(ValueError, match='requires'):
        JunqiEnv(record_public_history=True)


def test_missing_or_duplicate_history_is_rejected():
    before = position(PieceType.SHIZH, PieceType.LVZH)
    after, result = before.step(Action(Seat.SOUTH, (2, 7), (1, 7)))
    with pytest.raises(ValueError, match='complete public history'):
        ObserverKnowledge(after, Seat.SOUTH)
    knowledge = ObserverKnowledge(before, Seat.SOUTH)
    knowledge.observe(before, after, result)
    with pytest.raises(ValueError, match='missing, duplicated'):
        knowledge.observe(before, after, result)

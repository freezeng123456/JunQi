"""Completion masks have exact, public-only semantics on enumerable games."""
from dataclasses import replace
import itertools
import random

import numpy as np
import pytest

from junqi_core.identity import TYPE_TO_INDEX
from junqi_core.board import index_to_pos
from junqi_core.knowledge import ObserverKnowledge, PublicEvent
from junqi_core.move_gen import PieceRef
from junqi_core.rules import Event, PieceType, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import Action, GameState
from junqi_rl.belief.constraints import ConstraintBudgetExceeded, _immobility_domains, compile_constraints
from junqi_rl.belief.joint import IDENTITY_INPUT_SCHEMA, IdentityConstraintInput, build_joint_input


def event(left, right, outcome=Event.EAT):
    return PublicEvent(1, 1, 0, 1, left, right, outcome, False, False, False, False, ())


def toy_inputs(domains, owners=None, counts=None, events=(), known=None, own_known=None):
    n = len(domains)
    support = np.zeros((1, n, 12), dtype=bool)
    for i, allowed in enumerate(domains):
        support[0, i, allowed] = True
    owners = np.asarray(owners if owners is not None else [1] * n, dtype=np.int64)
    inventory = np.zeros((1, 4, 12), dtype=np.int64)
    for owner, labels in (counts or {1: list(range(n))}).items():
        for label in labels:
            inventory[0, owner, label] += 1
    observer_known = np.full((1, n + 1), -1, dtype=np.int64)
    if own_known is not None:
        observer_known[0, n] = own_known
    return IdentityConstraintInput(
        IDENTITY_INPUT_SCHEMA, np.array([0]), np.arange(n)[None],
        np.zeros((1, n), dtype=np.int64), np.zeros((1, n), dtype=np.int64),
        owners[None], np.ones((1, n), dtype=bool),
        np.asarray([known if known is not None else [-1] * n]), support, inventory,
        np.zeros((1, n, 3), dtype=np.int64), observer_known, (tuple(events),))


def enumerated(inputs):
    validator = compile_constraints(inputs)
    domains = [np.flatnonzero(d) for d in inputs.type_domains[0]]
    answers = []
    for labels in itertools.product(*domains):
        try:
            validator.validate(labels)
        except ValueError:
            continue
        answers.append(labels)
    return answers


def assert_all_prefix_masks(inputs):
    complete = enumerated(inputs)
    constraints = compile_constraints(inputs)
    assert complete
    for size in range(len(complete[0])):
        for prefix in {x[:size] for x in complete}:
            expected = {x[size] for x in complete if x[:size] == prefix}
            assert set(np.flatnonzero(constraints.allowed_next(prefix))) == expected
    for labels in complete:
        masks = constraints.teacher_masks(labels)
        assert masks[np.arange(len(labels)), labels].all()
    return constraints, complete


def test_exact_inventory_matching_rejects_a_hall_violation():
    # There are too many pieces restricted to {0,1}, although each individual
    # type has enough possible pieces and every domain is nonempty.
    inputs = toy_inputs([[0, 1], [0, 1], [0, 1], [0, 1, 2, 3]], counts={1: [0, 1, 2, 3]})
    constraints = compile_constraints(inputs)
    assert not constraints.allowed_next([]).any()


@pytest.mark.parametrize("outcome", [Event.EAT, Event.KILLED, Event.BOMB])
def test_public_combat_inventory_and_all_prefixes_match_enumeration(outcome):
    labels = [TYPE_TO_INDEX[t] for t in (PieceType.JUNZH, PieceType.SHIZH, PieceType.LVZH)]
    inputs = toy_inputs([labels] * 6, owners=[1, 1, 1, 2, 2, 2],
                        counts={1: labels, 2: labels}, events=[event(2, 5, outcome)])
    constraints, _ = assert_all_prefix_masks(inputs)
    assert constraints.stats['search_nodes'] > 0
    assert constraints.stats['cache_hits'] > 0


def test_known_dead_identity_and_own_combat_are_strict_constraints():
    labels = [TYPE_TO_INDEX[t] for t in (PieceType.JUNZH, PieceType.SHIZH, PieceType.LVZH)]
    inputs = toy_inputs([labels] * 3, counts={1: labels},
                        events=[event(0, 3)], own_known=labels[1], known=[-1, labels[2], -1])
    inputs = replace(inputs, alive=np.zeros((1, 3), dtype=bool))
    constraints, complete = assert_all_prefix_masks(inputs)
    assert complete == [tuple([labels[0], labels[2], labels[1]])]
    with pytest.raises(ValueError, match='known identity|combat'):
        constraints.validate([labels[1], labels[2], labels[0]])


def test_unknown_combat_endpoint_and_contradictory_known_fact_are_rejected():
    inputs = toy_inputs([[0, 1], [0, 1]], events=[event(0, 2)])
    with pytest.raises(ValueError, match='absent'):
        compile_constraints(inputs)
    with pytest.raises(ValueError, match='Contradictory'):
        compile_constraints(toy_inputs([[0], [1]], known=[1, -1]))


def test_resource_exhaustion_is_not_reported_as_infeasibility_or_cached():
    labels = [TYPE_TO_INDEX[t] for t in (PieceType.JUNZH, PieceType.SHIZH, PieceType.LVZH)]
    inputs = toy_inputs([labels] * 6, owners=[1, 1, 1, 2, 2, 2],
                        counts={1: labels, 2: labels}, events=[event(2, 5, Event.BOMB)])
    constrained = compile_constraints(inputs, max_search_nodes=1)
    with pytest.raises(ConstraintBudgetExceeded):
        constrained.allowed_next([])
    assert constrained.stats['budget_exhaustions'] == 1
    constrained.max_search_nodes = 1000
    np.testing.assert_array_equal(constrained.allowed_next([]), compile_constraints(inputs).allowed_next([]))


def test_neural_conditional_sampling_distribution_matches_enumeration():
    # The conditional model assigns unequal mass to complete valid layouts;
    # feasibility masks must not turn it into a uniform assignment strategy.
    inputs = toy_inputs([[0, 1], [0, 1, 2], [0, 1, 2]])
    constraints, assignments = assert_all_prefix_masks(inputs)
    logits = np.asarray([[0.9, -0.4, 0.1], [0.2, 0.8, -0.2], [-0.1, 0.5, 0.4]])
    target = {}
    for assignment in assignments:
        probability = 1.0
        for i, label in enumerate(assignment):
            mask = constraints.allowed_next(assignment[:i])[:3]
            probs = np.exp(logits[i]) * mask
            probs /= probs.sum()
            probability *= probs[label]
        target[assignment] = probability
    assert sum(target.values()) == pytest.approx(1.0)
    assert max(target.values()) - min(target.values()) > 0.1
    rng = np.random.default_rng(1703)
    counts = dict.fromkeys(assignments, 0)
    for _ in range(5000):
        prefix = []
        for row in logits:
            probs = np.exp(row) * constraints.allowed_next(prefix)[:3]
            probs /= probs.sum()
            prefix.append(int(rng.choice(3, p=probs)))
        counts[tuple(prefix)] += 1
    for assignment, probability in target.items():
        assert counts[assignment] / 5000 == pytest.approx(probability, abs=0.025)


def test_teacher_masks_depend_on_only_the_causal_prefix():
    inputs = toy_inputs([[0, 1, 2]] * 3)
    first = compile_constraints(inputs).teacher_masks([0, 1, 2])
    second = compile_constraints(inputs).teacher_masks([0, 2, 1])
    np.testing.assert_array_equal(first[:2], second[:2])
    assert not np.array_equal(first[2], second[2])


def test_no_move_elimination_uses_public_occupancy_and_movement_classes_only():
    stronghold = index_to_pos(Seat.SOUTH, 26)
    pieces = {(2, 7): PieceRef(Seat.SOUTH, PieceType.DARK, piece_id=0),
              stronghold: PieceRef(Seat.SOUTH, PieceType.DARK, piece_id=26),
              # An opaque opposing type would fail if movement read a blocker
              # identity rather than only its public presence and owner.
              (1, 7): PieceRef(Seat.WEST, object(), piece_id=30)}
    original = dict(pieces)
    domains = _immobility_domains(pieces, Seat.SOUTH)
    assert pieces == original
    expected = {TYPE_TO_INDEX[PieceType.JUNQI], TYPE_TO_INDEX[PieceType.DILEI]}
    assert set(np.flatnonzero(domains[0])) == expected
    assert domains[26].all()  # Every identity is immobile inside a stronghold.


def test_random_small_relations_agree_with_exhaustive_feasibility():
    rng = np.random.default_rng(7931)
    labels = [TYPE_TO_INDEX[t] for t in (PieceType.JUNZH, PieceType.SHIZH, PieceType.LVZH)]
    for _ in range(16):
        domains = [list(rng.choice(labels, size=int(rng.integers(1, 4)), replace=False))
                   for _ in range(6)]
        events = [event(int(rng.integers(0, 3)), int(rng.integers(3, 6)),
                        (Event.EAT, Event.KILLED, Event.BOMB)[int(rng.integers(3))])
                  for _ in range(2)]
        inputs = toy_inputs(domains, owners=[1, 1, 1, 2, 2, 2],
                            counts={1: labels, 2: labels}, events=events)
        complete = enumerated(inputs)
        constraints = compile_constraints(inputs)
        for prefix in [()] + [x[:3] for x in complete]:
            expected = {x[len(prefix)] for x in complete if x[:len(prefix)] == prefix}
            assert set(np.flatnonzero(constraints.allowed_next(prefix))) == expected


def test_real_75_identity_midgame_truth_and_generated_masks_pass_constraints():
    state = GameState.new_game(generate_random_setup(random.Random(1703)), show_mode=ShowMode.DARK)
    knowledge = ObserverKnowledge(state, Seat.SOUTH)
    rng = np.random.default_rng(1703)
    for _ in range(100):
        if state.terminated:
            break
        aid = int(rng.choice(state.legal_action_ids(state.turn)))
        src, dst = divmod(aid, 289)
        after, result = state.step(Action(state.turn, (src % 17, src // 17),
                                         (dst % 17, dst // 17)))
        knowledge.observe(state, after, result)
        state = after
    inputs = build_joint_input([knowledge.snapshot()])
    assert any(e.has_combat for e in inputs.public_history[0])
    truth = [TYPE_TO_INDEX[PieceType(int(state.piece_type_arr[pid]))] for pid in inputs.piece_ids[0]]
    constraints = compile_constraints(inputs)
    constraints.validate(truth)
    masks = constraints.teacher_masks(truth)
    assert masks[np.arange(75), truth].all()
    generated = []
    for _ in range(75):
        allowed = np.flatnonzero(constraints.allowed_next(generated))
        assert allowed.size
        generated.append(int(rng.choice(allowed)))
    constraints.validate(generated)
    # Hidden labels are used solely as this offline validator's targets.
    hidden = (state.piece_seat_arr >= 0) & (state.piece_seat_arr != Seat.SOUTH)
    state.piece_type_arr[hidden] = -127
    poisoned = build_joint_input([knowledge.snapshot()])
    np.testing.assert_array_equal(compile_constraints(poisoned).allowed_next([]), masks[0])


@pytest.mark.parametrize('prefix', [[True], [0.5], [-1], [12], [[0]]])
def test_invalid_causal_labels_are_rejected(prefix):
    with pytest.raises(ValueError):
        compile_constraints(toy_inputs([[0, 1], [0, 1]])).allowed_next(prefix)

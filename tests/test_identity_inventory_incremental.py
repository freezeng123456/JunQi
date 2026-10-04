"""Incremental inventory supports preserve deterministic exact search traces."""
import itertools

import numpy as np
import pytest

from junqi_core.identity import TYPE_TO_INDEX
from junqi_core.rules import Event, PieceType
from junqi_rl.belief import constraints as module
from junqi_rl.belief.constraints import ConstraintBudgetExceeded, compile_constraints
from junqi_rl.belief.joint import build_joint_input
from tests.test_identity_constraints import enumerated, event, toy_inputs


def outcome(solver, prefix):
    try:
        return solver.allowed_next(prefix).tolist()
    except ConstraintBudgetExceeded as error:
        return type(error).__name__, str(error)


def compare(reference, candidate, prefix):
    assert outcome(reference, prefix) == outcome(candidate, prefix)
    assert reference.stats == candidate.stats
    assert reference._cache == candidate._cache
    assert reference.matching_cache_stats == candidate.matching_cache_stats
    assert reference._matching_cache == candidate._matching_cache
    assert set(reference._masks) == set(candidate._masks)
    for key in reference._masks:
        np.testing.assert_array_equal(reference._masks[key], candidate._masks[key])


@pytest.mark.parametrize('budget', [1, 2, 5, 20, 1000])
@pytest.mark.parametrize('result', [Event.EAT, Event.KILLED, Event.BOMB])
def test_all_toy_prefixes_budget_failure_retries_and_cache_evictions(budget, result, monkeypatch):
    monkeypatch.setattr(module, '_MATCHING_CACHE_LIMIT', 3)
    labels = [TYPE_TO_INDEX[t] for t in (PieceType.JUNZH, PieceType.SHIZH, PieceType.LVZH)]
    inputs = toy_inputs([labels] * 6, owners=[1, 1, 1, 2, 2, 2],
                        counts={1: labels, 2: labels}, events=[event(2, 5, result)])
    reference = compile_constraints(inputs, max_search_nodes=budget)
    candidate = compile_constraints(inputs, max_search_nodes=budget, inventory_backend='incremental')
    prefixes = [()] + list(itertools.product(labels, repeat=2))
    for prefix in prefixes:
        compare(reference, candidate, prefix)
    reference.max_search_nodes = candidate.max_search_nodes = 1000
    answers = enumerated(inputs)
    for prefix in sorted({x[:size] for x in answers for size in range(6)}):
        compare(reference, candidate, prefix)
        expected = {x[len(prefix)] for x in answers if x[:len(prefix)] == prefix}
        assert set(np.flatnonzero(candidate.allowed_next(prefix))) == expected
        reference.allowed_next(prefix)  # Keep query and cache-hit accounting aligned.
    assert candidate.matching_cache_stats['evictions'] > 0


def test_random_domain_propagation_has_identical_mutations_even_when_infeasible():
    rng = np.random.default_rng(20261004)
    labels = [TYPE_TO_INDEX[t] for t in (PieceType.JUNZH, PieceType.SHIZH, PieceType.LVZH)]
    for _ in range(80):
        domains = [list(rng.choice(labels, int(rng.integers(1, 4)), replace=False)) for _ in range(6)]
        inputs = toy_inputs(domains, owners=[1, 1, 1, 2, 2, 2], counts={1: labels, 2: labels},
                            events=[event(2, 5, (Event.EAT, Event.KILLED, Event.BOMB)[int(rng.integers(3))])])
        reference = compile_constraints(inputs)
        candidate = compile_constraints(inputs, inventory_backend='incremental')
        before, after = list(reference._domains), list(candidate._domains)
        assert reference._propagate(before) == candidate._propagate(after)
        assert before == after
        assert reference._matching_cache == candidate._matching_cache
        compare(reference, candidate, ())


@pytest.mark.parametrize('plies', [161, 200])
def test_full_history_q12_teacher_and_random_complete_sample(plies):
    from tests.test_facts_joint_search import q12_fixture
    env, history = q12_fixture(plies)
    assert history[160][10] == (0,)
    inputs = build_joint_input([env.knowledge_snapshot(env.state.turn)])
    targets = [TYPE_TO_INDEX[PieceType(int(env.state.piece_type_arr[pid]))] for pid in inputs.piece_ids[0]]
    reference = compile_constraints(inputs)
    candidate = compile_constraints(inputs, inventory_backend='incremental')
    np.testing.assert_array_equal(reference.teacher_masks(targets), candidate.teacher_masks(targets))
    assert reference.stats == candidate.stats
    assert reference._cache == candidate._cache
    rng = np.random.default_rng(901)
    prefix = []
    for _ in range(75):
        compare(reference, candidate, prefix)
        # Read the already stored mask; do not add queries only on one side.
        prefix.append(int(rng.choice(np.flatnonzero(candidate._masks[tuple(prefix)]))))
    reference.validate(prefix)
    candidate.validate(prefix)


def test_backend_is_explicit_and_scan_remains_default():
    inputs = toy_inputs([[0, 1], [0, 1]])
    assert compile_constraints(inputs).inventory_backend == 'scan'
    assert compile_constraints(inputs).implementation == module.CONSTRAINT_IMPLEMENTATION
    assert compile_constraints(inputs, inventory_backend='incremental').implementation == module.INCREMENTAL_IMPLEMENTATION
    with pytest.raises(ValueError, match='Inventory backend'):
        compile_constraints(inputs, inventory_backend='silent-fallback')


def test_prefix_and_mask_cache_clearing_preserves_the_current_query():
    inputs = toy_inputs([[0, 1, 2]] * 3)
    reference = compile_constraints(inputs)
    candidate = compile_constraints(inputs, inventory_backend='incremental')
    for solver in (reference, candidate):
        # Unreachable keys fill the fixed 8192-entry caches without performing
        # unrelated expensive solves. The tested query triggers both clear paths.
        solver._cache = {(-1, i): None for i in range(8192)}
        solver._masks = {(-1, i): np.zeros(12, dtype=bool) for i in range(8192)}
    compare(reference, candidate, ())
    assert len(candidate._cache) < 8192
    assert len(candidate._masks) == 1
    for prefix in ((0,), (0, 1), (2,), (2, 1)):
        compare(reference, candidate, prefix)

"""Matching reuse must preserve exact witnesses, masks, and bounded search."""
import itertools

import numpy as np
import pytest

from junqi_core.identity import TYPE_TO_INDEX
from junqi_core.rules import Event, PieceType
from junqi_rl.belief import constraints as module
from junqi_rl.belief.constraints import (
    CompiledIdentityConstraints, ConstraintBudgetExceeded, compile_constraints,
)
from tests.test_identity_constraints import enumerated, event, toy_inputs


class UncachedConstraints(CompiledIdentityConstraints):
    def _matching_assignment(self, domains, group, inventory):
        return self._uncached_matching_assignment(domains, group, inventory)


def uncached_copy(compiled):
    return UncachedConstraints(domains=compiled._domains, owners=compiled._owners,
                               inventory=compiled._inventory, relations=compiled._relations,
                               max_search_nodes=compiled.max_search_nodes)


def assert_query_parity(cached, uncached, prefix):
    outcomes = []
    for solver in (cached, uncached):
        try:
            outcomes.append(solver.allowed_next(prefix).tolist())
        except ConstraintBudgetExceeded:
            outcomes.append("budget_exceeded")
    assert outcomes[0] == outcomes[1]
    assert cached.stats == uncached.stats
    assert cached._cache == uncached._cache
    return outcomes[0]


def test_every_bitset_keeps_original_ascending_iteration_order():
    for bits in range(4096):
        remaining, expected = bits, []
        while remaining:
            bit = remaining & -remaining
            expected.append(bit.bit_length() - 1)
            remaining ^= bit
        assert list(module._values(bits)) == expected


@pytest.mark.parametrize("outcome", [Event.EAT, Event.KILLED, Event.BOMB])
def test_all_satisfying_and_invalid_prefixes_keep_witnesses(outcome, monkeypatch):
    # Force eviction during normal solving; caching must only change its cost.
    monkeypatch.setattr(module, "_MATCHING_CACHE_LIMIT", 3)
    labels = [TYPE_TO_INDEX[t] for t in (PieceType.JUNZH, PieceType.SHIZH, PieceType.LVZH)]
    inputs = toy_inputs([labels] * 6, owners=[1, 1, 1, 2, 2, 2],
                        counts={1: labels, 2: labels},
                        events=[event(2, 5, outcome)])
    cached = compile_constraints(inputs)
    uncached = uncached_copy(cached)
    assignments = enumerated(inputs)
    prefixes = sorted({x[:size] for x in assignments for size in range(6)} |
                      set(itertools.product(labels, repeat=2)))
    for prefix in prefixes:
        actual = assert_query_parity(cached, uncached, prefix)
        expected = {x[len(prefix)] for x in assignments if x[:len(prefix)] == prefix}
        assert set(np.flatnonzero(actual)) == expected
    assert len(cached._matching_cache) <= 3
    assert cached.matching_cache_stats["evictions"] > 0


@pytest.mark.parametrize("budget", [1, 2, 5, 20, 1000])
def test_budget_exhaustion_and_retry_have_identical_node_accounting(budget):
    labels = [TYPE_TO_INDEX[t] for t in (PieceType.JUNZH, PieceType.SHIZH, PieceType.LVZH)]
    inputs = toy_inputs([labels] * 6, owners=[1, 1, 1, 2, 2, 2],
                        counts={1: labels, 2: labels},
                        events=[event(2, 5, Event.BOMB)])
    cached = compile_constraints(inputs, max_search_nodes=budget)
    uncached = uncached_copy(cached)
    assert_query_parity(cached, uncached, ())
    if budget == 1:
        assert cached.stats["budget_exhaustions"] == 1
        assert () not in cached._masks
    cached.max_search_nodes = uncached.max_search_nodes = 1000
    for prefix in [tuple(labels[:i]) for i in range(4)]:
        assert_query_parity(cached, uncached, prefix)


def test_matching_cache_covers_failure_inventory_group_and_mutable_results():
    solver = CompiledIdentityConstraints(domains=[3] * 4, owners=[0, 0, 1, 1],
                                         inventory=[[1, 1] + [0] * 10] * 2, relations=[])
    inventory = (1, 1) + (0,) * 10
    first = solver._matching_assignment([3] * 4, (0, 1), inventory)
    expected = first.copy()
    first.clear()
    assert solver._matching_assignment([3] * 4, (0, 1), inventory) == expected
    assert set(solver._matching_assignment([3] * 4, (2, 3), inventory)) == {2, 3}
    alternate = solver._matching_assignment([3] * 4, (0, 1), (2, 0) + (0,) * 10)
    assert set(alternate.values()) == {0}
    assert solver._matching_assignment([1] * 4, (0, 1), inventory) is None
    hits = solver.matching_cache_stats["hits"]
    assert solver._matching_assignment([1] * 4, (0, 1), inventory) is None
    assert solver.matching_cache_stats["hits"] == hits + 1
    other = uncached_copy(solver)
    assert not other._matching_cache

"""Exact completion masks for facts-only autoregressive identity generation.

This module computes feasibility, never identity probabilities. A candidate is
allowed only if domains, initial army inventories, and every recorded combat
relation have a complete satisfying assignment. Domains include public path
restrictions and elimination by absence of legal moves. Full game-history
reconstruction remains an independent check of the complete engine broadcast.

Returned masks define the support of each neural conditional. They do not make
the resulting autoregressive distribution uniform over complete assignments.
No infeasible proposal is repaired and no budget failure becomes a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from junqi_core.board import index_to_pos
from junqi_core.identity import TRACKED_TYPES
from junqi_core.move_gen import PieceRef, legal_moves_from
from junqi_core.rules import CAMP_INDICES, Event, PieceType, Seat

from .joint import IDENTITY_INPUT_SCHEMA, IdentityConstraintInput

CONSTRAINT_SCHEMA = "junqi.identity-domain-inventory-combat-q12.v2"
CONSTRAINT_IMPLEMENTATION = "junqi.constraints.bit-table-matching-cache.v1"
_N_TYPES = 12
_BIT_VALUES = tuple(tuple(label for label in range(_N_TYPES) if bits & (1 << label))
                    for bits in range(1 << _N_TYPES))
_MATCHING_CACHE_LIMIT = 256


class ConstraintBudgetExceeded(RuntimeError):
    """Feasibility is unknown because a bounded exact search ran out of nodes."""


@dataclass
class _Budget:
    remaining: int


def _values(bits: int):
    return _BIT_VALUES[bits]


def _immobility_domains(pieces, seat):
    """Exact per-piece type restrictions when a public seat has no moves.

    All other piece types are opaque blockers. Movement has only three type
    classes: immobile, ordinary movable, and engineer. No rank probabilities
    or simulator identities participate in this check.
    """
    restrictions = {}
    for pos, piece in tuple(pieces.items()):
        if piece.seat != seat:
            continue
        possible = np.ones(_N_TYPES, dtype=bool)
        for representative, predicate in (
            (PieceType.PAIZH, lambda t: not t.is_immobile and not t.is_engineer),
            (PieceType.GONGB, lambda t: t.is_engineer),
        ):
            pieces[pos] = PieceRef(seat, representative, piece_id=piece.piece_id)
            if legal_moves_from(pieces, pos, seat):
                possible &= np.asarray([not predicate(kind) for kind in TRACKED_TYPES])
        pieces[pos] = piece
        restrictions[piece.piece_id] = possible
    return restrictions


def _q12_domains(events):
    """Replay only public occupancy to extract no-legal-move eliminations.

    The broadcast lists flag surrender, empty-army death, then Q12 deaths in
    engine order. Therefore a listed seat with remaining pieces whose flag
    was not captured necessarily has no legal action at that point. Removing
    each such army before checking the next preserves Q12 cascade blockers.
    """
    if not any(event.seats_died for event in events):
        return {}
    pieces = {index_to_pos(Seat(owner), slot):
              PieceRef(Seat(owner), PieceType.DARK, piece_id=owner * 30 + slot)
              for owner in range(4) for slot in range(30) if slot not in CAMP_INDICES}
    restrictions = {}
    for event in events:
        src, dst = (event.src % 17, event.src // 17), (event.dst % 17, event.dst // 17)
        attacker, defender = pieces.get(src), pieces.get(dst)
        if (attacker is None or attacker.piece_id != event.attacker_pid
                or (-1 if defender is None else defender.piece_id) != event.defender_pid):
            raise ValueError('Public history occupancy or piece identity mismatch')
        if event.event in (Event.MOVE, Event.EAT):
            del pieces[src]
            pieces[dst] = attacker
        elif event.event is Event.KILLED:
            del pieces[src]
        elif event.event is Event.BOMB:
            del pieces[src]
            del pieces[dst]
        else:
            raise ValueError('Unknown public movement/combat event')
        surrendered = event.defender_pid // 30 if event.flag_captured else None
        for owner in event.seats_died:
            seat = Seat(owner)
            if owner != surrendered:
                for pid, domain in _immobility_domains(pieces, seat).items():
                    restrictions[pid] = restrictions.get(pid, np.ones(_N_TYPES, dtype=bool)) & domain
            for pos, piece in tuple(pieces.items()):
                if piece.seat == seat:
                    del pieces[pos]
    return restrictions


class CompiledIdentityConstraints:
    """Reusable constraints for one observer row, with exact prefix caches.

    ``validate`` returns None or raises ValueError. ``allowed_next`` returns a
    boolean array of length 12; all-false means the prefix has no completion.
    Exhausting ``max_search_nodes`` raises ConstraintBudgetExceeded instead.
    ``teacher_masks`` uses only each preceding causal prefix to compute masks.
    """

    schema = CONSTRAINT_SCHEMA
    implementation = CONSTRAINT_IMPLEMENTATION

    def __init__(self, *, domains, owners, inventory, relations,
                 max_search_nodes=100_000):
        if type(max_search_nodes) is not int or max_search_nodes < 1:
            raise ValueError("max_search_nodes must be a positive integer")
        self.max_search_nodes = max_search_nodes
        self.size = len(domains)
        self._domains = tuple(int(d) for d in domains)
        self._owners = tuple(int(o) for o in owners)
        self._inventory = tuple(tuple(int(n) for n in row) for row in inventory)
        self._groups = tuple(tuple(i for i, o in enumerate(self._owners) if o == owner)
                             for owner in range(len(self._inventory)))
        self._relations = tuple(relations)
        self._arcs = []
        for left, right, matrix in self._relations:
            self._arcs.append((left, right, tuple(
                sum(1 << j for j in np.flatnonzero(row)) for row in matrix)))
            self._arcs.append((right, left, tuple(
                sum(1 << j for j in np.flatnonzero(row)) for row in matrix.T)))
        self._cache: dict[tuple[int, ...], tuple[int, ...] | None] = {}
        self._masks: dict[tuple[int, ...], np.ndarray] = {}
        # Per-observer and bounded: no retention across states or games. The
        # key contains every argument of the exact matching problem. Retaining
        # its deterministic witness preserves branching and search-node counts.
        self._matching_cache: dict[tuple, dict[int, int] | None] = {}
        self.matching_cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.stats = {"queries": 0, "cache_hits": 0, "search_nodes": 0,
                      "budget_exhaustions": 0}

    def _labels(self, labels, *, complete=False):
        a = np.asarray(labels)
        if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
            raise ValueError("Identity labels must be a one-dimensional integer sequence")
        if len(a) > self.size or (complete and len(a) != self.size):
            raise ValueError("Identity assignment has the wrong length")
        if a.size and ((a < 0).any() or (a >= _N_TYPES).any()):
            raise ValueError("Identity labels must be between 0 and 11")
        return tuple(int(x) for x in a)

    def validate(self, labels) -> None:
        labels = self._labels(labels, complete=True)
        if any(not (self._domains[i] & (1 << label)) for i, label in enumerate(labels)):
            raise ValueError("Assignment violates a known identity or public type domain")
        counts = np.zeros((len(self._inventory), _N_TYPES), dtype=np.int64)
        for owner, label in zip(self._owners, labels):
            counts[owner, label] += 1
        if not np.array_equal(counts, self._inventory):
            raise ValueError("Assignment violates initial army inventory")
        for left, right, matrix in self._relations:
            if not matrix[labels[left], labels[right]]:
                raise ValueError("Assignment violates a public combat relation")

    def _matching_assignment(self, domains, group, inventory):
        key = (group, inventory, tuple(domains[i] for i in group))
        if key in self._matching_cache:
            self.matching_cache_stats["hits"] += 1
            result = self._matching_cache[key]
        else:
            self.matching_cache_stats["misses"] += 1
            result = self._uncached_matching_assignment(domains, group, inventory)
            if len(self._matching_cache) >= _MATCHING_CACHE_LIMIT:
                del self._matching_cache[next(iter(self._matching_cache))]
                self.matching_cache_stats["evictions"] += 1
            self._matching_cache[key] = result
        # Callers cannot mutate a retained witness and affect later queries.
        return None if result is None else result.copy()

    def _uncached_matching_assignment(self, domains, group, inventory):
        """Capacitated bipartite matching: exact within each army."""
        occupied = [[] for _ in range(_N_TYPES)]

        def augment(piece, visited):
            for label in _values(domains[piece]):
                if label in visited:
                    continue
                visited.add(label)
                if len(occupied[label]) < inventory[label]:
                    occupied[label].append(piece)
                    return True
                for index, previous in enumerate(occupied[label]):
                    if augment(previous, visited):
                        occupied[label][index] = piece
                        return True
            return False

        for i in sorted(group, key=lambda i: domains[i].bit_count()):
            if not augment(i, set()):
                return None
        return {i: label for label, pieces in enumerate(occupied) for i in pieces}

    def _propagate(self, domains):
        changed = True
        while changed:
            changed = False
            for left, right, supports in self._arcs:
                narrowed = sum(1 << label for label in _values(domains[left])
                               if supports[label] & domains[right])
                if not narrowed:
                    return False
                if narrowed != domains[left]:
                    domains[left] = narrowed
                    changed = True
            for group, inventory in zip(self._groups, self._inventory):
                for label, capacity in enumerate(inventory):
                    bit = 1 << label
                    possible = [i for i in group if domains[i] & bit]
                    fixed = sum(domains[i] == bit for i in possible)
                    if fixed > capacity or len(possible) < capacity:
                        return False
                    if fixed == capacity:
                        for i in possible:
                            if domains[i] != bit:
                                domains[i] ^= bit
                                if not domains[i]:
                                    return False
                                changed = True
                    elif len(possible) == capacity:
                        for i in possible:
                            if domains[i] != bit:
                                domains[i] = bit
                                changed = True
        return all(self._matching_assignment(domains, group, inventory) is not None
                   for group, inventory in zip(self._groups, self._inventory))

    def _solve(self, domains, budget):
        if budget.remaining == 0:
            self.stats["budget_exhaustions"] += 1
            raise ConstraintBudgetExceeded(
                f"Exact identity feasibility exceeded {self.max_search_nodes} search nodes")
        budget.remaining -= 1
        self.stats["search_nodes"] += 1
        if not self._propagate(domains):
            return None
        # Once every surviving pair-domain rectangle is allowed, three exact
        # army matchings complete the proof without branching over free pieces.
        unresolved = set()
        for left, right, supports in self._arcs[::2]:
            if any(supports[label] & domains[right] != domains[right]
                   for label in _values(domains[left])):
                unresolved.update(i for i in (left, right) if domains[i].bit_count() > 1)
        if not unresolved:
            labels = [-1] * self.size
            for group, inventory in zip(self._groups, self._inventory):
                assignment = self._matching_assignment(domains, group, inventory)
                if assignment is None:
                    return None
                for i, label in assignment.items():
                    labels[i] = label
            return tuple(labels)
        piece = min(unresolved, key=lambda i: domains[i].bit_count())
        for label in _values(domains[piece]):
            child = domains.copy()
            child[piece] = 1 << label
            witness = self._solve(child, budget)
            if witness is not None:
                return witness
        return None

    def _completion(self, prefix, budget, prepared_domains=None):
        if prefix in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[prefix]
        domains = list(self._domains if prepared_domains is None else prepared_domains)
        for i, label in enumerate(prefix):
            domains[i] &= 1 << label
            if not domains[i]:
                return None
        witness = self._solve(domains, budget)
        # Bound retained prefixes, while keeping each current witness usable.
        if len(self._cache) >= 8192:
            self._cache.clear()
        self._cache[prefix] = witness
        return witness

    def allowed_next(self, prefix) -> np.ndarray:
        prefix = self._labels(prefix)
        if len(prefix) == self.size:
            raise ValueError("A complete identity assignment has no next token")
        self.stats["queries"] += 1
        if prefix in self._masks:
            self.stats["cache_hits"] += 1
            return self._masks[prefix].copy()
        budget = _Budget(self.max_search_nodes)
        allowed = np.zeros(_N_TYPES, dtype=bool)
        domains = list(self._domains)
        for i, label in enumerate(prefix):
            domains[i] &= 1 << label
            if not domains[i]:
                return allowed
        # Apply cheap deterministic implications once for this entire query.
        # Types with exhausted inventory or impossible public pairs need no
        # separate completion search.
        if not self._propagate(domains):
            return allowed
        witness = self._cache.get(prefix)
        for label in _values(domains[len(prefix)]):
            candidate = prefix + (label,)
            if witness is not None and witness[len(prefix)] == label:
                self._cache[candidate] = witness
                allowed[label] = True
            else:
                found = self._completion(candidate, budget, domains)
                allowed[label] = found is not None
        if len(self._masks) >= 8192:
            self._masks.clear()
        self._masks[prefix] = allowed.copy()
        return allowed

    def teacher_masks(self, labels) -> np.ndarray:
        labels = self._labels(labels, complete=True)
        self.validate(labels)
        masks = np.stack([self.allowed_next(labels[:i]) for i in range(self.size)])
        if not masks[np.arange(self.size), labels].all():
            raise ValueError("Training target has no feasible causal completion")
        return masks


def compile_constraints(inputs: IdentityConstraintInput, row=0, *, max_search_nodes=100_000
                        ) -> CompiledIdentityConstraints:
    """Compile one facts-only row; simulator truth and probabilities are absent.

    Smaller synthetic armies are accepted for exhaustive correctness checks.
    Every identity referred to by combat must occur among the latent variables
    or have an explicitly known observer type; missing identities are rejected.
    """
    if not isinstance(inputs, IdentityConstraintInput) or inputs.schema != IDENTITY_INPUT_SCHEMA:
        raise ValueError("Constraints require the facts-only identity input schema")
    if type(row) is not int or not 0 <= row < len(inputs.observer_seats):
        raise ValueError("Invalid observer row")
    ids = np.asarray(inputs.piece_ids[row])
    owners = np.asarray(inputs.owners[row])
    domains = np.array(inputs.type_domains[row], copy=True)
    inventory = np.asarray(inputs.initial_inventory[row])
    known = np.asarray(inputs.known_types[row])
    observer_known = np.asarray(inputs.observer_known_types[row])
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids) or (ids < 0).any()):
        raise ValueError("Piece ids must be unique nonnegative integers")
    if domains.shape != (len(ids), _N_TYPES) or domains.dtype != np.bool_:
        raise ValueError("Type domains must be a boolean (pieces, 12) array")
    if (inventory.ndim != 2 or inventory.shape[1] != _N_TYPES
            or inventory.dtype.kind not in "iu" or (inventory < 0).any()):
        raise ValueError("Initial inventory must contain nonnegative integer counts")
    if (owners.shape != ids.shape or owners.dtype.kind not in "iu"
            or (owners < 0).any() or (owners >= len(inventory)).any()):
        raise ValueError("Every piece requires a valid relative owner")
    if (known.shape != ids.shape or known.dtype.kind not in "iu"
            or (known < -1).any() or (known >= _N_TYPES).any()):
        raise ValueError("Known identities must be -1 or a type label")
    if (observer_known.ndim != 1 or observer_known.dtype.kind not in "iu"
            or (observer_known < -1).any() or (observer_known >= _N_TYPES).any()
            or (ids >= len(observer_known)).any()):
        raise ValueError("Observer known identities must cover every piece id")
    for owner in range(len(inventory)):
        if int(inventory[owner].sum()) != int((owners == owner).sum()):
            raise ValueError("Initial inventory does not account for all latent identities")
    for i, label in enumerate(known):
        if label >= 0:
            domains[i] &= np.arange(_N_TYPES) == label
        public_label = observer_known[ids[i]]
        if public_label >= 0:
            domains[i] &= np.arange(_N_TYPES) == public_label
    index = {int(pid): i for i, pid in enumerate(ids)}
    relations = {}

    def known_label(pid):
        if not 0 <= pid < len(observer_known) or observer_known[pid] < 0:
            raise ValueError("Combat refers to an identity absent from observer knowledge")
        return int(observer_known[pid])

    events = inputs.public_history[row]
    if any(event.seats_died for event in events):
        if len(ids) != 75:
            raise ValueError('Public elimination constraints require all 75 other-army identities')
        for pid, domain in _q12_domains(events).items():
            if pid in index:
                domains[index[pid]] &= domain
            elif not domain[known_label(pid)]:
                raise ValueError('Known identity contradicts public no-legal-move elimination')

    for event in events:
        if not event.has_combat:
            continue
        left, right = index.get(event.attacker_pid), index.get(event.defender_pid)
        matrix = np.array(event.allowed_type_pairs(), copy=True)
        if left is not None and right is not None:
            if left == right:
                raise ValueError("A piece cannot fight itself")
            if left > right:
                left, right, matrix = right, left, matrix.T
            key = left, right
            relations[key] = relations.get(key, np.ones((_N_TYPES, _N_TYPES), dtype=bool)) & matrix
        elif left is not None:
            domains[left] &= matrix[:, known_label(event.defender_pid)]
        elif right is not None:
            domains[right] &= matrix[known_label(event.attacker_pid)]
        elif not matrix[known_label(event.attacker_pid), known_label(event.defender_pid)]:
            raise ValueError("Known identities contradict a public combat result")
    if not domains.any(-1).all():
        raise ValueError("Contradictory public identity restrictions")
    bits = [sum(1 << int(label) for label in np.flatnonzero(row)) for row in domains]
    return CompiledIdentityConstraints(
        domains=bits, owners=owners, inventory=inventory,
        relations=[(left, right, matrix) for (left, right), matrix in relations.items()],
        max_search_nodes=max_search_nodes)

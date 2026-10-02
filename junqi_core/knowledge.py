"""Persistent observer facts and public events, without identity probabilities.

Recording is optional. Boolean identity domains express legal restrictions,
never normalized weights. Unknown dead identities remain latent variables.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from .combat_memory import ORDINARY_RANKS
from .identity import ADMISSIBLE_BY_INDEX, NUM_TRACKED_TYPES, TRACKED_TYPES, TYPE_TO_INDEX
from .rules import (
    Event,
    PieceType,
    Seat,
    ShowMode,
    resolve_combat,
    siling_reveals_dst,
    siling_reveals_src,
)

if TYPE_CHECKING:
    from .state import GameState, MoveResult

KNOWLEDGE_SCHEMA = "junqi.observer-facts.v1"
_INITIAL_DOMAINS = np.tile(np.asarray([
    [kind in allowed for kind in TRACKED_TYPES] for allowed in ADMISSIBLE_BY_INDEX
], dtype=bool), (4, 1))
_RANKS = np.asarray([
    ORDINARY_RANKS.index(kind) + 1 if kind in ORDINARY_RANKS else 0
    for kind in TRACKED_TYPES
])
_INITIAL_DOMAINS.setflags(write=False)
_RANKS.setflags(write=False)


@dataclass(frozen=True, slots=True)
class PublicEvent:
    step: int
    seat: int
    src: int
    dst: int
    attacker_pid: int
    defender_pid: int
    event: Event
    flag_reveal_src: bool
    flag_reveal_dst: bool
    flag_captured: bool
    engineer_revealed: bool
    seats_died: tuple[int, ...]

    @property
    def has_combat(self) -> bool:
        return self.defender_pid >= 0

    def allowed_type_pairs(self) -> np.ndarray:
        """Exact game table for this broadcast, including non-revelations."""
        if not self.has_combat:
            raise ValueError("A movement event has no combat identity relation")
        return _combat_pairs(self.event, self.flag_reveal_src,
                             self.flag_reveal_dst, self.flag_captured)


@cache
def _combat_pairs(
    event: Event, reveal_src: bool, reveal_dst: bool, flag_captured: bool,
) -> np.ndarray:
    allowed = np.zeros((NUM_TRACKED_TYPES, NUM_TRACKED_TYPES), dtype=bool)
    for i, attacker in enumerate(TRACKED_TYPES):
        if attacker.is_immobile:
            continue
        for j, defender in enumerate(TRACKED_TYPES):
            outcome = resolve_combat(attacker, defender)
            allowed[i, j] = (
                outcome is event
                and siling_reveals_src(attacker, defender, outcome) == reveal_src
                and siling_reveals_dst(attacker, defender, outcome) == reveal_dst
                and defender.is_flag == flag_captured
            )
    allowed.setflags(write=False)
    return allowed


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class KnowledgeSnapshot:
    schema: str
    observer: Seat
    show_mode: ShowMode
    step: int
    initial_cells: np.ndarray
    cells: np.ndarray
    alive: np.ndarray
    known_types: np.ndarray
    move_counts: np.ndarray
    active_eat_counts: np.ndarray
    passive_survive_counts: np.ndarray
    type_domains: np.ndarray
    events: tuple[PublicEvent, ...]


class ObserverKnowledge:
    """Record one observer's information from the start of a game.

    Only explicitly visible types are copied. Debug/private MoveResult fields,
    hidden remaining counts, old belief tensors, and rule priors are ignored.
    This reference interface leaves policy observations and their schema intact.
    """

    def __init__(self, state: GameState, observer: Seat) -> None:
        if state.move_counter != 0:
            raise ValueError("Observer knowledge requires the complete public history from reset")
        self.observer = Seat(observer)
        self.show_mode = state.show_mode
        self.step = 0
        self.events: list[PublicEvent] = []
        self.initial_cells = np.where(
            state.zero_x >= 0, state.zero_y.astype(np.int16) * 17 + state.zero_x, -1)
        self.known_types = np.full(120, -1, dtype=np.int8)
        visible = state.piece_seat_arr == int(self.observer)
        if self.show_mode is ShowMode.BRIGHT:
            visible = state.piece_seat_arr >= 0
        elif self.show_mode is ShowMode.HALF_DARK:
            visible |= state.piece_seat_arr == int(self.observer.teammate)
        # Index first, so hidden entries need not be read at all.
        for pid in np.flatnonzero(visible):
            self.known_types[pid] = TYPE_TO_INDEX[PieceType(int(state.piece_type_arr[pid]))]
        self._sync_public(state)

    def _set_known(self, pid: int, kind: PieceType) -> None:
        label = TYPE_TO_INDEX[kind]
        if self.known_types[pid] not in (-1, label):
            raise ValueError("Conflicting public identity facts")
        self.known_types[pid] = label

    def _sync_public(self, state: GameState) -> None:
        self.alive = state.alive.copy()
        self.cells = np.where(state.alive,
            state.pos_y.astype(np.int16) * 17 + state.pos_x, -1)
        self.move_counts = state.move_count_arr.copy()
        self.active_eat_counts = state.active_eat_arr.copy()
        self.passive_survive_counts = state.passive_surv_arr.copy()
        self.is_engineer = state.combat_memory.is_gongb[int(self.observer)].copy()
        self.not_engineer = state.combat_memory.not_gongb[int(self.observer)].copy()
        self.rank_floor = state.combat_memory.rank_floor[int(self.observer)].copy()
        for pid in np.flatnonzero(self.is_engineer):
            self._set_known(int(pid), PieceType.GONGB)
        # Revealed flag locations are public; all other identities stay hidden.
        for seat in np.flatnonzero(state.seat_flag_revealed_arr):
            ids = np.flatnonzero(state.alive & (state.piece_seat_arr == seat))
            flags = ids[state.piece_type_arr[ids] == PieceType.JUNQI.value]
            for pid in flags:
                self._set_known(int(pid), PieceType.JUNQI)

    def observe(self, previous: GameState, current: GameState, result: MoveResult) -> None:
        """Incrementally append one public event before the old state is lost."""
        if previous.show_mode is not self.show_mode or current.show_mode is not self.show_mode:
            raise ValueError("Observer visibility changed within one game")
        if previous.move_counter != self.step or current.move_counter != self.step + 1:
            raise ValueError("Public history is missing, duplicated, or belongs to another game")
        sx, sy = result.src
        dx, dy = result.dst
        attacker = int(previous.cell_piece_id[sy * 17 + sx])
        defender = int(previous.cell_piece_id[dy * 17 + dx])
        if attacker < 0:
            raise ValueError("Public event has no source piece")
        event = PublicEvent(
            current.move_counter, int(result.seat), sy * 17 + sx, dy * 17 + dx,
            attacker, defender, result.event, bool(result.flag_reveal_src),
            bool(result.flag_reveal_dst), bool(result.flag_captured),
            bool(current.combat_memory.is_gongb[:, attacker].all()),
            tuple(int(seat) for seat in result.seats_died_this_step),
        )
        if event.flag_reveal_src:
            self._set_known(attacker, PieceType.SILING)
        if event.flag_reveal_dst:
            self._set_known(defender, PieceType.SILING)
        if event.flag_captured:
            self._set_known(defender, PieceType.JUNQI)
        if event.event is Event.KILLED and event.flag_reveal_src:
            self._set_known(defender, PieceType.DILEI)
        self.events.append(event)
        self.step = current.move_counter
        self._sync_public(current)

    def snapshot(self) -> KnowledgeSnapshot:
        domains = _INITIAL_DOMAINS.copy()
        moved = (self.move_counts > 0) | (self.active_eat_counts > 0)
        for kind in (PieceType.JUNQI, PieceType.DILEI):
            domains[moved, TYPE_TO_INDEX[kind]] = False
        domains[self.not_engineer, TYPE_TO_INDEX[PieceType.GONGB]] = False
        constrained = self.rank_floor > 0
        domains[constrained] &= _RANKS[None, :] >= self.rank_floor[constrained, None]
        for pid in np.flatnonzero(self.known_types >= 0):
            label = int(self.known_types[pid])
            if not domains[pid, label]:
                raise ValueError("Known identity contradicts public restrictions")
            domains[pid] = False
            domains[pid, label] = True
        return KnowledgeSnapshot(
            schema=KNOWLEDGE_SCHEMA,
            observer=self.observer,
            show_mode=self.show_mode,
            step=self.step,
            initial_cells=_readonly_copy(self.initial_cells),
            cells=_readonly_copy(self.cells),
            alive=_readonly_copy(self.alive),
            known_types=_readonly_copy(self.known_types),
            move_counts=_readonly_copy(self.move_counts),
            active_eat_counts=_readonly_copy(self.active_eat_counts),
            passive_survive_counts=_readonly_copy(self.passive_survive_counts),
            type_domains=_readonly_copy(domains),
            events=tuple(self.events),
        )

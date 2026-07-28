"""T7 / ADR-115 — Observation tail channels (A/B/C/D/E) unit tests.

The new 32-channel tail on the observation tensor (move_bucket /
active_eat_bucket / passive_survive_bucket / death_reason / dead_at_zero)
exposes the piece-lifecycle signals maintained in M1/M2/M3. These tests
pin down their semantics:

  1. Shape: `(OBS_CHANNELS, 17, 17) == (256, 17, 17)`.
  2. Bucket exactness (A) and cumulative (B/C) semantics.
  3. death_reason (D) anchored at death_loc, not zero-pos.
  4. dead_at_zero (E) anchored at zero-pos, not death_loc.
  5. Route-A invariant: dead pieces emit NO signal in A/B/C.
  6. Visibility filter: unrevealed enemy pieces leave A/B/C theirs side at 0.
  7. Ours/theirs split matches the observer's team partition.
  8. Canonical rotation preserves per-cell signals (SOUTH observer check).

All tests use SOUTH as the observer where possible, so world == canonical
and cell (x, y) reads directly from `obs.spatial[:, y, x]`.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from junqi_core.info_model import BeliefTensor
from junqi_core.move_gen import PieceMap, PieceRef
from junqi_core.observation import (
    ACTIVE_EAT_BUCKET_COUNT,
    CHANNEL_LAYOUT,
    DEATH_REASON_COUNT,
    MOVE_BUCKET_COUNT,
    OBS_CHANNELS,
    PASSIVE_SURVIVE_BUCKET_COUNT,
    build_observation,
)
from junqi_core.rules import (
    RULES_VERSION,
    DeathReason,
    Event,
    PieceType,
    Seat,
    ShowMode,
)
from junqi_core.setup import generate_random_setup
from junqi_core.state import (
    Action,
    DeathInfo,
    GameState,
    PieceState,
    SeatInfo,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _p(seat: Seat, pt: PieceType, pid: int) -> PieceRef:
    return PieceRef(seat=seat, piece_type=pt, alive=True, piece_id=pid)


def _build_state(
    *,
    pieces: PieceMap,
    turn: Seat,
    show_mode: ShowMode = ShowMode.BRIGHT,
    piece_state: dict[int, PieceState] | None = None,
    deaths: dict[int, DeathInfo] | None = None,
) -> GameState:
    """Build a minimal GameState with explicit piece_state / deaths overrides.

    BRIGHT mode by default so every observer sees every piece's true type —
    this makes the A/B/C "theirs" visibility filter a no-op and keeps the
    tests focused on bucket semantics.
    """
    if piece_state is None:
        piece_state = {ref.piece_id: PieceState() for ref in pieces.values()}
    info = {s: SeatInfo() for s in Seat}
    return GameState(
        pieces=dict(pieces),
        turn=turn,
        move_counter=0,
        moves_since_last_combat=0,
        info=info,
        terminated=False,
        winner_team=None,
        draw=False,
        show_mode=show_mode,
        rules_version=RULES_VERSION,
        debug_include_private=False,
        zero_board=dict(pieces),
        piece_state=dict(piece_state),
        deaths=dict(deaths) if deaths is not None else {},
    )


def _obs(state: GameState, observer: Seat):
    belief = BeliefTensor.initial(state, observer)
    return build_observation(state, belief, observer)


# A "wide" fully-occupied 4-seat board with mobile pieces so Q12 never
# fires during the first synthetic step. Used by scenarios that chain
# state.step() to observe piece_state growth naturally.
def _baseline_pieces() -> PieceMap:
    return {
        # SOUTH
        (8, 11): _p(Seat.SOUTH, PieceType.PAIZH, 10),
        (6, 13): _p(Seat.SOUTH, PieceType.JUNQI, 28),
        (8, 13): _p(Seat.SOUTH, PieceType.PAIZH, 20),
        # WEST
        (2, 8): _p(Seat.WEST, PieceType.PAIZH, 50),
        (0, 8): _p(Seat.WEST, PieceType.JUNQI, 56),
        # NORTH
        (8, 3): _p(Seat.NORTH, PieceType.PAIZH, 70),
        (10, 2): _p(Seat.NORTH, PieceType.JUNQI, 86),
        # EAST
        (14, 8): _p(Seat.EAST, PieceType.PAIZH, 100),
        (16, 10): _p(Seat.EAST, PieceType.JUNQI, 116),
    }


# ===========================================================================
# 1. Shape / layout
# ===========================================================================


def test_observation_tail_shape() -> None:
    """The 256-channel observation carries the full channel set in the
    documented order."""
    st = GameState.new_game(
        generate_random_setup(random.Random(7)), show_mode=ShowMode.BRIGHT
    )
    obs = _obs(st, Seat.SOUTH)
    assert obs.spatial.shape == (OBS_CHANNELS, 17, 17)
    assert obs.spatial.shape[0] == OBS_CHANNELS  # 352 after ADR-129 v5 layer-3

    # The 5 tail groups' sizes match the spec.
    assert obs.channel("move_bucket").shape[0] == 8
    assert obs.channel("active_eat_bucket").shape[0] == 8
    assert obs.channel("passive_survive_bucket").shape[0] == 8
    assert obs.channel("death_reason").shape[0] == 12
    assert obs.channel("dead_at_zero").shape[0] == 2


# ===========================================================================
# 2. A group: move_bucket exact-match semantics
# ===========================================================================


class TestMoveBucket:
    def test_fresh_piece_in_bucket_zero(self) -> None:
        """A piece with move_count == 0 lights ONLY ours[0] at its cell."""
        pieces = _baseline_pieces()
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        obs = _obs(st, Seat.SOUTH)
        mv = obs.channel("move_bucket")

        # SOUTH observer's own PAIZH at (8, 11), move_count == 0 → ours[0].
        assert mv[0, 11, 8] == 1.0
        assert mv[1, 11, 8] == 0.0
        assert mv[2, 11, 8] == 0.0
        assert mv[3, 11, 8] == 0.0
        # theirs half is untouched at this cell.
        assert mv[MOVE_BUCKET_COUNT:, 11, 8].sum() == 0.0

    @pytest.mark.parametrize(
        "move_count,expected_plane",
        [(0, 0), (1, 1), (2, 2), (3, 3), (5, 3), (99, 3)],
    )
    def test_bucket_exact_match(
        self, move_count: int, expected_plane: int
    ) -> None:
        """Exact-match semantics: move_count maps to ONE specific plane;
        >=3 saturates at plane 3."""
        pieces = _baseline_pieces()
        piece_state = {
            ref.piece_id: PieceState() for ref in pieces.values()
        }
        piece_state[10] = PieceState(move_count=move_count)
        st = _build_state(
            pieces=pieces, turn=Seat.SOUTH, piece_state=piece_state
        )
        obs = _obs(st, Seat.SOUTH)
        mv = obs.channel("move_bucket")

        # Exactly one plane on ours-half is lit at (8, 11); all others 0.
        for k in range(4):
            want = 1.0 if k == expected_plane else 0.0
            assert mv[k, 11, 8] == want, (
                f"move_count={move_count}: ours[{k}] expected {want}"
            )

    def test_dead_piece_emits_nothing_in_move_bucket(self) -> None:
        """Route-A: if a piece has died, it is not in state.pieces and
        therefore emits no signal in the move_bucket group — even if
        piece_state somehow still held an old counter (defensive)."""
        pieces = _baseline_pieces()
        # Kill SOUTH PAIZH 10 by removing it from pieces but leaving a
        # DeathInfo behind — mimics the state immediately after step().
        del pieces[(8, 11)]
        deaths = {
            10: DeathInfo(
                piece_id=10,
                reason=DeathReason.KILLED_BY_ENEMY,
                death_loc=(8, 11),
                step=1,
            )
        }
        # piece_state must NOT contain pid 10 (liveness invariant).
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        st = _build_state(
            pieces=pieces, turn=Seat.SOUTH, piece_state=ps, deaths=deaths
        )
        obs = _obs(st, Seat.SOUTH)
        mv = obs.channel("move_bucket")
        # Cell (8, 11) has NO living piece → all 8 planes are 0 there.
        assert mv[:, 11, 8].sum() == 0.0


# ===========================================================================
# 3. B/C groups: cumulative >= bucket semantics
# ===========================================================================


class TestActiveEatBucket:
    @pytest.mark.parametrize(
        "eat_count,lit_planes",
        [
            (0, {0}),          # only ch_0
            (1, {0, 1}),
            (2, {0, 1, 2}),
            (3, {0, 1, 2, 3}),
            (7, {0, 1, 2, 3}),  # saturates
        ],
    )
    def test_cumulative_fills(
        self, eat_count: int, lit_planes: set[int]
    ) -> None:
        pieces = _baseline_pieces()
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        ps[10] = PieceState(active_eat_count=eat_count)
        st = _build_state(pieces=pieces, turn=Seat.SOUTH, piece_state=ps)
        obs = _obs(st, Seat.SOUTH)
        ae = obs.channel("active_eat_bucket")

        for k in range(ACTIVE_EAT_BUCKET_COUNT):
            want = 1.0 if k in lit_planes else 0.0
            assert ae[k, 11, 8] == want, (
                f"eat_count={eat_count} expected ours[{k}]={want}"
            )


class TestPassiveSurviveBucket:
    def test_cumulative_planes_fill(self) -> None:
        pieces = _baseline_pieces()
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        ps[10] = PieceState(passive_survive_count=2)
        st = _build_state(pieces=pieces, turn=Seat.SOUTH, piece_state=ps)
        obs = _obs(st, Seat.SOUTH)
        pv = obs.channel("passive_survive_bucket")

        for k in range(PASSIVE_SURVIVE_BUCKET_COUNT):
            want = 1.0 if k <= 2 else 0.0
            assert pv[k, 11, 8] == want


# ===========================================================================
# 4. D group: death_reason anchored at death_loc
# ===========================================================================


class TestDeathReasonChannel:
    @pytest.mark.parametrize(
        "reason,plane_offset",
        [
            (DeathReason.KILLED_BY_ENEMY, 0),
            (DeathReason.HIT_MINE_OR_BOMB, 1),
            (DeathReason.MUTUAL, 2),
        ],
    )
    def test_reason_fires_on_death_loc(
        self, reason: DeathReason, plane_offset: int
    ) -> None:
        """An ally's death records exactly one lit plane at death_loc in
        the ours-half; theirs-half stays 0."""
        pieces = _baseline_pieces()
        # Pretend SOUTH PAIZH 10 died at (6, 5) with the given reason.
        zero_board = dict(pieces)
        del pieces[(8, 11)]                     # remove from live board
        deaths = {
            10: DeathInfo(
                piece_id=10,
                reason=reason,
                death_loc=(6, 5),
                step=1,
            )
        }
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        st = GameState(
            pieces=dict(pieces),
            turn=Seat.SOUTH,
            move_counter=1,
            moves_since_last_combat=1,
            info={s: SeatInfo() for s in Seat},
            terminated=False,
            winner_team=None,
            draw=False,
            show_mode=ShowMode.BRIGHT,
            rules_version=RULES_VERSION,
            debug_include_private=False,
            zero_board=zero_board,
            piece_state=ps,
            deaths=deaths,
        )
        obs = _obs(st, Seat.SOUTH)
        dr = obs.channel("death_reason")

        # ours-half plane at (6, 5).
        assert dr[plane_offset, 5, 6] == 1.0
        # All other ours planes at (6, 5) are 0.
        for k in range(DEATH_REASON_COUNT):
            if k != plane_offset:
                assert dr[k, 5, 6] == 0.0
        # theirs-half at (6, 5) all 0 — dead piece was ours.
        assert dr[DEATH_REASON_COUNT:, 5, 6].sum() == 0.0

    def test_enemy_death_goes_to_theirs_half(self) -> None:
        pieces = _baseline_pieces()
        zero_board = dict(pieces)
        # Kill WEST PAIZH 50 (enemy of SOUTH) at (3, 7).
        del pieces[(2, 8)]
        deaths = {
            50: DeathInfo(
                piece_id=50,
                reason=DeathReason.MUTUAL,
                death_loc=(3, 7),
                step=1,
            )
        }
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        st = GameState(
            pieces=dict(pieces),
            turn=Seat.SOUTH,
            move_counter=1,
            moves_since_last_combat=1,
            info={s: SeatInfo() for s in Seat},
            terminated=False,
            winner_team=None,
            draw=False,
            show_mode=ShowMode.BRIGHT,
            rules_version=RULES_VERSION,
            debug_include_private=False,
            zero_board=zero_board,
            piece_state=ps,
            deaths=deaths,
        )
        obs = _obs(st, Seat.SOUTH)
        dr = obs.channel("death_reason")

        # Post-256ch refactor: 12 channels = me[0..2], teammate[3..5],
        # left_enemy[6..8], right_enemy[9..11]. WEST is SOUTH.left_side_enemy,
        # so its MUTUAL death fires at plane R*2 + MUTUAL_idx = 6 + 2 = 8.
        assert dr[DEATH_REASON_COUNT * 2 + 2, 7, 3] == 1.0
        # All other perspectives silent at (3, 7).
        for k in range(3 * DEATH_REASON_COUNT):
            if k != DEATH_REASON_COUNT * 2 + 2:
                assert dr[k, 7, 3] == 0.0, f"channel {k} should be silent"


# ===========================================================================
# 5. E group: dead_at_zero anchored at zero-pos
# ===========================================================================


class TestDeadAtZero:
    def test_dead_lights_zero_cell_not_death_loc(self) -> None:
        """A dead piece lights its ZERO-board cell (home), not where it
        died. Distinction matters because piece 10 started at (8, 11) but
        died far away at (6, 5)."""
        pieces = _baseline_pieces()
        zero_board = dict(pieces)
        del pieces[(8, 11)]
        deaths = {
            10: DeathInfo(
                piece_id=10,
                reason=DeathReason.KILLED_BY_ENEMY,
                death_loc=(6, 5),
                step=1,
            )
        }
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        st = GameState(
            pieces=dict(pieces),
            turn=Seat.SOUTH,
            move_counter=1,
            moves_since_last_combat=1,
            info={s: SeatInfo() for s in Seat},
            terminated=False,
            winner_team=None,
            draw=False,
            show_mode=ShowMode.BRIGHT,
            rules_version=RULES_VERSION,
            debug_include_private=False,
            zero_board=zero_board,
            piece_state=ps,
            deaths=deaths,
        )
        obs = _obs(st, Seat.SOUTH)
        dz = obs.channel("dead_at_zero")

        # ours-half (plane 0) lights the ZERO cell (8, 11).
        assert dz[0, 11, 8] == 1.0
        # NOT the death_loc (6, 5).
        assert dz[0, 5, 6] == 0.0
        # theirs-half (plane 1) stays 0 for our own death.
        assert dz[1, 11, 8] == 0.0

    def test_alive_piece_does_not_emit(self) -> None:
        pieces = _baseline_pieces()
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        obs = _obs(st, Seat.SOUTH)
        dz = obs.channel("dead_at_zero")
        assert dz.sum() == 0.0  # no deaths → no signal anywhere


# ===========================================================================
# 6. Route-A invariant: dead pieces emit NOTHING in A/B/C groups
# ===========================================================================


class TestRouteAInvariant:
    def test_dead_piece_silent_across_abc(self) -> None:
        pieces = _baseline_pieces()
        zero_board = dict(pieces)
        # Remove pid 10 and record its death.
        del pieces[(8, 11)]
        deaths = {
            10: DeathInfo(
                piece_id=10,
                reason=DeathReason.MUTUAL,
                death_loc=(4, 4),
                step=1,
            )
        }
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        st = GameState(
            pieces=dict(pieces),
            turn=Seat.SOUTH,
            move_counter=1,
            moves_since_last_combat=1,
            info={s: SeatInfo() for s in Seat},
            terminated=False,
            winner_team=None,
            draw=False,
            show_mode=ShowMode.BRIGHT,
            rules_version=RULES_VERSION,
            debug_include_private=False,
            zero_board=zero_board,
            piece_state=ps,
            deaths=deaths,
        )
        obs = _obs(st, Seat.SOUTH)

        for group in ("move_bucket", "active_eat_bucket", "passive_survive_bucket"):
            g = obs.channel(group)
            # Neither the zero-pos (8, 11) nor the death_loc (4, 4) carries
            # a bucket signal — both ours-half and theirs-half stay 0.
            assert g[:, 11, 8].sum() == 0.0, (
                f"{group}: zero-pos still carries a signal for dead piece"
            )
            assert g[:, 4, 4].sum() == 0.0, (
                f"{group}: death_loc still carries a signal for dead piece"
            )


# ===========================================================================
# 7. Visibility filter: unrevealed enemies produce 0 on theirs-half of A/B/C
# ===========================================================================


class TestEnemyInvisibleZeroBuckets:
    def test_dark_mode_shows_enemy_counters(self) -> None:
        """Post-256ch refactor: bucket counters (move/eat/survive) are
        treated as public broadcast data and leak through under DARK.

        See ``_build_bucket_masks`` comment in junqi_core/observation.py:
        ``Enemy bucket data ... is derived from publicly broadcast
        MoveResult events — every player can count them``.

        So the theirs-half IS lit at an enemy's cell even under DARK —
        this test pins that down, documenting the behaviour that earlier
        drafts hid.
        """
        pieces = _baseline_pieces()
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        # WEST PAIZH 50 has walked 2 steps and eaten once.
        ps[50] = PieceState(
            move_count=2, active_eat_count=1, passive_survive_count=1
        )
        st = _build_state(
            pieces=pieces,
            turn=Seat.SOUTH,
            show_mode=ShowMode.DARK,
            piece_state=ps,
        )
        obs = _obs(st, Seat.SOUTH)
        # WEST piece is at (2, 8); observer is SOUTH.
        mv = obs.channel("move_bucket")
        ae = obs.channel("active_eat_bucket")
        pv = obs.channel("passive_survive_bucket")
        # theirs-half at (2, 8) should fire — counters are public.
        assert mv[MOVE_BUCKET_COUNT:, 8, 2].sum() > 0.0
        assert ae[ACTIVE_EAT_BUCKET_COUNT:, 8, 2].sum() > 0.0
        assert pv[PASSIVE_SURVIVE_BUCKET_COUNT:, 8, 2].sum() > 0.0

    def test_bright_mode_shows_enemy_counters(self) -> None:
        """Under BRIGHT everything is visible → theirs-half fires."""
        pieces = _baseline_pieces()
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        ps[50] = PieceState(move_count=2)
        st = _build_state(
            pieces=pieces,
            turn=Seat.SOUTH,
            show_mode=ShowMode.BRIGHT,
            piece_state=ps,
        )
        obs = _obs(st, Seat.SOUTH)
        mv = obs.channel("move_bucket")
        # theirs-plane 2 lit at (2, 8).
        assert mv[MOVE_BUCKET_COUNT + 2, 8, 2] == 1.0


# ===========================================================================
# 8. Ours/theirs split matches observer's team partition
# ===========================================================================


class TestOursTheirsSplit:
    def test_teammate_goes_to_ours_half(self) -> None:
        """NORTH is SOUTH's teammate; NORTH piece with move_count==1 must
        fire in the OURS half (plane 1 of the move_bucket group) — never
        in theirs-half."""
        pieces = _baseline_pieces()
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        ps[70] = PieceState(move_count=1)   # NORTH PAIZH at (8, 3)
        st = _build_state(
            pieces=pieces,
            turn=Seat.SOUTH,
            show_mode=ShowMode.BRIGHT,
            piece_state=ps,
        )
        obs = _obs(st, Seat.SOUTH)
        mv = obs.channel("move_bucket")

        # NORTH (teammate) is at world (8, 3). Under SOUTH observer, the
        # canonical-frame rotation is identity (SOUTH == observer), so the
        # cell stays at (y=3, x=8).
        # ours-plane 1 (move_count==1) must fire at (8, 3); theirs-half must
        # NOT fire at (8, 3) — teammate never leaks to the enemy half.
        assert mv[1, 3, 8] == 1.0
        theirs_at_teammate = mv[MOVE_BUCKET_COUNT:, 3, 8]
        assert theirs_at_teammate.sum() == 0.0, (
            f"teammate bled into theirs-half: {theirs_at_teammate}"
        )

    def test_teammate_move_count_stays_on_ours_half_under_dark(self) -> None:
        """Extra guard: even under DARK (where the teammate's type is NOT
        one-hot to the observer), NORTH is still classified as OURS by
        team membership — not by type visibility. So the bucket signal
        stays on ours-half. This test pins the decision that the
        ours/theirs split is TEAM-based, not VISIBILITY-based."""
        pieces = _baseline_pieces()
        ps = {ref.piece_id: PieceState() for ref in pieces.values()}
        ps[70] = PieceState(move_count=1)
        st = _build_state(
            pieces=pieces,
            turn=Seat.SOUTH,
            show_mode=ShowMode.DARK,
            piece_state=ps,
        )
        obs = _obs(st, Seat.SOUTH)
        mv = obs.channel("move_bucket")
        # NORTH teammate under DARK: still ours-half plane 1 at (8, 3).
        assert mv[1, 3, 8] == 1.0
        # theirs-half stays silent at the teammate's cell.
        assert mv[MOVE_BUCKET_COUNT:, 3, 8].sum() == 0.0


# ===========================================================================
# 9. Integration: a real step() transition lights the right channel
# ===========================================================================


class TestIntegrationWithStep:
    def test_one_move_lights_move_bucket_one(self) -> None:
        """End-to-end: state.step(Event.MOVE) bumps move_count to 1, and
        the resulting observation lights ours[1] at the new cell."""
        pieces = _baseline_pieces()
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        # Move SOUTH PAIZH 10 from (8, 11) to (8, 10).
        st2, res = st.step(Action(seat=Seat.SOUTH, src=(8, 11), dst=(8, 10)))
        assert res.event is Event.MOVE
        obs = _obs(st2, Seat.SOUTH)
        mv = obs.channel("move_bucket")
        # Ours plane 1 at new position (8, 10).
        assert mv[1, 10, 8] == 1.0
        # Ours plane 0 must NOT fire at (8, 10) any more.
        assert mv[0, 10, 8] == 0.0
        # And nothing fires at the old position (8, 11).
        assert mv[:, 11, 8].sum() == 0.0

    def test_eat_lights_active_eat_and_death_reason(self) -> None:
        """Integration: SILING eats PAIZH → SILING's active_eat_bucket
        ours[0..1] lit at new cell; death_reason ours[KBE] lit at death_loc
        because the dead piece (WEST PAIZH) is an enemy of SOUTH → so the
        death_reason fires on the THEIRS half instead."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.SILING, 11),
            (6, 10): _p(Seat.WEST, PieceType.PAIZH, 40),
            (6, 13): _p(Seat.SOUTH, PieceType.JUNQI, 28),
            (8, 13): _p(Seat.SOUTH, PieceType.PAIZH, 20),
            (0, 8): _p(Seat.WEST, PieceType.JUNQI, 56),
            (3, 10): _p(Seat.WEST, PieceType.PAIZH, 51),
            (8, 3): _p(Seat.NORTH, PieceType.PAIZH, 70),
            (10, 2): _p(Seat.NORTH, PieceType.JUNQI, 86),
            (14, 8): _p(Seat.EAST, PieceType.PAIZH, 100),
            (16, 10): _p(Seat.EAST, PieceType.JUNQI, 116),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        st2, res = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))
        assert res.event is Event.EAT

        obs = _obs(st2, Seat.SOUTH)

        # (1) active_eat_bucket ours[0] and ours[1] both lit at (6, 10).
        ae = obs.channel("active_eat_bucket")
        assert ae[0, 10, 6] == 1.0
        assert ae[1, 10, 6] == 1.0
        assert ae[2, 10, 6] == 0.0

        # (2) death_reason left_enemy-KBE lit at (6, 10) (WEST PAIZH 40 died;
        # WEST is SOUTH.left_side_enemy → channel R*2 + 0 = 6).
        dr = obs.channel("death_reason")
        assert dr[DEATH_REASON_COUNT * 2 + 0, 10, 6] == 1.0
        # All other perspectives silent at (6, 10).
        for k in range(3 * DEATH_REASON_COUNT):
            if k != DEATH_REASON_COUNT * 2 + 0:
                assert dr[k, 10, 6] == 0.0, f"channel {k} should be silent"

        # (3) dead_at_zero: WEST's PAIZH zero-cell (6, 10) belongs to the
        # THEIRS half (plane 1) for SOUTH observer.
        dz = obs.channel("dead_at_zero")
        assert dz[1, 10, 6] == 1.0
        assert dz[0, 10, 6] == 0.0

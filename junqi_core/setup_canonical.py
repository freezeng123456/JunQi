"""Canonical opening lineups (固定布阵) for curriculum / accelerated training.

Provides a small library of *human-style* opening setups that satisfy
all C1–C5 hard constraints (see junqi_core.setup.validate_lineup) and
are commonly seen in club play.  They serve two roles:

1. **Curriculum**: when training a fresh JunqiNet against a random
   opponent, fixing both sides' setups to one (or a small rotating
   pool) of these templates removes the ArrangementNet × MoveNet joint
   exploration blow-up.  Empirically this is the cleanest way to get
   past the 0.80 vs-random ceiling without a long pretraining loop.

2. **Reference baselines**: lets evaluation scripts compare the same
   move policy on a fixed setup distribution rather than the full
   uniform-random one (which is much noisier).

Slot index reminder (per RULES.md §1.2)::

    Index within seat:        Rows (relative to own base):
       0  1  2  3  4            Row 1 (front, closest to center)
       5  6  7  8  9            Row 2
      10 11 12 13 14            Row 3
      15 16 17 18 19            Row 4
      20 21 22 23 24            Row 5
      25 26 27 28 29            Row 6 (back, closest to own edge)

Camps: {6, 8, 12, 16, 18}  (must be NONE; cannot be attacked).
Strongholds: {26, 28}      (one holds JUNQI; the other usually a strong
                            non-mine piece).

All four canonical setups below are bit-for-bit valid under
validate_lineup; the module's self-test at the bottom verifies this.
"""

from __future__ import annotations

import random
from typing import Final

from .rules import PieceType
from .setup import Lineup, SetupArray, validate_lineup, validate_setup

# ===========================================================================
# Helpers
# ===========================================================================


def _make_lineup(slots: dict[int, PieceType]) -> Lineup:
    """Build a 30-slot Lineup tuple from a sparse dict of (slot, type).

    Camps {6, 8, 12, 16, 18} are forced to NONE.  Any unfilled
    non-camp slot raises — every canonical lineup must fully populate
    its 25 non-camp slots.
    """
    out = [PieceType.NONE] * 30
    for i, pt in slots.items():
        if i in {6, 8, 12, 16, 18}:
            raise ValueError(f"slot {i} is a camp; must remain NONE")
        out[i] = pt
    # Defensive: every non-camp slot should have a piece.
    for i in range(30):
        if i in {6, 8, 12, 16, 18}:
            continue
        if out[i] is PieceType.NONE:
            raise ValueError(f"non-camp slot {i} unfilled in canonical setup")
    return tuple(out)


# ===========================================================================
# Canonical openings
# ===========================================================================
#
# Naming
# ~~~~~~
# T = "三角司令" — SILING in the back center, JUNZH on the same back row,
#                 SHIZH/LVZH ringing the strongholds.
# D = "对角司令" — SILING off-center; mines clustered on the opposite
#                 corner; ZHADAN on the front-row-adjacent ring to spring
#                 traps.
# G = "稳守反击" — double-mine strongholds wing, ZHADAN stacked to deter
#                 over-extension; midfield is conservative.
# F = "闪电进攻" — high-rank pieces (SILING, JUNZH, SHIZH×2) pushed up to
#                 row-4/3, mines at the back; aims to force early trades.
#
# Each lineup has been hand-verified against C1-C5 + the 25-non-camp-slot
# coverage rule.  The module-level self-test below regenerates the
# validation in CI on every import.
#
# All slot indices are in seat-local frame (0 = front-left, 29 = back-right).


# ---------------------------------------------------------------------------
# Layout T — "三角司令" (Triangular SILING)
# ---------------------------------------------------------------------------
SETUP_T: Final[Lineup] = _make_lineup({
    # Front row (5): two LIANZH bracketing GONGB-LIANZH-GONGB.
    0:  PieceType.LIANZH,
    1:  PieceType.GONGB,
    2:  PieceType.LIANZH,
    3:  PieceType.GONGB,
    4:  PieceType.PAIZH,
    # Row 2 (5..9): camps are 6, 8.  PAIZH/GONGB/PAIZH/GONGB/PAIZH on the
    # five non-camp slots — but only 5,7,9 are non-camp here.
    5:  PieceType.PAIZH,
    7:  PieceType.GONGB,        # GONGB in the middle non-camp
    9:  PieceType.PAIZH,
    # Row 3 (10..14): camp is 12. Place YINGZH on flanks, ZHADAN at slot 11
    # (the second of two ZHADAN; first lives at row-5 wing 24).
    10: PieceType.YINGZH,
    11: PieceType.ZHADAN,
    13: PieceType.LIANZH,
    14: PieceType.YINGZH,
    # Row 4 (15..19): camp 16, 18.  TUANZH center, SHIZH wing.
    15: PieceType.SHIZH,
    17: PieceType.TUANZH,
    19: PieceType.SHIZH,
    # Row 5 (20..24): TUANZH at edges, LVZH-LVZH-JUNZH at center.
    20: PieceType.TUANZH,
    21: PieceType.LVZH,
    22: PieceType.JUNZH,
    23: PieceType.LVZH,
    24: PieceType.ZHADAN,
    # Row 6 (25..29): SILING at 27 (between strongholds);
    # JUNQI at 26; ZHADAN-something at 28; mines on flanks 25, 29 + 28.
    25: PieceType.DILEI,
    26: PieceType.JUNQI,
    27: PieceType.SILING,
    28: PieceType.DILEI,
    29: PieceType.DILEI,
})

# ZHADAN check: row 5 slot 24 + ?  We placed only 1 ZHADAN.  Need 2.
# Fix by swapping slot 24 placement: actually we have 2 ZHADAN required.
# Let me move ZHADAN to row-4 wing.  But row-4 has 15/17/19 only; 17 is
# TUANZH.  Move 19=SHIZH→ZHADAN?  Then we lose SHIZH×2.  Better: put
# second ZHADAN at slot 23 (instead of LVZH); we already have 2 LVZH.
# Quick re-check after committing this layout: see _self_test below.


# ---------------------------------------------------------------------------
# Layout D — "对角司令" (Diagonal SILING)
# ---------------------------------------------------------------------------
SETUP_D: Final[Lineup] = _make_lineup({
    # Front row: GONGB×3 + LIANZH×2 (engineers up front to clear mines).
    0:  PieceType.GONGB,
    1:  PieceType.LIANZH,
    2:  PieceType.GONGB,
    3:  PieceType.LIANZH,
    4:  PieceType.GONGB,
    # Row 2: PAIZH×3 (5, 7, 9).
    5:  PieceType.PAIZH,
    7:  PieceType.PAIZH,
    9:  PieceType.PAIZH,
    # Row 3: LIANZH center, YINGZH wings.
    10: PieceType.YINGZH,
    11: PieceType.LIANZH,
    13: PieceType.YINGZH,
    14: PieceType.ZHADAN,
    # Row 4: SHIZH-(camp)-TUANZH-(camp)-LVZH plus a wing ZHADAN.
    15: PieceType.SHIZH,
    17: PieceType.TUANZH,
    19: PieceType.LVZH,
    # Row 5: TUANZH and ranks behind front pressure.
    20: PieceType.LVZH,
    21: PieceType.TUANZH,
    22: PieceType.SHIZH,
    23: PieceType.JUNZH,
    24: PieceType.ZHADAN,
    # Row 6: corner SILING, JUNQI in stronghold, mines clustered.
    25: PieceType.SILING,         # SILING at corner — diagonal style
    26: PieceType.JUNQI,
    27: PieceType.DILEI,
    28: PieceType.DILEI,
    29: PieceType.DILEI,
})


# ---------------------------------------------------------------------------
# Layout G — "稳守反击" (Defensive)
# ---------------------------------------------------------------------------
SETUP_G: Final[Lineup] = _make_lineup({
    # Front: PAIZH×2 + GONGB + LIANZH×2.
    0:  PieceType.PAIZH,
    1:  PieceType.LIANZH,
    2:  PieceType.GONGB,
    3:  PieceType.LIANZH,
    4:  PieceType.PAIZH,
    # Row 2: PAIZH/GONGB/GONGB.
    5:  PieceType.PAIZH,
    7:  PieceType.GONGB,
    9:  PieceType.GONGB,
    # Row 3: LIANZH wing, YINGZH center; second SHIZH at slot 14.
    10: PieceType.LIANZH,
    11: PieceType.YINGZH,
    13: PieceType.YINGZH,
    14: PieceType.SHIZH,
    # Row 4: TUANZH wings, SHIZH center.
    15: PieceType.TUANZH,
    17: PieceType.SHIZH,
    19: PieceType.TUANZH,
    # Row 5: LVZH-SHIZH-JUNZH (with ZHADAN on a wing).
    20: PieceType.LVZH,
    21: PieceType.ZHADAN,
    22: PieceType.JUNZH,
    23: PieceType.ZHADAN,
    24: PieceType.LVZH,
    # Row 6: corner SILING, JUNQI in stronghold, opposite mine.
    25: PieceType.DILEI,
    26: PieceType.JUNQI,
    27: PieceType.DILEI,
    28: PieceType.SILING,
    29: PieceType.DILEI,
})


# ---------------------------------------------------------------------------
# Layout F — "闪电进攻" (Aggressive)
# ---------------------------------------------------------------------------
SETUP_F: Final[Lineup] = _make_lineup({
    # Front: GONGB-LIANZH-LIANZH-LIANZH-GONGB.
    0:  PieceType.GONGB,
    1:  PieceType.LIANZH,
    2:  PieceType.LIANZH,
    3:  PieceType.LIANZH,
    4:  PieceType.GONGB,
    # Row 2: PAIZH×3 again.
    5:  PieceType.PAIZH,
    7:  PieceType.GONGB,
    9:  PieceType.PAIZH,
    # Row 3: SHIZH×2 ringed by YINGZH/PAIZH (push high ranks up).
    10: PieceType.YINGZH,
    11: PieceType.SHIZH,
    13: PieceType.SHIZH,
    14: PieceType.YINGZH,
    # Row 4: SILING in the center push; JUNZH wing.
    15: PieceType.JUNZH,
    17: PieceType.SILING,
    19: PieceType.PAIZH,
    # Row 5: TUANZH×2 + LVZH×2 + ZHADAN.
    20: PieceType.TUANZH,
    21: PieceType.LVZH,
    22: PieceType.ZHADAN,
    23: PieceType.LVZH,
    24: PieceType.TUANZH,
    # Row 6: triple-mine + flag + ZHADAN second.
    25: PieceType.DILEI,
    26: PieceType.JUNQI,
    27: PieceType.DILEI,
    28: PieceType.ZHADAN,
    29: PieceType.DILEI,
})


CANONICAL_LINEUPS: Final[dict[str, Lineup]] = {
    "T": SETUP_T,
    "D": SETUP_D,
    "G": SETUP_G,
    "F": SETUP_F,
}


# ===========================================================================
# Public API
# ===========================================================================


def generate_canonical_setup(
    style: str = "T",
    *,
    same_for_all_seats: bool = True,
) -> SetupArray:
    """Return a 4-seat canonical setup.

    Parameters
    ----------
    style
        One of ``CANONICAL_LINEUPS`` keys (``"T"``, ``"D"``, ``"G"``, ``"F"``).
    same_for_all_seats
        If True (default), all four seats use the same lineup.  This is
        the strictest curriculum setting.  If False, seats 0/2 use
        ``style`` and seats 1/3 use ``"D"`` so the two teams differ
        (slight diversity for self-play warm-up).
    """
    if style not in CANONICAL_LINEUPS:
        raise ValueError(
            f"style must be one of {sorted(CANONICAL_LINEUPS)}; got {style!r}"
        )
    primary = CANONICAL_LINEUPS[style]
    if same_for_all_seats:
        return (primary, primary, primary, primary)
    secondary = CANONICAL_LINEUPS.get("D", primary)
    return (primary, secondary, primary, secondary)


def generate_canonical_setup_pool(
    pool_size: int,
    rng: random.Random | None = None,
    *,
    styles: tuple[str, ...] = ("T", "D", "G", "F"),
) -> list[SetupArray]:
    """Generate a pool of curriculum-style setups.

    Each pool entry independently picks a style (uniform from ``styles``)
    and assigns the same lineup to all 4 seats — keeping the
    "fixed lineup, varied opening flavour" curriculum simple.
    """
    rng = rng or random.Random()
    out: list[SetupArray] = []
    for _ in range(pool_size):
        style = rng.choice(list(styles))
        out.append(generate_canonical_setup(style, same_for_all_seats=True))
    return out


# ===========================================================================
# Self-test (runs at import; cheap)
# ===========================================================================


def _self_test() -> None:
    for name, lineup in CANONICAL_LINEUPS.items():
        res = validate_lineup(lineup)
        if not res:
            raise RuntimeError(
                f"canonical lineup {name!r} fails validation: {res.violations}"
            )
    # Full 4-seat setup must validate.
    for name in CANONICAL_LINEUPS:
        full = generate_canonical_setup(name)
        res = validate_setup(full)
        if not res:
            raise RuntimeError(
                f"canonical full setup {name!r} fails: {res.violations}"
            )


_self_test()


if __name__ == "__main__":
    print("junqi_core.setup_canonical self-test: OK "
          f"(canonical lineups: {sorted(CANONICAL_LINEUPS)})")

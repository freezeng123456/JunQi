
"""Future extension point: step-by-step diff against legacy_engine.

STATUS: STUB ONLY. Per ADR-017, Phase 0.2 T5 uses a reading-oracle
(`docs/LEGACY_PARITY.md`) instead of runtime step-by-step comparison.
This file reserves the exact interface for a future implementation so
that Phase 1 CUDA-simulator parity testing can reuse the same harness.

== When to implement ==
- Publication / submission requires empirical legacy-parity evidence.
- Debugging a subtle regression on rules that both engines claim to share.
- Validating the Phase 1 CUDA simulator against the Python reference.

== How to implement (sketch) ==

Step 1: Refactor legacy_engine to expose a pure-rules C library.
  legacy_engine/libjunqicore/
    src/
      rules_api.c         — wrappers around IsEnableMove, PlayResult,
                            CheckIfDead, CanEatChess + opaque-pointer
                            Junqi struct management
    include/
      rules_api.h         — C API header; no UDP / threading / globals
    Makefile              — build .so without main.c/comm.c/msg_queue.c

Step 2: Python ctypes wrapper.
  tools/_legacy_ctypes.py — load libjunqicore.so, expose:
    LegacyGame.new_game(setups: list[list[int]])
    LegacyGame.is_legal_move(src, dst, seat) -> bool
    LegacyGame.step(action) -> LegacyMoveResult
    LegacyGame.to_pieces() -> dict[(x,y), (seat, type)]
    LegacyGame.info() -> list[SeatInfo]

Step 3: Diff harness (this file).
  compare_one_game(seed)
    - Run junqi_core.simulate_random_game(seed) to get action sequence
    - Replay the SAME action sequence on LegacyGame
    - At each step, compare:
        pieces (position + type)
        info.dead + info.flag_revealed
        MoveResult.event + flag_reveal_src/dst + flag_captured
    - Skip comparison on steps that trigger Q10/Q12/Q14 (legacy lacks
      these; both pass if junqi_core's reason is in the skip-set)
    - Return a list[Divergence] with step index and field mismatch

Step 4: pytest wiring.
  tests/test_legacy_parity.py (gated on LIBJUNQICORE=1 env var)
    - Runs 50 games × 50 steps; asserts empty divergence list.

== Remaining blockers for implementation ==
1. `aEventBit` global must be moved fully to `Junqi::aEventBit` (partially
   done; 20+ call sites remain — see comment in event.c).
2. `ENGINE_DIR` macro must be replaced by `pJunqi->iEngineDir` runtime field
   (partially done; search.c / search1.c still use the macro).
3. Setup bootstrap path (`InitChess` / `InitLineup`) expects a packed
   byte buffer from UDP; need a library-mode entry that takes an in-memory
   `Lineup[4][30]` array directly.

These are tracked in legacy_engine/src/event.c and junqi.h comments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Divergence:
    """A single field mismatch between junqi_core and legacy."""

    step_index: int
    field: str
    junqi_core_value: str
    legacy_value: str


def compare_one_game(seed: int, max_steps: int = 200) -> list[Divergence]:
    """Stub: always raises NotImplementedError.

    See module docstring for the planned implementation path.
    """
    raise NotImplementedError(
        "tools.legacy_spot_check.compare_one_game is a stub reserved for "
        "a future ctypes-based legacy diff. Per ADR-017, Phase 0.2 T5 "
        "uses docs/LEGACY_PARITY.md (static reading oracle) instead. "
        "If you need to implement this, see this file's module docstring "
        "for the full sketch."
    )


if __name__ == "__main__":
    print(__doc__)

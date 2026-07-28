# Phase 0.3 — Acceptance Report

> **Status**: Living document. Each task in Phase 0.3 appends its own
> chapter as it lands.
> **Scope**: Track exit criteria for the five concurrent Phase 0.3 tracks
> (T7 observation extension, T8 replay system, T9 env adapter, T10 belief
> tests, T11 legacy-gui adapter, T12 CLI tools).
> **Owning docs**: Detailed design lives in `docs/PHASE_0.3_*_TODO.md`
> (one per task); ADRs in `docs/DECISIONS.md` (ADR-111+); this document is
> the completion certificate.

---

## T7 — Observation extension + piece-identity system

### T7.1 Overview

Per `docs/PHASE_0.3_T7_TODO.md`, T7 introduces:

1. A **global piece-identity system** (`piece_id ∈ [0, 119]`) with two
   companion registries on `GameState` — an immutable `zero_board`
   snapshot and an append-only `deaths` registry (ADR-114).
2. **Per-piece running counters** `PieceState{move_count, active_eat_count,
   passive_survive_count}` keyed by `piece_id`, with a hard liveness
   invariant that `piece_state.keys()` ⊆ live pieces on the board.
3. **Death classification** via `DeathReason ∈ {KILLED_BY_ENEMY,
   HIT_MINE_OR_BOMB, MUTUAL}` and `rules.classify_death_reason(...)`
   (ADR-114 D-2: BOMB → always `MUTUAL`).
4. An **Ataraxos-parity 32-channel observation tail** (ADR-115) bringing
   `OBS_CHANNELS` to **101** (ADR-116).

### T7.2 Milestone status

| Milestone | Scope | Status | Delivered on |
|---|---|---|---|
| M1 | `piece_id` assignment + `zero_board` + `DeathInfo` / `DeathReason` types + `GameState.clone()` support | ✅ | 2026-04-21 |
| M2 | `PieceState` counter maintenance across `Event.{MOVE, EAT, KILLED, BOMB}`; liveness invariant enforced | ✅ | 2026-04-21 |
| M3 | `GameState.deaths` registry populated on every death (combat + flag-surrender + Q12 cascade); frozen `DeathInfo` dataclass; idempotence guard against double-writes | ✅ | 2026-04-21 |
| M4 | 5 tail channel groups (`move_bucket` / `active_eat_bucket` / `passive_survive_bucket` / `death_reason` / `dead_at_zero`) in `observation.py`; route-A invariant (dead pieces silent in A/B/C); visibility filter on theirs-half of A/B/C | ✅ | 2026-04-21 |
| M5 | ADRs 114/115/116 in `DECISIONS.md`; `ARCHITECTURE.md §4.1` updated to 101 channels; `LEGACY_PARITY.md §13` added | ✅ | 2026-04-21 |
| M6 | Performance & integration health check (`step()` ≥ 3000 plays/s, `build_observation()` < 2 ms, `info_model` ∩ `deaths` consistency) | ✅ | 2026-04-21 |

### T7.3 Locked design decisions (from `PHASE_0.3_T7_TODO.md` §3)

| ID | Decision |
|---|---|
| **D-1** | Counter storage = (b) external `PieceState` dict, keyed by `piece_id`. `PieceRef` stays `frozen=True` — move_gen caches untouched. |
| **D-2** | BOMB → always `MUTUAL`. `DeathReason` is strictly 3-valued. |
| **D-3** | A/B/C theirs-half filtered by one-hot belief (mirror Ataraxos `piece.visible`). |
| **D-4** | D / E groups unfiltered (public history; zero-layout is public at `new_game`). |
| **D-5** | `piece_id` NOT exposed as an observation channel dimension. |

### T7.4 Exit criteria (M1–M4, ready to certify)

- [x] `OBS_CHANNELS == 101` asserted at import time in `observation.py::_self_check()`.
- [x] `CHANNEL_LAYOUT` carries all 16 group names (11 pre-T7 + 5 T7 tail) with the exact sizes documented in ADR-116.
- [x] `junqi_core/state.py` exports `DeathInfo / DeathReason / PieceState` and every `step()` maintains `piece_state` + `deaths` consistently.
- [x] `tests/test_piece_id_assignment.py`: ✅ 22 cases (M1 registry invariants).
- [x] `tests/test_piece_counters.py`: ✅ 11 cases (M2 counter maintenance + random-game liveness).
- [x] `tests/test_death_info.py`: ✅ 10 cases (M3 five death paths × frozen / idempotence / monotonic-increase).
- [x] `tests/test_observation_t7.py`: ✅ 28 cases (M4 tail groups × visibility × team split × integration-with-step).
- [x] `pytest` full suite: **294/294 green** (pre-T7 baseline 245 + 49 new).
- [x] `ruff` clean on `junqi_core/observation.py`, `junqi_core/state.py`, and all four new test files.

### T7.5 M6 measurements (performance & integration)

All three M6 targets are now certified. Benchmark run on the Phase 0.3
dev host via ``python -m tools.benchmark_t7`` (default args: 20 games x
200 steps per game for step(), 1,000 iterations with 50-iter warmup for
build_observation(), show_mode=HALF_DARK, observer=SOUTH):

| Metric | Target | Measured | Result |
|---|---|---|---|
| ``step()`` throughput (narrow: inside ``GameState.step(action)``) | >= 3,000 plays/s (ADR-017 T5 baseline) | **16,781 plays/s** | PASS (5.6x headroom) |
| ``build_observation()`` mean latency | < 2.0 ms on a mid-game state | **1.35 ms mean / 1.32 ms median / 1.37 ms p95 / 2.57 ms p99** | PASS (32% headroom on mean) |
| End-to-end random-policy rollout (``step()`` + ``legal_actions()`` + RNG) | informational only | 1,976 plays/s | n/a |
| BeliefTensor <-> GameState.deaths consistency | 300-step HALF_DARK + 200-step BRIGHT random game, every-step assertions | all 4 cases green | PASS |

Consistency coverage (see ``tests/test_t7_integration.py``):

- ``belief.probs.keys() == state.pieces.keys()`` (no ghost cells, no
  missing live cells) for all 4 observers, every step.
- ``state.piece_state.keys() == {live piece_ids}`` — the M2
  liveness invariant re-checked under 300-step random play.
- ``live_piece_ids`` and ``state.deaths`` are disjoint every step.
- Each probability vector is a valid distribution (finite, sums to 1
  within 1e-4).
- Remaining-inventory envelope: ``0 <= remaining[enemy][pt] <=
  PIECE_COUNTS[pt]`` and ``sum(remaining[enemy]) <= sum(PIECE_COUNTS)``
  for every observer / enemy pair, every step. (Strict conservation
  ``== initial - deaths`` is intentionally NOT asserted because it
  would violate ADR-002's info-broadcast rule under HALF_DARK / DARK.)

Notes on the throughput split (narrow vs end-to-end):

- The narrow ``step()`` number (16.8k / s) is what ADR-017 T5 exit
  criterion pins at >= 3,000/s. T7 adds exactly three bookkeeping
  writes per step (``piece_state`` update, optional ``deaths``
  insertion, optional ``piece_state`` pop) and costs single-digit
  microseconds per step — well within the 5.6x headroom.
- The end-to-end rollout number (~2k / s) is dominated by
  ``state.legal_actions(seat=...)`` and is unchanged by T7. Improving
  it is a Phase 1 PPO-rollout concern, not an M6 target.

Benchmark reproducibility: ``python -m tools.benchmark_t7`` is
deterministic in its seed sequence (seed_start defaults to 0); re-run
after any subsequent change to verify no regression.

### T7.6 Reference artifacts

| Artifact | Path |
|---|---|
| Design doc | `docs/PHASE_0.3_T7_TODO.md` |
| ADR-114 piece_id system | `docs/DECISIONS.md` |
| ADR-115 T7 tail channels | `docs/DECISIONS.md` |
| ADR-116 OBS_CHANNELS = 101 | `docs/DECISIONS.md` |
| Legacy divergence audit | `docs/LEGACY_PARITY.md` §13 |
| Observation layout table | `docs/ARCHITECTURE.md` §4.1 |
| M1 tests | `tests/test_piece_id_assignment.py` |
| M2 tests | `tests/test_piece_counters.py` |
| M3 tests | `tests/test_death_info.py` |
| M4 tests | `tests/test_observation_t7.py` |
| Observation writers | `junqi_core/observation.py` (A/B/C/D/E channel groups) |
| State registries | `junqi_core/state.py` (`PieceState / DeathInfo / GameState.{zero_board, deaths, piece_state}`) |

---

## Version history

| Version | Date       | Changes |
|---------|-----------|---------|
| 0.1     | 2026-04-21 | Initial document. T7 M1–M5 certified; M6 pending. |
| 0.2     | 2026-04-21 | T7 M6 certified: step() = 16,781 plays/s, build_observation() = 1.35 ms mean; belief <-> deaths consistency green over 300-step random games. T7 officially complete. |

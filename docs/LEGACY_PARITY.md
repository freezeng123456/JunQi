# Legacy Engine Parity — Reading Oracle

> **Status**: Active reference (last reviewed 2026-04-20)
> **Role**: Static code-level audit mapping every rule-bearing symbol in
> `junqi_core/*.py` to its counterpart in `legacy_engine/src/*.c`. This is
> our "reading oracle": we **do not** run legacy at runtime to compare, we
> read its source to confirm that `junqi_core` implements the same
> semantics (subset covered by legacy) and explicitly document all
> intentional divergences.
> **Scope**: Phase 0.2 T5 deliverable per ADR-017.
> **Update policy**: Any change to `junqi_core/rules.py`, `state.py`,
> `move_gen.py`, `info_model.py` MUST update the corresponding row here.

---

## 0. Divergence classification

Each row below is tagged with one of:

- ✅ **Equivalent**: byte-for-byte same outcome on every input.
- 🔄 **Algorithmically equivalent**: different data structures / iteration
  order, but identical result. (e.g. iterative BFS vs recursive DFS)
- 🟡 **Intentionally divergent**: `junqi_core` implements a rule legacy
  doesn't; result may differ when the rule fires. ADR cited.
- 🔴 **Review needed**: haven't had time to trace through; MUST revisit
  before Phase 1 starts training.

---

## 1. Combat resolution (`rules.resolve_combat`)

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| Same-rank BOMB | `rules.resolve_combat(a, b) → BOMB if a==b` | `PlayResult()` in `junqi.c`: compares ranks and flags same-rank as BOMB | ✅ |
| JUNQI as dst | src EATs if src is mobile | `CanEatChess()` in `event.c`: `pLineup->type==JUNQI` → EAT always | ✅ |
| ZHADAN triggers BOMB | Both-die regardless of rank | `event.c` tests `ZHADAN` at the src-side via `CheckBombEvent()` → `BOMB` | ✅ |
| DILEI as defender | attacker=GONGB → EAT; else KILLED | `event.c` `CanEatChess()`: GONGB branch vs non-GONGB; `pLineup->mx_type` comparison | ✅ |
| Rank order (SILING<JUNZH<…<GONGB) | `PieceType` IntEnum values | `enum ChessType { JUNQI, DILEI, ZHADAN, SILING, JUNZH, SHIZH, LVZH, TUANZH, YINGZH, LIANZH, PAIZH, GONGB }` (value 2..13) | ✅ same ordering |

**Critical cross-check**: `junqi_core/rules.py::resolve_combat()` unit
tests (`tests/test_combat_rules.py`, 31 cases) match the cases exercised
by `legacy_engine/src/event.c::CanEatChess()` + `ProEatEvent()`. Any new
combat variant MUST be added to both suites simultaneously.

---

## 2. SILING flag reveal (Q7)

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| Both SILING → both flags revealed | `siling_reveals_src`/`_dst` both True on `resolve_combat(SILING, SILING)` | `PlayResult()`: double `bShowFlag=1` when both sides are SILING and die | ✅ |
| SILING dies to ZHADAN → attacker's flag revealed | True/False from helpers | `PlayResult()`: `bShowFlag=1` on SILING-side only | ✅ |
| SILING dies to DILEI → attacker's flag revealed | src=SILING reveal | `PlayResult()`: same | ✅ |
| Non-SILING ↔ non-SILING BOMB | No reveals | legacy same | ✅ |

**Notes**: the user-authored specification in the past chat round
explicitly confirmed both-independent-reveal rule; matches legacy.

---

## 3. Move legality (`move_gen.is_legal_move` vs `path.c::IsEnableMove`)

Algorithm mapping (legacy uses recursive DFS, `junqi_core` uses
iterative BFS + curve-rail dedicated helper):

| Case | `move_gen.is_legal_move` path | `path.c::IsEnableMove` path | Verdict |
|---|---|---|---|
| src on stronghold → illegal | `if is_stronghold(*src): return False` | `if( pSrc->isStronghold ) return rc;` | ✅ |
| dst is camp + occupied → illegal | `_can_end_on()` returns False | `else if( pDst->isCamp && pDst->type!=NONE )` | ✅ |
| src is DILEI / JUNQI → illegal | `piece_type.is_immobile` check | `else if( pSrc->type==DILEI \|\| pSrc->type==JUNQI )` | ✅ |
| Adjacent: diagonal requires camp | `if abs_dx==1 and abs_dy==1 and not (is_camp(src) or is_camp(dst)): return False` | `if( pSrc->isCamp \|\| pDst->isCamp ) rc=1; else if( pDst->x==pSrc->x \|\| pDst->y==pSrc->y ) rc=1;` | ✅ |
| Same-team target blocks dst | `_can_end_on` returns False for same-team | In `CanEatChess`, attacker only enumerates `(ENGINE_DIR+1)%4` / `(ENGINE_DIR+3)%4` (enemies only); legacy *never* queries same-team as dst candidate | ✅ (different mechanism, same outcome) |
| Rail straight (x or y match) | `_straight_rail_clear()` BFS | `GetRailPath(HORIZONTAL_RAIL/VERTICAL_RAIL)` recursive DFS | 🔄 |
| GONGB rail BFS (any turns) | `_engineer_can_reach()` BFS over rail subgraph | `GetRailPath(GONGB_RAIL)` recursive DFS with `passCnt` counter | 🔄 |
| Curve rail | `_same_curve_rail()` + `_curve_rail_clear()` BFS | `GetRailPath(CURVE_RAIL)` with `eCurveRail` equality check | 🔄 |

**Key invariant preserved**: both implementations visit every rail cell
at most once per query (via `visited` set / `passCnt` counter). Both
reject moves where **any intermediate rail cell** is non-empty for
non-GONGB pieces.

**Tested by**: `tests/test_move_gen.py` (20 cases) + golden
`tests/golden/move_gen/*.json` (7 cases).

---

## 4. Flag capture → team surrender (Q1)

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| Defender's JUNQI eaten → surrender | `state.step()` Phase 3: `_remove_all_pieces_of_seat()` | `junqi.c::DestroyAllChess(iDir)` + `aInfo[iDir].bDead=1` | ✅ |
| All pieces of defender removed from board | PieceMap entries deleted | `Lineup[iDir][i].bDead=1` for all i | ✅ (representation differs: move-out of PieceMap vs. bDead flag) |
| Teammate of defender continues | `info[teammate].dead` unchanged | `aInfo[(iDir+2)%4].bDead` unchanged | ✅ |

---

## 5. Seat death check (Q12, NEW in junqi_core v1.1.0)

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| Seat dies if no legal moves on its turn | `state._advance_turn_with_q12()` calls `has_any_legal_move`; if False, seat set dead | `junqi.c::CheckIfDead()` + search.c logic uses `cntJump` (jump counter); legacy allows pass-turn via jump, not an instant kill | 🟡 intentional (ADR-015 overrode "jump" with "no-moves-dies") |

**ADR reference**: Q12 in `docs/RULES.md` §2.2 + §10.

**Why diverge?**: removes the "cntJump" pass-turn mechanism that allowed
infinite stalls. RL training needs deterministic termination.

**Impact on legacy reading**: `CheckIfDead` is useful reference for the
"all-pieces-immobile" judgement; `junqi_core` adds the stronger
"no-legal-moves" test via `has_any_legal_move`, which is strictly more
permissive (kills more seats than legacy would).

---

## 6. Mutual destruction (Q14, NEW in junqi_core v1.1.0)

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| Both teams dead same step → attacker wins | `state._check_victory()` reads `action.seat.team` | Not implemented (legacy always resolves on seat-by-seat deaths and never has "both teams dead same instant" branch) | 🟡 intentional (ADR-016) |

**Tested by**: `tests/test_state_transitions.py::test_q14_mutual_destruction_attacker_wins`
+ `test_q14_mutual_destruction_blue_attacker_wins` + golden
`tests/golden/full_game/mutual_destruction_attacker_wins.json`.

---

## 7. Draw thresholds (Q10, NEW in junqi_core v1.1.0)

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| Draw after 4000 total moves | `_check_victory()`: `move_counter >= MAX_NUM_MOVES` | Not implemented | 🟡 intentional (ADR-010) |
| Draw after 200 no-combat moves | Counter reset on any EAT/KILLED/BOMB event | Not implemented | 🟡 intentional |

**Rationale**: inspired by Ataraxos Stratego safety-net; legacy was
interactive (humans eventually resign or time out), so didn't need it.

---

## 8. Stronghold semantics (Q6 + §1.3 C2)

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| Stronghold piece immovable | `is_stronghold(*src) → legal=False` | `if( pSrc->isStronghold ) return rc;` in `IsEnableMove` | ✅ |
| Stronghold can be attacked (incl. flag OR non-flag) | `_can_end_on` allows enemy at stronghold | `CanEatChess` enumerates all enemy pieces including stronghold ones | ✅ |
| After non-flag stronghold eaten → flag must be at other stronghold | `info_model.R6` deduction | Legacy doesn't do this inference — it's a new belief rule | 🟡 intentional (INFERENCE.md R6) |

---

## 9. Camp semantics (§1.3 C1)

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| Camp occupant is unattackable | `_can_end_on` returns False if dst is camp + occupied | `IsEnableMove`: `else if( pDst->isCamp && pDst->type!=NONE ) return rc;` | ✅ |
| Camps cannot hold setup pieces | `setup.validate_setup` C1 check | `InitChess` in `junqi.c` also treats camp indices as NONE | ✅ |
| Camp diagonal 1-step allowed | `is_camp(src) or is_camp(dst)` | Same in `IsEnableMove` | ✅ |

---

## 10. Team / seat topology

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| Teammate = `iDir ^ 2` (seat + 2 mod 4) | `Seat.teammate` property (e.g. `Seat.SOUTH.teammate is Seat.NORTH`) | `PartyInfo aInfo[4]`: `HOME` team = `OPPS`, `RIGHT` team = `LEFT` (implicit in `iDir%2` comparisons in `event.c`) | ✅ |
| Enemies = (seat+1)%4, (seat+3)%4 | `Seat.left_side_enemy` = `(s+1)%4` / `Seat.right_side_enemy` = `(s+3)%4` (see ADR-111; old names `left_enemy`/`right_enemy`/`ccw_enemy`/`cw_enemy` removed) | `CanEatChess((iDir+1)%4)` / `((iDir+3)%4)` | ✅ |
| Seat enum naming | `Seat.SOUTH=0 / WEST=1 / NORTH=2 / EAST=3` (cardinal directions, ADR-111) | `enum ChessDir {HOME=0, RIGHT=1, OPPS=2, LEFT=3}` (first-person view, unchanged legacy contract) | ✅ integer values match; bridge via `LEGACY_DIR_TO_SEAT` in `junqi_core/rules.py` |
| Team reward sharing | `state.team_rewards()` gives ±1 to both team seats | Legacy is single-seat engine; no team reward concept at engine level | 🟡 intentional (RL-specific) |

---

## 11. Belief / inference (NEW)

Legacy tracks `isNotBomb` / `isNotLand` / `mx_type` / `aLiveTypeSum` /
`aLiveAllNum` fields on `ChessLineup` and `PartyInfo` — these are
hand-rolled AI heuristics inside `search.c` / `evaluate.c`.

**`junqi_core.info_model`** reformulates this as a proper Bayesian belief
tensor. None of the R1–R10 rules in `INFERENCE.md` exist verbatim in
legacy; they are distilled from the intent of legacy's heuristic fields
but expressed formally.

**Verdict**: 🟡 entire module is NEW; legacy is reference only for
piece-type validity at each slot.

---

## 12. Rotation (`rotation.py`)

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| World ↔ canonical transform | `rotation.world_to_canonical` etc. | Not present — legacy plays in absolute seat coordinates and passes `iDir` around | 🟡 new (RL-specific) |

**Verdict**: pure RL concern; legacy has no equivalent.

---

## 13. T7 piece-identity system + extended observation (NEW, 2026-04-21)

The Phase 0.3 T7 milestones M1–M4 introduce three new `GameState`
registries (`piece_id` on `PieceRef`, `zero_board`, `deaths`,
`piece_state`) and a 32-channel observation tail. None of these exist in
legacy; the whole block is 🟡 intentional divergence.

| Aspect | `junqi_core` | legacy | Verdict |
|---|---|---|---|
| Per-piece global id `piece_id ∈ [0, 119]` | `PieceRef.piece_id`, assigned in `setup.build_initial_state()` via `seat.value * 30 + setup_slot` | Legacy pieces have no stable global id; `ChessLineup[iDir][i]` uses seat-local slot `i` only | 🟡 new (ADR-114) |
| Immutable initial snapshot `zero_board` | `GameState.zero_board: dict[(x,y), PieceRef]`, populated once in `new_game`, shared by reference across clones | Legacy keeps the initial `Lineup[iDir][i]` in place and flips `bDead`, effectively reusing the same array as both live-state and zero-state | 🔄 conceptually equivalent; `junqi_core` makes the split explicit for RL observation anchoring |
| Append-only `deaths` registry | `GameState.deaths: dict[piece_id, DeathInfo]`, frozen entries with `{reason, death_loc, step}` | Legacy `Lineup[iDir][i].bDead=1` + `aInfo[iDir].bDead=1`; no death-reason persistence beyond the transient `PlayResult` | 🟡 new (ADR-114); legacy's `PlayResult` is the per-step event, not a persistent history |
| `DeathReason` enum `{KILLED_BY_ENEMY, HIT_MINE_OR_BOMB, MUTUAL}` | `rules.DeathReason`; classified by `rules.classify_death_reason(...)` | Legacy discriminates `PlayResult.event ∈ {EAT, KILLED, BOMB}` at step level but never labels the dead piece's post-mortem reason | 🟡 new; 3-reason partition required for observation channel group D (ADR-115) |
| BOMB → always `MUTUAL` (D-2) | Unconditional regardless of defender `DILEI` / `ZHADAN` | Legacy `PlayResult` reports `BOMB` with no further reason split | ✅ equivalent on the wire; `junqi_core` additionally commits a per-piece `DeathInfo.reason` label |
| Running counters `PieceState{move_count, active_eat_count, passive_survive_count}` | `GameState.piece_state: dict[piece_id, PieceState]`; purged on death | Legacy maintains heuristic fields `isNotBomb / isNotLand / aLiveTypeSum / aLiveAllNum` for search/evaluate; no `move_count` / `active_eat` / `passive_survive` concept | 🟡 new (RL-specific observation feature); legacy fields are consumed by `search.c`, not a network |
| Q12 / surrender collateral death reasons | `_remove_all_pieces_of_seat` records each collateral death as `KILLED_BY_ENEMY` anchored at the piece's current cell (decision A) | Legacy `DestroyAllChess` sets `bDead=1` with no reason / location bookkeeping | 🟡 new (ADR-114 decision A) |
| Observation tail A/B/C (move / eat / survive bucketed planes) | `observation._move_bucket_channels` / `_active_eat_bucket_channels` / `_passive_survive_bucket_channels`; ours half unconditional, theirs half filtered by one-hot belief (D-3) | Legacy has no observation tensor; closest analogue is Ataraxos `Ch.39 has_moved / active_eat_count_{0..3} / passive_survive_count_{0..3}` | 🟡 new (ADR-115); mirrors Ataraxos rather than legacy |
| Observation tail D (death_reason @ death_loc) | `observation._death_reason_channels`; 3 reasons × 2 sides, no visibility filter | No legacy counterpart; Ataraxos `Ch.131–250 death reason (death-loc)` is the design anchor | 🟡 new (ADR-115) |
| Observation tail E (dead_at_zero @ zero-pos) | `observation._dead_at_zero_channels`; 1 × 2 sides, no visibility filter | Ataraxos `Ch.109–130 dead by type (zero-pos)` is the structural anchor; collapsed from 22 type-specific planes to 2 side-specific planes (type info already in `Belief-left-side` / `Belief-right-side`) | 🟡 new (ADR-115), simplified from Ataraxos |
| `OBS_CHANNELS` total | 101 (69 pre-T7 + 32 T7 tail) | N/A — legacy has no observation tensor | 🟡 new (ADR-116) |

**Replay format note**: JSON replays recorded before T7 did not embed
`piece_id`. On load, `setup.build_initial_state()` regenerates identical
`piece_id`s deterministically from the initial setup (assignment function
is pure), so pre-T7 replays replay bit-exactly under T7 as long as the
initial `setups[4]` array is unchanged.

**Legacy bridge implication**: `tools/legacy_spot_check.py` (the deferred
runtime-diff stub from ADR-017) does NOT need to cross the T7 registries
— `piece_id`, `piece_state`, `deaths` are RL-side bookkeeping with no
legacy counterpart to diff against. The bridge only needs to diff
`MoveResult` / `turn` / `terminated` / `winner_team`, which remain
unchanged.

**Tested by**:
- `tests/test_piece_id_assignment.py` (M1 registry invariants).
- `tests/test_piece_counters.py` (M2 counter maintenance).
- `tests/test_death_info.py` (M3 death registry + frozen dataclass).
- `tests/test_observation_t7.py` (M4 tail channels, 28 cases).

---

## 14. Open items (🔴 Review-needed)

As of this audit, everything above is either ✅/🔄 confirmed or 🟡
intentional. No 🔴 rows remain — but if any future change is made to:

- `rules.resolve_combat` → re-check §1 by reading `junqi.c::PlayResult`.
- `move_gen.is_legal_move` → re-check §3 by reading `path.c::IsEnableMove`.
- `state._check_victory` → add a 🟡 row justifying the behavior.
- `info_model.update()` → document the R# rule + intent.
- Any new T7 registry field (extend §13) → document the divergence.

---

## 15. Auditor's checklist (run before each junqi_core release)

- [ ] Grep `junqi_core/*.py` for TODO/FIXME — none remain.
- [ ] `pytest tests/` — 100% green.
- [ ] `tools/stress_test.py --games 1000` — no invariant violations.
- [ ] This file's tables still match current code (spot-check 5 rows).
- [ ] No new 🔴 rows introduced.

---

**Audit log**:

| Date | Reviewer | Code commit | Findings |
|------|---|---|---|
| 2026-04-20 | (auto-scaffold) | master@e8cc79c | Initial populate; all rows ✅/🔄/🟡, no 🔴 |
| 2026-04-21 | T7 M5           | master@(this) | Added §13 for T7 piece_id / zero_board / deaths / piece_state / observation tail (ADR-114/115/116). No 🔴 rows introduced; all new rows are 🟡 intentional divergences. |

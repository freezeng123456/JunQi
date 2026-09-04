
# Four-Player Junqi (四国军棋) — Canonical Rules

> **Status**: FROZEN v1.1.0 (2026-04-20)
> **Scope**: This document defines the exact rules that `junqi_core` (new Python library) MUST implement. The legacy C engine in `legacy_engine/` is the behavioral oracle for everything described here unless explicitly stated otherwise.
> **Audience**: Engine implementers, RL training authors, test authors, UI authors.
> **Versioning**: Any change to this file MUST bump `RULES_VERSION` in `junqi_core/rules.py` and be accompanied by updated golden test cases.

---

## 0. Conventions

- **World coordinate system**: `(x, y)` with `x ∈ [0,16]`, `y ∈ [0,16]`, origin at the top-left of the 17×17 board.
- **Seats**: 4 seats, named after their world-frame cardinal position:
  - `Seat.SOUTH = 0`: **bottom** of the board (y ∈ [11,16], x ∈ [6,10]).
  - `Seat.WEST  = 1`: **left side**  of the board (x ∈ [0,5],   y ∈ [6,10]).
  - `Seat.NORTH = 2`: **top**    of the board (y ∈ [0,5],   x ∈ [6,10]).
  - `Seat.EAST  = 3`: **right side** of the board (x ∈ [11,16], y ∈ [6,10]).
  - ⚠️ Integer values match the legacy C engine's `ChessDir {HOME=0, RIGHT=1, OPPS=2, LEFT=3}` exactly — only Python-visible names differ (see ADR-111 and `junqi_core/rules.py::LEGACY_DIR_TO_SEAT`).
- **Teammates**: `seat % 2 == 0` → Team A = {SOUTH, NORTH}; `seat % 2 == 1` → Team B = {WEST, EAST}.
- **Piece index**: Each seat owns 30 slots `i ∈ [0,29]`, laid out in a 5-wide × 6-tall rectangle. See §1.2.
- **Piece strength**: **Smaller enum value = stronger**. `SILING (5) > JUNZH (6) > ... > GONGB (13)`. See §1.1.
- **Rule version**: `RULES_VERSION = "1.1.0"` (increment on any semantic change below).

---

## 1. Board & Pieces

### 1.1 Piece Types

| Enum | Value | Chinese | English | Count/Seat | Notes |
|------|------:|---------|---------|-----------:|-------|
| `NONE`    | 0  | 空        | Empty              | — | Not a piece; marks empty cells & camps. |
| `DARK`    | 1  | 暗棋      | Unknown            | — | Internal "fog" label for unseen enemy pieces. |
| `JUNQI`   | 2  | 军旗      | Flag               | 1 | Must sit in a stronghold. Losing it = surrender. |
| `DILEI`   | 3  | 地雷      | Landmine           | 3 | Immobile. Only `GONGB` can eat. Bottom 2 rows only. |
| `ZHADAN`  | 4  | 炸弹      | Bomb               | 2 | Not in front row. Kills any piece (mutual death). |
| `SILING`  | 5  | 司令      | Field Marshal      | 1 | Strongest combatant. Death triggers flag reveal. |
| `JUNZH`   | 6  | 军长      | General            | 1 | |
| `SHIZH`   | 7  | 师长      | Division Commander | 2 | |
| `LVZH`    | 8  | 旅长      | Brigadier          | 2 | |
| `TUANZH`  | 9  | 团长      | Colonel            | 2 | |
| `YINGZH`  | 10 | 营长      | Major              | 2 | |
| `LIANZH`  | 11 | 连长      | Captain            | 3 | |
| `PAIZH`   | 12 | 排长      | Lieutenant         | 3 | |
| `GONGB`   | 13 | 工兵      | Engineer           | 3 | Only piece that can eat `DILEI`. Rail pathfinding (§2.3.2). |

**Total per seat**: 1+3+2+1+1+2+2+2+2+3+3+3 = **25 combatants + 3 immobile + 1 flag = 25 placed pieces + 5 camps** = 25 pieces in 30 slots (5 slots are camps; see §1.2).

**Authoritative source**: `legacy_engine/src/junqi.h` line 20, enum `ChessType`.

### 1.2 Seat Layout (30 slots per seat)

Each seat owns a **5-wide × 6-tall** rectangle. Indexing within a seat:

```
 Index within seat:        Rows (relative to own base):
   0  1  2  3  4            Row 1 (front,  closest to center)
   5  6  7  8  9            Row 2
  10 11 12 13 14            Row 3
  15 16 17 18 19            Row 4
  20 21 22 23 24            Row 5
  25 26 27 28 29            Row 6 (back,   closest to own edge)
```

**Camps (行营)**: 5 cells — indices `{6, 8, 12, 16, 18}` — always empty; enemies cannot attack a piece inside its own camp.
**Strongholds (大本营)**: 2 cells — indices `{26, 28}` — the Flag must sit in one of these two.

**Authoritative source**: `legacy_engine/src/junqi.c::SetBoardCamp`.

### 1.3 Placement Hard Constraints (Q13 / setup validation)

A setup `Lineup[4][30]` is **legal** iff for every seat `d ∈ {0,1,2,3}`:

| # | Constraint | Enforcement |
|---|-----------|-------------|
| C1 | Camps must be empty: `Lineup[d][i] = NONE` for `i ∈ {6,8,12,16,18}` | **Hard** (Q13) |
| C2 | Flag (`JUNQI`) must sit in a stronghold: exactly one of `Lineup[d][26], Lineup[d][28]` is `JUNQI`; all other slots are not. | **Hard** |
| C3 | Landmines (`DILEI`) only in the back two rows (`i ≥ 20`), and never in a stronghold occupied by the flag. | **Hard** |
| C4 | Bombs (`ZHADAN`) not in the front row (`i < 5`). | **Hard** |
| C5 | Exact piece counts per §1.1 column "Count/Seat". | **Hard** |

**Any violation → reject setup, do not start game.** The `junqi_core.setup.validate_setup()` function MUST enforce C1–C5.

> **Note**: Legacy engine's `isNotBomb`/`isNotLand` fields on `ChessLineup` are **inference priors about ENEMY pieces** (used by the AI), NOT setup-validation constraints. Our new library must keep these concepts strictly separated (see §7.2 belief model).

### 1.4 Board Geometry

The 17×17 board contains:

- **4 seat zones** (each 5×6 = 30 cells) at the 4 edges.
- **1 central combat zone** (九宫格, NineGrid): 9 cells at `(x, y) ∈ {6, 8, 10} × {6, 8, 10}`.
- **Empty cells** between seat zones (rails pass through these).

**Coordinate formulas** (from `legacy_engine/src/junqi.c::SetChess`):

| Seat         | x-formula  | y-formula  | Range               |
|--------------|------------|------------|---------------------|
| SOUTH (0)    | `10 - i%5` | `11 + i/5` | x∈[6,10], y∈[11,16] |
| WEST  (1)    | `5 - i/5`  | `10 - i%5` | x∈[0,5],  y∈[6,10]  |
| NORTH (2)    | `6 + i%5`  | `5 - i/5`  | x∈[6,10], y∈[0,5]   |
| EAST  (3)    | `11 + i/5` | `6 + i%5`  | x∈[11,16],y∈[6,10]  |

NineGrid cells: `(10-(i%3)*2, 6+(i/3)*2)` for `i ∈ [0,8]`, i.e. 9 cells on a 2-step grid inside the central 5×5.

### 1.5 Railways (铁路)

Railway cells are the **outer ring** of each 5×6 seat zone (i.e. `i/5==0 || i/5==4 || i%5==0 || i%5==4` and `i<25`), plus the connecting rails across the board.

Railway types:
- **Horizontal rail**: same x (running up/down in world coords since x is column).
  - Wait — let me re-verify with the code: `HORIZONTAL_RAIL` triggers when `pDst->point.x == pSrc->point.x` (same x, different y) → so "horizontal" in code naming actually means **moving along the y-axis** (vertically on screen). We preserve the legacy naming to avoid diverging from the C code.
- **Vertical rail**: same y (`pDst->point.y == pSrc->point.y`).
- **Curve rail**: 4 special curve segments labeled `RAIL1..RAIL4` (see `SpcRail` enum).
- **GONGB rail**: virtual super-mode for engineers; can traverse ANY sequence of rail cells regardless of direction (subject to §2.3.2).

**Rail travel rule (non-engineer)**: All intermediate cells on the chosen rail path must be empty; direction cannot change except via curve rails.
**Rail travel rule (engineer)**: See §2.3.2.

**Authoritative source**: `legacy_engine/src/path.c::IsSameRail`, `GetRailPath`.

---

## 2. Movement

### 2.1 Move Atomicity (Q4)

- Exactly **one move per turn**, per acting seat.
- Turn order: `SOUTH (0) → WEST (1) → NORTH (2) → EAST (3) → SOUTH ...`.
- First move at game start: **SOUTH** (seat 0). For human-vs-AI, the human may select any seat; the turn order then rotates starting from SOUTH regardless.

### 2.2 Skip / Pass (Q12)

- **There is NO skip action.** The legacy `JUMP_EVENT` capability is disabled in the RL engine.
- If a seat has **zero legal moves** on its turn:
  1. That seat is **immediately marked dead** (`PartyInfo.bDead = 1`).
  2. All of its surviving pieces are removed from the board (see Q1, §5.1).
  3. Turn passes to the next seat.
- If both members of a team are dead, the opposing team wins (§5.3).

> Rationale: forbidding pass eliminates stalling loopholes and simplifies RL action masking.

### 2.3 Legal Moves

A piece at `src` can move to `dst` iff ALL of:

1. `src` is occupied by a piece owned by the acting seat.
2. `dst` is either empty, or occupied by an enemy piece that is NOT in its own camp.
3. `src` is not a stronghold cell (flags and pieces already in strongholds cannot move).
4. `src.type ∉ {DILEI, JUNQI}` (landmines and flags are immobile).
5. One of three movement modes applies: §2.3.1, §2.3.2, or §2.3.3.

**Authoritative source**: `legacy_engine/src/path.c::IsEnableMove`.

#### 2.3.1 Adjacent Move (adjacent / diagonal via camps)

- `dst` is one of the 8 cells `(src.x ± 0/1, src.y ± 0/1)` (excluding `src` itself), AND either:
  - `src` or `dst` is a camp (diagonal OK via camps), OR
  - `dst.x == src.x` or `dst.y == src.y` (non-diagonal).

#### 2.3.2 Railway Move (non-engineer)

- Both `src` and `dst` are railway cells.
- A **straight** rail path exists from `src` to `dst` with:
  - All intermediate cells empty (`type == NONE`).
  - Direction preserved (same x-line or same y-line).
  - Curves are only allowed via the 4 designated curve-rail segments.

#### 2.3.3 Engineer (`GONGB`) Railway Move (Q1 re-confirm)

- Engineer can traverse **any connected sequence of empty railway cells**, changing direction freely at any rail junction.
- Implemented as BFS over the rail subgraph from `src` avoiding occupied cells.
- Multiple shortest paths may exist; the UI displays **the first one found by the standard BFS** (Q3).
- `dst` must be a railway cell; the final cell can be occupied by an enemy (that would be a combat move).

### 2.4 Path Blocking (Q8)

- For ANY rail move, every intermediate cell on the chosen path must be `type == NONE`.
- Own pieces, ally pieces, enemy pieces, camps (which are NONE by rule anyway) all count as "blocked" if non-empty.
- Adjacent moves (§2.3.1) have no "intermediate" cells.

### 2.5 Camp Occupation (Q9)

- A camp (`isCamp == true`) is **safe**: an enemy piece cannot attack a piece sitting inside its own camp.
- A **camp is public**: any of the 4 seats may occupy any of the 20 camp cells on the board (not just camps in their own zone). First-come, first-served; no ownership.
- A piece inside a camp may move out normally; the attacker cannot follow into the camp while the camp is occupied by any piece.

> Rationale: original 四国军棋 only allows own camps as refuge by convention, but many online variants (and the legacy engine) treat all 20 camps uniformly. We follow legacy engine behavior (Q9 option a).

### 2.6 Stronghold (大本营) Attack (Q6)

- Strongholds (`isStronghold == true`) are **not** safe zones.
- Any enemy piece satisfying §2.3 can attack a piece sitting in a stronghold.
- This is intentional: attacking strongholds is a general tool for **probing the enemy's landmine distribution** (since landmines sit in rows `index ≥ 20` and strongholds are at `index ∈ {26, 28}`). Whether the attack hits the flag, a mine, or a plain piece is itself informative. The three-mine formation is just one special case of this probe strategy; the general pattern is "sacrifice a cheap piece to a stronghold to constrain the opponent's back-rank belief".

---

## 3. Combat

When a piece moves onto a cell occupied by an enemy piece, combat resolves **atomically** per the following table.

### 3.1 Combat Table

Let `A = src.type` (attacker, always the moving piece) and `B = dst.type` (defender).

**The rows are ordered, and the order is part of the rule.** Evaluate top to
bottom and take the first match.

| # | Condition | Outcome | Attacker | Defender |
|--:|-----------|---------|----------|----------|
| 1 | `A == ZHADAN` or `B == ZHADAN` | **BOMB** (mutual death) | Removed | Removed |
| 2 | `B == JUNQI` | **FLAG_CAPTURED** → defender's team surrenders (§5.2) | Moves to `dst` | Removed |
| 3 | `A == GONGB` && `B == DILEI` | **EAT** | Moves to `dst` | Removed |
| 4 | `B == DILEI` (A != GONGB) | **KILLED** | Removed | Stays |
| 5 | `A == B` (same rank) | **BOMB** (mutual death) | Removed | Removed |
| 6 | `A < B` (A is stronger, lower enum value) | **EAT** | Moves to `dst` | Removed |
| 7 | `A > B` (A is weaker) | **KILLED** | Removed | Stays |

Row 1 comes first because 炸弹和敌方任何棋子相遇则同归于尽，**包括军旗和地雷**.
A bomb never survives and never carries off the flag; it takes whatever it
touches down with it. Note that `flag_captured` is decided by `B == JUNQI`
alone, independent of the event, so a bomb reaching the flag still ends the
defender's game — it just does not leave a piece standing on the stronghold.

> An earlier revision of this table listed the flag and mine rows first and
> restricted row 4 to `A ∈ [SILING..GONGB]`, which made rows 1 and 4 disagree
> about `ZHADAN` vs `DILEI`. The implementation resolved that ambiguity the
> wrong way for both `ZHADAN` vs `DILEI` and `ZHADAN` vs `JUNQI`.

**Authoritative source**: `legacy_gui/src/rule.c::CompareChess`, which tests
`ZHADAN` before the flag and the mine for exactly this reason.
`tests/test_combat_rules.py::test_combat_matches_legacy_table` compares all
120 valid `(A, B)` pairs against a literal transcription of it.

### 3.2 SILING Flag-Reveal Rule (Q7)

When a `SILING` (司令) dies in combat, **its owning seat's flag is revealed** (`PartyInfo.bShowFlag |= 2`). This applies per-seat independently:

| Scenario | Outcome | Flag-of-src reveal | Flag-of-dst reveal |
|----------|---------|:----------------:|:----------------:|
| SILING eats weaker piece | EAT | — | — |
| SILING vs SILING | BOMB (both die) | ✅ reveal | ✅ reveal |
| SILING vs ZHADAN | BOMB (both die) | ✅ (src died) | ❌ |
| ZHADAN vs SILING | BOMB (both die) | ❌ | ✅ (dst died) |
| SILING vs DILEI | KILLED | ✅ (src died) | ❌ |
| SILING eats JUNQI | FLAG_CAPTURED | — | — (defender surrenders anyway) |

**Inference consequence (Q7 key insight)**: Other players observe `(event, src_side_revealed?, dst_side_revealed?)`. If a BOMB event occurs AND exactly ONE side revealed its flag, then the non-revealing side's piece MUST have been `ZHADAN` (since same-rank bomb would reveal both, and a silent death means non-SILING). This is a **hard deduction** that the belief model MUST encode. See §7.3.1.

**Authoritative source**: `legacy_engine/src/junqi.c` lines 720-735 (the two independent `if (pSrc->pLineup->type == SILING)` / `if (pDst->pLineup->type == SILING)` blocks).

### 3.3 Information Broadcast (Q2) — THE CORE NOVELTY OF 四国军棋

After every move, the engine broadcasts a **MoveResult** to all 4 seats. The content is **identical for all observers** (no private info to combatants) and contains ONLY:

| Field | Content | Notes |
|-------|---------|-------|
| `src_pos` | `(x, y)` | Where the moving piece came from |
| `dst_pos` | `(x, y)` | Where it moved to |
| `event`   | One of `{MOVE, EAT, KILLED, BOMB}` | No piece type revealed |
| `flag_reveal_src` | bool | True iff acting seat's flag was revealed this turn (only on SILING death) |
| `flag_reveal_dst` | bool | True iff defending seat's flag was revealed this turn |
| `flag_captured`   | bool | True iff `B == JUNQI` was captured |

**Critically NOT broadcast**:
- The type of the attacker (`A`).
- The type of the defender (`B`).
- Any internal belief/probability state.

**Why this matters**: This is what makes 四国军棋 HARDER than Stratego. In Stratego, combatants SEE each other's type; spectators see the result. In 四国军棋, **nobody directly sees any type** (except via SILING-flag-reveal and JUNQI-capture). Partial information can only be inferred through the §7 reasoning table.

### 3.4 Combat Inference Cheat-Sheet (what OTHER players can deduce)

Given `(event, A_visible, B_visible, flag_reveals)`, observers can sometimes narrow down piece types:

| Observable | Deducible | Example |
|-----------|-----------|---------|
| EAT, src moved onto dst | `A ≤ B` (A stronger or equal rank doesn't happen on EAT — wait: same-rank is BOMB, so A < B) | A is strictly stronger than B |
| KILLED, src removed | `A > B` OR `B == DILEI` | A is weaker OR walked into a mine |
| BOMB, both removed | `A == B == ZHADAN` is impossible (bombs don't move? — they do; A can be ZHADAN). More precisely: `A ∈ {ZHADAN}` OR `B ∈ {ZHADAN}` OR `A == B` | Combined with flag reveals, often resolves exactly (see Q7 insight) |
| FLAG_CAPTURED | `B == JUNQI`, `A` is any piece | Defender surrenders |

**Special deduction cases** (RULES.md does not enumerate the belief algorithm; see `docs/INFERENCE.md` to be written in Phase 0.2). A few canonical examples:

1. **Engineer-reveals-mine**: If A eats B where B was on `index ≥ 20` (back 2 rows) and B was immobile up to this point, and A survived, then A is likely `GONGB` and B was `DILEI`. Observers can mark A as engineer.
2. **Lieutenant-beats-engineer**: If A kills B and A was known to be weak (e.g. `PAIZH`), observers learn B's rank was `≥ PAIZH`, typically `GONGB`.
3. **SILING-on-mine**: KILLED + src-flag-revealed → src was `SILING`, dst was `DILEI`.
4. **Bomb-into-SILING** (Q7): BOMB + exactly-one-side-revealed → revealed side was `SILING`, other side was `ZHADAN`.
5. **Double SILING**: BOMB + both-sides-revealed → both were `SILING`.

These are **formal deductions** that the belief network should learn or be constrained to.

### 3.5 Stronghold Removal on Surrender (Q1)

- When a team surrenders (flag captured, all-dead, or both allies dead):
  1. **All remaining pieces of BOTH teammates are removed from the board** (option A).
  2. The surrendering team's `PartyInfo.bDead = 1` for both seats.
  3. Turn rotation skips dead seats.
- Surrendered pieces do NOT become spoils; they vanish.

### 3.6 Voluntary Surrender

- **Not allowed during training** (forced to play until natural termination).
- May be allowed in human-vs-AI mode via explicit UI action (future Phase 5).

---

## 4. Information Model (summary, detail in §7)

This is the **defining feature** of 四国军棋 AI difficulty:

- Pieces start **hidden** to enemies (暗棋 mode) — OR —
- Optional **semi-暗棋 training mode**: teammates see each other's full setup at start (Q11 option a); enemies still see nothing.

- After each combat, ONLY `{event, positions, flag_reveal_flags}` are broadcast (§3.3).
- No private channels; all 4 players see the same broadcast.
- Piece types can only be **inferred** from combat outcomes via §3.4 logic.

---

## 5. Termination

### 5.1 Team Death

A seat is **dead** when any of:
- Its flag is captured.
- It has zero legal moves on its turn (Q12).
- Both of its pieces are exhausted via combat (edge case; practically subsumed by above).

A team is **defeated** when BOTH of its seats are dead.

### 5.2 Victory

The game ends with a **winner team** as soon as:
- The opposing team is defeated (either flag captured or both seats dead).

Reward assignment for RL:
- Winner team's both seats get `+1` each.
- Loser team's both seats get `-1` each.
- Draws: all 4 seats get `0` (§5.3).

### 5.2a Mutual Destruction — Attacker Wins (Q14)

If a single `step()` drives **both teams** to complete defeat simultaneously
(i.e. in the same action, all remaining pieces on one team die AND all
remaining pieces on the other team die), the **attacking team** (the team of
the acting seat `action.seat`) is declared the winner.

**Rationale**: the attacker took the initiative to break the stalemate and is
credited with the decisive action. Standard 四国军棋 convention used to avoid
awarding a draw on every "last SILING vs last SILING" end-of-game scenario.

**Canonical example**: both teams are reduced to a single piece each (say,
two lone SHIZH's); the acting team's SHIZH attacks the opposing SHIZH; the
resulting same-rank BOMB kills both last pieces → both teams are
simultaneously defeated → **acting team wins, not a draw**.

**Implementation contract** (see `junqi_core.state`):
- `check_game_over()` MUST inspect `action.seat.team` when both teams are
  found dead in the same `step()`.
- If at least one seat from the attacker's team was alive BEFORE this step
  (which is always true since the action was executed), the attacker wins.
- The only remaining "draw" outcome is hitting the §5.3 step thresholds.

### 5.3 Draw Thresholds (Q10)

To prevent infinite stalls, a draw is declared if ANY of:

| Threshold | Value | Counter Reset |
|-----------|------:|---------------|
| `max_num_moves` | 4000 | Never (global move counter) |
| `max_num_moves_between_attacks` | 200 | Reset to 0 on any `EAT`/`KILLED`/`BOMB`/`FLAG_CAPTURED` event |

First threshold hit → game ends as **draw** (all rewards = 0).

> Inspired by Ataraxos (Stratego) which uses identical safety-net design.

### 5.4 First-Move Rule (Q5)

- **Training (self-play)**: seat order is always `SOUTH → WEST → NORTH → EAST`; SOUTH always moves first.
  - Single network weights across all 4 seats; no first-move bias because the network learns to play from whichever seat.
  - Canonical rotation (see `docs/ARCHITECTURE.md`) normalizes the view so the network never sees its seat index.
- **Human-vs-AI**: human may choose any of the 4 seats; internal order still `0→1→2→3`, but the human's seat is set accordingly.

---

## 6. Setup Phase (Deployment / 布阵)

### 6.1 Setup Source

Each seat provides a 30-element `Lineup[d][0..29]` array. This is either:
- Precomputed from a **setup library** (distribution of expert setups).
- Generated by a **setup network** (future RL component).
- Provided by a **human player** (GUI / web frontend).

### 6.2 Setup Validation

All 4 setups pass `validate_setup()` which enforces §1.3 constraints C1-C5. Failure → engine refuses to start; caller must retry.

### 6.3 Setup Visibility

- **Own**: full visibility always.
- **Teammate (training, Q11 option a)**: full visibility from game start.
- **Teammate (暗棋 full-information-hiding mode, future)**: hidden; same as enemy.
- **Enemy**: hidden always; only type-prior on `isNotBomb` (index<5 cells cannot be bombs) and `isNotLand` (index<20 cells cannot be landmines) is public knowledge, because enemies know the §1.3 constraints too.

> 💡 **Important**: The `isNotBomb` / `isNotLand` priors are **consequences** of the hard setup constraints (§1.3). They are not extra rules, they are just publicly-known structural facts. The belief network should receive them as observation channels (§7.2).

---

## 7. Belief Model (advanced — subset here, full spec in `docs/INFERENCE.md`)

### 7.1 Public State (visible to all)

- Every cell's `(x, y, is_occupied, owner_seat_if_occupied, is_camp, is_stronghold, is_railway)`.
- Every seat's `(alive, flag_revealed)`.
- Move history (sequence of MoveResult broadcasts).

### 7.2 Belief Channels (observation tensor)

For every enemy piece still on the board, a 12-dim probability distribution over `{JUNQI, DILEI, ZHADAN, SILING, JUNZH, SHIZH, LVZH, TUANZH, YINGZH, LIANZH, PAIZH, GONGB}`.

Priors (at game start):
- `index < 5`: P(ZHADAN) = 0, P(DILEI) = 0. Renormalize over remaining 10 types.
- `5 ≤ index < 20`: P(DILEI) = 0. Renormalize over 11 types.
- `index ∈ {26, 28}`: P(JUNQI) boosted significantly (must sit in one of these 2 cells).
- Other constraints from piece counts (Bayesian over remaining inventory).

### 7.3 Deductive Updates

After each combat event, exact logic prunes the belief tensor:

#### 7.3.1 SILING-reveal rule (Q7 deduction)
- BOMB event + only src flag revealed → src = `SILING`, dst = `ZHADAN`.
- BOMB event + only dst flag revealed → dst = `SILING`, src = `ZHADAN`.
- BOMB event + both revealed → both = `SILING`.
- BOMB event + neither revealed → neither was `SILING`; types inferred from rank equality or any-bomb rule.

#### 7.3.2 Stronghold probing — landmine/flag distribution
- Strongholds sit at `index ∈ {26, 28}`, i.e. in the back-rank landmine zone.
- Possible outcomes when attacking a stronghold with a known cheap piece (e.g. `PAIZH`):
  a. `EAT`: the stronghold contained the flag → game-ending capture.
  b. `KILLED` and attacker's flag NOT revealed: the stronghold contained a `DILEI`.
  c. `EAT` or `KILLED` against another ranked piece: the stronghold contained a non-flag piece; the flag is in the OTHER stronghold cell.
  d. `BOMB`: the stronghold contained a `ZHADAN` (rare on index 26/28 because bombs are allowed in back rows but players rarely place them there).
- Each outcome sharply narrows the belief over the enemy's back-rank layout. The three-landmine formation is a special sub-pattern where both strongholds + an adjacent cell all carry mines; the probe either confirms it (case b) or refutes it (case c).

#### 7.3.3 Engineer-signature
- Engineer is the ONLY piece that can traverse multi-segment rail paths with direction changes. First time an enemy piece does this → `P(GONGB) = 1` for that piece.
- Engineer is the ONLY piece that can eat a landmine. EAT event on a `index ≥ 20` cell that was immobile → attacker = `GONGB`, defender = `DILEI`.

### 7.4 Imperfect Deductions

Not every combat gives certainty. E.g., EAT with `A` and `B` both unknown → we only learn `A < B` (A's enum value less than B's). This is a **lattice constraint**; the belief network must maintain consistent distributions under all such constraints.

---

## 8. Action Space (for RL)

### 8.1 Action Definition

An action is `(src_cell, dst_cell)` pair, where both are cells on the 17×17 board.

- Maximum raw action space: `289 × 289 = 83521`.
- Actually-reachable space is much smaller (pieces only move to adjacent or rail-reachable cells).
- At each step, a **legal action mask** of shape `[289, 289]` (sparse) restricts the policy.

### 8.2 Legal Action Generation

Implemented in `junqi_core.move_gen.generate_legal_actions(state, seat)`:

1. For each of seat's 30 pieces still alive:
   a. Compute reachable cells via §2.3 (adjacent + rail).
   b. For each reachable cell that passes §2.1 + §2.3 constraints, add to action list.
2. Return list of `(src, dst)` tuples.
3. Construct mask tensor `mask[src_flat, dst_flat] = 1` for each legal action.

### 8.3 Canonical Action Rotation

Network outputs actions in **canonical view** (current actor rotated to the canonical SOUTH / bottom position). Before executing, `canonical_to_world(action, actor_seat)` rotates back to world coordinates.

See `docs/ARCHITECTURE.md` §3 for the rotation math.

---

## 9. Replay Protocol

Every game MUST be fully recordable and replayable bit-exactly. The replay record consists of:

```
ReplayFile := {
  rules_version: "1.0.0",
  seed: int,                           // RNG seed (if any stochastic element)
  setups: [Lineup[4][30]],             // all 4 setups
  moves: [MoveRecord, ...],            // sequence of moves
  final_state: GameOutcome,
}

MoveRecord := {
  seat: int,
  src: (x, y),
  dst: (x, y),
  result: MoveResultData,              // broadcast result (§3.3)
  private_diff: { ... }                // OPTIONAL: for debugging, raw type changes
}
```

The `junqi_core.replay` module MUST verify: given `setups` + `moves`, replaying produces identical `MoveResult` broadcasts and identical final state.

---

## 10. Rule Clarification Index (the 14 canonical answers)

| # | Question | Decision | Section |
|---|----------|----------|--------:|
| Q1 | Piece fate on surrender | All pieces removed | §3.5 |
| Q2 | Combat info broadcast | `{event, pos, flag_reveal}` only; no types | §3.3 |
| Q3 | Engineer path choice | Auto-pick shortest (UI concern) | §2.3.3 |
| Q4 | Moves per turn | Exactly 1 | §2.1 |
| Q5 | First-move seat | Training: SOUTH; Human: any | §5.4 |
| Q6 | Stronghold attackable | Yes, by any enemy | §2.6 |
| Q7 | SILING flag-reveal | Per-seat independent; both die → both reveal | §3.2 |
| Q8 | Rail path blocking | All intermediate cells must be empty | §2.4 |
| Q9 | Camp occupation | Any seat may enter any camp | §2.5 |
| Q10 | Draw thresholds | `4000` total; `200` since last combat | §5.3 |
| Q11 | Teammate setup visibility | Visible from start (training) | §6.3 |
| Q12 | Skip/pass action | Forbidden; no-moves = seat dies | §2.2 |
| Q13 | No pieces in camps | Hard constraint in setup validator | §1.3 (C1) |
| Q14 | Mutual destruction | Attacker's team wins (not a draw) | §5.2a |

---

## 11. Version History

| Version | Date       | Changes |
|---------|-----------|---------|
| 1.0.0   | 2026-04-19 | Initial canonical specification. Frozen. |
| 1.1.0   | 2026-04-20 | Added §5.2a (Q14): mutual-destruction resolution awards victory to the attacker's team (non-breaking addition). |

---

## Appendix A: Cross-Reference to Legacy Engine

| Rule § | Legacy file / function | Notes |
|--------|-----------------------|-------|
| §1.1 piece enum    | `junqi.h::ChessType` line 20 | Direct copy |
| §1.2 seat layout   | `junqi.c::SetChess`, `SetBoardCamp` | Coordinate math preserved |
| §1.3 setup C1–C5   | NEW — legacy has no validator | New library must enforce |
| §1.5 rails         | `path.c::IsSameRail`, `GetRailPath` | Direct port |
| §2.3 movement      | `path.c::IsEnableMove` | Direct port |
| §3.1 combat table  | `junqi.c::JudgeChess`, `PlayResult` | Direct port + SILING flag-reveal (§3.2) |
| §3.2 SILING reveal | `junqi.c` lines 720-735 | Two independent `if` blocks |
| §3.3 broadcast     | `event.c` event system | Our engine strips private info |
| §5.3 draw          | NEW — legacy has no draw logic | RL-specific |
| §7 belief          | `ChessLineup::isNotLand / isNotBomb / mx_type` | Legacy conflates prior with inference; we separate |

---

## Appendix B: Example Scenarios (for test authors)

See `tests/golden/` for concrete JSON scenarios. Categories:

- `tests/golden/move_gen/`: basic moves, rail BFS, curve rails, GONGB special.
- `tests/golden/battle/`: full combat table coverage.
- `tests/golden/siling_flag/`: all 6 SILING scenarios from §3.2.
- `tests/golden/stronghold/`: attacks on strongholds, non-flag strongholds.
- `tests/golden/inference/`: §3.4 special deductions, §7.3 belief updates.
- `tests/golden/setup_validation/`: C1-C5 enforcement.
- `tests/golden/full_game/`: end-to-end oracle runs from legacy engine.

Each scenario JSON contains:
- `description`: human-readable intent
- `rules_version`: required version
- `setup` or `pre_state`: initial condition
- `action`: what is being tested
- `expected`: expected outcome (observable fields only unless `debug_include_private` is true)

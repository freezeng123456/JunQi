
# Information Model & Deterministic Inference

> **Status**: DRAFT (evolving with `junqi_core/info_model.py`)
> **Scope**: Specifies the belief representation for hidden-information 四国
> 军棋 and all *deterministic* inferences the engine applies after each step.
> Probabilistic / model-based inference (MCTS priors, NN belief heads) is
> OUT of scope here — this document describes only rules that can be
> proved from the public observation sequence.
> **Versioning**: Tied to RULES.md version. Any change to inference rules
> bumps `INFERENCE_VERSION` in `junqi_core/info_model.py`.

---

## 0. Notation

- `observer ∈ Seat`: the seat whose belief we maintain (4 copies per game).
- `about ∈ Seat`: the seat the belief is *about*.
- `piece_id ≡ (x, y)`: the **current** world-frame position of a piece
  (belief follows the piece as it moves; on death belief is deleted).
- `p[t] ∈ [0, 1]`: probability the piece at `(x, y)` has type `t ∈ PieceType`.
  We only track the 12 concrete types (exclude `NONE` / `DARK`).
- Show modes:
  - **BRIGHT**: everyone sees everything → belief is always one-hot.
  - **HALF_DARK** (training default, Q11): observer sees own + teammate's
    pieces as one-hot; enemy pieces have distributions.
  - **DARK**: reserved for the future; not used in v1.x.

---

## 1. Belief representation (`BeliefTensor`)

For each observer, we store:

```
belief[observer] : Dict[(x, y), ndarray[12]]
```

Invariants (every value must hold at all times):

- **I1**: `sum(belief[observer][(x, y)]) == 1.0`, up to floating-point ε.
- **I2**: `belief[observer][(x, y)][t] >= 0` for all `t`.
- **I3**: Teammate and own pieces are one-hot (dict lookup returns the
  distribution `[0, ..., 1, ..., 0]` matching the true type). In BRIGHT
  mode this extends to enemies.
- **I4**: Dead seats have no entries at all (their pieces are removed from
  `state.pieces`, so belief has nothing to track).
- **I5**: `(x, y) ∈ belief[observer]` iff `(x, y) ∈ state.pieces` AND the
  owning seat is alive.

A **global inventory constraint** is ALSO tracked for each enemy seat:

```
remaining[observer][about] : Dict[PieceType, int]
```
— the number of each piece type still believed alive on `about`'s pieces.
It starts at `PIECE_COUNTS` minus any types that must be dead by
deterministic reasoning. Used to zero out impossible types.

---

## 2. Initial prior (game start)

At `t = 0`, the observer has not seen any move. It knows:

**Setup constraints (C1–C5 from RULES §1.3)** restrict the possible
placements of each piece type:

| Piece type | Setup constraint                                          |
|------------|-----------------------------------------------------------|
| `JUNQI`    | MUST be on exactly one stronghold index ∈ {26, 28}        |
| `DILEI`    | MUST be in back two rows (indices 20..29)                 |
| `ZHADAN`   | MUST NOT be in front row (indices 0..4)                   |
| (others)   | Any non-camp index allowed                                |

Plus: camp indices {6,8,12,16,18} are always empty, strongholds hold no
non-flag piece.

### 2.1 Observer's rules for each position on `about`'s board

Let `S` = set of piece types consistent with the index `i` of the position
(applying the table above). The uniform prior over `S` is too crude; we
refine using **inventory counts**: given `PIECE_COUNTS[t]` copies exist,
the probability of type `t` at a random admissible slot is proportional to
`PIECE_COUNTS[t]` / (number of admissible slots for `t` globally).

**Formal formula** (for enemy pieces, HALF_DARK):

For each index `i` on seat `about`'s setup:
- Let `A(i)` = { piece types allowed at index `i` given constraints }
- For each `t ∈ A(i)`:
  `p_initial[i][t] = c_t(i) · PIECE_COUNTS[t] / Z(i)`
  where `c_t(i) = 1` if `t ∈ A(i)` else `0`, and `Z(i)` normalizes to 1.

This gives a closed-form, constraint-aware initial prior. A more refined
prior would use inclusion-exclusion over the joint lineup, but that is
PSPACE-hard in the general case; in practice the simple per-slot prior is
sufficient for opening-phase heuristics and the NN refines it later.

### 2.2 Teammate (Q11): always one-hot

SOUTH observer sees NORTH's setup as one-hot (they are on the same team,
`seat % 2 == 0`). Ditto for the SOUTH–NORTH and WEST–EAST pairs. This
is the Q11 "teammate setup visible from start" rule.

---

## 3. Deterministic inferences per event

Let `result = MoveResult(seat, src, dst, event, flag_reveal_src,
flag_reveal_dst, flag_captured, ...)`.

We update `belief[observer]` after *every* step, for every observer that
did not already have the information. Order of rules matters — apply in
the listed order. Each rule is **sound** (never eliminates a true state).

### R1. Piece migration

- **MOVE**: `belief[observer][dst] ← belief[observer][src]`; delete `src`.
- **EAT**:  `belief[observer][dst] ← belief[observer][src]`; delete both
  `src` and the old `dst` key.
- **KILLED**: delete `src`. `dst` keeps its distribution.
- **BOMB**: delete both `src` and `dst`.

(Then re-assert invariant I5 — any cell not in `state.pieces` has no
belief entry.)

### R2. Rank-ordering inferences

When combat occurs with `event = EAT` (attacker wins), we learn:

- The attacker's rank > defender's rank (loosely; concrete rules in §5 of
  RULES.md). For each observer that does NOT know the attacker or the
  defender's type:
  - Zero out `p[attacker][t]` for all `t` whose strength < defender's max
    feasible strength under its current `p[defender]`. Re-normalize.
  - Symmetric cut on `p[defender]` for `t` stronger than attacker.
- Edge cases (same rank → BOMB; JUNQI involved; etc.) have their own
  rules below.

**Note**: this is the only "soft" step — it doesn't zero out individual
types unless the cut is deterministic. A common case IS deterministic:
if `p[defender]` was already one-hot at some type `t_d`, then
`p[attacker][t_a] = 0` for every `t_a` with `strength(t_a) ≤ strength(t_d)`.

### R3. Q7 SILING flag reveal (§5.1 of RULES)

On combat where either side was SILING:

- `flag_reveal_src = True` ⇔ attacker WAS SILING AND died in this fight.
  → set `state.info[seat_src].flag_revealed = True`.
  → if observer hadn't known `attacker_piece.type`, they now know it was
    SILING. Set `belief[observer][src_pos_before_combat][SILING] = 1`
    (one-hot). But `src` is deleted by R1; the implication is stored
    purely in "SILING is removed from `about=seat_src`'s remaining
    inventory".
- `flag_reveal_dst = True` ⇔ defender WAS SILING AND died.
  → symmetric.

### R4. Flag capture reveals the flag (Q1)

- `flag_captured = True` ⇒ the defender was JUNQI. Because the defender
  is removed (per §3.5 surrender), no belief update on the cell itself is
  needed — the key just gets deleted. But we DO update remaining
  inventory: `remaining[observer][seat_dst] = {}` (everything dies).

### R5. Landmine signatures

If the defender occupied a stronghold OR the event was KILLED on a
previously-suspected back-row cell, combined with the resolved event, we
may infer DILEI:

- If event = `KILLED` AND attacker's type is known to be NOT `GONGB` AND
  attacker's strength is known ≥ `DILEI`'s pseudo-strength (only SILING
  and `GONGB` can win vs DILEI, per RULES §5.2–5.4): then defender MUST
  be DILEI OR a higher-rank piece that actually beat attacker. Closer:
  any non-GONGB piece that died to a stationary piece at a back-row cell,
  the stationary piece is either DILEI or a higher rank.
  - Practical rule (conservative): if attacker rank was known ≥ PAIZH
    (which cannot beat DILEI) and attacker died → defender's type ∈
    {DILEI, or any piece higher than attacker}. Zero out all others.
- If event = `EAT` AND attacker is known to be `GONGB` AND defender dies
  → defender was DILEI (GONGB is the only piece that EATs DILEI).
  Strictly: `belief[observer][dst_before][DILEI] = 1` (but dst is deleted).
  Use inventory: decrement `remaining[observer][seat_dst][DILEI]` by 1.

### R6. Triangle-mine stronghold deduction (Q6)

When a player eats a piece at a stronghold and the event is `EAT`:
- That piece was NOT the flag (flag would trigger R4 instead).
- So `remaining[observer][seat_dst][non-JUNQI type]` bookkeeping applies,
  and observer learns the flag is at the OTHER stronghold of `seat_dst`.
- Set `belief[observer][(other_stronghold_pos_of_seat_dst)][JUNQI] = 1`
  if that cell still holds a piece (it should; JUNQI never moves).

### R7. GONGB signature

If a piece eats a DILEI (determined by R5 inverse or by revealing
defender's type via other paths), that piece is provably `GONGB`:
- Set `belief[observer][dst_after][GONGB] = 1`.
- Decrement `remaining[observer][seat_src][GONGB] -= 1`.

### R8. Bomb confirmation

If event = `BOMB` and neither piece was SILING (no flag reveal) and
neither was JUNQI, and the src type was not known:
- At least ONE side was ZHADAN, OR both sides have the exact same rank.
- Inventory: if observer already knew the defender's type was `T`, then
  either the attacker was type `T` (same rank → BOMB) OR `ZHADAN`.
- Practical: adjust `p[src]` and the remaining inventories accordingly.
  Implemented as a soft cut (never zeroes out unless deterministic).

### R9. Q12 seat death

When a seat dies via Q12 (no legal moves), all its remaining pieces are
removed. For the observer:
- Delete all `belief[observer][(x,y)]` where owner seat is the dead one.
- Clear `remaining[observer][dead_seat]`.
- Re-normalize global inventory constraints on surviving enemy seats.

### R10. Mutual destruction (Q14) — terminal

When `terminated_after = True` with `winner_team`, no further belief
updates are needed. Belief is frozen.

---

## 4. Consistency maintenance (normalization)

After any deterministic rule fires, **re-normalize** each affected
`belief[observer][(x, y)]` so `sum == 1`. If the zeroing produced a
zero-sum distribution, raise a loud error — this indicates a rule
implementation bug (we eliminated the true state).

Debug mode (`debug_include_private=True` on GameState) runs an extra
check after each update:
- For each `(x, y)` in belief, `belief[observer][(x, y)][true_type] > 0`.
  If not, print diagnostic and fail fast.

---

## 5. Rendering to tensor (for NN input)

For a neural network / policy head, an observer's belief is rendered as:

```
tensor : ndarray[17, 17, 12]  # world-frame
tensor[x, y, :] = belief[observer][(x, y)]  # zero-filled where no piece
```

Plus auxiliary channels (appended by the feature extractor, NOT by this
module):
- `seat_one_hot[x, y, 4]`: which seat owns the piece at (x, y)
- `flag_revealed[4]`: per-seat flag-revealed bits
- `turn_indicator[4]`: one-hot current turn
- `dead_seats[4]`: per-seat dead bits

---

## 6. Implementation bound (compute budget)

Each `step()` triggers at most O(pieces) work for belief updates — about
100 entries in the worst case. All rules are O(1) per piece; the global
pass is O(total_pieces). Training simulator target: < 50 µs per step for
belief maintenance on CPU.

---

## 7. Version History

| Version | Date       | Changes |
|---------|-----------|---------|
| 0.1.0   | 2026-04-20 | Initial draft: R1–R10, invariants I1–I5, BRIGHT/HALF_DARK. |


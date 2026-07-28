# JunQi-RL Architecture

> **Status**: DRAFT (2026-04-19)
> **Scope**: Engineering-level design doc covering codebase layout, module boundaries, coordinate systems, observation/action tensors, and the canonical rotation math that unifies 4-seat play under a single network weight.
> **Companion**: `RULES.md` (what the engine does) | This doc (how it's organized and how RL interacts with it) | `DECISIONS.md` (why these choices).

---

## 1. Repository Layout

Decision **D1 = A** (new Python library independent of legacy C code) and **D2 = B** (legacy GUI kept as visualization tool until Web frontend ships) give us:

```
JunQi/                              ← single monorepo
├── docs/
│   ├── RULES.md                    ← canonical rule spec (frozen v1.0.0)
│   ├── ARCHITECTURE.md             ← this file
│   ├── DECISIONS.md                ← ADRs: every major design choice + rationale
│   ├── INFERENCE.md                ← belief model detailed spec (Phase 0.2)
│   └── TRAINING.md                 ← RL training recipes (Phase 2+)
│
├── junqi_core/                     ← pure-Python engine (NEW, authoritative for RL)
│   ├── __init__.py
│   ├── rules.py                    ← piece enums, combat table, RULES_VERSION
│   ├── board.py                    ← 17×17 grid, rail graph, curve rails, NineGrid
│   ├── move_gen.py                 ← legal move generation (BFS engineer, rails)
│   ├── state.py                    ← GameState dataclass, transition, reward
│   ├── setup.py                    ← setup validator (C1-C5) + random generator
│   ├── info_model.py               ← belief tensor, deductive updates (§7 RULES)
│   ├── rotation.py                 ← world ↔ canonical view transforms
│   ├── observation.py              ← observation tensor builder for RL
│   ├── action.py                   ← action space encoding + legal-action mask
│   ├── replay.py                   ← replay record + replay verification
│   └── constants.py                ← dimensions, magic numbers, thresholds
│
├── junqi_rl/                       ← RL-specific code (Phase 1+)
│   ├── env/
│   ├── agents/
│   ├── networks/
│   ├── training/
│   └── utils/
│
├── junqi_viz/                      ← Web visualization (Phase 4)
│   ├── backend/                    ← FastAPI + WebSocket inference server
│   └── frontend/                   ← Vue/React board + belief heatmap
│
├── tests/
│   ├── golden/                     ← JSON oracle-generated test cases
│   │   ├── move_gen/
│   │   ├── battle/
│   │   ├── siling_flag/
│   │   ├── stronghold/
│   │   ├── inference/
│   │   ├── setup_validation/
│   │   └── full_game/
│   ├── test_rules.py
│   ├── test_rotation.py
│   ├── test_legacy_parity.py
│   └── conftest.py
│
├── legacy_engine/                  ← renamed from ENGINE/; read-only oracle
├── legacy_gui/                     ← renamed from GUI/; visualization until Phase 6
│
├── tools/
│   ├── oracle_runner.py            ← spawns legacy_engine, harvests golden cases
│   └── replay_viewer.py            ← CLI replay inspector
│
├── pyproject.toml
├── Makefile
├── .pre-commit-config.yaml
└── README.md
```

**Invariant**: `legacy_engine/` and `legacy_gui/` are **read-only** after initial rename. No new features added to C code; bugs there inform `junqi_core/` but are not fixed upstream.

---

## 2. Module Dependency Graph

```
rules.py  ←  leaf (no intra-package deps)
board.py  ← rules
move_gen.py ← rules, board, state
state.py  ← rules, board, setup
setup.py  ← rules
info_model.py ← rules, state
rotation.py ← rules  (pure math)
observation.py ← state, info_model, rotation
action.py ← state, move_gen, rotation
replay.py ← state

junqi_rl.env   ← junqi_core.*
junqi_rl.agents ← junqi_rl.env
junqi_rl.networks ← (torch only)
junqi_rl.training ← all of junqi_rl.*
```

Strict DAG; no cycles.

---

## 3. Coordinate Systems & Canonical Rotation

The **mathematical core** of making a single-weight network play all 4 seats symmetrically.

### 3.1 World Coordinates

- `x ∈ [0,16]` columns, `y ∈ [0,16]` rows. Origin top-left.
- **SOUTH (0)**: y ∈ [11,16] bottom, x ∈ [6,10].
- **WEST  (1)**: x ∈ [0,5]   left side,   y ∈ [6,10].
- **NORTH (2)**: y ∈ [0,5]   top,         x ∈ [6,10].
- **EAST  (3)**: x ∈ [11,16] right side,  y ∈ [6,10].
- NineGrid: `(x, y) ∈ {6,8,10} × {6,8,10}`.

> **Naming note (ADR-111)**: Seat names are cardinal directions anchored to the fixed world-frame board geometry. Integer values `SOUTH=0 / WEST=1 / NORTH=2 / EAST=3` are identical to legacy `ChessDir`'s `HOME=0 / RIGHT=1 / OPPS=2 / LEFT=3` respectively; only the Python-visible names changed. See `junqi_core/rules.py::LEGACY_DIR_TO_SEAT` for the explicit mapping.

Replays, golden tests, legacy interop — **all use world coordinates**.

### 3.2 Canonical Coordinates

Every network query for seat `s` first rotates the board so seat `s` appears at the SOUTH (bottom) position of the canonical image. After rotation:
- Acting seat at bottom (canonical y ∈ [11,16]).
- Teammate at top (canonical y ∈ [0,5]).
- `s.left_side_enemy`  (seat `(s+1) % 4`) at canonical **x ∈ [0, 5]**   — the **image left half**.
- `s.right_side_enemy` (seat `(s+3) % 4`) at canonical **x ∈ [11, 16]** — the **image right half**.

> **Naming policy (ADR-111)**: the enemy-of-acting-seat properties are named after which image half they occupy in the canonical frame, not after rotation direction or first-person viewpoint. `left_side_enemy` literally sits on the image left; `right_side_enemy` literally sits on the image right. This eliminates the mental flip that plagued earlier `left_enemy`/`right_enemy` and `ccw_enemy`/`cw_enemy` iterations.

Network outputs in canonical frame; we rotate back before applying.

### 3.3 Rotation Math (rotating around board center (8,8))

| Acting seat `s` | World → Canonical |
|:---------------:|-------------------|
| **0 (SOUTH)**   | identity: `(x, y) → (x, y)` |
| **1 (WEST)**    | `(x, y) → (y, 16-x)` |
| **2 (NORTH)**   | `(x, y) → (16-x, 16-y)` |
| **3 (EAST)**    | `(x, y) → (16-y, x)` |

> **Implementation note**: in numpy, these correspond to
> `np.rot90(plane, k=+1)` (WEST), `k=2` (NORTH), `k=-1` (EAST).
> The "CCW/CW" labels depend on whether y-axis points up (math) or down (image);
> we use the math convention for the formulas above (y-axis up, 0° = right).
> Verified by `junqi_core.rotation._self_check` on all 17×17 × 4 cells.

**Verification** (each seat's "center camp" maps to canonical SOUTH center `(8, 13)`):
- SOUTH: `(8, 13) → (8, 13)` ✓
- WEST:  `(3,  8) → (8, 13)` ✓
- NORTH: `(8,  3) → (8, 13)` ✓
- EAST:  `(13, 8) → (8, 13)` ✓

**Canonical → World** (for action un-rotation):

| Seat        | Canonical → World |
|:-----------:|-------------------|
| 0 (SOUTH)   | identity |
| 1 (WEST)    | `(x, y) → (16-y, x)` |
| 2 (NORTH)   | `(x, y) → (16-x, 16-y)` |
| 3 (EAST)    | `(x, y) → (y, 16-x)` |

### 3.4 Teammate Position in Canonical Frame

After rotation, for every seat the teammate ends up at the top zone (canonical y ∈ [0,5]):
- SOUTH (team A) → teammate NORTH is at top → identity keeps NORTH at top ✓
- NORTH (team A) → teammate SOUTH is at bottom → 180° sends SOUTH to top ✓
- WEST  (team B) → teammate EAST  at x∈[11,16] → +90° sends it to y∈[0,5] top ✓
- EAST  (team B) → teammate WEST  at x∈[0,5]  → -90° sends it to y∈[0,5] top ✓

**Invariant after rotation**: Teammate always "up"; enemies always on the image's "left" and "right" sides. The network never sees the seat index.

### 3.5 Symmetry Test (Phase 0 must-pass)

`tests/test_rotation.py::test_canonical_symmetry`:
- Construct a game state.
- Query the canonical observation tensor from each of 4 seats.
- After aligning the acting-seat's "own piece" channels, the 4 tensors should be identical up to a possible swap of `left-side-enemy` and `right-side-enemy` channel groups (this swap is allowed because e.g. SOUTH's `right_side_enemy` is NORTH's `left_side_enemy` under the 180° rotation between them).
- Any mismatch → rotation bug.

### 3.6 Action Mask Rotation

Legal action mask is computed in **world frame** (physics of the game), then rotated to canonical frame for the network. Sampled action goes back to world frame before execution.

---

## 4. Observation Tensor Layout

### 4.1 Spatial Tensor (canonical frame)

Shape: `[C, 17, 17]`, dtype `float32`. Channel breakdown:

| Block | Count | Content |
|:-----:|------:|---------|
| Piece-own             | 12 | One-hot location per piece type (JUNQI..GONGB), live pieces only — observer's own pieces |
| **Prob-teammate**     | 12 | Per-cell probability distribution over the teammate's piece type. **Degenerates to one-hot** under BRIGHT / HALF_DARK (Q11 teammate-visible); becomes a non-degenerate posterior under DARK. Sourced from `BeliefTensor.get(teammate_pos)`, which already encodes the correct sharpness per `show_mode` (see `info_model.py::_observer_sees_truth`). This unifies the "teammate is always visible" and "teammate needs inference" cases under a single channel group, enabling seamless curriculum BRIGHT → HALF_DARK → DARK without schema break. |
| Dark-teammate         | 1  | 1 where teammate has a piece but type is hidden (i.e. DARK mode); zero under BRIGHT / HALF_DARK. Provides an explicit existence mask when Prob-teammate has spread probability mass |
| Piece-left-side-enemy | 1  | 1 where `left_side_enemy` has a piece (type always hidden); image left half (x ∈ [0, 5]) |
| Piece-right-side-enemy| 1  | 1 where `right_side_enemy` has a piece; image right half (x ∈ [11, 16]) |
| Belief-left-side      | 12 | Per-cell P(`left_side_enemy`'s piece type = k); 0 if empty |
| Belief-right-side     | 12 | Same for `right_side_enemy` |
| Dead flags            | 4  | 1-plane per seat. **Observer-sorted order**: `[me, teammate, left_side_enemy, right_side_enemy]` (not absolute seat id) — keeps channel semantics frame-invariant across observers, matching the canonical-rotation philosophy |
| Flag revealed         | 4  | 1-plane per seat, same observer-sorted order as Dead flags |
| Board static          | 6  | camps, strongholds, rails, nine-grid, curve-rail-1, curve-rail-2 |
| Turn/history          | 4  | steps since last combat (norm), total steps (norm), is-my-turn, game-phase bucket |
| **T7 tail — `move_bucket`**            | 8  | Per-piece `move_count` bucket (exact-match `{0, 1, 2, ≥3}`), split 4 ours + 4 theirs. Anchored at the piece's current cell. `theirs` half fires only when observer's belief at that cell is one-hot (visibility filter, ADR-114 D-3). See ADR-115. |
| **T7 tail — `active_eat_bucket`**      | 8  | Per-piece `active_eat_count` cumulative bucket (`{0, ≥1, ≥2, ≥3}`), 4 ours + 4 theirs. Same visibility filter as `move_bucket`. |
| **T7 tail — `passive_survive_bucket`** | 8  | Per-piece `passive_survive_count` cumulative bucket, 4 ours + 4 theirs. Same visibility filter. |
| **T7 tail — `death_reason`**           | 6  | Per-death `DeathReason` one-hot (`{KILLED_BY_ENEMY, HIT_MINE_OR_BOMB, MUTUAL}`), 3 ours + 3 theirs. Anchored at `DeathInfo.death_loc`. No visibility filter — deaths are public (ADR-002). |
| **T7 tail — `dead_at_zero`**           | 2  | 1 at each dead piece's `zero_board` cell, 1 ours + 1 theirs. No visibility filter — zero-layout is public at `new_game`. |

**Total**: 101 channels (69 pre-T7 + 32 T7 tail), pinned by ADR-116 and enforced in `junqi_core/observation.py::OBS_CHANNELS`. Indices `[0, 69)` are bit-compatible with pre-T7 checkpoints so zero-padding the input conv's channel axis is sufficient to resume training on the 101-channel schema. The `ours / theirs` split is **team-based, not visibility-based**: teammate pieces always belong to `ours`, even under DARK.

### 4.2 Global (Non-Spatial) Features

Shape: `[G]`, dtype `float32`, `G = 28`. 1D features concatenated before policy/value heads:
- Remaining piece counts: 12 scalars for `left_side_enemy` + 12 scalars for `right_side_enemy` (24 total; sourced from `BeliefTensor.remaining[enemy_seat]`).
- Flag revealed for each of 4 seats in observer-sorted order `[me, teammate, left_side, right_side]` (4 scalars, 0/1).

(Game phase and time-budget scalars are broadcast into the spatial Turn/history channels instead of duplicated here.)

### 4.3 Batch Tensor Layout

For vectorized self-play:
- `obs_spatial`: `[B, C, 17, 17]`
- `obs_global`:  `[B, G]`
- `action_mask`: `[B, 289, 289]` boolean

---

## 5. Action Space

### 5.1 Flat Action Space

- Slots: `17 × 17 × 17 × 17 = 83521` (`src × dst`).
- Flat index: `a = sx*17³ + sy*17² + dx*17 + dy`.
- Typical legal count: ≤ 200 (< 0.3% of flat space).

### 5.2 Action Mask

`junqi_core.action.build_mask(state, seat)` → `[17,17,17,17]` bool in world frame; `rotation.rotate_mask(mask, seat)` brings to canonical.

### 5.3 Network Output

Policy outputs 83521 logits → masked softmax → categorical sample → unrotate → execute.

### 5.4 No-Op

**Forbidden** (Q12). If `mask.sum() == 0`, `env.step()` auto-kills the seat and advances turn. Policy never queried on dead seat.

---

## 6. Legacy Engine Parity Testing

### 6.1 Parity Test Protocol

`tests/test_legacy_parity.py`:
1. Spawn `legacy_engine/bin/JunQiEngine` subprocess (UDP localhost).
2. For 1000 random valid setups:
   - Feed same setup to both legacy and `junqi_core`.
   - Step in lockstep with the same random-but-legal moves.
   - Assert every `MoveResult` (position, event, flag flags) matches.
3. Pass criterion: 1000/1000.

### 6.2 Golden Case Generation

`tools/oracle_runner.py`:
- Input: scenario JSON with `expected = null`.
- Replays scenario in legacy engine, fills `expected`.
- Commit after human review.

### 6.3 Known Intentional Divergences

| Area | Legacy | New |
|------|--------|-----|
| Draw rules | None | 4000 steps / 200 no-combat |
| Skip action | Up to 5 per seat | Forbidden; no-moves → seat dies |
| Info broadcast | Exposes types to combatants/UI | Stripped to positions + event kind |

Divergences annotated per-scenario via `legacy_divergence: "..."` field.

---

## 7. Replay System

### 7.1 Recording

`junqi_core.replay.Recorder` captures:
- At start: `rules_version`, `setups[4]`, `seed`.
- Per step: `seat`, `(src, dst)` in world frame, `MoveResult`.
- At end: `outcome`, `final_state_hash`.

### 7.2 Verification

`verify(replay_file)`:
1. Load setups, fresh GameState.
2. Apply each action via `step()`.
3. Assert recorded `MoveResult` matches computed.
4. Assert final hash matches.
Any mismatch → non-determinism → bug.

### 7.3 Format

JSON default. MessagePack binary for large-scale training logs.

### 7.4 Viz Integration

- `legacy_gui`: consumes JSON via adapter (Phase 0-3).
- `junqi_viz` web: consumes JSON directly (Phase 4+).

---

## 8. Process & Concurrency Model

| Phase | Mode | Description |
|------:|------|-------------|
| 0-1   | Single-env debug | 1 process, 1 `GameState`, 4 seats rotate. Unit tests, golden cases. |
| 1+    | Vectorized (T4) | 1 process, `N`=64–256 `GameState` instances; CPU move-gen + GPU batched inference. |
| 2+    | Multi-process (H20) | Workers (16–64 CPUs) host `M`=4–8 envs each; central GPU learner; shared-memory / Ray / TorchRPC trajectories. |
| 3+    | 2-GPU H20 | Data-parallel via `torch.distributed` (NCCL). Sync if NVLink, async PS-style otherwise. |

---

## 9. Tooling

### 9.1 Python

- Python 3.10+ (3.11 preferred for perf).
- PyTorch 2.1+ (CUDA 12.x).
- Core: `torch`, `numpy`, `pydantic>=2`, `gymnasium`, `pytest`, `hypothesis`.
- Training: `tqdm`, `wandb`, `pyyaml`, `tensorboard`.
- Web (Phase 4): `fastapi`, `uvicorn`, `websockets`.

### 9.2 Legacy C

- `make -C legacy_engine` → `legacy_engine/bin/JunQiEngine`.
- `make -C legacy_gui` → GUI binary.
- Read-only after rename.

### 9.3 Dev Quality

- `ruff` for lint + format.
- `mypy --strict` on `junqi_core/`.
- `pre-commit` hooks.
- `pytest --cov=junqi_core` with ≥ 90% coverage gate.

---

## 10. Roadmap

| Phase | Weeks | Target |
|------:|:-----:|--------|
| 0.1   | 1     | `RULES.md`, `ARCHITECTURE.md`, `DECISIONS.md`, 100+ golden tests |
| 0.2   | 2     | `junqi_core` implementation; all golden tests pass; legacy parity ≥ 95% |
| 0.3   | 1     | Replay system + CLI tools + legacy_gui replay-mode integration |
| 1     | 1     | Gymnasium env, random/heuristic agents, vectorized self-play on T4 |
| 2     | 2     | Supervised pretrain from legacy engine games (awaits H20) |
| 3     | 3     | PPO self-play + belief network; full 四国军棋 training |
| 4     | 2     | Web visualization backend + frontend |
| 5     | 1     | Human-vs-AI pipeline |
| 6     | —     | Retire `legacy_gui` once `junqi_viz` feature-complete |

---

## 11. Open Phase 0.2 Questions

1. **INFERENCE.md**: detailed belief-update algorithm formalization, alongside `info_model.py`.
2. **Action masking efficiency**: sparse (CSR) vs dense boolean benchmark.
3. **Turn-skipping for dead seats**: codify in `state.py` (skip dead, continue with living teammate).
4. **Replay determinism**: pin all RNG sources; hash table ordering; float-vs-int math.

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 (DRAFT) | 2026-04-19 | Initial architecture decisions post Q13 rule freeze, D1/D2/D3 resolved. |
| 0.1.1         | 2026-04-20 | Phase 0.3 T7 seat rename (ADR-111): HOME/RIGHT/OPPS/LEFT → SOUTH/WEST/NORTH/EAST; `ccw_enemy`/`cw_enemy` → `left_side_enemy`/`right_side_enemy`. Integer values and JSON payloads unchanged. |
| 0.1.2         | 2026-04-20 | Phase 0.3 T7 observation design: §4.1 `Piece-teammate` semantically upgraded to `Prob-teammate` (mode-agnostic probability distribution; one-hot degenerate under BRIGHT/HALF_DARK, posterior under DARK). Total channel count remains 69; no schema break. §4.1 Dead/Flag-revealed clarified as observer-sorted. §4.2 pinned at 28 dims with explicit flag-reveal 4 scalars. Fixed duplicate rows in §4.1 table. |
| 0.1.3         | 2026-04-21 | Phase 0.3 T7 M1–M4 delivery: §4.1 extended with 32-channel Ataraxos-parity tail (`move_bucket` / `active_eat_bucket` / `passive_survive_bucket` / `death_reason` / `dead_at_zero`); spatial `OBS_CHANNELS` pinned at 101 (ADR-116). Global features §4.2 unchanged at 28. Indices `[0, 69)` remain bit-compatible with pre-T7 checkpoints. See ADR-114 (piece_id system) and ADR-115 (tail design). |

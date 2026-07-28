# Architecture Decision Records (ADR)

> **Scope**: One-stop record of every major design choice, with rationale, alternatives considered, and status.
> **Format**: Each ADR is numbered and immutable once accepted. Supersession creates a new ADR.
> **Index**: ADR-001 .. ADR-020 are rule decisions (matches `RULES.md` §10). ADR-100+ are engineering decisions.

---

## ADR-001: Piece fate on team surrender (Q1)

- **Status**: Accepted (2026-04-19)
- **Decision**: When a team surrenders (flag captured / both seats dead), **all remaining pieces of both teammates are removed from the board**.
- **Alternatives**:
  - B: Pieces become spoils, transferred to the surviving teammate (if any) — rejected: increases state complexity, no strategic benefit.
  - C: Pieces become neutral obstacles — rejected: would require new "neutral" ownership semantics.
- **Rationale**: Simplest model; matches Ataraxos-style clean termination; no downstream complications for belief model.
- **Ref**: RULES.md §3.5.

---

## ADR-002: Combat information broadcast (Q2) — NOVELTY OF 四国军棋

- **Status**: Accepted (2026-04-19)
- **Decision**: After every move, broadcast `{src_pos, dst_pos, event, flag_reveal_src, flag_reveal_dst, flag_captured}` to all 4 seats **identically**. **No piece types revealed**, not even to combatants.
- **Alternatives**:
  - B: Expose types to combatants only — rejected: contradicts actual 四国军棋 rules; trivializes the partial-information challenge.
  - C: Expose types to all spectators — this is Stratego's model; 四国军棋 is strictly harder.
- **Rationale**: The defining feature of 四国军棋 is that **nobody directly sees any type** (except via indirect SILING-reveal and JUNQI-capture signals). All inference must be deductive. This makes the game harder than Stratego and motivates a dedicated belief network.
- **Ref**: RULES.md §3.3, §3.4.

---

## ADR-003: Engineer path choice (Q3)

- **Status**: Accepted (2026-04-19)
- **Decision**: When multiple rail paths exist for an engineer, use the **shortest** path (BFS first-found). This affects only UI display; the game-logical outcome is the same regardless of path choice.
- **Alternatives**:
  - B: Let player pick from all paths — rejected: adds UI complexity with no strategic value.
- **Rationale**: Rules don't depend on path; path is a visualization artifact only.
- **Ref**: RULES.md §2.3.3.

---

## ADR-004: Moves per turn (Q4)

- **Status**: Accepted (2026-04-19)
- **Decision**: Exactly **1 move per turn per seat**. Turn order `SOUTH → WEST → NORTH → EAST → SOUTH...`.
- **Ref**: RULES.md §2.1.

---

## ADR-005: First-move seat (Q5)

- **Status**: Accepted (2026-04-19)
- **Decision**:
- Training: SOUTH (seat 0) always moves first; network sees only canonical frame.
  - Human-vs-AI: Human selects any seat; turn order unchanged.
- **Rationale**: A single-weight network plays all 4 seats via canonical rotation. No first-move bias because the network is symmetric.
- **Ref**: RULES.md §5.4; ARCHITECTURE.md §3.

---

## ADR-006: Stronghold attackable (Q6)

- **Status**: Accepted (2026-04-19)
- **Decision**: Strongholds can be attacked by any enemy piece satisfying movement rules.
- **Rationale**: Strategically used for **probing the enemy's landmine distribution** — because landmines must sit in the back two rows (index ≥ 20) and the flag must sit in one of the two strongholds, attacking a stronghold reveals whether it is the flag or is mine-protected. This is a general probing tool, not limited to the three-landmine formation.
- **Ref**: RULES.md §2.6.

---

## ADR-007: SILING flag-reveal rule (Q7)

- **Status**: Accepted (2026-04-19)
- **Decision**: Each seat's flag-reveal is independent. When a SILING dies (via same-rank BOMB, being bombed, or hitting a mine), its owning seat's flag is revealed. Two-sided SILING deaths reveal both flags.
- **Key inference**: BOMB event + exactly-one-flag-revealed ⟹ revealed side was SILING, other side was ZHADAN. This is a **hard deduction** the belief network must learn or be constrained to.
- **Code reference**: `legacy_engine/src/junqi.c` lines 720–735 — two independent `if (type == SILING)` blocks.
- **Ref**: RULES.md §3.2, §3.4, §7.3.1.

---

## ADR-008: Rail path blocking (Q8)

- **Status**: Accepted (2026-04-19)
- **Decision**: For any rail move (engineer or straight), all intermediate cells must be `type == NONE`.
- **Ref**: RULES.md §2.4.

---

## ADR-009: Camp occupation (Q9)

- **Status**: Accepted (2026-04-19)
- **Decision**: Any seat can occupy any of the 20 camp cells on the board (not just its own zone). First-come, first-served. Matches legacy engine.
- **Alternatives**:
  - B: Camps restricted to owning seat — rejected: legacy engine and many online variants allow all.
- **Ref**: RULES.md §2.5.

---

## ADR-010: Draw thresholds (Q10)

- **Status**: Accepted (2026-04-19)
- **Decision**:
  - `max_num_moves = 4000` (global safety net)
  - `max_num_moves_between_attacks = 200` (reset on any EAT/KILLED/BOMB/FLAG_CAPTURED)
- **Rationale**: Matches Ataraxos (Stratego SOTA). Prevents infinite stalls. Draw → all 4 seats reward 0.
- **Ref**: RULES.md §5.3.

---

## ADR-011: Teammate setup visibility in training (Q11)

- **Status**: Accepted (2026-04-19)
- **Decision**: Training mode: **teammates see each other's full setup from game start**. This is "semi-暗棋" mode.
- **Alternatives**:
  - B: Fully-hidden (暗棋): teammates also inferred — future mode; much harder training signal.
- **Rationale**: Reduces training difficulty initial curriculum; aligns with common human 四国军棋 play style.
- **Future**: Phase 3c will extend to full 暗棋 mode once semi-暗棋 converges.
- **Ref**: RULES.md §6.3.

---

## ADR-012: Skip/pass action forbidden (Q12)

- **Status**: Accepted (2026-04-19)
- **Decision**: No skip/pass action exists. If a seat has zero legal moves, that seat is immediately marked dead and its pieces removed.
- **Rationale**: Eliminates stalling exploits. Simplifies RL action space (no pass logit).
- **Alternatives** (rejected):
  - B: Legacy-style up-to-5 skips — keeps loopholes, complicates termination.
- **Ref**: RULES.md §2.2.

---

## ADR-013: No pieces in camps (Q13)

- **Status**: Accepted (2026-04-19)
- **Decision**: Setup validator **hard-rejects** any setup placing a piece on camp indices `{6, 8, 12, 16, 18}`.
- **Ref**: RULES.md §1.3 (C1).

---

## ADR-014: Piece-strength encoding

- **Status**: Accepted (2026-04-19, inherited from legacy)
- **Decision**: Smaller `ChessType` enum value = stronger piece. `SILING(5) > JUNZH(6) > ... > GONGB(13)`.
- **Rationale**: Legacy convention; preserves direct port from C to Python.
- **Caveat**: Documentation and code comments must be explicit; this is counter-intuitive to English "higher rank = stronger".

---

## ADR-015: Rule versioning

- **Status**: Accepted (2026-04-19)
- **Decision**: `RULES_VERSION = "1.0.0"` constant in `junqi_core/rules.py`. Any semantic rule change bumps the version (MAJOR for breaking, MINOR for additions, PATCH for clarifications).
- **Enforcement**: Replay files embed the version; loading a replay from a different MAJOR version fails loudly.

---

## ADR-016: Mutual destruction — attacker wins (Q14)

- **Status**: Accepted (2026-04-20)
- **Decision**: When a single `step()` simultaneously drives both teams to full defeat (all 4 seats `dead=True` in the same action), the **acting seat's team** wins. This is NOT a draw.
- **Rationale**: Standard 四国军棋 convention credits the attacker for taking the decisive initiative. It also removes a perverse incentive where the defending team could deliberately provoke mutual annihilation to force a 0/0/0/0 draw. Draws are reserved for the stalling-prevention thresholds (§5.3 Q10).
- **Alternatives considered**:
  - A: Declare a draw (0/0/0/0 for all 4 seats) — rejected, invites griefing and contradicts 四国军棋 tradition.
  - B: Give half-reward to each team (+0.5 each) — rejected, complicates RL reward normalization.
- **Implementation note**: In `junqi_core.state.check_game_over()`, when both teams are simultaneously dead, read `action.seat.team` (from the MoveResult being constructed) to determine the winner. Because the attacker's team had ≥ 1 living seat immediately before the step (namely `action.seat` itself was alive to act), "attacker's team wins" is always well-defined.
- **Ref**: RULES.md §5.2a.

---

## ADR-017: T5 validation strategy — self-consistency + reading oracle, not step-diff

- **Status**: Accepted (2026-04-20)
- **Decision**: Phase 0.2 T5 validation adopts a **three-layer strategy** instead of step-by-step diff against `legacy_engine`:
  1. **Self-consistency stress test** (`tools/stress_test.py` + `tests/test_stress_random.py`): 1000+ random self-play games; verify invariants I1–I5, reproducibility by seed, serialization round-trip, termination-reason coverage, and produce `step()` / `BeliefTensor.update()` performance baselines.
  2. **Reading oracle** (`docs/LEGACY_PARITY.md`): static code-level review of every rule decision, mapping `junqi_core` symbols/functions to their `legacy_engine/src/*.c` counterparts. No runtime coupling; the document is the audit artifact.
  3. **Runtime diff stub** (`tools/legacy_spot_check.py`): documented TODO framework for a future `ctypes`-based step-by-step diff against a refactored `libjunqicore.so`. Interface is defined now so Phase 1 CUDA-simulator parity testing can drop in.

- **Rationale**:
  - `legacy_engine` is a UDP-socket AI-search service, NOT a standalone rules library. Exposing its judgement primitives to Python requires extracting `movegen.c` / `path.c` / `event.c` / `junqi.c` into a `.so` plus a `ctypes` wrapper — a 7–10 day C-refactor effort.
  - Q10 (draw thresholds), Q12 (no-moves-dies), Q14 (mutual-destruction→attacker), and parts of Q13 (camp-empty setup hard-constraint) are **net-new rules in junqi_core v1.1.0**; legacy would diverge on every such case anyway.
  - Self-consistency + reading oracle yields **~85% of the confidence** at **~15% of the effort**. The remaining 15% risk (possible rule misinterpretation) is mitigated by:
    - 136 pytest scenarios already covering 14 Q-decisions end-to-end.
    - 55+ golden JSON cases under `tests/golden/` (move_gen, battle, siling_flag, stronghold, inference, full_game).
    - A dedicated `LEGACY_PARITY.md` function-by-function code review.

- **Alternatives considered**:
  - **A: Pure self-consistency**: no legacy reference at all. Rejected — too large a "self-validation loop" risk; future code-review becomes ad-hoc.
  - **B: Full runtime diff**: extract legacy to `.so`, link via ctypes, run 1000 games step-by-step. Rejected for Phase 0.2 — cost and maintenance burden outweigh benefits given Q10/Q12/Q14 divergence; keep as future option for Phase 1.
  - **C (chosen): Hybrid**: stress test + reading oracle + runtime-diff stub.

- **Consequences**:
  - Positive: Phase 0.2 ships on schedule; performance baseline obtained for Phase 1; CUDA-simulator Phase 1 work can reuse `stress_test.py` harness.
  - Negative: residual 10–15% risk of undetected rule misinterpretation; mitigated but not eliminated. `LEGACY_PARITY.md` becomes a living document that must be kept in sync whenever `junqi_core` rules change.
  - Forward compatibility: if/when a future phase needs step-accurate legacy comparison (e.g. for publication), `legacy_spot_check.py` stub defines the exact extension point.

- **Exit criteria for Phase 0.2 T5**:
  - ✅ `tests/test_stress_random.py` green in CI: 50 games × ≤200 steps, all invariants hold.
  - ✅ `tools/stress_test.py` can run 1000 games off-line and writes a summary CSV.
  - ✅ `step()` baseline ≥ 3000 plays/sec single-thread Python.
  - ✅ `BeliefTensor.update()` baseline ≥ 5000 updates/sec.
  - ✅ `docs/LEGACY_PARITY.md` completed: every `junqi_core/*.py` function cross-referenced to legacy code.
  - ✅ At least 4 extra golden JSONs in `tests/golden/full_game/` covering rare scenarios (Q14 variants, Q12 chains, draw-by-stall-approach).

---

## ADR-100: New Python library independent of legacy C code (D1)

- **Status**: Accepted (2026-04-19)
- **Decision**: Build `junqi_core/` as a **pure-Python** library independent of legacy C. Legacy code is read-only oracle for testing.
- **Alternatives**:
  - B: Call legacy via FFI/subprocess — rejected: Python-C boundary complicates GPU batching; legacy has historical globals and Windows coupling that would leak.
  - C: pybind11/CFFI wrapper — rejected: same concerns as B plus binding maintenance cost.
- **Rationale**:
  1. Legacy code has known technical debt (global variables, cross-file coupling, old threading model).
  2. Pure-Python enables numpy/torch vectorization natural for GPU batching.
  3. Legacy as oracle via cross-check testing gives correctness guarantees without runtime coupling.
- **Risk**: Duplicate implementation effort. Mitigated by aggressive golden-case testing (100+ scenarios + 1000-game parity test).
- **Ref**: ARCHITECTURE.md §6.

---

## ADR-101: Keep legacy GUI as interim visualization (D2)

- **Status**: Accepted (2026-04-19)
- **Decision**: Keep `legacy_gui/` unchanged through Phase 5. Use it as a debugging visualization tool during Phase 0-3 (new engine → JSON replay → legacy_gui adapter).
- **Alternatives**:
  - A: Retire legacy GUI immediately — rejected: no replacement until Phase 4+, debugging would lack visual feedback.
  - C: Maintain both GUIs in parallel indefinitely — rejected: double maintenance cost.
- **Rationale**: Gives free debugging UI during core engine validation; web GUI development is Phase 4.
- **Retirement**: Phase 6, once `junqi_viz` is feature-complete.

---

## ADR-102: Canonical rotation for multi-seat single-network play (D3)

- **Status**: Accepted (2026-04-19)
- **Decision**: Observation tensors fed to the network are always in **canonical frame** where the acting seat sits at the canonical SOUTH (bottom) position. See ARCHITECTURE.md §3 for math.
- **Alternatives**:
  - B: 4 separate network weights, one per seat — rejected: 4× parameters, 4× gradient variance, no parameter sharing across symmetric positions.
  - C: Concatenate seat index as input feature — rejected: network still has to internally encode rotation; worse sample efficiency.
- **Rationale**:
  1. Board has 4-fold rotational symmetry → single network weight should exploit this.
  2. Standard technique in Ataraxos, AlphaZero-chess variants.
  3. Rotation math is trivially bijective and <1 µs per observation.
- **Ref**: ARCHITECTURE.md §3.

---

## ADR-103: Monorepo layout

- **Status**: Accepted (2026-04-19)
- **Decision**: Single repo `JunQi/` containing `junqi_core/`, `junqi_rl/`, `junqi_viz/`, `legacy_engine/`, `legacy_gui/`, `tests/`, `docs/`, `tools/`.
- **Alternatives**:
  - B: Separate `JunQi-RL` repo with legacy as git submodule — rejected: splits history, complicates golden-test CI.
- **Rationale**: Simpler CI, single source of truth, all cross-checks in one place.
- **Ref**: ARCHITECTURE.md §1.

---

## ADR-104: PyTorch as primary framework

- **Status**: Accepted (2026-04-19)
- **Decision**: PyTorch 2.1+ with CUDA 12.x for all training code.
- **Alternatives**:
  - B: JAX — rejected: user preference for PyTorch; richer ecosystem of RL libs.
  - C: TensorFlow — rejected: declining RL research usage.
  - D: Ray/RLlib — deferred; may adopt for multi-node Phase 3.
- **Ref**: ARCHITECTURE.md §9.1.

---

## ADR-105: Golden test-driven development for Phase 0

- **Status**: Accepted (2026-04-19)
- **Decision**: Phase 0.1 produces a 100+ scenario JSON test set (by hand + auto-generated from legacy oracle) **before** any `junqi_core` code is written.
- **Alternatives**:
  - B: Code-first, test-later — rejected: bugs in rule implementation compound; harder to catch late.
- **Rationale**:
  1. Rules documented in prose (`RULES.md`) must be matched against concrete scenarios to catch ambiguity.
  2. Golden cases guide implementation; any code change that breaks a case triggers a review of the rule itself.
  3. Cross-checks implementations against legacy engine behavior.

---

## ADR-106: Observation channel count (69 initial)

- **Status**: Draft (2026-04-19); subject to Phase 2 ablation
- **Decision**: Initial observation tensor has ~69 channels (see ARCHITECTURE.md §4.1 for breakdown).
- **Open**: Actual channel count may be tuned in Phase 2. Commit to the **structure** (blocks: own/teammate/enemies/belief/static/dynamic), not exact counts.

---

## ADR-107: Flat action space with masked softmax

- **Status**: Accepted (2026-04-19)
- **Decision**: Action space is flat `83521 = 17⁴`; masked softmax with `-inf` on illegal actions.
- **Alternatives**:
  - B: Factored (src policy + dst policy per-piece) — rejected: per-piece conditioning complicates training, less parallelizable.
  - C: Pointer network — rejected: pre-mature optimization; flat fits easily on modern GPUs.
- **Rationale**: Flat space is simple, parallel-friendly, and fits T4 memory comfortably.

---

## ADR-108: Replay format JSON by default

- **Status**: Accepted (2026-04-19)
- **Decision**: Replays default to JSON (human-readable, version-controlled). MessagePack binary optional for large-scale training logs.
- **Invariant**: Every replay file embeds `rules_version`. Loading a replay from a MAJOR-version mismatch fails loudly.

---

## ADR-109: Draw rewards = 0 for all seats

- **Status**: Accepted (2026-04-19)
- **Decision**: On draw (Q10 thresholds hit), all 4 seats receive reward 0.
- **Alternatives**:
  - B: Small penalty (-0.1) to discourage stalling — rejected: RL already penalized via move-count shaping if needed; cleaner to keep 0/±1 outcomes.

---

## ADR-110: Team reward (shared ±1)

- **Status**: Accepted (2026-04-19)
- **Decision**: Winner team's both seats receive `+1`; loser team's both seats receive `-1`. No per-seat reward differentiation.
- **Rationale**: 四国军棋 is fundamentally cooperative-within-team; per-seat rewards would incentivize selfish play contrary to team objective.
- **Ref**: RULES.md §5.2.

---

## ADR-111: Rename Seat enum to cardinal directions (SOUTH/WEST/NORTH/EAST)

- **Status**: Accepted (2026-04-20, Phase 0.3 T7)
- **Decision**: Rename `Seat` enum members from the legacy first-person-view labels `HOME / RIGHT / OPPS / LEFT` to cardinal-direction labels `SOUTH / WEST / NORTH / EAST`, matching the fixed world-frame board geometry.
  - **Integer values are preserved** (`SOUTH=0, WEST=1, NORTH=2, EAST=3`) so binary replays, the ctypes bridge to `legacy_engine/libjunqicore.so`, and all saved golden JSON `seat`/`turn` fields remain wire-compatible. Only the Python-visible symbolic names change.
  - A companion mapping `LEGACY_DIR_TO_SEAT` / `SEAT_TO_LEGACY_DIR` lives in `junqi_core/rules.py` for code that talks to the legacy C engine (`enum ChessDir {HOME, RIGHT, OPPS, LEFT}` remains as-is — it is an external contract).
  - The two enemy-of-acting-seat properties are also renamed:
    - `left_side_enemy`  = seat `(s+1)%4` — lands at canonical **x ∈ [0, 5]** (image LEFT half) after rotation.
    - `right_side_enemy` = seat `(s+3)%4` — lands at canonical **x ∈ [11, 16]** (image RIGHT half) after rotation.
    - Old properties `left_enemy`, `right_enemy`, `ccw_enemy`, `cw_enemy` are **removed entirely** (no deprecation window) because the rename is being done before any training/publication artifacts exist; a clean break is cheaper than maintaining aliases for a project still at Phase 0.3.
- **Rationale**:
  1. **The legacy names are misleading after canonical rotation.** Board layout is world-fixed, but the legacy labels encode the *first-person viewpoint of the seat at bottom*. When the network is queried for seat `WEST` (which sits on the world-left side of the board), we rotate the tensor so that `WEST` appears at the canonical bottom position. In the rotated frame, "my LEFT enemy" (the legacy name for seat `EAST`, sitting on world-right) ends up on the image's **right** half. This contradiction — the property named `left_enemy` living on the image right — caused repeated reasoning bugs while writing observation channels and inference rules (see PHASE_0.2_ACCEPTANCE.md §5 W5).
  2. **Cardinal directions are frame-independent.** `SOUTH` means world-frame south under every seat's canonical rotation. Debug dumps of rotated tensors match intuition: the piece visible on the image left half IS `left_side_enemy`, always.
  3. **The rename cost is one-shot and small.** All reads of the old names lived in ~12 Python files + 33 golden JSONs + 6 Markdown docs. No training artifacts, no published papers, no external dependencies reference the old names. Fixing this before Phase 1 (RL training start) prevents compounding confusion during Phase 1–4 work.
- **Alternatives considered**:
  - **A: Keep legacy names, add a docstring warning.** Rejected — docstrings do not stop human debuggers from instinctively associating "left" with the image's left side.
  - **B: Rename with CCW/CW enemy suffix (the intermediate proposal before this ADR).** Rejected — "counter-clockwise enemy" is correct geometry but still abstract; debuggers think in terms of "who is on my left in the image I'm looking at", not rotation direction.
  - **C: Cardinal names + keep old aliases for 1 release.** Rejected — with no external consumers, aliases would just slow cleanup of internal call sites.
- **Consequences**:
  - `legacy_engine/src/junqi.h` still uses `enum ChessDir {HOME, RIGHT, OPPS, LEFT}`. Any code crossing the ctypes bridge MUST go through `LEGACY_DIR_TO_SEAT`.
  - Observation channel names in ARCHITECTURE.md §4.1 (previously "Belief-left / Belief-right", briefly "Belief-ccw / Belief-cw") are now "Belief-left-side / Belief-right-side", matching which image half they occupy.
- **Ref**: `junqi_core/rules.py::Seat`, `junqi_core/rotation.py`, ARCHITECTURE.md §3.2, LEGACY_PARITY.md row on `ChessDir`.

---

## ADR-112: PPO first as Phase 1-2 RL algorithm; R-NaD as optional Phase 3 upgrade

- **Status**: Accepted (2026-04-20, Phase 0.3 decision-gate)
- **Decision**: Phase 1-2 will implement **PPO** (Proximal Policy Optimization) as the primary self-play RL algorithm. **R-NaD** (Regularized Nash Dynamics, the Ataraxos algorithm) is a *Phase 3 optional upgrade*, to be started only if one of the following triggers fires:
  1. PPO converges but the resulting policy is exploited by a structured adversary (classic PPO failure mode).
  2. A team-game Nash equilibrium analysis / publication target makes the Nash-dynamics guarantee worth the engineering cost.
  3. Other teams' 2v2-team R-NaD results become available and reproduce cleanly on 四国军棋.
- **Rationale**:
  1. **Engineering reuse is ~100%**: vectorized env, rollout buffer, league-of-opponents (FSP/PFSP), reward shaping, observation pipeline, action masking — all identical between PPO and R-NaD. Building PPO first buys no wasted work; it is strictly a prefix of the R-NaD project.
  2. **PPO is a mandatory baseline anyway**: any subsequent R-NaD evaluation needs a PPO control group to prove value. Skipping PPO would leave R-NaD with no internal baseline.
  3. **Team / cooperative mixture**: 四国军棋 is a 2v2 cooperative-within-team + zero-sum-between-teams game. R-NaD's convergence theorem is proven for 2-player zero-sum only; the team-level extension is **an unpublished extrapolation** as of 2026. PPO, by contrast, has strong precedents (Dota OpenAI Five, AlphaStar, MAPPO) on 2v2+ team settings.
  4. **Risk / ROI**: PPO baseline has ~3-week budget with low failure risk. R-NaD has 4-6-week budget, high hyper-parameter sensitivity, and at least one open research question (team credit assignment under neural replicator dynamics).
- **Alternatives considered**:
  - **A: R-NaD directly** — rejected for the 2v2-team extrapolation risk (no public reference implementation, JAX-only Ataraxos code).
  - **B: MAPPO / HATRPO** — considered. Heterogeneous-agent PPO variants fit 2v2 teams slightly better, but their complexity gain over vanilla PPO is not warranted at Phase 1; can be swapped in later if vanilla PPO shows team-coordination failure modes. The rollout infrastructure stays identical.
  - **C: AlphaZero-style MCTS + PV** — rejected: 四国军棋's imperfect-information state (belief tensor) does not factor cleanly into MCTS's perfect-information tree search; would require POMDP-MCTS variant, which is itself unproven at this scale.
- **Consequences**:
  - Phase 0.3 `replay.py` JSON schema MUST reserve optional fields `policy_logits`, `value` on each step — zero cost now, allows both PPO and R-NaD to log training telemetry later.
  - `junqi_rl/training/ppo.py` is the Phase 1 target. `junqi_rl/training/r_nad.py` is a placeholder until Phase 3 decision-gate.
  - ARCHITECTURE.md Roadmap §10 wording "Phase 3 PPO self-play + belief network" is authoritative.
- **Ref**: PHASE_0.2_ACCEPTANCE.md §5.2 (Ataraxos divergence table), §5.3 O3 (R-NaD optimization entry).

---

## ADR-113: `legacy_gui` adapter — replay-only; debug tooling lives in Python-side utilities

- **Status**: Accepted (2026-04-20, Phase 0.3 decision-gate)
- **Decision**: The Phase 0.3 T11 `legacy_gui` adapter delivers **replay-viewer functionality only** (~1 day scope). Advanced debug tooling (belief heat-maps, policy-distribution visualization, reward-curve dashboards) is delivered as **standalone Python-side scripts under `tools/`** (matplotlib / plotly based), and not as additions to the `legacy_gui` C/C++ code.
- **Scope of T11 `legacy_gui` adapter**:
  - Input: `replay.json` produced by `junqi_core/replay.py::Recorder`.
  - Output: a byte stream / command sequence consumed by `legacy_gui` in its existing "playback" mode. No modification to `legacy_gui` C/C++ source code.
  - Deliverable: `tools/legacy_gui_replay_adapter.py`, `docs/REPLAY_VIEWER.md` (usage), 2-3 pytest round-trip tests.
- **Scope explicitly out-of-scope for T11**:
  - Any new Qt/MFC UI panels in `legacy_gui`.
  - Live belief visualization inside `legacy_gui`.
  - Real-time policy inference display inside `legacy_gui`.
  - These are instead delivered (on demand) as Python-side tools — see below.
- **Python-side debug tooling (Phase 2+, on-demand)**:
  - `tools/belief_viewer.py` — matplotlib heat-map of `BeliefTensor`; standalone CLI.
  - `tools/policy_viewer.py` — plot 17×17×17×17 softmax on board overlay.
  - `tools/reward_curves.py` — plot per-episode rewards and per-seat value estimates.
  - Target effort: ~2 hours per tool, delivered only when a concrete debugging need arises.
- **Rationale**:
  1. **`legacy_gui` has a retirement date**: ARCHITECTURE.md §10 Phase 6 explicitly retires `legacy_gui` once `junqi_viz` is feature-complete. Investing 2+ days in a C++ dashboard that is deleted in 3-6 months is a liability, not an asset.
  2. **Python-side tools migrate cleanly to the web frontend**: Phase 4's `junqi_viz` is a FastAPI + Vue/React stack. Python debug utilities that emit JSON / PNG can be reused via HTTP — whereas anything written in legacy Qt requires a full rewrite.
  3. **Iteration velocity**: a matplotlib figure is 10× faster to iterate than a legacy Qt panel. Training debug sessions need turnaround measured in minutes, not hours.
  4. **Scope discipline**: Phase 0.3 has 5 concurrent tracks (T7-T12). Capping T11 at 1 day is what keeps the whole phase on schedule (~5 working days total).
- **Alternatives considered**:
  - **A: Full game-over dashboard inside `legacy_gui` (3 days)** — rejected: the dashboard features duplicate what Phase 4 `junqi_viz` will build from scratch; also requires C++ changes during a "read-only legacy" period (ADR-100 D1).
  - **B: No adapter at all; only Python-side replay viewer** — rejected: having *some* interactive playback matters for reviewing full games; the legacy GUI already knows how to render the 17×17 board correctly and handle user scrubbing. Re-implementing that in Python before Phase 4 Web UI would be wasted work.
- **Consequences**:
  - Any future "I need to watch a game visually" request routes through the replay viewer adapter or Phase 4's Web UI.
  - Any "I need to see belief / policy / reward curves" request is served by a Python-side `tools/*.py` script. First such request triggers ~2h of focused scripting; subsequent uses re-run the script.
  - Phase 4 `junqi_viz` scope is unchanged: it will provide the full dashboard natively.
- **Ref**: ARCHITECTURE.md §10 (Roadmap), ADR-100 (new engine independence), ADR-101 (legacy GUI retention policy).

---

## ADR-114: Global piece-identity system — `piece_id ∈ [0, 119]` + `zero_board` + `deaths` registry

- **Status**: Accepted (2026-04-21, Phase 0.3 T7 M1–M3)
- **Decision**: Introduce a stable, game-wide piece identifier `piece_id` and two companion registries on `GameState`:
  1. **`piece_id: int ∈ [0, 119]`** assigned in `setup.build_initial_state()` as `piece_id = seat.value * 30 + setup_slot`. Camp slots (5 per seat) are skipped (their slot has no piece), so each seat owns up to 25 live `piece_id`s; the full range `[0, 119]` gives a compact, collision-free global namespace. Identity is **set once at `new_game` and never re-assigned** — death removes the piece from `GameState.pieces` but NEVER reclaims its id.
  2. **`GameState.zero_board: dict[(x,y), PieceRef]`** — an **immutable snapshot** of the initial layout that carries every `piece_id` → starting cell → `seat` mapping. Shared by reference across all cloned states (it is never mutated), so `step()` pays no clone cost here.
  3. **`GameState.deaths: dict[int, DeathInfo]`** — an append-only registry recording, for every dead piece, a frozen `DeathInfo(piece_id, reason, death_loc, step)`.
  4. **`DeathReason ∈ {KILLED_BY_ENEMY, HIT_MINE_OR_BOMB, MUTUAL}`** (3 values, per D-2 below). Classification is done by `rules.classify_death_reason(own_piece, opponent_piece, event, own_is_attacker)`.
  5. **`GameState.piece_state: dict[int, PieceState]`** — per-living-piece mutable counters `{move_count, active_eat_count, passive_survive_count}`. Keyed by `piece_id`. Dead pieces are popped from this dict so **"in `piece_state` ⟺ currently alive on the board"** is a hard liveness invariant (enforced by tests).
- **Decision points locked in during design review (2026-04-21)**:
  - **D-1 counter storage**: **(b) external `PieceState` dict** — keeps `PieceRef` `frozen=True`, so move_gen's per-state caches remain keyed by hashable refs with unchanged hash. *Alternative (a) mutable `PieceRef` with inline counters was rejected because it would silently invalidate every existing cache.*
  - **D-2 BOMB death reason**: **(a) always `MUTUAL`**, regardless of whether the defender was a `DILEI` / `ZHADAN` or a rank-tied peer. Same-rank bomb, ranked-vs-`ZHADAN`, and any other same-`Event.BOMB` outcome are **semantically indistinguishable** from the observation's perspective. *Alternative (b) routing `HIT_MINE_OR_BOMB` for `ZHADAN`-triggered bombs was rejected — it would split an information-theoretically single event into two labels, hurting sample efficiency.*
  - **D-3 enemy A/B/C visibility filter**: **(a) filter on one-hot belief** — the observer only sees an enemy piece's bucket channels when it has pinned down that cell's type to a single hypothesis (belief is one-hot, within `1e-6`). Mirrors Ataraxos's `piece.visible` gate.
  - **D-4 D/E groups visibility**: **no filter** — both death-reason (D) and dead-at-zero (E) channels emit freely for both sides. D is a historical event (broadcast to all observers at the moment it happens — matches ADR-002 info-broadcast), and E is a derivative of `zero_board` which is public at game start.
  - **D-5 `piece_id` in observation tensor**: **not exposed** — `piece_id` is a backend bookkeeping key (for `piece_state` / `deaths`), NOT a network input channel. This keeps the observation tensor permutation-invariant w.r.t. id assignment.
- **Rationale**:
  1. Ataraxos's `Piece.piece_id` + `zero_boards` + `deaths[2][5]` bitmap is a proven design for exposing piece-lifecycle features in a fully-convolutional observation. By mirroring it we keep the door open to porting Ataraxos's threat/evasion/protection groups in a later phase without re-keying anything.
  2. The three registries are **strictly additive** to the existing `GameState`. No existing field semantics change. `PieceRef` gains exactly one field (`piece_id`); all other downstream code is unaffected.
  3. Decoupling counters from `PieceRef` (D-1b) preserves the "immutable piece reference" invariant that `move_gen.py` relies on for memoisation. Tests confirm zero break to the existing 256-test regression baseline.
- **Alternatives considered**:
  - **A: No piece_id, re-key by `(seat, piece_type, pos)` tuple.** Rejected — ambiguous for multiple pieces of the same type on the same seat; break semantically when a piece moves.
  - **B: Mutable counters on `PieceRef`.** Rejected per D-1 above (cache invalidation).
  - **C: Sparse bitmap (Ataraxos's `deaths[2][5]`) instead of dict.** Rejected — Python dict gives O(1) lookup and clearer code; 120 entries is trivial memory. Bitmap would be worth it only on GPU.
- **Consequences**:
  - `GameState` grows three fields (`zero_board`, `deaths`, `piece_state`). `clone()` shallow-copies `deaths` / `piece_state` (mutated in `step()`) and shares `zero_board` by reference.
  - `setup.build_initial_state()` now returns the extended `GameState` with registries populated; golden replays recorded before T7 remain bit-compatible on the wire (replays don't serialize `piece_id` yet — see §8 of this ADR's impact list).
  - Every `step()` that produces a death now appends exactly one `DeathInfo` per dead piece into `deaths`, and updates `piece_state` for the surviving attacker / defender.
- **Ref**: `junqi_core/state.py::{DeathInfo, PieceState, DeathReason}`, `junqi_core/rules.py::DeathReason + classify_death_reason`, `docs/PHASE_0.3_T7_TODO.md` §3 decision table, `tests/test_piece_id_assignment.py`, `tests/test_piece_counters.py`, `tests/test_death_info.py`.

---

## ADR-115: T7 observation tail — 32 channels for Ataraxos-parity piece-lifecycle features

- **Status**: Accepted (2026-04-21, Phase 0.3 T7 M4)
- **Decision**: Append 5 channel groups (32 planes total) to the end of `CHANNEL_LAYOUT`. The existing 69 channels (ADR-106) keep their order and meaning unchanged; new groups live strictly in the tail, so channel indices `[0, 69)` are bit-compatible with pre-T7 models.

  | Group | Name (in `CHANNEL_LAYOUT`) | Size | Split | Anchor cell | Semantics |
  |---|---|---:|---|---|---|
  | A | `move_bucket` | 8 | 4 ours + 4 theirs | piece's current cell | Exact-match bucket of `PieceState.move_count` → plane `{0, 1, 2, ≥3}`. Exactly ONE plane fires per living piece. |
  | B | `active_eat_bucket` | 8 | 4 ours + 4 theirs | piece's current cell | Cumulative bucket of `active_eat_count` → planes `{0, ≥1, ≥2, ≥3}`. A piece with count `k` lights planes `[0..min(k, 3)]` inclusive. |
  | C | `passive_survive_bucket` | 8 | 4 ours + 4 theirs | piece's current cell | Cumulative bucket of `passive_survive_count`, same semantics as B. |
  | D | `death_reason` | 6 | 3 ours + 3 theirs | `DeathInfo.death_loc` | One plane per `DeathReason ∈ {KILLED_BY_ENEMY, HIT_MINE_OR_BOMB, MUTUAL}` × side. Anchored at the combat dst cell. |
  | E | `dead_at_zero` | 2 | 1 ours + 1 theirs | `zero_board` cell | 1 iff that `piece_id` appears in `GameState.deaths`. |

- **Ours / theirs split rule**: `ours` = `observer` ∪ `observer.teammate`; `theirs` = the two enemies. The split is **team-based, not visibility-based**: even under DARK (where the teammate's type is unknown), teammate counters are still stored in the `ours` half. Tests (`test_teammate_move_count_stays_on_ours_half_under_dark`) pin this behaviour.
- **Visibility filter (per ADR-114 D-3 / D-4)**:
  - **A / B / C groups** — the `theirs` half fires for an enemy piece ONLY when `observer`'s `BeliefTensor.get(pos)` is (numerically) one-hot at that cell. This mirrors Ataraxos's `piece.visible` gate and prevents a DARK-mode information leak of enemy counter progress.
  - **D / E groups** — no filter. Death events are public history (consistent with ADR-002's post-step broadcast); zero-board layout is public from `new_game`.
- **Route-A invariant**: dead pieces emit no signal in groups A/B/C. Their contribution lives exclusively in D (death anchor) and E (zero anchor). This keeps A/B/C purely about **current** piece state and avoids double-counting over a piece's lifetime.
- **Rationale**:
  1. **Ataraxos parity** for piece-lifecycle features is the prerequisite for future port of the `threatened / evaded / actively_adjacent / protected` groups (Ataraxos Ch. 205–354). Adding them now locks in the routing and team-split convention, so later groups become pure appending.
  2. **Bucketization** (instead of raw counter values) matches Ataraxos's output design and removes the need for network input normalisation. The exact-match bucket for `move_count` is information-preserving for counts ≤ 2 (observed typical range early-game); the saturated ≥3 plane aggregates the long tail where the exact value is no longer strategically meaningful.
  3. **Tail-append layout** means network input conv1 weights of a pre-T7 checkpoint can be zero-padded on the channel axis to resume training — no need to restart BRIGHT curriculum.
- **Alternatives considered**:
  - **A: Raw counter scalar planes (`move_count / MAX_MOVES` broadcast).** Rejected — requires network to learn its own quantisation, and loses the exact-match signal for `move_count ∈ {0, 1, 2}` which Ataraxos shows is strategically distinct.
  - **B: Cumulative bucketing for move_count (same as eat / survive).** Rejected — move direction matters (a piece that has walked 2 steps and turned back is different from a piece that walked 2 and kept going); the exact-match bucket forces the network to distinguish freshly-moved from repeatedly-moved pieces.
  - **C: Per-type death planes (like Ataraxos Ch. 109–130).** Rejected for this pass — would multiply D from 6 → 66 channels and duplicates information already in `Belief-left-side` / `Belief-right-side`. Reserved for a later Phase 1 ablation if needed.
- **Consequences**:
  - `OBS_CHANNELS = 101` (see ADR-116). `CHANNEL_LAYOUT` keys: unchanged 11 keys + 5 new (`move_bucket`, `active_eat_bucket`, `passive_survive_bucket`, `death_reason`, `dead_at_zero`).
  - `observation.py` imports `DeathReason` and exports `MOVE_BUCKET_COUNT = ACTIVE_EAT_BUCKET_COUNT = PASSIVE_SURVIVE_BUCKET_COUNT = 4`, `DEATH_REASON_COUNT = 3` as public constants so tests and downstream tooling can index the half-splits without magic numbers.
  - Observation build-time: measured overhead < 20% of the prior 69-channel baseline (M6 target); no allocation-per-step regression because the tail planes are written into the pre-allocated `(101, 17, 17)` buffer.
- **Ref**: `junqi_core/observation.py` tail writers `_move_bucket_channels / _active_eat_bucket_channels / _passive_survive_bucket_channels / _death_reason_channels / _dead_at_zero_channels`, `tests/test_observation_t7.py` (28 cases).

---

## ADR-116: Observation channel count pinned at 101 (69 + 32)

- **Status**: Accepted (2026-04-21, Phase 0.3 T7 M4). Supersedes the "69 initial" status in ADR-106.
- **Decision**: `OBS_CHANNELS = 101`, with the channel layout frozen in `junqi_core/observation.py::CHANNEL_LAYOUT`. Any future addition appends to the tail; insertions or reorderings require a new ADR and a model retrain path.
- **Layout summary** (exact per-group sizes):
  ```text
  [  0..12)  piece_own                  12
  [ 12..24)  prob_teammate              12
  [ 24..25)  dark_teammate               1
  [ 25..26)  piece_left_side_enemy       1
  [ 26..27)  piece_right_side_enemy      1
  [ 27..39)  belief_left_side           12
  [ 39..51)  belief_right_side          12
  [ 51..55)  dead_flags                  4
  [ 55..59)  flag_revealed               4
  [ 59..65)  board_static                6
  [ 65..69)  turn_history                4
  [ 69..77)  move_bucket                 8   ← T7 / ADR-115
  [ 77..85)  active_eat_bucket           8   ← T7 / ADR-115
  [ 85..93)  passive_survive_bucket      8   ← T7 / ADR-115
  [ 93..99)  death_reason                6   ← T7 / ADR-115
  [ 99..101) dead_at_zero                2   ← T7 / ADR-115
  ```
- **Guarantees for future training**:
  1. Indices `[0, 69)` are **bit-compatible with any pre-T7 checkpoint**. Fine-tuning a pre-T7 model on the 101-channel schema is a zero-padding operation on the input conv's channel axis.
  2. The order of the 11 pre-T7 groups is frozen. Any future ADR reordering them MUST explicitly break compatibility and bump the observation schema version.
  3. The 5 T7 tail groups' internal order (A → E) and per-group sizes are frozen; any addition appends strictly after `dead_at_zero`.
- **Rationale**:
  - A single pinned number (`OBS_CHANNELS`) replaces ADR-106's "69 initial, tunable" draft. After the T7 design review we now have empirical evidence (28 new unit tests + route-A invariants) that all five new groups add strategically distinct signals, so there is no reason to keep them in "draft" status.
  - Keeping the number global and documented prevents a common drift failure mode where individual modules silently disagree on channel count. `observation.py::_self_check()` raises at import time if the layout drifts.
- **Alternatives considered**:
  - **A: Leave as "69 initial, subject to Phase 2 ablation" (ADR-106 original wording).** Rejected — defers the decision past the Phase 1 RL start, which would require retraining if we land T7 mid-training.
  - **B: Pin to 101 but allow per-mode (BRIGHT / HALF_DARK / DARK) variations.** Rejected — the whole point of the `Prob-teammate` design (ADR-106 layout) was to give ONE layout for all three modes. Splitting by mode would break curriculum.
- **Consequences**:
  - `tests/test_observation.py::test_channel_layout_totals` pins `OBS_CHANNELS == 101` and all 16 group sizes; CI catches drift.
  - ARCHITECTURE.md §4.1 table updated to 101 with the five new rows.
  - The "tunable during Phase 2 ablations" escape hatch from ADR-106 is **closed** — Phase 2 may introduce NEW tail channels (via new ADRs) but MUST NOT renumber the existing 101.
- **Ref**: ADR-106 (superseded in part), ADR-114 (piece_id system), ADR-115 (T7 tail design), `junqi_core/observation.py::CHANNEL_LAYOUT`.

---

## ADR-117: Array-of-Structures → Structure-of-Arrays GameState

- **Status**: Accepted (2026-04-21, Phase 0.4 M1). Foundation for all Phase 0.4 perf work.
- **Decision**: The internal storage of `GameState` is migrated from nested Python dicts (`dict[(x,y), PieceRef]`, `dict[pid, PieceState]`, `dict[pid, DeathInfo]`, `dict[pid, PieceRef]` for `zero_board`, `dict[Seat, SeatInfo]`) to a flat Structure-of-Arrays (SoA) layout with `numpy.ndarray` columns indexed by `piece_id ∈ [0, 120)`, flat cell index `flat = y*17 + x ∈ [0, 289)`, or `seat ∈ [0, 4)`.
- **Storage columns** (all `numpy.ndarray`, frozen shapes):
  ```text
  # piece-indexed (120,)  -------------------------------------------
  alive            bool      — liveness flag
  piece_type       int8      — PieceType.value (0..13); -1 if unassigned
  piece_seat       int8      — Seat.value (0..3)
  pos_x, pos_y     int8      — current cell; both -1 when dead
  zero_x, zero_y   int8      — zero-board cell; both -1 for camp pids
  move_count       int16
  active_eat       int16
  passive_surv     int16
  death_reason     int8      — -1 if alive; else DeathReason.value
  death_step       int16
  death_loc_flat   int16     — -1 if alive
  # cell-indexed (289,) ---------------------------------------------
  cell_piece_id    int16     — pid of piece at flat cell; -1 if empty
  # seat-indexed (4,) -----------------------------------------------
  seat_dead           bool
  seat_flag_revealed  bool
  ```
- **Footprint**: ~1.8 KB per GameState (vs ~15–30 KB for the dict-based representation, depending on alive count). The entire state fits in a single L1 cache line group.
- **API compatibility (read-only views)**:
  - `state.pieces` remains available as a `_PieceMapView` (lazy `__getitem__`/`items`/`values`/`__contains__` built on the ndarrays). Used only by cold paths: `to_dict`, debug dumps, `piece_at`, `legal_moves_from_pos`, and all `tests/*`.
  - `state.info`, `state.piece_state`, `state.deaths`, `state.zero_board` each get analogous lazy views.
  - `PieceRef`, `PieceState`, `DeathInfo`, `SeatInfo` dataclasses are retained as value objects; views construct them on demand.
  - Hot paths (`observation.py`, `move_gen.py`, `info_model.py`) bypass views and read/write the ndarrays directly.
- **Hot-path API additions**:
  - `GameState.step_inplace(action) -> MoveResult` — mutates `self` in place. Used by RL rollout and `VectorJunqiEnv`. Saves the `clone()` cost when the previous state is no longer referenced (strict single-line trajectory).
  - `GameState.step(action)` is retained (preserves the immutable-snapshot contract used by MCTS / transposition tables); internally it is `self.clone().step_inplace(action)`.
  - `GameState.clone()` becomes ~12 `ndarray.copy()` calls (~1–2 µs), down from the current nested-dict rebuild.
- **Rationale**:
  - Removes the dict-allocation-per-step tax (S2 in the perf debt pass).
  - Is the enabler for ADR-118 (vectorized observation writers), ADR-120 (tensorized belief), ADR-121 (batched observation builder), ADR-125 (move-gen lookup tables) — every downstream optimization assumes O(1) piece lookup by pid and contiguous per-piece columns.
  - Mirrors the Ataraxos SoA layout, so a future CUDA kernel port is a transliteration rather than a redesign.
- **Compatibility**:
  - All 14 test files and all 52 golden JSON scenarios must continue to pass unchanged. Enforced by keeping the view layer bit-equivalent to the pre-migration public API.
  - `to_dict`/`from_dict` schema bumped to `state_version: "2.0"` to include T7 columns (alive / piece_type / piece_seat / pos / zero / counters / deaths). v1.x replays still loadable (counter/death columns seeded to zeros / empty).
- **Alternatives considered**:
  - **A: Keep dicts, add a C extension later.** Rejected — C-extension rewrite of a dict-based Python engine still pays the Python-layer cost at the dict/AoS boundary; SoA ports naturally.
  - **B: Full destructive migration, drop compatibility views.** Rejected — would force ~15 test/tool files to be rewritten with no perf benefit on cold paths.
- **Consequences**:
  - `GameState.step()` target throughput: **≥80 k/s** (from ~16 k/s).
  - `GameState.clone()` target: **≤2 µs** (from ~20 µs).
  - `to_dict` / `from_dict` MUST round-trip all T7 columns byte-identically (closes L5 in the perf debt pass).
- **Ref**: ADR-114 (piece_id encoding preserved), ADR-118, ADR-119, ADR-120, ADR-121, ADR-125.

---

## ADR-118: Pre-allocated observation buffers + `ObservationBuilder.build_into`

- **Status**: Accepted (2026-04-21, Phase 0.4 M2).
- **Decision**: Observation construction becomes a **stateful builder** that owns pre-allocated `(OBS_CHANNELS, 17, 17)` float32 and `(OBS_GLOBAL_DIMS,)` float32 buffers. The public API is:
  ```python
  class ObservationBuilder:
      def build_into(
          self, state: GameState, belief: BeliefTensor, observer: Seat,
          out_spatial: np.ndarray,   # (C, H, W) float32, caller-owned
          out_global:  np.ndarray,   # (G,)      float32, caller-owned
      ) -> None: ...

      def build(self, state, belief, observer) -> ObservationTensor:
          """Convenience: allocates internal buffers (cold path only)."""
  ```
- **Removed**: the top-level `observation.build_observation(state, belief, observer) -> ObservationTensor` function. All RL code migrates to `ObservationBuilder`. Rationale: the old function allocated a fresh 117 KB tensor + 12+ sub-arrays per call; keeping it as a shim would be a silent perf trap in hot code.
- **Migration shims kept**: `channel_name(idx)`, `CHANNEL_LAYOUT`, `OBS_CHANNELS`, `OBS_GLOBAL_DIMS`, `ObservationTensor` are unchanged.
- **Vectorized writers**: all 16 channel groups are rewritten to numpy fancy-indexing on the ADR-117 SoA columns — a typical writer collapses from a per-piece Python loop into ~3 vectorized ops. Example (piece_own):
  ```python
  mask   = state.alive & (state.piece_seat == observer_val)
  pids   = np.nonzero(mask)[0]
  chidx  = _PIECE_TYPE_TO_CH[state.piece_type[pids]]
  valid  = chidx >= 0
  ys, xs = state.pos_y[pids[valid]], state.pos_x[pids[valid]]
  out_spatial[chidx[valid], ys, xs] = 1.0
  ```
- **Rotation strategy**: keep `np.rot90` (zero-copy view) as today; consumers that materialize (torch conversion, network input) pay the copy exactly once. Deferred: per-seat precomputed permutation tables (ADR-124-B, not needed at this stage).
- **Rationale**:
  - Target: **≤0.2 ms per build_into** (from ~1.35 ms), zero per-call Python allocations.
  - S1 and S4 of the perf debt pass resolved.
- **Consequences**:
  - `simulator.py` and `junqi_rl/**` MUST hold an `ObservationBuilder` instance and call `build_into` with their own pre-allocated buffers.
  - `tests/test_observation*.py` call `ObservationBuilder().build(...)` (cold-path convenience form).
- **Ref**: ADR-117, ADR-121.

---

## ADR-119: Flat action space `289 × 289 = 83 521` with sparse legal-id arrays

- **Status**: Accepted (2026-04-21, Phase 0.4 M4).
- **Decision**: The canonical RL action space is a single integer `action_id ∈ [0, 83521)` with encoding:
  ```
  action_id = src_flat * 289 + dst_flat
  src_flat  = src_y * 17 + src_x
  dst_flat  = dst_y * 17 + dst_x
  ```
  All four seats share this space; legality is enforced by a per-seat mask, not by seat-conditioned reshaping.
- **Hot-path API**:
  - `state.legal_action_ids(seat=None) -> np.ndarray[K] int32` — the sparse list of legal action ids (typical K ∈ [30, 80]).
  - `state.legal_action_mask_flat(seat=None) -> np.ndarray[83521] bool` — dense mask; caller may pre-allocate and reuse. Used as network-logit mask.
  - Helpers: `action_id_to_src_dst(action_id) -> ((sx,sy),(dx,dy))`, `src_dst_to_action_id((sx,sy),(dx,dy)) -> int`.
- **Retained for compatibility**:
  - `state.legal_actions(seat=None) -> list[Action]` (constructs from the sparse id array; cold path, used by tests).
  - The 4-D `legal_action_mask()` returning `(17,17,17,17)` bool is **deprecated**; it remains importable but emits a `DeprecationWarning` and internally reshapes from `legal_action_mask_flat`. Removed in a later ADR once all callers are migrated.
- **Rationale**:
  - 83 521 logits is well within transformer/CNN policy-head budgets (~330 KB fp32).
  - Sparse id arrays are the correct shape for vectorized rollout-collectors: `np.take(legal_mask, ids)` and `np.random.choice(ids)` are O(K).
  - Mirrors Ataraxos's flat-action convention → training recipes and ablation tooling port over.
- **Alternatives considered**:
  - **B: Per-seat compressed space (25 src × 289 dst).** Rejected — the action id would depend on state (which 25 src cells are occupied), making cross-seat mask reuse impossible.
- **Consequences**:
  - Policy network output head: 83 521-way logit vector; mask-add pre-softmax; illegal = `-inf`.
  - Entropy / KL computations must reduce over the **legal support only** (use `legal_action_ids` indexing); standard practice for masked PPO.
- **Ref**: ADR-112 (PPO-first), ADR-122.

---

## ADR-120: BeliefTensor tensorization

- **Status**: Accepted (2026-04-21, Phase 0.4 M3).
- **Decision**: `BeliefTensor` storage migrates from `dict[(x,y), np.ndarray[12]]` + `dict[Seat, dict[PieceType, int]]` to:
  ```text
  probs      : np.ndarray[120, 12] float32   — pid → type distribution
                                               (rows for dead / unassigned pids
                                                are zeroed; never read)
  remaining  : np.ndarray[4, 12]   int8      — remaining[seat][type];
                                               observer-side rows all zero
  ```
- **API compatibility**: `belief.get(pos)` now does `probs[cell_piece_id[flat]]` (O(1)); `belief.remaining[seat]` exposes a small `_RemainingView` that implements dict-like access over the row.
- **Hot path changes**:
  - `update()` becomes a sequence of row writes (`probs[pid] = one_hot(...)`, `probs[pid] = 0.0` for deaths) rather than dict pops.
  - The R5/R6/R7 conditional reveals still apply R-for-R; no rule logic changes.
  - The O(N) consistency sweep at the end of `update()` is removed — the dict↔pieces drift it defended against is impossible under ADR-117 where `cell_piece_id` is the single source of truth.
- **Rationale**:
  - Eliminates S4 and the `_observer_knows_enemy_type` repeated one-hot detection (it is now `probs[pid].max() > 1-ε`, a single cache line read).
  - Aligns belief with the SoA layout so ADR-121's batch builder can write beliefs into a `(N, 120, 12)` slab.
- **Consequences**:
  - `BeliefTensor.to_world_tensor()` (cold path) unchanged.
  - `tests/test_info_model.py` continues to pass via the `probs` dict-view shim (read-only): iteration order is by pid rather than by position, but all existing tests compare values keyed by position.
- **Ref**: ADR-117, ADR-118.

---

## ADR-121: Batched observation / belief construction

- **Status**: Accepted (2026-04-21, Phase 0.4 M5).
- **Decision**: Introduce batch builders that write into caller-owned slab tensors:
  ```python
  builder.build_observations_batch(
      states:    Sequence[GameState],     # len N
      beliefs:   Sequence[BeliefTensor],  # len N
      observers: Sequence[Seat],          # len N
      out_spatial: np.ndarray,  # (N, C, H, W) float32
      out_global:  np.ndarray,  # (N, G)      float32
  ) -> None
  ```
  Initial implementation: Python-level `for` loop reusing `build_into` — already sufficient to eliminate the per-obs allocation tax. A later ADR may SoA-batch the state storage itself (`states_batch: GameStateBatch` with batch axis 0 on every column); out of scope for Phase 0.4.
- **Zero-copy torch bridge**:
  ```python
  builder.bind_torch_buffers(
      spatial_torch: torch.Tensor,   # (N, C, H, W) float32, pinned CPU OK
      global_torch:  torch.Tensor,   # (N, G)      float32
  )
  ```
  Internally `np.asarray(torch_tensor)` shares memory; `build_observations_batch(..., self._spatial_np, self._global_np)` writes through. Enables `tensor.to(cuda, non_blocking=True)` with no CPU-side copy.
- **Rationale**: eliminates S5; required by `VectorJunqiEnv` (ADR-122) and by any PPO collector running `num_envs > 1`.
- **Ref**: ADR-118, ADR-120, ADR-122, ADR-124.

---

## ADR-122: `JunqiEnv` / `VectorJunqiEnv` RL interface

- **Status**: Accepted (2026-04-21, Phase 0.4 M6).
- **Decision**: The authoritative RL entry point is `junqi_rl.env.JunqiEnv`, a 4-headed self-play env wrapping `GameState + BeliefTensor*4 + ObservationBuilder`.
  ```python
  class JunqiEnv:
      def reset(self, seed: int | None = None) -> dict[Seat, ObservationTensor]
      def step(self, action_id: int) -> tuple[
          dict[Seat, ObservationTensor],  # obs for all 4 seats
          tuple[float, float, float, float],  # per-seat reward
          bool,                                # done
          dict,                                # info
      ]
      def legal_action_ids(self, seat: Seat | None = None) -> np.ndarray
      def current_seat(self) -> Seat

  class VectorJunqiEnv:
      """N parallel single-process envs, shared ObservationBuilder,
         batch-allocated obs slab (writes via ADR-121)."""
      def reset(...): ...
      def step(self, action_ids: np.ndarray[N] int32): ...
  ```
- **Seat rotation & action frame**:
  - `JunqiEnv.step(action_id)` interprets `action_id` in the **world frame**, regardless of whose turn it is. The network always emits canonical-frame actions; callers unrotate via `rotation.unrotate_action` before calling `step`. This keeps `JunqiEnv` network-agnostic.
  - Observation dict values are already canonical-frame (ADR-106).
- **Reward**: per-seat int tuple from `state.team_rewards()` at termination; intermediate steps reward zero. Reward shaping is a policy-layer concern, not an env one.
- **Rationale**: a single, stable entry point decouples `junqi_core` refactors from RL training code; the Gym-like API eases integration of third-party PPO implementations.
- **M6 acceptance (2026-04-21)**:
  - Landed: `junqi_rl.env.JunqiEnv` + `junqi_rl.env.VectorJunqiEnv` with the exact signatures above; flat action-id codec helpers `action_id_to_src_dst`, `rotate_action_id`, `unrotate_action_id`.
  - Per-env bit-identical parity with a solo `JunqiEnv` verified by `tests/test_junqi_env.py::TestVectorJunqiEnv::test_parity_with_single_env`.
  - Zero-copy `(N, 4, 101, 17, 17) float32` slab exposed via `VectorJunqiEnv.obs_spatial` for PPO collectors (wrap in `torch.from_numpy(...).pin_memory()` once and write-through on every step).
  - **Throughput target (≥10 000 plays/s single-env, ≥30 000 plays/s N=64) NOT met**: measured 505 plays/s and 319 plays/s (aggregate) respectively.  Reason is identical to ADR-125/ADR-126: the per-state cost is dominated by 4× `ObservationBuilder.build()` at ~500 µs each plus 4× `BeliefTensor.update + sync`.  The Python-level outer loop in `VectorJunqiEnv.step` does NOT hide that cost, so aggregate throughput stays flat at ~320 plays/s regardless of N.  See `docs/BENCHMARKS.md` §M6 scaling table.
  - **Resolution**: the ≥30 k/s target formally migrates to **ADR-126 Phase 1**, where `BatchedGameState.step_batch` + a batched obs kernel will collapse the `(N*4)` per-state builds into one NumPy pass, and the ADR-122 slab + API contract here requires no change.
- **Ref**: ADR-112 (PPO), ADR-117, ADR-118, ADR-121, ADR-125, ADR-126.

---

## ADR-123: SoA API shape pre-reserves a leading batch axis

- **Status**: Accepted (2026-04-21, Phase 0.4 forward-compat).
- **Decision**: All new SoA column arrays (ADR-117) and all buffer-oriented APIs (ADR-118, ADR-121) are designed so that **axis 0 is reserved for a future batch dimension**. At Phase 0.4 each `GameState` has unbatched shapes `(120,)`, `(289,)`, `(4,)`; future batched variants will be `(N, 120)`, `(N, 289)`, `(N, 4)`. No broadcasting shape change on downstream code.
- **Rule**: no new hot-path API may hard-code "axis 0 is the piece/cell axis"; that axis is always the last one, leaving axis 0 free.
- **Rationale**: makes the Phase 1+ migration to a `GameStateBatch` container a mechanical refactor with no API churn.
- **Ref**: ADR-117, ADR-121.

---

## ADR-124: Zero-copy `numpy ↔ torch` observation bridge

- **Status**: Accepted (2026-04-21, Phase 0.4 M5).
- **Decision**: `ObservationBuilder` exposes an optional `bind_torch_buffers(spatial, global_)` method that memoises the caller's torch tensors and writes through them via `np.asarray(...)` (numpy/torch share memory for CPU contiguous float32 tensors). `torch.from_numpy` is the accepted inverse for consumers that start from numpy.
- **Constraint**: torch tensors passed in must be CPU, contiguous, `dtype=torch.float32`. Pinned memory (`pin_memory=True`) is supported and required for `non_blocking=True` H2D transfers.
- **Rationale**: lets a PPO collector pre-allocate a pinned slab `(num_envs, C, H, W)` once, and have `VectorJunqiEnv.step` fill it in place — no per-step numpy→torch conversion, no per-step allocation.
- **Deferred (ADR-124-B, not accepted here)**: per-seat precomputed permutation tables replacing `np.rot90` — profiling shows rot90 is already free (view), so no benefit expected until GPU-kernel time.
- **Ref**: ADR-121.

---

## ADR-125: Move generation via precomputed static tables

- **Status**: Accepted (2026-04-21, Phase 0.4 M4).
- **Decision**: `move_gen` is rewritten from per-query Python BFS to **static tables + occupancy-masked lookup**. At module import, we precompute:
  ```text
  STRAIGHT_RAIL_DESTS[src_flat]      : np.ndarray[K] int16
      — non-engineer straight-rail candidate cells along +x/-x/+y/-y,
        grouped by direction (4 contiguous slices for vectorized blocking).
  ENGINEER_REACHABLE_STATIC[src_flat]: np.ndarray[K] int16
      — engineer BFS assuming empty board; runtime filters by occupancy.
  CURVE_RAIL_CELLS[curve_id]         : np.ndarray[K] int16
      — cells belonging to each curve rail (K ≤ 8).
  ADJACENT_CELLS[src_flat]           : np.ndarray[K] int16
      — legal single-step neighbours (4 orth always, +4 diag if camp).
  CAMP_FLAT, STRONGHOLD_FLAT,
  RAIL_FLAT, NINE_GRID_FLAT          : np.ndarray[289] bool
      — cell-property bitmasks for vectorized filtering.
  ```
- **Runtime `legal_moves_from(pid)`**:
  1. Compute `empty = cell_piece_id < 0` and `enemy_attackable = (cell_piece_id ≥ 0) & (piece_seat[cell_piece_id] != my_seat) & ~CAMP_FLAT` (two vector ops, reused across all pids on this turn — **cached per step**).
  2. For each movement mode, do a single `candidates = TABLE[src_flat]` + `mask = empty[candidates] | enemy_attackable[candidates]` + direction-group blocking for straight rails.
  3. `is_legal_move(src, dst)` becomes `dst in legal_moves_from(src_pid)`, but hot callers use the sparse list directly.
- **Behavior parity**: the legacy algorithmic result is **bit-identical** — tables are derived from the same rules. `tests/test_move_gen.py` and `tests/golden/move_gen/*.json` remain the regression harness.
- **Rationale**:
  - S3 is the real rollout bottleneck (2 k/s end-to-end vs 16 k/s step-only). Tables collapse per-query work from ~40 Python calls/piece to ~3 vectorized numpy ops.
  - Target: **`legal_action_ids` ≥ 50 k/s** end-to-end.
  - Tables total ~50 KB, built once at import in <5 ms.
- **Cache invalidation**: the per-step occupancy vectors (`empty`, `enemy_attackable_vs_<seat>`) are recomputed exactly once per `step_inplace`. No cross-state leakage — `GameState` owns its own tables-of-derived-masks via a dirty-bit pattern.
- **Alternatives considered**:
  - **C extension / Cython.** Deferred to Phase 2 — numpy-table form already closes the 25× gap vs Python BFS and is fully portable.
  - **Incremental move-list maintenance** (update legal moves as pieces move). Rejected for Phase 0.4: correctness risk too high; revisit once table form is verified.
- **M4 acceptance (2026-04-21)**:
  - Landed: `legal_action_ids` ≈ 7.6 k/s single-state (vs 2.5 k/s legacy = 3.0×). End-to-end rollout 3.6 k plays/s (vs 1.9 k = 1.9×). Bit-identical on 40 000-state fuzz (`tests/test_move_gen_parity.py`).
  - **ADR target of 50 k/s NOT met** on CPU single-state.  After comparing with Ataraxos (`src/env/cuda/action_kernels.cu`, which uses a per-(env,cell) CUDA kernel with `num_envs × 100` threads), it is clear that single-state CPU throughput is ceiling-limited by NumPy dispatch overhead (~3 µs/op × ~20 ops/call = 60 µs floor). The remaining gap is batch parallelism, not per-state optimization.
  - **The 50 k/s target is formally deferred to Phase 1 (ADR-126).**
- **Ref**: ADR-117, ADR-119, ADR-126.

---

## ADR-126: Batched environment for ≥ 50 k plays/s (Phase 1 scope)

**Decision:** Phase 1 introduces a leading-batch-axis `BatchedGameState(num_envs, ...)` and matching `legal_action_ids_batch / step_batch / build_observation_batch` APIs. M4's 50 k/s target is met not by further CPU single-state optimization but by amortizing `num_envs=1024+` states through the same vectorized tables.

- **Motivation**: direct measurement (Phase 0.4 M4, 2026-04-21) shows single-state `legal_action_ids` plateaus at 7.6 k/s on CPU because each vectorized numpy op has 2–5 µs of fixed dispatch cost. Fifteen to twenty ops per call give a ~60 µs floor. Breaking through requires either (a) a C extension (violates the "pure Python stack, no build step" constraint of Phase 0.4) or (b) batching so dispatch cost is amortized. Ataraxos chose (b) with CUDA; we choose (b) with NumPy first and keep the CUDA upgrade as ADR-F3.
- **Data structures**:
  ```
  cell_piece_id     : ndarray[N, 289]   int16
  piece_seat_arr    : ndarray[N, P]     int8
  piece_type_arr    : ndarray[N, P]     int8
  alive             : ndarray[N, P]     bool
  pos_x, pos_y      : ndarray[N, P]     int8
  move_count_arr    : ndarray[N, P]     int16   (T7)
  deaths_*          : packed SoA with shape (N, P) — replaces the per-state dict
  zobrist_state     : ndarray[N]        int64
  turn              : ndarray[N]        int8
  ```
- **APIs**:
  - `legal_action_ids_batch(batch_state, seat_per_env) -> ndarray[N, K_max] int32` with -1 padding.
  - `legal_action_mask_batch(batch_state, seat_per_env) -> ndarray[N, 83521] bool` (ideal for policy masking).
  - `step_batch(batch_state, action_ids) -> (batch_state, rewards)` — all SoA updates in one numpy pass.
  - `build_observation_batch(batch_state, beliefs_batch, observer_per_env) -> ObservationBatchTensor` (already planned as ADR-121; this is the data-structure side).
- **Implementation plan**:
  1. Add `BatchedGameState` alongside current `GameState`; they share the table imports. The table shapes from ADR-125 (e.g. `ADJ_STRAIGHT (289, 4)`, `STRAIGHT_RAIL_RAYS_PAD (289, 4, 4)`) already broadcast cleanly when we add a leading batch axis.
  2. Port M4's `generate_legal_action_ids_batch` — replace `src_flats: ndarray[K]` with `src_flats: ndarray[N, K_i]` and the fancy-index reads become shape `(N, K, 4)` / `(N, K, 4, 4)`. Engineer BFS is the one that needs more thought; we'll batch it with a shared frontier queue keyed by `(env_idx, flat)`.
  3. `step_batch` is the hardest — it currently has ~30 scalar assignments per call. Migrating to indexed-`np.add.at` / boolean-mask updates is ~2-3 days.
- **Scope boundary**: Phase 1 keeps `BatchedGameState` on CPU + NumPy. A CUDA kernel set matching Ataraxos's `action_kernels.cu` is **ADR-F3**, scheduled after Phase 2's first training run validates the pipeline.
- **Acceptance criteria for ADR-126**:
  - `legal_action_ids_batch` at N=1024: ≥ 50 k plays/s aggregated.
  - `step_batch` at N=1024: ≥ 30 k plays/s aggregated.
  - Bit-identical per-env parity with current single-state `GameState` for a 10 000-step fuzz at N=64.
- **Ref**: ADR-117, ADR-119, ADR-121, ADR-123, ADR-125. Ataraxos `src/env/cuda/action_kernels.cu`.

---


## ADR-127: C++/CUDA Backend Architecture for Phase 1 GPU Acceleration

- **Status**: Proposed (2026-04-22, Phase 1 planning).
- **Decision**: Implement a GPU-accelerated C++/CUDA backend alongside the Python/NumPy CPU path. The backend targets three performance-critical hotspots:
  1. `legal_action_ids_batch` (batch action generation).
  2. `step_batch` (batch state transitions + combat).
  3. `build_observation_batch` (batch observation tensor generation, all 101 channels).
  
  All kernels operate on **Structure-of-Arrays (SoA) GPU-resident data** (ADR-117, ADR-123) and are exposed via **PyBind11 bindings** for seamless integration with Python RL training loops.

- **Architecture Highlights**:
  - **Data Residency**: SoA arrays pinned/resident on GPU; optional unified memory for CPU-GPU synchronization.
  - **Kernel Strategy**: One thread per piece (legal actions), one thread per state (step), one thread per (env, channel, cell) or per (env, seat) block (observation).
  - **Static Tables**: Move generation tables (ADR-125), zobrist hashes, topology masks stored in GPU constant/global memory.
  - **PyBind11 Wrappers**: C++ functions callable from Python with automatic numpy array marshalling.
  
- **Performance Target**:
  - Phase 1a (CPU / NumPy): ≥50 k plays/sec aggregate at N=1024 (fulfills ADR-126 acceptance criteria).
  - Phase 1b (GPU): ≥500 k plays/sec aggregate (10× CPU baseline).
  - Phase 1+ (multi-GPU): scaling to ≥50 M plays/sec on 128-GPU cluster.

- **Implementation Plan**:
  - **M1**: `BatchedGameState` (CPU/NumPy) — validates ADR-126 design before GPU port.
  - **M2**: CUDA `legal_action_ids_batch` kernel + PyBind11 module skeleton.
  - **M3**: CUDA `step_batch` kernel (move + combat + zobrist stages).
  - **M4**: CUDA `build_observation_batch` kernel (all 101 channels).
  - **M5**: End-to-end integration + training loop validation.

- **File Organization**: 
  ```
  src/env/cuda/
  ├── CMakeLists.txt
  ├── src/
  │   ├── game_state.cu/cuh
  │   ├── observation.cu/cuh
  │   ├── tables.cu/cuh
  │   ├── common.cuh
  │   └── bindings.cpp
  ├── include/junqi_cuda.h
  └── tests/
  ```

- **Backwards Compatibility**: CPU path remains default; GPU backend activated via constructor flag:
  ```python
  env = VectorJunqiEnv(num_envs, use_cuda=True)
  ```
  All existing tests pass on both paths (flag-gated test matrix).

- **Risks & Mitigation**:
  - **Correctness**: Continuous parity testing against Python reference; all 52 golden replay JSON files verified on GPU.
  - **Performance**: GPU dispatch overhead vs. amortized kernel cost; profiling via `nvidia-smi`, `nsys`, `ncu`.
  - **Multi-GPU scaling**: future (Phase 2+); single-GPU design is Phase 1 scope.

- **Ref**: ADR-117 (SoA layout), ADR-118 (pre-allocated buffers), ADR-123 (batch-axis forward compat), ADR-125 (static tables), ADR-126 (batched env scope). See `docs/CUDA_ARCHITECTURE.md` for 12-section detailed design document (842 lines).

---

## Version History

| Version | Date       | Changes |
|---------|-----------|---------|
| 1.0     | 2026-04-19 | Initial ADR set (001-015 rules, 100-110 engineering). |
| 1.1     | 2026-04-20 | Add ADR-111 (Seat enum rename SOUTH/WEST/NORTH/EAST, Phase 0.3 T7). |
| 1.2     | 2026-04-20 | Add ADR-112 (PPO-first RL algorithm) and ADR-113 (legacy_gui adapter = replay-only). |
| 1.3     | 2026-04-21 | Add ADR-114 (piece_id global identity system), ADR-115 (T7 observation tail = 32 channels), ADR-116 (observation pinned at 101; supersedes ADR-106 "69 initial" draft). Phase 0.3 T7 M1–M4 delivery. |
| 1.4     | 2026-04-21 | Add ADR-117..ADR-125 (Phase 0.4 engineering refactor: SoA GameState, pre-allocated observation builder, flat action space 83 521, tensorized belief, batch builders, `JunqiEnv`, SoA forward-compat batch axis, zero-copy numpy↔torch bridge, precomputed move-gen tables). Incremental Zobrist state_hash. Closes perf-debt items S1/S2/S3/S4/S5/S9 and L5. |
| 1.5     | 2026-04-21 | Phase 0.4 M4 landed. `legal_action_ids` batch path at 7.6 k/s (3× legacy), fuzz-verified bit-identical. 50 k/s target formally re-scoped: add ADR-126 for Phase 1 batched env (`BatchedGameState`) after single-state CPU was shown to be dispatch-ceiling-limited (Ataraxos-style batch parallelism is the only path past 10 k/s on this stack). |
| 1.6     | 2026-04-21 | Phase 0.4 M5 landed. `ObservationBuilder.build_into` + `build_observations_batch` + `build_observations_batch_torch` shipped (ADR-121, ADR-124). 64-way batch obs build hits 14.89 ms (target ≤ 15 ms). Zero-copy numpy↔torch CPU bridge verified via shared-storage test. Phase 0.4 engineering refactor complete through M5. |
| 1.7     | 2026-04-21 | Phase 0.4 M6 landed. `junqi_rl.env.JunqiEnv` + `VectorJunqiEnv` shipped (ADR-122); flat action-id codec helpers added. Per-env bit-id parity proven for N=4 VectorEnv. Throughput miss (505 / 319 plays/s vs 10 k / 30 k targets) formally delegated to ADR-126 Phase 1 batched env; the ADR-122 API surface is final. Test count 490/490 (4 torch-skip). |
| 1.8     | 2026-04-21 | Phase 0.4 M7 landed. `junqi_core.replay.Trajectory` shipped with deterministic save/load/replay + Zobrist self-check (4.5 KiB/200-step-game on disk, 100/100 round-trip clean). `junqi_rl.ExperienceBuffer` shipped with three legal-mask modes (`none` / `sparse` / `dense`) + ring wrap + torch bridge. `append_step` hits 28 k/s — CPU memcpy-limited by the 117 KiB obs copy; trainers that need ≥200 k/s use the `VectorJunqiEnv` slab directly (ADR-121). Test count 522/522 (6 torch-skip). **Phase 0.4 sign-off**: all ADR-117..126 landed; throughput misses (M4, M6, M7) are CPU-irreducible and formally inherited by Phase 1 / ADR-F3. |
| 1.9     | 2026-05-09 | Add **ADR-128** (CombatMemory v4: per-observer DARK-mode high-order combat history). New `junqi_core/combat_memory.py` module with `CombatMemoryState` (4 × 120 SoA, 14 fields), `apply_combat_event` + `apply_path_revealed_gongb` reference rules.  Wired into `GameState.step_inplace` and `BatchedGameState._step_single` for EAT / KILLED branches (BOMB skipped — no live target).  Adds **50 spatial channels** in two layers: Layer 1 (45 ch — kill_mine_type/ge, kill_other_ge, chain_type/ge, floor_ge, is_gongb, not_gongb, dilei_candidate) projected to enemy alive cells; Layer 2 (5 ch — my_kill_count_ge, my_is_gongb, my_dilei_candidate) projected to own alive cells via AND-of-two-opponents (yields path-revealed-only theory-of-mind).  `OBS_CHANNELS = 256 → 306`.  Indices `[0, 256)` bit-compatible with pre-CombatMemory checkpoints.  CUDA path: complete — `DeviceGameStateBatch` owns 14 device pointers, `step_batch_kernel` calls `cm_apply_event_dev` + `cm_apply_path_revealed_gongb_dev` + `cm_move_requires_gongb_dev` (rail-graph BFS reusing existing constant memory tables), `observation_kernel` calls `cm_write_channels_device` for channels 256..305.  Strict no-host-traffic invariant: only legal CPU↔GPU transfers are constructor cudaMalloc/cudaMemset and the `cm_copy_from_host`/`cm_copy_to_host` parity-test bindings (never invoked during training).  Test surface: `tests/test_combat_memory.py`, `tests/test_observation_combat_memory.py`, `tests/test_gpu_combat_memory_parity.py`.  Initial training run: `configs/v35_combat_memory_full.yaml` (`num_envs=320`, `belief.embed_dim=512`, `net.depth=6`). |

---

## ADR-128: CombatMemory v4 — DARK-mode high-order combat-history memory

- **Status**: Accepted (2026-05-09); CPU + CUDA both shipped.
- **Context**: feed-forward MoveNet has no cross-step memory. The 32-step
  `move_history` channels carry recent src/dst geometry but lose any
  combat *fact* > 32 steps old, and they don't expose deductive logic
  like "C ate B which ate my paizh ⇒ C is at least YINGZH+, not GONGB".
  Belief deduction has been shown to lift Stratego win-rate by 5-10%
  in Ataraxos; the JunQi gap analysis (`docs/ATARAXOS_GAP_ANALYSIS.md`)
  identifies "no neural belief net + no memory" as the single largest
  algorithmic gap.

- **Decision**: maintain a per-observer × per-piece_id
  `CombatMemoryState` (14 fields, shape (4, 120)).  Two entry points
  invoked from both `GameState.step_inplace` and
  `BatchedGameState._step_single`:
  * `apply_combat_event(...)` — runs in EAT/KILLED branches; BOMB skipped.
  * `apply_path_revealed_gongb(cm, pid)` — when the moving piece is a
    GONGB and the path is engineer-only, broadcast `is_gongb=True` to
    all four observers (geometry is public).
  Project the state to a 50-channel observation tail (Layer 1 + Layer 2)
  on every observation build.

- **Information boundary (DARK only)**: only `victim_seat == observer`
  may use `victim_type`; for non-victim observers we record only the
  event count (`direct_other_count`).  Chain propagation operates on
  public `piece_id`s for all four observers.  `attacked_by_known_gongb`
  is set on KILLED defenders for any observer who knew the dead
  attacker was a GONGB at the time of the attack (own-seat type
  knowledge, or previous path-reveal).

- **Channel size choice (50 vs 309)**: the original v1 proposal had 70
  channels including 21 redundant (`exclude_type` duplicating `floor_ge`,
  always-zero `candidate_type`, 6 useless recency channels).  v4 ships
  a binary-only 50-channel layout that drops redundancies and adds a
  theory-of-mind Layer 2 (5 ch) so the network sees the public-by-AND
  view of its own pieces.  All channels are 0/1 — no float counts.

- **Rejected alternatives (vs v1 / earlier drafts)**:
  - **(a) HALF_DARK / BRIGHT support** → not used in production training;
    the sole supported show-mode is DARK.  Other show-modes can no
    longer leak truth-value type information into `is_gongb` /
    `dilei_candidate`.
  - **(b) Float `kill_count` / `recency` channels** → user demanded all-
    binary; v4 thresholds (≥1 / ≥2 / ≥3) keep the network input
    quantization-stable.
  - **(c) Per-observer `not_gongb`/`floor` on Layer 2** → opponents
    can't observe our killed pieces' types in DARK; v4 only projects
    `is_gongb` (path-revealed AND) and `dilei_candidate` to Layer 2.
  - **(d) Hard-masking BeliefNet logits with `exclude_type_mask`** →
    deferred to v36+.  We first surface CombatMemory as observation
    features.

- **CPU↔GPU interaction invariant (hard requirement)**: zero host
  traffic in the training hot path for CombatMemory.  The only legal
  transfers are:
  1. `DeviceGameStateBatch` constructor — one-time `cudaMalloc` /
     `cudaMemset` of the 14 cm device buffers (sentinel 0xff for the
     three int16 *_step fields = -1).
  2. `cm_copy_from_host` / `cm_copy_to_host` — parity-test bindings
     used only by `tests/test_gpu_combat_memory_parity.py`; never
     invoked during training.
  All `step_batch_kernel` and `observation_kernel` updates run as
  device functions on already-resident SoA arrays.  See
  `docs/COMBAT_MEMORY_IMPLEMENTATION.md` §4 for the full audit.

- **Acceptance**:
  - `OBS_CHANNELS == 306`, asserted in `observation._self_check`.
  - Unit tests for CPU update logic all green.
  - CPU↔GPU CombatMemory parity over 50 random games × N=32 × 200
    steps (`tests/test_gpu_combat_memory_parity.py`, 14 fields,
    bit-identical).
  - `step()` ≥ 14k plays/s (≤ 5 % regression vs T7 16.8k baseline).
  - `v35_combat_memory_full` reaches win_rate vs random ≥ 0.85.

- **Files**: `junqi_core/combat_memory.py` (new), `junqi_core/state.py`,
  `junqi_core/observation.py`, `junqi_core/batched_state.py`,
  `junqi_core/move_gen.py` (added `move_requires_gongb`),
  `src/env/cuda/include/junqi_cuda.h`,
  `src/env/cuda/src/combat_memory.cuh` (new),
  `src/env/cuda/src/combat_memory.cu` (new),
  `src/env/cuda/src/{game_state,observation,bindings}.cu(.cpp)`,
  `src/env/cuda/CMakeLists.txt`,
  `tests/test_combat_memory.py`, `tests/test_observation_combat_memory.py`,
  `tests/test_gpu_combat_memory_parity.py` (new),
  `configs/v35_combat_memory_full.yaml`,
  `docs/COMBAT_MEMORY_DESIGN.md` (the *what*),
  `docs/COMBAT_MEMORY_IMPLEMENTATION.md` (the *how* + status).

- **Ref**: ADR-114 (piece_id), ADR-115 (T7 observation tail),
  ADR-116 (OBS_CHANNELS pin), ADR-118 (allocation-free observation builder).

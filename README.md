# JunQi — Four-Player 四国军棋 AI

Reinforcement-learning framework for the Chinese 4-player imperfect-information board game 四国军棋 (Junqi / Military Chess), with a legacy C engine and GTK GUI preserved as visualization and oracle testing tools.

---

## Project layout

```
JunQi/
├── docs/                     # Authoritative specifications
│   ├── RULES.md              # Canonical rules v1.0.0 (FROZEN)
│   ├── ARCHITECTURE.md       # Engineering design doc (repo layout, canonical rotation, RL tensors)
│   ├── DECISIONS.md          # ADRs for all 13 rule decisions + 11 engineering decisions
│   └── INFERENCE.md          # (Phase 0.2) Detailed belief-update algorithm
│
├── junqi_core/               # Pure-Python rule engine (NEW, authoritative)
│   ├── rules.py              # Piece enums, combat table, SILING flag-reveal (Q7)
│   ├── board.py              # 17×17 geometry, camps/strongholds/rails, NineGrid
│   └── setup.py              # C1-C5 setup validator + random generator
│
├── junqi_rl/                 # (Phase 1+) RL environment, agents, training
├── junqi_viz/                # (Phase 4+) Web visualization backend/frontend
│
├── tests/
│   ├── golden/               # Schema-enforced JSON test cases
│   │   ├── _dsl.py           # Fluent Case / SetupCase builder w/ autofill
│   │   ├── scenarios.py      # Catalog of 49 hand-written core scenarios
│   │   ├── SCHEMA.md         # Test JSON schema v1.0
│   │   ├── move_gen/         # 8 scenarios — adjacent / rail / engineer
│   │   ├── battle/           # 10 scenarios — combat table coverage
│   │   ├── siling_flag/      # 7 scenarios — Q7 flag-reveal logic
│   │   ├── stronghold/       # 6 scenarios — ADR-006 stronghold probing
│   │   ├── inference/        # 8 scenarios — §3.4 deductive logic
│   │   ├── setup_validation/ # 8 scenarios — C1-C5 hard constraints
│   │   └── full_game/        # 3 scenarios — integration (placeholder)
│   ├── test_combat_rules.py  # 31 combat golden regressions (all green)
│   └── test_setup_validation.py  # 9 setup validator regressions (all green)
│
├── legacy_engine/            # (renamed from ENGINE/) — frozen C oracle
├── legacy_gui/               # (renamed from GUI/)    — GTK visualization
│
├── tools/                    # Headless replay_viewer + oracle utilities
│
├── pyproject.toml            # Python packaging + ruff/mypy/pytest config
└── Makefile                  # builds legacy_engine/ and legacy_gui/
```

---

## Phase 0.1 — Rule Freeze ✅ DONE

All 13 rule-clarification questions (Q1-Q13) resolved and codified. See `docs/RULES.md` §10 for the full decision table.

Key facts the new engine preserves:

- **4 seats** — `Seat.SOUTH(0), Seat.WEST(1), Seat.NORTH(2), Seat.EAST(3)`; teams by `seat % 2` (SOUTH+NORTH vs WEST+EAST). Integer values identical to legacy `ChessDir {HOME, RIGHT, OPPS, LEFT}`, see ADR-111.
- **12 piece types**, smaller enum = stronger (`SILING=5 < ... < GONGB=13`).
- **25 pieces per seat** in 30 slots; 5 camps always empty (Q13 hard constraint).
- **Information broadcast** after each move: `{positions, event ∈ {MOVE,EAT,KILLED,BOMB}, flag_reveals, flag_captured}` — **no piece types ever revealed** (Q2, the core novelty vs Stratego).
- **SILING flag-reveal** — per-seat, triggered when that side's SILING dies (Q7).
- **Draw thresholds** — `4000` total moves, `200` since last combat (Q10, Ataraxos-aligned).

---

## Quick start

```bash
# 1. Install Python deps (dev extras include pytest, ruff, mypy)
pip install -e '.[dev]'

# 2. Regenerate all golden JSON cases from the DSL catalog
PYTHONPATH=. python3 -m tests.golden.scenarios

# 3. Run all tests (should be 40 passed)
PYTHONPATH=. python3 -m pytest tests/ -v

# 4. Self-test individual junqi_core modules
python3 -m junqi_core.rules   # prints "junqi_core.rules self-test: OK"
python3 -m junqi_core.board   # prints board topology summary
python3 -m junqi_core.setup   # prints "junqi_core.setup self-test: OK"

# 5. Inspect a saved human/RL replay without starting GTK
PYTHONPATH=. python3 tools/replay_viewer.py path/to/game.npz --step 120 --validate
```

See [`docs/REPLAY.md`](docs/REPLAY.md) for the seekable replay API and RL
policy/belief recording format.

Legacy C engine (optional, oracle for parity testing):

```bash
make -C legacy_engine        # builds legacy_engine/bin/JunQiEngine
make -C legacy_gui           # builds GTK GUI (requires X11 + GTK3)
```

---

## Decisions index

- **ADR-001..ADR-015**: all 13 rule decisions (Q1-Q13 + piece strength + versioning).
- **ADR-100..ADR-110**: engineering decisions (independent Python library, monorepo, PyTorch, canonical rotation, flat action space, etc.).

See `docs/DECISIONS.md`.

---

## Roadmap

| Phase | Status | Target |
|------:|:------:|--------|
| 0.1 | ✅ Done | Rule freeze + golden tests + `junqi_core/{rules,board,setup}.py` |
| 0.2 | 🚧 Next | `move_gen.py`, `state.py`, `info_model.py` + full golden parity |
| 0.3 | ✅ | Replay system + CLI tools + `oracle_runner.py` |
| 1   | ⏳ | Gymnasium env, vectorized self-play on T4 |
| 2   | ⏳ | Supervised pretrain from legacy engine games (awaits H20) |
| 3   | ⏳ | PPO self-play + belief network |
| 4   | ⏳ | Web visualization (FastAPI + Vue) |
| 5   | ⏳ | Human-vs-AI pipeline |
| 6   | ⏳ | Retire `legacy_gui` once `junqi_viz` feature-complete |

---

## License

Proprietary — internal Tencent project.

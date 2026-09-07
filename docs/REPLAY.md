# Replay workflow

JunQi now has one replay contract for the GTK client, headless tools, and RL
recordings.

## Safe NPZ loading

Replay NPZ files store numeric arrays and Unicode strings and are loaded with
`allow_pickle=False`. This applies to core trajectories, policy trajectories,
and viewer format detection. Legacy NPZ files containing object arrays are
rejected with `ValueError: Object arrays cannot be loaded when allow_pickle=False`.
Re-export recordings from their original trusted game/action data with the current
writer; do not enable pickle to open an unknown replay. JSON and native JQL replay
workflows are unchanged.

## Core trajectory

`junqi_core.replay.Trajectory` stores the four starting lineups plus a compact
world-frame action log. It is deterministic and self-checking:

```python
from junqi_core.replay import Trajectory
from junqi_core.replay_viewer import ReplayViewer

trajectory = Trajectory.load("game.npz")
trajectory.validate()
viewer = ReplayViewer(trajectory)
frame = viewer.seek(120)
print(frame.state.move_counter, frame.result)
```

`ReplayViewer.next()`, `previous()`, and `seek(step)` return cloned state
snapshots. A UI can therefore scrub freely without mutating the saved
trajectory or sharing mutable engine state between frames.

## Local visual viewer

Install the optional web dependencies and point the viewer at one explicit
replay file:

```bash
python -m pip install -e ".[viz]"
junqi-replay runs/game.npz
# open http://127.0.0.1:8765
```

The service binds to `127.0.0.1` by default and exposes only the replay selected
on the command line; it has no arbitrary filesystem browsing endpoint. The UI
supports play/pause, first/previous/next/last, timeline scrubbing, keyboard
arrows, playback speed, piece labels, event summaries, alive counts and RL
Top-K/value inspection.

## Original GTK native client

The repository's original native client is `legacy_gui`, not the web page. It
keeps the original 733×688 four-player board, background, menu layout and the
original four color-specific sprite strips (`orange.bmp`, `purple.bmp`,
`green.bmp`, and `blue.bmp`) under `legacy_gui/res`. The right-hand area is used
for the replay inspector: current step, action
coordinates and piece names, combat event, per-seat alive counts, and team
totals.

Build it on a machine with GTK3 development files installed:

```bash
make -C legacy_gui PROFILE=release
```

The legacy client reads compact `.jql` files. Convert a modern RL `.npz`
recording first:

```bash
python scripts/export_legacy_jql.py \
  runs/victory.npz \
  runs/victory.jql \
  --metadata runs/victory.native.json
```

Then launch `legacy_gui/bin/JunQiGUI runs/victory.jql`, or use the native
client's 「文件」→「打开复盘」 menu. The exporter validates the modern
trajectory before writing the legacy header, four 30-piece lineups, and
coordinate action log.

## RL policy replay

`junqi_rl.analysis.record_game_with_policy(..., record_beliefs=True)` writes a
`TrajectoryWithPolicy` file containing the chosen action, top-K policy
probabilities, value estimate, and the actual rule-based belief tensor at each
step. Belief snapshots use shape `(T, 4, 12, 289)` and are no longer silent
all-zero placeholders.

Policy replay schema v2 also stores an `action_source` for every step:

- `policy_sample`
- `policy_greedy`
- `random_opponent`

Schema v1 files remain readable and default to `policy_sample`. Top-K action
IDs are stored in full 17×17 world coordinates, so terminal and web viewers
decode the same move.

```bash
python tools/replay_viewer.py runs/game.npz --all --json > runs/game.jsonl
python tools/replay_viewer.py runs/game.npz --step 120 --validate
```

The JSONL output is backend-neutral and can feed a notebook, web dashboard, or
future GTK overlay. It includes step/turn, public combat event, alive-piece
counts, and policy value/top-K explanations when present.

## Automatic training progress replays

`scripts/train.py` records greedy EMA-policy games against a random opponent
after each evaluation. Relevant `TrainConfig` fields are:

```yaml
eval_every: 50
eval_num_games: 128
eval_record_games: 1
eval_record_beliefs: false
league_max_checkpoints: 12
league_eval_games: 16
```

Outputs are written below `save_dir`:

```text
replays/eval_000050_00_team0.npz
league.json
```

Evaluation uses paired seeds and runs the policy on both teams before merging
exact win/loss/draw/ongoing counts. Metrics include a Wilson 95% confidence
interval. The league registry stores checkpoint SHA-256 values, bounded
history, Elo ratings and deterministic historical-opponent sampling.

# Replay workflow

JunQi now has one replay contract for the GTK client, headless tools, and RL
recordings.

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

## RL policy replay

`junqi_rl.analysis.record_game_with_policy(..., record_beliefs=True)` writes a
`TrajectoryWithPolicy` file containing the chosen action, top-K policy
probabilities, value estimate, and the actual rule-based belief tensor at each
step. Belief snapshots use shape `(T, 4, 12, 289)` and are no longer silent
all-zero placeholders.

```bash
python tools/replay_viewer.py runs/game.npz --all --json > runs/game.jsonl
python tools/replay_viewer.py runs/game.npz --step 120 --validate
```

The JSONL output is backend-neutral and can feed a notebook, web dashboard, or
future GTK overlay. It includes step/turn, public combat event, alive-piece
counts, and policy value/top-K explanations when present.

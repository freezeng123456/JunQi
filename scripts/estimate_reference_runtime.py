#!/usr/bin/env python3
"""Estimate from measured production-size rollouts; distinguish queue time."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('benchmark');p.add_argument('--target',type=int,default=1000000)
a=p.parse_args();root=Path(a.benchmark)
rows=[json.loads(line) for line in (root/'game_progress.jsonl').read_text().splitlines()]
assert len(rows)>=3
first,last=rows[0],rows[-1]
elapsed=last['elapsed_s']-first['elapsed_s']
steps=last['tracked_environment_steps']-first['tracked_environment_steps']
games=last['completed_games']-first['completed_games']
assert elapsed>0 and steps>0 and games>0
steps_per_s=steps/elapsed;games_per_s=games/elapsed
# The preceding completed 20,427-game formal baseline measured this game length.
reference_mean_moves=847.616
result=dict(world_size=last['world_size'],warmup_rollouts_excluded=1,
    measured_interval_s=elapsed,measured_environment_actions=steps,measured_completed_games=games,
    actions_per_second=steps_per_s,completed_games_per_second=games_per_s,
    target_completed_games=a.target,
    hours_from_recent_completed_games=a.target/games_per_s/3600,
    hours_at_previous_baseline_game_length=a.target*reference_mean_moves/steps_per_s/3600,
    previous_baseline_mean_moves=reference_mean_moves,
    queue_time_included=False,
    caveat='Three early rollouts are a throughput probe, not a long-run speed guarantee; game length and setup readiness change during learning.')
assert all(math.isfinite(v) for v in result.values() if isinstance(v,float))
(root.parent.parent/'throughput_estimate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

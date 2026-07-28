"""Read training metrics from tensorboard events."""
import sys
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator('/data/home/freezeng/data/workspace/JunQi/exps/t11_smoke/logs')
ea.Reload()
tags = ea.Tags()['scalars']
print('Available tags:', sorted(tags))

for tag in sorted(tags):
    events = ea.Scalars(tag)
    if events:
        first = events[0]
        latest = events[-1]
        print(f'{tag:40s}: first={first.value:+.4f} latest={latest.value:+.4f} (step {latest.step}, n={len(events)})')

"""junqi_rl.belief — Neural belief network training subpackage.

Contains:

* :mod:`junqi_rl.belief.buffer`         — CPU replay buffer for (obs, labels) tuples.
* :mod:`junqi_rl.belief.reveal_tracker` — emits labels on terminal events (P1.3).
* :mod:`junqi_rl.belief.inference`      — uploads BeliefNet output to GpuRollout (P1.5).
* ``junqi_rl.training.belief_ppo``      — CE trainer that consumes the buffer (P1.4).

Design summary (docs/P1_BELIEF_NET_PLAN.md §5):

The belief net lives *in parallel* with JunQi's hand-coded deductive rules
(R1, R4, R5/R7, R6, R9, I5 on GPU). Deductive rules give bit-exact updates
for deterministic events (flag captured → that cell has JUNQI with prob 1).
The neural net interpolates for everything else. Layering means the net
doesn't have to learn what the rules already know, which keeps the
training signal sharp.
"""

from junqi_rl.belief.buffer import BeliefBuffer, BeliefSample
from junqi_rl.belief.inference import (
    belief_logits_to_upload_shape,
    refresh_beliefs_neural,
)
from junqi_rl.belief.reveal_tracker import RevealTracker

__all__ = [
    "BeliefBuffer",
    "BeliefSample",
    "RevealTracker",
    "refresh_beliefs_neural",
    "belief_logits_to_upload_shape",
]

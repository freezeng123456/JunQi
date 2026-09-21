# Continued self-play within the existing H20 lease

The 128-rollout vs-random pilot was a phase checkpoint, not the user's stopping
condition. Its paired head-to-head result did not justify promoting the candidate
or inferring a learning ceiling. Continue from the original audited guarded_s501
raw policy with a fresh optimizer; retain the completed pilot unchanged.

Phase B uses the same 942,292-parameter v4 policy, 128 environments, 512 steps,
BF16, fixed learning rate 1e-5, fixed magnet coefficient 0.002, 0.75 advantage keep,
and one PPO epoch. EMA, learned belief and arrangement training remain disabled.
All four seats now use the learner. Terminal team outcomes replace asymmetric
piece-loss shaping, whose incentive conflicts with engagement in self-play.
This is a training recipe change, not an ablation that identifies either change's
individual causal effect. Historical draw-rate comments are motivation, not new
evidence of current behavior.

Four real self-play rollouts with a random and fixed-baseline evaluation are
audited, privately backed up and then resumed with optimizer/counters intact.
The long stage has a 20,000-rollout ceiling, subject to the lease deadline and
guards. It saves every 128 rollouts and evaluates every 256 against 512 random
games and 256 paired baseline games. These repeated fixed-seed monitoring games
are for diagnosis/selection, not independent strength evidence. Three consecutive
random win rates below 99% (after rollout 512) stop this stage for diagnosis.
Nonfinite training or evaluation failures also require diagnosis. A stage stop
does not finish the larger training objective or automatically pause follow-up.

The server controller and app follow-up operate at two-hour intervals. The
controller uploads audited stable raw/full checkpoints, resolved configuration,
code and logs to the existing private HF repo under selfplay_20260914; model.pt
continues to denote the verified original baseline. Tokens live only in RAM.
Server-owned one-shot timers stop training September 16 at 07:00 Asia/Shanghai,
with a hard stop at 07:10. The 11:00 reclaim deadline leaves time for independent
evaluation and complete local recovery; no recurring clock polling is used.

Before the final held-out evaluation choose the highest monitored H2H checkpoint
that retains >=99% random wins (ties: earliest rollout), with a finite/reload audit.
Freeze its hash before opening held-out results. Fresh final seeds: GPU
setup=5191400/game=6191400 (2048 each), CPU setup=5291400/game=6291400 (512 each),
direct H2H setup=5391400/game=6391400 (2048 games). Candidate promotion requires
no observed vs-random regression on either paired backend and a pair-bootstrap
95% direct-H2H score lower bound above 50%. Otherwise preserve baseline as the
recommended model, retain every candidate and negative result, and use remaining
lease time for a diagnosed continuation, not blind resubmission or repeated use
of the same held-out set. No GitHub push is authorized by this protocol.

# JunQi two-direction continuation, 2026-09-19

Training source is frozen at 2cae8a9efa555a1de618d95b71826fd1dc3ce5c9.
All arms restore the exact 20,000-rollout full checkpoint (policy, optimizer,
1,920,034 update counter). Source SHA256 is recorded in provenance.json.
No EMA, learned belief, arrangement, reward shaping or architecture changes.

A: fixed uniform-legal magnet temperature 0.002, seed 9192, server A GPU 0.
B: fixed uniform-legal magnet temperature 0.006, seed 9192, server B GPU 0.
C: independent-seed replication of direction A, seed 9193, server B GPU 1.
C is a stability check, not a third training direction or hyperparameter search.
Other settings identical: LR 1e-5, 128 envs, 512 steps, BF16, one PPO epoch.
Four smoke rollouts are discarded. Formal training starts from identical weights
and optimizer state; initial counters are preserved rather than relabeled zero.

Primary target: 24,096 total rollouts (4,096 new) in A and B. Compare this fixed
checkpoint, never select by held-out performance. If either arm misses target,
use largest numbered checkpoint in the intersection of both available runs;
report actual common budget and partial status. C uses same target if feasible.
B may extend to 26,144; that extra-budget result is reported separately.

Fresh evaluation suite: 2,048 GPU/BF16 paired games per H2H matchup, alternating
team assignments with equal setup/random seeds. Setup seed 9,191,900; game seed
10,191,900. Matchups: A vs B, A vs start, B vs start, start vs original; C vs start
and C vs B are secondary. Random-opponent suites: 512 games, setup 11,191,900,
game 12,191,900. Extra-budget B uses separate seeds 13,191,900 / 14,191,900.
Report strict win fraction and score=(wins+draws/2)/N separately; pair-bootstrap
95% intervals group both seats for each seed. Do not pool evolving policies or
repeated monitoring games as independent evidence. No inference about human skill.

A and C soft cutoff September 19 08:30 Asia/Shanghai; 180-second hard stop grace.
B training cutoff 12:30; evaluation and publication cutoff 14:20. Lease ends A
10:00, B 15:00. Each GPU hosts one compute subprocess. GPU1 transitions from C
to evaluation. Server backups every two hours, plus matched and terminal saves.
Only leased GPUs are used; no Slurm or extra resources acquired.

Publish exact checkpoints, raw policies, configs, source and logs to existing
private HF repository under two_directions_20260919. Publish comparison reports,
per-game evidence and reproducible scripts to a new results branch in existing
GitHub JunQi repository. Credentials are only in RAM and excluded from artifacts.
This protocol is authorized by the user's September 19 request, including GitHub.
Status may be running, partial, failed or completed; submission is not completion.

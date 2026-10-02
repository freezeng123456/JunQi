# Joint BeliefNet versus frozen policy: Beijing V100 pilot

Status after the user's 2026-10-01 design correction: this document records the completed **legacy rule-input pilot**. The next experiment removes heuristic rule probabilities from both policy and belief inputs, trains a new fact-input policy, and freezes that policy for direct, marginal-search and joint-search comparisons. See the updated [review and replacement design](JOINT_BELIEF_REVIEW_20261001.md). The facts-only policy path was implemented on 2026-10-02; the new joint-belief/search migration remains pending. See [the new baseline run](FACTS_ONLY_TRAINING_20261002.md). The results below remain historical evidence.

## Experiment definition

Completion scope: this pilot trained one move-policy network and a separate joint-belief network. Its paired matches compare direct inference against optional search around the same move weights. Sharing the move policy is a valid controlled design for the Ataraxos staged approach; independently training a second move network is not a prerequisite. The [2026-10-01 review](JOINT_BELIEF_REVIEW_20261001.md) identifies missing public constraints, stale belief particles and uneven action-value estimates that limit this pilot. Here, policy means the learned neural action distribution; fallback bypasses search and samples that distribution.

This first experiment compares the same frozen move policy in two modes:

- Baseline: rule-constrained marginal probabilities in the existing 317 spatial channels.
- Candidate: identical move-policy weights, plus an autoregressive joint BeliefNet and a bounded search update.

Both use `junqi.graph317.cls.v1`: 317 spatial channels, zero global scalars, CLS pooling. The historical N/M/R checkpoints retain their original input contracts. This pilot does not retrain a second move policy and does not measure the isolated effect of autoregression versus an equally budgeted marginal sampler. A later ablation should hold search constant to answer that question.

The approach follows the staged frozen-policy belief training and sampled-world search in [Ataraxos](https://arxiv.org/html/2511.07312v1). It adapts two-player Stratego to four-player DARK JunQi: all three other armies, including the teammate, are hidden. This small-network pilot is not a full paper reproduction.

## Components and information boundary

`junqi_rl/belief/legacy_joint.py` (originally `joint.py`) builds the historical pilot's metadata for 75 tokens: all 25 original identities in each other army, including dead pieces. The sequence is relative owner, then original slot. Inputs contain current/initial location, public counters, public rule support and known original inventory. Hidden types are read only by a separate supervised-target function. The current `joint.py` instead packages facts/history under a new schema and is incompatible with this pilot's network.

`JointBeliefNet` reuses the marginal model's CNN and board encoder, restricts attention to the 129 playable cells, and adds a causal decoder. Parallel teacher forcing sees only preceding true labels. Sampling encodes the board once, projects cross-attention memory once, and appends self-attention K/V caches. Hall capacity masks over the 12 piece types exclude choices that make the remaining assignment impossible. Neural joint decoding runs only when requested; ordinary PPO collection stays unchanged.

Checkpoint contract `junqi.belief317.autoregressive.v1` records architecture, dimensions, sequence, DARK visibility, model config and SHA-256 of its frozen parent policy. The loader rejects mismatched parent weights or input contracts. Optimizer state and RNG state support resumed joint training.

`junqi_rl/analysis/legacy_joint_search.py` constructs each counterfactual world from the observer's known own lineup and sampled identities. It replays public actions and broadcasts to reject inconsistent worlds. It never copies the actual opponent/teammate identities or their private combat memory into a search world. Each sampled player receives its own legitimate observation reconstructed within that world. Counterfactual observations and future policy calls are batched; the rules engine runs on CPU in this initial backend.

A per-observer particle cache advances surviving sampled worlds through new public events. Rejection conditioning is approximate with this small particle budget; it does not maintain exact opponent-policy likelihood weights. If no sampled world survives, the candidate uses its frozen blueprint and records a fallback. After failed generation, it waits 32 public plies before trying again, avoiding repeated decoding at nearly unchanged states. High fallback rates limit the interpretation of a weak or negative result.

## Frozen pilot budgets

| Stage | Budget |
|---|---|
| Move policy | Fresh initialization; seed 20261001; 512 × 32 environments × 128 steps = 2,097,152 environment steps |
| Move architecture | Graph stem 64 × 2; transformer 128 × 4; categorical value; combat v1 and tactical v2 features |
| Move training | Float32; Adam learning rate 0.0001; self-play; no reward shaping; no arrangement training |
| Joint dataset | 1,024 observer rows from frozen-policy self-play; seed 20261002; four observers of a position remain in the same split; environment-disjoint validation |
| Joint network | CNN 32 × 2; encoder/decoder 64 dimensions, two layers each; four heads; dropout zero |
| Joint optimizer | 512 updates, minibatch 16, Adam 0.00005, gradient clip 0.5 |
| Search | Up to two particles, eight top-policy candidate actions, four plies; eight proposals per decode, up to two batches; alpha 0.002, beta 0.02 |
| Evaluation | 16 fixed setups × both team assignments = 32 matches; seed 20261003; up to 4,000 actual plies per match |

Search averages each candidate's return over sampled worlds, follows the frozen blueprint for every simulated player, and signs the leaf value for the root team. It performs one magnetic mirror-descent update using a uniform-over-pieces-and-destinations magnet. Unvisited root actions retain the blueprint value estimate. Candidate truncation and the small rollout budget differ from the original paper and are reported explicitly.

Validation reports conditional NLL and teacher-forced accuracy over unknown tokens, not unconditional marginal identification accuracy. The head-to-head report includes wins, losses, true draws, unfinished games, descriptive Wilson intervals, search fallbacks, proposal acceptance, explored-action coverage, actual simulated steps and p50/p95 decision latency. Unfinished matches are never relabeled as draws. Team-swapped outcomes are paired, so per-match Wilson intervals do not establish an independent paired significance test.

## Run and resume

The frozen archive, manifest, resolved configs and launch script live under the task's `work/beijing-v100-preflight/launch-20261001/experiment-20261001-v3`. Each source file has its own SHA-256. Launch one V100 with a two-hour scheduler limit. Build `junqi_cuda` for sm70 on the allocated node, then run GPU input/history/gradient gates, initial and resumed PPO smoke checks, policy training, joint training, one resumed joint update and paired evaluation.

The launcher's working directory is `SLURM_SUBMIT_DIR`, because Slurm executes a copied script under its own spool directory. Failed earlier attempts are retained separately.

The historical 28-dimensional count reduction uses `atomicAdd`, so its float32 result can differ by a few last bits. Its long-history regression uses rtol/atol 1e-6 only for global floats. Spatial channels, legal masks and boolean flags retain exact comparisons; the long-history test covers both zero-global and legacy modes.

Evaluation atomically saves after every match. `legacy_compare_joint_search.py --resume` verifies protocol and checkpoint hashes, preserves completed matches and deterministically reruns unfinished ones. The original command is a compatibility shim. Training/evaluation completion requires actual logs and checkpoint audits; source tests and job submission alone do not establish it.

## Observed result: 2026-10-01, job 2347635

The frozen v3 run completed on Beijing node `g0035`, Tesla V100-SXM2-32GB, PyTorch 2.5.1+cu121. Slurm reports `COMPLETED`, exit `0:0`, elapsed `00:31:35`. All 329 source hashes matched before submission. The allocated-node gate suite passed 50 tests with four skips; it includes both CUDA and CPU search tests.

The fresh baseline completed 512 rollouts, 2,097,152 environment steps and 12,264 optimization updates. Its 939,061-parameter policy uses rule-constrained marginal beliefs, with the learned marginal BeliefNet disabled. Checkpoint audits found all 360 training-checkpoint tensors finite. Initial/resumed PPO smoke checks advanced the optimizer and changed policy weights. The log's `kept_mean=nan` and `kept_min=nan` denote unused statistics in global minibatch grouping, rather than nonfinite model weights; `nan_skip` and `grad_skip` remained zero.

The joint model has 301,516 parameters. Collecting its 1,024 rows took 26.74 seconds; 512 optimizer updates took 10.20 seconds. On 256 environment-disjoint validation rows, conditional NLL fell from 1.74204 to 1.58248, and teacher-forced accuracy rose from 26.38% to 34.68% over 18,732 unknown targets. These conditional metrics do not measure unconditional identification accuracy or playing strength. A separate resumed update advanced the joint optimizer from 512 to 513 and changed 99 network tensors. Evaluation used the original 512-update joint checkpoint.

| Mode | Wins | Losses | Draws | Unfinished | Win rate | Score, draw = 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 17 | 9 | 6 | 0 | 53.125% | 62.5% |
| Joint BeliefNet + search | 9 | 17 | 6 | 0 | 28.125% | 37.5% |

All 16 starting setups were verified identical across their two team assignments. The 32 matches took 450.06 seconds. Resampling starting setups as 16 clusters while preserving both assignments gives a descriptive candidate-score 95% percentile bootstrap interval of 21.875%–53.125% (10,000 samples, seed 20261004). This small pilot does not establish a robust strength ranking.

Across 6,371 candidate decisions, 6,004 (94.24%) fell back to the frozen policy because no public-history-consistent particle was available. Search ran on 367 decisions; it evaluated 2,936 candidate actions out of 15,716 legal actions at those decisions. There were 799 joint refreshes, 11,680 proposals and 280 accepted worlds, with early stopping after enough particles. Public replay used 117,664 world steps and future rollouts used 17,952 world steps. Mean candidate decision time was 61.78 ms, including policy inference, generation and search. Per-match p50/p95 values remain in the original report; they are not averaged into a global percentile.

Of the 6,004 fallbacks, 5,369 decisions were inside the failed-generation retry cooldown. The 94.24% figure is therefore the fraction of decisions without search, not an immediate rejection rate for model samples. The joint network supplies sampled worlds to search; it does not replace the rule-marginal channels fed into the frozen move policy in this pilot.

Sixteen full 75-identity assignments took 148.74 ms in the standalone sampling measurement. Encoder reuse, incremental decoder K/V, particle caching, batched policy/value calls and failed-refresh cooldown are implemented. Public replay still runs on CPU. The high rejection rate makes better history-conditioned proposals and constrained rejuvenation a higher priority than increasing the rollout budget. Low conditional NLL alone does not guarantee that a 75-piece assignment satisfies all past combats.

This candidate underperformed in the observed matches. The experiment compares baseline against joint belief **plus** a small search budget; it neither isolates the value of joint belief nor reproduces the full Ataraxos system. Preserve the artifacts as legacy evidence. The next ablation uses a newly trained fact-input policy, improves valid-world coverage, compares marginal and joint samplers under the same search budget, and expands evaluation seeds.

The task-local recovery directory is `work/beijing-v100-preflight/launch-20261001/results-2347635/` beside the repository's parent `outputs/` directory. It contains the original reports/logs/configs, source manifest, artifact hashes, a derived aggregate report, browser completion proof, and three hash-verified model files. Local CPU strict loading validated both inference models, the joint model's exact parent-policy hash, finite weights, and equality of the baseline inference and training policy weights. This local recovery check complements the actual V100 training evidence.

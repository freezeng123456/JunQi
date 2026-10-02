# 训练配置索引

## 当前事实模型

| 配置 | 用途 |
| --- | --- |
| `facts_base.yaml` | 独立的 317/0 事实模型与 PPO 配方；作为父配置使用 |
| `v100_facts.yaml` | 原始 V100 512-rollout 实验；解析值保留 |
| `v100_facts_games.yaml` | 当前 CLI 默认入口；按累计完整对局结束，默认 2,000 局 |

`python scripts/train.py --validate-only` 解析当前默认配置。显式 `--config` 可选择
历史实验；CLI 和 `--set` 的覆盖优先级保持一致。构造 dataclass 的默认值继续服务
历史加载，不能替代新训练的显式 profile。没有提供 resume 时从新权重开始。

2026-10-03 提取 facts 基础配置时，44 份既有 YAML 的解析值逐字段核对相同。
历史源码/config 哈希按原实验归档保存，不用今天的配置文件哈希覆盖历史记录。

## 历史配置

以下文件用于明确的历史复现或对照；文件名中的 `current`、`default`、`clean`
表示当时的用途。它们不代表当前事实主线，原始内容与继承关系保留。

- `ataraxos_selfplay.yaml`
- `ataraxos_selfplay_value_all.yaml`
- `ataraxos_selfplay_value_all_ddp2.yaml`
- `default.yaml`
- `guarded_continue_20260914.yaml`
- `h20_10m_current.yaml`
- `h20_phase_a_90_random_20260817.yaml`
- `h20_phase_b_selfplay_30k_20260817.yaml`
- `h20_recover_30k_20260817.yaml`
- `h20_repaired_selfplay_30k_20260817.yaml`
- `iteration_baseline.yaml`
- `iteration_combat_outcomes.yaml`
- `iteration_joint_belief.yaml`
- `iteration_joint_belief_clean.yaml`
- `phase_a_random_no_belief_20260913.yaml`
- `selfplay_continue_20260914.yaml`
- `selfplay_iter.yaml`
- `small.yaml`
- `t4_10m_engineering_opt.yaml`
- `t4_optimized.yaml`
- `t4_smoke.yaml`
- `train_beat_random.yaml`
- `train_beat_random_v10.yaml`
- `train_beat_random_v11.yaml`
- `train_beat_random_v12.yaml`
- `train_beat_random_v13.yaml`
- `train_beat_random_v14.yaml`
- `train_beat_random_v4.yaml`
- `train_beat_random_v6.yaml`
- `train_beat_random_v7.yaml`
- `train_beat_random_v8.yaml`
- `train_beat_random_v9.yaml`
- `v100_spatial_belief.yaml`
- `v100_spatial_rules.yaml`
- `v35_combat_memory_full.yaml`
- `v37_bugfixes_1500R.yaml`
- `v38_value_rescue_100R.yaml`
- `v40_dark_pid_features.yaml`
- `v41_curriculum_T.yaml`
- `v42_planB_T_vs_random.yaml`
- `v43_planB_belief_on.yaml`
- `v44_planB_v6_eaten_by_pid.yaml`

# 强化学习高效迭代方案

## 目标

用“正确性门禁 → 低成本筛选 → 多种子复验 → 长程训练”的漏斗替代直接启动长
实验。每次只验证一个假设，同时比较样本效率、训练稳定性和吞吐，不再只看某个
checkpoint 的单次胜率。

当前仓库已有两条重要证据：

- `docs/PROGRESS.md` 显示 v17–v32 多个配置在随机对手上约 0.80 附近进入平台，
  单纯增加 rollout 数没有突破，主要矛盾不是算力不足。
- `docs/PROGRESS_2026Q2_CURRICULUM.md` 显示 128 局评估会出现很大波动，固定四方
  镜像布局还会泄漏 `piece_id → piece_type`。因此必须使用 paired seed、多种子和
  置信区间。

## 0. 先完成正确性门禁

以下任一项未通过时，不比较算法优劣：

1. CPU/CUDA 的规则、观测、合法动作和终局奖励 parity 全部通过。
2. 同一配置、同一 seed 的短 rollout 能复现；训练日志中的
   `train/nan_skip_total`、`train/grad_skip_total` 均为 0。
3. 审计每种 setup 模式的实际分布，确认训练和评估没有信息泄漏。
4. 修复 setup pool 的进程级耦合后再做正式比较：
   `GpuRollout._ensure_reset_pool` 当前使用模块级 `_reset_pool_uploaded`，
   ArrangementNet 刷新也会改写同一个 CUDA pool。训练环境和缓存的评估环境并不
   真正拥有独立分布。
5. 不用 `disable_arr_train` 模拟“无 ArrangementNet”。该开关只跳过优化，
   仍会生成并刷新布阵。无该子系统时应设置 `arr.enabled: false`。
   同理，无 BeliefNet 时设置 `belief.enabled: false`，避免仍收集大容量监督样本。

## 1. 四级实验漏斗

### L0：配置检查

只解析配置，不分配模型和环境：

```bash
python3 scripts/train.py \
  --config configs/iteration_baseline.yaml \
  --validate-only
```

未知字段、非法 dtype、错误的刷新周期和互斥 setup 模式应立即失败。

### L1：运行时冒烟

用 2 个 rollout 验证编译、显存、收集、反向传播、保存和评估链路：

```bash
python3 scripts/train.py \
  --config configs/iteration_baseline.yaml \
  --set total_rollouts=2 env__num_envs=16 env__steps_per_env=64 \
        eval_every=1 eval_num_games=32 save_every=1 \
        save_dir=exps/smoke_candidate
```

门槛：

- 无异常、NaN、Inf、梯度跳过和 ongoing 评估局；
- `rollout/n_policy_kept > 0`；
- checkpoint 可加载并继续一个 rollout；
- 记录 `time/collect_s`、`time/train_s` 和峰值显存，作为性能回归基线。

### L2：单假设筛选

以 `configs/iteration_baseline.yaml` 为 control，每个 candidate 只改一个变量。
先运行 30 个 rollout，使用相同的 3 个训练 seed；每 5–10 个 rollout 做 256 局
paired-seed 评估。

比较：

- 横轴统一用环境步数，而不是 rollout 编号或运行时长；
- 报告三个 seed 的中位数、最差 seed、评估 AUC 和末段均值；
- 同时报吞吐，避免用明显更高的计算量换来很小的胜率提升；
- 不以单个 `ckpt_best.pt` 判定胜负。

候选只有在训练健康指标全部通过、至少 2/3 seed 优于 control，且 paired score
中位提升超过 0.02 时进入 L3。该 0.02 是筛选阈值，不是模型发布标准。

### L3：复验

将入选项扩展到 100 个 rollout，仍使用同一组 seed。最终 checkpoint 对两个队伍
位置各运行 1024 局，共 2048 局；报告胜/负/和/ongoing、平均步数、Wilson 95%
区间以及对历史 checkpoint 的交叉对局。

正式结论应使用 paired-game bootstrap 的“candidate − control”区间；两个模型各自
的 Wilson 区间不能代替差值检验。

### L4：长程训练

只有 L3 通过的配置才扩大 rollout 数或启用 DDP。多 GPU 首先用于并行独立假设和
独立 seed；确认单个配方后，才用 DDP 提升该配方的数据吞吐。

## 2. 假设优先级

按当前证据依次测试：

1. **数据与评估正确性**：setup pool 隔离、两队位置平衡、固定 paired seed。
2. **PPO 更新比**：以 v32 已使用的 `num_epochs_per_rollout=1` 为 control，只测试
   1 与 2；避免默认 4 epoch 让旧样本反复更新并降低墙钟数据量。
3. **对手分布**：随机对手 control，对比“随机 / EMA 自博弈 / 历史联赛”的固定
   混合比例，重点观察 defensive-stall、平均局长和最差 seed。
4. **BeliefNet**：单独开启，先要求留出集 CE、top-1、校准误差优于库存均匀先验，
   再判断是否提升对局。不要同时改 MoveNet。
5. **ArrangementNet**：BeliefNet 结论稳定后再开启；记录实际 setup 熵、重复率和
   各座位覆盖率，防止布阵策略改变成为隐藏混杂变量。
6. **网络扩容和搜索**：只有数据管线与辅助网络已证明有效后再做，避免用参数量
   掩盖训练目标问题。

## 3. 每个实验必须保存的证据

- Git commit、完整 `cfg.runtime.yaml`、训练 seed、评估 seed、设备和 world size；
- 环境步数、collect/train 时间、峰值显存；
- policy/value/entropy/KL loss，return/advantage 分布，保留样本比例；
- NaN/梯度跳过累计值；
- 精确胜负和局数，不只保存浮点胜率；
- 最终 checkpoint、固定种子的 replay 和与 control 的 paired 对局结果。

发现 NaN、规则 parity 失败、setup 分布不符、有效策略样本为 0，或连续两次评估
明显低于 control 下界时立即停止；这类运行只用于诊断，不进入模型排名。

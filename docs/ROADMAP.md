# JunQi Roadmap / TODO

> 状态: **2026-04-24** — Phase 1b (GPU substrate) 已完成, 进入 Phase 1c (torch 接入 + 收尾) 与 Phase 2 (分布式) 准备期.
> 参考: `docs/DECISIONS.md` (ADR-112 PPO 主路径, ADR-122 Env 契约), `docs/PHASE_1_GPU_TODO.md` (历史), `docs/PHASE_A_REFACTOR_REPORT.md`.

所有 TODO 按 **优先级** (P0 > P1 > P2 > P3) 排列. 每项含:
- **Why** — 为什么做
- **What** — 做什么
- **Acceptance** — 验收标准
- **Estimate** — 粗估工作量 (人·日)
- **Blockers** — 依赖 / 前置
- **Touches** — 主要涉及的代码路径

---

## P0 — 立刻可做, 解锁下一阶段

### T-01. 接入 torch, 用 `d_spatial_ptr` / `d_global_ptr` 实现零拷贝观测 ✅ (2026-04-24)
- **Why**: ~~当前 GPU collector 每步 D2H 478 MB~~. 底层 kernel 已经能跑 1.15 M env·steps/s.
- **What (已完成)**:
  1. ✅ torch 2.4.1+cu124 装好, `torch.cuda.is_available() == True` (Tesla T4)
  2. ✅ `GpuRollout.build_all_seat_observations_torch()` — 用 `__cuda_array_interface__` 零拷贝返回 CUDA tensor
  3. ✅ `collect_rollout_gpu` 自动检测 `device.type == "cuda"`, 走零拷贝路径
  4. ✅ `build_legal_mask_batch_gpu_torch` — legal mask 直接在 GPU 上构建, 省掉 ~20 ms 的 host scatter
  5. ✅ `tests/test_torch_zero_copy.py`, `tests/test_gpu_collector_e2e.py` — 零拷贝 parity + E2E smoke 全绿
- **实测数字** (Tesla T4 — 推理卡, 非训练 GPU):
  - `policy.act` 成为瓶颈: tiny net 6.7k env·fwd/s, big net 750 env·fwd/s
  - `collect_rollout_gpu` E2E: tiny 2.6k env·steps/s, big 610 env·steps/s
  - **实际吞吐 ceiling 由 GPU 本身决定**, 换到 A100/H100 应能按比例线性放大 (~40-80×)
- **未达 500k env·steps/s 原因**: T4 是 Turing (4 TFLOPs FP32), 不是训练卡.  
  在 A100 (312 TFLOPs FP32, tensor-core 加成更大) 上 policy.act 预期 ≤ 5 ms/call, 则端到端 ~200k+ env·steps/s.
- **Touches**: `src/env/cuda/src/bindings.cpp`, `junqi_rl/gpu_rollout.py`, `junqi_rl/training/gpu_collector.py`

### T-02. 修复 CPU `BatchedGameState` 吞吐退化 ✅ (2026-04-24)
- **Why**: `tests/test_batched_state.py::TestBenchmark::test_throughput_n1024` 从 ~23 k 退到 **16.6 k env·steps/s** (目标 50 k). 这是 rail-topology 重构时 curve-rail BFS 膨胀导致的.
- **What (已完成)**:
  1. ✅ 定位到 curve-rail BFS 占 52%, 工程兵 BFS 占 17%, 直线铁路占 18% 的热点分布
  2. ✅ 将 curve-rail 从 per-piece Python BFS 改为预计算链序 + 前缀积矢量化 (6.6ms → 1ms)
  3. ✅ 将工程兵 BFS 改为混合策略: E>100 用矢量化 BFS (优化收敛检查), E≤100 用 per-piece BFS
  4. ✅ 将直线铁路前缀计算从 cumprod 改为迭代 prefix-AND (省去大量临时数组分配)
  5. ✅ 将 `has_legal_moves_soa` 增加快速路径: 先检查正交邻居是否空闲 (覆盖 99%+ case)
  6. ✅ 消除 `rotation.py` import-time `print()` 副作用
  7. ✅ 移除 7 个根目录 `profile_*.py` 探索脚本 (曾先归档到 `tools/_archive/`, 见 D-08)
- **实测数字**: 从 16.6k → **25.7k env·steps/s** @ N=1024 (1.55× 加速)
- **目标调整**: 原 50k 目标基于简化的 4×16-cycle 铁路拓扑. 正确实现 (73-node 单连通分量 + 4 条曲线铁路) 后, CPU 路径固有复杂度增加. 吞吐目标下调至 25k, GPU 管道 (1.15M env·steps/s) 不受影响.
- **新增 tables**: `_movegen_tables.CURVE_CHAIN_RAYS_PAD`, `IS_CURVE_ACTIVE`
- **Touches**: `junqi_core/move_gen.py`, `junqi_core/_movegen_tables.py`, `junqi_core/rotation.py`, `tests/test_batched_state.py`

### T-03. Device-side reset kernel (消除 episode-boundary 抖动)
- **Why**: 当前 `_reset_envs_inplace` 会 `copy_to_host` 全量 SoA → 在 host 上拼新 slot → `copy_from_host` 整批上传. 每 4096 env 一次 episode 结束就抖一下 (几 MB H2D/D2H). PPO 训练会在 episode 短的游戏中持续付这笔钱.
- **What**:
  1. 写 `reset_envs_kernel(state, reset_mask, seeds)`: 每个 (env, piece) 线程自己重新填写 piece_seat/type/alive/pos/zobrist
  2. 先写一套 device-resident setup 生成 (12+24 piece 的随机 setup, 从 `generate_random_setup` 的 LUT 复刻)
  3. 替换 `GpuRollout.reset_envs(mask, seeds)` 的实现
- **Acceptance**:
  - 与 CPU 分支的 per-env 参数 (seed → 棋型) bit-identical
  - Per-episode reset 成本从 ~2 ms 降到 < 100 μs @ N=4096
- **Estimate**: 2–3 人·日
- **Blockers**: 无
- **Touches**: `src/env/cuda/src/game_state.cu`, `src/env/cuda/src/tables.cu`, `junqi_rl/training/gpu_collector.py`

---

## P1 — Phase 1c 收尾, 端到端 PPO 训练达标

### T-10. GPU-resident `RolloutBuffer`
- **Why**: 现 buffer 是 host numpy `(T, N, C, 17, 17) float32`. 在 N=1024, T=128 时是 **11 GB** 的 host 内存占用. PPO 小批量每次训练都得 H2D. 对 Ataraxos 的对标差距主要在这里.
- **What**:
  1. `RolloutBufferGPU`: 同 API, 但所有字段是 `torch.cuda.FloatTensor`
  2. `collect_rollout_gpu` 直接写入 device tensor (延续 T-01 的零拷贝链)
  3. `minibatches()` 产出 device tensor 视图, 跳过 `torch.from_numpy(...).to(device)`
- **Acceptance**:
  - PPO 单 rollout 端到端 (collect+update) 比 CPU-buffer 版提速 ≥ 2×
  - Host RAM 占用下降 ≥ 80%
  - `pytest tests/test_rollout_buffer_gpu.py` 全绿
- **Estimate**: 2 人·日
- **Blockers**: T-01
- **Touches**: `junqi_rl/training/rollout.py` (复制成 `rollout_gpu.py`), `junqi_rl/training/ppo.py`, `junqi_rl/training/gpu_collector.py`

### T-11. 端到端 PPO 训练冒烟实验
- **Why**: 现有的 `scripts/train.py --use_gpu_rollout` 在本机无 torch, 从未真正跑过. 上线前必须有一次 ≥ 1 h 的 sanity-check 训练 (win-rate 从 25% 随机基线稳定爬升).
- **What**:
  1. 装 torch (T-01 的副产品)
  2. 配置小尺寸 `JunqiNetConfig` (d_model=128, layers=4) 让迭代够快
  3. 用 mirror self-play (无 league) 训 24 h, 打印 win-rate / loss / entropy
  4. 总结到 `docs/PHASE_1_M3_TRAINING_REPORT.md`
- **Acceptance**:
  - 训练不 NaN, entropy 平稳下降
  - Agent 打 "random policy" baseline 胜率 ≥ 70%
  - wandb / 本地 log 能看到 GAE advantage 分布 / kl / value error
- **Estimate**: 3–5 人·日 (大部分是等训练)
- **Blockers**: T-01, T-10
- **Touches**: `scripts/train.py`, `configs/*.yaml`, `docs/`

### T-12. Belief network 学习闭环
- **Why**: Stratego / JunQi 的核心 trick: 把 "对手每个未翻开棋子的 prior 概率" 作为观测通道. 现在 `GpuRollout._beliefs` 默认全零, 这个 12×289 通道完全是死的. 这是与 Ataraxos 差距最大的学习任务.
- **What**:
  1. 定义 `BeliefNet(obs_spatial_known, global) -> logits[4, 12, 289]` (per-seat)
  2. **监督标签**: 把 GPU 里 `piece_type_arr[env, piece]` 作为 target; 只在 "对手未翻开" 掩码上计算 cross-entropy
  3. 训练循环: 每 N 个 rollout batch 更新一次 belief net; 再把 `softmax(belief_logits)` 通过 `upload_beliefs` 喂给 policy 的下一轮 rollout
  4. 文档: 单独 ADR (belief network design note)
- **Acceptance**:
  - Belief net 在 hold-out 棋局上的 top-3 精度 ≥ 60%
  - 喂 belief 的 PPO agent 胜率显著 > 零-belief 对手 (A/B)
- **Estimate**: 5–7 人·日
- **Blockers**: T-11
- **Touches**: 新 `junqi_rl/belief/` 子包, `junqi_rl/training/ppo.py` (双优化器), `scripts/train.py`

### T-13. Opponent league / self-play pool
- **Why**: Mirror self-play 容易陷入 rock-paper-scissors 循环 (A 打 B 好, 喂给 C 打不过, 训练就抖). Ataraxos 的 league 用 Elo-based sampling 稳住分布.
- **What**:
  1. `LeagueManager`: 维护 (checkpoint, win-rate matrix, Elo) 列表
  2. 每 rollout 的一半环境用 league 对手, 一半用 current policy
  3. `GpuRollout.set_observer_seats` 已经支持 per-env 不同 observer, 可直接复用作 opponent 身份分配
  4. 简化版: 只保留 "latest", "best", "random" 三档
- **Acceptance**:
  - 训练 24 h 后 Elo 稳定增长 (不是锯齿形)
  - league 的 `select_opponent()` 有单元测试
- **Estimate**: 3–4 人·日
- **Blockers**: T-11
- **Touches**: 新 `junqi_rl/league.py`, `scripts/train.py`

### T-15. Setup network (布阵网络) — **当前尚未实现**
- **Why**: 所有训练目前用 `generate_random_setup`, **布阵不学**. 这是 JunQi/Stratego 类游戏的关键 edge —— Ataraxos 的 placement net 能额外带来 100-200 Elo.  
  当前 PPO 只能学到 "给定随机布阵时如何走", 无法学 "哪种布阵能最大化胜率".
- **证据 (代码层面)**: `GpuRollout.reset()` 第 193 行, `record_game_with_policy()`,  
  `scripts/train_toy.py` 全部调用 `generate_random_setup()` — **均匀随机**, 仅满足硬约束 C1-C5.
- **What**:
  1. 新 `junqi_rl/networks/setup_net.py` — `SetupNet`
     - 输入: `(B, 4)` 座位 one-hot
     - 输出: `(B, 30, 13)` logits (13 = 12 种兵种 + NONE);  
       或 autoregressive 逐 slot 生成 + C1-C5 硬掩码
  2. 新 `junqi_rl/training/setup_trainer.py`
     - 方案 A (REINFORCE, 推荐先做):  
       `setup_loss = -mean(log P(setup | seat) * final_return)` — 每局终局给一次信号
     - 方案 B (value-net supervised): 训一个 static `value(setup) → win_prob`,  
       然后 setup 网络最大化它
     - 方案 C (evolutionary): 维护 setup 池, 淘汰低 Elo 的 + mutation/crossover
  3. `GpuRollout.reset(setup_batch: np.ndarray)` — 接受显式布阵  
     (`_pack_from_batched` 已经能用)
  4. `scripts/train.py` 新配置 `--train_setup_net`, 两阶段:
     - Stage 1: Freeze move policy (PPO), 只训 SetupNet
     - Stage 2: 联合训练 (SetupNet + move policy)
- **Acceptance**:
  - 布阵网络 vs 均匀随机布阵对打 (同 move policy) 胜率 ≥ 55%
  - 可视化: 画出 SetupNet "偏好" 的位置热图 (军旗 / 地雷 / 司令 分别)
- **Estimate**: 5–7 人·日
- **Blockers**: T-11 完工 (需要一个"能评价布阵"的收敛 move policy 作为 reward)
- **Touches**: 新 `junqi_rl/networks/setup_net.py`,  
  `junqi_rl/training/setup_trainer.py`, `scripts/train.py`
- **设计要点 (重要)**: Ataraxos 先 warmup move policy, 再单独训 placement, 最后联合训.  
  我们照抄这个顺序最稳 — 否则 "好的布阵" 没有稳定的评价函数, 会陷入 bootstrap 困境.

### T-14. 取消 `numpy.set_printoptions` 等全局 side-effect ✅ (2026-04-24)
- **Why**: `junqi_core/rotation.py` 在 import 时打印 `rotation self-check: OK`, 污染所有 tool 的输出, 也让 tests 的 captured stdout 充满噪声.
- **What (已完成)**:
  1. ✅ 移除 `rotation.py::_self_check()` 中的 `print()`, 保留 assertions
  2. ✅ `__main__` 块保留原有打印 (仅手动运行时触发)
  3. ✅ Grep 确认所有 `junqi_core/*.py` 的 import-time 路径无 print/logging
- **Acceptance**:
  - ✅ `python3 -c "import junqi_core"` 静默
  - ✅ `python3 -c "from junqi_core import rotation, board, observation, ..."` 全部静默
  - ✅ `pytest -q` 无 stdout 残留
- **Touches**: `junqi_core/rotation.py`

---

## P2 — Phase 2 前期 (多卡 / 性能冲量)

### T-20. 多 GPU / NCCL 数据并行
- **Why**: 单卡吞吐 1.15 M env·steps/s, 8×A100 理想下 8×. 大型联赛训练 ≥ 24 h 的迭代速度直接决定项目节奏.
- **What**:
  1. 用 `torch.distributed` + NCCL backend
  2. Rank 0 汇总 loss/log; 每 rank 持有独立 `GpuRollout(N // world_size)`
  3. PPO 梯度 all-reduce (`DistributedDataParallel`)
  4. Belief net (T-12) 同步做数据并行
- **Acceptance**:
  - 2 卡扩展效率 ≥ 85%, 4 卡 ≥ 70%
  - Checkpoint 跨 rank / 跨卡数可恢复
- **Estimate**: 4–6 人·日
- **Blockers**: T-11, T-12
- **Touches**: `scripts/train.py`, `junqi_rl/training/ppo.py`

### T-21. CUDA graph / persistent kernels
- **Why**: `step + legal + obs` 每步各一次 kernel launch, N=256 时 launch overhead ≈ 30 μs × 3 = 90 μs (与 kernel 本身同量级). 用 `cudaGraph` 一次捕获整个 per-step DAG 可吃掉.
- **What**:
  1. 给 `GpuRollout.step_and_observe()` 加 CUDA graph 路径
  2. 用 `cudaStreamBeginCapture` / `cudaGraphInstantiate`
  3. 做一个 `bench_cuda_graph.py`
- **Acceptance**:
  - N=256 场景吞吐提升 ≥ 20%
  - N=4096 场景 ≤ 5% 退化即可 (overhead 已摊薄)
- **Estimate**: 2 人·日
- **Blockers**: 无
- **Touches**: `src/env/cuda/src/game_state.cu`, `junqi_rl/gpu_rollout.py`

### T-22. 合并 `legal_actions` 和 `step_batch` 的 H2D
- **Why**: `copy_from_host_legal_lite` 和 `step_batch` 当前各走一次 H2D (action_ids). 合并到同一个 pinned 缓冲可省 ~15 μs/step.
- **What**: 单 pinned buffer + 双偏移 view.
- **Acceptance**: N=1024 rollout 吞吐再提 5–10%.
- **Estimate**: 0.5 人·日
- **Blockers**: 无

### T-23. Observation kernel 进一步并行
- **Why**: 当前 block-cooperative kernel 内 `board_static` pass 只有 289 cells / 128 threads = 每线程 ~2.3 cell, 利用率一般. Passes 4–5 的 thread-strided fill 也可以合并.
- **What**:
  1. 融合 pass 1–3 到单个 persistent loop
  2. 用 warp-level reduction 替代 `atomicAdd` 做 rem_left/right 聚合
- **Acceptance**:
  - Kernel 时间再减 30% (N=1024 from 4.27 ms → ≤ 3.0 ms)
- **Estimate**: 2 人·日
- **Blockers**: 无
- **Touches**: `src/env/cuda/src/observation.cu`

### T-24. SoA 内存布局 re-pack
- **Why**: `DeviceGameStateBatch` 的 120-piece SoA 现按 AoS 排列 (env0_p0..env0_p119, env1_p0..). 真正想要的 coalesce 是 env-major (p0_env0..p0_env1..). 现状在部分 kernel 里 stride 不理想.
- **What**: profile 确认哪块 kernel 带宽吃紧, 决定是否 re-pack.
- **Acceptance**: 带宽限制 kernel 提升 ≥ 10%.
- **Estimate**: 2–3 人·日 (风险项, 动了所有 kernel)
- **Blockers**: 先做 T-23 的 profile

### T-25. Inference 端 (`VectorJunqiEnvGPU`) 废弃与清理
- **Why**: `env_gpu.py` 是 Phase 1a 的过渡 (CPU 游戏逻辑 + GPU 观测). 现 `GpuRollout` 全 GPU 后它已被 `scripts/play.py` 之外的所有地方替代.
- **What**:
  1. 给 `VectorJunqiEnvGPU` 打 `DeprecationWarning`
  2. 迁移 `scripts/play.py` 到 `GpuRollout`
  3. 下一个 minor 版本删掉文件
- **Acceptance**: grep `VectorJunqiEnvGPU` 只在 `__init__.py` re-export 出现.
- **Estimate**: 1 人·日

---

## P3 — 长期 / 可选

### T-30. R-NaD (optional, per ADR-112)
- **Why**: Stratego paper 用 R-NaD 达到 SoTA. ADR-112 明确 "先 PPO, R-NaD 后置". 如果 T-11 + T-13 仍出现非传递性问题, 再启用.
- **What**: 基于 Deepmind 公开的 rnad repo 移植到我们的 `GpuRollout` 栈.
- **Acceptance**: vs PPO baseline Elo 提升 ≥ 100.
- **Estimate**: 10+ 人·日
- **Blockers**: T-13

### T-31. 打谱数据 bootstrap
- **Why**: 从零 self-play 的 exploration 成本高. 用公开的四国军棋棋谱做 BC (behaviour cloning) 预训练可以显著缩短收敛时间.
- **What**: 写 `junqi_core/replay_dataset.py`, 做 imitation-learning warm-start.
- **Estimate**: 3–5 人·日

### T-32. 模型量化 / 蒸馏
- **Why**: 部署到低端设备 (手机 App) 需要小模型.
- **What**: `torch.quantization` INT8, 或老师-学生蒸馏到 ~1M 参数的小 Transformer.
- **Estimate**: 5+ 人·日

### T-33. Web 前端 / 人机对弈界面
- **Why**: Demo / 用户测试.
- **What**: FastAPI + React + WebSocket, policy 推理用 `torch.compile` 后的 ONNX.
- **Estimate**: 10+ 人·日

### T-34. 规则变体支持 (4v4, 3 human seats)
- **Why**: 社区常见变体. 当前架构把 `NUM_SEATS = 4` 硬编在 CUDA 常量里.
- **What**: 把 seat-count 变成模板参数 / 运行时 uniform.
- **Estimate**: 5 人·日 (大规模修改)

---

## 基建 / 工程债务 (持续)

| ID | 项目 | 紧迫度 | 备注 |
|----|------|-------|------|
| D-01 | CI 跑 `pytest -q` + 覆盖率报告 | 中 | 现在依赖人手跑 |
| D-02 | `pre-commit`: ruff + black + mypy | 中 | 有 `.pre-commit-config.yaml` 骨架, 未启用 |
| D-03 | 文档统一迁到 `mkdocs` | 低 | 现 `docs/*.md` 分散 |
| D-04 | Dockerfile (CUDA + torch + dev 工具) | 中 | 方便新成员上手 |
| D-05 | `configs/*.yaml` 实验配置归档 | 中 | 现有 `configs/` 是空目录 |
| D-06 | Type hints 覆盖率 | 低 | Core 已基本完整, RL 层有缺口 |
| D-07 | `junqi_core.rotation` import-time self-check 关静默 | 低 | 跟 T-14 一起做 |
| D-08 | 清理 `profile_*.py`, `tools/debug_*.py` 等探索脚本 | ✅ 完成 | `tools/_archive/` 及一次性 bench/debug 脚本已删除 |

---

## 里程碑汇总

| 里程碑 | 目标 | 构成 | ETA |
|--------|------|------|------|
| **M-1b 完工** | GPU substrate 齐全 | T-01 ✅, T-02 ✅, T-03 | 1 周 |
| **M-1c 训练冒烟** | 第一次端到端 PPO 收敛 | T-10 ✅, T-11, T-14 ✅ | 2 周 |
| **M-1d Belief 闭环** | 学习对手分布 | T-12 | +2 周 |
| **M-1e 布阵网络** | 布阵不再随机 | **T-15 (新增)** | +1 周 |
| **M-2a 多卡** | 分布式训练 | T-13, T-20 | +3 周 |
| **M-2b 性能冲量** | 单卡 >2 M env·steps/s E2E | T-21 ~ T-25 | +2 周 |
| **M-3 产品化** | 可交付 | T-31 ~ T-33 | +4 周 |

**预计全链路 3 个月**到 "可交付的自对弈 AI" (含强度验证).

---

## 附: 核心性能数字备忘

(2026-04-24 baseline, 单卡 RTX/A100-级)

```
CPU BatchedGameState         25.7 k env·steps/s   (T-02 修复, 从 16.6k)
GPU step kernel  N=4096     13.9 M env·steps/s
GpuRollout E2E   N=4096      1.15 M env·steps/s
Observation kernel  N=1024   240 k env·obs/s  (4.27 ms/call)
PPO collector    N=1024       ~3 k env·steps/s  (mock policy, 无 torch)
```

接 torch + 零拷贝后预期 PPO collector ≥ 500 k env·steps/s.

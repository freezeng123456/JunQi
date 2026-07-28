# JunQi 项目现状分析 (2026 Q2)

**日期**：2026-05-17
**分析者**：本次接手
**作用**：为本轮特征通道优化与"打过 90% 胜率"训练目标提供基线

---

## 0. TL;DR

- 项目主线已落地 **CombatMemory v6 layer-4**（`OBS_CHANNELS = 412`，
  commit `0c9ecca`，本轮新增）；v5 layer-3 (commit `a9d25f1`,
  `OBS_CHANNELS = 352`) 仍兼容。
- **四暗信息边界**：`test_dark_visibility_contract.py` +
  `test_observation_combat_memory_v5.py` + `test_observation_combat_memory_v6.py`
  钉死「仅 victim_seat==observer 时可见被吃子类型/slot/反向 killer pid」；
  v41 Plan A（四席同 canonical 布阵）因 piece_id→type 泄漏被判无效，
  **v42/v43 Plan B**（我方 T + 敌方随机）为正确训练分布；**v44**（本轮新增）
  在 v43 + v6 layer-4 反向投影上长跑。
- **训练天花板（截至 2026-05）**：v38 best **0.766**；v41（泄漏）**0.852**
  不可信；v42 Plan B R50 **0.555**（R40 stall 塌陷后恢复）。目标 **90%**
  vs-random 尚未达到；下一步 **v44** = Plan B + BeliefNet 训练 + v6
  反向投影 + `eval_random_dense.py`（2048 局 Wilson CI）。
- **结论**：规则/观测实现无作弊；v6 layer-4 把"我哪只 pid 被哪个敌 pid
  吃过"的反向锚信息显式给 CNN，预期能进一步缓解 v37 value collapse 与
  identity confusion；下文更新至 412 通道布局。

---

## 1. 现状：特征通道（Observation channel layout）

`OBS_CHANNELS = 412`（256 基座 + 50 CombatMemory v4 + 46 v5 layer-3 +
60 v6 layer-4；详见 `junqi_core/observation.py::CHANNEL_LAYOUT`）。
按用途归并：

| 类型 | 通道 | 通道数 | 内容简介 |
|---|---|---:|---|
| 主体身份 | `piece_own` | 12 | 自家棋按 12 种类型激活（仅 SOUTH 视角） |
| 主体身份 | `prob_teammate` | 12 | 队友棋按 BeliefTensor 概率分布展开（DARK 下 ≠ one-hot） |
| 主体身份 | `dark_teammate` | 1 | DARK 下队友 occupancy 掩码 |
| 主体身份 | `piece_left/right_side_enemy` | 2 | 左右侧敌人 occupancy 掩码 |
| 主体身份 | `belief_left/right_side` | 24 | 左右侧敌人 belief 概率分布 |
| 状态 | `dead_flags` | 3 | 队友/左/右是否阵亡 |
| 状态 | `flag_revealed` | 4 | 4 家旗已亮 |
| 静态 | `board_static` | 6 | 行营/大本营/铁路/九宫/弯道 |
| 状态 | `turn_history` | 2 | 平局进度 + 总步数进度 |
| 计数 | `move_bucket` | 8 | 我方/敌方每个棋子 move_count 桶（exact） |
| 计数 | `active_eat_bucket` | 8 | 我方/敌方 active eat 桶（cumulative） |
| 计数 | `passive_survive_bucket` | 8 | 我方/敌方 survive 桶（cumulative） |
| 历史 | `death_reason` | 12 | 4 视角 × 3 死因（KBE/HMB/MUTUAL）的死亡格 |
| 历史 | `dead_at_zero` | 2 | 死亡棋投影回起始格（我/敌） |
| 主体 | `piece_id` | 120 | 4 座 × 30 槽 one-hot（活子位置） |
| 历史 | `move_history` | 32 | 最近 32 步 src=-1, dst=+1 |
| **CombatMemory v4** | `cm_kill_mine_type` | 12 | 该敌子直接吃过我哪些 **类型**（multi-hot） |
| | `cm_kill_mine_ge` | 3 | 直接吃过我 ≥1/2/3 子（按 pid bitmap popcount） |
| | `cm_kill_other_ge` | 3 | 直接吃过盟友的 ≥1/2/3 子（DARK 下我看不到具体类型） |
| | `cm_chain_type` | 12 | 链式吃过的我方类型 multi-hot |
| | `cm_chain_ge` | 3 | 链式 ≥1/2/3 桥接我方 |
| | `cm_floor_ge` | 9 | rank 下界 cumulative：≥工兵 ≥排长 … ≥司令 |
| | `cm_is_gongb` | 1 | 已确证为工兵 |
| | `cm_not_gongb` | 1 | 已确证非工兵 |
| | `cm_dilei_candidate` | 1 | 在后两排且未被已知工兵攻击过 |
| | `cm_my_kill_count_ge` | 3 | theory-of-mind：对手对我棋的最低 kill count 视角 |
| | `cm_my_is_gongb` | 1 | 对手 AND-aggregate 后认为我是工兵 |
| | `cm_my_dilei_candidate` | 1 | 对手 AND 后我可能是地雷 |
| **CombatMemory v5** | `cm_kill_mine_count` | 12 | 该敌子直接吃过我各类型 **枚数**（÷3 归一化） |
| | `cm_kill_mine_slot` | 30 | 该敌子吃过 observer 的 slot-i（非全局 pid） |
| | `cm_recency` | 4 | direct/chain/floor 时间衰减（τ=32/256/64/128） |
| **CombatMemory v6** | `cm_eaten_by_pid` | 60 | **反向锚**：observer 自家 30 pid 上，标记吃过它的 60 个敌 pid（30 left + 30 right）。direct + chain，DARK 安全。 |
| **全局 (28 维)** | remaining_left/right | 24 | 左/右家剩余 12 类计数 |
| | flag_revealed | 4 | 4 家旗已亮 |

---

## 2. 你问的核心问题：AI **能否**识别对方棋子吃过我哪些子？

**答：能但严重缩水。**

### 2.1 内部 state 是足够的

`junqi_core.combat_memory.CombatMemoryState` 维护了：

```
direct_ate_my_pid_lo/hi  : (4, 120) uint64×2  — 这枚 killer 直接吃过 observer 的哪些 piece_id
direct_ate_my_type_mask  : (4, 120) uint16    — 上述受害者的 12 类 multi-hot
direct_other_count       : (4, 120) int16     — 直接吃过盟友（DARK 下不可知类型）几枚
chain_pid_lo/hi          : (4, 120) uint64×2  — 链式 120-pid bitmap
chain_ate_my_type_mask   : (4, 120) uint16
rank_floor               : (4, 120) int8      — 普通军阶下界
is_gongb / not_gongb     : (4, 120) bool
attacked_by_known_gongb  : (4, 120) bool
```

按规则（`apply_combat_event` in `combat_memory.py`）：

* `victim_seat == observer` 时，`direct_ate_my_pid` / `_type_mask` 都按真值
  写入；observer 的"自己被吃"信息是公开真实的；
* `victim_seat != observer` 时，DARK 规则下 observer 不知道 victim 的真
  实类型，仅累加 `direct_other_count`。
* chain 传播时所有 observer 都更新，但 chain_type_mask 只把 `victim` 是
  自己棋的类型加进去。

**这套设计是正确的，且不作弊。** golden tests `test_combat_memory.py` 验
证了它精确实现了"我看到自己的棋丢了类型 + 不看见敌方真实类型"。

### 2.2 通道投影（v4 + v5 layer-3）

Layer 1（v4）在敌方活子格上导出类型 multi-hot、≥k 计数、链式、rank floor、
工兵/地雷候选等（见上表）。

**v5 layer-3（ADR-129，已上线）** 在**同一敌方活子格**上追加：

- `cm_kill_mine_count[12]`：按类型 popcount（÷3 饱和归一化）；
- `cm_kill_mine_slot[30]`：observer 本地 slot 位图（哪一格布阵槽被该敌子吃过）；
- `cm_recency[4]`：直接/链式/floor 事件的新鲜度。

因此 **AI 现在可以识别**：「敌方棋子 K 当前在 (x,y)，曾吃过我的 slot-i /
类型 t 共 n 枚」——只要 victim 是 observer 自己的子（四暗合法信息）。

**仍存在的压缩/缺口**：

1. **slot ≠ 全局 pid 的跨视角命名**：slot 是 observer 布阵索引，不是
   120 维全局 piece_id one-hot 的反向索引表（但配合 `piece_id` 通道可
   在多数局面挂钩）。
2. **未投影 `eaten_by_pid` 反向表**（我活子格上「被哪些敌 pid 吃过」）：
   P0 设计稿中的 60+60 通道族尚未实现；v5 仅在**杀手格**上给 slot 信号。
3. **盟友被吃**：仍只有 `cm_kill_other_ge` 计数，无类型（DARK 正确）。
4. **链式**：`cm_chain_type` 仍 multi-hot，v5 未单独给 chain 的 slot 位图。

### 2.3 进一步的瑕疵

| 位置 | 问题 | 严重度 |
|---|---|---|
| `cm_kill_mine_type` | multi-hot 不带 count；K 吃 1 枚排长 vs 吃 2 枚排长在该通道上一致 | 中 |
| `cm_chain_type` | 同上；且与 cm_kill_mine_type 在 K 自己直接吃的位上重复（chain 包含 direct） | 低 |
| `cm_kill_other_ge` | 完全不区分类型，盟友损失只能数 | 中（暂无好办法） |
| `cm_dilei_candidate` | 仅看"未被已知工兵攻击过"，不看"敌方派工兵已经清雷的位置" | 中 |
| `cm_floor_ge` | 给 9 通道 cumulative one-hot，正确但稀疏 | 低 |
| v5 `cm_kill_mine_slot` | 杀手格上有 observer slot 位图；缺「我活子格←敌 pid」反向投影 | 中（P0 余量） |
| 缺失：threat / evasion / active-adjacency family | 完全没有，参考 Ataraxos Appendix C | 中-高 |
| 缺失：starting-square provenance | piece_id 已 one-hot 当前格，但没有"出生格" | 低 |
| 缺失：对手吃过的我**具体 pid** 投到我自家活子 | Layer 2 不够细 | 中 |

---

## 3. 训练现状：v17–v38

| 区段 | 主要改动 | 结果 |
|---|---|---|
| v17–v32（T4） | 1.03M JunqiNet, Ataraxos 对齐 | win_rate 卡 0.80；data scale 没解决 |
| v33–v34（H20） | DDP 6 卡自对弈, big move/belief net | OOM、NaN、collapse 反复 |
| v35（H20） | CombatMemory v4 上线（DARK-only） | 落地通过 961 测试，未做 long run |
| v36–v37（H20 单卡） | bugfixes + 1500R | best win 0.836@R50，end 0.5±0.2 |
| v38 (诊断) | reward_shaping=off + value-on-random-seats=on + adv_filt 0.2 | best win 0.766，仍未越过 0.80 |

诊断脚本（`scripts/diagnose_v37_collapse.py` / `diagnose_value_quality.py`）
已经定位到的关键失败模式是 **value head 退化为常数 + advantage 退化
为 reward-shaping 噪声 → 策略梯度跟着噪声打转**。v38 把 shaping 关掉、
adv_filt 收紧到 0.20、并打开 value 在所有座位的训练，这是正确方向但
还没等到充分长跑。

---

## 4. 此轮要做的优化（按 ROI 排序）

### 4.1 P0 — 提升观察通道的"对方吃我"可识别性

**已落地（v5，+46 ch，306→352）**：`cm_kill_mine_count` / `cm_kill_mine_slot` /
`cm_recency`。单测：`tests/test_observation_combat_memory_v5.py`。

**待做（可选下一 PR，+~88 ch → 394）**：

1. **`cm_kill_mine_count[12]`（12 通道）**
   将 multi-hot 升级到"K 吃我几枚 X 类"——单 cell 上每类 0/1/2/3 计数，
   归一化到 [0,1]（÷3）。直接从 `direct_ate_my_pid_lo/hi` & `_type_mask`
   推算（按 pid 计数+按 type 求 popcount，参考下面实现）。

2. **`cm_eaten_by_pid[60]`（60 通道）**
   投到 SOUTH 视角下的 60 个**敌方 piece_id**：
   `cm_eaten_by_pid[k][y][x] = 1` 当且仅当 SOUTH 自家某活子在 (x,y) 处
   且它**已经被** SOUTH 视角下编号为 k 的敌方 piece_id 吃过。
   这恰好填上了"我哪只 pid 被哪只敌 pid 吃了"。
   注：60 = 30(WEST) + 30(EAST) — 队友与自己不算。

   实施细节：
   - 我方活子 mpid 上读取 `chain_pid_lo/hi[obs, killer_pid]` 是否包含 mpid 不行
     ——chain_pid 是 killer 视角的 victim 集合；我们要的是反向。
   - **新增** `eaten_by_pid_lo/hi[obs, mpid]`（uint64×2）：每个我方
     pid mpid 维护"哪些敌方 pid 吃过我（含链式）"——SO 8 (4×120×16) bytes
     = 7.7 KB / state，可忽略。
   - 通道仅按 SOUTH 视角下 30 (WEST) + 30 (EAST) 的敌方 pid 投影；DARK 下
     "盟友被吃" 不暴露给我（信息边界）。

3. **`cm_kill_mine_pid[60]`（60 通道）**
   反向同时也加 K 视角的 60 通道：在敌方活子 K 上，标记"K 已经吃过
   我哪些 pid"。这是直接读 `direct_ate_my_pid_lo/hi[obs, K]` 与
   `chain_pid_lo/hi[obs, K] & my_pid_mask`。**这才让模型能跟踪
   "排长 #20 被 K 吃了"**。

4. **`cm_recency[6]`（6 通道）**
   3 维 fresh / 3 维 stale 的 sigmoid(time_since_event/τ)，对应：
   - direct_ate_my：`last_direct_step`
   - chain_my：`last_chain_step`
   - rank_floor 升级：`rank_floor_step`

   时间窗 τ 取 32（rolling）和 256（long-term），共 6 通道。

> 备注：方案 2 与 3 看似冗余，但 (我视角) 与 (敌视角) 是不同的"锚点"，
> CNN 不能从一个推出另一个；显式两份对网络更友好。Ataraxos 的
> threat/evasion/protection 系列也是双向各跑一份。

#### 数据结构改动

`combat_memory.py::CombatMemoryState` 新增字段
`eaten_by_pid_lo/hi : (4, 120) uint64×2`，并在 `apply_combat_event`
里两端同步更新（killer 已有；victim_pid 也要把 killer 写进
`eaten_by_pid` 在所有 observer 对应的 victim 上）。

> ⚠️ DARK 信息边界
>
> `eaten_by_pid` 的 observer 维度是关键：对每个 observer，仅当
> `victim_seat == observer`（"我看到自己丢了棋"）写入，否则不可见。
> Theory-of-Mind 投影不暴露 mpid 的"吃我者 pid 集合"。

### 4.2 P1 — 修 v37 value collapse（与 P0 同步落地）

v38 的三连击已经是对症的；本轮额外把以下做实：

1. **打开 self-play 的"主对手 EMA"** — 现在 random_opponent=True 时
   对家直接走 uniform；目标 90% vs random 时，前 200 rollouts 用
   ema_decay 0.999 的 self-play warmup 拉一拉 V，再切回 vs-random
   evaluation（**保留 random_opponent=True 评估，仅训练时打开 self-play
   对手池**）。
2. **value loss 上加 BCE-shape 正则**，监控 V.std/V.corr ≥ 0.005，
   一旦 V.std<1e-3 立即打 warning + 强制把 lr ×0.5。

### 4.3 P2 — 训练配置 (`configs/v40_dark_pid_features.yaml`)

继承 v38 的参数，主要调整：

```yaml
env:
  num_envs: 384            # H20 单卡可承受新 obs 增长 (306→394 = +28%)
  steps_per_env: 512
ppo:
  reward_shaping: false
  num_epochs_per_rollout: 2   # Ataraxos 1，T4 配置 4，新中点
  adv_filt_rate: 0.25         # 比 v38 略放宽，新通道带来更稳的 V → 更稳的 |adv|
random_opponent: true
train_value_on_random_seats: true
```

DDP 双卡 (`scripts/launch_h20_ddp.sh`)：
- ranks=2，每卡 384 envs，相同种子族但 rank-offset 10000 保证多样性；
- 评估 in-place 每 25 R 一次 vs-random。

### 4.4 P3 — 评估卷尺

新增 `scripts/eval_random_dense.py`：每个 ckpt 运行 2048 局 vs-random，
按 confidence interval（Wilson 95%）报告。这是判定"是否真的过 90%"的
唯一标准。

---

## 5. 工作计划（本轮 PR）

| # | 任务 | 状态 |
|---|---|---|
| 1 | 阅读 + 总结现状（本文） | ✅ |
| 2 | 修 build_cuda.py nvcc 路径，跑通 978 测试 | ✅ |
| 3 | P0 v5 layer-3（46 ch）+ GPU parity | ✅ `a9d25f1` |
| 4 | v40/v42/v43 配置 + `eval_random_dense.py` | ✅ v43 + eval 脚本 |
| 5 | BeliefNet DDP NaN 全 rank 同步 skip | ✅ `belief_ppo.py` |
| 6 | `eaten_by_pid` 反向投影（layer 4） | ✅ `0c9ecca` (v6) |
| 7 | v44 配置（v6 + Plan B + Belief） | ✅ `configs/v44_planB_v6_eaten_by_pid.yaml` |
| 8 | CUDA kernel 同步 v6（layer-4 写入） | 待办（CPU only 暂可训） |
| 9 | 长跑至 Wilson95 下界 ≥ 0.90 | 待 H20 训练 |
| 7 | 长跑至 Wilson95 下界 ≥ 0.90 | 待 H20 训练 |

---

## 6. 风险与开放问题

- **新增通道带来 obs slab 增长 ~30%**，rollout buffer 显存压力上升；
  GPU 端必须更新 `junqi_cuda::observation.cu` 或保持 CPU obs path。
  本次先以 CPU 路径落地 + GPU 路径加 TODO；GPU kernel 更新留下一 PR。
- **CombatMemory v4 CUDA kernel 仍未实装新增字段** — 现有 `state.cu`
  对 CombatMemory 的更新逻辑完整，但一旦新增 `eaten_by_pid` 字段，
  CPU/GPU parity 测试需要同步扩展。
- **bf16 + autocast 在新通道下偶发 logit overflow**：已经有 NaN guard，
  但若新通道激活值范围更大，建议把 `cm_kill_mine_count` 显式 ÷3 归一。
- **真"90% vs random"是否就是终点？** Ataraxos 类似设定下基线 ~95%，
  但他们 random opponent 不太能形成稳定 stall 局面；JunQi 的随机走可能
  会撞炸弹（无差别）导致天花板 < 100%。预期值取 0.92 ± 0.02 较真实。

# Phase 0.3 T7 — Ataraxos-parity TODO

> **目标**：在 `junqi_core` 观测层引入与 Ataraxos 严格对齐的 **piece_id 全局身份系统**，并以"Ataraxos 风格"落地新增的行为/死亡通道（A/B/C/D/E 共 32 个新通道，合计 `69 + 32 = 101` 通道）。
>
> **路线**：**路线 A（严格对齐 Ataraxos）**
> - A/B/C 组（行走 / 主动吃 / 被动幸存）只遍历当前棋盘、只写活子；死子在这些通道上信号消失。
> - D 组（死因 × death-loc） + E 组（dead_at_zero × zero-pos）承担死子全部信号，D/E 需要 piece_id 才能正确定位。
>
> **参照源码**：
> - `ataraxos/src/env/stratego.h`（`NUM_BOARD_STATE_CHANNELS`、通道注释）
> - `ataraxos/src/env/cuda/infostate_kernels.cu`（`BoardStateKernel__InvisiblesEmptyAndMoved`、`ThreatEvadeActiveAdj`、`Deaths`、`DeathReasons`）
> - `ataraxos/src/env/stratego_board.h`（`Piece{piece_id, has_moved, ...}`、`StrategoBoard`、`zero_boards`、`deaths` bitmap）

---

## 0. 版本与索引

| 项目 | 值 |
|---|---|
| Phase | 0.3 |
| Task | T7（观测扩展 + 全局身份系统） |
| Depends on | T5 规则验证已完成、T6 Seat/rotation 稳定（ADR-111） |
| Produces | 101 通道观测张量、`PieceRef.piece_id`、`GameState.zero_board`、`GameState.deaths` bitmap |
| Ownership | observation.py / state.py / move_gen.py / setup.py / info_model.py / tests/golden/ |
| ADR 预占编号 | **ADR-114**（piece_id 全局身份系统）、**ADR-115**（T7 新增 32 通道，严格对齐 Ataraxos）、**ADR-116**（观测通道总数 69 → 101） |

---

## 1. 设计快照（对齐 Ataraxos）

### 1.1 身份系统：`piece_id ∈ [0, 119]`

| Ataraxos | JunQi（本项目） |
|---|---|
| 40 颗子，`piece_id ∈ [0, 39]` | **120 颗子（4 家 × 每家 25 活子；军旗/地雷也编号，但不参与行走）**，`piece_id ∈ [0, 119]` |
| 约定 `piece_id == zero-board 格子索引` | 约定 `piece_id = seat.value * 30 + setup_slot`，其中 `setup_slot ∈ [0, 29]`（营地格在 setup 里为 NONE，该 slot 的 piece_id 不分配/置为 `0xff`） |
| 身份不变：从出生到死亡 `piece_id` 固定 | 同左 |

**原则**：一旦棋子落入初始布阵（`new_game`），`piece_id` 立即分配、终生不变；死亡只改 `deaths` bitmap 与 `alive`，`piece_id` 不回收。

### 1.2 双棋盘结构：`board` + `zero_board`

| 字段 | 语义 | 写入时机 |
|---|---|---|
| `GameState.pieces`（已存在）| 当前棋盘：`(x,y) -> PieceRef`，只含活子 | 每步 `step()` 更新 |
| `GameState.zero_board`（**新增**）| 初始棋盘快照：`(x,y) -> PieceRef`（含 `piece_id`）| 仅 `new_game` 时一次性写入，**永不更改** |
| `GameState.deaths`（**新增**，bitmap）| `Dict[Seat, bytearray(ceil(30/8))]` 或 `dict[piece_id -> DeathInfo]` | 每次有子死亡时置位 |
| `GameState.death_info`（**新增**）| `dict[piece_id -> DeathInfo{reason, death_loc, step}]` | 每次有子死亡时写入 |

其中 `DeathInfo.reason ∈ {KILLED_BY_ENEMY, HIT_MINE_OR_BOMB, MUTUAL}`，`death_loc` 为战场世界坐标（死亡发生时的 dst 格）。

### 1.3 通道布局（101 通道）

复用现有 69 通道不变，在尾部追加 32 通道：

| 通道组 | 数量 | 锚点 | 语义 |
|---|---|---|---|
| 现有 `piece_own..turn_history` | 69 | — | 保持 ADR-106 不动 |
| **A. `move_count_bucket_ours/theirs`** | 4 × 2 = 8 | 活子当前格 | 走了 0 / 1 / 2 / ≥3 步 |
| **B. `active_eat_count_bucket_ours/theirs`** | 4 × 2 = 8 | 活子当前格 | 主动吃了 0 / ≥1 / ≥2 / ≥3 个 |
| **C. `passive_survive_bucket_ours/theirs`** | 4 × 2 = 8 | 活子当前格 | 被动幸存 0 / ≥1 / ≥2 / ≥3 次 |
| **D. `death_reason_ours/theirs`** | 3 × 2 = 6 | **death-loc** | killed_by_enemy / hit_mine_or_bomb / mutual |
| **E. `dead_at_zero_ours/theirs`** | 1 × 2 = 2 | **zero-pos** | 1 = 该格初始子已死 |
| **新增合计** | **32** | | |
| **总计** | **101** | | |

### 1.4 bucket 语义定义（与 Ataraxos 原则对齐）

| bucket | move_count | active_eat_count | passive_survive |
|---|---|---|---|
| ch_0 | `== 0` 精确 | `== 0` 精确 | `== 0` 精确 |
| ch_1 | `== 1` 精确 | `≥ 1` | `≥ 1` |
| ch_2 | `== 2` 精确 | `≥ 2` | `≥ 2` |
| ch_3 | `≥ 3` | `≥ 3` | `≥ 3` |

**注意**：移动步数用"精确相等 + 末档 ≥"（因为走很多步和走很少步方向相反）；吃子/幸存用"全部 ≥"累加式（历史已定，见前序讨论）。
**敌方的这些通道只遍历可见的活子**（与 Ataraxos `piece.visible` 判定同构）——不可见子不向敌方视角泄露其 bucket。

---

## 2. 代码改动清单（按文件）

> ✅ = 新增、🛠 = 修改、🧪 = 新增测试

### 2.1 `junqi_core/state.py` 🛠

- [ ] **扩展 `PieceRef`**（或新建 `dataclass(frozen=False) PieceTracker` 作为可变补充，见 §3 决策点）：
  - `piece_id: int`（0..119）
  - `move_count: int`
  - `active_eat_count: int`
  - `passive_survive_count: int`
  - `alive` 保留
- [ ] **`GameState` 新增字段**：
  - `zero_board: dict[tuple[int,int], PieceRef]`（只读快照）
  - `deaths: dict[int, DeathInfo]`（key = piece_id）
- [ ] **新增 `DeathInfo`**：
  ```python
  @dataclass(frozen=True, slots=True)
  class DeathInfo:
      piece_id: int
      reason: DeathReason  # KILLED_BY_ENEMY / HIT_MINE_OR_BOMB / MUTUAL
      death_loc: tuple[int, int]
      step: int
  ```
- [ ] **`DeathReason` 枚举**（放在 `rules.py` 或 `state.py`）：3 个值。
- [ ] **`GameState.clone()`** 深拷贝 `zero_board`（浅拷贝 ok，永不修改）+ `deaths` dict。
- [ ] **`new_game` 写入 piece_id + zero_board**（见 §2.4）。
- [ ] **`step()` 维护计数器与 `deaths`**：
  - 成功移动（`Event.MOVE`）：攻击方 `move_count += 1`。
  - 战斗吃子（`src_piece` 存活 + `dst_piece` 死亡）：`src_piece.active_eat_count += 1`；`dst_piece` 记录 `DeathReason.KILLED_BY_ENEMY`（若 dst 是地雷/炸弹死因记为 `HIT_MINE_OR_BOMB`）。
  - 战斗守子（`dst_piece` 存活 + `src_piece` 死亡）：`dst_piece.passive_survive_count += 1`；`src_piece` 记录对应死因。
  - 同归（`MUTUAL`）：两子都记 `DeathReason.MUTUAL`（ADR-016 撞雷/炸弹时也归 HIT_MINE_OR_BOMB，具体映射见 §3 决策点 D-2）。
  - **死子不再累加任何计数**（与 §讨论一致）。
- [ ] **可变性取舍**：若保留 `PieceRef` frozen，需在每次 `step()` 构造新的 `PieceRef`（参考 `dataclasses.replace`）；若改为 mutable 需审视其它调用点（move_gen 的缓存）。

### 2.2 `junqi_core/setup.py` 🛠

- [ ] **初始 piece_id 分配**：`new_game` 阶段遍历每个 seat 的 30-slot lineup，对 `piece_type != NONE` 的槽位按 `piece_id = seat.value * 30 + slot_index` 赋值，营地槽位标 `0xff`/跳过。
- [ ] 输出 `(pieces, zero_board)` 元组供 `GameState.new_game` 使用。

### 2.3 `junqi_core/rules.py` 🛠（小改）

- [ ] 新增 `DeathReason` 枚举 + 3 个常量。
- [ ] 新增工具函数 `classify_death_reason(attacker_type, defender_type, event) -> DeathReason`。

### 2.4 `junqi_core/move_gen.py` 🛠（兼容性）

- [ ] `PieceMap` 值类型升级：下游只读 `seat / piece_type / alive`，新字段不影响 move_gen。**验证 hash 不变**：`PieceRef` 若仍 frozen，新增字段进入 hash 会失效现有缓存 → 需把新增计数字段移出 `PieceRef`，放入独立的 `PieceState[piece_id]` 字典。详见 §3 决策点 D-1。

### 2.5 `junqi_core/observation.py` 🛠（主改造）

- [ ] **新增 5 个通道组常量**：
  ```python
  _CH_MOVE_BUCKET_SIZE: Final[int] = 4 * 2
  _CH_ACTIVE_EAT_BUCKET_SIZE: Final[int] = 4 * 2
  _CH_PASSIVE_SURVIVE_BUCKET_SIZE: Final[int] = 4 * 2
  _CH_DEATH_REASON_SIZE: Final[int] = 3 * 2
  _CH_DEAD_AT_ZERO_SIZE: Final[int] = 1 * 2
  ```
- [ ] **追加到 `CHANNEL_LAYOUT`** 尾部：`move_bucket`, `active_eat_bucket`, `passive_survive_bucket`, `death_reason`, `dead_at_zero`。
- [ ] **更新 `OBS_CHANNELS` 断言**：`assert OBS_CHANNELS == 101`。
- [ ] **新增 5 个 writer 函数**：
  - `_move_bucket_channels(state, observer)` — 遍历 `state.pieces`（活子）按当前格写 bucket。
  - `_active_eat_bucket_channels(state, observer)` — 同上。
  - `_passive_survive_bucket_channels(state, observer)` — 同上。
  - `_death_reason_channels(state, observer)` — 遍历 `state.deaths` 按 `death_loc` 写。
  - `_dead_at_zero_channels(state, observer)` — 遍历 `state.zero_board`，若对应 `piece_id ∈ state.deaths` 则在 zero_board 格上置 1。
- [ ] **敌方可见性过滤**：与现有 `piece_left_side_enemy` 通道一致（只对已揭示/可见的敌子写入；未揭示一律 0）。
- [ ] **坐标变换**：统一走 `observer` 的 `rotate_to_pov`（与现有通道同构），避免重复实现。

### 2.6 `junqi_core/info_model.py` 🛠（评估影响）

- [ ] 验证 `purge_dead_pieces` 不再真的删除 `zero_board`、`deaths` 与 belief 之间的一致性。
- [ ] 若 info_model 依赖 `pieces` 迭代，确认 piece_id 不泄露给敌方（与 ADR-002 战斗广播规则一致）。

### 2.7 `junqi_core/__init__.py` 🛠

- [ ] 导出 `DeathReason`、`DeathInfo`、`OBS_CHANNELS=101`、`CHANNEL_LAYOUT`。

---

## 3. 决策点（✅ 已于 2026-04-21 全部定稿）

| ID | 问题 | 定稿 | 落地方式 |
|---|---|---|---|
| **D-1** | 计数器放在哪里？ | **方案 (b)**：新增 `PieceState: dict[piece_id -> {move_count, active_eat_count, passive_survive_count}]`，与 `PieceRef` 解耦 | `PieceRef` 保持 frozen，不破坏 move_gen 缓存；`GameState` 新增 `piece_state` 字段 |
| **D-2** | 同归的死因归类 | **方案 (a)**：无论是否涉及地雷/炸弹，同归一律记 `MUTUAL` | `DeathReason` 枚举只 3 值：`KILLED_BY_ENEMY` / `HIT_MINE_OR_BOMB` / `MUTUAL`；纯净三分类 |
| **D-3** | 敌方视角 bucket 通道是否对"不可见活子"置 0？ | **方案 (a)**：是，对齐 Ataraxos `piece.visible` | observation 的 A/B/C 组 writer 对 `observer` 视角下不可见的敌子一律置 0 |
| **D-4** | E 组（dead_at_zero）对敌方 zero-pos 是否写？ | **写**（zero-pos 是死子历史锚点，与当前兵种身份无关，不构成身份泄露） | `_dead_at_zero_channels` 我方/敌方两个通道均正常写 |
| **D-5** | piece_id 在 observation 张量中是否直接暴露？ | **方案 (a)**：不暴露 | `piece_id` 只作 `PieceState / deaths / death_info` 的 dict key，张量通道不含 piece_id 维度 |

**后续动作**：
- 以上定稿同步写入 ADR-114（piece_id 身份系统）、ADR-115（T7 新增 32 通道）、ADR-116（通道总数 69 → 101）。
- 立即进入 M1 实施阶段。

---

## 4. 实施步骤（阶段化里程碑）

### M1 — 身份系统落地（预计 0.5 天）
- [ ] state.py：`DeathInfo / DeathReason` 定义。
- [ ] state.py / setup.py：`piece_id` 分配 + `zero_board` 构造。
- [ ] state.py：`GameState.deaths` / `death_info` 字段 + `clone()` 支持。
- [ ] 单元测试：`test_piece_id_assignment.py`（断言 piece_id 唯一、zero_board 不可变、`new_game` 后稳定）。

### M2 — 计数器维护（预计 0.5 天）
- [ ] 按 D-1 决定的结构，把 `move_count / active_eat_count / passive_survive_count` 纳入 `step()`。
- [ ] 单元测试：`test_piece_counters.py`（用 golden scenario 跑 ≥1 局，逐步断言计数值）。

### M3 — 死亡记录（预计 0.5 天）
- [ ] `step()` 识别死因并写 `DeathInfo`。
- [ ] 单元测试：`test_death_info.py`（覆盖 KILLED/MINE/MUTUAL 三种死因 × 复现 `tests/golden/battle/` 全部 9 个战斗 golden）。

### M4 — 观测通道扩展（预计 1 天）
- [ ] observation.py：5 组 writer + layout 扩展 + 101 断言。
- [ ] 更新 `tests/golden/SCHEMA.md` 中 observation channel count。
- [ ] 金标回归：`pytest tests/test_golden_replay.py`（若存 observation 快照需重录）。
- [ ] 新增 `tests/test_observation_t7.py`：
  - 通道形状断言 `(101, 17, 17)`。
  - bucket 单位测试：构造 3 步移动 → `move_bucket[3]` 在当前格为 1。
  - death_reason：构造撞雷 → `death_reason[HIT_MINE_OR_BOMB]` 在战场格为 1。
  - dead_at_zero：构造一次吃子 → 死子 zero-pos 对应通道为 1。
  - 活/死共存：死子的 A/B/C 通道**全 0**（路线 A 的核心不变量）。

### M5 — 文档 + ADR（预计 0.5 天）
- [ ] 在 `docs/DECISIONS.md` 落 **ADR-114 / ADR-115 / ADR-116**。
- [ ] 在 `docs/ARCHITECTURE.md` 更新观测张量维度。
- [ ] 在 `docs/PHASE_0.2_ACCEPTANCE.md` 或新建 `docs/PHASE_0.3_ACCEPTANCE.md` 追加 T7 验收清单。
- [ ] 在 `docs/LEGACY_PARITY.md` 备注：legacy C 引擎不含 piece_id，回放兼容层（若将来重启）需读取 `zero_board` 占位。

### M6 — 性能与一致性体检（预计 0.5 天）
- [ ] `step()` 性能：100 步 × 1000 局 < 5s（参考现有基线）。
- [ ] observation build：单次 < 2ms（101 通道扩展不退化现有基线 20% 以上）。
- [ ] `info_model` 信念更新与 `deaths` 字典交叉一致（死子不再出现在信念分布中）。

**总预估**：**3.5 工作日**。

---

## 5. 风险与兜底

| 风险 | 兜底策略 |
|---|---|
| `PieceRef` frozen 改动引发 move_gen 缓存击穿 | 采用 D-1(b) 方案：把计数独立到 `PieceState` 字典，`PieceRef` 不变 |
| golden 快照大规模重录 | M4 前先跑 `tests/test_golden_replay.py` 看哪些 golden 含 observation；仅对这些重录并 code-review 变更 |
| 敌方 bucket 通道误泄可见性 | M4 单测 `test_enemy_invisible_piece_zero_buckets`：构造一颗未揭示敌子，3 步移动后，敌方视角 bucket 全 0 |
| piece_id 编号规则与将来棋子扩展不兼容 | 保留 `0xff` 表示无效 id；若未来更改编号规则，只改 setup.py 的单一分配函数 |
| zero_board 体积过大拖慢 `clone()` | `zero_board` 永不变更 → `clone()` 直接共享引用 |

---

## 6. 验收清单（Done 标准）

- [ ] `OBS_CHANNELS == 101` 断言通过，`CHANNEL_LAYOUT` 命名完整。
- [ ] `tests/test_observation_t7.py` 全绿（≥ 6 个用例）。
- [ ] `tests/test_piece_id_assignment.py` / `test_piece_counters.py` / `test_death_info.py` 全绿。
- [ ] `pytest` 全库通过（含现有 golden）。
- [ ] `DECISIONS.md` 收录 ADR-114/115/116。
- [ ] `ARCHITECTURE.md` 观测维度同步。
- [ ] `docs/PHASE_0.3_ACCEPTANCE.md` T7 章节勾选完成。

---

## 7. 与 Ataraxos 源码的对照表（自查用）

| Ataraxos 概念 | 源码位置 | JunQi 对应物 |
|---|---|---|
| `Piece.piece_id` | `stratego_board.h` | `PieceRef.piece_id`（或 `PieceState` key） |
| `StrategoBoard` | `stratego_board.h` | `GameState.pieces` |
| `zero_boards` | `env_state.h:38` | `GameState.zero_board` |
| `deaths[2][5]` bitmap | `stratego_board.h` | `GameState.deaths: dict[piece_id -> DeathInfo]`（JunQi 不用 bitmap，用 dict 更直观） |
| `Ch.39 has_moved` | `infostate_kernels.cu:408` | 合并在 A 组 `move_bucket`（我们用 4 档替代单 bit） |
| `Ch.109-130 dead by type (zero-pos)` | `infostate_kernels.cu:447-479` | **E 组** `dead_at_zero_{ours,theirs}`（军棋简化为 2 通道，不按 11 个兵种细分） |
| `Ch.131-250 death reason (death-loc)` | `infostate_kernels.cu:505-543` | **D 组** `death_reason_{ours,theirs} × 3` 档死因 |

---

## 8. 未决后续（**不在本 T7 范围**）

- 是否引入 Ataraxos 的 `threatened / evaded / actively_adjacent` 三组兵种位图（当前 JunQi 信念层 `info_model.py` 已覆盖大部分语义，暂不直接移植）。
- 是否引入 `protected / was_protected_by` 四组（Ataraxos Ch.251-354）— 等 Phase 0.4 再评估。
- piece_id one-hot embedding 作为 transformer token 输入 — 保留到 Phase 1 RL 架构设计时讨论。

---

_Last updated: 2026-04-21_

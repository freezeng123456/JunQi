# CombatMemory 设计：基于公开战斗事件的高阶逻辑记忆

状态：设计稿 v1.0  
目标：为 `MoveNet` / `BeliefNet` 提供精确、可解释、非作弊的战斗历史记忆，以弥补纯当前局面 observation 无法回忆多步前战斗事件的问题。

---

## 1. 背景与动机

当前 `JunQi` observation 已包含：

- `piece_id`：120 通道，追踪每个棋子的公开身份；
- `move_history`：最近 32 步源/目标位置；
- `active_eat_bucket` / `passive_surv_bucket`：吃子/被攻击后存活的计数 bucket；
- `death_reason` / `death_loc`：死亡原因与死亡位置；
- `belief_*`：对敌方暗子的类型概率。

这些特征仍不足以表达如下历史逻辑：

```text
我的排长 A 被右边未知棋 B 吃；
之后 B 又被对家未知棋 C 吃；
那么从我的视角：
  B 至少是连长；
  C 至少是营长；
  C 与“吃过我排长的 B”存在传递战斗关系。
```

`MoveNet` 是 feed-forward 网络，每一步只看当前 observation，不具备跨步记忆。若 observation 不显式包含上述历史事实，模型需要从有限的 `move_history` 中自行反推，样本效率低且超过 32 步会丢失。因此需要由环境维护一套持久的、每个 observer 私有的战斗记忆，并在每一步投影到当前棋盘上。

---

## 2. 设计目标

CombatMemory 需要支持：

1. **精确直接记忆**：当前敌子直接吃过我的哪些具体 `piece_id` 和类型；
2. **高阶链式记忆**：当前敌子是否吃掉过“曾经吃过我的棋”的棋，并继承其关联；
3. **普通军阶强度下界**：根据公开战斗结果推断某暗子至少多强；
4. **类型排除**：根据战斗/位置规则排除不可能类型，如“非后两排吃子不可能是工兵”；
5. **特殊类型候选**：记录地雷、炸弹、工兵等特殊事件嫌疑；
6. **时间新鲜度**：记录直接/链式/floor/special 记忆最近更新时间；
7. **非作弊信息边界**：只使用当前 observer 合法知道的信息，不泄露敌方真实类型或不可见 teammate 私有信息。

---

## 3. 信息边界

### 3.1 可以进入当前 observer observation 的信息

对 observer 自己而言，以下是合法公开/私有信息：

- 我的棋被谁吃了；
- 我的棋是什么类型；
- 吃我棋的敌方 `piece_id` 是谁；
- 后续谁吃掉了这个敌方 `piece_id`；
- 根据上述事件可逻辑推出的强度下界、类型排除、特殊嫌疑。

### 3.2 不应直接进入 observation 的信息

不应直接给模型：

- 敌方棋子的真实类型；
- 敌方被吃棋子的真实类型，除非该类型已由规则公开；
- teammate 的私有棋子类型，如果规则设定下 teammate 不可见。

因此所有基于 `victim_type` 的强推断只在：

```text
victim_seat == observer
```

时生效。若未来规则允许 teammate 信息共享，可单独扩展。

---

## 4. 核心状态结构

CombatMemory 按 `env × observer × piece_id` 存储。这里 `observer ∈ {0,1,2,3}`，`piece_id ∈ [0,119]`。

### 4.1 精确直接记忆

```cpp
uint64_t direct_ate_my_pid_lo[env, observer, pid];  // bits 0..63
uint64_t direct_ate_my_pid_hi[env, observer, pid];  // bits 64..119
uint16_t direct_ate_my_type_mask[env, observer, pid];
int16_t  last_direct_step[env, observer, pid];
```

含义：`pid` 这个棋直接吃过 `observer` 自己的哪些 `piece_id` / 类型。

例：B 吃了我的排长 A：

```text
direct_ate_my_pid[B].set(A)
direct_ate_my_type_mask[B].set(PAIZH)
last_direct_step[B] = step
```

### 4.2 高阶链式记忆

```cpp
uint64_t chain_ate_my_pid_lo[env, observer, pid];
uint64_t chain_ate_my_pid_hi[env, observer, pid];
uint16_t chain_ate_my_type_mask[env, observer, pid];
int16_t  last_chain_step[env, observer, pid];
```

含义：`pid` 这个棋吃掉过某个“直接或间接吃过 observer 棋子”的棋，并继承其与 observer 棋子的关联。

例：B 吃我的排长 A，C 又吃 B：

```text
chain_ate_my_pid[C].set(A)
chain_ate_my_type_mask[C].set(PAIZH)
```

如果 D 再吃 C，则继续继承：

```text
chain_ate_my_pid[D].set(A)
chain_ate_my_type_mask[D].set(PAIZH)
```

### 4.3 普通军阶强度下界

```cpp
int8_t  rank_floor[env, observer, pid];
int16_t rank_floor_step[env, observer, pid];
```

`rank_floor` 定义：

```text
0 = unknown
1 = GONGB+   工兵+
2 = PAIZH+   排长+
3 = LIANZH+  连长+
4 = YINGZH+  营长+
5 = TUANZH+  团长+
6 = LVZH+    旅长+
7 = SHIZH+   师长+
8 = JUNZH+   军长+
9 = SILING+  司令+
```

`JUNQI`、`DILEI`、`ZHADAN` 不进入普通 rank floor，走特殊候选/排除逻辑。

### 4.4 类型排除

```cpp
uint16_t exclude_type_mask[env, observer, pid];
```

每个 bit 对应一个 tracked type，例如：

```text
exclude_type_mask[observer][pid].has(GONGB)
```

表示该 observer 可推断 `pid` 不可能是工兵。

### 4.5 类型候选与特殊提示

```cpp
uint16_t candidate_type_mask[env, observer, pid];
uint8_t  special_hint_mask[env, observer, pid];
int16_t  special_hint_step[env, observer, pid];
```

`candidate_type_mask` 表示强候选类型，如可能是地雷、炸弹、工兵。

`special_hint_mask` 建议定义：

```text
bit0 = MINE_SUSPECT          // 可能是地雷
bit1 = BOMB_SUSPECT          // 可能是炸弹
bit2 = MINE_CLEAR_SUSPECT    // 曾经处理/吃掉地雷，可能是工兵
bit3 = EQUAL_TRADE_SEEN      // 发生过同归/互兑
bit4 = FLAG_CAPTURE_SEEN     // 捕获过军旗
```

---

## 5. 普通军阶与辅助函数

### 5.1 普通军阶顺序

从弱到强：

```text
GONGB < PAIZH < LIANZH < YINGZH < TUANZH < LVZH < SHIZH < JUNZH < SILING
```

特殊棋：

```text
JUNQI, DILEI, ZHADAN
```

不参与普通 rank floor。

### 5.2 next_stronger

```text
next_stronger(GONGB)  = PAIZH+
next_stronger(PAIZH)  = LIANZH+
next_stronger(LIANZH) = YINGZH+
next_stronger(YINGZH) = TUANZH+
next_stronger(TUANZH) = LVZH+
next_stronger(LVZH)   = SHIZH+
next_stronger(SHIZH)  = JUNZH+
next_stronger(JUNZH)  = SILING+
next_stronger(SILING) = special/unknown
```

如果某棋吃了司令，不能直接推断出普通军阶更强；通常应记录炸弹/特殊嫌疑，而不是更新普通 floor。

---

## 6. 事件归一化

每次战斗先归一化成：

```cpp
struct CombatEvent {
    int killer_pid;       // 单方面胜者，仍存活
    int victim_pid;       // 死者
    int killer_seat;
    int victim_seat;
    int killer_type;      // 引擎真值，仅内部使用，不直接泄露
    int victim_type;      // 仅 victim_seat == observer 时可用于 observer
    int victim_pos_flat;
    int event;            // EAT / KILLED / BOMB / MUTUAL
    int death_reason;
    int step;
    bool killer_survived;
    bool victim_died;
    bool both_died;
};
```

对 observation 的更新必须按 observer 权限过滤。

---

## 7. 更新规则

### 7.1 单方面吃子

统一处理两种情况：

```text
EAT:    attacker 吃 defender，killer = attacker, victim = defender
KILLED: defender 吃 attacker，killer = defender, victim = attacker
```

若：

```text
victim_seat == observer
victim_died == true
killer_survived == true
```

更新直接记忆：

```cpp
set_bit(direct_ate_my_pid[killer], victim_pid);
direct_ate_my_type_mask[killer] |= bit(victim_type);
last_direct_step[killer] = step;
```

### 7.2 普通军阶强度下界

若 `victim_type` 是普通军阶，且不是特殊 death reason：

```cpp
rank_floor[killer] = max(rank_floor[killer], next_stronger(victim_type));
rank_floor_step[killer] = step;
exclude_type_mask[killer] |= all_types_weaker_or_equal(victim_type);
```

例：

```text
吃排长 → 至少连长；排除工兵/排长
吃师长 → 至少军长；排除师长及以下普通子
```

### 7.3 工兵排除

若 killer 单方面吃了 observer 的非地雷棋：

```cpp
if (victim_type != DILEI) {
    exclude_type_mask[killer] |= bit(GONGB);
}
```

因为工兵不能靠普通军阶吃掉非地雷棋。

### 7.4 工兵候选 / 处理地雷

若 killer 吃了 observer 的地雷，且地雷位置在后两排：

```cpp
candidate_type_mask[killer] |= bit(GONGB);
special_hint_mask[killer] |= MINE_CLEAR_SUSPECT;
special_hint_step[killer] = step;
```

此时不更新普通 rank floor，也不排除工兵。

### 7.5 撞地雷/撞炸弹嫌疑

若 observer 的棋攻击敌方 X 后死，且 death reason 为 `HIT_MINE_OR_BOMB`：

- 若 X 存活且在后两排：

```cpp
candidate_type_mask[X] |= bit(DILEI);
special_hint_mask[X] |= MINE_SUSPECT;
```

- 若双方同死或事件指向 bomb/mutual：

```cpp
candidate_type_mask[X] |= bit(ZHADAN);
special_hint_mask[X] |= BOMB_SUSPECT;
```

不要更新普通 rank floor。

### 7.6 位置合法性排除

每次写 observation 时可动态加入：

```text
当前位置不在后两排 → exclude DILEI
当前位置在前排     → exclude ZHADAN
当前位置不在大本营 → exclude JUNQI
```

这些可以不写入状态，而在 observation kernel 内即时计算。

---

## 8. 高阶链式传播

当 killer 杀死 victim 时，对每个 observer：

```cpp
chain_ate_my_pid[killer]  |= direct_ate_my_pid[victim];
chain_ate_my_pid[killer]  |= chain_ate_my_pid[victim];
chain_ate_my_type[killer] |= direct_ate_my_type[victim];
chain_ate_my_type[killer] |= chain_ate_my_type[victim];

if (victim had any direct/chain memory) {
    last_chain_step[killer] = step;
}
```

强度下界传播：

```cpp
if (rank_floor[victim] > UNKNOWN) {
    rank_floor[killer] = max(rank_floor[killer], next_floor(rank_floor[victim]));
    rank_floor_step[killer] = step;
    exclude_type_mask[killer] |= weaker_or_equal_from_floor(rank_floor[victim]);
    exclude_type_mask[killer] |= bit(GONGB);
}
```

---

## 9. 示例推演

### 9.1 排长被吃，然后吃子者又被吃

```text
A = 我的排长
B = 右边未知棋
C = 对家未知棋

B 吃 A
C 吃 B
```

第一步：

```text
direct_ate_my_pid[B] = {A}
direct_ate_my_type_mask[B] = {PAIZH}
rank_floor[B] = LIANZH+
exclude_type_mask[B] includes GONGB, PAIZH
```

第二步：

```text
chain_ate_my_pid[C] = {A}
chain_ate_my_type_mask[C] = {PAIZH}
rank_floor[C] = YINGZH+
exclude_type_mask[C] includes GONGB, PAIZH, LIANZH
```

当前 C 所在格 observation 写：

```text
chain_ate_my_piece_id_A = 1
chain_ate_my_type_PAIZH = 1
floor_ge_paizh = 1
floor_ge_lianzh = 1
floor_ge_yingzh = 1
exclude_gongb = 1
exclude_paizh = 1
exclude_lianzh = 1
```

### 9.2 工兵/地雷特例

```text
X 吃了我的地雷，位置在后两排
```

更新：

```text
candidate_gongb[X] = 1
mine_clear_suspect[X] = 1
rank_floor[X] 不变
exclude_gongb[X] 不置位
```

```text
我的棋攻击 X 后死，X 仍活，X 在后两排
```

更新：

```text
candidate_dilei[X] = 1
mine_suspect[X] = 1
rank_floor[X] 不变
```

---

## 10. Observation 通道设计

新增 channel group：

```text
combat_memory
```

建议完整通道：

```text
direct_ate_my_piece_id[120]
direct_ate_my_type[12]
chain_ate_my_piece_id[120]
chain_ate_my_type[12]
floor_ge_*[8]
exclude_type[12]
candidate_type[12]
special_hint[5]
recency[8]
```

其中：

### 10.1 floor_ge cumulative channels

```text
floor_ge_paizh
floor_ge_lianzh
floor_ge_yingzh
floor_ge_tuanzh
floor_ge_lvzh
floor_ge_shizh
floor_ge_junzh
floor_ge_siling
```

如果 floor 是营长+，则点亮：

```text
floor_ge_paizh
floor_ge_lianzh
floor_ge_yingzh
```

### 10.2 recency channels

```text
direct_recent_32
direct_recent_128
chain_recent_32
chain_recent_128
floor_recent_32
floor_recent_128
special_recent_32
special_recent_128
```

### 10.3 总通道数

```text
120 + 12 + 120 + 12 + 8 + 12 + 12 + 5 + 8 = 309
```

当前 observation 为 256 通道，新增后：

```text
OBS_CHANNELS = 565
```

---

## 11. 对 MoveNet / BeliefNet 的影响

### 11.1 MoveNet

MoveNet 当前步可以直接看到：

```text
这个敌子直接吃过我的哪些棋；
这个敌子吃掉过吃我棋的棋；
这个敌子至少是什么等级；
它不可能是什么；
它可能是地雷/炸弹/工兵。
```

因此它不需要从最近 32 步里自行恢复历史事件。

### 11.2 BeliefNet

BeliefNet 可以学习：

```text
floor_ge_yingzh → 低阶类型概率下降
exclude_gongb → 工兵概率下降到 0
mine_clear_suspect → 工兵概率上升
mine_suspect → 地雷概率上升
bomb_suspect → 炸弹概率上升
```

后续可选 hard mask：

```python
logits[exclude_type_mask] = -inf
logits[rank_below_floor] = -inf
```

建议先作为 observation 特征，不先 hard mask，避免规则实现错误导致不可恢复偏差。

---

## 12. 显存与训练配置影响

新增 309 个通道后：

```text
OBS_CHANNELS: 256 → 565
```

observation buffer 约增加：

```text
565 / 256 ≈ 2.2x
```

当前 v34 大模型 H20 显存已约 82GB，直接用 `num_envs=512` 可能 OOM。建议 CombatMemory 实验配置：

```yaml
env:
  num_envs: 256
  steps_per_env: 512
```

如果显存仍高，可降到：

```yaml
env.num_envs: 192
ppo.minibatch_size: 512
belief.infer_chunk_size: 64
```

---

## 13. 实现范围

需要修改：

```text
junqi_core/state.py
junqi_core/observation.py
junqi_core/batched_state.py
src/env/cuda/include/junqi_cuda.h
src/env/cuda/src/game_state.cu
src/env/cuda/src/observation.cu
src/env/cuda/src/bindings.cpp
tests/*
```

可能需要更新：

```text
junqi_rl/training/rollout_gpu.py
junqi_rl/networks/junqi_net.py
junqi_rl/networks/belief_net.py
```

多数网络代码读取 `OBS_CHANNELS` 常量，理论上自动适配，但 buffer 显存和 checkpoint 形状会变化，旧模型不可直接加载。

---

## 14. 测试计划

### 14.1 CPU 逻辑测试

```text
test_combat_memory_direct_ate_my_paizh
test_combat_memory_defender_kills_my_piece
test_combat_memory_transitive_chain
test_combat_memory_mine_clear_suspect
test_combat_memory_mine_suspect
test_combat_memory_bomb_suspect
test_combat_memory_exclude_gongb_on_non_mine_kill
test_combat_memory_no_floor_update_on_mutual
test_combat_memory_reset_clears_state
```

### 14.2 CUDA parity

```text
test_gpu_combat_memory_direct
test_gpu_combat_memory_chain
test_gpu_combat_memory_special
test_gpu_obs_combat_memory_parity
```

### 14.3 Observation shape

```text
OBS_CHANNELS == 565
RolloutBufferGPU allocates new shape
JunqiNet forward accepts new shape
BeliefNet forward accepts new shape
```

### 14.4 Non-cheating tests

确认 observer 只使用自己的 victim type：

```text
victim_seat == observer       → 可更新 direct type/floor
victim_seat != observer       → 不使用 victim true type
```

---

## 15. 实验计划

实现后新开：

```text
v35_combat_memory_full
```

建议配置：

```yaml
env.num_envs: 256
steps_per_env: 512
net.depth: 6
net.embed_dim: 256
belief.net.embed_dim: 512
```

对比基线：

```text
v34_big_move_belief，无 CombatMemory
v35_big_move_belief_combat_memory，有 CombatMemory
```

统一做 4096 局评估：

```text
win / loss / draw / ongoing / avg_len
```

成功标准：

```text
win_rate 明显提升
loss_rate 明显下降
draw_rate 不显著上升
BeliefNet 对敌方类型预测更稳定
```

---

## 16. 结论

完整 CombatMemory 不是粗略“吃几个子”的替代品，而是一套高阶、可解释、视角安全的历史逻辑系统。它可以表达：

```text
B 吃了我的排长 → B 至少连长，且不可能是工兵/排长
C 吃了 B → C 继承 B 与我排长的链式关系，C 至少营长
X 让我撞死且在后排 → X 可能地雷
Y 吃了我的地雷 → Y 可能工兵
```

这类信息正是当前 feed-forward MoveNet/BeliefNet 很难从单步 observation 中恢复的长期战斗记忆。完整实现后，它有望成为比单纯扩大模型更有效的 observation 增强方向。

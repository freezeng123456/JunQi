# Ataraxos 配方与 JunQi 参数等效性分析

本说明基于 Ataraxos 官方 Stratego 仓库（代码版本为当前仓库 `main`）及其论文预印本；不把二手博客作为参数来源。官方仓库入口：[AtaraxosAI/stratego](https://github.com/AtaraxosAI/stratego)，论文：[Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search](https://arxiv.org/abs/2511.07312)。JunQi 对应配置为 [`configs/ataraxos_selfplay.yaml`](../configs/ataraxos_selfplay.yaml) 与当前训练代码。

## 结论

可以从 `ckpt_001500.pt` 开始切换为 Ataraxos 风格的参数版本，但“等效”只能分为两层：

1. **算法语义等效**：可以做到。保留输入输出不变，使用 PPO、完整合法动作支持上的 `KL(π_new || π_collect)`、对数值优势的高分位筛选、rollout 单位的幂律退火，以及独立的布阵网络训练。
2. **数值/训练轨迹等效**：不能直接声称。JunQi 的动作空间、棋盘、网络结构、布阵表示和收集器不同；Ataraxos 论文的具体常数应视为移植起点，而不是保证相同曲线的魔法数字。

尤其要避免把 `alpha` 和动作采样温度混为一谈：在 Ataraxos 的 RL 实现里，`temperature` 是对均匀/加权 magnet 的正则温度；`alpha` 是论文对该反向 KL/磁力正则系数的记号。它不是 softmax sampling temperature，也不应改变模型接口。

### 当前 master 基线的有意偏离

当前实际部署基线保留 Ataraxos 的 rollout-global 0.75 quantile 与 timestep batching，但按实验目标把 move-policy magnet 初始系数从论文的 `0.05` 提高到 `0.1`，仍按 rollout 时钟执行 `alpha(t)=0.1/t^0.3`，且不设正下限：`temperature_coef=0.1`、`temperature_decay=0.3`、`temperature_floor=0.0`。这是为了维持更多探索而作的明确实验选择，并非声称与论文常数严格等价。ARR 仍有自己独立的同形 schedule。

## 官方实现中的参数含义

下面的含义直接对应官方代码：[`pyengine/core/rl.py`](https://github.com/AtaraxosAI/stratego/blob/main/pyengine/core/rl.py)、[`pyengine/core/buffer.py`](https://github.com/AtaraxosAI/stratego/blob/main/pyengine/core/buffer.py) 和 [`pyengine/arrangement/buffer.py`](https://github.com/AtaraxosAI/stratego/blob/main/pyengine/arrangement/buffer.py)。

| Ataraxos 概念 | 官方实现语义 | JunQi 映射/建议 |
|---|---|---|
| `clip_range=0.2` | PPO importance ratio 的裁剪区间 `[0.8,1.2]`。 | `ppo.clip_range: 0.2`，可直接等效。 |
| `td_lambda=0.8` | 用于 value return 的 TD(λ) 追踪；不是 GAE λ。 | JunQi 当前实现若暴露该字段，设 `0.8`。 |
| `gae_lambda=0.5` | 用于 policy advantage 的 GAE 追踪，终止状态处截断。 | 设 `0.5`；这比常见 PPO 的 0.95 更短视，是 Ataraxos 特征。 |
| `vf_coef=1.0`, `policy_coef=1.0` | value loss 与 policy loss 的组合权重。 | 保持 1.0/1.0。 |
| `max_grad_norm=0.267` | 每个更新后的全局梯度范数裁剪。 | 设 0.267；不要与动作温度混淆。 |
| `kl_coef=0.1` | 新策略相对冻结 collection policy 的完整反向 KL：代码计算 `sum π_new(log π_new-log π_collect)`，在完整合法动作支持上求和。 | 当前 master 已固定使用完整反向 KL，只需 `kl_coef: 0.1`；已删除的 `kl_mode` 不应再出现在新配置里。 |
| move `alpha=0.05/t^0.3` | `alpha * KL(π_new || magnet)` 是 magnet 正则强度，不是动作采样 softmax 温度。论文公式不含固定 floor/ceil。 | 使用 rollout 时钟、`temperature_coef: 0.05`、`temperature_decay: 0.3`、`temperature_floor: 0.0`。从 rollout 1500 按全局时钟续算时，首个新更新约为 `0.00557`。 |
| advantage 0.75 quantile | 每个 rank 对整个 rollout buffer 的全部有效位置，以原始 `|A|` 计算一个 0.75 分位阈值；随后每个 simulator timestep 只训练超过该统一阈值且 `|A|>=0.01` 的位置，因此单行不保证恰好保留 25%。 | JunQi 的字段语义是“保留比例”，所以使用 `adv_filt_rate: 0.25`、`adv_filt_thresh: 0.01`、`adv_filter_scope: rollout` 和 `minibatch_group: timestep`；`1.0` 是关闭分位过滤。 |
| `lr_coef=0.5`, `lr_decay=1.1`, `lr_ceil=1e-4`, `lr_floor=5e-6` | 学习率按训练迭代的幂律 schedule 更新，并夹在上下界。 | 设为 0.5/1.1/1e-4/5e-6；续训时保持全局 rollout。rollout 1500 附近仍处于 `1e-4` 上限是公式本身的正常结果，约到 rollout 2300 后才明显下降。 |
| `weight_decay=0` | AdamW 的权重衰减为零。 | 保持 0。官方训练容器使用 AdamW，见 [`pyengine/core/train_container.py`](https://github.com/AtaraxosAI/stratego/blob/main/pyengine/core/train_container.py)。 |
| `num_epochs_per_rollout=1` | 每批新 rollout 数据只进行一个 epoch；buffer 内再按 minibatch 迭代。 | 设 1，避免改变 on-policy 数据复用量。 |
| 202 per-step batches | 官方按 202 个 simulator step 分组训练；过滤前每组每 GPU 为 1,536 个位置，过滤后不超过约 1/4；只做一个 epoch。 | JunQi 应使用 `minibatch_group: timestep`。此模式每个采集时刻做一次更新，`minibatch_size` 不再决定拆批大小。 |
| `uniform_magnet` | 官方可使用合法动作均匀分布作为 magnet；非 uniform 模式使用按 origin 加权的合法动作分布。 | JunQi 的 `magnet_shape=piece_then_dest` 对应“按棋子/起点分层”的加权 magnet，不等于 flat legal-action uniform；应作为结构映射而非常数映射。 |
| setup `alpha=0.1/t^0.3`, KL `0.1` | 布阵网络有独立的温度、KL、学习率和 PPO 更新；官方 setup 使用完整游戏结果（MC，对应 TD(1)/GAE(1)），并把熵项加入布阵优势。 | `arr.enabled=true` 时独立设置 `reg_temp_init: 0.1`、`reg_temp_decay: 0.3`、`reg_temp_floor: 0.0`，不能只复制 move-policy 的参数。JunQi 当前 `n_arr/pool_size` 是容量/采样工程参数，不是论文中的模型超参数。 |
| `arr_lr=5e-5`, `arr_batch_size=1024`, `arr_num_epoch_per_train=5` | 布阵网络每次训练最多 5 个 epoch，批大小 1024，AdamW/零 weight decay，梯度上限 0.5。 | 若 JunQi 的 ARR 接口一致，直接映射；否则至少保持“独立 optimizer、独立 schedule、独立 batch”。 |

## 退火与续训的关键问题

论文把这些 schedule 写成 training iteration 的函数。JunQi 的 Ataraxos 配置明确用 `num_rollout` 作为时钟；checkpoint 会恢复 `num_rollout`、`num_train_step` 和 optimizer state，但继续采用新配置，而不会把旧 checkpoint 中保存的 `cfg` 覆盖回来。因此从 1500 续训可以直接换配方，不过必须先决定使用哪一种时钟语义：

- **绝对时钟（论文公式的字面续算）**：把 checkpoint 1500 视作 training iteration 1500。第 1501 次更新的 move alpha 约为 `0.05/1501^0.3 = 0.00557`，会从当前固定 `0.1` 突降约 17.9 倍；setup alpha 约为 `0.01115`。这在公式上最直译，但不是平滑切换。
- **阶段时钟（把 1500 当作新配方的起点）**：新阶段第一次 move 更新从 alpha `0.05` 开始，再按阶段内 iteration 衰减。这更适合作为“从已有能力开始训练一遍 Ataraxos 动态阻尼”，但当前代码没有 schedule origin/offset 字段，不能只靠现有 YAML 精确表达。当前 ARR 本来就在按全局 rollout 使用 `0.1/t^0.3`，因此不建议把 setup 时钟重置到 `0.1`；那会把 rollout 1500 左右约 `0.01115` 的 setup alpha 突然放大约 9 倍。
- **采样时钟等效**：Ataraxos 每次迭代每 GPU 是 `1536×202` 个位置，而 JunQi 当前为 `2048×512`。若按采样量重标定，系数和衰减速度还会变化；由于双方 GPU 数、动作空间和四人组队博弈都不同，不能把这种换算称为严格复现。

无论选哪一种，续训都应：

- 从 checkpoint 恢复 optimizer state 和全局训练计数；
- 明确记录 schedule 使用绝对时钟还是阶段时钟；
- 若严格照论文幂律形式，把 rollout 模式下的 `temperature_floor` 设为 `0.0`；`temperature_ceil` 在 JunQi rollout 模式中不参与计算；
- 让 `alpha=0.1` 的旧配置与 Ataraxos 的 `temperature` 正则系数分开记录。把 `alpha` 固定为 0.1 并不等价于 Ataraxos 的 `0.05/t^0.3` 退火。

还有一个 checkpoint 恢复细节：主 PPO 每轮都会用新配置重新计算并写入 optimizer 学习率，因此主策略的新 LR schedule 会生效；ARR trainer 当前直接加载旧 optimizer state，而且没有每轮重写 LR，所以 checkpoint 中旧的 ARR 学习率会覆盖 YAML 的 `arr.ppo.lr`。若切换到官方 `5e-5`，应在加载后只重设 ARR optimizer 的 `lr`（保留 Adam moments），或在框架中增加明确的 resume-hyperparameter 覆盖逻辑。

## 推荐的 JunQi 等效参数档

以下是“使用绝对 rollout 时钟、语义上最接近 Ataraxos”的可解析参数清单，而不是立即执行命令。未列字段继续继承 `configs/ataraxos_selfplay.yaml`：

```yaml
ppo:
  num_epochs_per_rollout: 1
  minibatch_group: timestep
  adv_filter_scope: rollout
  clip_range: 0.2
  gamma: 1.0
  td_lambda: 0.8
  gae_lambda: 0.5
  vf_coef: 1.0
  policy_coef: 1.0
  kl_coef: 0.1
  adv_filt_rate: 0.25       # JunQi 字段为保留比例，即 |A| 最大的约 25%
  adv_filt_thresh: 0.01
  uniform_magnet: true
  magnet_shape: piece_then_dest
  temperature_schedule_unit: rollout
  temperature_coef: 0.05
  temperature_decay: 0.3
  temperature_floor: 0.0
  lr_schedule_unit: rollout
  lr_coef: 0.5
  lr_decay: 1.1
  lr_floor: 5.0e-6
  lr_ceil: 1.0e-4
  max_grad_norm: 0.267
  weight_decay: 0.0
  ema_decay: 0.999

arr:
  enabled: true
  reg_temp_init: 0.1
  reg_temp_decay: 0.3
  reg_temp_floor: 0.0
  reg_norm: 10.0
  ppo:
    lr: 5.0e-5
    clip_range: 0.2
    policy_coef: 1.0
    vf_coef: 0.5
    ent_pred_coef: 1.0
    kl_coef: 0.1
    batch_size: 1024
    num_epoch_per_train: 5
    max_grad_norm: 0.5
```

JunQi 的 ARR buffer 已固定用 `td_lambda=1.0`、`gae_lambda=1.0` 处理完成的布阵数据，所以不需要在 YAML 中重复这两个字段。

若采用推荐的阶段时钟版本，只需增加类似 `ppo.temperature_schedule_origin_rollout: 1500` 的纯训练配置字段；它只改变 move schedule 的横坐标，不改变网络、checkpoint 张量形状或模型输入输出。ARR 应继续用全局 rollout 时钟，除非明确要做一个 setup 退火重启实验。

## 不应宣称等效的项目

- Ataraxos 的 `alpha`/温度正则与 JunQi 当前固定 `alpha=0.1` 不是同一 schedule。
- Ataraxos 的优势过滤是高绝对优势尾部过滤；JunQi `adv_filt_rate=1.0` 明确关闭过滤。
- 完整 reverse KL 与 sampled proxy 的期望目标不同；后者只用已采样动作，不能称为严格等效。
- `uniform_magnet` 与 `piece_then_dest` 只在动作分布结构相同且归一化方式一致时才等效；需要单元测试逐状态比较分布。
- ARR 的 `n_arr`、pool size、重新配对方式决定布阵覆盖和吞吐，并非 Ataraxos 论文常数的直接替代。

## 直接来源

- Ataraxos 官方仓库与训练入口：[`README.md`](https://github.com/AtaraxosAI/stratego/blob/main/README.md)、[`scripts/train/rl_main.py`](https://github.com/AtaraxosAI/stratego/blob/main/scripts/train/rl_main.py)。
- RL 配置、PPO、reverse-KL/magnet loss、schedule：[`pyengine/core/rl.py`](https://github.com/AtaraxosAI/stratego/blob/main/pyengine/core/rl.py)。
- 优势过滤与 TD/GAE：[`pyengine/core/buffer.py`](https://github.com/AtaraxosAI/stratego/blob/main/pyengine/core/buffer.py)。
- 布阵训练与独立 ARR 温度：[`pyengine/arrangement/buffer.py`](https://github.com/AtaraxosAI/stratego/blob/main/pyengine/arrangement/buffer.py)。
- 论文：[`arXiv:2511.07312`](https://arxiv.org/abs/2511.07312)。

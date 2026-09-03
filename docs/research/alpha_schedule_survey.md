# RL 中正则系数 alpha 的设置：一手资料调研与 JunQi 建议

日期：2026-09-03

## 结论先行

JunQi 的 `alpha` 应被视为“策略到探索参考分布（magnet policy）的 reverse-KL 权重”，不能直接套用 PPO 的 old-policy KL 系数，也不能直接套用 SAC 的 entropy temperature。对这种系数，最有直接可比性的公开一手资料是 Ataraxos：其 move policy 使用

\[
L_\pi=-\min(r\delta,\operatorname{clip}(r,0.8,1.2)\delta)
 +0.1\,KL(\pi_\theta,\pi_{\theta_t})
 +\alpha_t KL(\pi_\theta,\rho),
\]

其中论文把 `rho` 定义为“均匀选可动棋子、再在该棋子的合法着法中均匀选择”的 magnet policy；公开最终配置为

\[
\alpha_t=0.05/t^{0.3}.
\]

Ataraxos 同时明确指出：退火过慢会使棋力发育不足，退火过快会导致熵快速坍缩、后续学习容量耗尽。因此，JunQi 目前观察到的熵下降不能只靠“alpha 越大越好”解释，必须联合观察 entropy、`KL(policy, magnet)`、old-policy KL、clip rate、return 和 h2h。

Ataraxos 的论文和公开代码还有两个需要明确记录的差别：论文最终超参数表只写 `0.05/t^0.3`，公开代码的通用调度器则额外执行 `clip(..., floor=0.001, ceil=0.1)`；论文描述的是分层 magnet，公开代码 `RLConfig` 的默认值却是 `uniform_magnet=True`（全体合法动作均匀）。因此“论文严格对齐”和“公开代码默认值对齐”不是完全相同的实验。

本项目的推荐顺序是：先以 Ataraxos 论文的 `0.05/t^0.3` 作为可复现锚点；若目标是保留更长探索期，再在保持 magnet、优势过滤和学习率不变的前提下，独立测试带时间尺度的幂律退火，例如 `0.1/sqrt(1+(t-1)/50)`。不要用未经标定的固定 entropy guard 替代 schedule；若更换 magnet 分布，也必须重新标定 alpha，因为 `KL(pi,rho)` 的数值尺度会改变。

## 1. 必须区分的三个 alpha

### 1.1 JunQi / Ataraxos magnet alpha

这是 `alpha_t KL(pi, rho)` 的系数，`rho` 是探索参考分布。它直接把 policy 拉向一个规定的随机分布；在 reverse KL 约定下，正则项越大，策略越接近 `rho`。JunQi 当前代码中的 alpha 属于这一类。

Ataraxos 的 move-learning 公式和 `rho` 的定义见论文附录 D.4，表 21–22：[Ataraxos 论文（arXiv PDF）](https://arxiv.org/pdf/2511.07312)，尤其 PDF 第 28–29 页（文本行 1364–1417）。

### 1.2 PPO old-policy KL 系数

PPO 的 old-policy KL 惩罚或 early-stop 阈值约束的是新策略与“采样数据的旧策略”之间的更新幅度，不是探索参考分布。OpenAI Spinning Up 的 PPO 实现采用 clip objective，并用 `target_kl` 做近似 KL early stopping；文档给出的常用 `target_kl` 是 `0.01` 或 `0.05`，clip ratio 通常为 `0.1–0.3`：[OpenAI Spinning Up PPO 文档](https://spinningup.openai.com/en/latest/algorithms/ppo.html)。Ataraxos 的 move loss 则同时有 `0.1 KL(pi_theta, pi_theta_t)` 和 magnet KL；两者作用不同。

### 1.3 SAC temperature alpha

SAC 中的 alpha 是最大熵目标里的 entropy temperature，典型项是 `alpha * H(pi(.|s))`（或等价的 `-alpha log pi`）。SAC 的后续版本把 alpha 当作可学习的对偶变量，通过目标熵约束自动调节，而不是预先规定一条时间退火曲线：[SAC 原论文](https://arxiv.org/abs/1801.01290)，[SAC Algorithms and Applications](https://arxiv.org/abs/1812.05905)。这与 JunQi 的 `KL(pi,rho)` 不等价：只有当 `rho` 是均匀分布时，KL 到 `rho` 才与熵正则相差一个状态相关常数；JunQi 的 legal-action/magnet 分布通常不是全动作空间均匀分布。

## 2. Ataraxos 的直接证据

Ataraxos 的核心设计是动态阻尼：弱策略阶段使用强正则和较大的更新，强策略阶段使用弱正则和较小的更新。论文明确说明，正则包括 setup 的最大熵和 move 的 magnet reverse KL，并按不同幂律退火；退火过慢会让能力发育不足，过快会造成 entropy collapse：[论文第 5 页](https://arxiv.org/pdf/2511.07312)（第 435–451 行）。

最终 move 配置给出：

| 项 | Ataraxos 公布值 |
|---|---:|
| magnet reverse-KL 系数 | `0.05 / t^0.3` |
| old-policy reverse-KL 系数 | `0.1` |
| importance-ratio clip | `0.2`（即 `[0.8,1.2]`） |
| advantage quantile | `0.75` |
| advantage magnitude threshold | `0.01` |
| advantage lambda | `0.5` |
| outcome lambda | `0.8` |
| Adam learning-rate | `clip(0.5/t^1.1, 5e-6, 1e-4)` |
| gradient norm | `0.267` |
| epochs per iteration | `1` |
| value-loss coefficient | `1` |

来源：论文附录 D.4 表 22 和公式 (6)，[PDF 第 28–29 页](https://arxiv.org/pdf/2511.07312)。论文还报告其过滤掉绝对优势低于全体有效数据 0.75 quantile 的样本，并且同时要求 `|advantage|>=0.01`；这不是 alpha schedule，但会显著改变 policy 梯度的有效强度，因而不能在比较 alpha 时忽略。

Ataraxos 的 setup 网络另有一个不同的 entropy coefficient：`0.1/t^0.3`，见公式 (3) 和表 20，[PDF 第 26–27 页](https://arxiv.org/pdf/2511.07312)。因此“Ataraxos 的 alpha=0.1”如果不说明是 setup 还是 move，会混淆两个不同正则项；move magnet alpha 的公开初值是 `0.05`，不是 `0.1`。

公开代码提供了更具体、但与论文写法并不完全重合的工程默认值。`RLConfig` 中 move 的 `temperature_coef=0.05`、`temperature_decay=0.3`、`temperature_floor=0.001`、`temperature_ceil=0.1`；实际调度器是

\[
\alpha_t=\operatorname{clip}\left(\frac{0.05}{(t+1)^{0.3}};\ 0.001,\ 0.1\right).
\]

代码来源：[Ataraxos `pyengine/core/rl.py`](https://github.com/AtaraxosAI/stratego/blob/master/pyengine/core/rl.py)。这个 `0.001` 下限约到第 46 万个训练 iteration 才会生效，所以在 JunQi 的 3000-rollout 量级内，`floor=0` 与 `floor=0.001` 数值上没有区别。公开代码默认 `uniform_magnet=True`，而论文描述的是 piece-then-destination magnet；比较复现结果时应明确采用哪一个定义。

## 3. 常见 alpha 机制及量级

### 固定系数

固定系数常见于实验基线，优点是可解释、易复现；缺点是早期探索与后期收敛需要的正则强度不同。Ataraxos 论文的设计动机正是固定正则无法覆盖整个训练过程，因此最终采用退火而非固定 move magnet 系数。对于 JunQi，固定 `0.1` 可以作为短程 sanity check，不宜直接作为数千 rollout 的最终方案。

### 幂律退火

形式通常为

\[
\alpha_t=\alpha_0 t^{-p},\qquad p>0.
\]

Ataraxos 的一手配置是 move `alpha_0=0.05,p=0.3`，setup `alpha_0=0.1,p=0.3`。幂律的优势是简单、无突变、长程仍有正则；风险是 `p` 对中前期影响很大，且不同正则项/动作空间不能共用同一数值而不做标定。

工程上更容易控制的是带时间尺度的形式

\[
\alpha_t=\alpha_{\min}+(\alpha_0-\alpha_{\min})\left(1+\frac{t}{\tau}\right)^{-p}.
\]

其中 `alpha_0` 决定初始正则强度，`tau` 决定高正则维持多久，`p` 决定长尾衰减速度，`alpha_min` 决定是否长期保留残余正则。它没有“公认默认常数”；这些数值必须随奖励/优势尺度、KL 的 reduction 方式和 magnet 定义重新标定。

### 线性或分段退火

常见工程形式是从 `alpha_0` 线性降到 `alpha_min`，或在 warm-up 后分段下降。它便于指定“到第 N 个 rollout 降到多少”，但在终点有 kink，且 floor 会造成长期残余正则。对于 JunQi，只有在明确需要稳定下界时才使用 floor；不应再引入未经实验依据的 entropy guard。

### 目标熵自适应

SAC 的做法是优化温度变量，使策略熵接近目标熵；SAC Applications 论文将其表述为带温度的约束/对偶优化，并强调自动调温能减少手工调参：[SAC Applications](https://arxiv.org/abs/1812.05905)。这套机制适合“直接最大化熵”的 SAC 目标，但不能未经推导地替换 JunQi 的 magnet reverse-KL：JunQi 的参考分布携带合法动作结构，目标应是相对参考分布的偏离，而不是单独控制 Shannon entropy。

如果未来在 JunQi 中做反馈式 alpha，更稳妥的控制量应是 `KL(pi,rho)`，或按合法动作数归一化的熵 `H(pi)/log(|A_legal|)`，而不是固定的原始熵数值。可以在 `log(alpha)` 空间做很慢的对偶/反馈更新，并使用 EMA、上下界和滞回区间抑制振荡；这应作为独立实验功能，而不是直接替代经过验证的开环基线。

### KL 约束/Lagrange 自适应

PPO-Penalty 将 old-policy KL 放入目标，并自动调整 penalty coefficient；OpenAI 对 PPO 的说明明确区分了 PPO-Penalty 与 PPO-Clip：[OpenAI PPO 说明](https://openai.com/index/openai-baselines-ppo/)。这类自适应系数适合控制每次 policy update 的步长，不等价于把 policy 拉回 magnet。若 JunQi 未来采用自适应 alpha，应基于观测到的 `KL(pi,rho)` 或 entropy 设定控制目标，并保持 old-policy KL 单独管理。

## 4. 对 JunQi 的建议

### 当前实现处于什么位置

当前 `configs/ataraxos_selfplay.yaml` 使用 `0.1/t^0.3`、`floor=0` 和 rollout 时钟；因为配置未显式写 `magnet_shape`，实际继承默认 `uniform_legal`。所以它是一个“探索加强的 JunQi 变体”，不是严格的 Ataraxos 论文复现：move alpha 在 floor 生效前始终是论文值的 2 倍，magnet 结构也与论文文字定义不同。

| rollout `t` | Ataraxos 论文 `0.05/t^0.3` | JunQi 当前 `0.1/t^0.3` | 探索候选 `0.1/sqrt(1+(t-1)/50)` |
|---:|---:|---:|---:|
| 1 | 0.0500 | 0.1000 | 0.1000 |
| 100 | 0.0126 | 0.0251 | 0.0579 |
| 1000 | 0.00630 | 0.0126 | 0.0218 |
| 3000 | 0.00453 | 0.00905 | 0.0128 |

第三列和第四列不是文献推荐值，只是便于下一次有 GPU 时进行受控 A/B 的工程候选。

### 建议的论文锚点

先建立一个明确可复现的基线：

```yaml
magnet_shape: piece_then_dest
temperature_schedule_unit: rollout
temperature_coef: 0.05
temperature_decay: 0.3
temperature_floor: 0.0
adv_filt_rate: 0.25
adv_filt_thresh: 0.01
kl_coef: 0.1
```

这里的 `0.05/t^0.3` 是 Ataraxos 论文 move 项的直接对应，而不是声称 JunQi 与 Ataraxos 的模型、奖励尺度或 action space 完全相同。若 JunQi 继续使用 `uniform_legal` 而不是 `piece_then_dest`，应把它标为“JunQi-scaled baseline”，并用冷启动 `KL(pi,rho)` 实测结果重新决定初始 alpha。若目标是复现公开代码默认值，则应另设 `uniform_legal` 且 `floor=0.001` 的基线，并明确它不是论文文字所述的 magnet。

### 探索优先的候选

如果首要目标是延缓熵下降，可以测试

\[
\alpha_t=\frac{0.1}{\sqrt{1+(t-1)/50}},
\]

但它必须是独立候选，不要与 `adv_filt_rate`、magnet shape、learning rate 同时修改。与 `0.1/t^{0.3}` 相比，该曲线在早期衰减更慢、后期仍会继续下降；其设计目标是避免冷启动后 alpha 迅速变小，又避免长期固定 floor。

### 监控与停训判据

每个 rollout 至少记录：`alpha_t`、`H`、`Hc`、`KL(pi,rho)`、`KL(pi,pi_old)`、approx-KL、clip rate、return、value loss、gradient norm、skip 计数和固定协议 h2h。若 alpha 增大但 `KL(pi,rho)` 不降，可能是 policy 梯度/优势尺度主导；若 `KL(pi,pi_old)` 与 clip rate 突然升高，则更像更新过大，而不是 magnet alpha 不够。

## 5. 主要风险

1. **尺度不可直接搬运。** Ataraxos 的 `0.05` 依赖其奖励定义、动作分布、网络和 batch；JunQi 只能把它当量级锚点。
2. **参考分布改变会改变 KL 尺度。** `piece_then_dest` 与 `uniform_legal` 不是同一 rho；切换后必须重新测初始 KL 和正则 loss。
3. **优势过滤会伪装成 alpha 效果。** 保留 top-25% 会改变有效梯度分布；比较 alpha 时应固定过滤设置。
4. **熵不是唯一目标。** 过高熵可能代表没有学到，过低熵可能代表过早锁定；必须结合 h2h、return 和开局/前几步分布。
5. **不要把 guard 当作 schedule。** 固定 entropy guard 与幂律 schedule 是两种不同机制。Ataraxos 论文最终表没有列 floor，但公开代码默认有 `0.001` floor；在约 46 万次 move update 前它不生效。除非训练尺度会触及该区间，或有明确控制目标和消融证据，否则 floor 不应主导当前选择。

## 一手来源

- Sokota et al., *Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search*, arXiv:2511.07312：<https://arxiv.org/abs/2511.07312>；公式与最终超参 PDF：<https://arxiv.org/pdf/2511.07312>
- Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*, arXiv:1801.01290：<https://arxiv.org/abs/1801.01290>
- Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, arXiv:1812.05905：<https://arxiv.org/abs/1812.05905>
- OpenAI, *Proximal Policy Optimization*（官方算法说明）：<https://openai.com/index/openai-baselines-ppo/>
- OpenAI Spinning Up, *Proximal Policy Optimization*（官方实现文档与默认量级）：<https://spinningup.openai.com/en/latest/algorithms/ppo.html>
- Ataraxos 官方代码仓库（实现入口）：<https://github.com/AtaraxosAI/stratego>

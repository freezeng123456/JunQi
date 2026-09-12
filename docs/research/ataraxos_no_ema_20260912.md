# 无 EMA 的 Belief 联合学习：Ataraxos 一手资料核查

核查日期：2026-09-12。论文为 arXiv:2511.07312v1；官方仓库本日读取的 HEAD 为 `92db29e8ffc323b1b8a2804b5c3f84695d036b05`。下文区分原作事实、当前已实现的改动、尚未运行的候选方案。旧四格实验使用 EMA，其数值不能归属于本版本。

## 先澄清 EMA 的实际使用

Ataraxos 不是无 EMA 系统。论文在策略评估中使用指数平均参数；官方策略训练器保存并评估原始与平均策略，官方 belief 训练器也有 `ema_decay=0.99`、更新及保存平均权重的代码。因而原实现中“Belief EMA 0.999 与 Ataraxos 一致”的注释不准确。我们移除 EMA 是明确的本地设计取舍，并不构成 EMA 普遍无效的结论。[论文 §2.6、D.3–D.4](https://arxiv.org/html/2511.07312v1#S2.SS6)、[官方策略实现](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/core/rl.py#L295)、[官方 belief 训练器](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/belief/belief.py#L24)

代码已移除 BeliefNet、布阵网络的参数平均对象，直接使用当前训练参数；行棋策略原本已经使用当前参数，本次再删除残留 EMA 类与配置字段。新检查点不保存平均权重。旧检查点的额外 EMA 项被忽略，明确恢复原始 `net`/`policy`；旧 YAML 的 `ema_decay` 仅作为兼容输入发出警告，不进入有效配置，其他未知字段仍报错。没有改写旧实验数据或旧检查点。

## 最值得借鉴：约束实际策略变化

官方策略目标包含 PPO 裁剪、面向采样策略的 reverse KL，以及面向探索先验的 reverse KL。这两个 KL 有不同作用：前者限制一次策略更新偏离采样行为的程度，后者保留探索倾向。对应的简化目标是

\[
L_\pi=L_{\mathrm{PPO}}+\lambda D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{collect}})
+\alpha_tD_{\mathrm{KL}}(\pi_\theta\Vert\rho)+c_vL_V.
\]

原代码在完整合法动作分布上计算 KL；`rho` 支持均匀合法动作与先均匀选棋、再均匀选目的地两种实现，不能仅凭默认值声称具体实验使用其中哪种。论文描述后者。现有 JunQi 也支持这两种先验，但联合实验继承的默认是 `uniform_legal`；切到 `piece_then_dest` 应单独记录，不与 EMA 移除的效果混为一谈。[官方目标](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/core/rl.py#L540)、[官方分层均匀先验](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/utils/helper.py#L20)

论文强调探索正则与更新幅度共同随训练阶段减小；只快速撤掉正则，可能造成不稳定。这为我们提供的是机制启发，不是对军棋联合学习的收敛保证。当前短试验固定学习率与正则；不能把它写成已经实现并验证了论文的完整动态退火方案。[论文 §2.4](https://arxiv.org/html/2511.07312v1#S2.SS4)

对联合学习的关键推论：PPO 的策略 KL 只能约束固定输入上的策略更新，不能自动约束 BeliefNet 改变输入后造成的行为变化。因此每轮先固定 belief 及采样观测，完成策略更新，再更新 belief，下一轮才使用新预测。精确复制采样策略是 PPO 的参考分布，不是时间平均。参数、归一化状态及采样观测都必须保持一致。本次保留该快照，并验证其数值完全相等、存储独立。

## Belief 应描述整盘隐藏配置

Ataraxos 的 belief 使用带因果顺序的 encoder–decoder；训练时右移真实棋型序列，推理时逐枚采样。完整配置概率按条件概率相乘，能够表达棋子之间的相关性。当前 JunQi 是每格 12 类边缘预测，尚未实现这一联合分布。[官方 belief 网络](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/networks/belief_transformer.py#L113)

官方采样循环逐步执行硬掩码：已分配棋型扣减剩余数量；移动过的棋子不能分配为不能移动的棋型；还检查当前分配是否会令后续棋子的移动约束无解。这个机制值得优先借鉴，例如不能让多个独立高置信预测共同超出军长、炸弹等剩余数量。当前 JunQi 的规则投影约束每格允许类型，不能据此宣称已经满足整组分配的一致性。[官方约束](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/belief/masking.py#L21)、[官方采样](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/belief/sampling.py#L19)

## 监督目标、对手分布与搜索用途

官方 belief 用独立 Adam 优化交叉熵，行棋和布阵模型只用于生成数据。终局标记用于定位完整对局，再从对局历史回取中间状态；“最终策略生成数据”不等于“只用终局盘面”。官方代码也记录相对规则基线的诊断。[官方训练目标](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/belief/belief.py#L173)、[历史位置采样](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/belief/buffer.py#L70)

论文先训练最终行棋/布阵策略，再训练 belief，用它为测试时搜索采样隐藏局面。这不是把不断变化的 belief 概率在线输入 PPO 的联合学习证明。我们当前的同局交替训练属于延伸方案。[论文 §2.5、D.5](https://arxiv.org/html/2511.07312v1#S2.SS5)

我们的建议是让 belief 只为真实棋型的预测负责：输入是玩家可知历史，模拟器真值只作为监督标签；阻断策略梯度进入 belief，避免预测为迎合价值函数而变得过度确定。评估用保留对局的 NLL、Brier、错误高置信率及规则基线差值，按对局划分数据，加入未参与训练的对手与布阵。官方网络的 dropout=0.2 可以作为泛化消融，但它不能代替对手分布覆盖。[官方网络配置](https://github.com/AtaraxosAI/stratego/blob/92db29e8ffc323b1b8a2804b5c3f84695d036b05/pyengine/networks/belief_transformer.py#L25)

低 belief 熵可能来自有效证据，不能直接定义为坍缩；需要检查“自信但错误”。策略熵也受合法动作数影响，需看归一化值、动作集中度和实际对局强度。以上评估建议是我们的推论，尚未完成新版本的长期或棋力验证。

## 当前配置与下一次可检验的问题

`iteration_joint_belief.yaml` 现在已无 EMA，仍保留上一轮的 25% 规则混合、渐入与发布检查，方便单独研究 EMA 移除。另提供未运行的 `iteration_joint_belief_clean.yaml`：直接使用当前 belief 的规则掩码后预测，取消概率混合、渐入、belief-to-rule KL 截断和动作分布触发的降权。初始监督预热、硬规则约束、空标签/非有限更新保护继续生效；这些条件不平均参数。

干净候选的一轮过程为：固定当前 belief 采样 → 用保存观测更新行棋网络 → 用独立 CE 更新 belief → 下一轮使用最新参数。初始预热阶段不接入随机初始化预测。该方案减少额外控制变量，但仍会发生由新 belief 引起的输入分布变化；不能保证不坍缩。先用“训练 belief 但不接入策略”的对照回答接入是否有效，再单独比较分层动作先验、自回归 belief，避免一次改变所有因素后无法归因。

建议后续先核验不带混合的交替训练，并以固定保留集与对局结果验收；再决定是否投入自回归 belief。测试时搜索是更后面的独立实验，不能用搜索收益替代原策略收益。

## 本地验证

- 定向回归：52 passed / 1 skipped。覆盖直接训练参数、旧 EMA 项忽略、空监督不触发 Adam、精确采样策略快照及训练入口。
- 全量本地回归：1,380 passed / 171 skipped / 70 warnings，87.80 秒。66 条是历史配置 EMA 字段兼容警告，另 4 条为原有依赖弃用和旧观测语义提示；跳过项需要 CUDA、原生扩展等本机不具备的条件。
- 四份真实历史检查点均成功恢复，原始 Belief 权重逐张量最大误差 0；原始与旧 EMA 的最大差异在 0.004660–0.005671，因此该检查确实区分了两套权重。源文件 SHA-256 在检查前后不变。
- 使用真实的行棋、布阵与 belief 训练器执行共同保存/读取，新的三个网络状态均无 EMA，Belief 权重逐张量完全一致。
- 简洁候选通过训练入口的 `--validate-only`，但本机没有 CUDA 扩展；这是配置校验，不是 GPU 运行结果。
- 未启动新的 GPU 训练。无 EMA 版本的长期稳定性、棋力和候选配置效果均未验证，先前四格试验保留为使用 EMA 的历史证据。

本地证据文件位于工作区 `outputs/JunQi-no-ema-tests-20260912.log`、`outputs/JunQi-no-ema-config-20260912.log` 和 `outputs/JunQi-no-ema-checkpoint-audit-20260912.json`。

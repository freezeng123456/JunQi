# 317 通道模型：按需特征与统一 checkpoint 入口

状态说明（2026-10-03）：当前默认采用 [事实输入路径](FACTS_ONLY_TRAINING_20261002.md)，317 个空间通道、0 个全局输入、语义 v5。可选模块按各自声明的输入语义决定能否接入；combat v1 / tactical v2 / relational v1 仅支持历史概率输入。下文包含这些历史模块的兼容说明，不能直接当作新基线配置。

本实现以 N/M/R 使用的源码 `741527582ae01c297f59f247a723f69f09afddfe` 为基线，固定观测语义 v4、317×17×17 空间输入和 129×129 动作空间。原始 N/M/R 使用 28 维全局摘要；新模型支持关闭摘要并使用可学习 CLS。它不转换历史 CNN、其他通道布局或 v3 观测。

28 维全局输入全部是已有空间特征的确定性汇总或重复编码。用户选择的新实验设置 `net.use_global_summary: false`，保持全部 317 空间通道，跳过摘要生成/缓存/输入。该字段缺失时保留旧架构以加载历史文件。来源、BeliefNet 职责、迁移与验证见 [28 维全局输入审计](GLOBAL_INPUT_28.md)。

## 历史概率模块配置

```yaml
net:
  use_global_summary: false
  policy_features:
    combat: 1
    tactical: 2
    # relational: 1
```

三个原有布尔字段 `combat_outcome_features`、`tactical_features`、`relational_features` 继续有效；`true` 等价于请求对应的已登记版本。新字段与旧字段取并集，同一模块只创建一次；版本冲突会报错。需要关闭模块时，应同时删除对应的新请求并关闭旧开关。显式空字典不会覆盖旧开关。

| 配置 | 请求的派生模块 |
|---|---|
| 基础网络 | `{}`，且旧开关全关 |
| N | `{combat: 1}` |
| M | `{combat: 1, tactical: 2}` |
| R | `{combat: 1, tactical: 2, relational: 1}` |

未知模块、错误版本、字符串伪布尔值均在配置校验/网络构造时拒绝。缺少一个训练时启用的模块不能靠填零兼容，严格权重加载会拒绝键不匹配。

## 执行与开销

1. `feature_specs.py` 只在构造模型时解析请求；启用的模块才导入和实例化。
2. 网络保存固定的执行顺序，推理时不查询注册表、不读取 manifest、不查版本。
3. 不启用派生模块时，不创建特征上下文，不额外 gather 观测，不增加参数。
4. `PolicyFeatureContext` 只提取请求的具名通道组，保留张量设备和自动求导。M/R 共享的通道与身份分布在本次调用内复用；实例不存入网络，不跨 forward、玩家或局面保留。
5. tactical/relational 原来各自 gather 全部 317 通道；现在 tactical 需要 38 个通道，二者合用共需要 52 个通道。combat 保留原先低精度观测的加法顺序，并复用己方棋子通道。这里是通道提取量变化，不是已测得的整体加速比；启用模块的通道依赖在构造时合并为静态索引，推理时一次 gather；实际收益仍需在目标 GPU 测量。
6. 纯 value 前向不计算派生模块。共享 policy/value 前向只对 policy 子集计算，包含仅开启 tactical 或 relational 的情况。

基础 317 通道编码和 GraphStem 继续处理完整输入；本轮按需的是其上的派生特征模块。权重为零也不会自动跳过启用模块，否则新模块无法获得训练梯度。

网络保留 `combat_head.*` / `tactical_head.*` / `relational_head.*` 权重路径、参数遍历顺序和模块初始化顺序。相同配置可沿用原 checkpoint，无需为了接口调整重新训练。N/M/R 的 R36384 完整归档已于 2026-09-30 取回并完成 CPU 核验；权重保存在仓库外的模型审计目录中。

这里的权重/顺序兼容指相同全局输入选项。关闭摘要会将 `global_proj` 替换成 `cls_token`，登记新架构 `junqi.graph317.cls.v1`，不能直接完整恢复旧 N/M/R 的优化器。显式迁移可保留共享权重并重新训练适应。

## Checkpoint 使用

```python
from junqi_rl.training.checkpoint import load_evaluation_policy, export_evaluation_policy

policy = load_evaluation_policy("ckpt_036384.pt", device="cuda")
export_evaluation_policy(policy, "candidate_policy.pt")
restored = load_evaluation_policy("candidate_policy.pt", device="cpu")
```

训练文件和轻量推理导出共用 `policy`、`train_cfg.net`、`checkpoint_meta` 契约。新增元信息包含架构 ID、实际启用的特征版本、全局维度，以及通道布局/棋盘紧凑编号的哈希；继续校验既有语义与特征版本字段。检查发生在权重写入模型前。

加载完整训练文件时先反序列化到 CPU，再只把行棋网络移到指定设备，避免优化器占用推理显存。推理导出使用原子写入，权重和配置可通过 `torch.load(..., weights_only=True)` 读取；不包含优化器、布阵或 belief，也不支持完整续训。标准加载器为兼容历史 Python 配置对象仍使用可信 checkpoint 加载路径。

缺少嵌入配置的旧 `state_dict` 或 `{net: ...}` 文件必须明确传 `net_config=`；自带配置的文件不允许用外部不同配置覆盖。旧布尔字段和等价的新版本映射视为相同配置。`eval_random_dense.py` 已改用此入口，并沿用 YAML 的 `extends` 解析。

旧文件没有观测语义元信息时，保留原来的警告兼容行为；它不证明原训练输入与当前输入等价。明确写着 v3 或 tactical v1 的文件会被拒绝。新导出描述当前构造的 v4 模型，不能将其当成旧语义运行的历史复现。

## 后续添加同一输入契约下的特征

实现一个 `nn.Module`，其 `forward(obs_spatial, *, context)` 返回 `(B, 16641)` 的 logits 残差；需要合法动作掩码的模块签名为 `forward(obs_spatial, legal_mask, *, context)`。可使用 `context.channels(name)` 或 `context.pieces()` 获取共享输入。直接单独调用模块时，也应允许 `context=None` 并创建本地上下文。

在应用启动时用 `register_policy_feature(PolicyFeatureSpec(...))` 注册模块 ID、版本、稳定的模块属性名、可信代码位置、所需 `channels`、是否需要合法掩码，以及 `supported_identity_encodings`。随后在 `policy_features` 中声明版本即可；无需改网络的分派代码。模块路径来自应用代码，checkpoint 只允许选择已注册 ID。

新模块若基于事实输入学习，可显式声明 `supported_identity_encodings=("known_only",)`；真正支持两种语义的模块可同时声明两个值。为保护历史兼容，未声明时仅支持 `legacy_probabilities`。兼容性在配置校验与网络构造时检查，forward 不增加语义分支。该声明是模块作者的契约，需要信息边界测试支持；它本身不会自动验证模块内部算法。未启用的模块不导入、不创建、不 gather。当前事实基线保持空模块列表，未添加新的 BeliefNet。

注册时版本不可重复覆盖。已创建模型的执行计划固定；本接口支持已经登记的 317/v4 概率输入和 317/v5 事实输入；新的布局或语义仍需新增架构适配，不能复用本接口绕过 checkpoint 校验。

## 验证状态

2026-10-01：无全局摘要路径、BeliefNet 接入和评估隔离的定向 CPU 回归共 281 项通过，2 项 CUDA 检查跳过；真实 N/M/R 抽样输出仍保持一致。详情与 V100 待执行检查见 [28 维全局输入审计](GLOBAL_INPUT_28.md)。下面保留 9 月 30 日的按需特征核验记录。

- 已执行：10 项不依赖 PyTorch 的注册/版本/惰性构造/索引规划测试；实际棋盘上 5 种启用组合的 NumPy 索引映射对照；改动 Python 文件语法检查；`git diff --check`。
- 已执行 PyTorch 2.5.1 CPU 定向回归：81 项通过、6 个子测试通过；2 项 CUDA 优化器验证因本机无 CUDA 跳过。覆盖启用/禁用组合、旧配置与新配置的权重顺序等价、按需 gather 对完整 gather 的数值与梯度等价、无 combat 的共享前向、上下文隔离、导出/加载、元信息拒绝和 CPU 加载路径。
- 已通过系统代理安装测试依赖。
- 已取回 N/M/R 的 R36384 完整 checkpoint，逐一验证既有 SHA-256、严格重载、有限权重与优化器张量、训练计数和 AdamW 全状态恢复。使用固定种子的 8 个合法玩家视角局面，对照原始 `741527582ae01c297f59f247a723f69f09afddfe` 源码：三者 FP32 合法动作 log-probability、value 均完全一致，贪心动作相同。纯推理导出经 `weights_only=True` 读取、统一入口重载后，权重逐张量相同。
- 北京 V100 GPU 训练/保存/恢复和实际性能测试仍待执行；CPU 核验不证明 GPU 吞吐或模型棋力，不报告 FPS 提升。

本次执行的定向回归（无需真实 checkpoint）：

```bash
python -m pytest -q tests/test_policy_feature_specs.py tests/test_optional_policy_features.py tests/test_checkpoint_compat.py tests/test_load_evaluation_policy.py tests/test_junqi_net_evaluate_actions.py tests/test_combat_outcome_features.py tests/test_training_config.py
```

历史 `test_feature_focus_20260921.py` 的 migration 测试依赖外部起点文件，需要取回真实归档后再执行。目标 V100 应在相同 batch、精度和权重下测 N/M/R 的前向与训练步，确认共享 gather 对实际吞吐的影响。

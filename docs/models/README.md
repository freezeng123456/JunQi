# 模型与证据索引

[registry.json](registry.json) 是机器可读索引，登记本地已回收的权重、输入契约、训练计数、父模型和证据文件的 SHA-256。权重与完整实验归档保留在仓库外。

| 模型 ID | 输入 | 角色 / 训练计数 |
| --- | --- | --- |
| `facts-original-r512` | 317/0，事实 v5 | 原始参考；512 次 rollout，完整局数未知 |
| `facts-counted-g2013` | 317/0，事实 v5 | 原始参考后累计新增 2,013 局 |
| `facts-counted-g8010` | 317/0，事实 v5 | 累计新增 8,010 局；开发集选出的对照 |
| `facts-counted-g10017` | 317/0，事实 v5 | 累计新增 10,017 局；当前冻结 π₀，已通过 V100 验收 |
| `facts-joint-g2000-e10` | 317/0 + 公开历史，联合信念 q12.v2 | π₀ 采集 2,000 局；训练 10 epoch，validation 选中第 10 个 |
| `legacy-n-r36384` | 317/28，历史概率输入 | N 历史对照；36,384 次 rollout |
| `legacy-m-r36384` | 317/28，历史概率输入 | M 历史对照；36,384 次 rollout |
| `legacy-r-r36384` | 317/28，历史概率输入 | R 历史对照；36,384 次 rollout |
| `legacy-probability-r512` | 317/0，历史概率输入 | 旧 joint 试验的策略父模型，区别于原始事实基线 |
| `legacy-joint-v1` | 历史概率输入的联合信念 | 未获得提升的旧搜索试验；不接当前事实接口 |

这里的 10,017 局不包括最初 512 次 rollout 中未知的完整局数。布局数、环境步数、rollout 批次和对局数分别记录。BeliefNet 的 2,000 局是冻结 π₀ 采集的独立监督数据，没有继续更新 π₀。当前索引包含上表十个已回收模型，默认模型仍为 `facts-counted-g10017`。

## 冻结 π₀ 的 V100 验收

2026-10-03，北京 V100 作业 `2348994` 完成，退出码 `0:0`，用时 1 分 14 秒。验收源码为 `6c6633220b67faffa0f01a5fa5c4f7a556a97f0b`，设备为 Tesla V100-SXM2-32GB，PyTorch `2.5.1+cu121`。

- 原 `ckpt_000890.pt` 严格加载通过，权重 SHA-256 保持 `ef60cd5f21086e77a61a4744b57b267d51590fae3df1d659fc87b09458661aa1`。
- 三种明暗模式、48 个观察位置的全部 317 通道，CPU/CUDA 最大差为 0；策略输出最大绝对差约 `1.14e-5`，通过预定容差。
- 短 PPO 训练的更新数从 3 增至 9；保存恢复后的动作、参数、优化器和完整局记录与不中断执行精确一致。
- GPU 测试报告为 **27 passed，0 failed，0 skipped**。

该短训练使用新优化器和 17 步规则上限，仅用于验收；其中四局不计入冻结模型的训练量，也没有替换 π₀ 权重。审计 JSON、JUnit XML、调度记录、协议和回收归档均在 `registry.json` 中登记路径与哈希。归档 SHA-256 为 `5a3945c331e4849eb743532a20a7d95aa2b065bce6a0d53cc91d6b8951af099e`。

这次早期验收源码中的信念约束版本仍为 `junqi.identity-domain-inventory-combat.v1`，证明的是冻结基线在上述源码下的验收。后续 `q12.v2` BeliefNet 的验收、训练和对局证据见下节。

## Facts BeliefNet：已回收训练与正式对局

北京 V100 作业 `2348998` 以 `COMPLETED / 0:0` 完成，用时 1 小时 29 分 53 秒；冻结源码为 `b7813532640e5238e64b49a5c47e9dc9219feda6`。设备为 Tesla V100-SXM2-32GB，PyTorch `2.5.1+cu121`。**34 项 GPU 测试全部通过，无失败或跳过**；π₀ 验收和 BeliefNet 训练验收均通过，后者的参数、优化器和随机状态恢复与不中断训练精确一致。验收短局与权重独立保存，不计入正式训练数据。

`facts-joint-g2000-e10` 使用事实输入与公开历史，约束版本为 `junqi.identity-domain-inventory-combat-q12.v2`，不使用旧规则概率或 28 维全局特征。冻结 π₀ 采集的 **2,000 个完整对局**产生 16,000 个观察位置；按初始布局分组划分如下，同一局和同一布局不会跨 split。

| 数据集 | 完整局数 | 观察位置数 |
| --- | ---: | ---: |
| Train | 1,595 | 12,760 |
| Validation | 198 | 1,584 |
| Test | 207 | 1,656 |

模型有 **327,436 个参数**，完成 10 个 epoch、3,990 次更新；只按 validation 条件 NLL 选模，选中第 10 个 epoch。Validation NLL 从 `1.441892` 降至 `1.279635`；独立 test NLL 为 `1.295129`，teacher-forced 准确率为 **43.51%**。这些指标衡量给定真实前缀时的条件身份预测，不等同于边缘身份准确率或棋力。

选中 checkpoint 的 SHA-256 为 `0c9d9718681fb61ebce73bb7dd3a949417706316c88b3415314f2eb6e92595bd`，父策略仍是上节的 `ef60cd5f…`，未被替换。

正式评测比较 **B = 同一冻结 π₀ + BeliefNet + 搜索** 与 **A = 冻结 π₀**。使用 100 个新布局，每个布局交换 B 所在队伍，完成全部 200 局。搜索配置固定为 1 个联合样本、深度 1、β = 0.5；这次比较衡量整个 B 系统的效果。

| B 的结果 | 实测值 | 95% 配对布局 bootstrap 区间 |
| --- | ---: | ---: |
| 胜 / 负 / 和 | 106 / 92 / 2 | — |
| 胜率 | 53.00% | 47.00%–59.00% |
| 得分率（和棋计 0.5） | 53.50% | 47.75%–59.50% |

两个区间都包含 50%，**目前没有确认棋力提升，默认继续使用冻结 π₀**。12,618 次搜索决策全部成功，12,618 个提案全部被接受，没有提案拒绝或采样预算失败；覆盖全部 621,224 个合法候选动作。工程 pilot 的 4 局单独归档，不混入正式 200 局。

本地恢复审计通过，核对了 **4,212 个产物**，包括父模型与数据绑定、validation 选模、heldout test 分母、完整配对记录及胜率区间重算。完整归档 SHA-256 为 `2d4364cc8518bebf2e378d38fa5a301434ec8932a1b2cee620473ea7eba43946`，与远端一致。登记的恢复审计不重跑 GPU、神经 NLL 或对局；原始实验记录与各文件哈希保存在归档和索引中。

## 本地核验

在仓库根目录执行，将两个占位路径替换成本地归档根目录：

```bash
python scripts/audit_model_registry.py \
  --artifact-root beijing-v100-preflight=/path/to/work/beijing-v100-preflight \
  --artifact-root model-audit=/path/to/outputs/junqi-model-audit-20260928 \
  --output /path/to/model-registry-audit.json
```

工具只用 Python 标准库检查清单、文件存在与哈希，不反序列化权重。任一模型、root 或文件缺失，以及哈希不符，都会返回非零状态。模型的 CPU 严格加载结果另记在清单的 `verification` 字段；哈希检查通过不能替代 GPU 验收或棋力评测。

`training_upload_archive` 和 `training_source_manifest` 指向原实验冻结文件；N/M/R 的 `compatibility_reference_source_commit` 仅是兼容性核验的参考源码。当前工程清理的提交不能冒充这些旧模型的训练源码。

评测原始逐局记录、协议及汇总也登记了哈希。历史 1,200 局评测参与了 π₀ 选择；本次正式 BeliefNet 对比使用独立的新布局集合，结果与历史选模评测分别登记。完整恢复审计入口为 `scripts/audit_facts_belief_results.py`，其通过记录作为新模型的 `recovery_audit` 证据。

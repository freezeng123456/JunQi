# 模型与证据索引

[registry.json](registry.json) 是机器可读索引，登记本地已回收的权重、输入契约、训练计数、父模型和证据文件的 SHA-256。权重与完整实验归档保留在仓库外。

| 模型 ID | 输入 | 角色 / 训练计数 |
| --- | --- | --- |
| `facts-original-r512` | 317/0，事实 v5 | 原始参考；512 次 rollout，完整局数未知 |
| `facts-counted-g2013` | 317/0，事实 v5 | 原始参考后累计新增 2,013 局 |
| `facts-counted-g8010` | 317/0，事实 v5 | 累计新增 8,010 局；开发集选出的对照 |
| `facts-counted-g10017` | 317/0，事实 v5 | 累计新增 10,017 局；当前冻结 π₀，已通过 V100 验收 |
| `legacy-n-r36384` | 317/28，历史概率输入 | N 历史对照；36,384 次 rollout |
| `legacy-m-r36384` | 317/28，历史概率输入 | M 历史对照；36,384 次 rollout |
| `legacy-r-r36384` | 317/28，历史概率输入 | R 历史对照；36,384 次 rollout |
| `legacy-probability-r512` | 317/0，历史概率输入 | 旧 joint 试验的策略父模型，区别于原始事实基线 |
| `legacy-joint-v1` | 历史概率输入的联合信念 | 未获得提升的旧搜索试验；不接当前事实接口 |

这里的 10,017 局不包括最初 512 次 rollout 中未知的完整局数。布局数、环境步数、rollout 批次和对局数分别记录。新 facts BeliefNet 的最终权重、训练记录和对局证据待完整回收核验后另行登记；当前索引只包含上表九个已回收模型。

## 冻结 π₀ 的 V100 验收

2026-10-03，北京 V100 作业 `2348994` 完成，退出码 `0:0`，用时 1 分 14 秒。验收源码为 `6c6633220b67faffa0f01a5fa5c4f7a556a97f0b`，设备为 Tesla V100-SXM2-32GB，PyTorch `2.5.1+cu121`。

- 原 `ckpt_000890.pt` 严格加载通过，权重 SHA-256 保持 `ef60cd5f21086e77a61a4744b57b267d51590fae3df1d659fc87b09458661aa1`。
- 三种明暗模式、48 个观察位置的全部 317 通道，CPU/CUDA 最大差为 0；策略输出最大绝对差约 `1.14e-5`，通过预定容差。
- 短 PPO 训练的更新数从 3 增至 9；保存恢复后的动作、参数、优化器和完整局记录与不中断执行精确一致。
- GPU 测试报告为 **27 passed，0 failed，0 skipped**。

该短训练使用新优化器和 17 步规则上限，仅用于验收；其中四局不计入冻结模型的训练量，也没有替换 π₀ 权重。审计 JSON、JUnit XML、调度记录、协议和回收归档均在 `registry.json` 中登记路径与哈希。归档 SHA-256 为 `5a3945c331e4849eb743532a20a7d95aa2b065bce6a0d53cc91d6b8951af099e`。

验收源码中的信念约束版本仍为 `junqi.identity-domain-inventory-combat.v1`。后续 `q12.v2` 网络与训练模型需要各自的 GPU 和对局证据；本次结果证明的是上述源码下冻结基线的验收，不能据此登记新 BeliefNet 或宣称其棋力提升。

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

评测原始逐局记录、协议及汇总也登记了哈希。这组 1,200 局评测参与了 π₀ 选择；后续正式 BeliefNet 对比要使用新的布局集合。

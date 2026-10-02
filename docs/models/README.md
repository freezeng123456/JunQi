# 模型与证据索引

[registry.json](registry.json) 是机器可读索引，登记本地已回收的权重、输入契约、训练计数、父模型和证据文件的 SHA-256。权重与完整实验归档保留在仓库外。

| 模型 ID | 输入 | 角色 / 训练计数 |
| --- | --- | --- |
| `facts-original-r512` | 317/0，事实 v5 | 原始参考；512 次 rollout，完整局数未知 |
| `facts-counted-g2013` | 317/0，事实 v5 | 原始参考后累计新增 2,013 局 |
| `facts-counted-g8010` | 317/0，事实 v5 | 累计新增 8,010 局；开发集选出的对照 |
| `facts-counted-g10017` | 317/0，事实 v5 | 累计新增 10,017 局；当前冻结 π₀ |
| `legacy-n-r36384` | 317/28，历史概率输入 | N 历史对照；36,384 次 rollout |
| `legacy-m-r36384` | 317/28，历史概率输入 | M 历史对照；36,384 次 rollout |
| `legacy-r-r36384` | 317/28，历史概率输入 | R 历史对照；36,384 次 rollout |
| `legacy-probability-r512` | 317/0，历史概率输入 | 旧 joint 试验的策略父模型，区别于原始事实基线 |
| `legacy-joint-v1` | 历史概率输入的联合信念 | 未获得提升的旧搜索试验；不接当前事实接口 |

这里的 10,017 局不包括最初 512 次 rollout 中未知的完整局数。布局数、环境步数、rollout 批次和对局数分别记录。新 BeliefNet 尚未训练。

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

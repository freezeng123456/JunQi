# JunQi：固定学习率的军棋特征对比结果

本轮已完成共同36,384轮主比较训练、15项共16,896局对打／诊断、模型严格重载与本地回收。主比较和最后参数均已保存到私有HF。可选38,432轮加训未完成，其不等预算终点单列保存。

兵力交换／身份不确定性组M对原模型N得分率52.49%，95%配对区间50.54%–54.42%；关系特征组R未证实对N的收益，对M得分率46.48%。这些是单次训练和多重比较下的探索结果，没有自动替换默认模型。

- [完整结论、胜负和与限制](CONCLUSION.zh.md)
- [最终状态与发布位置](STATUS.md)
- [固定预算和评估协议](reproduce/PROTOCOL.md)
- [独立结果复算](final_audit/independent_result_audit.json)
- [六份检查点的严格重载与优化器审计](final_audit/final_checkpoints_verified.json)
- [HF完整历史核验](final_audit/hf_full_history_verified.json)
- [旧F错误与历史保存说明](SUPERSEDED.md)
- [启动时的原始快照](startup/README.md)及[原始清单](startup/SHA256SUMS)

实验冻结源码为[741527582ae01c297f59f247a723f69f09afddfe](https://github.com/freezeng123456/JunQi/commit/741527582ae01c297f59f247a723f69f09afddfe)。恢复旧实验必须使用该提交、原始检查点和配置；本PR只归档证据，不切换运行中的策略。`source/`只展示相关文件，不是独立可运行的精简包。

[私有HF参数、优化器与完整历史](https://huggingface.co/a1390892757/junqi-guarded-continue-20260914/tree/main/feature_focus_20260921)。每组`primary/`为共同预算模型，`terminal/`为最后参数，`complete/`为完整历史。模型文件不提交到此GitHub目录。

启动证据已按原始清单逐字节复制到`startup/`，原清单仅移动并保留字节；逐局记录、原始报告与`FINAL_SHA256SUMS`保持不变。审计脚本按原实验目录布局运行，需要在环境中提供HF_TOKEN，凭据不在仓库内。新机器必须使用与实验一致的CUDA运行库与配置；本次PR整理没有重跑GPU实验。

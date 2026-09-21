# JunQi：固定学习率的军棋特征消融

状态：已通过本地和真实H20检查，N/M/R三组已进入正式训练。当前文件是预先固定的方案、实现分析和已核验启动证据；独立胜率结论尚未完成。后续N/M/R目录保存逐局结果及报告。

- [特征选择与作用分析](FEATURE_PLAN.zh.md)
- [当前状态和未完成边界](STATUS.md)
- [固定预算、种子和评估协议](reproduce/PROTOCOL.md)
- [旧F错误及历史保存说明](SUPERSEDED.md)
- [本地独立回收/重载证据](local_startup_audit.json)

实现冻结于[741527582ae01c297f59f247a723f69f09afddfe](https://github.com/freezeng123456/JunQi/commit/741527582ae01c297f59f247a723f69f09afddfe)。完整源码在该提交及私有HF的source.bundle；source/只展示本次相关文件，不是独立可运行的精简包。

[私有HF参数及审计文件](https://huggingface.co/a1390892757/junqi-guarded-continue-20260914/tree/main/feature_focus_20260921)。版本与SHA以回执为准；smoke仅验证实现，参数已丢弃，不作为正式训练起点或候选优胜模型。

复现实验需要冻结源码、start.pt、已记录哈希的CUDA运行库和同一配置。recover.py通过进程环境HF_TOKEN授权读取私有仓库；凭据不包含在发布文件中。本地审计脚本中的路径对应原实验工作目录，其他机器应按该目录结构放置或调整路径。环境规则与输入schema未改变。

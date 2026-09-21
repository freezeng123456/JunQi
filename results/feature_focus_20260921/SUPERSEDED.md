# Superseded experiments preserved on explicit user pivot

2026-09-21 用户将重心转到特征：旧学习率/加深网络/特征v1运行已保存并退出；A32288、B32624、F27432、D27263完整历史已回收到 outputs/JunQi-feature-focus-20260921/preserved。新实验固定学习率1e-5、4层，详见新目录。所有JunQi定时任务保持暂停。

另：F v1军衔价值公式方向错误，将强军衔映射成负值；原结果仅代表这一错误实现，不能用来判断预期中的正向材料价值特征是否有效。新实验使用修正版本2；保留v1源码和所有原始数据。

Archives contain original configs, logs, numbered checkpoints and upload receipts. Original F v1 must be loaded with frozen source cff8496c09a8d9cea2bf25ff48c65ec388e304ad; do not use corrected feature-v2 semantics on old F weights. No experimental result has been deleted or relabeled as a feature-v2 result.

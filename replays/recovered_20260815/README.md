# 已保存胜利复盘

这是目前还能从旧工作环境取回的完整胜局复盘，不是重新生成的示例。

## 复盘概况

- 结果：Team 0 获胜
- 步数：603
- 对局方式：Team 0 的 greedy policy 对随机合法动作
- 记录运行：`h20_long_300_256x1024_20260815`
- 记录元数据中的来源 commit：`2c238c9cd01dccb136b59bfe249559a7b4214648`
- 记录元数据中的 checkpoint 路径：`/root/JunQi/exps/h20_long_300_256x1024_20260815/ckpt_000300.pt`

checkpoint 文件本身已经找不到；本目录保存的是复盘数据和可离线查看的棋谱，不代表模型参数也已恢复。

## 文件

- `victory_team0_seed2026081500.npz`：现代 `TrajectoryWithPolicy` 记录，含 belief、Top-K、value 和动作来源。
- `victory_team0_seed2026081500.jql`：原生 GTK 客户端可读的紧凑复盘文件。
- `victory_team0_seed2026081500.json`：复盘元数据和校验信息。
- `moves.jsonl`：603 行逐步棋谱。
- `analysis.md`：第一轮离线统计和关键战斗表。

## 查看方式

现代复盘可以使用仓库的 Web 复盘工具；原生客户端可以直接打开 `.jql`，或从“文件”菜单选择“打开复盘”。原生客户端使用恢复后的原始 GTK 棋盘、原始颜色棋子条和右侧复盘信息。

## SHA-256

```text
27c7bf501b52db359cca5c8772d852c85d6db65ccb32e4ab75c28a15c274d271  victory_team0_seed2026081500.npz
bbbe3590e6b31928924d4c88e2733e30981c65edcaa0996db3583cef0067124c  victory_team0_seed2026081500.jql
3c83c58f27b6783a05b0a6ed082b4f804bb1847aaa73cfc64fc49838dbac0ad4  victory_team0_seed2026081500.json
8df52cb53f338b44c3fc2d86ca5caef85acb15a120df4568a2a05f2e9ed3fdb9  moves.jsonl
fccbee3a09fc0aba93d07b63895bb2c7a6a48563e802fcd6c1b6943fe5e53b20  analysis.md
```

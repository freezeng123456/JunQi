# JunQi — 四国军棋客户端与强化学习框架

JunQi 包含一套可玩的 GTK 客户端、遗留 C 对弈引擎、权威 Python
规则实现、CUDA 批量环境以及 PPO/BeliefNet/ArrangementNet 训练代码。

**唯一主干是 `main`。** 新开发和质量检查以更新后的 `origin/main` 为基线，
改动统一合入 `main`；不再维护或重新创建平行的 `master`。
原两条主干的整合与保留项见 [主干收敛记录](docs/engineering/single-main-20260908.md)。

项目当前的职责边界是：

- `junqi_core`：规则、状态、信息模型和复盘格式的权威实现。
- `junqi_rl`：训练环境、网络、采样器和评估工具。
- `src/env/cuda`：经过 Python parity 测试的加速实现。
- `junqi_viz`：人类对局与 RL 对局的可视化复盘服务。
- `legacy_engine`：遗留 C 引擎和规则对照实现。
- `legacy_gui`：桌面客户端；棋子继续使用项目原始资源。

## 快速开始

### 运行桌面游戏

macOS 需要 GTK3、XQuartz 和 `pkg-config`：

```bash
brew install gtk+3 pkg-config
make
./run_mac.sh
```

也可以使用 `make run`。启动脚本会同时启动 C 引擎和 GTK 客户端。
客户端启动后已经带有四方默认布局，直接点击右下角“开始”即可进入游戏；
需要换布局时，先点击对应一方的“调入布局”并选择 `.jql` 文件。

客户端自身的声音开关位于“设置 → 静音”。它会持久化到
`legacy_gui/bin/config.ini`，并立即终止仍在播放的声音进程。

### Windows 客户端

Windows 版本沿用同一个 GTK 界面、C 引擎、UDP 协议、复盘和静音逻辑，
使用 MSYS2 的 UCRT64/MinGW 工具链构建。先安装 [MSYS2](https://www.msys2.org/)，
在 **UCRT64** 终端执行：

```bash
pacman -Syu
pacman -S --needed mingw-w64-ucrt-x86_64-toolchain \
  mingw-w64-ucrt-x86_64-gtk3 mingw-w64-ucrt-x86_64-pkgconf
cd /path/to/JunQi
mingw32-make windows
```

构建结果位于 `legacy_engine/bin/windows/` 和
`legacy_gui/bin/windows/`。在仓库根目录双击 `run_windows.bat`，或在
PowerShell 执行 `./run_windows.ps1` 即可启动；脚本只会回收自己启动的
两个引擎进程，不会影响其他 JunQi 进程。

需要分发给没有 MSYS2 的机器时，在 UCRT64 终端执行：

```bash
bash tools/package_windows.sh
```

`dist/JunQi-windows/` 会包含棋盘资源、声音、启动脚本和 GTK/MinGW 运行库，
可直接压缩后分发。每次推送到 GitHub 后，Actions 的 **Windows client** 工作
流也会自动构建并上传同名 artifact。

### 安装 Python 核心

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python run_tests.py --profile core
```

核心测试不要求 Torch 或 NVIDIA GPU。在未安装 Torch 时，测试入口会明确
选择 core profile，而不会在 pytest 收集阶段失败。

### 安装强化学习依赖

```bash
python -m pip install -e ".[rl,dev]"
python run_tests.py --profile rl
```

CUDA 扩展需要在 NVIDIA Linux 主机上单独构建：

```bash
python build_cuda.py
python -m pytest -q tests/test_gpu_step_batch.py tests/test_gpu_obs_parity.py
```

## 常用命令

```bash
make test              # 无 Torch 也能运行的核心测试
make test-rl           # 要求 Torch 的完整 Python 测试
make lint-critical     # 阻止未定义名称、重复定义等高置信问题
make check             # lint-critical + core lint + typing + tests
make legacy-engine     # 构建 C 引擎
make legacy-gui        # 构建 GTK 客户端
make legacy-sanitize   # ASan/UBSan 构建 + 畸形 UDP 协议冒烟
```

GitHub Actions 分别验证 Python 核心、RL CPU 和遗留 C/GTK 构建。CUDA
parity 仍需在带 NVIDIA GPU 的执行器上运行。

## 复盘

普通对局和 RL 对局共用确定性的 `.npz` 轨迹格式。复盘支持前进、后退和
任意步定位，并验证最终状态哈希。

```bash
# 本地可视化界面
python -m pip install -e ".[viz]"
junqi-replay game.npz
# 浏览器访问 http://127.0.0.1:8765

# 终端检查/导出
python tools/replay_viewer.py game.npz --step 120 --validate
python tools/replay_viewer.py rl_game.npz --all --json
```

RL 轨迹可以包含每步 Top-K 动作概率、Value、Belief 快照及训练元数据。
训练每次评估默认在 `<save_dir>/replays/` 写入进度复盘，并在
`<save_dir>/league.json` 维护历史 checkpoint、Elo 和交叉对局结果。
详见 `docs/REPLAY.md`。

网页默认南方玩家视角，也可切换到西、北、东。四暗棋只直接显示本方棋型，
双明棋同时显示队友棋型；司令阵亡后，其军旗位置对所有玩家公开。
未明的棋子在接口、棋盘和动作详情中均显示为暗棋；确定的公开推断单独标注。
玩家视角只列出当前步及之前的公开事件，不返回训练内部的策略和元数据。
选择“全知复盘”可以查看完整棋谱诊断；这是本机复盘工具的显式功能，视角
选择不是多人对战的身份认证或权限隔离。

当前训练使用 **317 个事实通道、零全局量、`known_only`、观察语义 v5**。
未知身份通道为零，可见身份为 one-hot。公开历史和严格逻辑约束保留，手工身份
概率、概率混合及手工估值头不进入事实路径。CPU/GPU 评测和回放按模型的输入契约
创建观察；不同身份语义的对战需要独立适配器。

## 训练

当前入口为北京 V100 的完整对局配置 `configs/v100_facts_games.yaml`：

```bash
# 先检查配置；不启动训练，也不分配 GPU。
python scripts/train.py --validate-only

# 在已分配的 V100 节点上运行当前事实配置。
python scripts/train.py --config configs/v100_facts_games.yaml

# 续训时，累计完整局数目标应高于 checkpoint 已记录的局数。
python scripts/train.py --config configs/v100_facts_games.yaml \
  --resume /path/to/ckpt_000890.pt --target_completed_games 12000 \
  --save_dir exps/facts_continuation
```

配置继承为 `facts_base.yaml → v100_facts.yaml → v100_facts_games.yaml`。
事实基础配置独立定义模型、PPO 和关闭的可选组件。默认目标为新计数起点下累计
2,000 局；rollout 数只作为故障保护上限。历史 `v100_facts.yaml` 保留原始 512 次
rollout 的参数。H20、旧 `default.yaml`、课程学习和旧信念配置需显式指定，详见
[配置索引](configs/README.md)。底层 legacy 默认值用于解释历史权重。

固定策略为 `ckpt_000890.pt`，计数阶段累计完成 10,017 局；此前原始第 512 轮
checkpoint 的完整局数未知，不能把 512 当作局数。模型、父模型和证据文件见
[模型索引与核验说明](docs/models/README.md)，训练与评测结果见
[事实基线记录](docs/FACTS_ONLY_TRAINING_20261002.md)。当前联合 BeliefNet 的
采集、训练与对局协议见 [事实信念比较](docs/FACTS_BELIEF_COMPARISON_20261003.md)。
本轮 V100 验收、2,000 局数据训练及 200 局对比已完成并回收，见
[实测结果与证据](docs/FACTS_JOINT_COMPARISON_20261003.md)：联合信念＋搜索取得
106 胜、92 负、2 和，胜率 53%；95% 配对区间包含 50%，默认模型继续保留 π₀。

后续 [边缘与联合信念对照协议](docs/FACTS_BELIEF_ABLATION_20261004.md) 固定同一 π₀、
同一 2,000 局数据和搜索预算，使用两个信念训练种子比较边缘、联合及直接策略。
`train_facts_joint_belief.py --belief-kind marginal` 训练事实边缘模型；通用严格加载器
按架构与输入契约识别模型。边缘 NLL 与联合 teacher-forced NLL 分开解释。

历史信念原型使用 `scripts/legacy_train_joint_belief.py`、
`scripts/legacy_compare_joint_search.py` 和 `scripts/legacy_pilot_joint_belief.py`。
`train_joint_belief.py` 和 `compare_joint_search.py` 现已迁移到事实版本；
历史实验需显式使用 `legacy_*` 入口。历史原型不能消费当前事实策略。
`analysis.joint_search` 同样指向事实搜索，含糊的旧 `pilot_joint_belief.py` 别名已移除。

PPO 始终使用冻结的 collection policy，在完整合法动作分布上计算
`KL(π_new || π_collect)`。训练专用 `actions=` 快速路径只省去未使用的
随机动作采样，不以 sampled proxy 替换完整反向 KL。

训练结果不能只看单次对随机玩家胜率。正式比较至少应包含：

- 多随机种子；
- 固定 paired-seed 对局；
- 按完整布局对重采样的配对置信区间；
- 胜、负、和、平均步数与无战斗步数；
- 对随机玩家、EMA 策略和历史 checkpoint 的交叉评估。

## 规则与架构文档

- `docs/RULES.md`：冻结的规则合同。
- `docs/ARCHITECTURE.md`：Python/RL 架构。
- `docs/CUDA_ARCHITECTURE.md`：CUDA 状态和内核布局。
- `docs/DECISIONS.md`：架构决策记录。
- `docs/REPLAY.md`：复盘格式和工具。
- `docs/FACTS_ONLY_TRAINING_20261002.md`：当前事实基线、完整对局计数及后续步骤。
- `docs/models/registry.json`：模型血缘、输入契约与证据哈希索引。
- `docs/RL_ITERATION_PLAN.md`：历史实验漏斗；当前预算以完整对局为准。
- `docs/MAINTENANCE_2026-07-28.md`：本轮修复、验证边界与后续建议。
- `docs/PROGRESS_2026Q2_CURRICULUM.md`：历史课程学习实验。

## 安全

项目是内部 proprietary 软件。不要把访问 Token、密码、训练数据或私有
checkpoint 提交到仓库。推送使用凭据管理器或临时环境变量，禁止在 Git
remote URL 和 shell 脚本中硬编码密码。

## License

Proprietary — internal Tencent project.

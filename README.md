# JunQi — 四国军棋客户端与强化学习框架

JunQi 包含一套可玩的 GTK 客户端、遗留 C 对弈引擎、权威 Python
规则实现、CUDA 批量环境以及 PPO/BeliefNet/ArrangementNet 训练代码。

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
make check             # lint-critical + core tests
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

## 训练

```bash
# 当前 10.8M H20 自对弈配置：继承 Ataraxos 配方，只将
# ppo.adv_filt_rate 覆盖为 1.0（保留全部 transition）。
python scripts/train.py --config configs/h20_10m_current.yaml

# 通用小规模入口
python scripts/train.py --config configs/default.yaml
```

当前 10.8M 主线配置为 `configs/h20_10m_current.yaml`。它通过
`extends: ataraxos_selfplay.yaml` 复用模型、rollout、优化器、熵/KL 和
布阵池设置，仅关闭 advantage quantile 过滤。机器相关的续训 checkpoint、
输出目录等通过命令行参数传入。`configs/v42_*.yaml` 至
`configs/v44_*.yaml` 保留为历史课程学习实验记录。

PPO 始终使用冻结的 collection policy，在完整合法动作分布上计算
`KL(π_new || π_collect)`。训练专用 `actions=` 快速路径只省去未使用的
随机动作采样，不以 sampled proxy 替换完整反向 KL。

训练结果不能只看单次对随机玩家胜率。正式比较至少应包含：

- 多随机种子；
- 固定 paired-seed 对局；
- Wilson 置信区间；
- 胜、负、和、平均步数与无战斗步数；
- 对随机玩家、EMA 策略和历史 checkpoint 的交叉评估。

## 规则与架构文档

- `docs/RULES.md`：冻结的规则合同。
- `docs/ARCHITECTURE.md`：Python/RL 架构。
- `docs/CUDA_ARCHITECTURE.md`：CUDA 状态和内核布局。
- `docs/DECISIONS.md`：架构决策记录。
- `docs/REPLAY.md`：复盘格式和工具。
- `docs/RL_ITERATION_PLAN.md`：分级门禁、实验漏斗与模型晋级标准。
- `docs/MAINTENANCE_2026-07-28.md`：本轮修复、验证边界与后续建议。
- `docs/PROGRESS_2026Q2_CURRICULUM.md`：当前训练实验状态。

## 安全

项目是内部 proprietary 软件。不要把访问 Token、密码、训练数据或私有
checkpoint 提交到仓库。推送使用凭据管理器或临时环境变量，禁止在 Git
remote URL 和 shell 脚本中硬编码密码。

## License

Proprietary — internal Tencent project.

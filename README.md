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
客户端中载入或随机生成四方布局后，点击“开始”进入游戏。

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
```

GitHub Actions 分别验证 Python 核心、RL CPU 和遗留 C/GTK 构建。CUDA
parity 仍需在带 NVIDIA GPU 的执行器上运行。

## 复盘

普通对局和 RL 对局共用确定性的 `.npz` 轨迹格式。复盘支持前进、后退和
任意步定位，并验证最终状态哈希。

```bash
python tools/replay_viewer.py game.npz --step 120 --validate
python tools/replay_viewer.py rl_game.npz --all --json
```

RL 轨迹可以包含每步 Top-K 动作概率、Value、Belief 快照及训练元数据。
详见 `docs/REPLAY.md`。

## 训练

```bash
python scripts/train.py --config configs/default.yaml
```

当前实验主线使用随机敌方布阵，避免固定四方镜像布局造成
`piece_id → piece_type` 信息泄漏。最新实验配置位于
`configs/v42_*.yaml` 至 `configs/v44_*.yaml`。

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
- `docs/PROGRESS_2026Q2_CURRICULUM.md`：当前训练实验状态。

## 安全

项目是内部 proprietary 软件。不要把访问 Token、密码、训练数据或私有
checkpoint 提交到仓库。推送使用凭据管理器或临时环境变量，禁止在 Git
remote URL 和 shell 脚本中硬编码密码。

## License

Proprietary — internal Tencent project.

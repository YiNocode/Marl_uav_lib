## 项目简介

**marl_uav_lib** 是一个面向无人机（UAV）多智能体强化学习（Multi-Agent RL, MARL）的轻量级研究型代码库，主要目标：

- **复现与对比经典多智能体算法**：目前支持 **IPPO** 与 **MAPPO（集中式评价器 PPO）**，并提供 `RandomPolicy` 作为基线。
- **提供简洁清晰的训练/评估管线**：包含通用的 `MAC`（multi-agent controller）、`RolloutWorker`、`Trainer`、`Learner`、`Buffer` 等模块。
- **支持玩具 UAV 环境**：内置 `ToyUavEnv` 适合作为算法实验与调试的测试床。

非常适合：

- 做多智能体 RL 课程 / 论文中的 **小规模实验与验证**；
- 在统一框架下 **快速对比 IPPO / MAPPO 与随机策略基线**；
- 作为自己扩展新算法、新环境的起点。

---

## 环境与依赖

- **操作系统**：Linux / macOS / Windows（本仓库在 Windows 10 上开发验证）
- **Python 版本**：`>= 3.10`

主依赖（根据源码推断，推荐手动安装）：

- `numpy`
- `torch`
- `matplotlib`
- `pyyaml`
- （可选）`pytest`, `pytest-cov`（运行单元测试）

项目自带 `pyproject.toml`，但未显式列出上述运行时依赖，推荐在虚拟环境中手动安装。

### 安装步骤（推荐）

```bash
# 1. 创建并激活虚拟环境（示例）
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
# 或 source .venv/bin/activate  # Linux / macOS

# 2. 安装本地包（开发模式）
pip install -e .

# 3. 安装运行所需依赖
pip install numpy torch matplotlib pyyaml

# 4.（可选）安装测试依赖
pip install pytest pytest-cov
```

---

## 代码结构概览

核心目录结构（只列出与库本身强相关的部分）：

- `marl_uav/`
  - `agents/mac.py`：多智能体控制器（Multi-Agent Controller, MAC），封装共享策略对多个智能体的决策。
  - `buffers/`
    - `episode_buffer.py`：按 episode 存储的 on-policy 采样缓冲区。
    - `replay_buffer.py`：通用 replay buffer（off-policy 可用）。
  - `data/batch.py`：`Batch` / `EpisodeBatch` 等数据结构。
  - `envs/adapters/toy_uav_env.py`：玩具 UAV 多智能体环境适配器 `ToyUavEnv`。
  - `learners/on_policy/`
    - `ippo_learner.py`：IPPO 学习器。
    - `mappo_learner.py`：MAPPO 学习器（集中式 critic）。
  - `modules/encoders/`：MLP 等特征编码网络。
  - `modules/heads/`：策略头、价值头模块。
  - `policies/`
    - `actor_critic_policy.py`：标准 Actor-Critic 策略。
    - `centralized_critic_policy.py`：集中式 critic 策略（actor 用局部观测、critic 用全局 state）。
    - `random_policy.py`：随机策略，用作 baseline。
  - `runners/`
    - `rollout_worker.py`：负责环境交互与采样的 worker。
    - `trainer.py`：on-policy 训练主循环（使用 GAE 计算优势）。
    - `evaluator.py`：评估若干 episode 的平均回报等指标。
  - `utils/`
    - `rl.py`：包含 `compute_gae` 等 RL 工具函数。
    - `config.py`：加载 YAML 配置。

- `configs/`
  - `env/toy_uav.yaml`：玩具 UAV 环境配置。
  - `algo/ippo.yaml`：IPPO 超参数配置。
  - `algo/mappo.yaml`：MAPPO 超参数配置（含集中式 critic 标志）。
  - `model/*.yaml`：模型结构配置（MLP / RNN / attention / centralized_critic 等）。
  - `train/default.yaml`：默认训练顶层配置（指向 env / algo / model）。
  - `experiment/mappo_toy.yaml`：在 ToyUavEnv 上跑 MAPPO 的实验配置。

- `scripts/`
  - `train.py`：通用 IPPO / MAPPO 训练入口脚本（通过 `--train-config` 选择配置）。
  - `run_ippo_toy_uav.py`：在 ToyUavEnv 上对比 RandomPolicy vs IPPO，并绘制学习曲线。
  - `run_mappo_toy_uav.py`：在 ToyUavEnv 上对比 RandomPolicy vs MAPPO，并绘制学习曲线。
  - `eval.py`、`run_rollout_episode.py` 等：示例评估/采样脚本。

- `tests/`：针对 MAC、buffer、policy、centralized critic 等的单元测试及 smoke test。

---

## 快速开始

### 1. 使用通用训练脚本训练 IPPO / MAPPO

通用训练入口位于 `scripts/train.py`，内部会根据顶层 YAML 配置构建环境、策略、学习器与训练循环。

**默认训练（IPPO + ToyUavEnv + MLP）**：

```bash
cd e:\lyn\year_1\research\marl_uav_lib
python scripts/train.py
```

`scripts/train.py` 中默认使用：

- 顶层训练配置：`configs/train/default.yaml`
  - `env: configs/env/toy_uav.yaml`
  - `algo: configs/algo/ippo.yaml`
  - `model: configs/model/mlp.yaml`
  - 其余如 `seed`、`total_timesteps`、`log_interval` 等控制训练调度。

运行时会：

- 创建 `ToyUavEnv` 环境实例；
- 根据 `model` 配置构建 `ActorCriticPolicy` 或 `CentralizedCriticPolicy`；
- 用 `MAC` 封装多智能体决策；
- 使用 `Trainer` 执行「采样 → GAE/returns 计算 → learner.update → 打印日志」的主训练循环；
- 训练结束后，用 `Evaluator` 在若干 episode 上评估当前策略性能。

**使用 MAPPO + 集中式 critic 训练 ToyUavEnv（示例）**：

```bash
python scripts/train.py --train-config configs/experiment/mappo_toy.yaml
```

其中 `configs/experiment/mappo_toy.yaml` 指定：

- `env: configs/env/toy_uav.yaml`
- `algo: configs/algo/mappo.yaml`（`use_centralized_critic: true`）
- `model: configs/model/centralized_critic.yaml`

### 2. 对比 RandomPolicy vs IPPO / MAPPO（含画图）

我们提供了两个完整示例脚本，用于在 ToyUavEnv 上对比随机策略与学习到的策略：

- **RandomPolicy vs IPPO**：

```bash
python scripts/run_ippo_toy_uav.py
```

- **RandomPolicy vs MAPPO（集中式 critic）**：

```bash
python scripts/run_mappo_toy_uav.py
```

脚本行为概览：

- 首先在 ToyUavEnv 上多次评估 `RandomPolicy`，得到 baseline 平均回报；
- 然后对 IPPO / MAPPO 分别进行多 seed 训练，每个 epoch 后进行评估；
- 使用 `matplotlib` 绘制 **环境步数 vs 平均评估回报** 曲线，并用虚线标出 RandomPolicy 基线；
- 最后在控制台打印最终性能对比结论。

注意：这两个脚本假定运行在 CPU 上，如需 GPU 请自行调整 `device` 相关代码。

---

## 配置文件说明（简要）

### 1. 顶层训练配置：`configs/train/default.yaml`

主要字段：

- `env`：环境配置文件路径（如 `configs/env/toy_uav.yaml`）
- `algo`：算法配置文件路径（如 `configs/algo/ippo.yaml`）
- `model`：模型配置文件路径（如 `configs/model/mlp.yaml`）
- `seed`：随机种子
- `total_timesteps`：总训练步数（部分脚本目前使用 `num_epochs + rollout_steps` 形式控制）
- `log_interval` / `save_interval` / `eval_interval`：日志、模型保存与评估频率。

### 2. 环境配置：`configs/env/toy_uav.yaml`

关键字段：

- `env_id: toy_uav`
- `num_agents`：智能体数量（示例为 2）
- `episode_limit`：每个 episode 的最大步数
- `world_size`、`step_size`、`goal_reach_dist`：决定 UAV 的移动空间与到达目标判定阈值。

`ToyUavEnv` 会根据该配置设置观测维度 `obs_dim`、动作维度 `n_actions` 等（示例注释中为 `obs_dim=6`, `action_dim=5`）。

### 3. 算法配置：`configs/algo/ippo.yaml` 与 `configs/algo/mappo.yaml`

共同字段（部分名称略有差异）：

- 折扣与 GAE：`gamma`, `gae_lambda`
- PPO 裁剪：`clip_ratio`
- 损失系数：`ent_coef` / `vf_coef` 或 `entropy_coef` / `value_coef`
- 优化相关：`lr`, `epochs`, `max_grad_norm`
- MAPPO 额外：`use_centralized_critic: true`，以及可能的 `minibatch_size` 等。

`scripts/train.py` 在 `build_learner` 中会从这些字段中解析出学习器超参数。

### 4. 模型配置：`configs/model/*.yaml`

根据 `type` 字段决定使用的模型结构：

- `mlp`：MLP encoder + Actor-Critic policy（典型 IPPO 场景）
- `centralized_critic`：集中式 critic 策略（actor 用局部观测，critic 用全局 state）
- `rnn` / `attention` / `comm` 等文件可作为后续扩展的模型配置模板。

---

## 训练流程概要（内部实现）

虽然使用者通常只需调用脚本，但理解内部流程有助于自定义扩展：

1. **Rollout 阶段**：`RolloutWorker.collect_episode` 在环境中采样一个完整 episode，记录 `obs`, `actions`, `rewards`, `dones`, `values` 等。
2. **后处理阶段**：`Trainer._postprocess_episode` 使用 `compute_gae` 计算每个时间步的 `advantages` 和 `returns`，并打包成 `EpisodeBatch`。
3. **更新阶段**：`Learner.update(batch)`（如 `IPPOLearner` / `MAPPOLearner`）根据 PPO 损失函数更新策略/价值网络。
4. **日志与评估**：在 `Trainer.run` 中按 `log_interval` 打印训练指标；在脚本中使用 `Evaluator` 定期在若干 episode 上评估策略。

---

## 运行测试

如果你安装了 `pytest`：

```bash
pytest
```

`tests/` 目录中包含对 buffer、MAC、policy、centralized critic、GAE 等模块的单元测试，以及 IPPO 与 MAPPO 的 smoke test，用于快速验证实现是否正常工作。

---

## 自行扩展的建议

- **新增环境**：参考 `ToyUavEnv` 的接口与适配方式，在 `marl_uav/envs/adapters/` 中增加自定义环境 wrapper，并在 `configs/env/` 中添加相应 YAML。
- **新增算法**：在 `marl_uav/learners/` 中继承 `BaseLearner` 实现自己的 `update`，并在 `configs/algo/` 中写好超参数 YAML，最后在 `scripts/train.py` 的 `build_learner` 中加入分支。
- **新增模型结构**：在 `modules/encoders/`、`modules/heads/` 中添加新的网络模块，在 `policies/` 中组合成策略，并在 `configs/model/` 中配置好 `type` 与超参数。

如果你在扩展或运行过程中遇到具体问题，可以随时记录你使用的命令、配置文件和报错信息，方便快速定位。


# marl_uav_lib

面向 3v1 无人机追逃实验的 MARL 研究代码。当前主线是 **Dream-MAPPO**，支持两类仿真后端：

- **Genesis**：新的主要无人机仿真后端，使用 Genesis DroneEntity 和 RPM 控制。
- **PyFlyt**：保留的兼容后端，用于复现实验和对照。

本文只保留与 Dream-MAPPO、Genesis、PyFlyt、ex1/ex2 实验直接相关的内容。

## 主要实验

**ex1：结构感知围捕**

- 任务名：`pursuit_evasion_3v1_ex1`
- 重点：结构感知观测、围捕结构奖励、覆盖/聚拢/角度指标。
- PyFlyt 配置：`configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml`
- Genesis 配置：`configs/experiment/pursuit_evasion_dream_mappo_3v1_genesis.yaml`

**ex2：带圆柱障碍物的结构围捕**

- 任务名：`pursuit_evasion_3v1_ex2`
- 重点：在 ex1 基础上加入圆柱障碍物、障碍物观测、障碍物避让流形和碰撞惩罚。
- PyFlyt 配置：`configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2.yaml`
- Genesis 配置：`configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2_genesis.yaml`

Genesis 版 ex1/ex2 的训练参数与对应 PyFlyt 配置保持一致，主要差异仅是 `env` 指向 `configs/env/genesis_3v1.yaml`。

## 后端配置

**Genesis 后端**

- 环境配置：`configs/env/genesis_3v1.yaml`
- 后端选择：`backend: genesis`
- 动作空间：连续 `[vx, vy, yaw_rate, vz]`，动作上下界与 PyFlyt 3v1 保持一致。
- 控制路径：task 输出高层速度 setpoint，`GenesisBackend` 转换为四旋翼 RPM。
- Genesis 是可选依赖，只在请求 Genesis 后端时导入；未安装 Genesis 时 PyFlyt 路径不受影响。

**PyFlyt 后端**

- 环境配置：`configs/env/pyflyt_3v1.yaml`
- 后端选择：旧配置中的 `backend:` 字典仍按 PyFlyt 解释。
- 默认训练关闭渲染：`render: False`。

## 训练命令

在仓库根目录运行：

```bash
cd e:\lyn\year_1\research\marl_uav_lib
```

**Genesis + Dream-MAPPO ex1**

```bash
python scripts/train.py --train-config configs/experiment/pursuit_evasion_dream_mappo_3v1_genesis.yaml
```

**Genesis + Dream-MAPPO ex2**

```bash
python scripts/train.py --train-config configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2_genesis.yaml
```

兼容入口：

```bash
python scripts/train.py --train-config configs/train/genesis_3v1.yaml
```

`configs/train/genesis_3v1.yaml` 当前等价于 Genesis ex2 训练入口。

**PyFlyt + Dream-MAPPO ex1**

```bash
python scripts/train.py --train-config configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml
```

**PyFlyt + Dream-MAPPO ex2**

```bash
python scripts/train.py --train-config configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2.yaml
```

## Guarded 训练入口

`scripts/guarded_dream_mappo.py` 会为每次训练生成独立运行目录，并记录 stdout、monitor summary、checkpoint 和 TensorBoard 日志。

默认运行 Genesis + Dream-MAPPO ex2：

```bash
python scripts/guarded_dream_mappo.py
```

Genesis ex1：

```bash
python scripts/guarded_dream_mappo.py --experiment ex1 --backend genesis
```

Genesis ex2：

```bash
python scripts/guarded_dream_mappo.py --experiment ex2 --backend genesis
```

PyFlyt ex1 / ex2：

```bash
python scripts/guarded_dream_mappo.py --experiment ex1 --backend pyflyt
python scripts/guarded_dream_mappo.py --experiment ex2 --backend pyflyt
```

常用调试参数：

```bash
python scripts/guarded_dream_mappo.py --experiment ex2 --backend genesis --rollout-steps 128 --skip-eval
```

## 评估命令

`scripts/eval.py` 使用与训练相同的环境工厂，因此传入 Genesis 实验配置时会创建 Genesis 后端；传入 PyFlyt 配置时会创建 PyFlyt 后端。

**Genesis ex1**

```bash
python scripts/eval.py --config configs/experiment/pursuit_evasion_dream_mappo_3v1_genesis.yaml --seed 205 --train-seed 205 --episodes 20
```

**Genesis ex2**

```bash
python scripts/eval.py --config configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2_genesis.yaml --seed 205 --train-seed 205 --episodes 20
```

**PyFlyt ex1 / ex2**

```bash
python scripts/eval.py --config configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml --seed 205 --train-seed 205 --episodes 20
python scripts/eval.py --config configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2.yaml --seed 205 --train-seed 205 --episodes 20
```

如果 checkpoint 不在默认目录，可以显式指定：

```bash
python scripts/eval.py --config configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2_genesis.yaml --ckpt results/pursuit_evasion_dream_mappo_3v1_ex2_genesis/checkpoints/205/best.pt
```

## 日志和 Checkpoint

普通 `scripts/train.py` 训练默认输出到：

```text
results/<train_config_stem>/
```

例如 Genesis ex2：

```text
results/pursuit_evasion_dream_mappo_3v1_ex2_genesis/
  tb_/205/
  checkpoints/205/
    latest.pt
    best.pt
```

其中：

- TensorBoard 日志：`results/<run>/tb_/<seed>/`
- checkpoint：`results/<run>/checkpoints/<seed>/`
- `best.pt`：按 `train/avg_return` 保存的最佳模型。
- `latest.pt`：最近一次保存的模型。

Guarded 训练默认输出到：

```text
results/guarded_dream_mappo_runs/<config_stem>_<timestamp>/
```

目录内包含：

```text
train_stdout.log
train_monitor_summary.json
run_summary.json
eval_stdout.log              # 未使用 --skip-eval 时生成
tb_/<seed>/
checkpoints/<seed>/
```

## Smoke Test

Genesis 未安装时，Genesis smoke test 会 skip。

```bash
python scripts/smoke_test_genesis_3v1.py
pytest tests/test_genesis_backend_smoke.py -q
```

旧 PyFlyt 路径不会因为 Genesis 未安装而失败。训练 Genesis 前请先确认本机 Genesis 可用，并按你的机器情况设置 `configs/env/genesis_3v1.yaml` 中的：

- `backend_config.device: gpu | cpu`
- `backend_config.headless: true | false`
- `backend_config.dt`
- `backend_config.hover_rpm`
- `backend_config.max_rpm`

## 常见问题

**Genesis 初始化时出现 UnicodeEncodeError / 乱码，且 TensorBoard 目录为空**

这是 Windows 子进程 stdout/stderr 使用 GBK 编码导致的，Genesis banner 中的 Unicode 字符会触发日志写入失败。当前训练入口已经强制设置 UTF-8：

- `scripts/train.py`
- `scripts/eval.py`
- `scripts/guarded_dream_mappo.py`
- `scripts/smoke_test_genesis_3v1.py`

请重新运行训练命令。旧的失败 run 目录不会自动补写 TensorBoard event；新的 run 会在环境创建前写入 `run/alive`，路径仍为：

```text
results/guarded_dream_mappo_runs/<config_stem>_<timestamp>/tb_/<seed>/
```

## 关键配置文件

- `configs/env/genesis_3v1.yaml`：Genesis DroneEntity 后端、RPM 控制器、动作范围。
- `configs/env/pyflyt_3v1.yaml`：PyFlyt 兼容后端、动作范围、渲染/频率设置。
- `configs/algo/dream_mappo.yaml`：Dream-MAPPO 的 PPO/MAPPO 训练超参。
- `configs/model/dream_mappo_centralized.yaml`：Dream-MAPPO actor/critic 网络与几何动作头参数。
- `configs/experiment/pursuit_evasion_dream_mappo_3v1_genesis.yaml`：Genesis ex1。
- `configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2_genesis.yaml`：Genesis ex2。
- `configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml`：PyFlyt ex1。
- `configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2.yaml`：PyFlyt ex2。

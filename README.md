# marl_uav_lib

面向 3v1 无人机追逃实验的研究代码库，实现 **结构保持型多无人机协同围捕框架（Structure-Preserving Cooperative Encirclement, SCE）**。

> **重要说明：** 本项目**不**提出新的 MAPPO 优化器。MAPPO / actor–critic 仅作为**标准学习型闭环执行后端（execution backend）**。主要研究贡献在于：可变形围捕流形生成、基于运输的角色分配（论文目标为熵正则最优传输；见 `docs/paper/TODO.md` 与当前实现的差异）、拓扑感知的结构目标，以及残差技能保持微调。详见 [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md) 与 [`docs/paper/`](docs/paper/)。

**论文工作标题：** *Structure-preserving cooperative encirclement through deformable encirclement manifold generation and transport-based role allocation.*

## 方法流水线（叙事）

```text
逃逸者状态 + 环境上下文
  → 可变形闭合曲线围捕流形
  → 流形上的目标槽位采样
  → 运输式角色分配
  → 角色条件参考 / 拓扑感知结构引导
  → 基于 RL 的闭环执行策略（本仓库：MAPPO 风格后端）
```

配置与代码中仍保留 **`dream_mappo`** 等历史命名（兼容训练脚本）；在论文与文档中应理解为 **「完整 SCE 框架 + MAPPO 执行后端」**，而非新的 RL 算法。

## 仿真后端

- **Genesis**：主要无人机仿真后端（`configs/env/genesis_3v1.yaml`）。
- **PyFlyt**：兼容后端，用于复现与对照（`configs/env/pyflyt_3v1.yaml`）。

## 实验场景（与论文 E1–E7 对应）

| 论文块 | 仓库现状 | 入口 |
|--------|----------|------|
| **E1** 开敞空间围捕兼容性 | `configs/benchmark/e1_1_open_space_suite.yaml` | `scripts/benchmark_e1_1_open_space.py` |
| **E2** 障碍物环境 | `pursuit_evasion_3v1_ex2` | `configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2.yaml` |
| **E3–E4** 窄通道 / 多出口 | 待建场景 | 见 `docs/paper/TODO.md` |
| **E5–E7** 消融 / 结构指标 / 运行时 | 部分脚本已有 | `docs/paper/03_experiments.md` |

**ex1（结构感知围捕）**

- 任务：`pursuit_evasion_3v1_ex1`
- PyFlyt：`configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml`
- Genesis：`configs/experiment/pursuit_evasion_dream_mappo_3v1_genesis.yaml`

**ex2（圆柱障碍 + 可变形流形）**

- 任务：`pursuit_evasion_3v1_ex2`
- PyFlyt：`configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2.yaml`
- Genesis：`configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2_genesis.yaml`

## E1.1 主基准（开敞空间）

对比 **围捕框架实例与基线**，而非「MAPPO 变体之争」：

```bash
cd e:\lyn\year_1\research\marl_uav_lib

# 生成配置 + 训练 + 评估 + 汇总 CSV
python scripts/benchmark_e1_1_open_space.py --mode all

# 仅评估已有 checkpoint
python scripts/benchmark_e1_1_open_space.py --mode eval --methods dream_mappo_full mappo
```

方法键见 `configs/benchmark/e1_1_open_space_suite.yaml`：

- **`sce`**（推荐先跑）：可变形流形 + 熵正则 OT 角色分配 + 比例槽位跟踪（最简执行后端，无需训练）
- **`dream_mappo_full`**：同上几何/角色栈 + MAPPO 执行后端（需训练）
- 对照：`mappo`、`mappo_bc`、启发式基线等

```bash
python scripts/benchmark_e1_1_open_space.py --mode eval --methods sce oracle_slot mappo
```

结果目录：`results/e1_1_open_space_pyflyt/`。

## 训练命令（单实验）

**Genesis + SCE 框架（ex2）**

```bash
python scripts/train.py --train-config configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2_genesis.yaml
```

**PyFlyt + SCE 框架（ex1）**

```bash
python scripts/train.py --train-config configs/experiment/pursuit_evasion_dream_mappo_3v1.yaml
```

**E1.1 完整框架实例（开敞空间 benchmark）**

```bash
python scripts/train.py --train-config configs/experiment/e1_1_open_space_pyflyt_dream_mappo_full.yaml
```

**MAPPO 执行后端基线（无 ex1 结构模块）**

```bash
python scripts/train.py --train-config configs/experiment/e1_1_open_space_pyflyt_mappo.yaml
```

兼容入口 `configs/train/genesis_3v1.yaml` 当前等价于 Genesis ex2。

## Guarded 训练入口

`scripts/guarded_dream_mappo.py` 记录独立运行目录（历史脚本名保留）：

```bash
python scripts/guarded_dream_mappo.py --experiment ex2 --backend genesis
```

## 评估

```bash
python scripts/eval.py --config configs/experiment/pursuit_evasion_dream_mappo_3v1_ex2_genesis.yaml --seed 205 --train-seed 205 --episodes 20
```

结构指标后处理：`python scripts/pursuit_episode_log_stats.py`（逃逸角、角色稳定性等）。

## 日志与 Checkpoint

```text
results/<train_config_stem>/
  tb_/<seed>/
  checkpoints/<seed>/latest.pt
  checkpoints/<seed>/best.pt
```

## 关键配置（框架语义）

| 语义 | 配置文件 |
|------|----------|
| 执行后端超参 | `configs/algo/dream_mappo.yaml`, `configs/algo/mappo.yaml` |
| 流形 + 角色条件策略头 | `configs/model/dream_mappo_centralized.yaml` |
| E1.1 完整框架 | `configs/experiment/e1_1_open_space_pyflyt_dream_mappo_full.yaml` |
| MAPPO 基线 | `configs/experiment/e1_1_open_space_pyflyt_mappo.yaml` |
| 残差微调 | `configs/experiment/e1_1_open_space_pyflyt_mappo_bc.yaml` |

推荐在实验 YAML 注释中使用的标签：`framework: structure_preserving_encirclement`, `execution_backend: mappo`, `manifold: closed_curve`, `role_allocator: nearest`（或未来的 `entropic_ot`）。

## Smoke Test

```bash
python scripts/smoke_test_genesis_3v1.py
pytest tests/test_genesis_backend_smoke.py -q
```

## 文档

- [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md) — 框架叙事与命名对照
- [`docs/paper/`](docs/paper/) — 摘要、方法、实验大纲（Markdown 草稿）

## 常见问题

**Genesis 在 Windows 上出现 UnicodeEncodeError**

训练入口已强制 UTF-8：`scripts/train.py`, `scripts/eval.py`, `scripts/guarded_dream_mappo.py`。失败 run 的 TensorBoard 需重新训练生成。

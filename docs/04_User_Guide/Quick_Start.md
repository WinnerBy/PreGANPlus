# 快速开始

**创建日期**: 2026-01-14

---

## 🚀 快速开始

### 前置要求

- Python 3.8+
- Conda环境管理工具
- 足够的磁盘空间（建议至少10GB）

### 第一步：环境配置

```bash
# 激活 conda 环境（当前实验使用 pregan_env）
conda activate pregan_env

# 使用 python 而非 python3
```

### 第二步：运行实验

#### 方法1: 使用三阶段脚本（推荐）

```bash
# 阶段1：数据生成（推荐 400 步×8 容器）
python scripts/stage1_data_generation.py --steps 400

# 阶段2：模型训练（单方法或 --method-set gan/ablation/traditional/all）
python scripts/stage2_model_training.py --method PreGAN

# 阶段3：推理测试（支持 -y 遇错继续下一方法）
python scripts/stage3_inference_testing.py --method-set all -y --runs 10
```

详见 [scripts/README.md](../../scripts/README.md)。

#### 方法2: 手动运行单个方法

通过环境变量或脚本参数指定方法（见 `scripts/stage2_model_training.py`、`stage3_inference_testing.py`）；或编辑 `main.py` 切换 Recovery 类后运行：

```bash
conda activate pregan_env
python main.py -e "" -m 0
```

---

## 📊 完整实验流程（三阶段）

```bash
conda activate pregan_env

# 阶段1：数据生成
python scripts/stage1_data_generation.py --steps 400

# 阶段2：模型训练（单方法或 --method-set）
python scripts/stage2_model_training.py --method PreGAN

# 阶段3：推理测试（可加 -y、--runs）
python scripts/stage3_inference_testing.py --method-set all -y
```

详见 [Experimental_Workflow](../02_Experiments/Experimental_Workflow.md)、[scripts/README.md](../../scripts/README.md)。

---

## 📁 结果文件位置

- **Stage1 日志**: `experiment_logs/stage1/`
- **Stage2 日志**: `experiment_logs/stage2/`
- **Stage3 日志与 CSV**: `experiment_logs/stage3/`（含 `stage3_aggregated_5runs_selected.csv`、`stage3_aggregated_by_method.csv`）
- **当前结果说明**: [Stage3_Results_Analysis](../03_Results/Stage3_Results_Analysis.md)、[docs/README.md](../README.md)

---

## 🔧 配置说明

- **stage1_data_generation.py**: `--steps`（默认见脚本）、`--log-dir`
- **stage2_model_training.py**: `--method` / `--method-set gan|ablation|traditional|all`、`--encoder-only`、`-y`、`--steps`
- **stage3_inference_testing.py**: `--method` / `--method-set`、`--steps`、`--containers-per-step`、`--runs`、`-y`

详见各脚本 `--help` 与 [scripts/README.md](../../scripts/README.md)。

---

## 📈 查看结果

- **Stage3 汇总表**: `experiment_logs/stage3/stage3_aggregated_by_method.csv`、`stage3_aggregated_5runs_selected.csv`
- **解析日志**: `python scripts/parse_stage3_logs.py experiment_logs/stage3 --steps 600 --containers-per-step 10`
- **结果说明**: [Stage3_Results_Analysis](../03_Results/Stage3_Results_Analysis.md)、[SLO_SLA_Metrics_Explanation](../03_Results/SLO_SLA_Metrics_Explanation.md)

---

## ⚠️ 注意事项

1. **环境**: 使用 `pregan_env` conda 环境，命令用 `python` 而非 `python3`
2. **磁盘空间**: 建议至少 10GB 用于日志与 checkpoint
3. **运行时间**: 完整实验较长，建议使用 tmux/screen；Stage3 支持 `-y` 遇错继续下一方法
4. **模型文件**: 首次运行会训练编码器/GAN，checkpoint 在 `checkpoints/`、`checkpointsplus/`、`recovery/ablation_models/`

---

## 🐛 故障排除

### 问题1: 找不到模型文件

**错误**: `FileNotFoundError: ... .ckpt`

**解决**: 首次运行需要训练模型，确保有训练数据在 `recovery/PreGANSrc/data/simulator/`

### 问题2: 内存不足

**解决**: 减少 `--steps` 参数，或减少同时运行的方法数量

### 问题3: 权限错误

**解决**: 确保有写入权限，必要时使用 `chmod` 或 `chown`

---

## 🔗 相关文档

- [安装指南](Installation.md) - 环境配置和依赖安装
- [高级用法](Advanced_Usage.md) - 高级功能和自定义
- [实验流程说明](../02_Experiments/Experimental_Workflow.md) - 详细流程
- [实验参数配置](../02_Experiments/Experimental_Configuration.md) - 参数说明

---

**最后更新**: 2026-01

# PreGANPlus (MAMO-GAN)

**Migration-Aware Multi-Objective GAN for Fault-Tolerant Edge Computing**

<div align="center">
  <a href="https://github.com/WinnerBy/PreGANPlus/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-red.svg" alt="License">
  </a>
  <a>
    <img src="https://img.shields.io/badge/python-3.8%20%7C%203.10-blue.svg" alt="Python">
  </a>
  <a>
    <img src="https://img.shields.io/badge/PyTorch-2.x-green.svg" alt="PyTorch">
  </a>
</div>

---

## 概述

本项目实现 **MAMO-GAN**（Migration-Aware Multi-Objective GAN），一种用于边缘计算故障容忍的迁移感知多目标生成对抗网络。MAMO-GAN 在 FPE-GAN(PreGAN) 和 TF-GAN(PreGANPlus) 基础上，引入迁移感知生成器与多目标判别器，在少迁移、低能耗与 SLO 满足上取得综合优势。

### 方法命名（代码 ↔ 文档）

| 代码名称 | 文档/论文名称 |
|----------|----------------|
| PreGAN | FPE-GAN (Fault Prediction Encoder GAN) |
| PreGANPlus | TF-GAN (Transformer-based Fault GAN) |
| PreGANPlusEnhanced | MAMO-GAN (Migration-Aware Multi-Objective GAN) |

---

## 主要特性

- **迁移感知 Generator**：迁移成本预测与迁移门控，减少不必要迁移
- **多目标 Discriminator**：同时优化能量、时延、迁移代价
- **三阶段流程**：Stage1 数据生成 → Stage2 模型训练 → Stage3 推理测试
- **消融实验**：NoTransformer / NoGAT / NoMigrationAware / NoMultiObjective，基准为 PreGANPlusEnhanced

---

## 最新实验结果（Stage3 挑选 5 次，课程作业用）

**配置**：600 步×10 容器/步，16 主机；汇总表见 `experiment_logs/stage3/stage3_aggregated_5runs_selected.csv`。

| 方法 | 迁移次数 | 能耗 | SLA 违反(全体创建) |
|------|----------|------|--------------------|
| **PreGANPlusEnhanced (MAMO-GAN)** | **926.00±5.90** | 最低(GAN 系) | 12.83±0.14% |
| PreGANPlus (TF-GAN) | 929.80±8.13 | 略高 | 13.00±0.09% |
| AblationNoTransformer | 932.80±19.59 | — | 12.80±0.19% |
| PreGAN (FPE-GAN) | 942.20±8.66 | 最高(GAN 系) | 13.11±0.23% |
| CMODLB / DFTM / ECLB / PCFT | 见汇总表 | 见汇总表 | 见汇总表 |

**结论**：PreGANPlusEnhanced 在迁移次数与稳定性上综合最优；消融 NoTransformer 相对完整模型体现明显劣势。完整分析与指标说明见 [docs/03_Results/Stage3_Results_Analysis.md](docs/03_Results/Stage3_Results_Analysis.md)。

---

## 快速开始

### 1. 环境

```bash
git clone https://github.com/imperial-qore/PreGANPlus.git
cd PreGANPlus

# Conda（推荐）
conda env create -f environment.yaml
conda activate pregan_env

# 或 pip
pip install -r requirements.txt
```

### 2. 运行实验（三阶段）

```bash
# 阶段1：数据生成（默认 400 步×8 容器）
python scripts/stage1_data_generation.py

# 阶段2：模型训练（如 PreGANPlusEnhanced）
python scripts/stage2_model_training.py --method PreGANPlusEnhanced

# 阶段3：推理测试
python scripts/stage3_inference_testing.py --method PreGANPlusEnhanced
```

**一键全流程**（单方法）：

```bash
python scripts/run_experiment.py --all-stages --methods PreGANPlusEnhanced
```

**详细说明**：见 [docs/04_User_Guide/Quick_Start.md](docs/04_User_Guide/Quick_Start.md)、[scripts/README.md](scripts/README.md)。

---

## 文档

| 入口 | 说明 |
|------|------|
| [docs/README.md](docs/README.md) | 文档总索引（实验设置、Stage1/2/3、指标与论文） |
| [docs/01_Methods/](docs/01_Methods/README.md) | FPE-GAN / TF-GAN / MAMO-GAN 设计 |
| [docs/02_Experiments/](docs/02_Experiments/README.md) | 实验设置、Stage1 数据、Stage2 训练 |
| [docs/03_Results/](docs/03_Results/README.md) | Stage3 结果、SLO/SLA 与故障检测指标说明 |
| [docs/04_User_Guide/](docs/04_User_Guide/README.md) | 安装、快速开始、高级用法 |

---

## 项目结构（简要）

```
PreGANPlus/
├── docs/                    # 文档（方法 / 实验 / 结果 / 用户指南）
├── experiment_logs/         # 实验日志（stage1/2/3）
├── recovery/                # 模型实现
│   ├── PreGAN.py            # FPE-GAN
│   ├── PreGANPlus.py        # TF-GAN
│   ├── PreGANPlusEnhanced.py # MAMO-GAN
│   ├── Ablation.py          # 消融模型
│   └── PreGANSrc/           # 源码与 checkpoint
├── scripts/                 # 实验脚本
│   ├── stage1_data_generation.py
│   ├── stage2_model_training.py
│   ├── stage3_inference_testing.py
│   ├── run_experiment.py
│   └── archived/            # 已归档的旧脚本
├── main.py                  # 主入口
├── environment.yaml         # Conda 环境
└── README.md
```

---

## 引用

- **FPE-GAN**: Tuli et al., "PreGAN: Preemptive Migration Prediction Network for Proactive Fault-Tolerant Edge Computing", 2022
- **TF-GAN**: Tuli et al., "PreGAN+: Semi-Supervised Fault Prediction and Preemptive Migration in Dynamic Mobile Edge Environments", 2024

---

## License

BSD-3-Clause. Copyright (c) 2022, Shreshth Tuli. See LICENSE for details.

---

*最后更新: 2026-01*

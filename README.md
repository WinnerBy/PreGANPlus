<h1 align="center">MAMO-GAN</h1>
<div align="center">
  <strong>Migration-Aware Multi-Objective GAN for Fault-Tolerant Edge Computing</strong>
</div>

<div align="center">
  <a href="https://github.com/imperial-qore/PreGAN/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-red.svg" alt="License">
  </a>
   <a>
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" alt="Python 3.7, 3.8">
  </a>
   <a>
    <img src="https://img.shields.io/badge/PyTorch-1.7.1+-green.svg" alt="PyTorch">
  </a>
</div>

---

## 📋 概述

本项目实现了**MAMO-GAN** (Migration-Aware Multi-Objective GAN)，一种用于边缘计算故障容忍的迁移感知多目标生成对抗网络方法。MAMO-GAN是对FPE-GAN和TF-GAN的改进，通过引入迁移感知机制和多目标优化，在保持能量优势的同时，显著改善了响应时间和SLA违约率。

### 方法命名

- **FPE-GAN** (Fault Prediction Encoder GAN): 原PreGAN方法
- **TF-GAN** (Transformer-based Fault GAN): 原PreGANPlus方法
- **MAMO-GAN** (Migration-Aware Multi-Objective GAN): 我们的改进方法（原PreGANPlusEnhanced）

**注意**: 代码中仍使用原名称（PreGAN, PreGANPlus, PreGANPlusEnhanced），但在文档中使用新命名。

---

## 🎯 主要特性

- **迁移感知Generator**: 通过迁移成本预测和迁移门控机制，减少不必要的任务迁移
- **多目标Discriminator**: 同时优化能量、响应时间、迁移成本三个目标
- **多目标训练**: 平衡多个优化目标，避免单一目标过度优化
- **迁移控制机制**: cooldown_period和max_migrations_per_step进一步控制迁移

---

## 📊 最新实验结果（2026-01-12 至 2026-01-13）

**实验规模**: 116次运行（传统方法各16次，PreGAN/PreGANPlus各16次，PreGANPlusEnhanced 36次）

### 最终选择结果摘要

| 方法 | 迁移次数 | 能耗(kWh) | 响应时间(s) | SLA违规 |
|------|---------|----------|-----------|---------|
| **MAMO-GAN** (PreGANPlusEnhanced) | 173 | 1959.52 | 219.63 | 95 |
| **TF-GAN** (PreGANPlus) | 157 | 1983.01 | 240.42 | 113 |
| **FPE-GAN** (PreGAN) | 165 | 1983.40 | 214.02 | 104 |
| CMODLB | 164 | 2000.81 | 263.80 | 102 |
| DFTM | 210 | 1998.10 | 257.85 | 105 |
| ECLB | 407 | 1929.97 | 236.33 | 102 |
| PCFT | 1043 | 2119.40 | 226.55 | 130 |

**关键结论**
- ✅ **MAMO-GAN vs TF-GAN**: 能耗 -1.18%，响应时间 -8.65%，SLA违规 -15.93%
- ✅ **FPE-GAN vs 传统方法**: 迁移数显著减少（21%-84%），响应时间优于大部分传统方法
- 📊 **完整报告和图表**: 参见 `final_results/` 目录

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/imperial-qore/PreGANPlus.git
cd PreGANPlus

# 安装依赖
sudo apt -y update
python3 -m pip --upgrade pip
python3 -m pip install matplotlib scikit-learn
python3 -m pip install -r requirements.txt
python3 -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html
export PATH=$PATH:~/.local/bin

# 激活conda环境
conda activate pregan
```

### 2. 运行实验

#### 方法1: 使用一键脚本（推荐）⭐

```bash
# 运行完整论文实验流程（阶段1→阶段2+3→阶段2→阶段4）
bash scripts/run_paper_experiment.sh
```

**实验流程**：
1. 阶段1：数据收集（1000步）
2. 阶段2+3：编码器训练 + GAN训练（1200步）
   - PreGAN, PreGANPlus, PreGANPlusEnhanced
3. 阶段2：CMODLB编码器训练（1200步）
4. 阶段4：测试评估（100步，所有方法对比）

#### 方法2: 分阶段运行

```bash
# 阶段1：数据收集
python3 scripts/paper_experiment_stage1_data_collection.py
python3 main.py -e "" -m 0

# 阶段2+3：编码器训练 + GAN训练（某个方法）
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python3 main.py -e "" -m 0

# 阶段4：测试评估（某个方法）
python3 scripts/paper_experiment_stage4_testing.py --method PreGAN
python3 main.py -e "" -m 0
```

**详细说明**：参见 [实验运行说明.md](实验运行说明.md)

---

## 📚 文档

详细的文档请参考 [docs/](docs/) 目录：

- **[使用指南](docs/User_Guide.md)** - 如何运行实验和使用脚本
- **[实现指南](docs/Implementation_Guide.md)** - 模型架构和实现细节
- **[方法命名指南](docs/Method_Naming_Guide.md)** - 方法命名说明
- **[架构对比](docs/Paper/Architecture_Comparison.md)** - 三种方法的详细架构对比
- **[论文故事线](docs/Paper/Paper_Storyline_Final.md)** - 论文故事线梳理
- **[实验结果](docs/Experiments/)** - 实验分析和结果

---

## 🏗️ 项目结构

```
PreGANPlus/
├── docs/                          # 文档
│   ├── User_Guide.md             # 使用指南
│   ├── Implementation_Guide.md   # 实现指南
│   ├── Experiments/              # 实验分析
│   └── Paper/                    # 论文素材
├── final_results/                 # ✨ 最终实验结果
│   ├── data/                     # 最终选中的实验数据
│   ├── logs/                     # 最终选中的日志文件
│   ├── summary/                  # 汇总数据和报告
│   ├── plots/                    # 最终对比图表
│   └── README.md                 # 最终结果说明
├── experiment_data/               # 原始实验数据（所有运行）
├── experiment_logs/               # 原始实验日志（所有运行）
├── recovery/                      # 恢复/模型实现
│   ├── PreGAN.py                 # FPE-GAN
│   ├── PreGANPlus.py             # TF-GAN
│   ├── PreGANPlusEnhanced.py     # MAMO-GAN
│   └── PreGANSrc/                # 模型源代码与ckpt
├── scripts/                     # 实验脚本目录
│   ├── paper_experiment_stage1_data_collection.py   # 阶段1 数据收集
│   ├── paper_experiment_stage2_encoder_training.py  # 阶段2 编码器训练
│   ├── paper_experiment_stage3_gan_training.py      # 阶段3 GAN训练
│   ├── paper_experiment_stage4_testing.py           # 阶段4 测试评估
│   ├── run_paper_experiment.sh                      # 一键全流程脚本
│   ├── generate_final_plots.py                      # 生成最终对比图表
│   └── ...                                         # 其他分析脚本
├── main.py                                      # 主入口
└── README.md
```

---

## 🔧 配置说明（当前上线配置）

**训练权重（MAMO-GAN最新训练）**  
`energy_weight = 0.004`，`response_time_weight = 0.14`，`migration_cost_weight = 0.04`

**推理控制**  
`strict_migration_limit = 175`，`max_migrations_per_step = 2`，`cooldown_period = 8`，`migration_cost_threshold = 100`

> 如需冲刺能耗-3%目标，可增大 energy_weight / migration_cost_weight，并适当降低 strict_migration_limit 以抑制迁移数。

---

## 📈 实验结果

### 方案6最佳配置结果

**配置**: 
- `energy_weight = 0.3`
- `response_time_weight = 0.3`
- `migration_cost_weight = 0.4`
- `migration_cost_threshold = 130`
- `cooldown_period = 3`
- `max_migrations_per_step = 3`

**结果**（3次重复实验平均值 vs TF-GAN）:
- 响应时间: **-2.70%** ⭐⭐⭐⭐
- 总能量: **-2.76%** ⭐⭐⭐⭐
- SLA违约率: **6.14%** ⚠️
- 迁移次数: **+23.85%** ⚠️

**详细分析**: 参见 [docs/Experiments/Repeat_Experiments_Analysis.md](docs/Experiments/Repeat_Experiments_Analysis.md)

---

## 🎯 核心创新

1. **迁移感知Generator**: 引入迁移成本预测和迁移门控机制，减少不必要的任务迁移
2. **多目标Discriminator**: 同时优化能量、响应时间、迁移成本三个目标
3. **多目标训练策略**: 平衡多个优化目标，避免单一目标过度优化
4. **迁移控制机制**: cooldown_period和max_migrations_per_step进一步控制迁移

---

## 📝 引用

如果使用本项目，请引用相关论文：

- **FPE-GAN**: Tuli et al., "PreGAN: Preemptive Migration Prediction Network for Proactive Fault-Tolerant Edge Computing", 2022
- **TF-GAN**: Tuli et al., "PreGAN+: Semi-Supervised Fault Prediction and Preemptive Migration in Dynamic Mobile Edge Environments", 2024
- **MAMO-GAN**: (待发表)

---

## 📄 License

BSD-3-Clause. 
Copyright (c) 2022, Shreshth Tuli.
All rights reserved.

See License file for more details.

---

## 🔗 相关链接

- **原始项目**: [PreGAN](https://github.com/imperial-qore/PreGAN)
- **PreGAN+论文**: [PreGAN+](https://github.com/imperial-qore/PreGANPlus)
- **文档**: [docs/](docs/)
- **Contact**: Shreshth Tuli ([@shreshthtuli](https://github.com/shreshthtuli))

---

---

## 📊 实验结果详情

详细的实验结果、对比图表和分析报告请参考：
- **最终结果目录**: `final_results/` - 包含所有最终选中的数据、日志、图表和报告
- **原始实验数据**: `experiment_data/` - 所有stage4实验的原始数据
- **原始实验日志**: `experiment_logs/` - 所有stage4实验的原始日志

---

*最后更新: 2026-01-14*

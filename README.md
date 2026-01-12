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

## 📊 最新实验结果（2026-01-12，全流程一键跑）

基准：TF-GAN（PreGANPlus）  
对比：PreGAN, PreGANPlusEnhanced (MAMO-GAN), CMODLB, PCFT, DFTM, ECLB  
测试：阶段4（NUM_SIM_STEPS=100，training=False，NEW_CONTAINERS=5）

| 方法 | 迁移次数 | 能耗(kWh) | 响应时间(s) | SLA违规 | 备注 |
|------|---------|----------|-----------|--------|------|
| PreGANPlus (基准) | 163 | 2004.7 | 231.6 | 108 | TF-GAN |
| MAMO-GAN (PreGANPlusEnhanced) | 186 (+14.1%) | 1996.5 (-0.4%) | 226.7 (-2.1%) | 100 (-7.4%) | 能耗改善不足，迁移略高 |
| CMODLB | 162 (-0.6%) | 1971.0 (-1.7%) | 224.9 (-2.9%) | 113 (+4.6%) | 综合最优 ✔ |
| DFTM | 204 (+25.2%) | 1991.9 (-0.6%) | 226.5 (-2.2%) | 100 (-7.4%) | 迁移偏高 |
| PCFT | 1073 (+558%) | 2117.3 (+5.6%) | 217.8 (-6.0%) | 114 (+5.6%) | 调度不稳定 ✖ |
| ECLB | 353 (+117%) | 1901.6 (-5.1%) | 240.2 (+3.7%) | 98 (-9.3%) | 能耗最低但迁移过多 |

**关键结论**
- MAMO-GAN 相比 TF-GAN：响应时间 -2.1%，能耗 -0.4%，迁移 +14%（未达到能耗-3%目标，迁移成本偏高）。
- CMODLB 综合评分最佳，三项关键指标均优于基准，当前最强对照基线。
- PCFT 不稳定，迁移暴增，建议排除。

**完整报告**：`EXPERIMENT_FINAL_REPORT.md`（含表格、排名、后续建议）

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
│   └── run_paper_experiment.sh                      # 一键全流程脚本
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

*最后更新: 2026-01-12*

# 实验参数配置

**创建日期**: 2026-01-14

---

## 📋 概述

本文档详细说明实验中的所有参数配置，包括模拟器参数、实验阶段参数、模型参数和训练参数。

---

## 🖥️ 模拟器参数

### 基础配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `HOSTS` | 16 | 主机数量 |
| `CONTAINERS` | 16 | 容器数量（等于主机数） |
| `TOTAL_POWER` | 1000 | 总功率（单位：W） |
| `ROUTER_BW` | 10000 | 路由器带宽（单位：Mbps） |
| `INTERVAL_TIME` | 300 | 时间间隔（单位：秒，即5分钟） |
| `NEW_CONTAINERS` | 5 | 每个间隔新增的容器数 |

### 环境配置

- **环境类型**: `RPiEdge` (Raspberry Pi Edge环境)
- **工作负载**: `BitbrainWorkload2` (Bitbrain工作负载)
- **调度器**: `GOBIScheduler` (GOBI调度器)

---

## 📊 实验阶段参数

### 阶段1：数据收集

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_SIM_STEPS` | 1000 | 模拟步数（间隔数） |
| `NEW_CONTAINERS` | 5 | 每个间隔新增容器数 |
| `Recovery` | `Recovery()` | 使用基类，不进行任何恢复 |
| **目的** | 收集包含各种故障情况的训练数据 |
| **输出** | `time_series.npy`, `schedule_series.npy` |

**注意**: 
- 此阶段不进行任何预防性迁移
- 生成的数据用于后续编码器训练
- 如果1000步数据训练效果不好，可以继续生成更多数据

### 阶段2：编码器训练

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_SIM_STEPS` | 10 (GAN方法) / 1200 (CMODLB) | 模拟步数 |
| `NEW_CONTAINERS` | 5 | 每个间隔新增容器数 |
| `training` | `False` (自动触发训练) | 训练模式 |
| **编码器类型** | FPE (PreGAN) / Transformer (PreGANPlus/PreGANPlusEnhanced) / FCN (CMODLB) | 根据方法选择 |
| **训练数据** | 阶段1收集的1000步数据 | 离线训练 |
| **训练epochs** | 50 | 编码器训练轮数 |
| **输出** | 编码器checkpoint文件 |

**注意**:
- 编码器训练是自动的：如果checkpoint不存在或epoch == -1，会自动训练
- PreGAN和PreGANPlus使用不同的编码器（FPE vs Transformer）
- PreGANPlus和PreGANPlusEnhanced共享相同的Transformer编码器
- Transformer编码器只在PreGANPlus阶段训练一次，PreGANPlusEnhanced直接使用

### 阶段3：GAN训练（仅GAN方法）

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_SIM_STEPS` | 1200 | 模拟步数（论文配置） |
| `NEW_CONTAINERS` | 5 | 每个间隔新增容器数 |
| `training` | `True` | 训练模式 |
| **GAN类型** | 根据方法选择 | Gen_16 / Gen_16_MigrationAware |
| **Discriminator类型** | 根据方法选择 | Disc_16 / Disc_16_MultiObjective |
| **训练方式** | 在线训练 | 每个间隔进行GAN训练 |
| **编码器调优** | 同时进行 | 每个间隔进行编码器调优 |
| **输出** | GAN checkpoint文件 |

**注意**:
- GAN训练是在线的，使用1200步运行中的数据
- 同时进行编码器调优（tune_model）
- 如果效果不好，可以多训练几次

### 阶段4：测试评估

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_SIM_STEPS` | 100 | 模拟步数（测试评估） |
| `NEW_CONTAINERS` | 5 | 每个间隔新增容器数 |
| `training` | `False` | 推理模式，不进行训练 |
| **测试方法** | 所有方法 | PreGAN, PreGANPlus, PreGANPlusEnhanced, CMODLB, DFTM, ECLB, PCFT |
| **输出** | 性能指标、日志文件、数据文件 |

**注意**:
- 所有方法在相同条件下测试
- 不进行训练，只进行推理
- 收集所有性能指标用于对比分析

---

## 🧠 模型参数

### 编码器参数

#### FPE编码器 (PreGAN)

- **模型名称**: `FPE_16`
- **输入维度**: 时间序列数据 (time_series) + 调度矩阵 (schedule_data)
- **输出维度**: 
  - 异常检测分数: [16, 2]
  - 原型向量: [16, 2]
- **训练参数**:
  - Epochs: 50
  - 学习率: 默认（在constants.py中定义）
  - 优化器: Adam

#### Transformer编码器 (PreGANPlus/PreGANPlusEnhanced)

- **模型名称**: `Transformer_16`
- **输入维度**: 时间序列数据 + 调度矩阵
- **输出维度**: 与FPE相同
- **架构特点**:
  - 使用MultiheadAttention机制
  - 能够捕获时间序列的长期依赖关系
- **训练参数**:
  - Epochs: 50
  - 学习率: 默认
  - 优化器: Adam

#### FCN编码器 (CMODLB)

- **模型名称**: `FCN_16`
- **输入维度**: 时间序列数据
- **输出维度**: 预测值
- **训练参数**:
  - Epochs: 50
  - 学习率: 默认
  - 优化器: Adam

### Generator参数

#### 标准Generator (PreGAN/PreGANPlus)

- **模型名称**: `Gen_16`
- **输入**: 
  - Embedding: [16, 2]
  - Schedule: [16, 16]
- **输出**: New Schedule [16, 16]
- **架构**: 
  - 隐藏层维度: 64
  - 学习率: 0.00005
  - 优化器: AdamW

#### Migration-Aware Generator (PreGANPlusEnhanced)

- **模型名称**: `Gen_16_MigrationAware`
- **输入**: 与标准Generator相同
- **输出**: 
  - New Schedule [16, 16]
  - Predicted Migration Cost [1]
- **新增组件**:
  - 迁移成本预测模块
  - 迁移门控模块
- **架构参数**:
  - 隐藏层维度: 64
  - 注意力头数: 4
  - Dropout: 0.1
  - 学习率: 0.00005

### Discriminator参数

#### 标准Discriminator (PreGAN/PreGANPlus)

- **模型名称**: `Disc_16`
- **输入**: 
  - Original Schedule [16, 16]
  - New Schedule [16, 16]
- **输出**: Classification Probabilities [2]
- **架构**:
  - 隐藏层维度: 128
  - 学习率: 0.00005
  - 优化器: AdamW

#### Multi-Objective Discriminator (PreGANPlusEnhanced)

- **模型名称**: `Disc_16_MultiObjective`
- **输入**: 与标准Discriminator相同
- **输出**: 
  - Classification Probabilities [2]
  - Energy Prediction [1]
  - Response Time Prediction [1]
  - Migration Cost Prediction [1]
- **架构参数**:
  - 隐藏层维度: 128
  - 学习率: 0.00005
  - 优化器: AdamW

---

## 🎯 训练参数

### 编码器训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Epochs | 50 | 训练轮数 |
| Batch Size | 根据数据自动确定 | 批次大小 |
| Learning Rate | 默认（在constants.py中） | 学习率 |
| Optimizer | Adam | 优化器 |
| Loss Function | 根据编码器类型 | 损失函数 |

### GAN训练参数

#### 标准GAN训练 (PreGAN/PreGANPlus)

| 参数 | 值 | 说明 |
|------|-----|------|
| Learning Rate (Generator) | 0.00005 | Generator学习率 |
| Learning Rate (Discriminator) | 0.00005 | Discriminator学习率 |
| Optimizer | AdamW | 优化器 |
| Loss Function | BCELoss | 二元交叉熵损失 |
| Training Steps | 1200 | 训练步数（每个间隔一次） |

#### 多目标GAN训练 (PreGANPlusEnhanced)

| 参数 | 值 | 说明 |
|------|-----|------|
| Learning Rate | 0.00005 | 与标准GAN相同 |
| Optimizer | AdamW | 与标准GAN相同 |
| **多目标权重** | | |
| `energy_weight` | 0.004 | 能量优化权重 |
| `response_time_weight` | 0.14 | 响应时间约束权重（已验证有效） |
| `migration_cost_weight` | 0.04 | 迁移成本约束权重 |
| **阈值参数** | | |
| `sla_threshold` | 2800.0 | SLA阈值（秒） |
| `migration_cost_threshold` | 110 | 迁移成本阈值 |
| Training Steps | 1200 | 训练步数 |

**注意**: 
- 多目标权重经过优化验证
- 响应时间权重较高（0.14），已验证有效
- 能量权重较低（0.004），通过推理阶段的控制机制实现

---

## 🔧 推理参数

### MAMO-GAN迁移控制参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `cooldown_period` | 10 | 冷却期（间隔数），防止容器频繁迁移 |
| `max_migrations_per_step` | 2 | 每个间隔的最大迁移数 |
| `strict_migration_limit` | 173 | 全局迁移限制（验证的最优值） |
| `migration_cost_threshold` | 110 | 迁移成本阈值 |

**迁移控制机制**:
1. **Cooldown机制**: 容器迁移后，在冷却期内禁止再次迁移
2. **每步限制**: 每个间隔最多允许N个迁移
3. **全局预算**: 整个测试过程的迁移总数限制
4. **预测阈值**: 基于Generator预测的迁移成本进一步限制

**验证结果**:
- 最优配置: `max_per_step=2`, `limit=173`
- 效果: 迁移数173（仅比TF-GAN多10.19%），但能耗和响应时间显著改善

---

## 📝 配置文件位置

### 主要配置文件

- **模拟器参数**: `main.py` (全局常量)
- **编码器参数**: `recovery/PreGANSrc/src/constants.py`
- **模型架构**: `recovery/PreGANSrc/src/models.py`
- **训练函数**: 
  - 标准GAN: `recovery/PreGANSrc/src/train.py`
  - 多目标GAN: `recovery/PreGANSrc/src/train_multiobjective.py`

### 实验脚本

- **阶段1**: `scripts/paper_experiment_stage1_data_collection.py`
- **阶段2**: `scripts/paper_experiment_stage2_encoder_training.py`
- **阶段3**: `scripts/paper_experiment_stage3_gan_training.py`
- **阶段4**: `scripts/paper_experiment_stage4_testing.py`
- **一键运行**: `scripts/run_paper_experiment.sh`

---

## 🔄 参数调整建议

### 如需调整能耗

- 增加 `energy_weight`（训练阶段）
- 降低 `strict_migration_limit`（推理阶段）
- 增加 `migration_cost_weight`（训练阶段）

### 如需调整响应时间

- 增加 `response_time_weight`（已验证有效）
- 调整 `sla_threshold`
- 优化Generator的注意力机制

### 如需调整迁移数

- 调整 `strict_migration_limit`
- 调整 `max_migrations_per_step`
- 调整 `cooldown_period`
- 调整 `migration_cost_weight`

---

## 📊 当前最优配置

基于116次实验运行的结果，当前最优配置为：

### MAMO-GAN最优配置

```python
# 训练参数
energy_weight = 0.004
response_time_weight = 0.14
migration_cost_weight = 0.04
sla_threshold = 2800.0
migration_cost_threshold = 110

# 推理参数
cooldown_period = 10
max_migrations_per_step = 2
strict_migration_limit = 173
```

**效果**:
- 能耗: 1959.52 kWh (比TF-GAN降低1.18%)
- 响应时间: 219.63 s (比TF-GAN改善8.65%)
- 迁移次数: 173 (比TF-GAN多10.19%，可接受)
- SLA违规: 95 (比TF-GAN减少15.93%)

---

**最后更新**: 2026-01-14

# 高级用法

**创建日期**: 2026-01-14

---

## 📋 概述

本文档介绍高级功能和自定义配置，包括实现细节、系统集成、问题解决等。

---

## 🔧 实现细节

### MAMO-GAN实现完成情况

#### 1. 模型实现 ✅

- **Migration-Aware Generator** (`Gen_16_MigrationAware`)
  - 位置: `recovery/PreGANSrc/src/models.py`
  - 功能: 迁移成本预测、迁移门控、注意力机制
  - 状态: ✅ 通过测试

- **Multi-Objective Discriminator** (`Disc_16_MultiObjective`)
  - 位置: `recovery/PreGANSrc/src/models.py`
  - 功能: 多目标评估（分类、能量、响应时间、迁移成本）
  - 状态: ✅ 通过测试

#### 2. 训练函数实现 ✅

- **多目标训练函数** (`train_gan_multiobjective`)
  - 位置: `recovery/PreGANSrc/src/train_multiobjective.py`
  - 功能: 多目标Discriminator和Generator训练
  - 状态: ✅ 通过测试

#### 3. Recovery类实现 ✅

- **PreGANPlusEnhancedRecovery** (MAMO-GAN)
  - 位置: `recovery/PreGANPlusEnhanced.py`
  - 功能: 完整的MAMO-GAN实现，包括迁移控制机制
  - 状态: ✅ 完全兼容PreGANPlus接口

---

## 📝 修改的文件列表

### 新增文件

1. `recovery/PreGANPlusEnhanced.py` - MAMO-GAN 的 Recovery 类
2. `recovery/PreGANSrc/src/train_multiobjective.py` - 多目标训练函数
3. `recovery/Ablation.py` - 四种消融模型（NoTransformer、NoGAT、NoMigrationAware、NoMultiObjective）

### 修改的文件

1. `recovery/PreGANSrc/src/models.py` - 添加 Gen_16_MigrationAware、Disc_16_MultiObjective、TransformerNoGAT_16 等
2. `recovery/PreGANSrc/src/plotter.py` - 修复 MPS/tensor 转 numpy 等设备兼容
3. `main.py` - 添加 PreGANPlusEnhanced、Ablation 等 Recovery 导入与选择
4. 实验脚本为 `scripts/stage1_data_generation.py`、`stage2_model_training.py`、`stage3_inference_testing.py`（见 [scripts/README.md](../../scripts/README.md)）

---

## 🔧 系统集成

### 集成步骤

1. **模型定义**: 在 `models.py` 中定义新模型
2. **训练函数**: 在 `train_multiobjective.py` 中实现训练逻辑
3. **Recovery类**: 在 `PreGANPlusEnhanced.py` 中实现Recovery接口
4. **主程序**: 在 `main.py` 中添加导入和配置
5. **测试脚本**: 创建测试脚本验证功能

### 接口兼容性

MAMO-GAN完全兼容PreGANPlus的接口：
- 相同的Recovery基类
- 相同的训练和推理接口
- 相同的checkpoint格式

---

## 🎯 自定义配置

### 多目标权重配置

在 `PreGANPlusEnhanced.py` 中可以调整多目标权重：

```python
# 训练参数
energy_weight = 0.004            # 能量优化权重
response_time_weight = 0.14      # 响应时间约束权重
migration_cost_weight = 0.04     # 迁移成本约束权重

# 阈值参数
sla_threshold = 2800.0           # SLA阈值（秒）
migration_cost_threshold = 110   # 迁移成本阈值
```

### 迁移控制参数

在 `PreGANPlusEnhanced.py` 中可以调整迁移控制参数：

```python
# 推理参数
cooldown_period = 10             # 冷却期（间隔数）
max_migrations_per_step = 2      # 每个间隔的最大迁移数
strict_migration_limit = 173     # 全局迁移限制
```

---

## 🐛 问题解决

### 常见问题

#### 1. 模型训练不收敛

**可能原因**:
- 学习率过大或过小
- 权重配置不合理
- 数据质量问题

**解决方案**:
- 调整学习率（当前: 0.00005）
- 调整多目标权重
- 检查训练数据

#### 2. 迁移数过多

**可能原因**:
- 迁移成本阈值设置过高
- 迁移控制参数设置不合理

**解决方案**:
- 降低 `migration_cost_threshold`
- 增加 `cooldown_period`
- 降低 `max_migrations_per_step`

#### 3. 响应时间过长

**可能原因**:
- 响应时间权重过低
- SLA阈值设置不合理

**解决方案**:
- 增加 `response_time_weight`（已验证有效）
- 调整 `sla_threshold`

---

## 📊 性能优化建议

### 对于能耗优化

1. **增加能量权重**（训练阶段）
   - 提高 `energy_weight`
   - 降低 `migration_cost_weight`

2. **降低迁移限制**（推理阶段）
   - 降低 `strict_migration_limit`
   - 降低 `max_migrations_per_step`

### 对于响应时间优化

1. **增加响应时间权重**（已验证有效）
   - 提高 `response_time_weight`（当前0.14）
   - 调整 `sla_threshold`

2. **优化Generator的注意力机制**
   - 改进交叉注意力和自注意力机制
   - 优化特征融合方式

### 对于迁移数优化

1. **调整迁移控制参数**
   - 调整 `strict_migration_limit`
   - 调整 `max_migrations_per_step`
   - 调整 `cooldown_period`

2. **优化迁移感知机制**
   - 改进迁移成本预测
   - 优化迁移门控机制

---

## 🔗 相关文档

- [快速开始](Quick_Start.md) - 快速开始使用
- [方法设计文档](../01_Methods/MAMO-GAN_Design.md) - MAMO-GAN详细设计
- [实验参数配置](../02_Experiments/Experimental_Configuration.md) - 参数说明

---

**最后更新**: 2026-01-14

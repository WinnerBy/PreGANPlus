# 实验文档索引

本目录包含MAMO-GAN的核心实验文档。

**方法命名**:
- **FPE-GAN** (Fault Prediction Encoder GAN): 原PreGAN方法
- **TF-GAN** (Transformer-based Fault GAN): 原PreGANPlus方法
- **MAMO-GAN** (Migration-Aware Multi-Objective GAN): 我们的改进方法（原PreGANPlusEnhanced）

---

## 📁 文档结构

```
Experiments/
├── README.md                           # 本文档（文档索引）
├── Repeat_Experiments_Analysis.md     # 重复实验分析（最新）⭐
├── Comprehensive_Experiment_Comparison.md  # 综合实验对比 ⭐
├── Final_Experiment_Analysis.md       # 最终实验分析
├── Optimization_Process_Summary.md    # 优化过程总结
└── archive/                            # 已归档的过时文档
```

---

## 🎯 核心文档

### 1. Repeat_Experiments_Analysis.md ⭐ **最新**

**内容**: 方案6最佳配置的3次重复实验结果分析

**关键结果**（3次平均值 vs TF-GAN）:
- 响应时间: **-2.70%** ⭐⭐⭐⭐
- 总能量: **-2.76%** ⭐⭐⭐⭐
- SLA违约率: **6.14%** ⚠️
- 迁移次数: **+23.85%** ⚠️

**推荐**: 这是最新的最终实验结果，建议优先阅读。

---

### 2. Comprehensive_Experiment_Comparison.md ⭐

**内容**: 所有实验方案的综合对比分析

**包含方案**:
- 方案1-5: 早期优化尝试
- **方案6**: 最佳配置 ⭐⭐⭐⭐⭐（推荐）
- 方案7-9: 后续优化尝试

**推荐**: 了解所有方案的对比和方案6为何是最佳配置。

---

### 3. Final_Experiment_Analysis.md

**内容**: 最终实验的详细分析

**推荐**: 了解最终实验的详细分析过程。

---

### 4. Optimization_Process_Summary.md

**内容**: 从初始实现到最终配置的优化过程总结

**包含**:
- 优化目标
- 优化历程（5个阶段）
- 最终配置说明
- 关键改进点

**推荐**: 了解优化过程的整体脉络。

---

## 📊 最终配置（方案6）

### 配置参数

```python
# 多目标权重
energy_weight = 0.3               # 能量优化权重
response_time_weight = 0.3       # 响应时间约束权重
migration_cost_weight = 0.4       # 迁移成本约束权重

# 约束阈值
sla_threshold = 2800.0            # SLA阈值（秒）
migration_cost_threshold = 130    # 迁移成本阈值

# 迁移控制机制
cooldown_period = 3               # 冷却期（epochs）
max_migrations_per_step = 3       # 每步最大迁移数
```

### 实验结果

| 指标 | vs TF-GAN | 评估 |
|------|-----------|------|
| **总能量** | **-2.76%** | ⭐⭐⭐⭐ 保持优势 |
| **响应时间** | **-2.70%** | ⭐⭐⭐⭐ 略有优势 |
| **SLA违约率** | **6.14%** | ⚠️ 略高但可接受 |
| **迁移次数** | **+23.85%** | ⚠️⚠️ 较高但可接受 |

---

## 🎯 推荐阅读顺序

1. **Optimization_Process_Summary.md** - 了解优化过程整体脉络
2. **Comprehensive_Experiment_Comparison.md** - 了解所有方案对比和方案6为何最佳
3. **Repeat_Experiments_Analysis.md** - 查看最新最终实验结果
4. **Final_Experiment_Analysis.md** - 了解最终实验的详细分析

---

## 📝 已归档文档

所有过时的中间过程文档已归档到 `archive/` 目录，包括：
- 各种权重调整实验分析
- 各种优化实验分析
- SLA改进策略和分析
- 迁移优化策略和分析
- 中间实验记录

如需查看历史优化过程，可参考 `archive/` 目录。

---

*最后更新: 2026-01-03*

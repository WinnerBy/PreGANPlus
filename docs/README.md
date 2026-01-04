# MAMO-GAN 文档索引

欢迎查阅MAMO-GAN项目的文档。本文档提供了所有相关文档的索引和简要说明。

**方法命名**:
- **FPE-GAN** (Fault Prediction Encoder GAN): 原PreGAN方法
- **TF-GAN** (Transformer-based Fault GAN): 原PreGANPlus方法
- **MAMO-GAN** (Migration-Aware Multi-Objective GAN): 我们的改进方法（原PreGANPlusEnhanced）

---

## 📚 文档结构（已精简）

```
docs/
├── README.md                           # 本文档（文档索引）⭐
│
├── User_Guide.md                       # 使用指南（推荐）⭐
├── Implementation_Guide.md             # 实现指南（推荐）⭐
├── Method_Naming_Guide.md              # 方法命名指南
│
├── Experiments/                        # 实验相关文档（5个核心文档）
│   ├── README.md                       # 实验文档索引
│   ├── Repeat_Experiments_Analysis.md  # 重复实验分析（最新）⭐
│   ├── Comprehensive_Experiment_Comparison.md  # 综合实验对比 ⭐
│   ├── Final_Experiment_Analysis.md    # 最终实验分析
│   └── Optimization_Process_Summary.md # 优化过程总结
│
├── Paper/                              # 论文相关文档（2个核心文档）
│   ├── README.md                       # 论文文档索引
│   ├── Architecture_Comparison.md     # 架构对比文档（推荐）⭐
│   └── Paper_Storyline_Final.md       # 最终论文故事线（推荐）⭐
│
└── archive/                             # 已归档的过时文档
```

---

## 🎯 快速导航

### 我想...

#### 🚀 开始使用
- **使用指南** → [User_Guide.md](./User_Guide.md) ⭐ **推荐**
- **实现指南** → [Implementation_Guide.md](./Implementation_Guide.md) ⭐ **推荐**
- **方法命名** → [Method_Naming_Guide.md](./Method_Naming_Guide.md)

#### 📊 查看最新实验结果
- **重复实验分析** → [Experiments/Repeat_Experiments_Analysis.md](./Experiments/Repeat_Experiments_Analysis.md) ⭐ **最新（3次重复实验）**
- **综合实验对比** → [Experiments/Comprehensive_Experiment_Comparison.md](./Experiments/Comprehensive_Experiment_Comparison.md) ⭐
- **优化过程总结** → [Experiments/Optimization_Process_Summary.md](./Experiments/Optimization_Process_Summary.md)

#### 📝 了解架构设计
- **架构对比文档** → [Paper/Architecture_Comparison.md](./Paper/Architecture_Comparison.md) ⭐ **推荐（详细架构对比）**

#### 📖 撰写论文
- **最终论文故事线** → [Paper/Paper_Storyline_Final.md](./Paper/Paper_Storyline_Final.md) ⭐ **推荐**

---

## 📋 文档分类说明

### 核心文档（推荐阅读）

1. **User_Guide.md** - 使用指南
   - 如何运行实验
   - 脚本使用方法
   - 故障排除

2. **Implementation_Guide.md** - 实现指南
   - 模型架构说明
   - 实现细节
   - 配置参数

3. **Method_Naming_Guide.md** - 方法命名指南
   - FPE-GAN, TF-GAN, MAMO-GAN命名说明

### 实验文档 (Experiments/)

包含核心实验分析和优化过程总结（5个文档）。

**核心文档**:
- `Repeat_Experiments_Analysis.md` - 重复实验分析（最新，3次重复实验）⭐
- `Comprehensive_Experiment_Comparison.md` - 综合实验对比 ⭐
- `Optimization_Process_Summary.md` - 优化过程总结

### 论文文档 (Paper/)

包含论文故事线和架构设计文档（2个核心文档）。

**推荐阅读**:
- `Architecture_Comparison.md` - 详细架构对比（FPE-GAN, TF-GAN, MAMO-GAN）⭐
- `Paper_Storyline_Final.md` - 最终论文故事线 ⭐

---

## 📊 最新实验结果摘要

### 重复实验（方案6配置，3次重复实验平均值）

**配置**: MAMO-GAN最佳配置
- `energy_weight = 0.3`
- `response_time_weight = 0.3`
- `migration_cost_weight = 0.4`
- `migration_cost_threshold = 130`
- `cooldown_period = 3`
- `max_migrations_per_step = 3`

| 指标 | vs TF-GAN | 评估 |
|------|-----------|------|
| **总能量** | **-2.76%** | ⭐⭐⭐⭐ 保持优势 |
| **响应时间** | **-2.70%** | ⭐⭐⭐⭐ 略有优势 |
| **SLA违约率** | **6.14%** | ⚠️ 略高但可接受 |
| **迁移次数** | **+23.85%** | ⚠️⚠️ 较高但可接受 |

**详细分析**: 参见 [Experiments/Repeat_Experiments_Analysis.md](./Experiments/Repeat_Experiments_Analysis.md)

---

## 🎯 推荐阅读路径

### 新用户
1. [User_Guide.md](./User_Guide.md) - 了解如何使用
2. [Method_Naming_Guide.md](./Method_Naming_Guide.md) - 了解方法命名
3. [Paper/Architecture_Comparison.md](./Paper/Architecture_Comparison.md) - 了解架构设计
4. [Experiments/Repeat_Experiments_Analysis.md](./Experiments/Repeat_Experiments_Analysis.md) - 查看最新实验结果

### 开发者
1. [Implementation_Guide.md](./Implementation_Guide.md) - 了解实现细节
2. [Paper/Architecture_Comparison.md](./Paper/Architecture_Comparison.md) - 了解架构设计
3. [Experiments/Optimization_Process_Summary.md](./Experiments/Optimization_Process_Summary.md) - 了解优化过程

### 研究者
1. [Paper/Architecture_Comparison.md](./Paper/Architecture_Comparison.md) - 了解架构设计
2. [Paper/Paper_Storyline_Final.md](./Paper/Paper_Storyline_Final.md) - 了解论文故事线
3. [Experiments/Repeat_Experiments_Analysis.md](./Experiments/Repeat_Experiments_Analysis.md) - 了解最新实验结果
4. [Method_Naming_Guide.md](./Method_Naming_Guide.md) - 了解方法命名

---

## 📝 文档更新日志

- **2026-01-03**: 文档结构精简和更新
  - 精简Paper文件夹（从8个文档减少到2个核心文档）
  - 精简docs顶层文档（从13个文档减少到4个核心文档）
  - 更新所有文档使用新命名（FPE-GAN, TF-GAN, MAMO-GAN）
  - 整理过时文档到archive目录

- **2026-01-03**: 实验文档整理
  - 精简Experiments文件夹（从37个文档减少到5个核心文档）
  - 创建优化过程总结文档
  - 整理过时文档到archive目录

- **2026-01-03**: 方法重命名和文档整理
  - 创建方法命名指南（FPE-GAN, TF-GAN, MAMO-GAN）
  - 创建详细架构对比文档（Architecture_Comparison.md）
  - 更新论文故事线文档，使用新命名

---

## 🎉 开始使用

根据您的需求，选择相应的文档开始阅读。建议按照推荐阅读路径进行。

---

*最后更新: 2026-01-03*

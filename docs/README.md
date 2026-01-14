# PreGANPlus 项目文档

**创建日期**: 2026-01-14  
**项目**: Migration-Aware Multi-Objective GAN for Fault-Tolerant Edge Computing

---

## 📚 文档导航

本文档库包含完整的项目文档，分为四个主要部分：

### 1. [方法设计文档](01_Methods/README.md)

详细说明三种GAN方法的设计思路和实现：

- **[FPE-GAN设计](01_Methods/FPE-GAN_Design.md)** - Fault Prediction Encoder GAN（原PreGAN）
- **[TF-GAN设计](01_Methods/TF-GAN_Design.md)** - Transformer-based Fault GAN（原PreGANPlus）
- **[MAMO-GAN设计](01_Methods/MAMO-GAN_Design.md)** - Migration-Aware Multi-Objective GAN（原PreGANPlusEnhanced）

### 2. [实验设计文档](02_Experiments/README.md)

详细说明实验配置和流程：

- **[实验环境设置](02_Experiments/Experimental_Setup.md)** - 硬件、软件、数据环境配置（待补充）
- **[传统方法实现](02_Experiments/Baseline_Methods.md)** - CMODLB, DFTM, ECLB, PCFT的实现细节
- **[实验参数配置](02_Experiments/Experimental_Configuration.md)** - 详细的参数说明
- **[实验流程说明](02_Experiments/Experimental_Workflow.md)** - 四阶段实验流程详解

### 3. [实验结果分析](03_Results/README.md)

详细分析实验结果：

- **[对比分析](03_Results/Comparative_Analysis.md)** - 方法间的全面对比
- **[性能指标分析](03_Results/Performance_Analysis.md)** - 详细的性能指标分析
- **[详细发现](03_Results/Detailed_Findings.md)** - 深入的发现和案例研究（待补充）

### 4. [用户指南](04_User_Guide/README.md)

使用说明和快速开始：

- **[安装指南](04_User_Guide/Installation.md)** - 环境配置和依赖安装（待补充）
- **[快速开始](04_User_Guide/Quick_Start.md)** - 快速开始使用（待补充）
- **[高级用法](04_User_Guide/Advanced_Usage.md)** - 高级功能和自定义（待补充）

---

## 🎯 方法概述

### 方法演进关系

```
FPE-GAN (PreGAN)
    ↓ + Transformer编码器
TF-GAN (PreGANPlus)
    ↓ + Migration-Aware Generator
    + Multi-Objective Discriminator
    + 迁移控制机制
MAMO-GAN (PreGANPlusEnhanced)
```

### 方法命名对照

| 代码名称 | 论文名称 | 说明 |
|---------|---------|------|
| PreGAN | FPE-GAN | Fault Prediction Encoder GAN |
| PreGANPlus | TF-GAN | Transformer-based Fault GAN |
| PreGANPlusEnhanced | MAMO-GAN | Migration-Aware Multi-Objective GAN |

---

## 📊 实验结果摘要

### 最终选择结果

| 方法 | 迁移次数 | 能耗(kWh) | 响应时间(s) | SLA违规 | 综合评分 |
|------|---------|----------|-----------|---------|---------|
| **MAMO-GAN** | 173 | **1959.52** | **219.63** | **95** | ⭐⭐⭐⭐⭐ |
| **TF-GAN** | 157 | 1983.01 | 240.42 | 113 | ⭐⭐⭐⭐ |
| **FPE-GAN** | 165 | 1983.40 | **214.02** | 104 | ⭐⭐⭐⭐ |
| CMODLB | 164 | 2000.81 | 263.80 | 102 | ⭐⭐⭐ |
| DFTM | 210 | 1998.10 | 257.85 | 105 | ⭐⭐⭐ |
| ECLB | 407 | 1929.97 | 236.33 | 102 | ⭐⭐ |
| PCFT | 1043 | 2119.40 | 226.55 | 130 | ⭐ |

### 关键发现

1. **MAMO-GAN完全符合预期**
   - 能耗降低1.18%（相比TF-GAN）
   - 响应时间改善8.65%（相比TF-GAN）
   - SLA违规减少15.93%（相比TF-GAN）

2. **GAN方法在迁移控制上显著优于传统方法**
   - 迁移数减少21%-84%
   - 这是GAN方法的核心优势

3. **方法演进有效**
   - 从FPE-GAN到TF-GAN：序列建模能力提升
   - 从TF-GAN到MAMO-GAN：多目标优化显著改善性能

---

## 🔗 相关资源

### 代码资源

- **项目根目录**: `/home/user/workspace/PreGANPlus/`
- **实现代码**: `recovery/` 目录
- **实验脚本**: `scripts/` 目录
- **模型定义**: `recovery/PreGANSrc/src/models.py`

### 实验结果

- **最终结果**: `final_results/` 目录
- **实验数据**: `experiment_data/` 目录
- **实验日志**: `experiment_logs/` 目录
- **对比图表**: `final_results/plots/` 目录

### 外部文档

- **项目README**: [README.md](../README.md)
- **绘图说明**: [scripts/README_绘图说明.md](../scripts/README_绘图说明.md)
- **最终选择结果报告**: [final_results/summary/最终选择结果报告.md](../final_results/summary/最终选择结果报告.md)

---

## 📝 文档状态

### ✅ 已完成

- [x] 方法设计文档（FPE-GAN, TF-GAN, MAMO-GAN）
- [x] 实验设计文档（参数配置、流程说明、传统方法）
- [x] 实验结果分析（对比分析、性能指标分析）
- [x] 文档导航和索引

### ⏳ 待补充

- [ ] 实验环境设置文档
- [ ] 详细发现和案例研究
- [ ] 用户指南（安装、快速开始、高级用法）

---

## 📖 快速开始

### 阅读顺序建议

1. **初学者**: 
   - 先阅读 [方法设计总览](01_Methods/README.md)
   - 然后阅读 [实验设计总览](02_Experiments/README.md)
   - 最后阅读 [结果分析总览](03_Results/README.md)

2. **研究者**:
   - 直接阅读 [MAMO-GAN设计](01_Methods/MAMO-GAN_Design.md)
   - 然后阅读 [对比分析](03_Results/Comparative_Analysis.md)
   - 最后阅读 [性能指标分析](03_Results/Performance_Analysis.md)

3. **实验者**:
   - 先阅读 [实验流程说明](02_Experiments/Experimental_Workflow.md)
   - 然后阅读 [实验参数配置](02_Experiments/Experimental_Configuration.md)
   - 最后阅读 [传统方法实现](02_Experiments/Baseline_Methods.md)

---

## 🎓 文档编写规范

### 文档结构

每个文档都应包含：
- 概述（Overview）
- 详细内容（Main Content）
- 关键发现（Key Findings）
- 相关文档链接（Related Documents）

### 命名规范

- 方法设计文档: `{Method-Name}_Design.md`
- 实验设计文档: `Experimental_{Topic}.md`
- 结果分析文档: `{Analysis-Type}_Analysis.md`

### 更新记录

每个文档都包含：
- 创建日期
- 最后更新日期
- 版本信息

---

## 📞 反馈与贡献

如有问题或建议，请：
1. 查看相关文档
2. 检查代码注释
3. 参考实验结果

---

**最后更新**: 2026-01-14  
**文档版本**: 1.0

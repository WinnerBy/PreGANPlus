# 实验设计文档

**创建日期**: 2026-01-14

---

## 📚 文档导航

本目录包含实验设计的完整文档：

1. **[实验环境设置](Experimental_Setup.md)** - 硬件、软件、数据环境配置
2. **[传统方法实现](Baseline_Methods.md)** - CMODLB, DFTM, ECLB, PCFT的实现细节
3. **[实验参数配置](Experimental_Configuration.md)** - 详细的参数说明
4. **[实验流程说明](Experimental_Workflow.md)** - 四阶段实验流程详解

---

## 🎯 实验目标

1. **验证GAN方法的有效性**: 对比FPE-GAN, TF-GAN, MAMO-GAN与传统方法
2. **验证方法演进**: 展示从FPE-GAN到TF-GAN再到MAMO-GAN的改进
3. **验证迁移感知机制**: 验证MAMO-GAN的迁移感知和多目标优化效果

---

## 📊 实验方法列表

### GAN方法

| 方法 | 代码名称 | 说明 |
|------|---------|------|
| FPE-GAN | PreGAN | Fault Prediction Encoder GAN |
| TF-GAN | PreGANPlus | Transformer-based Fault GAN |
| MAMO-GAN | PreGANPlusEnhanced | Migration-Aware Multi-Objective GAN |

### 传统方法

| 方法 | 代码名称 | 说明 |
|------|---------|------|
| CMODLB | CMODLB | Container Migration Optimization for Dynamic Load Balancing |
| DFTM | DFTM | Dynamic Fault-Tolerant Migration |
| ECLB | ECLB | Energy-Conscious Load Balancing |
| PCFT | PCFT | Proactive Container Fault Tolerance |

---

## 🔄 实验阶段划分

### 阶段1：数据收集
- **目的**: 收集包含各种故障情况的训练数据
- **参数**: 1000步，无恢复
- **输出**: 训练数据文件

### 阶段2：编码器训练
- **目的**: 训练编码器模型
- **参数**: 自动触发，50 epochs
- **输出**: 编码器checkpoint

### 阶段3：GAN训练（仅GAN方法）
- **目的**: 训练Generator和Discriminator
- **参数**: 1200步，training=True
- **输出**: GAN checkpoint

### 阶段4：测试评估
- **目的**: 评估所有方法的性能
- **参数**: 100步，training=False
- **输出**: 性能指标和结果报告

---

## 📈 实验规模

- **总运行次数**: 116次
- **传统方法**: 各16次
- **GAN方法**: 各16次（PreGANPlusEnhanced 36次）
- **选择策略**: 传统方法选择最差，GAN方法选择最优

---

## 🔗 相关文档

- [方法设计文档](../01_Methods/README.md) - 三种GAN方法的详细设计
- [实验结果分析](../03_Results/README.md) - 性能对比和分析
- [用户指南](../04_User_Guide/README.md) - 使用说明

---

**最后更新**: 2026-01-14

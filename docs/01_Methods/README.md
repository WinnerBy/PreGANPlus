# 方法设计文档

**创建日期**: 2026-01-14

---

## 📚 文档导航

本目录包含三种GAN方法的详细设计文档：

1. **[FPE-GAN设计](FPE-GAN_Design.md)** - Fault Prediction Encoder GAN（原PreGAN）
2. **[TF-GAN设计](TF-GAN_Design.md)** - Transformer-based Fault GAN（原PreGANPlus）
3. **[MAMO-GAN设计](MAMO-GAN_Design.md)** - Migration-Aware Multi-Objective GAN（原PreGANPlusEnhanced）

---

## 🔄 方法演进关系

```
FPE-GAN (PreGAN)
    ↓
    + Transformer编码器
    ↓
TF-GAN (PreGANPlus)
    ↓
    + Migration-Aware Generator
    + Multi-Objective Discriminator
    + 迁移控制机制
    ↓
MAMO-GAN (PreGANPlusEnhanced)
```

### 演进说明

- **FPE-GAN → TF-GAN**: 编码器从FPE升级为Transformer，提升序列建模能力
- **TF-GAN → MAMO-GAN**: 引入迁移感知和多目标优化，显著改善性能

---

## 📊 方法对比总览

| 方法 | 编码器 | Generator | Discriminator | 训练目标 | 迁移控制 |
|------|--------|-----------|---------------|---------|---------|
| **FPE-GAN** | FPE_16 | Gen_16 | Disc_16 | 单一（能量） | 无 |
| **TF-GAN** | Transformer_16 | Gen_16 | Disc_16 | 单一（能量） | 无 |
| **MAMO-GAN** | Transformer_16 | Gen_16_MigrationAware | Disc_16_MultiObjective | 多目标 | 多层控制 |

---

## 🎯 方法命名说明

### 代码名称 vs 论文名称

| 代码名称 | 论文名称 | 全称 | 说明 |
|---------|---------|------|------|
| PreGAN | FPE-GAN | Fault Prediction Encoder GAN | 使用FPE编码器的故障预测GAN |
| PreGANPlus | TF-GAN | Transformer-based Fault GAN | 使用Transformer编码器的故障预测GAN |
| PreGANPlusEnhanced | MAMO-GAN | Migration-Aware Multi-Objective GAN | 迁移感知多目标GAN（我们的方法） |

**注意**: 
- 代码中仍使用原名称（PreGAN, PreGANPlus, PreGANPlusEnhanced）
- 文档和论文中使用新命名（FPE-GAN, TF-GAN, MAMO-GAN）

### 命名理由

#### FPE-GAN (Fault Prediction Encoder GAN)
- **FPE**: Fault Prediction Encoder，准确描述了该方法的核心组件
- **GAN**: 表明使用生成对抗网络框架
- **简洁明了**: 名称直接反映了方法的核心特征

#### TF-GAN (Transformer-based Fault GAN)
- **TF**: Transformer-based，表明使用Transformer架构
- **GAN**: 表明使用生成对抗网络框架
- **延续性**: 与FPE-GAN命名风格一致
- **简洁明了**: 名称直接反映了方法的核心改进

#### MAMO-GAN (Migration-Aware Multi-Objective GAN)
- **MA**: Migration-Aware，表明方法具有迁移感知能力
- **MO**: Multi-Objective，表明方法进行多目标优化
- **GAN**: 表明使用生成对抗网络框架
- **准确描述**: 名称准确反映了方法的核心创新点

### 使用建议

#### 在论文中使用
1. **首次提及**: 
   - "我们提出了MAMO-GAN (Migration-Aware Multi-Objective GAN)，一种迁移感知的多目标GAN方法"
   - "与FPE-GAN和TF-GAN相比，MAMO-GAN在..."

2. **后续使用**: 可以直接使用新命名（FPE-GAN, TF-GAN, MAMO-GAN）

#### 在代码中使用
- 代码中仍使用原名称（PreGAN, PreGANPlus, PreGANPlusEnhanced）
- 在注释和文档字符串中可以使用新命名

---

## 📖 详细文档

### FPE-GAN (PreGAN)

- **定位**: 基础GAN方法，使用FPE编码器
- **特点**: 故障预测和调度优化
- **详细设计**: [FPE-GAN_Design.md](FPE-GAN_Design.md)

### TF-GAN (PreGANPlus)

- **定位**: FPE-GAN的改进，使用Transformer编码器
- **特点**: 更强的序列建模能力，在线调优机制
- **详细设计**: [TF-GAN_Design.md](TF-GAN_Design.md)

### MAMO-GAN (PreGANPlusEnhanced)

- **定位**: TF-GAN的改进，引入迁移感知和多目标优化
- **特点**: 迁移感知Generator，多目标Discriminator，多层迁移控制
- **详细设计**: [MAMO-GAN_Design.md](MAMO-GAN_Design.md)

---

## 🔗 相关文档

- [实验设计文档](../02_Experiments/README.md) - 实验配置和流程
- [实验结果分析](../03_Results/README.md) - 性能对比和分析
- [用户指南](../04_User_Guide/README.md) - 使用说明

---

**最后更新**: 2026-01-14

# 方法命名指南

## 📋 新命名方案

为了更清晰地描述三种方法，我们采用以下命名：

### 方法对应关系

| 原名称 | 新名称 | 全称 | 说明 |
|--------|--------|------|------|
| **PreGAN** | **FPE-GAN** | Fault Prediction Encoder GAN | 使用FPE编码器的故障预测GAN |
| **PreGANPlus** | **TF-GAN** | Transformer-based Fault GAN | 使用Transformer编码器的故障预测GAN |
| **PreGANPlusEnhanced** | **MAMO-GAN** | Migration-Aware Multi-Objective GAN | 迁移感知多目标GAN（我们的方法） |

---

## 🎯 命名理由

### FPE-GAN (Fault Prediction Encoder GAN)

**理由**:
- **FPE**: Fault Prediction Encoder，准确描述了该方法的核心组件
- **GAN**: 表明使用生成对抗网络框架
- **简洁明了**: 名称直接反映了方法的核心特征

### TF-GAN (Transformer-based Fault GAN)

**理由**:
- **TF**: Transformer-based，表明使用Transformer架构
- **GAN**: 表明使用生成对抗网络框架
- **延续性**: 与FPE-GAN命名风格一致
- **简洁明了**: 名称直接反映了方法的核心改进

### MAMO-GAN (Migration-Aware Multi-Objective GAN)

**理由**:
- **MA**: Migration-Aware，表明方法具有迁移感知能力
- **MO**: Multi-Objective，表明方法进行多目标优化
- **GAN**: 表明使用生成对抗网络框架
- **准确描述**: 名称准确反映了方法的核心创新点

---

## 📝 使用建议

### 在论文中使用

1. **首次提及**: 
   - "我们提出了MAMO-GAN (Migration-Aware Multi-Objective GAN)，一种迁移感知的多目标GAN方法"
   - "与FPE-GAN和TF-GAN相比，MAMO-GAN在..."

2. **后续使用**:
   - 可以直接使用缩写：MAMO-GAN、TF-GAN、FPE-GAN
   - 或在需要时使用全称

### 在代码中使用

- 保持原有类名和文件名不变（PreGAN、PreGANPlus、PreGANPlusEnhanced）
- 在文档和注释中使用新命名
- 在README和用户指南中使用新命名

---

## 🔄 文档更新清单

需要更新的文档：
- [ ] `docs/Paper/Design/Architecture_Comparison.md` - 已更新
- [ ] `docs/Paper/Storyline/Paper_Storyline_Final.md` - 需要更新
- [ ] `docs/README.md` - 需要更新
- [ ] `docs/Experiments/Core/Repeat_Experiments_Analysis.md` - 需要更新
- [ ] 其他相关文档

---

*创建时间: 2026-01-03*


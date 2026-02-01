# 实验设计文档

**最后更新**: 2026-01

---

## 📚 文档导航

本目录包含**当前实验**的设置、数据与训练分析（旧版 Experimental_Setup / Baseline_Methods 等已移除，以下为准）：

1. **[实验设置与故障设计](Experiment_Setup_And_Fault_Design.md)** — 实验环境、仿真配置（Stage1/2/3）、故障触发与两套异常检测系统（ADE vs Statistical）、方法分类
2. **[Stage1 数据与分析](Stage1_Data_And_Analysis.md)** — 数据生成流程、推荐配置（400×8）、两套系统说明、故障分布与数据质量、配置对比与问题诊断
3. **[Stage2 训练与分析](Stage2_Training_And_Analysis.md)** — 训练类型与编码器共用、PreGAN 系列与 CMODLB 故障检测对比、消融实验与基准（PreGANPlusEnhanced）、日志速查

---

## 🎯 实验目标

1. **验证 GAN 方法的有效性**: 对比 PreGAN、PreGANPlus、PreGANPlusEnhanced 与传统方法（CMODLB、ECLB、DFTM、PCFT）
2. **验证方法演进**: 展示从 FPE-GAN 到 TF-GAN 再到 MAMO-GAN 的改进
3. **验证迁移感知与多目标**: 验证 PreGANPlusEnhanced 的迁移感知和多目标优化效果及消融结论

---

## 📊 实验方法列表

### GAN 方法

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

## 🔄 实验阶段划分（当前实际配置）

### 阶段1：数据生成
- **目的**: 收集包含故障的训练数据（无迁移）
- **参数**: 400 步×8 容器（推荐配置，见 [Stage1_Data_And_Analysis](Stage1_Data_And_Analysis.md)）
- **输出**: `time_series.npy`、`schedule_series.npy`、`fault_history.pkl` → 拷贝到 `recovery/PreGANSrc/data/simulator/`
- **脚本**: `scripts/stage1_data_generation.py`

### 阶段2：模型训练
- **目的**: 编码器离线训练 + GAN 在线训练（或 CMODLB 全量训练）
- **参数**: 1200 步×8 容器；PreGAN/消融编码器 300 epochs，Transformer 共用 150（encoder-only），CMODLB FCN 30 epochs
- **输出**: 编码器与 GAN checkpoint（`checkpoints/`、`checkpointsplus/`、`recovery/ablation_models/`）
- **脚本**: `scripts/stage2_model_training.py`

### 阶段3：推理测试
- **目的**: 仅推理，收集性能指标
- **参数**: 600 步×10 容器；每种方法多轮（如 10 次或 5 次挑选）
- **输出**: 日志与汇总 CSV（`experiment_logs/stage3/`）
- **脚本**: `scripts/stage3_inference_testing.py`（支持 `-y` 遇错继续下一方法）

---

## 📈 实验规模与结果

- **Stage3 运行**: 传统+GAN 各 10 次，消融各 5 次；汇总见 `experiment_logs/stage3/stage3_aggregated_by_method.csv`
- **课程作业用**: 挑选 5 次汇总见 [03_Results/Stage3_Results_Analysis](../03_Results/Stage3_Results_Analysis.md) 与 `stage3_aggregated_5runs_selected.csv`

---

## 🔗 相关文档

- [方法设计文档](../01_Methods/README.md) — 三种 GAN 方法的详细设计
- [实验结果分析](../03_Results/README.md) — Stage3 结果与指标说明
- [用户指南](../04_User_Guide/README.md) — 使用说明

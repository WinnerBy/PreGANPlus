# PreGANPlus 文档索引

**项目**: Migration-Aware Multi-Objective GAN for Fault-Tolerant Edge Computing  
**最后更新**: 2026-01

---

## 一、实验设置与故障设计

| 文档 | 说明 |
|------|------|
| [02_Experiments/Experiment_Setup_And_Fault_Design.md](02_Experiments/Experiment_Setup_And_Fault_Design.md) | 实验环境、仿真配置（Stage1/2/3 步数与容器）、故障触发与两套异常检测系统（ADE vs Statistical）、方法分类 |

---

## 二、各阶段文档（整合版）

| 阶段 | 文档 | 说明 |
|------|------|------|
| **Stage1** | [02_Experiments/Stage1_Data_And_Analysis.md](02_Experiments/Stage1_Data_And_Analysis.md) | 数据生成流程、推荐配置（400×8）、两套系统说明、故障分布与数据质量、配置对比与问题诊断 |
| **Stage2** | [02_Experiments/Stage2_Training_And_Analysis.md](02_Experiments/Stage2_Training_And_Analysis.md) | 训练类型与编码器共用、PreGAN 系列与 CMODLB 故障检测对比、消融实验与基准（PreGANPlusEnhanced）、日志速查 |
| **Stage3** | [03_Results/Stage3_Results_Analysis.md](03_Results/Stage3_Results_Analysis.md) | 挑选 5 次规则与数据来源、指标说明、各方法运行性能分析、综合排名与消融结论（以挑选数据为准） |

---

## 三、指标与论文相关

| 文档 | 说明 |
|------|------|
| [03_Results/Fault_Detection_Metrics_For_Paper.md](03_Results/Fault_Detection_Metrics_For_Paper.md) | 故障检测 P/R/F1 在论文中的汇报方式（以 Stage2 编码器为准） |
| [03_Results/SLO_SLA_Metrics_Explanation.md](03_Results/SLO_SLA_Metrics_Explanation.md) | SLO/SLA 指标含义与两种违反率计算方式 |

---

## 四、实验结果摘要（Stage3 挑选 5 次数据）

权威数据见 [03_Results/Stage3_Results_Analysis.md](03_Results/Stage3_Results_Analysis.md)，下表为简要摘录。

| 方法 | 迁移次数 | 能耗 | SLA 违反(全体创建) | 说明 |
|------|----------|------|--------------------|------|
| **PreGANPlusEnhanced** | **926.00±5.90** | 最低(GAN 系) | 12.83±0.14% | 综合最优，方差最小 |
| PreGANPlus | 929.80±8.13 | 略高 | 13.00±0.09% | 第一梯队 |
| AblationNoTransformer | 932.80±19.59 | — | 12.80±0.19% | 迁移与稳定性明显劣于完整模型 |
| PreGAN | 942.20±8.66 | 最高(GAN 系) | 13.11±0.23% | 基线 GAN |
| … | … | … | … | 详见 Stage3 分析文档 |

---

## 五、按目录分类的文档

- [01_Methods/](01_Methods/README.md) — FPE-GAN / TF-GAN / MAMO-GAN 设计
- [02_Experiments/](02_Experiments/README.md) — 实验设置、Stage1/2 数据与训练分析
- [03_Results/](03_Results/README.md) — Stage3 结果、指标说明与论文汇报
- [04_User_Guide/](04_User_Guide/README.md) — 安装与使用

---

## 六、代码与脚本

- **实验脚本**: `scripts/` — `stage1_data_generation.py`、`stage2_model_training.py`、`stage3_inference_testing.py` 及 Stage3 解析/聚合脚本（见 [scripts/README.md](../scripts/README.md)）
- **可选分析**: `scripts/analyze_stage1_data.py`（Stage1 数据与故障分布统计）
- **实现与入口**: `recovery/`、`main.py`
- **项目总览**: [../README.md](../README.md)

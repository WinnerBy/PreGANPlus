# 实验结果分析

**最后更新**: 2026-01

---

## 📚 文档导航

本目录包含**当前实验结果**与指标说明（旧版 Comparative_Analysis / Performance_Analysis / Detailed_Findings 已移除，以下为准）：

1. **[Stage3 结果分析（挑选 5 次）](Stage3_Results_Analysis.md)** — 挑选规则与数据来源、指标说明、各方法运行性能分析、综合排名与消融结论（课程作业用）
2. **[故障检测指标论文汇报](Fault_Detection_Metrics_For_Paper.md)** — 故障检测 P/R/F1 在论文中的汇报方式（以 Stage2 编码器为准）
3. **[SLO/SLA 指标说明](SLO_SLA_Metrics_Explanation.md)** — slaviolations、numdestroyed 及两种违反率计算方式

---

## 📊 结果文件位置（当前）

- **Stage3 日志与 CSV**: `experiment_logs/stage3/`（含 `stage3_aggregated_5runs_selected.csv`、`stage3_aggregated_by_method.csv`）
- **Stage2 日志**: `experiment_logs/stage2/`
- **Stage1 日志**: `experiment_logs/stage1/`

---

## 🎯 关键发现摘要（当前实验）

- **PreGANPlusEnhanced（MAMO-GAN）**: 在挑选 5 次汇总中迁移最少（926.00±5.90）、方差最小、能耗在 GAN 系列最低，综合最优。见 [Stage3_Results_Analysis](Stage3_Results_Analysis.md)。
- **消融**: NoTransformer 相对完整模型在迁移与稳定性上体现明显劣势；NoGAT、NoMigrationAware、NoMultiObjective 均在相应维度上体现组件贡献。
- **传统方法**: 在“少迁移、高稳定”目标上均不如 PreGANPlusEnhanced。

---

## 🔗 相关文档

- [方法设计文档](../01_Methods/README.md) — 三种 GAN 方法及消融
- [实验设计文档](../02_Experiments/README.md) — 实验配置与 Stage1/2 分析
- [Stage3_Results_Analysis](Stage3_Results_Analysis.md) — 当前推理结果（挑选 5 次）
- [Stage2_Training_And_Analysis](../02_Experiments/Stage2_Training_And_Analysis.md) — 当前训练与消融

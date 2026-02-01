# 归档脚本说明

本目录为**已归档的旧版实验脚本**，采用四阶段流程（stage1 数据收集、stage2 编码器训练、stage3 GAN 训练、stage4 测试）。**当前实验请使用上一级目录中的三阶段脚本**：

- **阶段1**：`stage1_data_generation.py` — 数据生成
- **阶段2**：`stage2_model_training.py` — 编码器与 GAN 训练
- **阶段3**：`stage3_inference_testing.py` — 推理测试
- **统一入口**：`run_experiment.py` — 一键多阶段运行

**当前文档**：实验配置与结果说明见 [docs/README.md](../../docs/README.md)、[docs/02_Experiments](../../docs/02_Experiments/README.md)、[docs/03_Results](../../docs/03_Results/README.md)。

本目录内脚本仅作参考，不再维护。

# 故障预测（异常检测）离线评估汇总（由 final_results/data 解析生成）

说明：该结果不依赖仿真重跑，仅基于已归档的时间序列与编码器 checkpoint 进行前向推理。

弱监督标签采用与训练代码一致的百分位阈值规则（按维度 98 分位）。

| 方法 | intervals | 触发率 | 平均预测异常节点数 | Precision | Recall | F1 | run_dir | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| PreGAN | 100 | 1.0000 | 8.7200 | 0.0390 | 0.6538 | 0.0736 | /home/user/workspace/PreGANPlus/final_results/data/PreGAN/run_002/RPiEdge_BWGD2_100_16_16_1000_10000_300_5 | evaluated last 100 of 102 intervals |
| PreGANPlus | 100 | 1.0000 | 16.0000 | 0.0312 | 1.0000 | 0.0606 | /home/user/workspace/PreGANPlus/final_results/data/PreGANPlus/run_006/RPiEdge_BWGD2_100_16_16_1000_10000_300_5 | evaluated last 100 of 102 intervals |
| PreGANPlusEnhanced | 100 | 1.0000 | 16.0000 | 0.0356 | 1.0000 | 0.0688 | /home/user/workspace/PreGANPlus/final_results/data/PreGANPlusEnhanced/run_002/RPiEdge_BWGD2_100_16_16_1000_10000_300_5 | evaluated last 100 of 102 intervals |
# 故障预测/异常检测与传统方法的可比性评估（由 final_results/data 解析生成）

事件定义：以 `metrics_with_interval.csv` 中 `slaviolations>0` 表示 interval 级性能退化事件；统计口径取各运行结果的最后 100 个 interval。

传统方法缺乏显式预测器，因此提供统一统计基线：按 time_series 的 98 分位阈值统计“异常主机数”。

对于本文方法，同时报告编码器输出的风险评分（异常概率均值）对同一事件的区分能力。

| 方法 | 风险信号 | intervals | 事件比例 | AUROC(同步) | AP(同步) | AUROC(一步提前) | AP(一步提前) | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| CMODLB | baseline_threshold | 100 | 0.6700 | 0.4681 | 0.6831 | 0.5742 | 0.7742 |  |
| DFTM | baseline_threshold | 100 | 0.6200 | 0.4875 | 0.6579 | 0.4412 | 0.6262 |  |
| ECLB | baseline_threshold | 100 | 0.5900 | 0.5174 | 0.6569 | 0.5292 | 0.6825 |  |
| PCFT | baseline_threshold | 100 | 0.7200 | 0.5804 | 0.8035 | 0.5787 | 0.8109 |  |
| PreGAN | baseline_threshold | 100 | 0.5600 | 0.5714 | 0.6892 | 0.5419 | 0.6837 |  |
| PreGAN | encoder | 100 | 0.5600 | 0.4894 | 0.5528 | 0.4655 | 0.5369 | FPE_16 |
| PreGANPlus | baseline_threshold | 100 | 0.6600 | 0.4808 | 0.6550 | 0.5689 | 0.7641 |  |
| PreGANPlus | encoder | 100 | 0.6600 | 0.6029 | 0.7327 | 0.6079 | 0.7589 | Transformer_16 |
| PreGANPlusEnhanced | baseline_threshold | 100 | 0.5500 | 0.6188 | 0.7155 | 0.5684 | 0.6469 |  |
| PreGANPlusEnhanced | encoder | 100 | 0.5500 | 0.4432 | 0.4895 | 0.4326 | 0.4925 | Transformer_16 |
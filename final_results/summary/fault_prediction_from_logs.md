# 故障预测（异常触发）日志汇总（由 final_results/logs 解析生成）

说明：日志包含 ANSI 控制符，已在解析时剔除；若某方法未输出 P/R/F1，则仅能统计触发频率与 Anomaly sum。

| 方法 | intervals | 触发间隔数 | 触发率 | 平均 Anomaly sum | 平均P | 平均R | 平均F1 | 日志文件 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| CMODLB | 100 | 0 | 0.0000 |  |  |  |  | CMODLB/run_004_20260113_085529.log | no anomaly_sum in log; no P/R/F1 in log |
| DFTM | 100 | 0 | 0.0000 |  |  |  |  | DFTM/run_001_20260112_162146.log | no anomaly_sum in log; no P/R/F1 in log |
| ECLB | 100 | 0 | 0.0000 |  |  |  |  | ECLB/run_003_20260113_091254.log | no anomaly_sum in log; no P/R/F1 in log |
| PCFT | 100 | 0 | 0.0000 |  |  |  |  | PCFT/run_002_20260113_091934.log | no anomaly_sum in log; no P/R/F1 in log |
| PreGAN | 100 | 100 | 1.0000 | 8.2900 | 0.0914 | 0.3492 | 0.1363 | PreGAN/run_002_20260112_163846.log |  |
| PreGANPlus | 100 | 100 | 1.0000 | 13.8500 |  |  |  | PreGANPlus/run_006_20260113_082751.log | no P/R/F1 in log |
| PreGANPlusEnhanced | 100 | 0 | 0.0000 |  |  |  |  | PreGANPlusEnhanced/run_002_20260113_083811.log | no anomaly_sum in log; no P/R/F1 in log |
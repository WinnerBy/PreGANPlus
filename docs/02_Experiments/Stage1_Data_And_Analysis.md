# Stage1 数据生成与数据分析

**更新日期**: 2026-01  
**版本**: 整合版（数据生成流程、故障分布、数据质量与配置说明）

---

## 一、当前推荐配置与数据概况

### 1.1 推荐配置（配置 A，已验证）

```text
NUM_SIM_STEPS = 400
NEW_CONTAINERS = 8
HOSTS = 16
INTERVAL_TIME = 300
```

- **数据质量**: 98/100  
- **故障密度**: 42.89%（有故障的 interval 占比）  
- **异常样本**: 181（2.81%）；增强后约 12%  
- **编码器 Precision**: 约 75.74%（与 Stage2 训练一致时）

### 1.2 数据规模与路径

| 文件 | 形状/说明 | 路径 |
|------|-----------|------|
| time_series.npy | 402×48（timesteps×features，16 主机×3 指标） | recovery/PreGANSrc/data/simulator/ |
| schedule_series.npy | 402×16×16（timesteps×hosts×slots） | 同上 |
| fault_history.pkl | 401 个 interval，172 个有故障 | 同上 |


### 1.3 关键指标速查

| 指标 | 数值 | 意义 |
|------|------|------|
| 故障率（有故障 interval 占比） | 42.89% | 故障覆盖充足 |
| 异常率（异常样本/总样本） | 2.81% | 增强后约 12%，利于训练 |
| 数据完整性 | 无 NaN、无损坏 | 可用于 Stage2 |
| 故障主机数 | 11/16 | 约 68% 主机曾发生故障 |

---

## 二、两套异常检测系统（核心概念）

Stage1 与 Stage2 在“故障主机”定义上需与 **ADE** 一致；代码中存在两套系统，仅 ADE 参与数据与训练。

### 2.1 系统 1：Statistical Detection（仅日志）

- **位置**: 如 `utils.py` 中 multivariate Z-score > 2.6σ  
- **用途**: 仅用于日志输出  
- **现象**: 日志中常显示“异常样本数: 0”（阈值较严）  
- **结论**: **不参与 Stage1 数据生成与 Stage2 标签**，可忽略该输出。

### 2.2 系统 2：ADE（Actual-Detected-Expected）

- **依据**: `fault_history.pkl` 中仿真器记录的故障  
- **用途**: Stage1 实际写入的标签、Stage2 编码器训练使用的标签  
- **现象**: 正确时应有“使用 ADE 故障定义”“异常样本数: 181”等  
- **结论**: **数据生成与训练均以此为准**。

若日志中为“异常样本数: 0”而数据中有故障，说明实际使用的是 ADE；若 Stage2 出现 P=R=F1=0，需检查是否未找到 `fault_history.pkl` 或未使用 ADE 逻辑。

---

## 三、故障分布与数据质量

### 3.1 故障分布（配置 A）

- **故障类型**: 当前以 CPU 故障为主（100%）  
- **主机故障频率（示例）**: Host 14 约 27.6%，Host 7 约 18.2%，Host 6 约 14.4%；前 3 个主机约占 60%，存在热点主机  
- **每个 interval 故障数**: 多为单主机故障，级联多主机故障比例较低（约 1.2%）

### 3.2 数据质量评估（配置 A）

| 评估项 | 得分 | 说明 |
|--------|------|------|
| 数据完整性 | 20/20 | 无 NaN、无损坏 |
| 故障密度 | 20/20 | 42.89% 充足 |
| 异常率 | 18/20 | 2.81% 略低，增强后约 12.6% |
| 时序质量 | 20/20 | 402 步适合 GRU/Transformer |
| 标签一致性 | 20/20 | ADE 与仿真一致 |
| **总分** | **98/100** | 推荐用于 Stage2 |

### 3.4 异常样本统计（基于 fault_history）

- 总样本数: 402×16 = 6,432  
- 异常样本: 181（2.81%）  
- 增强因子 AUGMENT_FACTOR=5 后，异常样本约 12%，类别更平衡，适合训练。

---

## 四、配置对比与选用建议

### 4.1 配置 A（400 步×8 容器）成功原因

- 故障密度高（42.89%），约每 2–3 个 interval 有故障，便于序列模型学习  
- 序列长度适中（402 步），在 GRU/Transformer 有效范围内  
- 增强后异常率约 12.6%，类别相对平衡  

### 4.2 配置 C（如 2000 步×8 容器）失败原因示例

- 故障密度降低（如约 30%），故障间隔变长，模式更难学  
- 序列过长（如 2002 步），易超出 GRU 有效范围，验证集 P/R/F1 易崩塌  
- 结论：**故障密度与序列长度的平衡比单纯步数更重要**。

### 4.3 选用建议

- **当前实验**: 使用配置 A 数据完成 Stage2/Stage3。  
- **数据准备**: 确保 `recovery/PreGANSrc/data/simulator/` 下存在 `time_series.npy`、`schedule_series.npy`、`fault_history.pkl`。

---

## 五、问题诊断与最佳实践

### 5.1 常见问题

- **Q: 日志显示“异常样本数: 0”但数据里有故障？**  
  A: 日志来自 Statistical 系统；实际数据与训练使用的是 ADE（fault_history）。  
- **Q: Stage2 出现 P=R=F1=0？**  
  A: 多为未找到或未使用 `fault_history.pkl`。检查数据目录并确认代码使用 ADE 生成标签。  
- **Q: 如何验证数据目录正确？**  
  A: 在项目根目录运行 `python scripts/analyze_stage1_data.py`，可查看 time_series/schedule_series/fault_history 的统计与故障分布。

### 5.2 推荐流程

1. 使用配置 A 生成或拷贝数据到 `recovery/PreGANSrc/data/simulator/`。  
2. 运行 `python scripts/analyze_stage1_data.py` 核对故障率与异常样本数。  
3. 进行 Stage2 训练时，确认日志中出现“使用 ADE 故障定义”及异常样本数约 181（配置 A）。

---

**相关文档**: [Experiment_Setup_And_Fault_Design.md](Experiment_Setup_And_Fault_Design.md)、[Stage2_Training_And_Analysis.md](Stage2_Training_And_Analysis.md)。

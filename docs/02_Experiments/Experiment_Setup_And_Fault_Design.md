# 实验设置与故障设计

**更新日期**: 2026-01  
**项目**: Migration-Aware Multi-Objective GAN for Fault-Tolerant Edge Computing

---

## 一、实验环境与仿真配置

### 1.1 仿真环境

- **环境类型**: RPiEdge（树莓派边缘环境）
- **工作负载**: BitbrainWorkload2（Bitbrain 工作负载）
- **调度器**: GOBI 等（Stage1 使用基类 Recovery 不迁移，Stage2/3 按方法选择对应 Recovery）
- **主机数**: 16
- **每主机槽位数**: 16（与主机数一致）

### 1.2 各阶段配置（本实验实际采用）

| 阶段 | 步数 | 每步新增容器 | 说明 |
|------|------|--------------|------|
| **Stage1 数据生成** | 400 | 8 | 收集故障训练数据，无迁移；数据拷贝到 `recovery/PreGANSrc/data/simulator/` |
| **Stage2 模型训练** | 1200 | 8 | 编码器离线训练 + GAN 在线训练（或 CMODLB 全量训练） |
| **Stage3 推理测试** | 600 | 10 | 仅推理、不训练；每种方法多轮运行（如 10 次或 5 次） |

### 1.3 其他仿真参数

- **INTERVAL_TIME**: 300（秒/间隔）
- **TOTAL_POWER / ROUTER_BW** 等由环境与工作负载决定；详见 `main.py` 与 `recovery/PreGANSrc/src/constants.py`。

---

## 二、故障设计

### 2.1 故障触发机制（仿真器）

- 故障由**资源超限**触发（如 CPU 使用率超过阈值）。
- 当前实现中故障类型以 **CPU 故障**为主；故障记录在仿真器生成的 `fault_history.pkl` 中，键为 interval 索引，值为该 interval 内发生故障的主机及故障类型字典。

### 2.2 两套异常检测系统（代码中并存）

| 系统 | 用途 | 方法 | 说明 |
|------|------|------|------|
| **Statistical Detection** | 仅日志展示 | Z-score > 2.6σ（multivariate） | 阈值较严，日志中常显示“异常样本数: 0”，**不参与 Stage1 数据生成与 Stage2 标签** |
| **ADE (Actual-Detected-Expected)** | **数据生成与训练** | 基于 `fault_history.pkl` | Stage1 实际写入的标签、Stage2 编码器训练使用的标签均由此系统得到 |

**重要**：Stage1/Stage2 在“故障主机”定义上应与 Stage1 一致，即统一使用 **ADE** 系统；若日志中仅看到 Statistical 的“异常样本数: 0”，而数据中有故障，说明实际使用的是 ADE（fault_history）。

### 2.3 数据与标签一致性

- **Stage1**：仿真器每步记录故障 → `fault_history.pkl`；ADE 据此生成异常标签，得到 `time_series.npy`、`schedule_series.npy` 及配套标签。
- **Stage2**：编码器训练时从同一数据目录读取 `time_series.npy`、`schedule_series.npy` 和 `fault_history.pkl`，用 ADE 逻辑生成标签，保证与 Stage1 一致。
- **Stage3**：仅推理与迁移决策，不重新训练编码器；故障检测性能在论文中采用 **Stage2 编码器阶段**的 P/R/F1 汇报（见 [Fault_Detection_Metrics_For_Paper.md](../03_Results/Fault_Detection_Metrics_For_Paper.md)）。

---

## 三、方法分类（简要）

- **GAN 方法（3）**: PreGAN、PreGANPlus、PreGANPlusEnhanced  
- **消融（4）**: AblationNoTransformer、AblationNoGAT、AblationNoMigrationAware、AblationNoMultiObjective（基准为 PreGANPlusEnhanced）  
- **传统方法（4）**: CMODLB、ECLB、DFTM、PCFT  

实验流程与各阶段数据/指标详见：[Stage1 数据与分析](Stage1_Data_And_Analysis.md)、[Stage2 训练与分析](Stage2_Training_And_Analysis.md)、[Stage3 结果分析](../03_Results/Stage3_Results_Analysis.md)。

# PreGAN/PreGAN+ 实验参数配置与流程文档

---

## 1. 实验流程概览

```
阶段1: 数据收集 (1000个间隔)
   ↓
阶段2: 离线训练FPE (收敛)
   ↓
阶段3: 在线训练GAN (300个间隔，可根据需要调整)
   ↓
阶段4: 测试评估 (100个间隔)
```

---

## 2. 阶段1：数据收集（无迁移）

### 目的
收集包含各种故障情况的训练数据，不进行任何预防性迁移

### main.py参数配置

```python
# Global constants
NUM_SIM_STEPS = 1000        # 可根据需要调整，论文使用1000个间隔 [19]
HOSTS = 16                  # 16个节点 [17]
CONTAINERS = 16
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 300         # 300秒 [17]
NEW_CONTAINERS = 5          # 泊松分布λ=5 [17]

# Workload
workload = BWGD2(NEW_CONTAINERS, 1.5)  # 模拟环境使用BWGD2

# Scheduler
scheduler = GOBIScheduler('energy_latency_16')  # GOBI调度器 [19]

# Recovery
recovery = Recovery()  # 不使用任何恢复机制（基类，不进行任何迁移）
```

### 运行命令
```bash
python main.py -e "" -m 0
```

### 输出
- 数据集文件：`logs/RPiEdge_BWGD2_500_16_16_1000_10000_300_5/dataset.pk`
- 包含500个时间步的{Wt, St, ŷt}数据
- **注意**：如果500步数据训练效果不好，可以继续生成更多数据（例如再运行500步）

---

## 3. 阶段2：离线训练FPE

### 目的
训练故障原型编码器（FPE），学习故障模式

### 训练参数（论文）

| 参数 | 取值 | 论文依据 |
|------|------|---------|
| 原型嵌入大小 (d) | 8 | [19] |
| 时间窗口大小 (k) | 5 | [19] |
| 初始步长 (α₀) | 0.9 | [19] |
| 衰减率 (ε) | 0.05 | [19] |
| 故障类别数 (c) | 3 | [19] |
| 训练/测试分割 | 80%/20% | [19] |

### PreGAN特有参数（GRU）
- GRU隐藏层大小：128
- GRU层数：2

### PreGAN+特有参数（Transformer）
- 模型维度 (d_model)：256 [11]
- 注意力头数 (nhead)：8 [11]
- 编码器层数：6 [11]
- 前馈网络维度：1024 [11]

### 训练说明

**重要**：FPE和Transformer编码器的训练是**自动进行**的，不需要单独的训练脚本。

#### 自动训练机制

1. **首次运行**：当运行 `PreGANRecovery` 或 `PreGANPlusRecovery` 时：
   - 如果 checkpoint 文件不存在（`simulator_FPE_16.ckpt` 或 `simulator_Transformer_16.ckpt`）
   - 或者 checkpoint 中的 `epoch == -1`
   - 系统会自动调用 `train_model()` 进行训练

2. **训练数据来源**：
   - 路径：`recovery/PreGANSrc/data/{env_name}/`
   - 文件：`time_series.npy` 和 `schedule_series.npy`
   - 这些数据通常来自阶段1的数据收集，或之前的实验运行

3. **训练参数**：
   - 训练轮数：50 epochs（在 `constants.py` 中定义）
   - 训练完成后模型自动保存到 `checkpoints/` 目录
   - 模型会被冻结（`freeze(model)`），后续只进行推理

#### 实际使用

```python
# 在阶段3运行GAN训练时，FPE/Transformer会自动训练（如果未训练）
recovery = PreGANRecovery(HOSTS, environment, training=True)
# 如果FPE未训练，会自动训练；如果已训练，直接加载
```

**注意**：不需要手动调用 `train_fpe_offline()` 或 `save_fpe_weights()`，这些功能已集成在 `load_models()` 方法中。

---

## 4. 阶段3：在线训练GAN

### 目的
训练生成器和判别器，学习最优迁移策略

### main.py参数配置

```python
# Global constants
NUM_SIM_STEPS = 1200        # 可根据需要调整，论文使用1200个间隔 [19]
HOSTS = 16
CONTAINERS = 16
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 300
NEW_CONTAINERS = 5

# Workload
workload = BWGD2(NEW_CONTAINERS, 1.5)

# Scheduler
scheduler = GOBIScheduler('energy_latency_16')

# Recovery - 训练PreGAN (FPE-GAN)
recovery = PreGANRecovery(HOSTS, environment, training=True)
# FPE会自动加载（如果已训练）或自动训练（如果未训练）
# 训练完成后FPE会被自动冻结

# 或训练PreGANPlus (TF-GAN)
recovery = PreGANPlusRecovery(HOSTS, environment, training=True)
# Transformer会自动加载（如果已训练）或自动训练（如果未训练）
# 训练完成后Transformer会被自动冻结

# 或训练PreGANPlusEnhanced (MAMO-GAN)
recovery = PreGANPlusEnhancedRecovery(HOSTS, environment, training=True)
# Transformer会自动加载（如果已训练）或自动训练（如果未训练）
```

### GAN训练参数（论文）

| 参数 | 取值 | 论文依据 |
|------|------|---------|
| 生成器学习率 | 0.0001 | [15] |
| 判别器学习率 | 0.0001 | [15] |
| QoS权重 (β) | 0.5 | [16][22] |
| 训练间隔数 | 300 | 实际使用（论文使用1200 [19]，可根据需要调整） |
| **注意** | - | 如果效果不好，可以多训练几次 |
| **内存优化** | - | 中间保存频率：每50步保存一次stats |

### PreGANPlusEnhanced (MAMO-GAN) 多目标权重配置

| 参数 | 取值 | 说明 |
|------|------|------|
| energy_weight | 0.35 | 能量优化权重（保持能量优势） |
| response_time_weight | 0.45 | 响应时间约束权重（优先优化响应时间） |
| migration_cost_weight | 0.2 | 迁移成本约束权重（通过迁移控制机制限制） |
| sla_threshold | 2800.0 | SLA阈值（秒） |
| migration_cost_threshold | 130 | 迁移成本阈值 |

**权重调整说明**：
- 目标：使得能量和响应时间都比PreGANPlus好
- 当前配置：响应时间权重从0.3增加到0.45，优先优化响应时间
- 如果效果不理想，可以考虑增加训练步数（见数据量建议）

### 数据量建议

**论文配置 vs 实际配置**：

| 阶段 | 论文配置 | 实际配置 | 建议 |
|------|---------|---------|------|
| 阶段1：数据收集 | 1000步 | 500步 | 如果效果不好，可增加到1000步 |
| 阶段3：GAN训练 | 1200步 | 300步 | 如果效果不好，可逐步增加到600-800步 |

**数据量不足的潜在影响**：
- 数据收集不足可能导致故障模式覆盖不全，编码器学习不充分
- GAN训练不足可能导致生成器未充分学习最优迁移策略，多目标优化未收敛

**建议**：
1. **先使用当前配置（500步数据收集，300步GAN训练）测试权重调整效果**
2. **如果效果仍不理想，再考虑增加训练步数**：
   - 阶段1：增加到1000步（重新收集数据）
   - 阶段3：逐步增加到600-800步（可在已有300步基础上继续训练）

### 运行命令
```bash
python main.py -e "" -m 0
```

### 输出
- 训练好的模型：`models/pregan_complete.pth` 或 `models/pregan_plus_complete.pth`

---

## 5. 阶段4：测试评估

### 目的
在独立测试集上评估模型性能

### main.py参数配置

```python
# Global constants
NUM_SIM_STEPS = 100         # 论文使用100个间隔 [17]
HOSTS = 16
CONTAINERS = 16
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 300
NEW_CONTAINERS = 5

# Workload
workload = BWGD2(NEW_CONTAINERS, 1.5)

# Scheduler
scheduler = GOBIScheduler('energy_latency_16')

# Recovery - 测试PreGAN (FPE-GAN)
recovery = PreGANRecovery(HOSTS, environment, training=False)  # ⚠️ training=False
# 会自动加载已训练的checkpoint（FPE, Generator, Discriminator）

# 或测试PreGANPlus (TF-GAN)
recovery = PreGANPlusRecovery(HOSTS, environment, training=False)
# 会自动加载已训练的checkpoint（Transformer, Generator, Discriminator）

# 或测试PreGANPlusEnhanced (MAMO-GAN)
recovery = PreGANPlusEnhancedRecovery(HOSTS, environment, training=False)
# 会自动加载已训练的checkpoint（Transformer, Generator, Discriminator）

# 对比基线方法（传统方法不需要训练，可直接运行）
# recovery = PCFTRecovery(HOSTS, environment, training=False)  # 基于规则，无需训练
# recovery = DFTMRecovery(HOSTS, environment, training=False)  # 基于规则，无需训练
# recovery = ECLBRecovery(HOSTS, environment, training=False)  # 基于规则，无需训练
# recovery = CMODLBRecovery(HOSTS, environment, training=False)  # 有FCN模型，会自动训练（如果未训练）
```

### PreGAN+在线调优参数（可选）

| 参数 | 取值 | 论文依据 |
|------|------|---------|
| 标注数据比例 (ξ) | 0.4 | [13][15][24] |
| 调优学习率 | 0.00001 | [15] |

### 运行命令
```bash
python main.py -e "" -m 0
```

### 输出
- 性能评估报告：`logs/RPiEdge_BWGD2_100_16_16_1000_10000_300_5/`
- 包含能耗、响应时间、SLO违约率等指标

---

## 6. 评估指标

### 故障检测指标
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- 改进率（Improvement Ratio）[18][21]

### 故障诊断指标
- HitRate@100% [18]
- NDCG@100% [18]
- 故障类型分类准确率

### QoS性能指标
- 能耗（Energy Consumption，KW-hr）[17][22]
- 响应时间（Response Time，秒）[16][22]
- SLO违约率（SLO Violation Rate）[18][22]
- 开销比率（Overhead Ratio）[22][24]

---

## 7. 论文结果对照（Table III）

| 方法 | 能耗 (KW-hr) | 响应时间 (s) | SLO违约率 | 改进率 |
|------|-------------|-------------|----------|--------|
| PreGAN+ | 8.8721 | - | 0.0568 | 0.9014 |
| PreGAN | 9.0121 | - | 0.0621 | 0.7895 |
| CMODLB | 9.8234 | - | 0.0717 | 0.6234 |

**论文依据：** [22]

---

## 8. 快速参考表

| 阶段 | NUM_SIM_STEPS | training | recovery类型 | 论文依据 |
|------|--------------|----------|-------------|---------|
| 数据收集 | 500 | False | Recovery | 实际使用（论文1000 [19]） |
| FPE训练 | - | True | 单独脚本 | [11][19] |
| GAN训练 | 300 | True | PreGAN/PreGAN+ | 实际使用（论文1200 [19]） |
| 测试 | 100 | False | PreGAN/PreGAN+/PreGANPlusEnhanced | [17] |

**关键参数：**
- HOSTS = 16 [17]
- NEW_CONTAINERS = 5 [17]
- INTERVAL_TIME = 300 [17]
- embedding_size = 8 [19]
- window_size = 5 [19]
- α₀ = 0.9 [19]
- ε = 0.05 [19]
- β = 0.5 [16][22]
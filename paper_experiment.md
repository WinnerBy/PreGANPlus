# PreGAN/PreGAN+ 实验参数配置与流程文档

---

## 1. 实验流程概览

```
阶段1: 数据收集 (1000步)
   ↓
阶段2+3: 编码器训练（自动）+ GAN训练 (1200步)
   - PreGAN/PreGANPlus/PreGANPlusEnhanced
   - 编码器训练使用阶段1的1000步数据（离线训练）
   - GAN训练使用1200步数据（在线训练）
   ↓
阶段2: CMODLB编码器训练 (1200步)
   - CMODLB（不需要GAN训练）
   ↓
阶段4: 测试评估 (100步)
   - 所有方法对比
```

**注意**：阶段2和阶段3已合并，编码器训练在GAN训练开始时自动进行。

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

## 3. 阶段2+3：编码器训练 + GAN训练（合并）

### 目的
1. 自动训练编码器（FPE/Transformer/FCN），学习故障模式
2. 在线训练GAN，学习最优迁移策略

**重要更新**：阶段2和阶段3已合并。编码器训练在GAN训练开始时自动进行。

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
# 在阶段2+3运行GAN训练时，FPE/Transformer会自动训练（如果未训练）
recovery = PreGANRecovery(HOSTS, environment, training=True)
# 如果FPE未训练，会自动训练；如果已训练，直接加载
# 编码器训练完成后，会继续运行1200步的GAN训练
```

**注意**：
- 不需要手动调用 `train_fpe_offline()` 或 `save_fpe_weights()`，这些功能已集成在 `load_models()` 方法中
- 编码器训练使用阶段1收集的1000步数据（离线训练）
- GAN训练使用1200步数据（在线训练）

---

## 4. 阶段2+3：编码器训练 + GAN训练（合并）

### 目的
1. 自动训练编码器（FPE/Transformer），学习故障模式
2. 在线训练GAN，学习最优迁移策略

### main.py参数配置

```python
# Global constants
NUM_SIM_STEPS = 1200        # GAN训练步数，论文使用1200个间隔 [19]
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
# 然后继续运行1200步的GAN训练

# 或训练PreGANPlus (TF-GAN)
recovery = PreGANPlusRecovery(HOSTS, environment, training=True)
# Transformer会自动加载（如果已训练）或自动训练（如果未训练）
# 训练完成后Transformer不会被冻结（支持在线调优）
# 然后继续运行1200步的GAN训练

# 或训练PreGANPlusEnhanced (MAMO-GAN)
recovery = PreGANPlusEnhancedRecovery(HOSTS, environment, training=True)
# Transformer会自动加载（如果已训练）或自动训练（如果未训练）
# 注意：PreGANPlusEnhanced与PreGANPlus共享相同的Transformer编码器
# 如果PreGANPlus已训练，PreGANPlusEnhanced会直接使用已训练的Transformer（不重复训练）
# 然后继续运行1200步的多目标GAN训练
```

**重要说明**：
- ✅ **PreGAN**：使用FPE编码器（`checkpoints/simulator_FPE_16.ckpt`）
- ✅ **PreGANPlus**：使用Transformer编码器（`checkpointsplus/simulator_Transformer_16.ckpt`）
- ✅ **PreGANPlusEnhanced**：**共享PreGANPlus的Transformer编码器**
  - 两者使用相同的checkpoint文件：`checkpointsplus/simulator_Transformer_16.ckpt`
  - 如果先训练PreGANPlus，Transformer会被训练并保存
  - 然后训练PreGANPlusEnhanced时，会直接加载已训练的Transformer，**不重复训练**
  - 这样可以节省训练时间，避免重复训练

### GAN训练参数（论文）

| 参数 | 取值 | 论文依据 |
|------|------|---------|
| 生成器学习率 | 0.0001 | [15] |
| 判别器学习率 | 0.0001 | [15] |
| QoS权重 (β) | 0.5 | [16][22] |
| 训练间隔数 | 1200 | 论文配置 [19]，可根据需要调整 |
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

### 运行命令

**使用一键脚本（推荐）**：
```bash
bash scripts/run_paper_experiment.sh
```

**或手动运行**：
```bash
# 配置阶段2+3
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python3 main.py -e "" -m 0
```

### 输出
- 编码器模型：`recovery/PreGANSrc/checkpoints/simulator_FPE_16.ckpt`（PreGAN）
- 编码器模型：`recovery/PreGANSrc/checkpointsplus/simulator_Transformer_16.ckpt`（PreGANPlus/PreGANPlusEnhanced）
- GAN模型：`recovery/PreGANSrc/checkpoints/simulator_Gen_16.ckpt` 和 `simulator_Disc_16.ckpt`
- 训练日志：`experiment_logs/paper_experiment/stage2+3_training_*.log`

---

## 5. 阶段2：CMODLB编码器训练

### 目的
训练CMODLB的FCN编码器（CMODLB不需要GAN训练）

### main.py参数配置

```python
# Global constants
NUM_SIM_STEPS = 1200        # 使用1200步确保环境稳定
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

# Recovery
recovery = CMODLBRecovery(HOSTS, environment, training=False)
# FCN编码器会自动加载（如果已训练）或自动训练（如果未训练）
# 训练完成后FCN会被自动冻结
```

### 运行命令

```bash
python3 scripts/paper_experiment_stage2_encoder_training.py --method CMODLB
python3 main.py -e "" -m 0
```

### 输出
- FCN编码器模型：`recovery/CMODLBSrc/checkpoints/simulator_FCN_16.ckpt`
- 训练日志：`experiment_logs/paper_experiment/stage2_encoder_training_CMODLB_*.log`

---

## 6. 阶段4：测试评估

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
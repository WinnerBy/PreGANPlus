# MAMO-GAN 详细设计文档

**Migration-Aware Multi-Objective GAN for Fault-Tolerant Edge Computing**

**创建日期**: 2026-01-14  
**方法代码名称**: PreGANPlusEnhanced  
**论文命名**: MAMO-GAN

---

## 📋 概述

### 方法定位

MAMO-GAN (Migration-Aware Multi-Objective GAN) 是对 TF-GAN (PreGANPlus) 的改进方法，通过引入迁移感知机制和多目标优化，在保持能量优势的同时，显著改善了响应时间和SLA违约率。

### 核心设计理念

1. **迁移感知Generator**: 通过迁移成本预测和迁移门控机制，减少不必要的任务迁移
2. **多目标Discriminator**: 同时优化能量、响应时间、迁移成本三个目标
3. **多目标训练策略**: 平衡多个优化目标，避免单一目标过度优化
4. **迁移控制机制**: 推理阶段的cooldown_period和max_migrations_per_step进一步控制迁移

### 相对于TF-GAN的改进

| 方面 | TF-GAN | MAMO-GAN | 改进说明 |
|------|--------|----------|---------|
| Generator | 标准Generator | Migration-Aware Generator | 增加迁移成本预测和迁移门控 |
| Discriminator | 标准Discriminator | Multi-Objective Discriminator | 同时预测能量、响应时间、迁移成本 |
| 训练目标 | 单一目标（能量） | 多目标（能量+响应时间+迁移成本） | 平衡多个优化目标 |
| 迁移控制 | 无 | 多层迁移控制机制 | 训练和推理阶段都有控制 |

---

## 🏗️ 架构设计

### 编码器架构（与TF-GAN相同）

MAMO-GAN使用与TF-GAN相同的Transformer编码器：

- **模型名称**: `Transformer_16`
- **输入**: 时间序列数据 (time_series) + 调度矩阵 (schedule_data)
- **输出**: 
  - 异常检测分数 (anomaly_scores): [16, 2] - 每个主机的异常概率
  - 原型向量 (prototypes): [16, 2] - 每个主机的特征表示
- **架构特点**: 
  - 使用Transformer的注意力机制
  - 能够捕获时间序列的长期依赖关系
  - 相比FPE编码器，具有更强的序列建模能力

**编码器训练**:
- 与 TF-GAN 共用 Transformer_16，使用阶段1数据离线训练（当前推荐 400×8，见 [Stage1_Data_And_Analysis](../02_Experiments/Stage1_Data_And_Analysis.md)）
- Epochs 由 `constants.py` 的 `num_epochs` 决定（当前 300）；实践中常与 PreGANPlus 共用一次 encoder-only 训练（如 150 epoch）
- 训练完成后冻结编码器参数

---

### Migration-Aware Generator架构

#### 网络结构

```python
Gen_16_MigrationAware(
    embedding_proj: Linear(2 -> 64)           # Embedding投影层
    schedule_proj: Linear(16 -> 64)            # Schedule投影层
    cross_attn: MultiheadAttention(64, 4)     # 交叉注意力
    self_attn: MultiheadAttention(64, 4)      # 自注意力
    migration_cost_predictor: Sequential(      # 迁移成本预测模块
        Linear(64 -> 32),
        LeakyReLU(0.2),
        Linear(32 -> 1),
        ReLU()
    )
    migration_gate: Sequential(                # 迁移门控模块
        Linear(64 -> 32),
        LeakyReLU(0.2),
        Linear(32 -> 1),
        Sigmoid()
    )
    output: Sequential(                       # 输出层
        Linear(64 -> 64),
        LeakyReLU(),
        Linear(64 -> 16),
        Tanh()
    )
)
```

#### 输入输出

- **输入**:
  - `e`: [16, 2] - 编码器输出的embedding（原型向量）
  - `s`: [16, 16] - 当前调度矩阵
- **输出**:
  - `new_schedule`: [16, 16] - 新的调度矩阵
  - `predicted_migration_cost`: [1] - 预测的迁移次数

#### 关键组件详解

##### 1. Embedding和Schedule投影层

将不同维度的输入投影到相同的隐藏维度（64）：
- Embedding投影: `Linear(2 -> 64)` - 将2维原型向量投影到64维
- Schedule投影: `Linear(16 -> 16)` - 将16维调度向量投影到64维

##### 2. 交叉注意力机制 (Cross-Attention)

- **Query**: Schedule投影 (`s_proj`)
- **Key/Value**: Embedding投影 (`e_proj`)
- **作用**: 让embedding（故障信息）指导schedule的更新
- **输出**: 经过embedding指导的schedule表示

##### 3. 自注意力机制 (Self-Attention)

- **Query/Key/Value**: 交叉注意力后的schedule表示
- **作用**: 在schedule序列内部建模容器之间的依赖关系
- **输出**: 考虑容器间依赖的schedule表示

##### 4. 迁移成本预测模块 (Migration Cost Predictor)

**新增组件**，用于预测当前调度决策可能产生的迁移次数：

```python
migration_cost_predictor = Sequential(
    Linear(64 -> 32),
    LeakyReLU(0.2),
    Linear(32 -> 1),
    ReLU()  # 确保预测值非负
)
```

- **输入**: 融合后的特征向量的平均值 `[1, 64]`
- **输出**: 预测的迁移次数 `[1]`，范围 [0, 300]
- **作用**: 在生成调度决策前，预测可能产生的迁移成本

##### 5. 迁移门控模块 (Migration Gate)

**新增组件**，为每个容器预测迁移概率：

```python
migration_gate = Sequential(
    Linear(64 -> 32),
    LeakyReLU(0.2),
    Linear(32 -> 1),
    Sigmoid()  # 输出0-1之间的概率
)
```

- **输入**: 融合后的特征向量 `[16, 64]`（每个容器一个）
- **输出**: 迁移概率 `[16]`，范围 [0, 1]
- **作用**: 为每个容器预测是否应该迁移的概率

##### 6. 输出层与迁移约束

输出层生成调度增量，并应用迁移约束：

```python
# 生成原始增量
del_s_raw = output(s_fused)  # [16, 16]

# 应用迁移约束
migration_penalty = 1.0 - 0.9 * migration_gates  # 迁移概率高时减小增量
migration_cost_penalty = clamp(1.0 - predicted_cost / 150.0, 0.2, 1.0)  # 预测成本高时减小增量

# 最终增量（考虑迁移约束）
del_s = 4 * del_s_raw * migration_penalty * migration_cost_penalty

# 新调度
new_schedule = s + del_s
```

**迁移约束机制**:
- 如果容器的迁移概率高，大幅减小增量幅度（最多减小90%）
- 如果预测的迁移成本高，进一步减小增量
- 通过双重约束，有效减少不必要的迁移

---

### Multi-Objective Discriminator架构

#### 网络结构

```python
Disc_16_MultiObjective(
    shared: Sequential(                       # 共享特征提取层
        Linear(512 -> 256),
        LeakyReLU(0.2),
        Dropout(0.1),
        Linear(256 -> 128),
        LeakyReLU(0.2),
        Dropout(0.1)
    )
    classifier: Sequential(                   # 分类头（更好/更差）
        Linear(128 -> 64),
        LeakyReLU(0.2),
        Linear(64 -> 2),
        Softmax()
    )
    energy_predictor: Sequential(            # 能量预测头
        Linear(128 -> 64),
        LeakyReLU(0.2),
        Linear(64 -> 1)
    )
    response_time_predictor: Sequential(      # 响应时间预测头
        Linear(128 -> 64),
        LeakyReLU(0.2),
        Linear(64 -> 1)
    )
    migration_cost_predictor: Sequential(     # 迁移成本预测头
        Linear(128 -> 64),
        LeakyReLU(0.2),
        Linear(64 -> 1),
        ReLU()
    )
)
```

#### 输入输出

- **输入**:
  - `o`: [16, 16] - 原始调度矩阵
  - `n`: [16, 16] - 新调度矩阵
- **输出**:
  - `class_probs`: [2] - 分类概率（原始更好 vs 新调度更好）
  - `energy_pred`: [1] - 预测的能量消耗
  - `response_time_pred`: [1] - 预测的响应时间
  - `migration_cost_pred`: [1] - 预测的迁移次数

#### 关键组件详解

##### 1. 共享特征提取层

将原始调度和新调度拼接后提取特征：
- 输入: `concat([o.view(-1), n.view(-1)])` → `[512]`
- 输出: `[128]` 特征向量
- 作用: 提取两个调度的联合特征表示

##### 2. 分类头 (Classifier)

判断新调度是否优于原始调度：
- 输出: `[2]` - Softmax概率，`[0]`表示原始更好，`[1]`表示新调度更好
- 训练目标: 基于综合评分（能量+响应时间+迁移成本）判断

##### 3. 能量预测头 (Energy Predictor)

预测调度方案的能量消耗：
- 输出: `[1]` - 预测的能量值（单位：Kilowatt-hr）
- 训练目标: 最小化预测误差

##### 4. 响应时间预测头 (Response Time Predictor)

预测调度方案的响应时间：
- 输出: `[1]` - 预测的响应时间（单位：秒）
- 训练目标: 最小化预测误差，特别关注SLA阈值

##### 5. 迁移成本预测头 (Migration Cost Predictor)

预测调度方案产生的迁移次数：
- 输出: `[1]` - 预测的迁移次数（非负）
- 训练目标: 最小化预测误差

---

## 🎓 训练流程

### 多目标GAN训练

MAMO-GAN使用多目标训练函数 `train_gan_multiobjective`，同时优化三个目标：

#### 训练参数配置

```python
# 多目标权重（经过优化验证）
energy_weight = 0.004            # 能量优化权重
response_time_weight = 0.14      # 响应时间约束权重（已验证有效）
migration_cost_weight = 0.04     # 迁移成本约束权重

# 阈值参数
sla_threshold = 2800.0           # SLA阈值（秒）
migration_cost_threshold = 110   # 迁移成本阈值
```

#### Discriminator训练

Discriminator的损失函数包含四个部分：

1. **分类损失** (Classification Loss):
   - 基于综合评分判断新调度是否更好
   - 综合评分 = `0.8 * energy + 0.2 * response_time + 0.01 * migration_count`

2. **能量预测损失** (Energy Prediction Loss):
   - MSE损失，预测值与实际值的差异
   - 归一化: `loss / max(energy)^2`

3. **响应时间预测损失** (Response Time Prediction Loss):
   - MSE损失，预测值与实际值的差异
   - 归一化: `loss / sla_threshold^2`

4. **迁移成本预测损失** (Migration Cost Prediction Loss):
   - MSE损失，预测值与实际值的差异
   - 归一化: `loss / migration_cost_threshold^2`

**总损失**:
```python
disc_loss = class_loss + 
            0.2 * energy_loss_norm +
            0.1 * response_time_loss_norm +
            0.1 * migration_cost_loss_norm
```

#### Generator训练

Generator的损失函数包含四个部分：

1. **分类损失** (Classification Loss):
   - 鼓励Discriminator认为新调度更好
   - 目标: `[0, 1]` (新调度更好)

2. **能量约束损失** (Energy Constraint Loss):
   - 鼓励预测更低的能量
   - `gen_energy_loss = ReLU(energy_pred - orig_energy + 0.1) / orig_energy`

3. **响应时间约束损失** (Response Time Constraint Loss):
   - 惩罚超过SLA阈值的响应时间
   - `gen_response_time_loss = response_time_weight * ReLU(rt_pred - sla_threshold) / sla_threshold`
   - 同时惩罚实际响应时间超过SLA的情况

4. **迁移成本约束损失** (Migration Cost Constraint Loss):
   - **关键组件**，使用立方惩罚机制
   - 惩罚预测迁移成本超过阈值的情况
   - `gen_migration_cost_loss = migration_cost_weight * (excess^3 + excess^2 + excess)`
   - 同时惩罚实际迁移成本超过阈值的情况（权重2.0）

**总损失**:
```python
gen_loss = gen_class_loss +
           energy_weight * gen_energy_loss +
           response_time_weight * (gen_response_time_loss + gen_actual_response_time_loss) +
           migration_cost_weight * (gen_migration_cost_loss + gen_actual_migration_cost_loss)
```

#### 训练流程

1. **阶段2**: 编码器训练（与 TF-GAN 共用 Transformer_16）
   - 使用阶段1数据离线训练（当前 400×8）
   - Epochs 由 constants 决定（当前 300）
   - 训练完成后冻结

2. **阶段2（GAN 部分）**: GAN 训练
   - 运行 1200 步模拟（当前配置，见 [Experiment_Setup_And_Fault_Design](../02_Experiments/Experiment_Setup_And_Fault_Design.md)）
   - 每步 8 容器，每个间隔进行在线 GAN 训练
   - 可选编码器调优（tune_model）
   - 保存 checkpoint

---

## 🔍 推理流程

### 异常检测

与TF-GAN相同，使用Transformer编码器进行异常检测：

1. 获取当前时间序列数据
2. 运行编码器，得到异常检测分数和原型向量
3. 如果所有主机都未检测到异常，返回原始决策

### 调度决策生成

1. **生成新调度**:
   - 使用Migration-Aware Generator生成新调度
   - 同时获得预测的迁移成本

2. **多目标评估**:
   - 使用Multi-Objective Discriminator评估新调度
   - 获得分类概率、能量预测、响应时间预测、迁移成本预测

3. **决策选择**:
   - 如果原始调度更好（`class_probs[0] > class_probs[1]`），返回原始决策
   - 否则，应用迁移控制机制

### 迁移控制机制

MAMO-GAN在推理阶段实现了三层迁移控制：

#### 1. Cooldown机制

防止容器频繁迁移（迁移抖动）：

```python
migration_cooldown = {}  # {container_id: last_migration_epoch}
cooldown_period = 10     # 冷却期：10个间隔
```

- 每个容器迁移后，记录迁移时间
- 在冷却期内，禁止该容器再次迁移
- 有效防止迁移抖动

#### 2. 每步迁移限制

限制每个间隔的最大迁移数：

```python
max_migrations_per_step = 2  # 每个间隔最多2个迁移
```

- 收集所有潜在迁移，按优先级排序
- 只保留优先级最高的N个迁移
- 优先级基于调度变化幅度计算

#### 3. 全局迁移预算

限制整个测试过程的迁移总数：

```python
strict_migration_limit = 173  # 全局迁移限制
total_migrations = 0          # 当前迁移计数
```

- 跟踪总迁移数
- 如果达到限制，禁止后续迁移
- 确保迁移数在合理范围内

#### 4. 预测成本阈值

基于Generator预测的迁移成本进一步限制：

```python
migration_cost_threshold = 110  # 迁移成本阈值
```

- 如果预测的迁移成本超过阈值，且当前有多个潜在迁移
- 只保留优先级最高的1个迁移
- 进一步减少迁移

---

## 🔑 关键技术点

### 1. 迁移感知机制

**设计思想**: 在生成调度决策时，同时考虑迁移成本，避免不必要的迁移。

**实现方式**:
- Generator预测迁移成本
- Generator为每个容器预测迁移概率
- 在生成调度增量时，应用迁移约束

**效果**: 
- 相比TF-GAN，迁移次数仅增加10.19%（173 vs 157）
- 在保持迁移控制的同时，显著改善了能耗和响应时间

### 2. 多目标优化

**设计思想**: 同时优化能量、响应时间、迁移成本三个目标，避免单一目标过度优化。

**实现方式**:
- Discriminator同时预测三个目标
- Generator损失函数包含三个约束项
- 使用权重平衡不同目标的重要性

**效果**:
- 能耗降低1.18%
- 响应时间改善8.65%
- SLA违规减少15.93%

### 3. 迁移控制策略

**设计思想**: 在推理阶段通过多层机制控制迁移，确保迁移数在合理范围内。

**实现方式**:
- Cooldown机制防止抖动
- 每步限制防止突发迁移
- 全局预算确保总体控制
- 预测阈值进一步优化

**效果**:
- 迁移次数控制在合理范围（173次）
- 相比传统方法，迁移数显著减少（21%-84%）

### 4. 立方惩罚机制

**设计思想**: 对超过阈值的迁移成本使用立方惩罚，使超过阈值时惩罚更严重。

**实现方式**:
```python
gen_migration_cost_loss = migration_cost_weight * (excess^3 + excess^2 + excess)
```

**效果**:
- 有效抑制迁移成本超过阈值的情况
- 训练更稳定，收敛更快

---

## 📊 与 TF-GAN 的对比

| 方面 | TF-GAN | MAMO-GAN | 改进 |
|------|--------|----------|------|
| Generator | 标准 Generator | Migration-Aware Generator | ✅ 迁移感知 |
| Discriminator | 标准 Discriminator | Multi-Objective Discriminator | ✅ 多目标优化 |
| 训练目标 | 单一（能量） | 多目标（能量+响应时间+迁移成本） | ✅ 平衡优化 |
| 迁移控制 | 无 | 多层控制机制 | ✅ 迁移控制 |

**当前实验数值**（Stage3 挑选 5 次，600 步×10 容器）见 [Stage3_Results_Analysis](../03_Results/Stage3_Results_Analysis.md)：PreGANPlusEnhanced 在迁移、能耗、稳定性上综合最优。

---

## 💡 设计优势

1. **迁移感知**: 在生成决策时考虑迁移成本，减少不必要迁移
2. **多目标平衡**: 同时优化多个目标，避免单一目标过度优化
3. **迁移控制**: 多层控制机制确保迁移数在合理范围
4. **性能提升**: 在保持迁移控制的同时，显著改善能耗和响应时间

---

## 📝 代码位置

- **实现文件**: `recovery/PreGANPlusEnhanced.py`
- **Generator模型**: `recovery/PreGANSrc/src/models.py` - `Gen_16_MigrationAware`
- **Discriminator模型**: `recovery/PreGANSrc/src/models.py` - `Disc_16_MultiObjective`
- **训练函数**: `recovery/PreGANSrc/src/train_multiobjective.py` - `train_gan_multiobjective`

---

**最后更新**: 2026-01-14

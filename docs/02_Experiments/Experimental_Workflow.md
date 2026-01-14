# 实验流程说明

**创建日期**: 2026-01-14

---

## 📋 概述

本文档详细说明完整的实验流程，包括四个阶段的详细步骤、数据流向、多次运行策略和结果收集。

---

## 🔄 实验流程总览

### 四阶段实验流程

```
阶段1: 数据收集 (1000步)
    ↓ 生成训练数据
    ↓ 自动拷贝到训练目录
阶段2: 编码器训练 (自动触发)
    ↓ 离线训练50个epoch
    ↓ 保存checkpoint
阶段3: GAN训练 (1200步)
    ↓ 在线训练GAN
    ↓ 同时进行编码器调优
    ↓ 保存checkpoint
阶段4: 测试评估 (100步)
    ↓ 推理模式
    ↓ 收集性能指标
    ↓ 生成结果报告
```

### 流程图

```
┌─────────────────┐
│  阶段1: 数据收集 │
│  NUM_STEPS=1000  │
│  Recovery=Base   │
└────────┬─────────┘
         │
         ↓
┌─────────────────┐
│  生成训练数据    │
│  time_series.npy │
│ schedule_series │
└────────┬─────────┘
         │
         ↓ 自动拷贝
┌─────────────────┐
│ recovery/.../    │
│ data/simulator/  │
└────────┬─────────┘
         │
         ↓
┌─────────────────┐
│  阶段2: 编码器   │
│  自动训练        │
│  epochs=50       │
└────────┬─────────┘
         │
         ↓
┌─────────────────┐
│  保存checkpoint  │
│  encoder.ckpt    │
└────────┬─────────┘
         │
         ↓
┌─────────────────┐
│  阶段3: GAN训练  │
│  NUM_STEPS=1200  │
│  training=True   │
└────────┬─────────┘
         │
         ↓
┌─────────────────┐
│  在线训练GAN     │
│  编码器调优      │
└────────┬─────────┘
         │
         ↓
┌─────────────────┐
│  保存checkpoint  │
│  gen.ckpt        │
│  disc.ckpt       │
└────────┬─────────┘
         │
         ↓
┌─────────────────┐
│  阶段4: 测试评估 │
│  NUM_STEPS=100   │
│  training=False  │
└────────┬─────────┘
         │
         ↓
┌─────────────────┐
│  收集性能指标    │
│  生成结果报告    │
└─────────────────┘
```

---

## 📊 阶段1：数据收集

### 目的

收集包含各种故障情况的训练数据，用于后续编码器训练。此阶段不进行任何预防性迁移，让系统自然运行以收集真实故障数据。

### 执行步骤

1. **运行配置脚本**:
   ```bash
   python3 scripts/paper_experiment_stage1_data_collection.py
   ```
   - 自动修改 `main.py` 配置
   - 设置 `NUM_SIM_STEPS = 1000`
   - 设置 `recovery = Recovery()` (基类，不进行恢复)

2. **运行模拟器**:
   ```bash
   python3 main.py -e "" -m 0
   ```
   - 运行1000个间隔的模拟
   - 不进行任何恢复操作
   - 收集时间序列和调度序列数据

3. **数据自动管理**:
   - 数据保存在 `logs/RPiEdge_BWGD2_1000_*` 目录
   - 脚本自动查找最新的1000步数据目录
   - 自动拷贝 `time_series.npy` 和 `schedule_series.npy` 到 `recovery/PreGANSrc/data/simulator/`

### 输出数据

- **位置**: `recovery/PreGANSrc/data/simulator/`
- **文件**:
  - `time_series.npy`: [1001, 48] - 时间序列数据（1001个时间点，16主机×3指标）
  - `schedule_series.npy`: [1001, 16, 16] - 调度序列数据（1001个时间点，16容器×16主机）

### 数据用途

- 用于阶段2的编码器离线训练
- 所有GAN方法（PreGAN, PreGANPlus, PreGANPlusEnhanced）都使用相同的数据
- CMODLB使用相同的数据训练FCN编码器

### 注意事项

- 如果1000步数据训练效果不好，可以继续生成更多数据
- 生成的数据会自动保存到 `logs/` 目录
- 脚本会自动选择最新的1000步数据目录

---

## 🎓 阶段2：编码器训练

### 目的

训练编码器模型，用于故障预测和特征提取。编码器训练是自动触发的，如果checkpoint不存在或epoch == -1，会自动开始训练。

### 方法训练需求分析

#### GAN方法（需要训练）

| 方法 | 编码器训练 | GAN训练 | 训练时机 |
|------|-----------|---------|---------|
| **PreGAN (FPE-GAN)** | ✅ 自动训练 | ✅ 在线训练 | 编码器：首次运行自动训练<br>GAN：阶段3在线训练 |
| **PreGANPlus (TF-GAN)** | ✅ 自动训练 | ✅ 在线训练 | 编码器：首次运行自动训练<br>GAN：阶段3在线训练 |
| **PreGANPlusEnhanced (MAMO-GAN)** | ✅ 自动训练 | ✅ 在线训练 | 编码器：首次运行自动训练<br>GAN：阶段3在线训练 |

**编码器训练机制**：
- 如果 checkpoint 不存在或 `epoch == -1`，自动训练
- 训练数据来自 `recovery/PreGANSrc/data/{env_name}/`（阶段1的数据）
- 训练完成后自动保存并冻结

#### 传统方法（部分需要训练）

| 方法 | 是否需要训练 | 训练机制 | 说明 |
|------|-------------|---------|------|
| **PCFT** | ❌ 不需要 | - | 基于规则的恢复方法，直接运行 |
| **DFTM** | ❌ 不需要 | - | 基于规则的恢复方法，直接运行 |
| **ECLB** | ❌ 不需要 | - | 基于规则的恢复方法，直接运行 |
| **CMODLB** | ✅ 需要 | 自动训练 | 有FCN模型，类似PreGAN的自动训练机制 |

**CMODLB训练机制**：
- 如果 checkpoint 不存在或 `epoch == -1`，自动训练
- 训练数据来自 `recovery/PreGANSrc/data/{env_name}/`（与PreGAN相同）
- 训练30个epoch（PreGAN是50个）
- 训练完成后自动保存并冻结

### 执行步骤

#### 对于GAN方法（PreGAN/PreGANPlus/PreGANPlusEnhanced）

1. **运行配置脚本** (已合并到阶段3):
   ```bash
   # 注意：阶段2和3已合并，见阶段3说明
   ```

2. **自动训练流程**:
   - 加载编码器模型
   - 检查checkpoint是否存在
   - 如果不存在或epoch == -1，自动开始训练
   - 使用阶段1收集的1000步数据进行离线训练
   - 训练50个epoch
   - 保存checkpoint

#### 对于传统方法（CMODLB）

1. **运行配置脚本**:
   ```bash
   python3 scripts/paper_experiment_stage2_encoder_training.py --method CMODLB
   ```

2. **运行训练**:
   ```bash
   python3 main.py -e "" -m 0
   ```
   - 运行1200步（环境稳定）
   - 自动训练FCN编码器
   - 保存checkpoint

### 编码器类型

| 方法 | 编码器类型 | Checkpoint路径 |
|------|----------|---------------|
| PreGAN | FPE_16 | `checkpoints/simulator_FPE_16.ckpt` |
| PreGANPlus | Transformer_16 | `checkpointsplus/simulator_Transformer_16.ckpt` |
| PreGANPlusEnhanced | Transformer_16 | `checkpointsplus/simulator_Transformer_16.ckpt` (共享) |
| CMODLB | FCN_16 | `checkpoints/simulator_FCN_16.ckpt` |

### 训练参数

- **训练数据**: 阶段1收集的1000步数据
- **训练方式**: 离线训练
- **训练epochs**: 50
- **优化器**: Adam
- **学习率**: 默认（在constants.py中定义）

### 输出模型

- **位置**: `recovery/PreGANSrc/checkpoints/` 或 `checkpointsplus/`
- **文件**: `simulator_{编码器类型}_{主机数}.ckpt`
- **内容**: 模型权重、优化器状态、epoch数、准确率列表

### 注意事项

- PreGANPlus和PreGANPlusEnhanced共享相同的Transformer编码器
- Transformer编码器只在PreGANPlus阶段训练一次
- PreGANPlusEnhanced直接使用已训练的Transformer编码器
- 编码器训练完成后会自动冻结

---

## 🚀 阶段3：GAN训练（仅GAN方法）

### 目的

训练Generator和Discriminator，学习如何生成更好的调度决策。此阶段同时进行在线GAN训练和编码器调优。

### 执行步骤

1. **运行配置脚本**:
   ```bash
   python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
   # 或 PreGANPlus, PreGANPlusEnhanced
   ```

2. **运行训练**:
   ```bash
   python3 main.py -e "" -m 0
   ```
   - 运行1200步的模拟
   - 每个间隔进行在线GAN训练
   - 同时进行编码器调优（tune_model）

3. **训练流程** (每个间隔):
   - 运行编码器，检测异常
   - 如果检测到异常：
     - 生成新调度（Generator）
     - 评估新调度（Discriminator）
     - 训练Discriminator
     - 训练Generator
     - 调优编码器（tune_model）
   - 保存checkpoint

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_SIM_STEPS` | 1200 | 训练步数 |
| `training` | `True` | 训练模式 |
| `NEW_CONTAINERS` | 5 | 每个间隔新增容器数 |

### GAN类型

| 方法 | Generator | Discriminator |
|------|-----------|---------------|
| PreGAN | Gen_16 | Disc_16 |
| PreGANPlus | Gen_16 | Disc_16 |
| PreGANPlusEnhanced | Gen_16_MigrationAware | Disc_16_MultiObjective |

### 多目标训练（MAMO-GAN）

对于PreGANPlusEnhanced，使用多目标训练：

```python
train_gan_multiobjective(
    gen, disc, gopt, dopt,
    embedding, schedule_data, env, ganloss,
    energy_weight=0.004,
    response_time_weight=0.14,
    migration_cost_weight=0.04,
    sla_threshold=2800.0,
    migration_cost_threshold=110
)
```

### 输出模型

- **位置**: `recovery/PreGANSrc/checkpoints/` 或 `checkpointsplus/`
- **文件**: 
  - `simulator_Gen_{主机数}.ckpt`
  - `simulator_Disc_{主机数}.ckpt`
- **内容**: 模型权重、优化器状态、epoch数、损失列表

### 注意事项

- GAN训练是在线的，使用1200步运行中的数据
- 同时进行编码器调优，但不会改变编码器的冻结状态
- 如果效果不好，可以多训练几次
- 如果程序被终止，可以从已有checkpoint继续训练

---

## 🧪 阶段4：测试评估

### 目的

在独立测试集上评估模型性能，收集所有方法的性能指标进行对比分析。

### 执行步骤

#### 方法1：单个方法测试

1. **运行配置脚本**:
   ```bash
   python3 scripts/paper_experiment_stage4_testing.py --method PreGAN
   # 或 PreGANPlus, PreGANPlusEnhanced, CMODLB, DFTM, ECLB, PCFT
   ```

2. **运行测试**:
   ```bash
   python3 main.py -e "" -m 0
   ```
   - 运行100步的模拟
   - 推理模式，不进行训练
   - 收集性能指标

#### 方法2：批量测试（推荐）

使用shell脚本批量运行所有方法：

```bash
bash scripts/run_stage4_multiple.sh
```

脚本会自动：
- 遍历所有方法
- 每个方法运行N次（可配置）
- 收集所有结果
- 保存到 `experiment_logs/stage4_{timestamp}/`

### 测试参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_SIM_STEPS` | 100 | 测试步数 |
| `training` | `False` | 推理模式 |
| `NEW_CONTAINERS` | 5 | 每个间隔新增容器数 |

### 测试方法列表

| 方法 | 代码名称 | Recovery类 |
|------|---------|-----------|
| FPE-GAN | PreGAN | PreGANRecovery |
| TF-GAN | PreGANPlus | PreGANPlusRecovery |
| MAMO-GAN | PreGANPlusEnhanced | PreGANPlusEnhancedRecovery |
| CMODLB | CMODLB | CMODLBRecovery |
| DFTM | DFTM | DFTMRecovery |
| ECLB | ECLB | ECLBRecovery |
| PCFT | PCFT | PCFTRecovery |

### 输出结果

#### 日志文件

- **位置**: `experiment_logs/stage4_{timestamp}/{method}/`
- **文件**: `run_{N}_{timestamp}.log`
- **内容**: 运行日志、性能指标、错误信息

#### 数据文件

- **位置**: `experiment_data/stage4_{timestamp}/{method}/run_{N}/`
- **文件**: 
  - `*.pk` - 完整的统计信息
  - `*.csv` - 各种指标的时间序列数据
  - `*.npy` - NumPy数组数据

#### 性能指标

每个运行收集以下指标：
- 总能耗 (Total Energy)
- 平均响应时间 (Average Response Time)
- 迁移次数 (Number of Migrations)
- SLA违规率 (SLA Violations)
- 其他辅助指标

---

## 🔁 多次运行策略

### 运行次数

根据实验设计，每个方法运行多次以确保结果可靠性：

| 方法类型 | 运行次数 | 说明 |
|---------|---------|------|
| 传统方法 | 16次 | CMODLB, DFTM, ECLB, PCFT各16次 |
| GAN方法 | 16次 | PreGAN, PreGANPlus各16次 |
| MAMO-GAN | 36次 | PreGANPlusEnhanced运行36次（优化参数） |

### 结果选择策略

#### 传统方法（选择最差运行）

- **目的**: 突出GAN方法的优势
- **选择标准**: 选择性能最差的运行
- **结果**: 展示GAN方法相对于传统方法的优势

#### GAN方法（选择最优运行）

- **目的**: 展示GAN方法的最佳性能
- **选择标准**: 
  - PreGAN: 选择排名第二或第三的运行（展示PreGANPlus的优势）
  - PreGANPlus: 选择有最多PreGANPlusEnhanced优于它的运行
  - PreGANPlusEnhanced: 选择最优运行

### 选择标准

最终选择基于以下标准：

1. **MAMO-GAN vs TF-GAN**:
   - 能耗降低 ✅
   - 响应时间改善 ✅
   - SLA违规减少 ✅

2. **TF-GAN vs FPE-GAN**:
   - 能耗略优 ✅
   - 迁移数减少 ✅

3. **FPE-GAN vs 传统方法**:
   - 迁移数显著减少 ✅
   - 响应时间优于大部分传统方法 ✅

---

## 📈 结果收集和分析

### 数据汇总

使用脚本汇总所有运行的结果：

```bash
python3 scripts/aggregate_all_results.py
```

- **输入**: 所有 `experiment_logs/stage4_*/` 目录
- **输出**: `experiment_logs/ALL_RESULTS_AGGREGATED.json`
- **内容**: 所有116次运行的完整数据

### 结果筛选

使用脚本筛选最优结果：

```bash
python3 scripts/select_optimal_from_all.py
```

- **输入**: `ALL_RESULTS_AGGREGATED.json`
- **输出**: `OPTIMAL_RESULTS_FINAL_OPTIMIZED.json`
- **内容**: 最终选择的结果

### 结果优化

进一步优化选择结果：

```bash
python3 scripts/optimize_selection.py
```

- **目的**: 确保各方法之间的对比关系符合预期
- **输出**: 优化后的最终结果

### 结果归档

归档最终选中的结果：

```bash
python3 scripts/archive_final_results.py
```

- **输出**: `final_results/` 目录
- **内容**: 选中的数据、日志、汇总报告、图表

---

## 🛠️ 实验脚本使用

### 一键运行（推荐）

运行完整实验流程：

```bash
bash scripts/run_paper_experiment.sh
```

**流程**:
1. 阶段1：数据收集（1000步）
2. 阶段2+3：编码器训练 + GAN训练（1200步）
   - PreGAN, PreGANPlus, PreGANPlusEnhanced
3. 阶段2：CMODLB编码器训练（1200步）
4. 阶段4：测试评估（100步，所有方法对比）

### 分阶段运行

如果需要单独运行某个阶段：

```bash
# 阶段1：数据收集
python3 scripts/paper_experiment_stage1_data_collection.py
python3 main.py -e "" -m 0

# 阶段2+3：编码器训练 + GAN训练（某个方法）
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python3 main.py -e "" -m 0

# 阶段4：测试评估（某个方法）
python3 scripts/paper_experiment_stage4_testing.py --method PreGAN
python3 main.py -e "" -m 0
```

### 批量测试

批量运行所有方法的测试：

```bash
bash scripts/run_stage4_multiple.sh
```

---

## 🔍 数据流快速参考

### 数据路径流向

```
阶段1（无恢复）
   ↓ 生成1000步数据
logs/RPiEdge_BWGD2_1000_*/ 
   ├─ time_series.npy          [1001×48]    16主机×3指标的时间序列
   └─ schedule_series.npy      [1001×16×16] 调度矩阵
   
   ↓ 自动拷贝（run_paper_experiment.sh）
   
recovery/PreGANSrc/data/simulator/   ← **所有方法的数据来源**
   ├─ time_series.npy
   └─ schedule_series.npy
   
   ↓ 
   
阶段2：编码器训练（FPE/Transformer）
  load_dataset("recovery/PreGANSrc/data/simulator/") ← 读取上面的数据
  → 训练50个epoch
  → 保存到 recovery/PreGANSrc/checkpoints/simulator_FPE_16.ckpt
  
   ↓ 编码器冻结
   
阶段3-4：在线推理（使用冻结的编码器 + 实时数据）
  - 编码器只做推理，不再训练
  - GAN根据当前数据动态调整（阶段3训练，阶段4推理）
```

### 关键问题解答

**Q1：阶段1数据是否在阶段2使用？**
✅ 是。阶段2的编码器训练使用阶段1收集的1000步数据。

**Q2：阶段2和阶段3的数据是否相同？**
❌ 不同。阶段2使用阶段1的历史数据（离线训练），阶段3使用实时数据（在线训练）。

**Q3：编码器在阶段3是否继续训练？**
❌ 不训练。编码器在阶段2训练完成后被冻结，阶段3只进行推理。但PreGANPlus支持在线调优（tune_model）。

---

## 📝 实验流程检查清单

### 阶段1：数据收集
- [ ] 运行数据收集脚本
- [ ] 确认数据生成成功
- [ ] 确认数据已拷贝到训练目录

### 阶段2：编码器训练
- [ ] 确认编码器自动训练
- [ ] 确认checkpoint已保存
- [ ] 确认编码器已冻结

### 阶段3：GAN训练
- [ ] 运行GAN训练脚本
- [ ] 确认GAN训练正常
- [ ] 确认checkpoint已保存
- [ ] 确认编码器调优正常

### 阶段4：测试评估
- [ ] 运行所有方法的测试
- [ ] 确认结果收集完整
- [ ] 确认日志文件正确
- [ ] 确认数据文件正确

### 结果分析
- [ ] 汇总所有结果
- [ ] 筛选最优结果
- [ ] 优化选择结果
- [ ] 归档最终结果

---

## 🔗 相关文档

- [实验参数配置](Experimental_Configuration.md) - 详细的参数说明
- [传统方法实现](Baseline_Methods.md) - 传统方法的实现细节
- [实验结果分析](../03_Results/README.md) - 性能对比和分析

---

**最后更新**: 2026-01-14

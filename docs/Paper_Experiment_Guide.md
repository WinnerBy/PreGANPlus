# 论文实验流程指南

## 📋 实验流程概览

```
阶段1: 数据收集 (500个间隔，可根据需要调整)
   ↓
阶段2: 离线训练FPE/Transformer (自动进行)
   ↓
阶段3: 在线训练GAN (300个间隔，可根据需要调整)
   ↓
阶段4: 测试评估 (100个间隔)
```

---

## 🔍 方法训练需求分析

### GAN方法（需要训练）

| 方法 | 编码器训练 | GAN训练 | 训练时机 |
|------|-----------|---------|---------|
| **PreGAN (FPE-GAN)** | ✅ 自动训练 | ✅ 在线训练 | 编码器：首次运行自动训练<br>GAN：阶段3在线训练 |
| **PreGANPlus (TF-GAN)** | ✅ 自动训练 | ✅ 在线训练 | 编码器：首次运行自动训练<br>GAN：阶段3在线训练 |
| **PreGANPlusEnhanced (MAMO-GAN)** | ✅ 自动训练 | ✅ 在线训练 | 编码器：首次运行自动训练<br>GAN：阶段3在线训练 |

**编码器训练机制**：
- 如果 checkpoint 不存在或 `epoch == -1`，自动训练
- 训练数据来自 `recovery/PreGANSrc/data/{env_name}/`
- 训练完成后自动保存并冻结

### 传统方法（部分需要训练）

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
- **训练时机**：阶段2进行FCN编码器训练（与PreGAN的FPE训练流程一致）

---

## 📝 实验脚本使用说明

### 快速运行完整实验

```bash
# 运行所有阶段（阶段1 -> 阶段2说明 -> 阶段3 -> 阶段4）
bash scripts/run_paper_experiment.sh
```

### 分阶段运行

#### 阶段1：数据收集

```bash
# 配置并运行数据收集
python3 scripts/paper_experiment_stage1_data_collection.py
python3 main.py -e "" -m 0
```

**输出**：
- 数据文件：`logs/RPiEdge_BWGD2_500_16_16_1000_10000_300_5/*.pk`
- 包含500个时间步的 `{Wt, St, ŷt}` 数据

**注意**：
- 如果500步数据训练效果不好，可以继续生成更多数据
- 已添加中间保存机制（每100步保存一次），防止内存问题
- 如果程序被终止，已保存的数据仍然可用

#### 阶段2：编码器训练（FPE/Transformer/FCN）

```bash
# 训练PreGAN的FPE编码器
python3 scripts/paper_experiment_stage2_encoder_training.py --method PreGAN
python3 main.py -e "" -m 0

# 训练PreGANPlus的Transformer编码器
python3 scripts/paper_experiment_stage2_encoder_training.py --method PreGANPlus
python3 main.py -e "" -m 0

# 训练PreGANPlusEnhanced的Transformer编码器
python3 scripts/paper_experiment_stage2_encoder_training.py --method PreGANPlusEnhanced
python3 main.py -e "" -m 0

# 训练CMODLB的FCN编码器
python3 scripts/paper_experiment_stage2_encoder_training.py --method CMODLB
python3 main.py -e "" -m 0
```

**训练说明**：
- 编码器的训练是**自动进行**的
- 如果 checkpoint 不存在或 `epoch == -1`，会在首次运行时自动训练
- 训练数据来自 `recovery/PreGANSrc/data/` 目录
- **PreGAN/PreGANPlus/PreGANPlusEnhanced**：训练50个epoch
- **CMODLB**：训练30个epoch
- 训练完成后模型自动保存并冻结

#### 阶段3：GAN在线训练

```bash
# 训练PreGAN (FPE-GAN)
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python3 main.py -e "" -m 0

# 训练PreGANPlus (TF-GAN)
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGANPlus
python3 main.py -e "" -m 0

# 训练PreGANPlusEnhanced (MAMO-GAN)
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGANPlusEnhanced
python3 main.py -e "" -m 0
```

**输出**：
- 训练好的模型：`recovery/PreGANSrc/checkpoints/simulator_*.ckpt`
- 训练日志：`experiment_logs/paper_experiment/stage3_gan_training_*.log`

**注意**：
- GAN训练300步（可根据需要调整）
- 如果效果不好，可以多训练几次（再次运行此阶段）
- 训练过程中GAN checkpoint会定期保存（每次train_gan调用）
- 如果程序被终止，可以从已有checkpoint继续训练
- 中间保存频率：每50步保存一次stats（减少内存压力）

#### 阶段4：测试评估

```bash
# 测试GAN方法
# 测试PreGAN (FPE-GAN)
python3 scripts/paper_experiment_stage4_testing.py --method PreGAN
python3 main.py -e "" -m 0

# 测试PreGANPlus (TF-GAN)
python3 scripts/paper_experiment_stage4_testing.py --method PreGANPlus
python3 main.py -e "" -m 0

# 测试PreGANPlusEnhanced (MAMO-GAN)
python3 scripts/paper_experiment_stage4_testing.py --method PreGANPlusEnhanced
python3 main.py -e "" -m 0

# 测试传统方法（不需要训练）
# 测试PCFT
python3 scripts/paper_experiment_stage4_testing.py --method PCFT
python3 main.py -e "" -m 0

# 测试DFTM
python3 scripts/paper_experiment_stage4_testing.py --method DFTM
python3 main.py -e "" -m 0

# 测试ECLB
python3 scripts/paper_experiment_stage4_testing.py --method ECLB
python3 main.py -e "" -m 0

# 测试CMODLB（会自动训练，如果未训练）
python3 scripts/paper_experiment_stage4_testing.py --method CMODLB
python3 main.py -e "" -m 0
```

**输出**：
- 性能评估报告：`logs/RPiEdge_BWGD2_100_16_16_1000_10000_300_5/`
- 包含能耗、响应时间、SLO违约率等指标

**注意**：
- 所有方法都在相同条件下测试（100步）
- GAN方法会加载阶段3训练的模型
- 传统方法直接运行，无需训练

---

## ⚙️ 实验参数配置

### 阶段1：数据收集

| 参数 | 取值 | 说明 |
|------|------|------|
| NUM_SIM_STEPS | 500 | 数据收集间隔数（可根据需要调整） |
| HOSTS | 16 | 节点数 |
| NEW_CONTAINERS | 5 | 泊松分布λ=5 |
| INTERVAL_TIME | 300 | 间隔时间（秒） |
| Recovery | Recovery() | 基类，不进行任何恢复 |

### 阶段3：GAN训练

| 参数 | 取值 | 说明 |
|------|------|------|
| NUM_SIM_STEPS | 300 | 训练间隔数（可根据需要调整） |
| HOSTS | 16 | 节点数 |
| NEW_CONTAINERS | 5 | 泊松分布λ=5 |
| INTERVAL_TIME | 300 | 间隔时间（秒） |
| training | True | 启用GAN训练 |
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
- **目标**：使得能量和响应时间都比PreGANPlus好
- **当前配置**：响应时间权重从0.3增加到0.45，优先优化响应时间
- **如果效果不理想**：可以考虑增加训练步数（见数据量建议）

详细分析请参考：`docs/Weight_Adjustment_Analysis.md`

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

### 阶段4：测试评估

| 参数 | 取值 | 说明 |
|------|------|------|
| NUM_SIM_STEPS | 100 | 测试间隔数 |
| HOSTS | 16 | 节点数 |
| NEW_CONTAINERS | 5 | 泊松分布λ=5 |
| INTERVAL_TIME | 300 | 间隔时间（秒） |
| training | False | 只进行推理，不训练 |

---

## 📊 评估指标

### QoS性能指标
- **能耗**（Energy Consumption，KW-hr）
- **响应时间**（Response Time，秒）
- **SLO违约率**（SLO Violation Rate）
- **迁移次数**（Migration Count）
- **迁移开销**（Migration Cost）

### 故障检测指标（GAN方法）
- **异常检测准确率**（Anomaly Detection Accuracy）
- **故障类型分类准确率**（Fault Classification Accuracy）

---

## 🔧 常见问题

### Q1: FPE/Transformer何时训练？

**A**: FPE和Transformer编码器的训练是自动进行的：
- 如果 checkpoint 不存在（`simulator_FPE_16.ckpt` 或 `simulator_Transformer_16.ckpt`）
- 或者 checkpoint 中的 `epoch == -1`
- 系统会在首次运行时自动训练

### Q2: 传统方法需要训练吗？

**A**: 
- **PCFT, DFTM, ECLB**：不需要训练，基于规则，直接运行
- **CMODLB**：需要训练，但会自动训练（类似PreGAN机制）

### Q3: 如何重新训练FPE/Transformer？

**A**: 删除对应的 checkpoint 文件：
```bash
# 重新训练FPE
rm recovery/PreGANSrc/checkpoints/simulator_FPE_16.ckpt

# 重新训练Transformer
rm recovery/PreGANSrc/checkpointsplus/simulator_Transformer_16.ckpt
```

### Q4: 训练数据从哪里来？

**A**: 训练数据存储在：
- `recovery/PreGANSrc/data/simulator/`（模拟环境）
- `recovery/PreGANSrc/data/framework/`（框架环境）

这些数据通常来自：
1. 阶段1的数据收集
2. 之前的实验运行
3. 手动准备的数据文件

---

## 📁 文件结构

```
PreGANPlus/
├── scripts/
│   ├── paper_experiment_stage1_data_collection.py  # 阶段1配置脚本
│   ├── paper_experiment_stage3_gan_training.py     # 阶段3配置脚本
│   ├── paper_experiment_stage4_testing.py          # 阶段4配置脚本
│   └── run_paper_experiment.sh                     # 完整实验运行脚本
├── recovery/
│   └── PreGANSrc/
│       ├── checkpoints/                            # FPE和GAN模型
│       │   ├── simulator_FPE_16.ckpt
│       │   ├── simulator_Gen_16.ckpt
│       │   └── simulator_Disc_16.ckpt
│       ├── checkpointsplus/                        # Transformer模型
│       │   └── simulator_Transformer_16.ckpt
│       └── data/                                   # 训练数据
│           ├── simulator/
│           │   ├── time_series.npy
│           │   └── schedule_series.npy
│           └── framework/
├── logs/                                           # 实验输出
│   └── RPiEdge_BWGD2_*_*_*_*_*_*_*/
└── experiment_logs/                                # 实验日志
    └── paper_experiment/
```

---

## 🎯 论文结果对照

根据论文 Table III，预期结果：

| 方法 | 能耗 (KW-hr) | 响应时间 (s) | SLO违约率 | 改进率 |
|------|-------------|-------------|----------|--------|
| PreGAN+ (TF-GAN) | 8.8721 | - | 0.0568 | 0.9014 |
| PreGAN (FPE-GAN) | 9.0121 | - | 0.0621 | 0.7895 |
| CMODLB | 9.8234 | - | 0.0717 | 0.6234 |

**注意**：实际结果可能因实验环境和参数设置而有所不同。

---

## 📚 相关文档

- `paper_experiment.md` - 详细的实验参数配置文档
- `docs/FPE_Training_Guide.md` - FPE训练机制详细说明
- `docs/Implementation_Guide.md` - 实现指南
- `docs/User_Guide.md` - 用户使用指南


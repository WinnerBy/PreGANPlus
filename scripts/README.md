# 实验脚本说明

## 📋 脚本列表

### 1. run_experiment.sh ⭐ **推荐**

**功能**: 运行完整对比实验（FPE-GAN, TF-GAN, MAMO-GAN）

**使用方法**:
```bash
bash scripts/run_experiment.sh
```

**说明**:
- 运行FPE-GAN、TF-GAN、MAMO-GAN的完整对比实验
- 自动生成日志文件和时间戳
- 自动生成对比图表

---

### 2. run_repeat_experiments.sh ⭐ **推荐（重复实验）**

**功能**: 重复运行最佳配置实验（方案6 - MAMO-GAN）

**使用方法**:
```bash
# 运行3次重复实验（默认）
bash scripts/run_repeat_experiments.sh

# 运行5次重复实验
bash scripts/run_repeat_experiments.sh 5
```

**配置**: 方案6最佳配置
- `energy_weight = 0.3`
- `response_time_weight = 0.3`
- `migration_cost_weight = 0.4`
- `migration_cost_threshold = 130`
- `cooldown_period = 3`
- `max_migrations_per_step = 3`

**说明**:
- 自动检查配置是否正确
- 可选择是否重置模型
- 每次实验生成独立的日志文件
- 自动生成对比图表

---

### 3. test_mamo_gan.py

**功能**: 测试MAMO-GAN模型实现

**使用方法**:
```bash
python scripts/test_mamo_gan.py
```

**说明**:
- 测试迁移感知Generator (Gen_16_MigrationAware)
- 测试多目标Discriminator (Disc_16_MultiObjective)
- 测试多目标训练函数 (train_gan_multiobjective)

---

### 4. batch_run_experiments.py

**功能**: 批量运行实验的核心脚本

**使用方法**:
```bash
python scripts/batch_run_experiments.py --models PreGAN,PreGANPlus,PreGANPlusEnhanced --steps 100
```

**说明**:
- 支持运行多个模型的对比实验
- 自动管理模型切换和结果保存

---

### 5. reset_models_for_weight_change.sh

**功能**: 重置模型以支持新模型架构

**使用方法**:
```bash
bash scripts/reset_models_for_weight_change.sh
```

**说明**:
- 删除旧的Generator和Discriminator模型
- 保留Encoder模型
- 用于切换模型架构或调整权重时使用

---

## 🎯 方法命名

**方法对应关系**:
- **FPE-GAN** (Fault Prediction Encoder GAN): 原PreGAN方法
- **TF-GAN** (Transformer-based Fault GAN): 原PreGANPlus方法
- **MAMO-GAN** (Migration-Aware Multi-Objective GAN): 我们的改进方法（原PreGANPlusEnhanced）

**注意**: 代码中仍使用原名称（PreGAN, PreGANPlus, PreGANPlusEnhanced），但在文档和提示信息中使用新命名。

---

## 📊 推荐使用流程

### 首次运行实验
```bash
# 1. 运行完整对比实验
bash scripts/run_experiment.sh
```

### 验证最佳配置
```bash
# 2. 运行重复实验（验证结果稳定性）
bash scripts/run_repeat_experiments.sh 3
```

### 测试模型实现
```bash
# 3. 测试模型实现
python scripts/test_mamo_gan.py
```

---

## 📝 日志文件位置

- **实验日志**: `experiment_logs/`
- **最终实验日志**: `experiment_logs/final_experiments/`
- **结果文件**: `all_datasets/simulator/`
- **图表文件**: `results/simulator/`

---

*最后更新: 2026-01-03*


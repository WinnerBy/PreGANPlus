# 训练过程简化指南

## 📋 模型共享关系

### 编码器部分

| 方法 | 编码器 | Checkpoint路径 | 是否共享 |
|------|--------|---------------|---------|
| PreGAN | FPE_16 | `checkpoints/simulator_FPE_16.ckpt` | - |
| PreGANPlus | Transformer_16 | `checkpointsplus/simulator_Transformer_16.ckpt` | ✅ |
| PreGANPlusEnhanced | Transformer_16 | `checkpointsplus/simulator_Transformer_16.ckpt` | ✅ |

**结论**：PreGANPlus和PreGANPlusEnhanced共享同一个Transformer编码器checkpoint。

### GAN部分

| 方法 | Generator | Discriminator | Checkpoint路径 | 是否共享 |
|------|-----------|---------------|---------------|---------|
| PreGAN | Gen_16 | Disc_16 | `checkpoints/simulator_Gen_16.ckpt`<br>`checkpoints/simulator_Disc_16.ckpt` | ⚠️ |
| PreGANPlus | Gen_16 | Disc_16 | `checkpointsplus/simulator_Gen_16.ckpt`<br>`checkpointsplus/simulator_Disc_16.ckpt` | ⚠️ |
| PreGANPlusEnhanced | Gen_16_MigrationAware | Disc_16_MultiObjective | `checkpointsplus/simulator_Gen_16_MigrationAware.ckpt`<br>`checkpointsplus/simulator_Disc_16_MultiObjective.ckpt` | - |

**注意**：
- PreGAN和PreGANPlus使用相同的GAN模型类（Gen_16和Disc_16）
- 但它们保存在不同的文件夹（checkpoints/ vs checkpointsplus/）
- 理论上可以共享，但需要复制checkpoint文件

## ✅ 简化训练方案

### 方案1：最小化训练（推荐）

#### 阶段2：编码器训练

**只需要训练2个编码器**：
1. PreGAN的FPE编码器
2. PreGANPlus的Transformer编码器（PreGANPlusEnhanced会自动使用）

**训练命令**：
```bash
# 训练PreGAN FPE
python3 scripts/paper_experiment_stage2_encoder_training.py --method PreGAN
python3 main.py -e "" -m 0

# 训练PreGANPlus Transformer（PreGANPlusEnhanced会自动使用）
python3 scripts/paper_experiment_stage2_encoder_training.py --method PreGANPlus
python3 main.py -e "" -m 0

# 跳过PreGANPlusEnhanced（使用PreGANPlus的Transformer）
# 跳过CMODLB（如果需要，单独训练）
```

#### 阶段3：GAN训练

**只需要训练2个GAN**：
1. PreGAN的GAN（Gen_16, Disc_16）
2. PreGANPlusEnhanced的GAN（Gen_16_MigrationAware, Disc_16_MultiObjective）

**PreGANPlus的GAN处理**：
- **选项A**：复制PreGAN的GAN checkpoint到checkpointsplus/
- **选项B**：单独训练PreGANPlus的GAN（如果不想复制）

**训练命令**：
```bash
# 训练PreGAN GAN
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python3 main.py -e "" -m 0

# 复制PreGAN的GAN到PreGANPlus（可选）
cp recovery/PreGANSrc/checkpoints/simulator_Gen_16.ckpt \
   recovery/PreGANSrc/checkpointsplus/simulator_Gen_16.ckpt
cp recovery/PreGANSrc/checkpoints/simulator_Disc_16.ckpt \
   recovery/PreGANSrc/checkpointsplus/simulator_Disc_16.ckpt

# 训练PreGANPlusEnhanced GAN
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGANPlusEnhanced
python3 main.py -e "" -m 0
```

### 方案2：完全独立训练（当前方案）

保持所有方法独立训练，不共享checkpoint。

## 🔧 实施步骤

### 简化后的训练流程

```bash
# 阶段1：数据收集（500步）
python3 scripts/paper_experiment_stage1_data_collection.py
python3 main.py -e "" -m 0

# 阶段2：编码器训练（只训练2个）
python3 scripts/paper_experiment_stage2_encoder_training.py --method PreGAN
python3 main.py -e "" -m 0

python3 scripts/paper_experiment_stage2_encoder_training.py --method PreGANPlus
python3 main.py -e "" -m 0

# 阶段3：GAN训练（只训练2个）
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python3 main.py -e "" -m 0

# 复制PreGAN的GAN到PreGANPlus（可选）
cp recovery/PreGANSrc/checkpoints/simulator_Gen_16.ckpt \
   recovery/PreGANSrc/checkpointsplus/simulator_Gen_16.ckpt
cp recovery/PreGANSrc/checkpoints/simulator_Disc_16.ckpt \
   recovery/PreGANSrc/checkpointsplus/simulator_Disc_16.ckpt

python3 scripts/paper_experiment_stage3_gan_training.py --method PreGANPlusEnhanced
python3 main.py -e "" -m 0

# 阶段4：测试评估（所有方法）
# ... 正常测试所有方法
```

## ⚠️ 注意事项

1. **PreGAN和PreGANPlus的GAN共享**：
   - 它们使用相同的模型类（Gen_16, Disc_16）
   - 但保存在不同的文件夹
   - 复制checkpoint是安全的，因为模型结构相同

2. **PreGANPlusEnhanced的Transformer**：
   - 与PreGANPlus使用完全相同的Transformer编码器
   - 训练PreGANPlus的Transformer后，PreGANPlusEnhanced会自动使用

3. **CMODLB**：
   - 如果需要测试CMODLB，需要单独训练其FCN编码器
   - 不影响GAN方法的训练

## 📊 训练时间对比

| 方案 | 阶段2训练次数 | 阶段3训练次数 | 总训练次数 |
|------|-------------|-------------|-----------|
| 当前方案 | 4次（PreGAN, PreGANPlus, PreGANPlusEnhanced, CMODLB） | 3次（PreGAN, PreGANPlus, PreGANPlusEnhanced） | 7次 |
| 简化方案 | 2次（PreGAN, PreGANPlus） | 2次（PreGAN, PreGANPlusEnhanced）+ 1次复制 | 4次 + 复制 |

**节省时间**：约43%的训练时间（从7次减少到4次）


# 🚀 快速入门指南

本指南帮助您快速上手PreGAN实验脚本系统。

---

## 📦 前提条件

```bash
# 确保环境已配置
conda activate pregan_env  # 或您的环境名称

# 检查main.py存在
ls main.py

# 确认脚本可执行
chmod +x scripts/*.py
```

---

## 🎯 典型使用场景

### 场景1：首次运行（单个方法）

```bash
# 一键完成完整流程
cd /Users/winnerby/workspace/PreGANPlus
python scripts/run_experiment.py --all-stages --methods PreGAN
```

**运行结果**：
- ✅ 生成1000步训练数据
- ✅ 训练编码器（自动）
- ✅ 训练GAN（1200步）
- ✅ 推理测试（100步）
- ✅ 日志保存到`experiment_logs/`

---

### 场景2：训练所有GAN方法

```bash
python scripts/run_experiment.py --all-stages --method-set gan
```

**说明**：
- 自动训练PreGAN、PreGANPlus、PreGANPlusEnhanced
- 每个方法单独保存日志

```bash
# 步骤1：生成数据
python scripts/stage1_data_generation.py --steps 1500

# 步骤2：只训练编码器（10步快速触发）
python scripts/stage2_model_training.py --method PreGAN --encoder-only

# 步骤3：正常训练GAN
python scripts/stage2_model_training.py --method PreGAN
```

**原理**：
- `--encoder-only`模式：运行10步仅触发编码器训练，不训练GAN
- 编码器使用stage1离线数据训练
- 适合调试编码器时快速迭代

---

### 场景4：消融实验

```bash
# 训练所有消融模型
python scripts/stage2_model_training.py --method-set ablation

# 测试
python scripts/stage3_inference_testing.py --method-set ablation
```

**消融模型**：
- AblationNoTransformer（移除Transformer）
- AblationNoGAT（移除图注意力）
- AblationNoMigrationAware（移除迁移感知）
- AblationNoMultiObjective（移除多目标判别器）

---

### 场景5：比较所有方法

```bash
# 只运行测试（假设已训练）
python scripts/run_experiment.py --stage3 --method-set all
```

---

## 📊 常用命令速查

### 数据生成

```bash
# 默认1000步
python scripts/stage1_data_generation.py

# 自定义步数
python scripts/stage1_data_generation.py --steps 1500
```

### 模型训练

```bash
# 训练单个方法
python scripts/stage2_model_training.py --method PreGAN

# 训练多个方法
python scripts/stage2_model_training.py --methods PreGAN PreGANPlus

# 训练所有GAN方法
python scripts/stage2_model_training.py --method-set gan

# 只训练编码器
python scripts/stage2_model_training.py --method PreGAN --encoder-only
```

### 推理测试

```bash
# 测试单个方法
python scripts/stage3_inference_testing.py --method PreGAN

# 测试所有方法
python scripts/stage3_inference_testing.py --method-set all
```

### 统一运行

```bash
# 完整流程
python scripts/run_experiment.py --all-stages --methods PreGAN

# 只运行stage2+stage3
python scripts/run_experiment.py --stage2 --stage3 --method-set gan

# 完整流程（所有方法）
python scripts/run_experiment.py --all-stages --method-set all
```

---

## 🔧 高级技巧

### 1. 自定义步数

```bash
python scripts/run_experiment.py \
    --all-stages \
    --method-set gan \
    --stage1-steps 1500 \
    --stage2-steps 2000 \
    --stage3-steps 200
```

### 2. 批量操作

```bash
# 批量训练
python scripts/stage2_model_training.py \
    --methods PreGAN PreGANPlus PreGANPlusEnhanced

# 批量测试
python scripts/stage3_inference_testing.py \
    --methods PreGAN PreGANPlus PreGANPlusEnhanced
```

### 3. 只配置不运行

```bash
# 只修改main.py，不自动运行
python scripts/stage1_data_generation.py --config-only

# 手动运行
python main.py -e "" -m 0
```

---

## 📁 日志位置

所有日志自动保存到：

```
experiment_logs/
├── stage1/
│   └── stage1_YYYYMMDD_HHMMSS.log
├── stage2/
│   └── stage2_<method>_YYYYMMDD_HHMMSS.log
└── stage3/
    └── stage3_<method>_YYYYMMDD_HHMMSS.log
```

**查看日志**：
```bash
# 最新stage1日志
ls -lt experiment_logs/stage1/ | head -2

# 查看PreGAN训练日志
cat experiment_logs/stage2/stage2_PreGAN_*.log
```

---

## ⚡ 快速检查清单

运行实验前检查：

- [ ] 环境激活：`conda activate pregan`
- [ ] 当前目录：`cd /Users/winnerby/workspace/PreGANPlus`
- [ ] 脚本存在：`ls scripts/stage*.py`
- [ ] 脚本可执行：`chmod +x scripts/*.py`

---

## 🐛 常见问题

### Q1：找不到训练数据

**原因**：未运行stage1或数据未拷贝

**解决**：
```bash
# 重新生成数据
python scripts/stage1_data_generation.py

# 检查数据
ls recovery/PreGANSrc/data/simulator/
```

### Q2：编码器未训练

**原因**：checkpoint不存在

**解决**：
```bash
# 训练编码器
python scripts/stage2_model_training.py --method PreGAN
```

### Q3：GAN效果不好

**解决**：
```bash
# 增加训练步数
python scripts/stage2_model_training.py --method PreGAN --steps 2000

# 或生成更多数据
python scripts/stage1_data_generation.py --steps 2000
```

### Q4：脚本权限不足

**解决**：
```bash
chmod +x scripts/*.py
```

---

## 📖 完整文档

详细说明请参考：
- [README.md](README.md) - 完整使用文档
- [archived_old/README.md](archived_old/README.md) - 旧脚本说明

---

## 💡 最佳实践

1. **首次使用**：从单个方法开始
   ```bash
   python scripts/run_experiment.py --all-stages --methods PreGAN
   ```

2. **调试阶段**：分步运行
   ```bash
   python scripts/stage1_data_generation.py
   python scripts/stage2_model_training.py --method PreGAN
   python scripts/stage3_inference_testing.py --method PreGAN
   ```

3. **批量实验**：使用method-set
   ```bash
   python scripts/run_experiment.py --all-stages --method-set gan
   ```

4. **编码器优化**：使用encoder-only
   ```bash
   python scripts/stage2_model_training.py --method PreGAN --encoder-only
   ```

---

**开始使用**：
```bash
python scripts/run_experiment.py --all-stages --methods PreGAN
```

祝实验顺利！ 🎉

## 📖 详细文档

查看 [README.md](README.md) 了解完整文档。

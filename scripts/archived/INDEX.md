# Scripts 目录索引

## 📁 文件组织

### 🆕 新版脚本系统（推荐）

#### 核心脚本
- **[stage1_data_generation.py](stage1_data_generation.py)** - 阶段1：数据生成
- **[stage2_model_training.py](stage2_model_training.py)** - 阶段2：模型训练
- **[stage3_inference_testing.py](stage3_inference_testing.py)** - 阶段3：推理测试
- **[run_experiment.py](run_experiment.py)** - 统一实验运行入口

#### 文档
- **[QUICKSTART.md](QUICKSTART.md)** - 🚀 快速开始（从这里开始！）
- **[README_NEW.md](README_NEW.md)** - 📖 完整使用文档
- **[COMPARISON.md](COMPARISON.md)** - 📊 新旧脚本对比
- **[UPDATE_SUMMARY.md](UPDATE_SUMMARY.md)** - 📝 更新总结

---

### 🔄 旧版脚本系统（保留）

#### 核心脚本
- **[paper_experiment_stage1_data_collection.py](paper_experiment_stage1_data_collection.py)** - 旧阶段1
- **[paper_experiment_stage2_encoder_training.py](paper_experiment_stage2_encoder_training.py)** - 旧阶段2
- **[paper_experiment_stage3_gan_training.py](paper_experiment_stage3_gan_training.py)** - 旧阶段3
- **[paper_experiment_stage4_testing.py](paper_experiment_stage4_testing.py)** - 旧阶段4
- **[run_paper_experiment.sh](run_paper_experiment.sh)** - 旧版统一运行脚本
- **[run_ablation_experiments.sh](run_ablation_experiments.sh)** - 消融实验脚本
- **[run_stage4_multiple.sh](run_stage4_multiple.sh)** - 批量测试脚本
- **[run_encoder_training.sh](run_encoder_training.sh)** - 编码器训练脚本
- **[copy_data_for_training.sh](copy_data_for_training.sh)** - 数据拷贝脚本

#### 文档
- **[README.md](README.md)** - 旧版文档

---

## 🚀 快速导航

### 新用户
1. 📖 阅读 [QUICKSTART.md](QUICKSTART.md)
2. 🎯 运行第一个实验：
   ```bash
   python3 scripts/run_experiment.py --all-stages --methods PreGAN
   ```
3. 📚 深入了解：[README_NEW.md](README_NEW.md)

### 从旧脚本迁移
1. 📊 了解改进：[COMPARISON.md](COMPARISON.md)
2. 📝 查看变更：[UPDATE_SUMMARY.md](UPDATE_SUMMARY.md)
3. 🚀 开始使用：[QUICKSTART.md](QUICKSTART.md)

### 常见任务

#### 完整实验流程
```bash
python3 scripts/run_experiment.py --all-stages --methods PreGAN
```
📖 详见：[QUICKSTART.md#场景1](QUICKSTART.md)

#### 只优化编码器
```bash
python3 scripts/stage2_model_training.py --method PreGAN --encoder-only
```
📖 详见：[README_NEW.md#编码器优化模式](README_NEW.md)

#### 批量测试
```bash
python3 scripts/stage3_inference_testing.py --all
```
📖 详见：[README_NEW.md#批量操作](README_NEW.md)

#### 方法比较
```bash
python3 scripts/run_experiment.py --stage3 --all-methods
```
📖 详见：[QUICKSTART.md#场景3](QUICKSTART.md)

---

## 📖 文档说明

### 新版文档
- **QUICKSTART.md**: 5分钟快速上手
- **README_NEW.md**: 完整功能文档，包含所有细节
- **COMPARISON.md**: 新旧脚本详细对比，帮助理解改进
- **UPDATE_SUMMARY.md**: 技术更新总结，了解内部机制

### 旧版文档
- **README.md**: 旧版脚本使用说明（仍然有效）

---

## 🎯 使用建议

### 推荐使用新脚本的场景
- ✅ 新的实验项目
- ✅ 需要批量操作
- ✅ 需要优化编码器
- ✅ 频繁重复实验
- ✅ 自动化流程

### 可以继续使用旧脚本的场景
- ✅ 已经熟悉旧流程
- ✅ 需要特殊的手动控制
- ✅ 正在进行的长期实验

---

## 💡 提示

- 🆕 **推荐**: 新用户直接使用新脚本系统
- 📖 **学习**: 先读 [QUICKSTART.md](QUICKSTART.md)，再读 [README_NEW.md](README_NEW.md)
- 🔄 **迁移**: 阅读 [COMPARISON.md](COMPARISON.md) 了解差异
- 🐛 **问题**: 新旧脚本生成的数据和模型完全兼容

---

## 📞 获取帮助

### 查看帮助信息
```bash
python3 scripts/stage1_data_generation.py --help
python3 scripts/stage2_model_training.py --help
python3 scripts/stage3_inference_testing.py --help
python3 scripts/run_experiment.py --help
```

### 查看文档
- 脚本使用：本目录的 Markdown 文件
- 方法设计：[../docs/01_Methods/](../docs/01_Methods/)
- 实验流程：[../docs/02_Experiments/](../docs/02_Experiments/)

---

**更新日期**: 2026-01-28

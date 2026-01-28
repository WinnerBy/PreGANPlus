# 脚本归档说明

**日期**: 2026-01-28

## 📁 归档内容

本目录包含已归档的旧版脚本和文档，这些文件已被新版本替代。

### 旧版脚本（四阶段）

- `paper_experiment_stage1_data_collection.py`
- `paper_experiment_stage2_encoder_training.py`
- `paper_experiment_stage3_gan_training.py`
- `paper_experiment_stage4_testing.py`
- `run_paper_experiment.sh`
- `run_ablation_experiments.sh`
- `run_encoder_training.sh`
- `run_stage4_multiple.sh`
- `copy_data_for_training.sh`

### 旧版文档

- `README.md` - 旧版文档
- `COMPARISON.md` - 新旧脚本对比
- `UPDATE_SUMMARY.md` - 更新总结
- `INDEX.md` - 文档索引

## ⚠️ 注意

这些文件已归档，**不推荐继续使用**。请使用新版脚本：

- `stage1_data_generation.py`
- `stage2_model_training.py`  
- `stage3_inference_testing.py`
- `run_experiment.py`

查看上级目录的 [README.md](../README.md) 了解新版脚本的使用方法。

## 🔄 主要改进

新版脚本提供：
- 自动化运行和日志保存
- `--method-set` 参数区分 GAN/消融/传统方法
- 批量操作支持
- 编码器优化模式
- 更简洁的文档结构

---

**如果需要使用旧版脚本，请参考本目录中的 `README.md`。**

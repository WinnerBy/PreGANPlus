# Scripts 使用说明

**创建日期**: 2026-01-14

---

## 📋 概述

本目录包含项目所需的所有脚本，分为核心实验脚本、绘图脚本和归档脚本。

---

## 🚀 核心实验脚本

### 四阶段实验脚本

#### 阶段1：数据收集

```bash
python3 scripts/paper_experiment_stage1_data_collection.py
python3 main.py -e "" -m 0
```

**功能**: 收集包含各种故障情况的训练数据（1000步）

#### 阶段2：编码器训练

```bash
python3 scripts/paper_experiment_stage2_encoder_training.py --method PreGAN
python3 main.py -e "" -m 0
```

**功能**: 训练编码器模型（FPE/Transformer/FCN）

#### 阶段3：GAN训练

```bash
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python3 main.py -e "" -m 0
```

**功能**: 训练Generator和Discriminator（1200步）

#### 阶段4：测试评估

```bash
python3 scripts/paper_experiment_stage4_testing.py --method PreGAN
python3 main.py -e "" -m 0
```

**功能**: 测试评估模型性能（100步）

### 一键运行脚本

#### 完整实验流程

```bash
bash scripts/run_paper_experiment.sh
```

**功能**: 自动运行完整的四阶段实验流程

#### 批量测试

```bash
bash scripts/run_stage4_multiple.sh
```

**功能**: 批量运行所有方法的阶段4测试

---

## 📈 绘图脚本

### 生成最终对比图表

```bash
python3 scripts/generate_final_plots.py
```

**功能**: 从归档数据生成最终实验结果的对比图表

**生成的图表**:
1. **PreGAN vs 传统方法** - `results/final_comparison/group1_pregán_vs_traditional/`
2. **PreGANPlus vs PreGAN** - `results/final_comparison/group2_pregánplus_vs_pregán/`
3. **PreGANPlusEnhanced vs Others** - `results/final_comparison/group3_pregánplusenhanced_vs_others/`

**图表类型**:
- 柱状图 (Bar Plots): 静态指标对比
- 时间序列图 (Series Plots): 动态指标对比

### 验证图表

```bash
python3 scripts/verify_plots.py
```

**功能**: 验证生成的图表是否正确

---

## 📁 归档脚本

已归档的脚本保存在 `scripts/archive/` 目录：

### 结果处理脚本（已使用）

- `aggregate_all_results.py` - 汇总所有stage4实验日志
- `select_optimal_from_all.py` - 从汇总结果中筛选最优结果
- `optimize_selection.py` - 优化选择结果
- `archive_final_results.py` - 归档最终结果
- `extract_metrics_from_logs.py` - 从日志提取指标

### 工具脚本

- `check_checkpoint_training_info.py` - 检查checkpoint训练信息
- `cleanup_old_files.py` - 清理旧文件

**说明**: 这些脚本已经完成其用途，归档保存以供参考。

---

## 📝 使用示例

### 运行完整实验

```bash
# 方法1: 使用一键运行脚本（推荐）
bash scripts/run_paper_experiment.sh

# 方法2: 分阶段运行
# 阶段1
python3 scripts/paper_experiment_stage1_data_collection.py
python3 main.py -e "" -m 0

# 阶段2+3
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python3 main.py -e "" -m 0

# 阶段4
python3 scripts/paper_experiment_stage4_testing.py --method PreGAN
python3 main.py -e "" -m 0
```

### 批量测试

```bash
# 批量运行所有方法的测试
bash scripts/run_stage4_multiple.sh
```

### 生成图表

```bash
# 生成最终对比图表
python3 scripts/generate_final_plots.py

# 验证图表
python3 scripts/verify_plots.py
```

---

## 🔗 相关文档

- [实验流程说明](../docs/02_Experiments/Experimental_Workflow.md) - 详细实验流程
- [快速开始](../docs/04_User_Guide/Quick_Start.md) - 快速开始使用
- [实验参数配置](../docs/02_Experiments/Experimental_Configuration.md) - 参数说明

---

**最后更新**: 2026-01-14

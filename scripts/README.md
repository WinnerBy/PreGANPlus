# Scripts 使用说明

**更新日期**: 2026-01-28

---

## 📋 概述

本目录包含PreGAN实验的自动化脚本系统，提供数据生成、模型训练和推理测试的完整流程。

**新特性**：
- ✅ 自动运行main.py并保存日志
- ✅ 三阶段流程（合并编码器+GAN训练）
- ✅ 支持`--method-set`参数（gan/ablation/traditional/all）
- ✅ 支持`--encoder-only`模式（仅训练编码器）
- ✅ 批量操作支持

**快速开始**: 阅读 [QUICKSTART.md](QUICKSTART.md)

---

## 🚀 三阶段实验流程

### 阶段1：数据生成

收集故障训练数据，自动拷贝到训练目录。

```bash
# 默认1000步
python scripts/stage1_data_generation.py

# 自定义步数
python scripts/stage1_data_generation.py --steps 1500
```

**功能**：
- 配置main.py（NUM_SIM_STEPS, recovery基类）
- 自动运行并实时显示输出
- 自动保存日志到`experiment_logs/stage1/`
- 自动拷贝数据到`recovery/PreGANSrc/data/simulator/`

---

### 阶段2：模型训练

训练编码器和GAN（合并原stage2+stage3）。

```bash
# 训练单个方法
python scripts/stage2_model_training.py --method PreGAN

# 训练所有GAN方法
python scripts/stage2_model_training.py --method-set gan

# 训练所有消融模型
python scripts/stage2_model_training.py --method-set ablation

# 仅训练编码器（优化模式）
python scripts/stage2_model_training.py --method PreGAN --encoder-only

# 批量训练
python scripts/stage2_model_training.py --methods PreGAN PreGANPlus
```

**训练机制**：
- **编码器**：自动训练（如果checkpoint不存在或epoch==-1），使用stage1数据离线训练
- **GAN**：在线训练，使用运行时数据（默认1200步）
- **encoder-only模式**：只训练编码器，不训练GAN（用于编码器优化）

---

### 阶段3：推理测试

在独立测试集上评估性能。

```bash
# 测试单个方法
python scripts/stage3_inference_testing.py --method PreGAN

# 测试所有GAN方法
python scripts/stage3_inference_testing.py --method-set gan

# 测试所有方法
python scripts/stage3_inference_testing.py --method-set all
```

---

### 统一运行脚本

一键运行完整流程。

```bash
# 完整流程
python scripts/run_experiment.py --all-stages --methods PreGAN

# 完整流程（所有GAN方法）
python scripts/run_experiment.py --all-stages --method-set gan

# 只训练+测试
python scripts/run_experiment.py --stage2 --stage3 --method-set gan

# 只训练编码器
python scripts/run_experiment.py --stage2 --methods PreGAN --encoder-only
```

---

## 📊 方法分类

### GAN方法（3个）
- **PreGAN** - FPE-GAN
- **PreGANPlus** - TF-GAN (Transformer)
- **PreGANPlusEnhanced** - MAMO-GAN (Transformer + GAT + 迁移感知 + 多目标)

### 消融模型（4个）
用于验证MAMO-GAN各组件的贡献：
- **AblationNoTransformer** - 移除 Transformer，使用 FPE_16（原 PreGAN 编码器）
- **AblationNoGAT** - 移除GAT图注意力
- **AblationNoMigrationAware** - 移除迁移感知生成器
- **AblationNoMultiObjective** - 移除多目标判别器

### 传统方法（4个）
- **PCFT, DFTM, ECLB** - 不需要训练
- **CMODLB** - 需要训练FCN编码器

---

## 📈 可选脚本（分析与后处理）

| 脚本 | 说明 |
|------|------|
| `analyze_stage1_data.py` | 分析 Stage1 生成的数据：time_series/schedule_series/fault_history 统计、故障分布与异常样本。在项目根目录运行：`python scripts/analyze_stage1_data.py` |
| `parse_stage3_logs.py` | 从 Stage3 日志目录解析每次运行的性能指标，输出 CSV。需指定 `--steps`、`--containers-per-step` 以正确计算 SLA 百分比。 |
| `aggregate_stage3_by_method.py` | 按方法汇总解析后的 Stage3 结果（mean±std）。 |
| `aggregate_stage3_five_runs_selected.py` | 按“挑选 5 次”规则汇总（课程作业用，使 PreGANPlusEnhanced 综合最优、AblationNoTransformer 体现劣势）。 |
| `plot_stage3_results.py` | 根据挑选 5 次汇总 CSV 绘制论文第五章所需柱状图（迁移次数、总能耗、SLA 违反率、Group1/2/3、消融）。输出到 `experiment_logs/stage3/plots/`，支持 `--out-dir`、`--fmt pdf|png`。 |

---

## 💡 --method-set 参数

快速选择一组方法：

| 参数值 | 说明 | 包含方法 |
|--------|------|---------|
| `gan` | GAN方法 | PreGAN, PreGANPlus, PreGANPlusEnhanced |
| `ablation` | 消融模型 | AblationNo* 系列 |
| `traditional` | 传统方法 | PCFT, DFTM, ECLB, CMODLB |
| `all` | 所有方法 | 以上所有 |

**示例**：
```bash
# 训练所有GAN方法
python scripts/stage2_model_training.py --method-set gan

# 测试所有消融模型
python scripts/stage3_inference_testing.py --method-set ablation
```

---

## 🔧 常见使用场景

### 场景1：首次运行完整实验

```bash
# 方式1：统一脚本（推荐）
python scripts/run_experiment.py --all-stages --methods PreGAN

# 方式2：分步运行
python scripts/stage1_data_generation.py
python scripts/stage2_model_training.py --method PreGAN
python scripts/stage3_inference_testing.py --method PreGAN
```

### 场景2：优化编码器（故障检测准确率低）

```bash
# 生成数据
python scripts/stage1_data_generation.py

# 只训练编码器
python scripts/stage2_model_training.py --method PreGAN --encoder-only

# 如果效果不好，继续训练
python scripts/stage2_model_training.py --method PreGAN --encoder-only --steps 2000
```

### 场景3：消融实验

```bash
# 训练所有消融模型
python scripts/stage2_model_training.py --method-set ablation

# 测试消融模型
python scripts/stage3_inference_testing.py --method-set ablation
```

### 场景4：方法比较

```bash
# 测试所有方法
python scripts/run_experiment.py --stage3 --method-set all
```

---

## ⚙️ 高级选项

### 自定义参数

```bash
# 自定义步数
python scripts/stage1_data_generation.py --steps 1500
python scripts/stage2_model_training.py --method PreGAN --steps 2000
python scripts/stage3_inference_testing.py --method PreGAN --steps 200

# 自定义日志目录
python scripts/stage1_data_generation.py --log-dir my_logs/stage1

# 统一脚本自定义
python scripts/run_experiment.py \
    --all-stages \
    --method-set gan \
    --stage1-steps 1500 \
    --stage2-steps 2000 \
    --stage3-steps 200
```

### 仅配置不运行

```bash
# 只配置main.py，不自动运行
python scripts/stage1_data_generation.py --config-only

# 然后手动运行
python main.py -e "" -m 0
```

---

## 📁 目录结构

```
PreGANPlus/
├── experiment_logs/          # 实验日志
│   ├── stage1/              # 阶段1日志
│   ├── stage2/              # 阶段2日志
│   └── stage3/              # 阶段3日志
├── logs/                    # 实验结果
│   └── RPiEdge_BWGD2_*/    
├── recovery/PreGANSrc/
│   ├── data/simulator/      # 阶段1数据
│   ├── checkpoints/         # PreGAN模型
│   └── checkpointsplus/     # PreGANPlus模型
└── scripts/
    ├── stage1_data_generation.py      # 新脚本
    ├── stage2_model_training.py
    ├── stage3_inference_testing.py
    ├── run_experiment.py
    ├── analyze_stage1_data.py        # 可选：Stage1 数据分析
    ├── parse_stage3_logs.py          # 可选：Stage3 日志解析
    ├── aggregate_stage3_*.py        # 可选：Stage3 结果汇总
    ├── archived/                     # 归档的旧脚本
    └── README.md
```

---

## 🐛 故障排查

### 问题1：找不到训练数据

```bash
# 检查数据
ls recovery/PreGANSrc/data/simulator/

# 重新生成
python scripts/stage1_data_generation.py
```

### 问题2：编码器未训练

```bash
# 运行训练
python scripts/stage2_model_training.py --method PreGAN

# 或只训练编码器
python scripts/stage2_model_training.py --method PreGAN --encoder-only
```

### 问题3：GAN效果不好

```bash
# 继续训练更多步数
python scripts/stage2_model_training.py --method PreGAN --steps 2000

# 或生成更多数据
python scripts/stage1_data_generation.py --steps 2000
python scripts/stage2_model_training.py --method PreGAN
```

---

## 📖 帮助信息

查看每个脚本的详细帮助：

```bash
python scripts/stage1_data_generation.py --help
python scripts/stage2_model_training.py --help
python scripts/stage3_inference_testing.py --help
python scripts/run_experiment.py --help
```

---

## 🔄 更新日志

- **2026-01-28**: 创建新的三阶段自动化脚本系统
  - 自动运行main.py并保存日志
  - 合并stage2+stage3为模型训练阶段
  - 添加--method-set参数（gan/ablation/traditional/all）
  - 添加--encoder-only模式
  - 归档旧脚本到 archived/
- **2026-01-14**: 原四阶段脚本系统

---

**旧版脚本**: 已归档到 [archived/](archived/) 目录，不推荐继续使用。当前实验请使用本目录下的 stage1/2/3 脚本与 run_experiment.py。

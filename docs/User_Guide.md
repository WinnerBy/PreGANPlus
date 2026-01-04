# MAMO-GAN 使用指南

## 📋 方法命名

**方法对应关系**:
- **FPE-GAN** (Fault Prediction Encoder GAN): 原PreGAN方法
- **TF-GAN** (Transformer-based Fault GAN): 原PreGANPlus方法
- **MAMO-GAN** (Migration-Aware Multi-Objective GAN): 我们的改进方法（原PreGANPlusEnhanced）

**注意**: 代码中仍使用原名称（PreGAN, PreGANPlus, PreGANPlusEnhanced），但在文档和提示信息中使用新命名。

---

## 🚀 快速开始

### 方法1: 使用实验脚本（推荐）⭐

使用 `scripts/run_experiment.sh` 可以自动运行完整对比实验：

```bash
# 激活conda环境
conda activate pregan

# 运行完整对比实验（FPE-GAN, TF-GAN, MAMO-GAN）
bash scripts/run_experiment.sh
```

脚本会自动：
1. 运行所有方法的对比实验
2. 生成日志文件（带时间戳）
3. 生成对比图表

---

### 方法2: 使用批量运行脚本

使用 `scripts/batch_run_experiments.py` 可以自动运行多个方法并收集结果：

```bash
conda activate pregan

# 运行所有方法（包括MAMO-GAN）
python scripts/batch_run_experiments.py \
    --models PreGAN,PreGANPlus,PreGANPlusEnhanced \
    --steps 100

# 只运行我们的改进方法
python scripts/batch_run_experiments.py \
    --models PreGANPlusEnhanced \
    --steps 100
```

---

### 方法3: 运行重复实验（验证结果稳定性）⭐

使用 `scripts/run_repeat_experiments.sh` 可以重复运行最佳配置实验：

```bash
# 运行3次重复实验（默认）
bash scripts/run_repeat_experiments.sh

# 运行5次重复实验
bash scripts/run_repeat_experiments.sh 5
```

**配置**: 方案6最佳配置（MAMO-GAN）
- `energy_weight = 0.3`
- `response_time_weight = 0.3`
- `migration_cost_weight = 0.4`
- `migration_cost_threshold = 130`
- `cooldown_period = 3`
- `max_migrations_per_step = 3`

---

### 方法4: 在main.py中手动切换方法

编辑 `main.py` 的第119行，切换不同的Recovery方法：

```python
# 使用FPE-GAN (PreGAN)
recovery = PreGANRecovery(HOSTS, environment, training = True)

# 使用TF-GAN (PreGANPlus)
recovery = PreGANPlusRecovery(HOSTS, environment, training = True)

# 使用MAMO-GAN (PreGANPlusEnhanced)
recovery = PreGANPlusEnhancedRecovery(HOSTS, environment, training = True)
```

然后运行：
```bash
conda activate pregan
python main.py -e "" -m 2
```

---

## 📊 完整实验流程

### 标准对比实验

```bash
# 1. 激活conda环境
conda activate pregan

# 2. 运行所有方法的对比实验
python scripts/batch_run_experiments.py \
    --models PreGAN,PreGANPlus,PreGANPlusEnhanced \
    --steps 100

# 3. 生成对比图表
python grapher.py simulator

# 4. 查看结果
# 图表在 results/simulator/ 目录
# 数据文件在 all_datasets/simulator/<Model>/ 目录
```

---

## 🔧 配置说明

### batch_run_experiments.py 参数

- `--models`: 要运行的方法列表，用逗号分隔
  - 可选值：`PreGAN`, `PreGANPlus`, `PreGANPlusEnhanced`, `PCFT`, `DFTM`, `ECLB`, `CMODLB`
- `--steps`: 模拟步数（NUM_SIM_STEPS）
- `--dry-run`: 只显示将要运行的内容，不实际运行
- `--force-grapher`: 即使某些方法失败也运行grapher

### 示例

```bash
# 运行所有方法，100步
python scripts/batch_run_experiments.py \
    --models PreGAN,PreGANPlus,PreGANPlusEnhanced \
    --steps 100

# 只运行我们的方法，50步
python scripts/batch_run_experiments.py \
    --models PreGANPlusEnhanced \
    --steps 50

# 干运行（查看将要运行的内容）
python scripts/batch_run_experiments.py \
    --models PreGANPlusEnhanced \
    --steps 100 \
    --dry-run
```

---

## 📁 文件结构

运行实验后，文件结构如下：

```
PreGANPlus/
├── logs/                          # 原始日志和.pk文件
│   └── RPiEdge_BWGD2_.../
│       └── *.pk
├── all_datasets/                  # 按方法分类的.pk文件
│   └── simulator/
│       ├── PreGAN/                # FPE-GAN结果
│       │   └── *.pk
│       ├── PreGANPlus/            # TF-GAN结果
│       │   └── *.pk
│       └── PreGANPlusEnhanced/    # MAMO-GAN结果
│           └── *.pk
├── results/                       # grapher生成的图表
│   └── simulator/
│       ├── Bar-*.pdf              # 柱状图对比
│       └── Series-*.pdf           # 时间序列图
├── experiment_logs/               # 实验日志
│   ├── experiment_YYYYMMDD_HHMMSS.log
│   └── final_experiments/         # 最终实验日志
│       └── experiment_*.log
└── scripts/                       # 实验脚本
    ├── run_experiment.sh          # 运行完整对比实验 ⭐
    ├── run_repeat_experiments.sh  # 运行重复实验 ⭐
    └── batch_run_experiments.py   # 批量运行脚本
```

---

## 🎯 实验建议

### 1. 基础对比实验

运行三个方法进行对比：
```bash
bash scripts/run_experiment.sh
```

### 2. 验证结果稳定性

运行重复实验：
```bash
bash scripts/run_repeat_experiments.sh 3
```

### 3. 测试模型实现

```bash
python scripts/test_mamo_gan.py
```

---

## 📈 查看结果

### 1. 查看图表

```bash
# 图表保存在
results/simulator/

# 主要图表：
# - Bar-*.pdf: 柱状图对比
# - Series-*.pdf: 时间序列图
```

### 2. 查看数据文件

```bash
# 数据文件在
all_datasets/simulator/PreGANPlusEnhanced/

# 可以用Python加载查看：
import pickle
with open('all_datasets/simulator/PreGANPlusEnhanced/xxx.pk', 'rb') as f:
    stats = pickle.load(f)
    # 查看stats.metrics等
```

### 3. 查看日志

```bash
# 查看最新日志
ls -lt experiment_logs/*.log | head -1 | awk '{print $NF}' | xargs tail -f

# 查看所有日志
ls -lt experiment_logs/
```

---

## ⚠️ 注意事项

1. **环境要求**: 必须在 `pregan` conda环境下运行
2. **磁盘空间**: 确保有足够的磁盘空间存储日志和结果
3. **运行时间**: 完整实验可能需要较长时间，建议使用screen或tmux
4. **模型文件**: 首次运行会训练模型，后续运行会加载已训练的模型

---

## 🐛 故障排除

### 问题1: 找不到模型文件

**错误**: `FileNotFoundError: ... .ckpt`

**解决**: 首次运行需要训练模型，确保有训练数据在 `recovery/PreGANSrc/data/simulator/`

### 问题2: grapher找不到结果

**错误**: `KeyError` 或模型不在结果中

**解决**: 
1. 检查 `all_datasets/simulator/PreGANPlusEnhanced/` 目录是否有.pk文件
2. 确保grapher.py中的Models列表包含 `PreGANPlusEnhanced`

### 问题3: 内存不足

**解决**: 减少 `--steps` 参数，或减少同时运行的方法数量

---

## 🎯 推荐命令（复制即可使用）

### 完整对比实验（推荐）⭐

```bash
bash scripts/run_experiment.sh
```

### 重复实验（验证稳定性）⭐

```bash
bash scripts/run_repeat_experiments.sh 3
```

---

## 📝 更新日志

- ✅ 已添加MAMO-GAN到main.py
- ✅ 已添加MAMO-GAN到batch_run_experiments.py
- ✅ 已添加MAMO-GAN到grapher.py的Models列表
- ✅ 修复了plotter.py中的模型名称解析问题
- ✅ 创建了实验脚本（run_experiment.sh, run_repeat_experiments.sh）
- ✅ 更新了方法命名（FPE-GAN, TF-GAN, MAMO-GAN）

---

## 🎉 开始实验

现在你可以开始运行实验了！

```bash
conda activate pregan
bash scripts/run_experiment.sh
```

---

*最后更新: 2026-01-03*

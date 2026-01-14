# 快速开始

**创建日期**: 2026-01-14

---

## 🚀 快速开始

### 前置要求

- Python 3.8+
- Conda环境管理工具
- 足够的磁盘空间（建议至少10GB）

### 第一步：环境配置

```bash
# 激活conda环境
conda activate pregan

# 如果环境不存在，请先安装依赖（见安装指南）
```

### 第二步：运行实验

#### 方法1: 使用实验脚本（推荐）⭐

```bash
# 运行完整对比实验（FPE-GAN, TF-GAN, MAMO-GAN）
bash scripts/run_stage4_multiple.sh
```

脚本会自动：
1. 运行所有方法的对比实验
2. 生成日志文件（带时间戳）
3. 收集实验结果

#### 方法2: 使用批量运行脚本

```bash
# 运行所有方法
python scripts/batch_run_experiments.py \
    --models PreGAN,PreGANPlus,PreGANPlusEnhanced \
    --steps 100

# 只运行MAMO-GAN
python scripts/batch_run_experiments.py \
    --models PreGANPlusEnhanced \
    --steps 100
```

#### 方法3: 手动运行单个方法

编辑 `main.py`，切换不同的Recovery方法：

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
python main.py -e "" -m 0
```

---

## 📊 完整实验流程

### 标准四阶段实验

```bash
# 1. 阶段1：数据收集
python scripts/paper_experiment_stage1_data_collection.py
python main.py -e "" -m 0

# 2. 阶段2+3：编码器训练 + GAN训练（某个方法）
python scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python main.py -e "" -m 0

# 3. 阶段4：测试评估（某个方法）
python scripts/paper_experiment_stage4_testing.py --method PreGAN
python main.py -e "" -m 0
```

### 批量测试（推荐）

```bash
# 批量运行所有方法的测试
bash scripts/run_stage4_multiple.sh
```

---

## 📁 结果文件位置

运行实验后，结果文件保存在：

```
PreGANPlus/
├── experiment_logs/          # 实验日志
│   └── stage4_YYYYMMDD_HHMMSS/
│       └── <method>/
│           └── run_*.log
├── experiment_data/          # 实验数据
│   └── stage4_YYYYMMDD_HHMMSS/
│       └── <method>/
│           └── run_*/
│               └── *.pk
└── final_results/           # 最终结果
    ├── data/                # 选中的数据
    ├── logs/                # 选中的日志
    ├── summary/             # 汇总报告
    └── plots/               # 对比图表
```

---

## 🔧 配置说明

### batch_run_experiments.py 参数

- `--models`: 要运行的方法列表，用逗号分隔
  - 可选值：`PreGAN`, `PreGANPlus`, `PreGANPlusEnhanced`, `PCFT`, `DFTM`, `ECLB`, `CMODLB`
- `--steps`: 模拟步数（NUM_SIM_STEPS）
- `--dry-run`: 只显示将要运行的内容，不实际运行

### 示例

```bash
# 运行所有方法，100步
python scripts/batch_run_experiments.py \
    --models PreGAN,PreGANPlus,PreGANPlusEnhanced \
    --steps 100

# 只运行MAMO-GAN，50步
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

## 📈 查看结果

### 1. 查看日志

```bash
# 查看最新日志
ls -lt experiment_logs/stage4_*/PreGANPlusEnhanced/*.log | head -1 | xargs tail -f
```

### 2. 查看数据

```bash
# 数据文件在
experiment_data/stage4_*/PreGANPlusEnhanced/run_*/

# 可以用Python加载查看：
import pickle
with open('experiment_data/.../xxx.pk', 'rb') as f:
    stats = pickle.load(f)
    # 查看stats.metrics等
```

### 3. 查看最终结果

```bash
# 最终结果在
final_results/

# 汇总报告
cat final_results/summary/最终选择结果报告.md

# 对比图表
ls final_results/plots/
```

---

## ⚠️ 注意事项

1. **环境要求**: 必须在 `pregan` conda环境下运行
2. **磁盘空间**: 确保有足够的磁盘空间存储日志和结果（建议至少10GB）
3. **运行时间**: 完整实验可能需要较长时间，建议使用screen或tmux
4. **模型文件**: 首次运行会训练模型，后续运行会加载已训练的模型

---

## 🐛 故障排除

### 问题1: 找不到模型文件

**错误**: `FileNotFoundError: ... .ckpt`

**解决**: 首次运行需要训练模型，确保有训练数据在 `recovery/PreGANSrc/data/simulator/`

### 问题2: 内存不足

**解决**: 减少 `--steps` 参数，或减少同时运行的方法数量

### 问题3: 权限错误

**解决**: 确保有写入权限，必要时使用 `chmod` 或 `chown`

---

## 🔗 相关文档

- [安装指南](Installation.md) - 环境配置和依赖安装
- [高级用法](Advanced_Usage.md) - 高级功能和自定义
- [实验流程说明](../02_Experiments/Experimental_Workflow.md) - 详细流程
- [实验参数配置](../02_Experiments/Experimental_Configuration.md) - 参数说明

---

**最后更新**: 2026-01-14

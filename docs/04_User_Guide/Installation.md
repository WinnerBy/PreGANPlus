# 安装指南

**创建日期**: 2026-01-14

---

## 📋 概述

本文档详细说明系统要求、依赖安装、环境配置和验证安装的步骤。

---

## 💻 系统要求

### 操作系统

- **推荐**: Linux (Ubuntu 20.04+)
- **已测试**: Linux 6.12.54-linuxkit
- **其他**: macOS, Windows (需要相应调整)

### 硬件要求

- **CPU**: 建议多核CPU（实验使用16主机模拟）
- **内存**: 建议至少16GB RAM
- **存储**: 建议至少50GB可用空间
- **GPU**: 可选（用于加速训练，CPU也可运行）

---

## 🔧 依赖安装

### 方法1: 使用Conda（推荐）

#### 1. 创建 Conda 环境

```bash
# 创建名为 pregan_env 的 conda 环境（与当前实验一致）
conda create -n pregan_env python=3.9

# 激活环境
conda activate pregan_env
```

**注意**: 在 conda 环境下请使用 `python` 而非 `python3`，以使用该环境的解释器。

#### 2. 安装PyTorch

```bash
# 根据CUDA版本选择（如果有GPU）
# CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# CPU版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### 3. 安装其他依赖

```bash
# 安装基础依赖
pip install numpy pandas matplotlib scipy seaborn

# 安装DGL（图神经网络库）
pip install dgl

# 安装其他工具
pip install tqdm scikit-learn
```

### 方法2: 使用pip（如果没有Conda）

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装PyTorch
pip install torch torchvision torchaudio

# 安装其他依赖
pip install numpy pandas matplotlib scipy seaborn dgl tqdm scikit-learn
```

---

## ✅ 环境验证

### 1. 验证Python版本

```bash
python --version
# 应该显示 Python 3.8+ 或 3.9+
```

### 2. 验证关键库

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import dgl; print(f'DGL: {dgl.__version__}')"
```

### 3. 验证CUDA（如果有GPU）

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. 验证项目结构

```bash
# 检查关键目录
ls -d recovery/PreGANSrc/
ls -d scripts/
ls -d recovery/PreGANSrc/data/simulator/ 2>/dev/null || echo "数据目录不存在（首次运行需要先收集数据）"
```

---

## 📁 目录结构

确保项目目录结构正确：

```
PreGANPlus/
├── recovery/                    # 恢复方法实现
│   ├── PreGANSrc/              # GAN方法源码
│   │   ├── src/                # 模型定义
│   │   ├── data/               # 训练数据（首次运行后生成）
│   │   └── checkpoints/        # 模型checkpoint（训练后生成）
│   └── CMODLBSrc/              # CMODLB方法源码
├── scripts/                     # 实验脚本
├── experiment_data/            # 实验数据（运行后生成）
├── experiment_logs/            # 实验日志（运行后生成）
└── final_results/              # 最终结果（运行后生成）
```

---

## 🚀 快速验证

运行一个简单的测试验证安装：

```bash
# 激活环境
conda activate pregan_env

# 测试导入（使用 python 而非 python3）
python -c "
import torch
import numpy as np
import pandas as pd
import dgl
print('✅ 所有依赖安装成功！')
"
```

---

## 🔗 相关文档

- [快速开始](Quick_Start.md) - 快速开始使用
- [实验环境设置](../02_Experiments/Experimental_Setup.md) - 详细环境配置
- [高级用法](Advanced_Usage.md) - 高级功能和自定义

---

**最后更新**: 2026-01-14

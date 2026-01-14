# 实验环境设置

**创建日期**: 2026-01-14

---

## 📋 概述

本文档详细说明实验的硬件环境、软件环境、依赖安装和环境验证。

---

## 🖥️ 硬件环境

### 计算资源

- **CPU**: 建议多核CPU（实验使用16主机模拟）
- **内存**: 建议至少16GB RAM
- **存储**: 建议至少50GB可用空间（用于存储实验数据和日志）

### 存储配置

- **实验数据**: `experiment_data/` 目录
- **实验日志**: `experiment_logs/` 目录
- **模型checkpoint**: `recovery/PreGANSrc/checkpoints/` 目录
- **最终结果**: `final_results/` 目录

---

## 💻 软件环境

### 操作系统

- **推荐**: Linux (Ubuntu 20.04+)
- **已测试**: Linux 6.12.54-linuxkit

### Python版本

- **要求**: Python 3.8+
- **推荐**: Python 3.9 或 3.10

### 关键依赖库

- **PyTorch**: 1.12.0+ (用于深度学习模型)
- **NumPy**: 1.21.0+ (数值计算)
- **Pandas**: 1.3.0+ (数据处理)
- **Matplotlib**: 3.5.0+ (数据可视化)
- **DGL**: 0.9.0+ (图神经网络，用于FPE编码器)
- **SciPy**: 1.7.0+ (科学计算)
- **Seaborn**: 0.11.0+ (统计可视化)

---

## 🔧 环境配置

### Conda环境设置

#### 1. 创建Conda环境

```bash
# 创建名为pregan的conda环境
conda create -n pregan python=3.9

# 激活环境
conda activate pregan
```

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

# 安装其他依赖（如果有requirements.txt）
pip install -r requirements.txt
```

### 依赖安装步骤

#### 方法1: 使用requirements.txt（如果存在）

```bash
conda activate pregan
pip install -r requirements.txt
```

#### 方法2: 手动安装

```bash
conda activate pregan

# 核心依赖
pip install numpy==1.21.6
pip install pandas==1.3.5
pip install matplotlib==3.5.3
pip install scipy==1.7.3
pip install seaborn==0.12.0

# 深度学习
pip install torch==1.12.0
pip install dgl==0.9.1

# 其他工具
pip install tqdm
pip install scikit-learn
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

## 📁 目录结构要求

### 必需目录

```
PreGANPlus/
├── recovery/                    # 恢复方法实现
│   ├── PreGANSrc/              # GAN方法源码
│   │   ├── src/                # 模型定义
│   │   ├── data/               # 训练数据
│   │   └── checkpoints/        # 模型checkpoint
│   └── CMODLBSrc/              # CMODLB方法源码
├── scripts/                     # 实验脚本
├── experiment_data/            # 实验数据（运行后生成）
├── experiment_logs/            # 实验日志（运行后生成）
└── final_results/              # 最终结果（运行后生成）
```

### 数据目录

- **训练数据**: `recovery/PreGANSrc/data/simulator/`
  - `time_series.npy` - 时间序列数据
  - `schedule_series.npy` - 调度序列数据
- **首次运行**: 需要先运行阶段1数据收集，生成训练数据

---

## 🔗 相关文档

- [实验流程说明](Experimental_Workflow.md) - 详细实验流程
- [实验参数配置](Experimental_Configuration.md) - 参数说明
- [快速开始](../04_User_Guide/Quick_Start.md) - 快速开始使用

---

**最后更新**: 2026-01-14

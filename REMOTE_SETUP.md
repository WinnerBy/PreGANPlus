# 远程服务器运行指南

## 📋 快速开始

在远程服务器上运行改进的Stage1数据生成，实现高故障率数据生成（15-25%故障率）。

## 🚀 一键运行（推荐）

```bash
# 1. 克隆最新代码
git clone https://github.com/WinnerBy/PreGANPlus.git
cd PreGANPlus

# 2. 激活conda环境
conda activate pregan_env

# 3. 运行改进的Stage1（2000步，15个新容器）
# 预期耗时：60-90分钟，生成15-25%故障率的数据
python scripts/stage1_data_generation_improved.py --steps 2000 --new-containers 15

# 4. 查看输出数据
# - logs/RPiEdge_*/time_series.npy
# - logs/RPiEdge_*/schedule_series.npy  
# - logs/RPiEdge_*/fault_history.pkl
# 数据会自动复制到: recovery/PreGANSrc/data/simulator/
```

## ⚙️ 详细步骤

### 第1步：环境准备

```bash
# SSH连接到远程服务器
ssh user@remote_server

# 克隆代码
git clone https://github.com/WinnerBy/PreGANPlus.git
cd PreGANPlus

# 检查conda环境
conda env list
# 如果没有pregan_env，创建新环境：
conda create -n pregan_env python=3.10
conda activate pregan_env

# 安装依赖
pip install -r requirements.txt
# 或者使用环境文件
conda env update --file environment.yaml
```

### 第2步：运行Stage1改进版

```bash
# 激活环境
conda activate pregan_env

# 进入项目目录
cd PreGANPlus

# 运行改进的Stage1数据生成
python scripts/stage1_data_generation_improved.py \
  --steps 2000 \
  --new-containers 15 \
  --log-dir experiment_logs/stage1

# 监控运行进度
tail -f experiment_logs/stage1/stage1_improved_*.log
```

### 第3步：验证数据质量

```bash
# 查看生成的数据统计
python << 'EOF'
import numpy as np
from pathlib import Path
import pickle

data_dir = Path('recovery/PreGANSrc/data/simulator')

# 检查数据大小
time_series = np.load(data_dir / 'time_series.npy')
schedule = np.load(data_dir / 'schedule_series.npy')
with open(data_dir / 'fault_history.pkl', 'rb') as f:
    fault_history = pickle.load(f)

print(f"时间序列形状: {time_series.shape}")
print(f"调度序列形状: {schedule.shape}")
print(f"故障样本数: {len([f for f in fault_history.values() if f])}")
print(f"总样本数: {time_series.shape[0]}")

fault_rate = len([f for f in fault_history.values() if f]) / time_series.shape[0] * 100
print(f"故障率: {fault_rate:.2f}%")
print(f"预期范围: 15-25%")
EOF
```

## 📊 参数说明

### stage1_data_generation_improved.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--steps` | 2000 | 模拟步数（1步≈5分钟） |
| `--new-containers` | 15 | 每步新增容器数 |
| `--log-dir` | experiment_logs/stage1 | 日志输出目录 |

### 预期结果

- **故障率**：15-25%（目标提升）
- **故障样本数**：~3000个（从291增加）
- **正常样本数**：~13000个
- **生成耗时**：60-90分钟（取决于服务器性能）

## 🔄 后续流程

数据生成完成后，在远程服务器运行编码器训练：

```bash
# 训练所有GAN方法的编码器（共享Transformer）
python scripts/stage2_model_training.py \
  --method-set gan \
  --encoder-only \
  --steps 100

# 或训练所有消融模型的编码器
python scripts/stage2_model_training.py \
  --method-set ablation \
  --encoder-only \
  --steps 100
```

## 🚨 故障排除

### 问题1：ModuleNotFoundError: influxdb

```bash
# 解决方案：安装influxdb
pip install influxdb
```

### 问题2：GPU内存不足

```bash
# 减少容器数量
python scripts/stage1_data_generation_improved.py --steps 2000 --new-containers 10
```

### 问题3：运行速度慢

```bash
# 检查GPU是否被使用
nvidia-smi

# 或者在CPU上运行（较慢）
export CUDA_VISIBLE_DEVICES=''
python scripts/stage1_data_generation_improved.py ...
```

## 📁 数据目录结构

```
recovery/PreGANSrc/data/simulator/
├── time_series.npy          # 时间序列数据 (样本数 × 特征数)
├── schedule_series.npy       # 调度决策 (样本数 × 主机数)
└── fault_history.pkl        # 故障标签 {样本索引: 是否有故障}
```

## 💾 下载数据到本地

```bash
# 在本地机器上执行
rsync -avz user@remote_server:~/PreGANPlus/recovery/PreGANSrc/data/simulator/ ./data/

# 或使用scp
scp -r user@remote_server:~/PreGANPlus/recovery/PreGANSrc/data/simulator/ ./data/
```

## 🎯 性能对标

| 步数 | 新容器数 | 总容器 | 故障率 | 耗时 |
|------|---------|--------|--------|------|
| 1000 | 5 | 5000 | 1.82% | 30分钟 |
| 2000 | 15 | 30000 | 15-25% | 60-90分钟 |
| 3000 | 20 | 60000 | 20-30% | 90-120分钟 |

## 📝 日志和输出

- **运行日志**：`experiment_logs/stage1/stage1_improved_*.log`
- **生成数据**：`logs/RPiEdge_*/`
- **训练数据**：`recovery/PreGANSrc/data/simulator/`

## 🔗 相关文档

- [Stage1数据改进方案](docs/Stage1数据改进方案.md)
- [编码器改进_核心参考指南](docs/编码器改进_核心参考指南.md)
- [编码器改进综合执行方案](docs/编码器改进综合执行方案.md)

# TF-GAN 详细设计文档

**Transformer-based Fault GAN for Fault-Tolerant Edge Computing**

**创建日期**: 2026-01-14  
**方法代码名称**: PreGANPlus  
**论文命名**: TF-GAN

---

## 📋 概述

### 方法定位

TF-GAN (Transformer-based Fault GAN) 是对FPE-GAN的改进，将编码器从FPE升级为Transformer，提供更强的序列建模能力。这是方法演进的重要一步。

### 核心设计理念

1. **Transformer编码器**: 使用Transformer的注意力机制进行故障预测
2. **标准GAN**: 使用标准Generator和Discriminator进行调度优化
3. **在线调优机制**: 编码器在训练阶段进行在线调优，适应动态环境

### 相对于FPE-GAN的改进

| 方面 | FPE-GAN | TF-GAN | 改进说明 |
|------|---------|--------|----------|
| 编码器 | FPE_16 | Transformer_16 | Transformer更强的序列建模能力 |
| 训练机制 | 离线训练 | 在线调优 | 适应动态环境 |
| 迁移控制 | 无改进 | 无改进 | 迁移数减少4.85% |
| 能耗 | 1983.40 kWh | 1983.01 kWh | 略优 |
| 响应时间 | 214.02 s | 240.42 s | 更差 |

---

## 🏗️ 架构设计

### Transformer编码器架构

#### 网络结构

```python
Transformer_16(
    embedding: Linear(3 -> d_model)       # 输入嵌入层
    pos_encoder: PositionalEncoding()     # 位置编码
    encoder_layers: TransformerEncoderLayer * 2  # 2层Transformer编码器
    encoder: TransformerEncoder()         # Transformer编码器
    output_proj: Linear(d_model -> 2)      # 输出投影层
)
```

#### 输入输出

- **输入**:
  - `t`: [n_window, 48] - 时间序列数据（n_window个时间步，16主机×3指标）
  - `s`: [16, 16] - 当前调度矩阵
- **输出**:
  - `anomaly_scores`: [16, 2] - 每个主机的异常检测概率
  - `prototypes`: [16, 2] - 每个主机的原型向量

#### 关键组件详解

##### 1. 输入嵌入层

- **输入**: [n_window, 48] - 时间序列数据
- **输出**: [n_window, d_model] - 嵌入向量
- **作用**: 将输入特征映射到模型维度

##### 2. 位置编码

- **输入**: [n_window, d_model]
- **输出**: [n_window, d_model] - 带位置信息的嵌入向量
- **作用**: 为序列添加位置信息，使模型能够理解时间顺序

##### 3. Transformer编码器层

- **输入**: [n_window, d_model]
- **输出**: [n_window, d_model] - 编码后的特征
- **组件**:
  - 多头自注意力机制（Multi-Head Self-Attention）
  - 前馈神经网络（Feed-Forward Network）
  - 残差连接和层归一化
- **作用**: 捕获序列的长期依赖关系

##### 4. 输出投影层

- **输入**: [n_window, d_model] - Transformer编码后的特征
- **输出**: 
  - `anomaly_scores`: [16, 2] - 异常检测概率
  - `prototypes`: [16, 2] - 原型向量
- **作用**: 将编码特征映射到异常检测和原型向量

---

### Generator和Discriminator架构

TF-GAN使用与FPE-GAN相同的Generator和Discriminator：

- **Generator**: `Gen_16` - 标准Generator
- **Discriminator**: `Disc_16` - 标准Discriminator

详见 [FPE-GAN设计文档](FPE-GAN_Design.md#generator架构) 和 [FPE-GAN设计文档](FPE-GAN_Design.md#discriminator架构)。

---

## 🎓 训练流程

### 编码器训练

#### 训练数据

- **来源**: 阶段1收集的1000步数据
- **格式**: 
  - `time_series.npy`: [1001, 48] - 时间序列数据
  - `schedule_series.npy`: [1001, 16, 16] - 调度序列数据

#### 训练参数

- **Epochs**: 50
- **学习率**: 默认（在constants.py中定义）
- **优化器**: Adam
- **损失函数**: 异常检测损失 + 原型向量损失

#### 训练流程

1. **数据加载**: 加载阶段1收集的1000步数据
2. **窗口化**: 将时间序列转换为窗口数据
3. **前向传播**: 运行Transformer编码器，得到异常检测分数和原型向量
4. **损失计算**: 计算异常检测损失和原型向量损失
5. **反向传播**: 更新编码器参数
6. **保存checkpoint**: 每10个epoch保存一次

### GAN训练（与FPE-GAN相同）

#### 训练参数

- **Generator学习率**: 0.00005
- **Discriminator学习率**: 0.00005
- **优化器**: AdamW
- **损失函数**: BCELoss（二元交叉熵）

#### 训练流程（每个间隔）

1. **异常检测**: 运行Transformer编码器，检测故障
2. **如果检测到故障**:
   - **Discriminator训练**
   - **Generator训练**
   - **编码器调优**（新增）
3. **保存checkpoint**: 每10个间隔保存一次

---

## 🔍 推理流程

### 异常检测（与FPE-GAN相同）

1. **获取时间序列数据**: 从环境统计数据获取最新的时间序列
2. **数据预处理**: 归一化和窗口化
3. **运行编码器**: 得到异常检测分数和原型向量
4. **判断**: 如果所有主机都未检测到异常，返回原始决策

### 调度决策生成（与FPE-GAN相同）

1. **生成原型向量**: 只保留检测到异常的主机的原型向量
2. **生成新调度**: 使用Generator生成新调度
3. **评估**: Discriminator评估新调度是否优于原始调度
4. **决策选择**: 
   - 如果原始调度更好，返回原始决策
   - 否则，应用新调度

---

## 🔑 关键技术点

### 1. Transformer编码器的优势

**设计思想**: 使用Transformer的注意力机制，提供比FPE更强的序列建模能力。

**实现方式**:
- 多头自注意力机制捕获序列的长期依赖
- 位置编码使模型能够理解时间顺序
- 多层Transformer编码器提供更强的特征提取能力

**效果**: 
- 相比FPE编码器，能够捕获更长的依赖关系
- 在序列建模任务上表现更好

### 2. 在线调优机制

**设计思想**: 在训练阶段对编码器进行在线调优，使模型能够适应动态环境。

**实现方式**:
- 在GAN训练过程中，同时进行编码器调优
- 使用当前时间窗口的数据进行调优
- 保持编码器参数可训练（但checkpoint保存时冻结）

**效果**: 
- 编码器能够适应动态环境
- 提高故障预测的准确性

### 3. 迁移数减少

**设计思想**: 通过更强的序列建模能力，减少不必要的迁移。

**实现方式**:
- Transformer编码器提供更准确的故障预测
- 更准确的故障预测减少误报，从而减少不必要的迁移

**效果**: 
- 迁移数比FPE-GAN减少4.85%（157 vs 165）
- 能耗略优（-0.02%）

---

## 📊 与FPE-GAN的对比

| 方面 | FPE-GAN | TF-GAN | 改进 |
|------|---------|--------|------|
| 编码器 | FPE_16 | Transformer_16 | ✅ 更强的序列建模能力 |
| Generator | Gen_16 | Gen_16 | 相同 |
| Discriminator | Disc_16 | Disc_16 | 相同 |
| 训练机制 | 离线训练 | 在线调优 | ✅ 适应动态环境 |
| 能耗 | 1983.40 kWh | 1983.01 kWh | ✅ 略优 |
| 响应时间 | 214.02 s | 240.42 s | ❌ 更差 |
| 迁移次数 | 165 | 157 | ✅ 略少 |

---

## 💡 设计优势

1. **序列建模能力**: Transformer编码器提供比FPE更强的序列建模能力
2. **在线调优**: 编码器能够适应动态环境
3. **迁移控制**: 相比FPE-GAN，迁移数减少4.85%
4. **能耗优化**: 相比FPE-GAN，能耗略优

---

## ⚠️ 局限性

1. **响应时间**: 响应时间比FPE-GAN更差（+12.34%）
   - 这可能是因为Transformer在某些场景下过度优化能量
   - 这为MAMO-GAN的多目标优化提供了改进空间

2. **单一目标**: 只优化能量，可能导致其他指标变差
   - 这为MAMO-GAN的多目标优化提供了改进空间

---

## 📝 代码位置

- **实现文件**: `recovery/PreGANPlus.py`
- **编码器模型**: `recovery/PreGANSrc/src/models.py` - `Transformer_16`
- **Generator模型**: `recovery/PreGANSrc/src/models.py` - `Gen_16`
- **Discriminator模型**: `recovery/PreGANSrc/src/models.py` - `Disc_16`

---

## 🔄 方法演进

TF-GAN是FPE-GAN到MAMO-GAN的中间步骤：

```
FPE-GAN (基础)
    ↓ + Transformer编码器
TF-GAN (改进序列建模)
    ↓ + 迁移感知 + 多目标优化
MAMO-GAN (最优性能)
```

**演进意义**:
- 验证了Transformer编码器的有效性
- 为MAMO-GAN的多目标优化提供了基础
- 展示了方法演进的必要性

---

**最后更新**: 2026-01-14

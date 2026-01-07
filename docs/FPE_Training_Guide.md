# FPE 编码器训练指南

## 📋 FPE 训练机制说明

### 核心发现

**FPE（Fast Pattern Extraction）编码器的训练是在运行 `main.py` 时自动进行的，不需要单独的训练脚本。**

## 🔍 训练流程

### 1. 自动训练触发机制

在 `recovery/PreGAN.py` 的 `load_models()` 方法中：

```python
def load_models(self):
    # Load encoder model
    self.model, self.optimizer, self.epoch, self.accuracy_list = \
        load_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
    # Train the model if not trained
    if self.epoch == -1: self.train_model()  # ← 关键：自动训练
    # Freeze encoder
    freeze(self.model)
```

**训练触发条件**：
- 如果 checkpoint 文件不存在，`load_model()` 返回 `epoch = -1`
- 如果 `epoch == -1`，自动调用 `train_model()` 进行训练

### 2. 训练过程

`train_model()` 方法的执行流程：

```python
def train_model(self):
    self.model_plotter = Model_Plotter(self.env_name, self.model_name)
    folder = os.path.join(data_folder, self.env_name)  # 训练数据目录
    # 加载训练数据
    train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
    # 训练 num_epochs 个epoch（默认50个）
    for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
        loss, factor = backprop(...)  # 反向传播
        anomaly_score, class_score = accuracy(...)  # 计算准确率
        # 保存checkpoint
        save_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', ...)
```

### 3. 训练数据来源

训练数据存储在：
- **路径**: `recovery/PreGANSrc/data/{env_name}/`
- **文件**:
  - `time_series.npy` - 时间序列数据
  - `schedule_series.npy` - 调度序列数据

**数据生成**：
- 这些数据通常是通过之前的实验运行生成的
- 数据保存在 `logs/` 目录中，然后被处理并保存到 `data/` 目录

### 4. 训练参数

在 `recovery/PreGANSrc/src/constants.py` 中定义：

```python
num_epochs = 50  # 训练轮数
PERCENTILES = 98  # 异常检测百分位数
PROTO_DIM = 2  # 原型向量维度
PROTO_UPDATE_FACTOR = 0.2  # 原型更新因子
```

## 📁 相关文件位置

### Checkpoint 文件
- **FPE模型**: `recovery/PreGANSrc/checkpoints/simulator_FPE_16.ckpt`
- **Generator**: `recovery/PreGANSrc/checkpoints/simulator_Gen_16.ckpt`
- **Discriminator**: `recovery/PreGANSrc/checkpoints/simulator_Disc_16.ckpt`

### 训练数据
- **Simulator环境**: `recovery/PreGANSrc/data/simulator/`
- **Framework环境**: `recovery/PreGANSrc/data/framework/`

## ⚠️ 重要说明

### 1. 训练时机

- **首次运行**: 如果 checkpoint 不存在，会在运行 `main.py` 时自动训练
- **已训练模型**: 如果 checkpoint 存在且 `epoch >= 0`，直接加载，不进行训练
- **训练时间**: FPE训练需要一定时间（取决于数据量和epoch数）

### 2. 训练数据要求

- 训练数据必须存在于 `data/{env_name}/` 目录
- 如果数据不存在，训练会失败
- 数据通常来自之前的实验运行

### 3. 模型冻结

训练完成后，FPE编码器会被冻结（`freeze(self.model)`），在后续使用中不再更新参数。

## 🔧 如何重新训练 FPE

如果需要重新训练 FPE 模型：

### 方法1: 删除 checkpoint 文件

```bash
# 删除FPE checkpoint，下次运行时会自动重新训练
rm recovery/PreGANSrc/checkpoints/simulator_FPE_16.ckpt
```

### 方法2: 修改 checkpoint 中的 epoch

如果 checkpoint 存在但想重新训练，可以：
1. 加载 checkpoint
2. 将 epoch 设置为 -1
3. 保存 checkpoint

### 方法3: 创建单独的训练脚本（可选）

可以创建一个独立的训练脚本，但这不是必需的，因为训练已经在 `main.py` 中自动进行。

## 📊 训练监控

训练过程中会输出：
- **Loss**: 总损失（异常检测损失 + 三元组损失）
- **Anomaly Score**: 异常检测准确率
- **Class Score**: 类别分类准确率
- **Factor**: 原型更新因子

训练进度会显示在终端，并保存到 checkpoint 文件中。

## 💡 与 PreGANPlus 论文的对应

根据 PreGANPlus 论文：
- FPE 编码器需要**离线预训练**
- 训练使用历史时间序列数据和调度数据
- 训练完成后，编码器参数固定，用于在线推理

**当前实现符合论文描述**：
- ✅ FPE 在首次运行时自动训练（离线预训练）
- ✅ 训练完成后冻结参数
- ✅ 在线运行时只进行推理，不更新编码器参数

## 🔄 PreGANPlus 的区别

PreGANPlus 使用 Transformer 编码器，而不是 FPE：
- **Transformer 编码器**: `recovery/PreGANSrc/checkpointsplus/simulator_Transformer_16.ckpt`
- **训练机制**: 与 FPE 类似，也是在首次运行时自动训练
- **在线微调**: PreGANPlus 支持在线微调 Transformer 编码器（这是 PreGANPlus 的核心特性）

## 📝 总结

1. **FPE 训练是自动的**：在运行 `main.py` 时，如果检测到模型未训练，会自动进行训练
2. **不需要单独的训练脚本**：训练逻辑已经集成在 `PreGANRecovery` 类中
3. **训练数据来自历史实验**：存储在 `recovery/PreGANSrc/data/` 目录
4. **训练完成后模型被冻结**：确保在线运行时编码器参数不变

如果需要对 FPE 进行重新训练或调整训练参数，可以：
- 删除 checkpoint 文件强制重新训练
- 修改 `constants.py` 中的训练参数
- 更新训练数据


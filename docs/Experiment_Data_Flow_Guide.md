# 四步实验数据流完整指南

## 概览

本文档详细说明了论文实验的四个阶段如何使用数据、数据从何而来、如何确保数据一致性。

---

## 阶段1：数据收集（Data Collection）

### 目的
- 运行**无恢复机制**的模拟器（仅使用 Recovery 基类），采集真实的故障和异常数据
- 生成1000个时间步（每步300秒）的完整运行日志

### 运行配置（由 `paper_experiment_stage1_data_collection.py` 设置）
```python
NUM_SIM_STEPS = 1000  # 1000 个时间步
NEW_CONTAINERS = 5    # 每步创建5个新容器
recovery = Recovery() # 无恢复机制，仅收集数据
```

### 数据生成流程

```
main.py 启动
  ↓
initalizeEnvironment() → stats = Stats(workload, datacenter, scheduler)
  ↓
for step in range(1000):
    stepSimulation() → stats.saveStats()
      ↓
      stats.saveHostInfo() 保存：
        - CPU、RAM、Disk 指标（16个主机 × 3个指标）
        - 每步追加到 time_series: shape=[t, 48]
        - 调度矩阵（16容器 × 16主机）
        - 每步追加到 schedule_series: shape=[t, 16, 16]
  ↓
main.py 结束 → saveStats()
  ↓
stats.generateDatasets(dirname) 生成：
  📁 logs/RPiEdge_BWGD2_1000_16_16_1000_10000_300_5/
     ├── time_series.npy        (shape: [1001, 48])  # 1000步 + 初始行
     ├── schedule_series.npy    (shape: [1001, 16, 16])
     ├── [其他性能指标和图表]
```

**数据内容说明：**
- `time_series.npy`：
  - 维度：[1001时间步, 48维特征]
  - 每个时间步包含16个主机的CPU、RAM、Disk三个指标
  - 用于训练编码器（异常检测和原型学习）
  
- `schedule_series.npy`：
  - 维度：[1001时间步, 16容器, 16主机]
  - 每个时间步的当前调度矩阵（容器→主机分配）
  - 用于编码器输入和GAN训练

### 数据质量要求
- ✅ 数据应包含各种故障场景（因为没有恢复机制，会发生SLA违规、高能耗等）
- ✅ 数据量足够（1000步 ≈ 8.3小时运行时间，足以覆盖各类工作负载变化）
- ❌ 如果数据中出现明显的数据泄露（NaN、Inf），说明故障过于严重，应调整参数或增加数据量

---

## 阶段1 → 阶段2 数据转移

### 关键步骤：将阶段1数据拷贝到编码器训练目录

**在 `run_paper_experiment.sh` 中自动执行：**
```bash
# 找到最新生成的数据（按时间排序）
LATEST_LOG_DIR=$(find logs -maxdepth 1 -name "RPiEdge_BWGD2_1000*" -type d | sort -r | head -1)

# 拷贝到编码器训练数据目录
mkdir -p recovery/PreGANSrc/data/simulator
cp "$LATEST_LOG_DIR/time_series.npy" "recovery/PreGANSrc/data/simulator/time_series.npy"
cp "$LATEST_LOG_DIR/schedule_series.npy" "recovery/PreGANSrc/data/simulator/schedule_series.npy"
```

**或手动执行：**
```bash
# 选择你要使用的数据集
cp logs/RPiEdge_BWGD2_1000_16_16_1000_10000_300_5/time_series.npy recovery/PreGANSrc/data/simulator/
cp logs/RPiEdge_BWGD2_1000_16_16_1000_10000_300_5/schedule_series.npy recovery/PreGANSrc/data/simulator/
```

### 为什么需要这个步骤？
- 阶段1生成的数据保存在 `logs/` 目录（按运行时间戳组织，便于追溯）
- 所有编码器训练和推理都从固定位置 `recovery/PreGANSrc/data/simulator/` 读取
- 这样设计的好处：
  - 允许重复运行阶段1生成不同数据集，手动选择最佳数据
  - 阶段2-4的脚本不需要修改，自动使用 `recovery/PreGANSrc/data/simulator/` 中的数据
  - 便于对比不同数据集对模型性能的影响

---

## 阶段2：编码器训练（Encoder Training）

### 目的
- 使用阶段1的数据训练编码器（FPE、Transformer 或 FCN）
- 编码器学习时间序列异常检测和容器类型分类
- 训练好的编码器会冻结，在阶段3和4中仅用于推理

### 运行配置（由 `paper_experiment_stage2_encoder_training.py` 设置）
```python
NUM_SIM_STEPS = 10   # 短步数，仅用于触发编码器训练（不进行GAN训练）
NEW_CONTAINERS = 5
training = False     # 推理模式，但如果checkpoint不存在，编码器会自动训练
```

### 编码器训练流程

```python
# recovery/PreGAN.py (或 PreGANPlus.py)
class PreGANRecovery:
    def __init__(self, hosts, env, training=False):
        self.load_models()
    
    def load_models(self):
        # 尝试加载预存的编码器checkpoint
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, 'simulator_FPE_16.ckpt', 'FPE_16')
        
        # 关键：如果 epoch == -1，说明checkpoint不存在，自动训练
        if self.epoch == -1:
            self.train_model()  # ← 自动触发训练
    
    def train_model(self):
        # 从 recovery/PreGANSrc/data/simulator/ 加载数据
        folder = os.path.join(data_folder, self.env_name)  # = 'recovery/PreGANSrc/data/simulator'
        train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        
        # 使用阶段1的数据训练50个epoch
        for epoch in range(num_epochs):  # num_epochs=50
            loss, factor = backprop(epoch, self.model, train_time_data, ...)
            save_model(model_folder, 'simulator_FPE_16.ckpt', self.model, ...)
```

### 数据加载逻辑

```python
# recovery/PreGANSrc/src/utils.py
def load_dataset(folder, model):
    """
    folder = 'recovery/PreGANSrc/data/simulator'
    加载阶段1生成的数据进行训练
    """
    time_data = load_npyfile(folder, 'time_series.npy')        # [1001, 48]
    train_schedule_data = load_npyfile(folder, 'schedule_series.npy')  # [1001, 16, 16]
    
    # 数据预处理
    time_data = normalize_time_data(time_data)
    train_time_data = convert_to_windows(time_data, model)  # 转换为滑动窗口格式
    
    # 生成异常标签和类别标签（基于百分比阈值）
    anomaly_data, class_data = form_test_dataset(time_data)
    
    return train_time_data, train_schedule_data, anomaly_data, class_data
```

### 输出产物
- 训练好的编码器checkpoint，保存到 `recovery/PreGANSrc/checkpoints/simulator_FPE_16.ckpt`（或对应的Transformer_16等）
- 训练历史（loss、准确率）用于绘图

**关键保证：** ✅ 所有方法（PreGAN、PreGANPlus、CMODLB等）都使用**同一份** `recovery/PreGANSrc/data/simulator/` 数据训练编码器
- 这确保了公平的对比：不同方法只在GAN或其他模块上有差异，编码器来自相同数据

---

## 阶段3：GAN在线训练（Online GAN Training）

### 目的
- 在新的运行环境中，使用**在线学到的数据**训练GAN（生成器和判别器）
- GAN学习如何根据异常检测结果调整容器调度以优化能耗和响应时间
- 与阶段2不同，GAN不使用阶段1的离线数据，而是在运行中实时学习

### 运行配置（由 `paper_experiment_stage3_gan_training.py` 设置）
```python
NUM_SIM_STEPS = 1200  # 仅PreGANPlus；其他方法=300
NEW_CONTAINERS = 5
recovery = PreGANRecovery(HOSTS, env, training=True)  # 启用GAN在线训练
```

### GAN训练流程

```python
# recovery/PreGAN.py
class PreGANRecovery:
    def run_model(self, time_series, original_decision):
        # 1. 加载已训练的编码器（来自阶段2）
        self.model  # ← 已冻结，仅推理
        
        # 2. 使用编码器检测当前步骤的异常
        anomaly, prototype = self.run_encoder(...)
        
        # 3. 如果检测到异常，使用GAN生成新的调度
        if anomaly_detected:
            self.train_gan(embedding, schedule_data)  # ← 当前步的GAN训练
            return self.recover_decision(embedding, schedule_data, original_decision)
        else:
            return original_decision  # 无异常，返回原始决策
    
    def train_gan(self, embedding, schedule_data):
        """
        使用当前步的数据训练GAN一个batch
        - embedding: 编码器的异常原型输出
        - schedule_data: 当前的容器调度矩阵（来自 scheduler.result_cache）
        """
        # 训练判别器
        new_schedule = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule.detach())
        disc_loss = ...
        disc_loss.backward(); self.dopt.step()
        
        # 训练生成器
        probs = self.disc(schedule_data, new_schedule)
        gen_loss = ...
        gen_loss.backward(); self.gopt.step()
        
        # 保存checkpoint
        save_gan(model_folder, 'simulator_Gen_16.ckpt', 'simulator_Disc_16.ckpt', ...)
```

### 数据来源（在线生成，不使用阶段1数据）
- `embedding`：当前步骤编码器的异常原型输出
- `schedule_data`：当前的调度矩阵（来自 `scheduler.result_cache`）
- 这些数据来自运行时的实时统计，**不是阶段1的离线数据**

### 为什么GAN不用阶段1数据？
- 阶段1的数据是"无恢复"场景的数据，包含大量SLA违规
- GAN的目的是在**有编码器**的场景下学习最优调度
- 如果用阶段1数据，就会学习"应对故障"而不是"预防故障"
- 在线训练使GAN适应当前的运行环境

---

## 阶段4：测试评估（Testing and Evaluation）

### 目的
- 在独立的运行环境中，评估所有方法（包括传统方法如PCFT、DFTM）的性能
- 对比不同恢复机制的能耗、响应时间、SLA违规率

### 运行配置（由 `paper_experiment_stage4_testing.py` 设置）
```python
NUM_SIM_STEPS = 100   # 100个时间步的测试运行
NEW_CONTAINERS = 5
recovery = PreGANRecovery(HOSTS, env, training=False)  # 推理模式，不训练
```

### 测试流程

```python
# 对于预测性方法（PreGAN、PreGANPlus等）：
for step in range(100):
    stepSimulation(recovery=PreGANRecovery(..., training=False)):
        # 使用阶段2训练的编码器（已冻结）
        anomaly, prototype = encoder.forward(current_data)
        
        # 使用阶段3训练的GAN进行推理（不再训练）
        new_schedule = gan.forward(anomaly, current_schedule)
        
        # 评估新的调度
        metrics = evaluate(new_schedule)
```

### 数据来源
- **编码器数据**：使用阶段1数据 `recovery/PreGANSrc/data/simulator/` 作为标准化基准
  - 在 `normalize_test_time_data()` 中使用训练数据的范围进行归一化
- **当前运行数据**：实时从当前模拟环节获取（不同于阶段1）
- **GAN数据**：仅用于推理，不训练

### 评估指标
保存在日志中的指标：
- `numdestroyed`：测试期间销毁的容器数
- `nummigrations`：总迁移次数
- `energytotalinterval`：总能耗
- `avgresponsetime`：平均响应时间
- `avgmigrationtime`：平均迁移时间
- `slaviolations`：SLA违规数

---

## 数据一致性检查清单

### ✅ 确保数据流正确

| 阶段 | 数据输入 | 数据输出 | 检查方法 |
|------|--------|--------|--------|
| 1 | Recovery() 无恢复 | logs/RPiEdge_BWGD2_1000*/ | `ls -lh logs/` 检查文件大小 |
| 1→2 | logs 目录数据 | recovery/PreGANSrc/data/simulator/ | 检查文件是否拷贝成功 `ls -l recovery/PreGANSrc/data/simulator/` |
| 2 | 阶段1数据 | Checkpoint (*.ckpt) | 检查编码器文件 `ls -l recovery/PreGANSrc/checkpoints/simulator_*` |
| 3 | 阶段2编码器 + 实时数据 | GAN Checkpoint | 检查GAN文件是否存在 |
| 4 | 阶段2编码器 + 阶段3 GAN | 性能指标 | 查看 logs/ 中的汇总统计 |

### ❌ 常见问题及排查

**问题1：阶段2报错"Data not found"**
- 原因：阶段1数据未拷贝到 `recovery/PreGANSrc/data/simulator/`
- 解决：
  ```bash
  mkdir -p recovery/PreGANSrc/data/simulator
  cp logs/RPiEdge_BWGD2_1000*/time_series.npy recovery/PreGANSrc/data/simulator/
  cp logs/RPiEdge_BWGD2_1000*/schedule_series.npy recovery/PreGANSrc/data/simulator/
  ```

**问题2：不确定用了哪份数据**
- 解决：在 `recovery/PreGANSrc/data/simulator/` 中添加README或时间戳
  ```bash
  echo "Data source: logs/RPiEdge_BWGD2_1000_16_16_1000_10000_300_5/" > recovery/PreGANSrc/data/simulator/README
  ```

**问题3：多次运行阶段1，logs目录混乱**
- 解决：使用脚本自动选择最新数据（已在 `run_paper_experiment.sh` 中实现）
- 或手动备份：
  ```bash
  mkdir -p data_backups
  cp -r logs/RPiEdge_BWGD2_1000* data_backups/backup_$(date +%Y%m%d_%H%M%S)/
  ```

---

## 总结：论文实验流程的合理性

### ✅ 合理之处

1. **分阶段设计明确职责：**
   - 阶段1：数据收集（无预防）
   - 阶段2：编码器训练（学习异常检测）
   - 阶段3：GAN训练（学习最优调度）
   - 阶段4：性能评估（对比）

2. **数据使用科学：**
   - 阶段2和4共享阶段1数据的统计特性（用于标准化）
   - 阶段3使用在线数据（模拟真实场景下的学习）
   - 避免数据泄露（训练数据和测试数据分离）

3. **编码器冻结保证公平：**
   - 所有方法用同一编码器，只在恢复机制上有差异
   - 便于对比GAN、传统方法等的真实效果差异

### ⚠️ 需要注意的地方

1. **数据拷贝是手动步骤：** 需要在脚本中显式处理（已在更新的 `run_paper_experiment.sh` 中改进）

2. **main.py 的 training 参数命名可能误导：**
   - 编码器无论 training 参数如何都会自动训练（如果checkpoint不存在）
   - 建议在代码注释中明确说明这一行为

3. **GAN的"在线训练"与"推理"需要明确区分：**
   - 阶段3：`training=True` 时，GAN每步都训练
   - 阶段4：`training=False` 时，GAN仅做推理
   - 建议添加日志确认当前处于哪个模式

---

## 推荐工作流

```bash
# 1. 完整实验流程（自动处理数据）
bash scripts/run_paper_experiment.sh

# 2. 如果中断，可以手动继续

# 重新运行阶段2（编码器训练）
python3 scripts/paper_experiment_stage2_encoder_training.py --method PreGANPlus
python3 main.py -e "" -m 0

# 重新运行阶段3（GAN训练）
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGANPlus
python3 main.py -e "" -m 0

# 重新运行阶段4（测试）
python3 scripts/paper_experiment_stage4_testing.py --method PreGANPlus
python3 main.py -e "" -m 0

# 3. 检查结果
ls -lh experiment_logs/paper_experiment/
tail -30 experiment_logs/paper_experiment/stage4_testing_PreGANPlus_*.log
```

---

## 附录：数据文件格式说明

### time_series.npy
- **Shape:** (1001, 48)
- **含义：** 1001个时间步，每步48维特征（16主机 × 3指标）
- **指标顺序：** [Host0_CPU, Host0_RAM, Host0_Disk, Host1_CPU, ...]
- **值域：** [0, 1]（已归一化）

### schedule_series.npy
- **Shape:** (1001, 16, 16)
- **含义：** 1001个时间步，每步16个容器到16个主机的分配矩阵
- **意义：** `schedule[t][c][h]` 表示时刻t容器c是否分配到主机h（0或正数表示分配、负数表示未使用）
- **值域：** 取决于调度器实现，通常为 [-1, 主机ID]

---

**文档版本：** 1.0
**最后更新：** 2026年1月8日
**维护者：** PreGANPlus开发团队

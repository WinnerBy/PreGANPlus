# 四步实验数据流——快速参考

## 一句话总结
**阶段1生成1000步"故障数据"→ 拷贝到固定目录 → 阶段2用此数据训练编码器（冻结）→ 阶段3-4用编码器+GAN在不同环境中评估**

---

## 数据路径流向图

```
阶段1（无恢复）
   ↓ 生成1000步数据
logs/RPiEdge_BWGD2_1000_*/ 
   ├─ time_series.npy          [1001×48]    16主机×3指标的时间序列
   └─ schedule_series.npy      [1001×16×16] 调度矩阵
   
   ↓ 自动拷贝（run_paper_experiment.sh 第44-57行）
   
recovery/PreGANSrc/data/simulator/   ← **所有方法的数据来源**
   ├─ time_series.npy
   └─ schedule_series.npy
   
   ↓ 
   
阶段2：编码器训练（FPE/Transformer）
  load_dataset("recovery/PreGANSrc/data/simulator/") ← 读取上面的数据
  → 训练50个epoch
  → 保存到 recovery/PreGANSrc/checkpoints/simulator_FPE_16.ckpt
  
   ↓ 编码器冻结
   
阶段3-4：在线推理（使用冻结的编码器 + 实时数据）
  - 编码器只做推理，不再训练
  - GAN根据当前数据动态调整（阶段3训练，阶段4推理）
```

---

## 关键答案

### Q1：阶段1数据是否在阶段2使用？
**答案：✅ 是**
```python
# recovery/PreGAN.py line 42
train_time_data, train_schedule_data, ... = load_dataset(
    folder="recovery/PreGANSrc/data/simulator",  # ← 阶段1的数据
    model=self.model
)
# 用这个数据训练编码器50个epoch
```

### Q2：如何确定各阶段用哪个数据？
**答案：自动选择最新**
```bash
# run_paper_experiment.sh 第44行
LATEST_LOG_DIR=$(find logs -maxdepth 1 -name "RPiEdge_BWGD2_1000*" -type d | sort -r | head -1)
# sort -r：按时间降序 | head -1：取最新
```
- 如果logs中有多个1000步数据，脚本自动使用**最新**的那个
- 或手动指定：编辑脚本第44行，写死目录名

### Q3：为什么各阶段都用同一个 recovery/PreGANSrc/data/simulator/?
**答案：保证公平、便于管理**
- 所有方法共享同一组训练数据 → 编码器特性一致
- 只在恢复机制上有差异（GAN vs 传统方法）
- 用固定目录避免每个方法指定不同路径的复杂性

### Q4：training=False 真的会阻止编码器训练吗？
**答案：❌ 不会**
```python
# recovery/PreGANPlus.py line 35-38
def __init__(self, hosts, env, training=False):
    self.training = training  # 这个参数被忽略了！
    self.load_models()

def load_models(self):
    if self.epoch == -1:  # ← 只看这个
        self.train_model()  # 无论 training 是什么，都训练！
```
- **第一次运行**（无checkpoint）：编码器自动训练，不管training参数
- **后续运行**（有checkpoint）：编码器冻结，仅推理
- `training=False` 的作用：禁止GAN训练，仅做推理

---

## 实验合理性检查清单

| 检查项 | 状态 | 说明 |
|-------|------|------|
| 📊 数据分离 | ✅ | 阶段1是"无恢复故障场景"，包含真实异常，用于学习异常特征 |
| 🔄 编码器复用 | ✅ | 所有方法共享同一编码器，只在恢复模块不同，便于对比 |
| 🧊 编码器冻结 | ✅ | 阶段2后编码器冻结，不再更新，避免对比时引入新的变化因素 |
| ⏱️ 在线学习 | ✅ | 阶段3用运行时的实时数据训练GAN，模拟真实场景 |
| 🚫 无数据泄露 | ✅ | 阶段4的测试数据完全独立，不与阶段1或2的训练数据重叠 |
| 📌 可重复性 | ✅ | 自动选择最新数据，脚本记录所有步骤和日志，易于复现 |

---

## 常见问题排查

### 问题1：阶段2报错 "Data not found"
```
解决步骤：
1. 检查阶段1是否完成：ls logs/RPiEdge_BWGD2_1000*/
2. 检查数据是否拷贝：ls -l recovery/PreGANSrc/data/simulator/
3. 手动拷贝：
   cp logs/RPiEdge_BWGD2_1000_*/time_series.npy recovery/PreGANSrc/data/simulator/
```

### 问题2：运行了多次阶段1，不知道用哪个数据
```
解决方案：
- 当前脚本自动用最新的（sort -r | head -1）
- 如果要用特定时间戳的数据，编辑第44行改成：
  LATEST_LOG_DIR="logs/RPiEdge_BWGD2_1000_16_16_1000_10000_300_5"
- 或创建备份：
  cp logs/RPiEdge_BWGD2_1000_* data_backups/  # 自己管理
```

### 问题3：阶段2编码器没有训练，直接用旧checkpoint了
```
排查：
- 编码器只在 epoch==-1 时训练（checkpoint不存在）
- 检查checkpoint位置：ls recovery/PreGANSrc/checkpoints/
- 如果存在旧checkpoint，删除它重新训练：
  rm recovery/PreGANSrc/checkpoints/simulator_*
  # 然后重新运行阶段2
```

### 问题4：阶段3想重新训练GAN，但它加载了旧checkpoint
```
解决：
- 删除GAN checkpoint：rm recovery/PreGANSrc/checkpoints/simulator_Gen_* recovery/PreGANSrc/checkpoints/simulator_Disc_*
- 或在 stage3_gan_training.py 中添加 --reset-gan 参数
```

---

## 性能优化建议

| 建议 | 影响 | 实施难度 |
|------|------|--------|
| 增加阶段1步数（1000→2000）| ↑ 编码器准确度 | ⭐ |
| 增加阶段2 epoch（50→100）| ↑ 编码器收敛 | ⭐ |
| 增加阶段3步数（300→600）| ↑ GAN学习深度 | ⭐ |
| 增加阶段4步数（100→500）| ↑ 评估统计量 | ⭐ |
| 调整GAN学习率（1e-4→5e-5） | ↑ 训练稳定性 | ⭐⭐ |
| 使用不同seed运行多次阶段1 | ↑ 统计鲁棒性 | ⭐⭐ |

---

## 文件清单

```
数据文件：
  logs/RPiEdge_BWGD2_1000_*/
    ├─ time_series.npy          [1001, 48]
    └─ schedule_series.npy      [1001, 16, 16]
  recovery/PreGANSrc/data/simulator/  (拷贝的数据)
    ├─ time_series.npy
    └─ schedule_series.npy

模型文件：
  recovery/PreGANSrc/checkpoints/
    ├─ simulator_FPE_16.ckpt               (编码器)
    ├─ simulator_Transformer_16.ckpt      (可选)
    ├─ simulator_Gen_16.ckpt              (GAN生成器)
    └─ simulator_Disc_16.ckpt             (GAN判别器)

脚本：
  scripts/run_paper_experiment.sh         (主流程，包含数据拷贝)
  scripts/paper_experiment_stage1_data_collection.py
  scripts/paper_experiment_stage2_encoder_training.py
  scripts/paper_experiment_stage3_gan_training.py
  scripts/paper_experiment_stage4_testing.py

文档：
  docs/Experiment_Data_Flow_Guide.md      (详细说明)
  docs/User_Guide.md                      (整体使用)
```

---

## 一键验证数据完整性

```bash
# 复制以下命令一次执行
echo "=== 检查阶段1数据 ===" && \
ls -lh logs/RPiEdge_BWGD2_1000*/time_series.npy 2>/dev/null | tail -1 && \
echo "=== 检查拷贝的数据 ===" && \
ls -lh recovery/PreGANSrc/data/simulator/time_series.npy && \
echo "=== 检查编码器checkpoint ===" && \
ls -lh recovery/PreGANSrc/checkpoints/simulator_*FPE* && \
echo "✅ 所有数据完整！"
```

---

**最后更新：** 2026年1月8日  
**关联文档：** docs/Experiment_Data_Flow_Guide.md

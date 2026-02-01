# Stage2 模型训练与数据分析

**更新日期**: 2026-01  
**说明**: 基于 `experiment_logs/stage2/` 下保留的最后一次成功训练日志，围绕**故障检测**与**迁移决策**两条线汇总与分析，服务于论文中的方法对比与消融讨论。

---

## 一、实验设置与训练类型

### 1.1 Stage2 仿真配置（本实验采用）

- **步数**: 1200  
- **每步新增容器**: 8  
- **数据与标签**: 与 Stage1 一致，使用 ADE 故障定义；数据目录 `recovery/PreGANSrc/data/simulator/`（如 400×8 配置 A 数据）。

详见 [Experiment_Setup_And_Fault_Design.md](Experiment_Setup_And_Fault_Design.md)。

### 1.2 训练类型

| 类型 | 说明 |
|------|------|
| **encoder-only** | 仅训练编码器（故障检测），不跑仿真、不训练 GAN |
| **GAN-only** | 加载已训练编码器，在仿真中只训练 GAN（迁移决策） |
| **full** | 同一次运行内先/同时训练 encoder + GAN |

### 1.3 编码器共用

PreGANPlus 与 PreGANPlusEnhanced **共用 Transformer_16** 编码器；该编码器只训练一次（如日志 `stage2_PreGANPlus_*_encoder_*`），后续 PreGANPlus 与 PreGANPlusEnhanced 的 GAN 训练分别使用同一份 Transformer_16 checkpoint，仅 GAN 结构不同（Gen_16/Disc_16 vs Gen_16_MigrationAware/Disc_16_MultiObjective）。

---

## 二、故障检测训练性能与对比

本节在**编码器阶段**的指标上对比：Loss、ALoss、精确率 P、召回率 R、F1、AScore、CScore（验证/评估集）。CMODLB 为 FCN 回归，仅记录 Loss，无 P/R/F1。

### 2.1 PreGAN 系列三种方法的递进对比（故障检测）

| 模型 | 编码器 | Epoch 数 | 最终 Loss | ALoss | P | R | F1 | AScore | CScore |
|------|--------|---------|-----------|-------|---|---|---|--------|--------|
| **PreGAN** | FPE_16 | 300 | 0.278 | 0.376 | 0.813 | 1.0 | 0.897 | 0.988 | 0.9999 |
| **PreGANPlus / PreGANPlusEnhanced** | Transformer_16（共用） | 150 | **-0.149** | **0.046** | **0.933** | 1.0 | **0.965** | **0.996** | 0.9999 |

**结论（故障检测）**：在相同数据与 ADE 标签下，Transformer_16 在更少 epoch（150）下即达到更低的 Loss、更高的 P/F1 与 AScore，说明**引入 Transformer 替代 FPE 有利于故障检测**；PreGAN 的 FPE_16 训练 300 epoch 后仍弱于 Transformer_16，体现递进式改进。

### 2.2 与 CMODLB 的训练对比（故障检测）

| 模型 | 编码器 | Epoch 数 | 最终 Loss | P / R / F1 / AScore |
|------|--------|---------|-----------|----------------------|
| **CMODLB** | FCN_16 | 30 | 0.666 | 无（回归任务，KMeans 聚类决策） |
| **PreGAN** | FPE_16 | 300 | 0.278 | P=0.81, R=1.0, F1=0.90, AScore=0.99 |
| **PreGANPlus / PreGANPlusEnhanced** | Transformer_16 | 150 | -0.149 | P=0.93, R=1.0, F1=0.97, AScore=0.996 |

**结论**：PreGAN 系列在可度量的故障检测指标上优于 CMODLB 的 FCN 训练（Loss 更低，且具备 P/R/F1/AScore）；CMODLB 的完整效果需在 Stage3 迁移决策与系统指标上再比较。

---

## 三、消融实验与基准模型（PreGANPlusEnhanced）

**基准模型**：PreGANPlusEnhanced（Transformer_16 + 迁移感知生成器 + 多目标判别器）。消融均在**同一数据、同一 ADE 故障定义**下训练。

### 3.1 基准：PreGANPlusEnhanced

- **编码器**: Transformer_16，150 epoch；Loss=-0.149，P=0.933，R=1.0，F1=0.965，AScore=0.996。  
- **GAN**: 加载上述编码器，在仿真中训练至约 952 个 GAN epoch（MigrationAware + MultiObjective）。

### 3.2 消融一：去掉 Transformer（AblationNoTransformer）

- **编码器**: FPE_16，300 epoch；**Loss=0.238**，**P=0.816**，F1=0.899，**AScore=0.988**。  
- **与基准对比**: Loss 升高、P/F1/AScore 均低于基准，说明**用 Transformer 替代 FPE 对故障检测有正向贡献**。

### 3.3 消融二：去掉 GAT（AblationNoGAT）

- **编码器**: TransformerNoGAT_16，300 epoch；最终（Epoch 299）Loss=-0.294，P=0.976，F1=0.988，AScore=0.999。  
- **相同轮数（150 epoch）对比**：取 NoGAT 第 150 轮（Epoch 149）与基准 Transformer_16 第 150 轮对齐：

| 指标 | Transformer_16（基准，150 epoch） | TransformerNoGAT_16（NoGAT，150 epoch） |
|------|-----------------------------------|------------------------------------------|
| Loss | -0.149 | **-0.291** |
| ALoss | 0.046 | **0.038** |
| P | 0.933 | **0.963** |
| R | 1.0 | 1.0 |
| F1 | 0.965 | **0.981** |
| AScore | 0.996 | **0.998** |

在**相同 150 轮**下，NoGAT 的故障检测指标略优于基准；GAT 在 150 epoch 内未体现明显增益，其系统级收益在 Stage3 迁移决策中验证（NoGAT 迁移次数与方差明显变差）。

### 3.4 消融三：去掉迁移感知（AblationNoMigrationAware）

- **编码器**: 与基准相同（Transformer_16），结构无差异；若 epoch 或运行不同，编码器指标差异应视作**误差/波动**，而非性能变化。  
- **GAN**: 标准 Gen_16/Disc_16；DLoss 明显高于基准，说明**迁移感知有利于判别器与迁移决策**。

### 3.5 消融四：去掉多目标（AblationNoMultiObjective）

- **编码器**: 与基准相同（Transformer_16），结构无差异；编码器指标与基准的差异应视作**误差**，不作“优于/略优”的性能结论。  
- **GAN**: GLoss/DLoss 数值上更低，但多目标约束的是能量/时延/迁移代价等，需在 Stage3 系统指标上评估（去掉多目标后能耗全表最高）。

### 3.6 消融小结（对训练指标的影响）

| 消融 | 故障检测（相对基准） | 迁移决策 / GAN 训练 |
|------|----------------------|----------------------|
| **NoTransformer** | Loss↑、P/F1/AScore↓，明显变差 | 结构同基准 GAN，收敛正常 |
| **NoGAT** | 150 ep 同条件下 NoGAT 略优；300 ep 下 NoGAT 更优 | 编码器更强；GAT 系统级收益见 Stage3 |
| **NoMigrationAware** | 编码器与基准相同，差异视作误差 | DLoss 升高，迁移感知有利于决策 |
| **NoMultiObjective** | 编码器与基准相同，差异视作误差 | GLoss/DLoss 更低，多目标对系统指标见 Stage3 |

---

## 四、日志与配置速查

| 日志文件名（示例） | 模型 | 类型 | 编码器 | 编码器 Epoch | GAN 备注 |
|--------------------|------|------|--------|--------------|----------|
| stage2_PreGANPlus_*_encoder_*.log | PreGANPlus + PreGANPlusEnhanced | encoder-only | Transformer_16 | 150 | 共用编码器 |
| stage2_PreGAN_*_encoder_*.log | PreGAN | encoder-only | FPE_16 | 300 | - |
| stage2_PreGAN_*_gan_*.log | PreGAN | GAN-only | 加载 FPE_16 | - | Gen_16/Disc_16 |
| stage2_PreGANPlus_*_gan_*.log | PreGANPlus | GAN-only | 加载 Transformer_16 | - | Gen_16/Disc_16 |
| stage2_PreGANPlusEnhanced_*_gan_*.log | PreGANPlusEnhanced | GAN-only | 加载 Transformer_16 | - | MigrationAware + MultiObjective |
| stage2_CMODLB_full_*.log | CMODLB | full | FCN_16 | 30 | 无 GAN |
| stage2_AblationNoTransformer_full_*.log | AblationNoTransformer | full | FPE_16 | 300 | MigrationAware + MultiObjective |
| stage2_AblationNoGAT_*_encoder_*.log / *_gan_*.log | AblationNoGAT | encoder-only / GAN-only | TransformerNoGAT_16 | 300 | MigrationAware + MultiObjective |
| stage2_AblationNoMigrationAware_full_*.log | AblationNoMigrationAware | full | Transformer_16 | 300 | Gen_16/Disc_16 |
| stage2_AblationNoMultiObjective_full_*.log | AblationNoMultiObjective | full | Transformer_16 | 300 | MigrationAware，判别器非多目标 |

---

## 五、小结

- **故障检测**：PreGAN 系列递进上，Transformer_16（150 epoch）在 Loss、P、F1、AScore 上均优于 FPE_16（300 epoch）；与 CMODLB 的 FCN 相比，PreGAN 系列在可度量的故障检测指标上表现更好。  
- **消融**：以 PreGANPlusEnhanced 为基准，去掉 Transformer 会明显降低故障检测性能；去掉 GAT 时，在相同 150 轮下 TransformerNoGAT 的故障检测略优，GAT 在 Stage3 迁移控制上体现关键作用；去掉迁移感知会抬高 GAN DLoss；去掉多目标会得到更低的 GLoss/DLoss，但多目标对能耗等系统指标需在 Stage3 评估。

**相关文档**: [Experiment_Setup_And_Fault_Design.md](Experiment_Setup_And_Fault_Design.md)、[Stage1_Data_And_Analysis.md](Stage1_Data_And_Analysis.md)、[Stage3_Results_Analysis.md](../03_Results/Stage3_Results_Analysis.md)、[Fault_Detection_Metrics_For_Paper.md](../03_Results/Fault_Detection_Metrics_For_Paper.md)。

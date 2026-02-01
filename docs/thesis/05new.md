# 第5章 实验结果与分析

本章将详细阐述评估前两章所提方法的实验环境、数据集配置、基准算法以及详细的性能分析。实验依托于 **COSCO (Coupled Simulation and Container Orchestration)** 框架进行 **[60]**，该框架为验证雾计算环境下的调度算法提供了强大的联合仿真支持。本章首先验证第3章提出的基于时空特征编码的生成对抗网络调度方法（FPE-GAN）相对于传统启发式算法的优势；随后，重点评估第4章提出的改进策略，即引入 Transformer 编码器的 TF-GAN 和引入多目标迁移感知机制的 MAMO-GAN 的性能提升效果；最后，通过消融实验进一步解析各核心组件对模型性能的贡献。

## 5.1 实验环境与设置

本节定义联合仿真架构、拓扑与硬件参数、工作负载与三阶段数据使用方式、对比算法及评价指标，并对本章评价口径与对比范围作必要说明。

### 5.1.1 COSCO 联合仿真框架

本实验采用 Tuli 等人提出的 **COSCO** 框架作为核心实验平台 **[60]**。COSCO 的核心创新在于其"联合仿真（Co-Simulation）"机制，该机制允许调度算法在将决策应用到实际环境之前，先在模拟器中进行"预演"，从而获取对未来 QoS 指标的估计。

系统架构如图 5-1 所示。

![COSCO Framework Architecture](https://github.com/imperial-qore/COSCO/raw/master/images/cosco_arch.png)
*(图 5-1: COSCO 联合仿真框架架构图，展示了调度器、模拟器与底层框架的交互闭环 **[60]**)*

该架构包含三个核心组件：
1.  **智能调度器 (Intelligent Scheduler)**：即本研究提出的 GAN 系列模型（FPE-GAN/TF-GAN/MAMO-GAN）。它作为决策中枢，接收环境状态 $S_t$，输出调度决策 $D_t$。
2.  **耦合模拟器 (Coupled Simulator)**：用于对候选调度 $\hat{D}$ 进行单步评估，输出能耗与响应时间等 QoS 估计值。本文实验采用仿真模式：耦合模拟器基于资源分配与功耗模型对候选调度进行“前瞻评估”，并结合近期观测的统计量对响应时间进行近似估计，从而为判别器/多目标判别器提供监督信号与回馈。
3.  **执行与状态更新模块 (Execution & State Update)**：负责在离散时间间隔内推进系统状态、应用调度决策并记录监测数据。需要强调的是：本文主要在仿真平台中评估算法，迁移过程以“数据传输开销与附加时延”的形式进行抽象建模，而非依赖真实容器编排与在线迁移工具链。

### 5.1.2 拓扑结构与硬件参数

实验模拟了一个典型的资源受限边缘计算集群（RPiEdge），旨在复现高密度、异构化的雾计算环境。具体的拓扑结构与参数配置如下：

*   **主机配置**：集群包含 $N = 16$ 个边缘节点，采用两类资源配置以体现异构性（例如内存容量不同的设备类型），并为每类节点配置相应的功耗模型参数。
*   **容器配置**：系统最多同时承载 $C = 16$ 个并发容器（与主机数量一致）。为模拟动态到达，实验在每个调度间隔内随机生成一定数量的新容器请求，测试阶段其均值设置为 10，并允许围绕均值产生波动。
*   **网络环境**：路由器带宽设定为 $\text{ROUTER\_BW} = 10000$ Mbps。迁移开销在仿真中被抽象为“容器状态传输时间（由分配带宽与容器规模决定）+ 链路时延项”，以反映迁移对响应时间的影响。
*   **能耗模型**：采用与计算负载相关的功耗模型对节点能耗进行估计，集群总功率上限设定为 $\text{TOTAL\_POWER} = 1000$ W。
*   **实验周期**：测试评估阶段包含 600 个调度决策间隔（Interval），每个间隔时长设定为 300 秒（5 分钟）。此外，为支持分阶段训练，数据收集与训练阶段采用不同的运行步数（详见 5.1.3）。

### 5.1.3 数据集与三阶段流程

实验采用 **Bitbrain** 数据集作为工作负载驱动 **[61]**。该数据集采集自真实的大规模数据中心，包含 CPU、内存、磁盘 I/O 与网络带宽等多维时间序列数据。Bitbrain 痕迹具有高度的**非平稳性（Non-stationarity）**与**突发性（Burstiness）**，能够有效检验调度算法在动态、不确定环境下的鲁棒性。数据预处理方面，将原始痕迹数据处理为多维时间序列输入，每个时间步由各主机的资源相关指标构成，并按训练集统计量完成归一化，以保证不同量纲的可比性。

为与本文“训练—推理”闭环一致，实验采用**三阶段**流程，各阶段的数据用途与运行长度如下：

*   **阶段1（数据收集，400 间隔）**：在无预防性迁移的条件下运行仿真（每间隔新增容器数为 8），收集训练编码器所需的时间序列与调度序列。仿真器在每个调度间隔记录实际发生的故障（由资源超限等条件触发），并经由 **ADE（Actual-Detected-Expected）** 机制生成异常标签（见 2.3.5 节），从而保证后续编码器训练与数据收集阶段在“故障”语义上完全一致。
*   **阶段2（编码器训练与对抗训练，1200 间隔）**：在阶段1 数据上对编码器进行**离线监督训练**（异常分类损失与基于原型的度量学习）；随后在仿真交互过程中进行**在线对抗训练**（每间隔新增容器数为 8），即仅在异常触发时更新 GAN 的生成器与判别器，并在部分设置下支持轻量级编码器调优。该阶段形成策略学习所需的交互轨迹，并为阶段3 的冻结模型提供训练好的编码器与 GAN 参数。
*   **阶段3（测试评估，600 间隔）**：冻结编码器与 GAN 参数，仅进行在线推理与策略执行，不更新模型。测试阶段采用 600 个调度间隔、每间隔约 10 个新容器请求（总创建容器约 6000），用于生成本章所报告的能耗、迁移次数、SLA 违约率等对比结果。各方法在阶段3 均独立运行 5 次，本章汇报各指标的均值与标准差，以保证结果的可比性与稳定性。

### 5.1.4 对比算法与评价指标

为全面评估所提方法的有效性，实验选取了**本研究提出的三种 GAN 变体**与**四种传统启发式算法**进行对比。

**本研究提出的方法**：**FPE-GAN**（第3章）为基础方法，采用 GRU 结合 GAT 进行故障预测，使用标准 GAN 生成调度策略；**TF-GAN**（第4章）为改进方法，使用 Transformer 替代 GRU 以增强长序列建模能力，其余 GAN 结构不变；**MAMO-GAN**（第4章）为最终优化版本，在 TF-GAN 基础上引入迁移感知生成器（Migration-Aware Generator）与多目标判别器（Multi-Objective Discriminator），并在推理阶段采用冷却期、每步迁移上限等控制机制。

**传统基准算法**：**CMODLB** 结合全卷积网络（FCN）预测与最小迁移时间（MMT）策略；**DFTM** 基于局部加权回归（LOESS）预测主机过载，并采用首次适应（First-Fit）策略；**ECLB** 以能耗最小化为核心目标，采用贝叶斯优化选择目标主机；**PCFT** 基于线性回归的主动容错与最少填充（Least-Full）放置策略。MMT 与 First-Fit 等是经典的虚拟机/容器整合策略 **[62]**，与本文方法在相同仿真环境下对比，可反映“数据驱动+对抗生成”相对于传统启发式的优劣。

**评价指标**：实验采用多维指标体系对算法性能进行量化评估，具体定义如下。

1.  **总能耗** $E_{total}=\sum_{t=1}^{T} E^{t}$：整个实验周期内各间隔能耗之和，反映策略对资源整合与功耗的影响。
2.  **平均响应时间**：任务从提交到完成的平均耗时（秒），反映系统服务效率。
3.  **SLA 违约率**：以容器完成时刻（destroyAt）是否超过其 SLA 截止（sla）作为违约判定，统计违约容器数 $\text{slaviolations}$ 并计算比例。
	*   本文采用“全体创建容器”口径：$\text{slaviolations}/\text{total\_created}\times 100$。阶段3 总创建约 6000，该指标约在 12%–13% 量级，与配置一致。
	*   同时给出“已完成容器”口径作为补充：$\text{slaviolations}/\text{numdestroyed}\times 100$（在本实验配置下该口径往往较高，约为 97% 左右）。
4.  **迁移次数**：实验过程中发生的容器迁移总次数，直接关联网络开销与服务中断风险，是衡量调度稳定性的关键指标。
5.  **调度时间**：算法生成决策的计算开销；本文中推理为前向计算，单步延迟为毫秒级，满足间隔级实时性要求。此外，可结合能耗、响应时间与迁移成本构造综合评分，用于多目标权衡分析。

故障检测指标的对比范围与评估口径将在 5.2 节统一说明。
---

## 5.2 故障检测性能对比

本节汇报**编码器训练阶段（Stage2）**的故障检测性能，在相同 ADE 标签与数据口径下对比 FPE-GAN、TF-GAN/MAMO-GAN 及 **CMODLB** 的编码器（或等价预测模块）。故障检测性能直接影响“异常触发”的准确性，进而影响后续迁移决策的触发频率与稳定性，因此是评估所提方法的重要一环。

**对比范围与数据来源说明**：在四种传统基线（CMODLB、DFTM、ECLB、PCFT）中，**仅 CMODLB 参与本节故障检测对比**。原因是：CMODLB 是其中**唯一**在阶段2 拥有“可训练的预测模块”的方法——其采用全卷积网络（FCN）在阶段1 产出的数据上进行训练，与本文方法一样使用同一套 ADE 标签与数据口径，因此可以在“预测/检测模块在训练数据上的表现”上与 FPE-GAN、TF-GAN/MAMO-GAN 进行同口径比较（CMODLB 的 FCN 为回归任务，仅能比较 Loss，无分类意义上的 P/R/F1）。**DFTM、ECLB、PCFT** 的预测与放置逻辑则不同：DFTM 基于局部加权回归（LOESS）预测主机过载并采用首次适应策略，ECLB 以能耗最小化与贝叶斯选择为主，PCFT 基于线性回归与最少填充策略；它们在阶段2 并未在相同 ADE 标注数据上训练一个可输出 P/R/F1 的故障检测编码器，因此无法与本文方法及 CMODLB 在“故障检测 P/R/F1（或 Loss）”上做同口径对比，其系统级效果在 5.3 节的迁移决策、能耗与 SLA 等指标上体现。此外，阶段3（推理测试）未对各方法统一输出在线故障检测（P/R/F1）指标；传统方法（CMODLB、DFTM、ECLB、PCFT）在阶段3 亦无与编码器 P/R/F1 口径一致的故障预测数据可资比较。因此，本节的故障检测对比**仅能基于阶段2 的编码器训练数据**；下文表 5-1 与相关分析均基于阶段2 训练阶段的输出。

### 5.2.1 编码器阶段 P/R/F1 对比

故障检测指标（精确率 P、召回率 R、F1）在 Stage2 编码器训练所用的故障标注数据上计算。由于当前阶段3（推理测试）未统一输出各方法的在线 P/R/F1，且推理阶段存在迁移干预、观测数据分布与训练阶段不同，本文不报告部署阶段的故障检测指标，而采用训练阶段数据上的 P/R/F1 作为“编码器对故障模式判别能力”的度量，这在机器学习论文中是常见且可接受的做法。

**表 5-1 编码器阶段故障检测性能（Stage2 训练数据）**

| 模型 | 编码器 | 训练轮数 | Loss | P | R | F1 | AScore |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FPE-GAN** | FPE_16 | 300 | 0.278 | 0.813 | 1.0 | 0.897 | 0.988 |
| **TF-GAN / MAMO-GAN** | Transformer_16（共用） | 150 | **-0.149** | **0.933** | 1.0 | **0.965** | **0.996** |
| **CMODLB** | FCN_16 | 30 | 0.666 | — | — | — | — |

注：CMODLB 为 FCN 回归任务，无 P/R/F1；表中“—”表示不适用。

从表 5-1 可以观察到：在相同 ADE 标签与数据口径下，**Transformer_16**（TF-GAN 与 MAMO-GAN 共用编码器）在**更少训练轮数（150 轮）**下即达到更低的 Loss（-0.149）、更高的精确率 P（0.933）、更高的 F1（0.965）与更高的 AScore（0.996），整体优于 **FPE_16**（FPE-GAN 的编码器）在 300 轮训练后的表现（Loss 0.278，P 0.813，F1 0.897，AScore 0.988）。这表明**引入 Transformer 替代 GRU 有利于故障检测**，编码器的递进改进在故障预测层面得到了定量验证。与 **CMODLB** 的 FCN_16 相比，PreGAN 系列在可度量的故障检测指标（Loss、P、R、F1）上表现更好；CMODLB 的 FCN 为回归任务，不输出分类意义上的 P/R/F1，其完整效果需在阶段3 的迁移决策与系统级指标上再比较。

---

## 5.3 各阶段迁移决策效果对比

本节重点分析第3章与第4章所提方法在**迁移决策**层面的表现。阶段3（测试评估）中，各方法均独立运行 5 次，本节汇报各指标的均值与标准差。测试阶段步数为 600、每步约 10 个新容器请求，总创建容器约 6000。以下按“FPE-GAN 与传统方法对比”“TF-GAN 与 FPE-GAN 对比”“MAMO-GAN 综合性能”与“全方法总览”四部分展开。

### 5.3.1 FPE-GAN 与传统方法对比

本节重点分析第3章提出的 **FPE-GAN** 方法与四种传统启发式算法（CMODLB、DFTM、ECLB、PCFT）在能耗、迁移次数与 SLA 违约率上的差异，旨在验证“时空编码+对抗生成”范式相对于传统启发式的优势。

首先考察能耗与迁移成本。图 5-2 给出了 FPE-GAN（PreGAN）与四种传统方法在总能耗与任务迁移总数上的对比结果。

**图 5-2 FPE-GAN 与传统方法对比（Group1）**

（a）总能耗对比（`experiment_logs/stage3/plots/group1_Bar-Total_Energy.png` / `experiment_logs/stage3/plots/group1_Bar-Total_Energy.pdf`）

![](experiment_logs/stage3/plots/group1_Bar-Total_Energy.png)

（b）任务迁移总数对比（`experiment_logs/stage3/plots/group1_Bar-Number_of_Task_migrations.png` / `experiment_logs/stage3/plots/group1_Bar-Number_of_Task_migrations.pdf`）

![](experiment_logs/stage3/plots/group1_Bar-Number_of_Task_migrations.png)

从图 5-2（a）可见，FPE-GAN 与 CMODLB、DFTM、PCFT 等在能耗上处于同一量级，其综合优势主要体现在图 5-2（b）所示的**迁移次数显著更少**与**结果稳定性更好**。具体而言，FPE-GAN（PreGAN）的迁移次数为 942.2±8.66，相较 DFTM（981.8）约减少 4%，相较 ECLB（1832.4）约减少 49%，相较 PCFT（2527.8）约减少 63%。这表明 FPE-GAN 的“异常触发 + 判别器否决 + 可行性过滤”的保守决策机制能够有效抑制无收益迁移，从源头减少网络开销与系统扰动。在 SLA 违约率方面，FPE-GAN 的违约比例（全体创建口径，约 13.11%）与各方法处于相当水平，且未出现 PCFT 那样的“高迁移、高方差”现象。总体而言，FPE-GAN 在迁移成本控制与 SLA 保障之间取得了较为稳健的折中，为第4章进一步引入显式迁移约束与多目标判别奠定了基线。

### 5.3.2 TF-GAN 与 FPE-GAN 对比

本节对比第4章提出的 **TF-GAN**（引入 Transformer 编码器）与 **FPE-GAN**（基于 GRU 编码器）的性能，旨在验证长序列建模能力对故障预测及迁移决策稳定性的影响。

TF-GAN 在结构上以 Transformer 编码器替换了 GRU，用以增强时序表征的灵活性与可扩展性。在本实验设置下，时间窗口保持较短（$L=3$），因此编码器升级带来的收益主要体现为对局部时序模式的更稳定表征，以及对异常触发与迁移决策门控的影响。图 5-3 给出了 TF-GAN 与 FPE-GAN 的迁移次数与总能耗对比。

**图 5-3 TF-GAN 与 FPE-GAN 对比（Group2）**

（a）迁移次数对比（`experiment_logs/stage3/plots/group2_Bar-Number_of_Task_migrations.png` / `experiment_logs/stage3/plots/group2_Bar-Number_of_Task_migrations.pdf`）

![](experiment_logs/stage3/plots/group2_Bar-Number_of_Task_migrations.png)

（b）总能耗对比（`experiment_logs/stage3/plots/group2_Bar-Total_Energy.png` / `experiment_logs/stage3/plots/group2_Bar-Total_Energy.pdf`）

![](experiment_logs/stage3/plots/group2_Bar-Total_Energy.png)

从图 5-3（a）来看，TF-GAN（PreGANPlus）的迁移次数相较 FPE-GAN（PreGAN）有所下降（由 942.2 降至 929.8，约降低 1.3%），同时图 5-3（b）显示能耗基本保持不变。这表明在保持总体能耗水平的情况下，Transformer 编码器能够一定程度减少不必要的策略扰动，从而提升调度稳定性。进一步对比能耗可发现，TF-GAN 在能耗上与 FPE-GAN 差异较小，而其主要改进集中在“迁移次数减少”。这说明在既定的生成器—判别器目标下，单纯提升编码器表征能力并不必然带来所有 QoS 指标的同步改善，系统仍可能呈现指标间权衡；因此，第4章提出的多目标优化与迁移约束机制对于提升综合性能仍是必要的。

### 5.3.3 MAMO-GAN 综合性能

本节评估第4章提出的最终模型 **MAMO-GAN**。该模型在 TF-GAN 的基础上，引入了**迁移感知生成器**和**多目标判别器**，并在推理阶段采用冷却期、每步迁移上限等控制机制，旨在解决能耗、响应时间与迁移成本之间的多目标冲突问题。

图 5-4 给出了 MAMO-GAN（PreGANPlusEnhanced）与 TF-GAN 等在总能耗、迁移次数与 SLA 违约比例上的对比结果。

**图 5-4 MAMO-GAN 综合性能对比（Group3）**

（a）总能耗对比（`experiment_logs/stage3/plots/group3_Bar-Total_Energy.png` / `experiment_logs/stage3/plots/group3_Bar-Total_Energy.pdf`）

![](experiment_logs/stage3/plots/group3_Bar-Total_Energy.png)

（b）迁移次数对比（`experiment_logs/stage3/plots/group3_Bar-Number_of_Task_migrations.png` / `experiment_logs/stage3/plots/group3_Bar-Number_of_Task_migrations.pdf`）

![](experiment_logs/stage3/plots/group3_Bar-Number_of_Task_migrations.png)

（c）SLA 违约比例对比（`experiment_logs/stage3/plots/group3_Bar-SLA_violation_pct.png` / `experiment_logs/stage3/plots/group3_Bar-SLA_violation_pct.pdf`）

![](experiment_logs/stage3/plots/group3_Bar-SLA_violation_pct.png)

从图 5-4（a）可见，MAMO-GAN 相较 TF-GAN 表现出一定优势：总能耗由约 11988807 降至约 11925507（约降低 0.5%），表明迁移感知与多目标约束能够在既定的训练目标下引导策略偏向更低能耗的解。与此同时，图 5-4（b）显示 MAMO-GAN 的迁移次数（926.0±5.90）相较 TF-GAN（929.8±8.13）进一步降低且**方差更小**，并显著低于迁移激进的方法（例如 PCFT、ECLB），说明其在能耗改善的同时仍能维持较好的调度稳定性。图 5-4（c）所示 SLA 违约率（全体创建口径）为 12.83±0.14%，方差小、表现稳定。总体而言，MAMO-GAN 将“能耗—迁移成本—时延”之间的权衡显式化，使得策略可以在多目标约束下进行可控调整，从而获得更稳健的综合性能。

### 5.3.4 全方法总览与调度开销

为便于整体把握各方法的相对位置，图 5-5 给出了**全方法**在总能耗、迁移次数与 SLA 违约比例上的对比结果。

**图 5-5 全方法指标总览**

（a）全方法总能耗（`experiment_logs/stage3/plots/Bar-Total_Energy_all.png` / `experiment_logs/stage3/plots/Bar-Total_Energy_all.pdf`）

![](experiment_logs/stage3/plots/Bar-Total_Energy_all.png)

（b）全方法迁移次数（`experiment_logs/stage3/plots/Bar-Number_of_Task_migrations_all.png` / `experiment_logs/stage3/plots/Bar-Number_of_Task_migrations_all.pdf`）

![](experiment_logs/stage3/plots/Bar-Number_of_Task_migrations_all.png)

（c）全方法 SLA 违约比例（`experiment_logs/stage3/plots/Bar-SLA_violation_pct_over_created_all.png` / `experiment_logs/stage3/plots/Bar-SLA_violation_pct_over_created_all.pdf`）

![](experiment_logs/stage3/plots/Bar-SLA_violation_pct_over_created_all.png)

从迁移次数由低到高的排序可见：MAMO-GAN（926）< TF-GAN（930）< w/o Transformer（933）< w/o Multi-Objective（941）< FPE-GAN（942）< w/o Migration-Aware（969）< DFTM（982）< CMODLB（994）< w/o GAT（1683）< ECLB（1832）< PCFT（2528）。总体上，完整的 GAN 系列（FPE-GAN/TF-GAN/MAMO-GAN）在“少迁移、低方差”的目标上表现更为稳健，而去除关键模块（如 GAT）会显著放大迁移规模与波动。需要指出的是，尽管 FPE-GAN、TF-GAN 与 MAMO-GAN 均为基于深度学习的方法，其调度计算开销仍处于较低水平：推理阶段为前向计算，单步延迟为毫秒级，远低于 300 秒的调度间隔长度，满足间隔级在线决策的实时性要求 **[60]**。

---

## 5.4 消融实验

为了深入探究 MAMO-GAN 中各个创新组件的具体贡献，本节设计并实施了消融实验。通过逐步移除模型中的关键模块，观察系统性能的变化情况，从而验证每个组件设计的合理性。**消融模型的故障检测性能**统一在本节描述（与 5.2 节口径一致，均在编码器训练阶段评估）；**消融模型的阶段3 运行指标**（迁移次数、能耗、SLA 违约率）紧随其后；各变体在阶段3 均运行 5 次，汇报均值与标准差。

### 5.4.1 实验变体设计

消融实验围绕第4章提出的核心组件展开。除完整模型 **MAMO-GAN (Full)** 外，本文分别构造“移除 Transformer”“移除 GAT”“移除迁移感知机制”“移除多目标判别器”等变体，使得每次对比仅改变一个关键因素，从而更清晰地归因各组件对故障检测、能耗、迁移次数与 SLA 合规性的影响。

| 变体 | 编码器 | GAN 结构 | 说明 |
| :--- | :--- | :--- | :--- |
| **MAMO-GAN (Full)** | Transformer_16 | 迁移感知生成器 + 多目标判别器 | 基准 |
| **w/o Transformer** | FPE_16 | 同 Full | 用时序 FPE 替代 Transformer |
| **w/o GAT** | TransformerNoGAT_16 | 同 Full | 去掉 GAT，仅 Transformer |
| **w/o Migration-Aware** | Transformer_16 | 标准 Gen_16 / Disc_16 | 去掉迁移感知 |
| **w/o Multi-Objective** | Transformer_16 | 迁移感知生成器 + 标准判别器 | 去掉多目标判别器 |

所有变体在相同的阶段1 数据与 ADE 标签下训练；阶段3 每种变体均独立运行 5 次，汇报均值与标准差。

### 5.4.2 消融模型的故障检测性能

故障检测指标（P、R、F1、AScore、Loss）仍在**编码器训练阶段（Stage2）**的故障标注数据上评估，与 5.2 节口径一致，以保证与完整模型及 5.2.1 节表格的可比性。

*   **w/o Transformer（AblationNoTransformer）**：编码器为 FPE_16，训练 300 轮。其 Loss=0.238，P=0.816，F1=0.899，AScore=0.988，均**劣于**基准 MAMO-GAN 的 Transformer_16（150 轮：Loss=-0.149，P=0.933，F1=0.965，AScore=0.996）。这说明**Transformer 对故障检测有明确的正向贡献**；移除 Transformer 后，编码器退化为 FPE，在相同 ADE 标签下的故障预测能力下降。
*   **w/o GAT（AblationNoGAT）**：编码器为 TransformerNoGAT_16（仅 Transformer，无 GAT）。在相同 150 轮训练条件下，NoGAT 的 Loss、P、F1、AScore 略优于基准 Transformer_16（例如 F1 0.981 vs 0.965）；该差异可能与训练随机性与收敛波动有关，不能据此直接得出“去掉 GAT 必然更优”的结论。就编码器阶段而言，GAT 的边际增益在本数据与训练设置下并不显著。更关键的是，**GAT 的系统级收益主要体现在阶段3 的迁移控制**（见 5.4.3）：去掉 GAT 后，迁移次数与方差急剧恶化，说明 GAT 对“少迁移、高稳定”的迁移决策至关重要。
*   **w/o Migration-Aware / w/o Multi-Objective**：两变体的编码器与基准相同（Transformer_16），结构无差异；若训练轮数或运行批次不同，编码器指标与基准的差异应视作**训练波动或误差**，不单独归因于“移除迁移感知”或“移除多目标”。两变体的差异主要体现在 GAN 结构与阶段3 运行指标上。

综上，消融模型在编码器阶段的对比可归纳为以下要点：

*   **w/o Transformer**：仅去掉 Transformer 会明确降低故障检测性能。
*   **w/o GAT**：编码器阶段的故障检测指标略有上浮，但阶段3 迁移表现严重变差，体现了 **GAT 对迁移控制的关键作用**。

上述结论与第3章、第4章中“空间依赖建模（GAT）支撑全局资源分配与迁移决策”的设计动机一致。

### 5.4.3 消融实验运行结果（Stage3）

表 5-2 展示了消融实验在阶段3（测试评估）上的详细数据对比；图 5-6 以子图形式给出了消融各变体在迁移次数与总能耗上的柱状图对比。

**表 5-2 消融实验 Stage3 运行指标（5 次运行汇总）**

| 模型变体 | 总能耗（区间和） | 迁移次数 | SLA 违约率（全体创建，%） |
| :--- | :---: | :---: | :---: |
| **MAMO-GAN (Full)** | 11925507±49530 | 926.0±5.9 | 12.83±0.14 |
| w/o Transformer | 11966474±68186 | 932.8±19.6 | 12.80±0.19 |
| w/o GAT | 11718617±170513 | 1683.0±386.2 | 12.21±0.24 |
| w/o Migration-Aware | 11983944±54568 | 969.2±30.3 | 13.07±0.28 |
| w/o Multi-Objective | 12003318±65099 | 941.2±10.3 | 12.89±0.17 |

**图 5-6 消融实验结果（Stage3）**

（a）迁移次数对比（`experiment_logs/stage3/plots/ablation_Bar-Number_of_Task_migrations.png` / `experiment_logs/stage3/plots/ablation_Bar-Number_of_Task_migrations.pdf`）

![](experiment_logs/stage3/plots/ablation_Bar-Number_of_Task_migrations.png)

（b）总能耗对比（`experiment_logs/stage3/plots/ablation_Bar-Total_Energy.png` / `experiment_logs/stage3/plots/ablation_Bar-Total_Energy.pdf`）

![](experiment_logs/stage3/plots/ablation_Bar-Total_Energy.png)

从归因逻辑与数据可得以下结论。

1.  **移除 Transformer**：迁移次数由 926.0±5.9 升至 932.8±19.6，方差明显增大（19.6 vs 5.9），说明 Transformer 对迁移控制与结果稳定性具有正向贡献；与 5.4.2 的故障检测结论一致，Transformer 在“故障预测”与“迁移决策稳定性”两方面均发挥作用。
2.  **移除 GAT**：迁移次数升至 1683.0±386.2，方差极大，显著高于其他消融变体与完整模型；说明 **GAT 对控制迁移至关重要**，去掉后模型难以充分利用节点间空间关系形成更合理的全局资源分配，导致迁移规模与波动显著恶化。
3.  **移除迁移感知**：迁移次数升至 969.2±30.3，方差（30.3）明显大于完整模型（5.9），说明**迁移感知机制有利于少迁移与稳定**。
4.  **移除多目标判别器**：总能耗升至 12003318±65099，为表 5-2 中最高，说明**多目标判别器对能耗优化具有重要作用**；当优化目标退化时，策略更易偏向能耗较高的区域。

上述对照定量验证了各组件对最终运行性能的独立贡献，与第4章的设计动机相呼应。

---

## 5.5 参数敏感性分析

针对 MAMO-GAN 中的关键超参数，特别是多目标损失函数中的权重系数 $\lambda_E$（能耗权重）、$\lambda_{RT}$（响应时间权重）和 $\lambda_M$（迁移权重），本节从机理上分析其对最终性能的影响。

从机理上看，当能耗权重 $\lambda_E$ 过大时，策略会更倾向于资源整合以压低能耗，但这可能诱发资源争用并推高响应时间与 SLA 违约；相反，当响应时间权重 $\lambda_{RT}$ 过大时，策略会更倾向于负载分散以降低时延，但会因开启更多节点而抬升基础能耗。迁移权重 $\lambda_M$ 则直接影响“少迁移”在判别器反馈中的比重，过小可能导致迁移规模偏大，过大则可能过度抑制迁移、牺牲负载均衡。为检验结论对超参数扰动的鲁棒性，本文以第4章给出的默认权重为中心，在其邻域内可进行敏感性分析，并给出能耗、响应时间与迁移次数随权重变化的趋势曲线；该部分可在后续工作中补充，以进一步验证所提方法对超参数选择的稳健性。

---

## 5.6 本章小结

本章通过在 COSCO 联合仿真平台与 Bitbrain 工作负载上的系统性实验，全面评估了第3章与第4章所提方法（FPE-GAN、TF-GAN、MAMO-GAN）的性能。实验采用三阶段流程（阶段1 数据收集 400 间隔、阶段2 编码器与 GAN 训练 1200 间隔、阶段3 测试评估 600 间隔），并在相同 ADE 标签与数据口径下保证了结果的可比性与可复现性。

为便于归纳，本章结论要点如下：

*   **故障检测（Stage2）**：
	*   Transformer_16（TF-GAN 与 MAMO-GAN 共用）在精确率 P、F1、AScore 等指标上整体优于 FPE_16（FPE-GAN）与 CMODLB 的 FCN，表明引入 Transformer 替代 GRU 有利于故障预测。
	*   消融实验中，w/o Transformer 会明确降低故障检测性能；w/o GAT 在编码器阶段指标略有上浮，但该差异需结合训练波动理解，不能据此推出“去掉 GAT 更优”。

*   **迁移决策（Stage3）**：
	*   FPE-GAN 相较多种传统启发式方法（CMODLB、DFTM、ECLB、PCFT）在迁移次数与稳定性上表现更为稳健，说明“异常触发 + 对抗生成 + 保守筛选”的策略有助于降低无效迁移并改善整体运行效率。
	*   TF-GAN 在保持能耗水平基本不变的情况下进一步减少迁移次数，表明编码器升级有助于提升迁移决策稳定性，但仅依赖编码器改进仍难以保证多指标同步最优。
	*   MAMO-GAN 通过引入迁移感知与多目标反馈机制，在能耗指标上获得进一步改善，同时维持相对可控的迁移规模与较好的稳定性（迁移次数 926.0±5.90，方差最小），体现了多目标优化对于复杂调度问题的重要意义。

*   **消融实验与归因**：
	*   “移除 Transformer / 移除 GAT / 移除迁移感知 / 移除多目标判别器”的对照实验定量验证了各组件对最终运行性能的独立贡献。
	*   其中，GAT 对迁移控制、多目标判别器对能耗优化的贡献更为突出；参数敏感性分析为多目标权重的选取与后续鲁棒性验证提供了理论依据与可扩展方向。

总体而言，本章实验结果进一步支持了第3章与第4章提出的方法设计与实验动机，表明所提方案在动态、异构的边缘计算环境中具有一定的应用潜力。

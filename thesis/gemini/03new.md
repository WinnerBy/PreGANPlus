# 第3章 基于时空特征编码和生成对抗网络的主动迁移方法（FPE-GAN）

## 3.1 FPE-GAN 总体框架

边缘计算环境的动态性与不确定性，使得传统的基于阈值或简单回归的容错方法难以满足高可靠性（High Reliability）和低延迟（Low Latency）的服务等级协议（SLA）要求。针对这一难题，本章提出一种基于时空特征编码和生成对抗网络的**抢占式迁移**方法（Fault Prediction and Embedding based GAN, FPE-GAN）。该方法不仅关注单一节点的负载状态，更从全局视角捕捉故障在网络拓扑中的传播规律，实现了从"被动修复"到"主动防御"的范式转变。FPE-GAN 通过在故障发生前的预测窗口内提前执行迁移，能够显著降低服务中断时间，提高系统可靠性。

### 3.1.1 设计理念与核心思路

FPE-GAN 的设计遵循“感知-编码-生成-决策”的闭环控制逻辑。与“通过重构误差做无监督异常检测”的思路不同，本文实现的 FPE-GAN 采用**监督式故障预测编码器（FPE）**输出异常概率，并以其产生的**原型向量（prototype）**作为条件输入，驱动生成器产生候选调度；随后判别器对“原始调度 vs 新调度”进行二分类判断，从而实现“仅在需要且确有收益时才迁移”的保守主动容错策略。

总体架构如图 3-1 所示（此处插入 FPE-GAN 整体架构图），包含四个紧密耦合的模块：
1.  **多维状态感知模块**：负责从异构边缘节点采集 CPU、内存、I/O 及网络带宽等多维时序数据，并进行清洗与标准化处理。
2.  **时空特征编码器（Spatio-Temporal Encoder）**：这是本章的核心组件之一，由 GRU 和 GAT 构成，负责将非结构化的时序数据和图结构数据映射为高维稠密的潜在特征向量。
3.  **对抗式决策生成器（Adversarial Decision Generator）**：基于 GAN 架构，生成器试图根据潜在特征生成符合系统约束的迁移概率分布，而判别器则通过对抗训练不断提升生成策略的有效性与鲁棒性。
4.  **在线迁移执行模块**：将生成的调度矩阵转化为离散的迁移指令，并在实验平台中完成迁移执行。在仿真评估中，迁移过程被抽象为资源占用与通信开销对系统性能的影响；在工程系统中，该过程可进一步映射为检查点、状态传输与恢复等典型迁移步骤。

### 3.1.2 适用场景与假设

本章所提方法主要针对以下边缘计算典型故障场景：
*   **资源耗尽型故障**：由于突发流量导致 CPU 或内存利用率达到 100%，引发服务死锁。
*   **性能降级型故障**：虽然服务未完全中断，但响应时间显著超过 SLA 阈值。

假设边缘网络拓扑在短时间内保持相对稳定，且历史负载数据能够反映一定的周期性规律。

## 3.2 基于时空图网络的故障特征编码器

边缘节点的运行状态数据具有显著的**时空相关性（Spatio-Temporal Correlation）**。时间维度上，负载往往呈现出日周期性或小时级波动；空间维度上，物理邻近或逻辑互联的节点间存在负载传导效应。为了精确提取这些特征，本节设计了深度的时空特征编码器。

### 3.2.1 输入数据建模与预处理

定义边缘计算网络为无向图 $G = (\mathcal{V}, \mathcal{E})$，其中 $N = |\mathcal{V}|$ 为节点总数。
对于每个节点 $v_i$，在时间步 $t$ 采集特征向量 $x_{i,t} \in \mathbb{R}^F$，其中 $F$ 为特征维度（包括 CPU 利用率、内存占用率、出入带宽、磁盘 I/O 等）。

实验数据以时间序列形式组织为二维矩阵（每行对应一个时间点的全局观测）：

- 单步观测：$x_t \in \mathbb{R}^{48}$，由 \(16\) 个主机的 \(3\) 个指标拼接得到（\(16\times 3=48\)）；
- 调度状态：$s_t \in \mathbb{R}^{16\times 16}$，表示容器到主机的分配矩阵。

为消除量纲影响并与源码一致，本文采用**按维最大值归一化**（训练集最大值为基准）：
\[
\hat{x}_t^{(k)}=\frac{x_t^{(k)}}{\max(\mathcal{D}_{train}^{(k)})+\varepsilon},\ \ \varepsilon=10^{-8} \tag{3.1}
\]

编码器使用长度为 \(L=3\) 的滑动窗口，将连续 3 个时间点组成输入 \(t\in\mathbb{R}^{3\times 48}\)。

### 3.2.2 时序依赖性建模：深度 GRU 网络

在时间维度，采用门控循环单元（GRU）来捕捉**短期**负载变化趋势。由于本文窗口长度仅为 \(L=3\)，该模块重点建模“近几个 interval 的变化”，以满足在线推理的低开销需求。相比 LSTM，GRU 结构更简洁、参数更少，适合边缘侧快速前向计算。

对于节点 $i$，输入序列为 $X_i = (\hat{x}_{i,1}, \hat{x}_{i,2}, ..., \hat{x}_{i,L})$。GRU 层的计算过程如下：

1.  **重置门（Reset Gate）** $r_t$：决定丢弃多少历史信息，用于捕捉突发性的负载波动。
    $$
    r_{i,t} = \sigma(W_r \hat{x}_{i,t} + U_r h_{i,t-1} + b_r) \tag{3.2}
    $$
2.  **更新门（Update Gate）** $z_t$：决定保留多少历史状态，用于捕捉周期性的长期趋势。
    $$
    z_{i,t} = \sigma(W_z \hat{x}_{i,t} + U_z h_{i,t-1} + b_z) \tag{3.3}
    $$
3.  **候选隐藏状态** $\tilde{h}_{t}$：
    $$
    \tilde{h}_{i,t} = \tanh(W_h \hat{x}_{i,t} + U_h (r_{i,t} \odot h_{i,t-1}) + b_h) \tag{3.4}
    $$
4.  **最终隐藏状态** $h_{t}$：
    $$
    h_{i,t} = (1 - z_{i,t}) \odot h_{i,t-1} + z_{i,t} \odot \tilde{h}_{i,t} \tag{3.5}
    $$

在本文实验设置下，GRU 的输入维度为 48、隐藏维度为 3，得到每个时间步的隐状态序列 \(\in\mathbb{R}^{3\times 3}\)。

### 3.2.3 空间依赖性建模：多头图注意力网络

仅依靠时序特征无法感知网络拥塞或级联故障。为此，本研究引入图注意力网络（GAT），利用注意力机制动态聚合邻居节点的信息。为了增强模型的表达能力，采用**多头注意力机制（Multi-Head Attention）**。

假设使用 $K$ 个独立的注意力头。对于第 $k$ 个头，节点 $i$ 与其邻居 $j \in \mathcal{N}_i$ 之间的注意力系数 $e_{ij}^{(k)}$ 计算如下：

$$
e_{ij}^{(k)} = \text{LeakyReLU}\left( (\vec{a}^{(k)})^T [W^{(k)} h_i^{temp} \, \| \, W^{(k)} h_j^{temp}] \right) \tag{3.6}
$$

其中 $W^{(k)} \in \mathbb{R}^{D_{gat} \times D_{rnn}}$ 是线性变换矩阵，$\vec{a}^{(k)}$ 是注意力权重向量，$\|$ 表示拼接操作。

利用 Softmax 函数对系数进行归一化，得到注意力权重 $\alpha_{ij}^{(k)}$：

$$
\alpha_{ij}^{(k)} = \frac{\exp(e_{ij}^{(k)})}{\sum_{l \in \mathcal{N}_i} \exp(e_{il}^{(k)})} \tag{3.7}
$$

最后，将 $K$ 个头的输出进行拼接，得到节点 $i$ 的空间特征向量 $h_i^{spat}$：

$$
h_i^{spat} = \Bigg\|_{k=1}^{K} \sigma\left( \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k)} W^{(k)} h_j^{temp} \right) \tag{3.8}
$$

在本文实验设置下，边缘网络被建模为 \(N=16\) 个主机的**全连接图**（每对主机均连边），并以每个主机的 3 个指标作为节点特征。GAT 输出每个时间步的主机表示（16 维），随后对节点维进行聚合以得到与 GRU 输出在时间维上一致的表示。另需说明：上文“多头”用于给出一般化的 GAT 公式描述；在本文具体实现与实验设置中，可等价理解为单头注意力情形。

### 3.2.4 特征融合与潜在空间映射

为了综合利用时序与空间信息，本文采用“拼接 + 注意力融合 + 线性映射”的轻量结构：

$$
z_i = \text{MLP}_{fusion}([h_i^{temp} \, \| \, h_i^{spat}]) \tag{3.9}
$$

具体而言，GRU 输出与 GAT 输出在特征维拼接得到 19 维表示（3+16），再经注意力融合，展平为 57 维向量并映射为 \(16\times 10\) 的潜在表示。该潜在表示随后分别进入异常检测分支（2 类概率输出）与原型向量分支（2 维向量输出），为后续 GAN 提供条件输入。需要说明的是，本文主要通过训练流程与决策门控机制抑制过拟合与无效迁移，而非依赖复杂的正则化结构。

## 3.3 基于生成对抗网络的迁移决策生成

在获得故障特征编码后，FPE-GAN 并非仅依赖启发式规则直接给出迁移目标，而是引入生成对抗网络（GAN）对“候选调度是否更优”进行数据驱动的学习。需要强调的是：本文实现的异常检测来自**编码器的监督式分类输出**（正常/异常），GAN 的作用是**在异常触发时生成并筛选更优调度**，而不是通过“学习正常分布—重构误差”来完成异常检测。

### 3.3.1 生成器（Generator）结构与策略生成

生成器 \(G\) 的目标是基于编码器输出的**原型向量**与当前调度矩阵，生成一个“候选新调度”。与引入随机噪声的生成式建模不同，本文实现的生成器采用**调度增量（delta）**思想：输入为 \([e\in\mathbb{R}^{16\times 2},\ s\in\mathbb{R}^{16\times 16}]\) 的拼接展平向量（维度 \(16\times 2 + 16\times 16 = 288\)），经两层 MLP 输出 256 维增量并用 Tanh 约束范围，最后形成新调度：
\[
\Delta s = 4\cdot \tanh(\text{MLP}([e\ \|\ s])) ,\quad s_{new}=s+\Delta s \tag{3.10}
\]
其中系数 4 用于放大更新幅度，使调度变化更明显。

### 3.3.2 判别器（Discriminator）结构与多维 QoS 评估

判别器 \(D\) 的输入为“原始调度 \(s\)”与“新调度 \(s_{new}\)”的拼接展平向量（维度 \(512\)），输出为二分类概率 \(\text{probs}\in\mathbb{R}^{2}\)：分别表示“原始更优 / 新调度更优”。其监督信号来自仿真评估得到的综合得分：
\[
\text{score}=0.8\cdot E + 0.2\cdot R \tag{3.11}
\]
其中 \(E\) 为能耗，\(R\) 为响应时间（延迟）。若 \(\text{score}_{new}\le \text{score}_{orig}\)，则标签为 \([0,1]\)，否则为 \([1,0]\)。

### 3.3.3 损失函数设计与正则化

本文采用二元交叉熵损失进行对抗训练，核心目标是让判别器学习“新调度是否更优”的判别边界，并让生成器倾向于生成能被判别器判为“更优”的调度：

- **判别器损失**：
\[
\mathcal{L}_D=\text{BCE}(D(s,s_{new}),\ y),\quad
y=
\begin{cases}
[0,1], & \text{score}_{new}\le \text{score}_{orig}\\
[1,0], & \text{otherwise}
\end{cases}
\tag{3.12}
\]

- **生成器损失**（鼓励“新调度更优”）：
\[
\mathcal{L}_G=\text{BCE}(D(s,s_{new}),\ [0,1]) \tag{3.13}
\]

需要说明的是：资源可行性（如容量约束）在当前实现中主要由**执行侧过滤**（例如放置可行性检查）保障，而非通过显式正则项写入 GAN 损失。

### 3.3.4 训练算法流程

FPE-GAN 的训练过程采用交替优化策略。详细步骤如算法 3-1 所示。

**Algorithm 3-1: FPE-GAN Training Procedure**
```text
Input: Historical time-series dataset and schedule dataset, epochs=50, online training steps=1200
Output: Trained Encoder (FPE), Generator (Gen), Discriminator (Disc)

// Stage-2: Offline train Encoder (supervised)
1: Normalize time_series by per-dim max of training set
2: Build windows t ∈ R^{3×48} for each step, load s ∈ R^{16×16}
3: for epoch = 1..50 do
4:     anomaly_scores, prototypes = Encoder(t, s)
5:     L_anom = CrossEntropy(anomaly_scores, anomaly_labels)   // class-imbalance reweighting
6:     L_proto = TripletLoss(prototypes, prototype_classes)   // update class prototypes
7:     Update Encoder by Adam optimizer
8: end for
9: Freeze Encoder parameters

// Stage-3: Online train GAN (only when anomaly detected)
10: for step = 1..1200 do
11:     Get current window t and current schedule s from environment
12:     anomaly_scores, prototypes = Encoder(t, s)
13:     if no anomaly then continue
14:     embedding = mask prototypes by anomaly (normal nodes → 0)
15:     s_new = Gen(embedding, s)
16:     score_new = 0.8*Energy(s_new) + 0.2*Latency(s_new)
17:     score_org = 0.8*Energy(s) + 0.2*Latency(s)
18:     y = [0,1] if score_new ≤ score_org else [1,0]
19:     Update Discriminator by BCE(D(s, s_new.detach), y)
20:     Update Generator     by BCE(D(s, s_new), [0,1])
21: end for
```

## 3.4 在线推理与迁移执行

本节描述 FPE-GAN 在**运行期（每个调度间隔）**如何调用已训练的编码器与 GAN 生成迁移决策，并在本文实验平台中完成迁移执行。需要强调的是：本文实验以仿真平台为主要评估环境，在线推理与迁移执行发生在**每 300 秒一个调度间隔**的决策环节；本文不将“面向 Kubernetes 等编排器的在线服务化部署”作为实验前提。

### 3.4.1 在线推理机制

在线推理阶段强调“**保守触发、快速决策**”。在每个调度间隔，系统先基于基线调度策略得到原始放置决策，随后由恢复决策模块调用 FPE-GAN 对当前状态进行判别：仅当**检测到潜在故障**且**判别器认为新调度更优**时，才输出迁移后的决策；否则直接返回原始决策，从源头减少不必要迁移。

在运行期，模型输入来自两类状态信息：

- **时间序列状态**：包含 CPU、内存、带宽等监控指标，取最近 \(L=3\) 个时间步组成窗口；
- **当前调度矩阵**：表示容器（或任务）到主机的分配关系。

**Algorithm 3-2: FPE-GAN Online Inference and Proactive Migration (Interval-level)**
```text
Input: recent time-series window, current schedule matrix, baseline decision
Output: refined migration decision

1: s = current schedule matrix
2: t = normalize(recent time-series) and take last L=3 window
3: anomaly_scores, prototypes = Encoder(t, s)
4: if ∀i, argmax(anomaly_scores[i]) == NORMAL then
5:     return baseline decision            // no anomaly → no migration
6: end if
7: embedding[i] = 0 if i is NORMAL else prototypes[i]   // mask normal nodes
8: s_new = Generator(embedding, s)
9: probs = Discriminator(s, s_new)
10: if probs[ORIG] > probs[NEW] then
11:     return baseline decision            // discriminator veto
12: end if
13: decision = baseline decision
14: for each task/container do
15:     target host = argmax(s_new[row])
16:     if target differs from current host:
17:         add migration action into decision
18:     end if
19: end for
20: return decision
```

### 3.4.2 接口设计与回滚策略

在本文实验框架中，FPE-GAN 以“恢复决策模块”的形式嵌入调度闭环：基线调度器给出原始放置决策后，恢复决策模块对其进行修正并输出迁移决策；随后由执行层完成迁移（仿真评估中以抽象的迁移开销模型体现，工程系统中可对应检查点—传输—恢复的迁移流程）。

为避免“误判触发大量迁移”，本文采用**算法级保守策略**（对应实现中的多重“早退出”）：

- **无异常不迁移**：若编码器对所有节点均判为正常，直接返回原始决策；
- **判别器否决不迁移**：若判别器认为新调度不优于原调度，直接返回原始决策；
- **可行性过滤**：执行层对迁移动作进行资源与约束可行性检查，不可行迁移不会被执行。

需要说明的是：本文实验以仿真为主，迁移执行在仿真器中被抽象为资源占用与传输开销的影响；框架模式虽然通过控制器/Agent 提供了 checkpoint/migrate/restore 接口，但本章不将其作为实验依赖，因此不引入“Ping 双确认/降级运行”等工程化机制描述，避免与实验事实不一致。

## 3.5 理论分析与局限性

### 3.5.1 模型复杂度与收敛性分析

*   **计算复杂度**：GRU 的时间复杂度为 $O(T \cdot N \cdot D_{rnn}^2)$，GAT 的复杂度为 $O(K \cdot (N + |\mathcal{E}|) \cdot D_{gat})$。由于边缘网络规模 $N$ 通常较小（几十到几百），且图结构稀疏（$|\mathcal{E}| \ll N^2$），整体推理延迟可控制在毫秒级别。
*   **收敛性讨论**：GAN 的训练本质上是寻找生成器与判别器之间博弈的纳什均衡点。本文采用二分类对抗目标，并通过“仅在异常触发时训练”“交替更新判别器与生成器”“固定能耗/延迟加权评分生成标签”等策略，提升训练过程的可控性与稳定性。

### 3.5.2 方法局限性

尽管 FPE-GAN 在处理时空特征和生成策略方面表现出优势，但仍存在以下局限，这也为第 4 章的改进提供了动机：
1.  **长序列遗忘问题**：尽管使用了 GRU，但在处理极长的时间序列（如跨度数天的周期性负载）时，递归结构的梯度衰减问题仍无法完全避免，导致对超长周期规律的捕捉能力不足。
2.  **单目标优化的局限**：目前的损失函数主要关注负载均衡和故障规避，未能显式地对能耗、SLA 违约率等多个相互冲突的目标进行帕累托（Pareto）优化。
3.  **迁移震荡问题**：虽然 FPE-GAN 通过对抗训练能够在一定程度上缓解迁移震荡，但由于判别器采用间接惩罚方式，且未将迁移成本显式纳入损失函数，在负载相近、网络条件波动时仍可能出现任务在两个节点间反复迁移的现象。这种间接惩罚方式不足以稳定收敛到低迁移策略，需要显式的迁移成本建模。
4.  **模式崩溃风险**：在极端不平衡的数据集下，GAN 仍可能陷入模式崩溃，即生成器倾向于输出单一的"安全"策略（如总是迁移到资源最大的节点），而忽略了全局最优解。

## 3.6 本章小结

本章详细介绍了 FPE-GAN 的设计与实现。作为本文的基础方法，FPE-GAN 构建了“时空编码 + 对抗生成”的决策框架：编码器以短窗口（\(L=3\)）提取时空特征并输出异常概率与原型向量；当异常触发时，生成器基于原型向量与当前调度生成候选新调度，判别器结合仿真评估得分（能耗/延迟加权）判断新调度是否更优，从而以保守策略减少不必要迁移。理论分析和算法设计表明，该方法能够有效应对边缘环境的动态性。然而，针对其在长期依赖建模与多目标权衡上的不足，下一章将提出基于 Transformer 和多目标优化的改进方案 TF-GAN。
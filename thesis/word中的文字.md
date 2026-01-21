第1章 绪论
1.1 研究背景及意义
在数字化转型浪潮中，物联网（IoT）技术迅猛发展，引发了一场数据革命。近年来，物联网设备及数据量呈指数级增长[1]，给传统云计算模式带来严峻挑战。传统云计算长期作为数据处理核心范式，面对物联网海量、实时性要求高的数据时，弊端尽显。在传统架构下，数据从数据源到远程数据中心处理，传输路径漫长，导致显著的数据传输延迟，无法满足延迟敏感的物联网应用需求。同时，大量数据集中传输使网络带宽压力剧增，网络拥塞频发，不仅延迟更高，运营成本也大幅提升，严重制约数据处理效率与实时性。此外，物联网系统中的安全和隐私问题也不容忽视。在此背景下，边缘计算（MEC）这一新兴计算范式诞生[2]。其核心是将计算和存储资源部署在靠近数据源的网络边缘[3]。让数据能在源头附近快速处理，极大缩短传输距离，显著降低延迟，如同构建“高速通道”，提升系统响应速度，在诸多对实时性要求高的场景中展现独特优势。
尽管边缘计算展现出巨大的潜力，但仍面临复杂严峻的挑战。在设备资源层面，边缘设备受限明显[4]。与传统数据中心强大服务器相比，其硬件配置较低，计算能力有限，难以处理大规模、高复杂度的计算任务。存储容量也相对较小，无法长时间存储大量数据。网络环境方面[5]，边缘设备所处网络复杂多变、充满不确定性。网络故障如链路中断、节点失效时有发生，信号干扰问题普遍存在，这些因素都会导致设备间通信的不稳定。这些故障一旦出现，不仅影响单个设备正常运行，还可能波及整个边缘计算系统的稳定性与可靠性。在关键边缘计算应用场景中，系统容错能力至关重要。以自动驾驶为例[6]，车辆安全行驶高度依赖边缘计算设备对传感器数据的实时处理与准确决策。若设备出现故障，车辆易失去对路况的准确判断，导致行驶决策失误，引发严重交通事故，危及人身安全和财产。在远程医疗手术中[7]，医生需实时获取并准确分析患者生理数据，任何边缘计算设备故障都可能导致数据传输异常，影响医生判断与操作，危及患者生命安全。
因此，为确保设备稳定性，需要能够在故障发生前主动预测故障，并诊断根本原因问题，以便能够采取适当的修复措施。然而现有一些主动容错方法，如，动态容错迁移（DFTM）[13]、主动协调容错（PCFT）[17]、聚类[23][25]等方法，存在覆盖故障类型不全面、异变环境中适应缓慢、预测准确性相对低等问题，无法很好的准确预测故障并及时进行迁移决策。
基于上述背景和挑战，本课题旨在深入探究边缘计算环境中设备故障、资源状态变化等复杂因素，综合考虑不同任务的实时性要求和资源需求，以在线方式构建一个具有高度智能化和自适应能力的模型。该模型将集成先进的故障检测、诊断和分类算法，能够实时监测边缘设备的运行状态，准确识别各种潜在的故障类型和原因，并根据故障的严重程度和影响范围进行合理分类。同时，结合任务的实时性要求和资源需求，模型将运用优化的决策算法，制定出适当的抢占式迁移策略，确保任务在迁移过程中的数据完整性和服务连续性，实现边缘计算系统的高效、可靠运行[8][9]。
1.2 国内外研究现状
近年来，随着边缘计算技术的发展，逐步成为物联网时代的核心支撑技术，但因其复杂的环境，一些故障检测与迁移方法，本节对主动容错、故障预测及计算迁移方法的国内外研究现状进行分析，梳理现有方法的技术路径。
1.2.1 面向分布式计算的容错方法
容错旨在开发能够承受工作负载故障，并有效管理资源以维持最佳服务质量（QoS）的系统[8]。目前，大多数容错方法主要分为两类：反应式和主动式。反应式方法是在察觉到系统故障后才采取行动，通常通过检查点、复制或重新提交受故障影响的任务。主动式方法则是提前预测故障，并通过抢占式迁移、故障感知调度等的修复步骤，避免代价高昂的故障恢复过程。鉴于反应式方案在高度动态环境中往往难以保障良好的QoS，本文重点聚焦于主动式方法。
过去几年间，为提升边缘或云平台的服务可靠性，诸多容错方法已经被提出。其中一种常见做法是提供节点冗余和网络应急措施，以防止边缘组件出现长时间停机。但随着边缘设备数量增多，为每个节点配备冗余会面临能源和成本问题[9]，这种部署并不可行。另一种方法是在不同节点上复制任务的运行实例，然而对于资源受限的边缘设备而言，这容易引发资源竞争和故障，并非理想选择[10]。还有一种机制是通过对相应容器进行检查点操作，定期保存运行任务的执行状态。容器构建了一个虚拟化层，让运行的应用程序独立于底层硬件，便于在节点故障时高效恢复任务[11]。当节点发生故障，系统能够在其他设备上转移并恢复任务，即 “抢占式迁移”[12]。不过，在提前预测故障和决定恢复节点时，定期对容器进行检查点操作，可能会给系统和网络带来较大压力。相对而言，仅对需要恢复的容器进行检查点操作，可节省对所有运行任务定期检查点的额外开销。
近年来，众多不同方法被用于确定合适的迁移决策，以增强服务可靠性。为解决数据稳定可靠性问题，Sivagami等人[13]提出一种动态容错迁移（DFTM）方法，运用整数线性规划（ILP）公式来分析工作负载流量，从而确定主机上应迁移的任务以及恢复的目标主机。Vaswani等人[14]提出一种基于阈值的自适应容错（TBAFT）方法，主动将任务从过载设备迁移到资源使用率低的设备，借助自回归移动平均（ARIMA）模型预测主机的资源利用率指标，并将其与人为设定的阈值进行比较，进而执行任务迁移以避免争用。为保障运行性能的高可用性，Mohammed等人[15]提出一种故障转移策略（SFS），它只选择那些即将违反其截止期限的任务以减少迁移开销。然而，在优化目标主机选择时，该策略对SLO截止期限的建模未考虑其他QoS参数，在异构边缘节点环境中表现欠佳。为减少迁移成本，Ray等人[16]提出一种基于偏好的故障管理（PBFM）算法，它试图使用多目标整数线性规划公式在QoS改进和迁移成本之间取得平衡。此类方法无法扩展到实时操作，不适合关键任务边缘应用。
Liu等人[17]提出主动协调容错（PCFT）方法，利用粒子群优化（PSO）来降低一组任务的总体传输开销、网络消耗和总执行时间。该方法先通过预测资源恶化来预测运行主机中产生的故障，然后使用PSO找到抢占式迁移决策的目标主机。这种方法主要侧重于减少分布式云设置中的传输开销，但往往无法提高计算节点的I/O性能。Sharif等人[18]为提高容错能力、资源利用率和性能，提出节能检查点和负载均衡（ECLB）技术，运用贝叶斯方法和神经网络将主机分为过载、欠载和正常执行三类，依据分类结果决定合适的任务迁移，以减少过载主机数量。但该模型仅考虑计算过载，未涵盖其他故障类型。
Satpathy等人[19]提出使用另一种进化搜索（CSAVMP）方案来为任务队列做出实时迁移决策。该方法通过防止不必要的迁移来优化计算设置的功耗。Wang等人[20]提出一种基于双深度Q网络的在线SFC放置方案（DDQP），一种类似的方法，Longetal等人[21]利用FTAW使用深度Q网络为分布式边缘环境中的输入工作流类型工作负载确定最佳调度决策。然而此类强化学习方案在易变环境中适应缓慢[22]。
利用深度神经网络执行模糊聚类的方法，例如，Hu等人[23]提出一种自适应加权Gath-Geva（AWGG）聚类方法，是一种无监督模型，使用堆叠稀疏自编码器检测故障以减少检测时间。应用神经网络重建系统的最后状态的方法[24]，重建误差用作当前状态异常可能性的指标。例如，TopoMAD[24]利用由长短期记忆（LSTM）和变分自编码器（VAE）组成的拓扑感知神经网络检测故障。然而，重建误差仅针对最新状态获得，这限制了它们使用不适合易变环境的反应式故障恢复策略。
Negi等人[25]使用基于聚类的多目标动态负载均衡技术（CMODLB），避免云计算节点中的资源争用。该方法使用K-means对节点进行聚类并识别过载主机，使用PSO选择任务，并使用深度学习和模糊逻辑优化选择目标主机。任务从这些主机中使用PSO方法选择，目标主机使用深度学习和模糊逻辑优化选择。然而，缓慢的PSO优化导致恢复时间较长，在高度动态的系统中作用有限[22]。
选取部分现有方法进行相关比较，如表1所示。

表 1.1 不同参数下相关工作的比较**（√表示存在相应特征）

| 工作 | 方法 | 故障检测 | 故障修复 | 准确性 | 开销 | 能耗 | SLO |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DFTM**[13] | 整数线性规划 | √ | √ | | | √ | √ |
| **TBAFT**[14] | 过载预测与任务复制 | | √ | | √ | √ | √ |
| **PBFM**[16] | 基于线性规划的负载均衡 | | √ | | √ | √ | √ |
| **PCFT**[17] | 粒子群优化 | √ | √ | | √ | √ | √ |
| **ECLB**[18] | 贝叶斯优化与神经网络 | √ | √ | √ | | √ | √ |
| **TopoMAD**[24] | 基于拓扑的自动编码器与长短期记忆网络 | √ | | √ | √ | | |
| **CMODLB**[25] | K-均值聚类与群体优化 | √ | √ | √ | | √ | √ |

1.2.2 面向边缘计算的故障预测方法
在边缘计算领域，边缘网络由众多设备构成。通过在边缘节点部署各类轻量级服务，以满足用户的实时性需求。然而边缘节点故障问题也随之而来。故障预测作为一种主动式故障管理方式，能主动预防故障，有效提升系统可靠性[26]。以下对故障预测技术的研究现状展开概述。
基于统计分析的故障预测方法，采用降维方式对复杂数据信息进行简化。在最大程度保留关键故障信息的基础上，该方法对变量关系加以简化，降低了故障预测问题的复杂程度，从而提升了分析效率。Li等人[27]提出了一种新的方向主成分分析（diPCA）方法，通过将故障缩小到指定方向或相互正交的复合方向，diPCA可以加快故障检测速度并有助于精确的故障诊断。Soraya等人[28]对基于静态主成分分析（PCA）的两种经典故障预测方法进行研究，发现预测准确率存在一定误差。这类基于主成分分析的故障预测方法，在数据呈现非高斯分布的情形下，预测结果的准确性会大幅下降。
基于数学模型的故障预测方法，Tateo等人[29]通过统计概率模型的故障预测方法，利用大量统计的故障数据进行概率计算，获得可能发生故障的概率统计，此类方法往往需要大量的故障样本。Zhang等人[30]提出一种最优隶属函数的自适应模糊神经网络故障预测模型，利用故障分布函数高度概括故障发生规律，借助神经网络强大的自学习能力有效挖掘故障数据中的潜在信息，进而用故障分布函数拟合故障数据，并通过预设多种隶属函数形成一套隶属函数，以扩大所提模型在故障预测中的适用性。这类故障预测方法能够借助数学理论知识进行建模，得到有效故障预测结果，但该方法采用离散的数据进行分析，只能做出模糊性的描述，无法推测具体故障类型。
基于人工智能的故障预测方法，Huang等人[31]通过支持向量机（SVM）对卫星传感器进行故障检测和预测，利用正常和异常在轨模拟数据进行分类训练，实现卫星传感器的故障检测与预测。Ma等人[32]提出了一种基于多特征增强和改进长短期记忆网络的变压器故障预测方法，将改进的子通道阈值深度残差收缩网络与采用多目标粒子群算法和随机游走策略优化长短期记忆（LSTM）网络相结合，实现提前识别潜在异常情况并预先生成近似的故障模型。Liu等人[33]提出了一种联合损失卷积神经网络（JL-CNN）架构，通过共享参数和部分网络并行实现故障识别和RUL检测。
1.2.3 面向边缘计算的计算迁移方法
在边缘计算的架构体系里，计算迁移策略会按照多样的需求与实际场景情况，在边缘节点、云端以及终端设备之间，灵活且合理地对计算任务进行重新分配。这种操作的核心目的在于达成资源的高效利用，削减系统延迟，最终实现系统性能的全面提升。以下是一些边缘计算下的计算迁移方法：
在现今的边缘云环境体系中，各个节点表现出鲜明的特性差异，在容量大小、运行速度快慢、响应时间长短以及能量消耗高低等多个维度，均呈现出较为明显的不同。而且，用户在云边环境中发起任务请求具有随机性，计算节点的资源又呈现出异构性[34]，上述多种因素相互叠加、相互影响，致使任务调度[35]的优化工作依旧面临诸多难题。像轮询调度算法[27]、启发式算法[36]这类传统的方法，在处理边缘云中动态的任务迁移与资源分配优化问题时，逐渐显得力不从心。这主要是由于这些传统方法在运行过程中，其搜索空间会以指数级的幅度快速增长。这种急剧增长的搜索空间，无疑给设备带来了极为沉重的计算负担。特别是在大规模的应用场景中，这些方法根本无法满足多接入边缘计算（MEC）场景对于实时性决策的严苛要求。深度强化学习（DRL）的出现为应对复杂环境下的决策难题开辟了一条全新路径。智能体通过与环境进行不断的交互，并结合深度神经网络（DNN）的高维感知能力与强化学习（RL）的决策能力，自动学习在多种不同状态下应选取的、对应最优策略的动作组合，进而实现累积奖励的最大化[37]。
近年来，DRL已经在求解边缘云环境下一些复杂决策问题领域受到大量的关注。Tuli等人[38]提出了一种基于A3C的实时调度器，用于随机边缘云环境，允许跨多个代理同时进行去中心化学习。使用R2N2架构来捕获大量的主机和任务参数以及时间模式，以提供高效的调度决策。陈娟[34]等人提出了一种结合基于最大熵框架[39]的策略梯度强化学习（Soft Actor Critic,SAC）和图卷积网络（GCN)[40]的优化算法（SAC-GCN），以最小化任务的平均能耗、响应时间、任务迁移时间和服务等级协议（SLA)[41]的违约比例。Chen[42]等人提出了一种基于深度循环Q-learning（DRQNSM）的服务迁移决策算法，使用DRL框架来指导任务迁移的时间和位置，以最小化用户延迟和系统能耗。对于任务迁移的路径选择，Gao[43]等人提出了一种基于强化学习（RL）的单用户边缘计算服务迁移系统框架，模型中的智能体寻找最优路径。


第2章 相关理论与技术


2.1 注意力机制
注意力机制（Attention Mechanism）[44][45]是一种模拟人脑在载入过量信息时聚焦核心信息的模型，本质是通过给模型不同部分分配不同权重来筛选和聚焦重要信息，在故障特征提取中作用突出，不仅能聚焦关键特征、增强特征表示、过滤噪声干扰、自适应学习特征，还能提升基于深度学习的故障诊断模型的各项性能指标，减少误报漏报，提高系统可靠性与有效性。简单描述为一种输入为查询 Query 和键值对（Key,Value），输出为注意力值的映射函数。注意力机制实质上是一个寻址过程，通过给定一个任务相关的查询向量Query，通过计算与键向量Key的注意力分布并附加在值向量Value上，从而计算注意力值，如图2.1所示。
 
图2.1 注意力机制
2.1.1 自注意力机制
自注意力机制（Self-Attention Mechanism）是一种将单个序列的不同位置关联起来以计算统一序列的表示的注意力机制，允许模型在处理序列数据时，能够动态关注序列中不同位置，以更好地捕捉序列中的长距离依赖关系，面对设备运行的时间序列等故障数据，其长序列依赖特性显著，自注意力机制可并行计算序列中任意位置间的依赖关系，不受距离约束，且具备自适应特征加权能力，增强关键故障特征，抑制干扰信息，同时拥有并行计算高效性的优势，能并行计算所有位置的注意力权重，可在更短时间内完成大规模故障数据的特征提取任务，满足高实时性的故障诊断需求。具体如图2.2所示。
针对故障序列而言，对于输入序列中的每个元素，通过学习得到的权重矩阵与输入元素的线性变换得到查询、键和值向量。对于输入序列X=(x_1,x_2,…,x_n)，权重矩阵为W_Q、W_K、W_V，对于每个元素x_i，通过公式（2-1）、（2-2）和（2-3）得到Q_i、K_i、V_i，分别表示第i个元素的查询、键和值向量。
█(Q_i=W_Q x_i  #（2-1） )
█(K_i=W_K x_i  #（2-2） )
█(V_i=W_V x_i  #（2-3） )
对于每一个元素x_i、x_j，通过查询向量和键向量的内积，为了防止内积过大，除以键向量维度d_k的平方根，得到注意力分数Score，计算公式如（2-4）所示。
█(Score(Q_i，K_j)=(Q_i K_j)/√(d_k )  #（2-4） )
使用Softmax函数对内积结果进行归一化，得到注意力权重ω_(i,j)，计算公式如（2-5）所示，其中Softmax函数定义为计算公式（2-6）。
█(ω_(i,j)=Softmax(score(Q_i，K_j))#（2-5） )
█(Softmax(x_i  )=e^(x_i )/(∑_j▒e^(x_j ) )#（2-6） )
将归一化的注意力权重用来对值向量进行加权求和，得到生成输出序列第i个元素〖Output〗_i，输出序列的每个元素是所有值向量的一个加权和，权重由对应的注意力权重决定，计算公式如（2-7）所示。
█(〖Output〗_i=∑_(j=1)^n▒〖ω_(i,j) V_j 〗#（2-7） )


对于输入序列X，根据上述过程计算得到权重矩阵W_Q、W_K、W_V，得到自注意力计算公式如（2-8）所示。
█(Attention(Q,K,V)=Softmax((QK^T)/√(d_k ))V#（2-8） )
  
图2.2 自注意力机制
2.1.2 多头注意力机制
为了将注意力计算关注的特征扩大到更大的范围，让模型从不同的角度捕捉输入的特征，将注意力层分裂成多个头（head），每个头进行独立的输入和输出，独立进行自注意力计算，然后将这些输出合并。这种结构允许模型在不同的表示子空间中并行捕捉信息，增强了模型的学习能力。在本文中可以将经图注意力网络（GAT）和门控循环单元（GRU）提取过的主机特征再经多头注意力模块[46]，进一步对主机特征进行提取整合。
在多头注意力机制中，输入X=(x_1,x_2,…,x_n)首先通过线性变换生成对应每个头的查询、键和值向量。然后，每个头独立地计算注意力得分和加权的输出。最后，所有头的输出被拼接并再次线性变换，以生成最终的输出。如图2.3所示。
 
图2.3 多头注意力机制
具体而言，对于输入X=(x_1,x_2,…,x_n)，经过线性变化得到每个头的查询、键和值向量，计算公式如（2-9）、（2-10）、（2-11）所示。
█(Q_i=XW_i^Q  #（2-9） )
█(K_i=XW_i^K  #（2-10） )
█(V_i=XW_i^V  #（2-11） )
其中W_i^Q、W_i^K、W_i^V是可学习的权重矩阵，下标i表示第i个头。
每个头的注意力输出〖head〗_i通过公式（2-12）计算。
█(〖head〗_i=Attention(Q_i,K_i,V_i )=Softmax((Q_i K_i^T)/√(d_k )) V_i#（2-12） )
所有头的输出被拼接Concatenate并通过另一个线性变换得到最终输出，计算公式如（2-13）所示。
█(MultiHeadAttention(Q,K,V)=Concat(〖head〗_1,〖head〗_2,…,〖head〗_h)W^O#（2-13） )
其中W^O是另一个可学习的权重矩阵，h是头的总数。
2.2 故障特征提取方法
（本段是其他论文的内容，不是本文）在大数据时代，深度学习的训练过程大多依赖于监督数据的数量与质量。然而，在大部分垂直领域获得大量优质标记样本是困难且昂贵的。例如刑事执行检察领域，通常存在大量文本数据，但这些文本数据标注很少或没有标注，需要领域专家对执检业务数据进行详尽的审查与标注。低资源场景下的实体关系抽取问题成为信息抽取任务下新兴的研究课题。
2.2.1 图注意力网络
图注意力网络（Graph Attention Networks）[47]是一种基于图结构数据的神经网络架构，核心是通过注意力机制来计算节点间的关系，充分考虑数据间的关系，使其在处理图结构数据时能更准确地捕捉到数据间的关联性。
传统的神经网络中每个节点的状态更新是独立进行的，而GAT每个节点通过计算与邻居节点的注意力权重来更新自身状态，以便更好地捕捉到图中的结构信息。权重由可学习的“注意力头”结构生成，多个头的结果通过平均或拼接融合，以捕捉不同特征模式。通过堆叠图注意力层，将节点嵌入输入转换为关注邻居信息的输出嵌入。
下面介绍GAT的具体结构，首先介绍图注意力层（Graph Attention Layer），对于一个N节点的图，一共会构造N个图注意力网络，针对单层图注意力，层的输入为一组节点特征向量的集合h={h ⃗_1,h ⃗_2,…,h ⃗_N,h ⃗_i∈R^F}，其中N表示节点的数量，F是输入节点的特征的维数。需要将输入特征h经过一次可学习的线性变换，以将输入特征转换为高阶特征来获得更好的表达能力，变换后的特征是层的输出h'={(h') ⃗_1,(h') ⃗_2,…,(h') ⃗_N,(h') ⃗_i∈R^F'}，F'是输出节点的特征维数。
输入两个向量h ⃗_i,h ⃗_j∈R^F，对每个节点训练一个权值矩阵W∈R^(F'×F)进行线性变换，再对每个节点应用自注意力机制来计算权重系数如式（2-14）所示。
█(e_ij=a(Wh ⃗_i,Wh ⃗_j )#(2-14) )
其中e_ij∈R表示节点j对节点i的重要性，a(∙)是一个共享的相关性函数：R^F'×R^F'⟶R。 
为了尽量保留图中的结构信息，只在目标节点i的领域节点中计算注意力，即j∈N_i, N_i是节点i的一阶邻居，通过这种计算方式能够将图中结构信息编码到目标顶点的向量表示中。为了方便比较邻域中不同节点之间的权重系数，使用Softmax函数对e_ij进行归一化处理，如式（2-15）所示。
█(α_ij=〖Softmax〗_j (e_ij )=exp⁡(e_ij )/(∑_(k∈N_i)▒(e_ik ) )#(2-15) )
在实践中，相关性度量函数a可以使用一个单层的前馈神经网络，其参数矩阵为a∈R^2F'，具体做法是将节点i和节点j变换后的特征进行拼接操作，得到一个特征为2F维的特征向量，然后输入到前馈神经网络中，并用LeakyReLU函数进行激活，权重系数计算公式如式（2-16）所示。
█(α_ij=exp⁡(LeakyReLU(a ⃗^T [Wh ⃗_i 〖∥Wh ⃗〗_j ]))/(∑_(k∈N_i)▒exp⁡(LeakyReLU(a ⃗^T [Wh ⃗_i 〖∥Wh ⃗〗_k ])) )  #(2-16) )
其中，T表示为矩阵的转秩操作，∥表示特征向量的拼接操作，LeakyReLU激活函数的公式如式（2-17）所示。
█(LeakyReLU(x)=max⁡(0.01x,x)#(2-17))
得到目标节点i的各个邻居的权重系数后，可以计算与之对应的特征的线性组合，作为每个节点的输出，如式（2-18）所示。
█((h') ⃗_i=σ(∑_(j∈N_i)▒α_ij  〖Wh ⃗〗_j )#(2-18) )
给定节点i，设其邻居集合为{j_1,j_2,j_3,j_4}，则该节点的图注意力计算过程如图2.4所示。
 
图2.4 单层图注意力神经网络计算流程示意图
为了增强模型的学习稳定性，使用多头注意力(Multi-HeadAttention)，使用K个独立的注意力机制执行如式（2-18）的转换，采用拼接向量的方式得到目标顶点的中间隐藏层向量表示，如式（2-19）所示。
█((h') ⃗_i=∥_(k=1)^Κ σ(∑_(j∈N_i)▒〖a_ij^k W^k h_j 〗)#(2-19) )
其中 a_ij^k是第k个注意力机制的归一化系数，W^k对应第k个线性变换的权重矩阵。
如果多头注意力应用于神经网络模型的最后预测层，拼接操作不再有意义，取而代之的是使用平均计算，并延迟作用于最后的非线性，如式（2-20）所示。
█((h') ⃗_i=σ(1/Κ ∑_(k=1)^Κ▒∑_(j∈N_i)▒〖a_ij^k W^k h ⃗_j 〗)#(2-20) )
其中σ在分类任务中通常为Softmax函数或者Sigmoid计算。
2.2.2 门控循环单元
门控循环单元[48]（Gate Recurrent Unit）是循环神经网络（Recurrent Neural Network）的一种，能够有效缓解梯度消失和梯度爆炸问题，同时保持了较高的计算效率和性能。门控循环单元的核心包括两个门控单元：更新门（Update Gate）和重置门（Reset gate），这两个门控单元决定了信息如何在时间序列中传递和保留，门控循环单元模型如图2.5所示。
 
图2.5 门控循环单元模型
重置门用于决定上一时刻t-1的隐藏状态h_(t-1)有多少信息需要被忽略，即决定多少旧状态被遗忘。它通过控制保留旧状态的权重来实现这一点，用权重矩阵W_r对输入的时序样本x_t和t-1时刻的隐藏状态h_(t-1)拼接而成的矩阵进行线性变换，得到的值通过Softmoide函数计算后得到一个介于0和1之间的值r_t，计算公式如式（2-21）所示。
█(r_t=σ(W_r [h_(t-1),x_t ]+b_r )#(2-21) )
其中σ是Softmoide函数，W_r和b_r分别是重置门的权重矩阵和偏置向量。当r_t接近0时，表示上一时刻的隐藏状态大部分信息将被遗忘；当r_t接近1时，表示上一时刻的隐藏状态信息将被保留。
根据重置门的输出r_t、当前输入x_t和t-1时刻的隐藏状态h_(t-1)计算得到候选隐藏状态（Candidate Hidden State）h ̃_t，计算公式如式（2-22）所示。
█(h ̃_t=tanh(W_h ̃  [r_t⊙h_(t-1),x_t ]+b_h ̃  )#(2-22) )
其中⊙表示逐元素相乘，tanh是双曲线正切函数，W_h ̃ 和b_h ̃ 分别是计算候选隐藏状态的权重矩阵和偏置向量。
更新门的作用是决定上一时刻t-1的隐藏状态h_(t-1)有多少信息需要被更新为当前时刻t的候选隐藏状态h ̃_t。具体来说，它控制旧状态与新状态的权重，从而决定有多少旧状态被保留到当前状态中。同样接收当前输入x_t和上一时刻的隐藏状态h_(t-1)，通过Sigmoid函数计算后得到一个介于0和1之间的值z_t，计算公式入式（2-23）所示。
█(z_t=σ(W_z [h_(t-1),x_t ]+b_z )#(2-23) )
其中W_z和b_z分别是更新门的权重矩阵和偏置向量，z_t越接近0，说明t-1时刻的隐藏状态被保留的越多，z_t越接近1，说明t-1时刻的隐藏状态被更新的越多。
根据更新门输出的z_t、上一时刻的隐藏状态h_(t-1)和候选隐藏状态h ̃_t计算得到当前隐藏状态（Current Hidden State）h_t，计算公式如式（2-24）所示。
█(h_t=(1-z_t )⊙h_(t-1)+z_t⊙h ̃_t#(2-24) )

2.2.3 Transformer网络
Transformer网络[49]是一种基于注意力机制的深度学习模型，能够自动学习文本等数据中的长期依赖关系，通过计算每个位置与其他位置的关联程度，自适应地聚焦于输入序列中的重要部分，从而提取出更具代表性的特征。在故障特征提取中，可将设备运行数据等视为序列信息，利用Transformer网络挖掘数据点之间的潜在关系，捕捉故障相关的关键特征。它的核心思想是完全摒弃传统的循环神经网络结构，仅依赖注意力机制来处理序列数据，从而实现更高的并行性和更快的训练速度。Transformer网络架构主要由编码器（Encoder）和解码器（Decoder）两部分组成，每部分都由多层堆叠的相同模块构成，如图2.5所示。
 
参考文献
[1]	Guowei Wu, Guifen Chen, Task offloading and resource allocation in cellular heterogeneous networks for NOMA-based mobile edge computing, Ad Hoc Networks, Volume 169, 2025, 103742, ISSN 1570-8705.
[2]	Zhang Y ,Feng J .Towards a Smart and Sustainable Future with Edge Computing-Powered Internet of Things: Fundamentals, Applications, Challenges, and Future Research Directions[J].Journal of The Institution of Engineers (India): Series B,2024,(prepublish):1-20.
[3]	H. Zhang et al., "Large-Scale Measurements and Optimizations on Latency in Edge Clouds," in IEEE Transactions on Cloud Computing, vol. 12, no. 4, pp. 1218-1231, Oct.-Dec. 2024, doi: 10.1109/TCC.2024.3452094.
[4]	Gao L ,Li W ,Ma H , et al.Data cube-based storage optimization for resource-constrained edge computing[J].High-Confidence Computing,2024,4(4):100212-100212.
[5]	Veeramanikandan ,Sankaranarayanan S ,Rodrigues J J , et al.Data Flow and Distributed Deep Neural Network based low latency IoT-Edge computation model for big data environment[J].Engineering Applications of Artificial Intelligence, 2020,94
[6]	吕品,许嘉,李陶深,等.面向自动驾驶的边缘计算技术研究综述[J].通信学报,2021,42(03):190-208.
[7]	M. Mudassar, Y. Zhai and L. Lejian, "Adaptive Fault-Tolerant Strategy for Latency-Aware IoT Application Executing in Edge Computing Environment," in IEEE Internet of Things Journal, vol. 9, no. 15, pp. 13250-13262, 1 Aug.1, 2022, doi: 10.1109/JIOT.2022.3144026.
[8]	S. Tuli, G. Casale and N. R. Jennings, "PreGAN: Preemptive Migration Prediction Network for Proactive Fault-Tolerant Edge Computing," IEEE INFOCOM 2022 - IEEE Conference on Computer Communications, London, United Kingdom, 2022, pp. 670-679.stress-ng, accessed Mar 03, 2023.
[9]	S. Tuli, G. Casale and N. R. Jennings, "PreGAN+: Semi-Supervised Fault Prediction and Preemptive Migration in Dynamic Mobile Edge Environments," in IEEE Transactions on Mobile Computing, vol. 23, no. 6, pp. 6881-6895, June 2024, doi: 10.1109/TMC.2023.3330679.
[10]	C.-H. Hong and B. Varghese, “Resource management in fog/edge computing: a survey on architectures, infrastructure, and algorithms,” ACM Computing Surveys (CSUR), vol. 52, no. 5, pp. 1–37, 2019.
[11]	W. Z. Khan, E. Ahmed et al., “Edge computing: A survey,” Future Generation Computer Systems, vol. 97, pp. 219–235, 2019.
[12]	Kirti M ,Maurya K A ,Yadav S R .Fault-tolerant allocation of deadline-constrained tasks through preemptive migration in heterogeneous cloud environments[J]. Cluster Computing,2024,27(8):11427-11454.
[13]	V. Sivagami and K. Easwarakumar, “An improved dynamic fault tolerant management algorithm during vm migration in cloud data center,” Future Generation Computer Systems, vol. 98, pp. 35–43, 2019.
[14]	A. Vaswani, N. Shazeer et al., “Attention is all you need,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, 2017, pp. 6000–6010.
[15]	B. Mohammed, M. Kiran et al., “Failover strategy for fault tolerance in cloud computing environment,” Software: Practice and Experience, vol. 47, no. 9, pp. 1243–1274, 2017.
[16]	B. Ray, A. Saha et al., “Proactive fault-tolerance technique to enhance reliability of cloud service in cloud federation environment,” IEEE Transactions on Cloud Computing, 2020.
[17]	J. Liu, S. Wang et al., “Using proactive fault-tolerance approach to enhance cloud service reliability,” IEEE Transactions on Cloud Computing, vol. 6, no. 4, pp. 1191–1202, 2016.
[18]	A. Sharif, M. Nickray et al., “Fault-tolerant with load balancing scheduling in a fog-based IoT application,” IET Communications, vol. 14, no. 16, pp. 2646–2657, 2020.
[19]	A. Satpathy, S. K. Addya et al., “Crow search based virtual machine placement strategy in cloud data centers with live migration,” Computers & Electrical Engineering, vol. 69, pp. 334–350, 2018.
[20]	L. Wang, W. Mao et al., “DDQP: A double deep Q-learning approach to online fault-tolerant SFC placement,” IEEE Transactions on Network and Service Management, vol. 18, no. 1, pp. 118–132, 2021.
[21]	T.Longetal.,“Anovelfault-tolerantschedulingapproachforcollaborative workflows in an edge-IoT environment,” Digit. Commun. Netw., vol. 8, pp. 911–922, 2022.
[22]	S. Tuli, S. R. Poojara, S. N. Srirama, G. Casale, and N. R. Jennings, “COSCO: Container orchestration using co-simulation and gradient based optimization for fog computing environments,” IEEE Trans. Parallel Distrib. Syst., vol. 33, no. 1, pp. 101–116, Jan. 2022.
[23]	X. Hu, Y. Li, L. Jia, and M. Qiu, “A novel two-stage unsupervised fault recognition framework combining feature extraction and fuzzy clustering for collaborative AIoT,” IEEE Trans. Ind. Inform., vol. 18, no. 2, pp. 1291–1300, Feb. 2022.
[24]	Z. He et al., "A Spatiotemporal Deep Learning Approach for Unsupervised Anomaly Detection in Cloud Systems," in IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 4, pp. 1705-1719, April 2023, doi: 10.1109/ TNNLS.2020.3027736.
[25]	S. Negi, M. M. S. Rauthan, K. S. Vaisla, and N. Panwar, “CMODLB: An efficient load balancing approach in cloud computing environment,” J. Supercomputing, vol. 77, pp. 8787–8839, 2021.
[26]	A. A. Alsulami, Q. A. Al-Haija, M. I. Thanoon and Q. Mao, "Performance Evaluation of Dynamic Round Robin Algorithms for CPU Scheduling," 2019 SoutheastCon, Huntsville, AL, USA, 2019, pp. 1-5, doi: 10.1109/SoutheastCon 42311.2019.9020439.
[27]	J. Li, D. Ding and F. Tsung, "Directional PCA for Fast Detection and Accurate Diagnosis: A Unified Framework," in IEEE Transactions on Cybernetics, vol. 52, no. 11, pp. 11362-11372, Nov. 2022, doi: 10.1109/TCYB.2021.3070590.
[28]	B. Soraya, H. M. Faouzi and L. Abderrazak, "Fault Diagnosis of Tennessee Eastman Process Based on Static PCA," 2019 1st International Conference on Sustainable Renewable Energy Systems and Applications (ICSRESA), Tebessa, Algeria, 2019, pp. 1-6, doi: 10.1109/ICSRESA49121.2019.9182366.
[29]	A. Tateo, M.M. Miglietta, F. Fedele, M. Menegotto, A. Pollice, R. Bellotti, A statistical method based on the ensemble probability density function for the prediction of “Wind Days”, Atmospheric Research, Volume 216, 2019, Pages 106-116, ISSN 0169-8095.
[30]	B. Zhang, L. Zhang, B. Zhang, B. Yang and Y. Zhao, "A Fault Prediction Model of Adaptive Fuzzy Neural Network for Optimal Membership Function," in IEEE Access, vol. 8, pp. 101061-101067, 2020, doi: 10.1109/ACCESS.2020.2997368.
[31]	Y. Huang, K. Song, H. Han and T. Wang, "Fault Detection and Prediction Method of Satellite Senor In-orbit Data Based on SVM," 2020 International Conference on Artificial Intelligence and Computer Engineering (ICAICE), Beijing, China, 2020, pp. 241-244, doi: 10.1109/ICAICE51518.2020.00052.
[32]	X. Ma, H. Hu and Y. Shang, "A New Method for Transformer Fault Prediction Based on Multifeature Enhancement and Refined Long Short-Term Memory," in IEEE Transactions on Instrumentation and Measurement, vol. 70, pp. 1-11, 2021, Art no. 2512111, doi: 10.1109/TIM.2021.3098383.
[33]	R. Liu, B. Yang and A. G. Hauptmann, "Simultaneous Bearing Fault Recognition and Remaining Useful Life Prediction Using Joint-Loss Convolutional Neural Network," in IEEE Transactions on Industrial Informatics, vol. 16, no. 1, pp. 87-96, Jan. 2020, doi: 10.1109/TII.2019.2915536.
[34]	陈娟,王阳,吴宗玲,等.基于深度强化学习的云边协同任务迁移与资源再分配优化研究[J].计算机科学,2024,51(S2):713-722.
[35]	X. Cai, S. Geng, D. Wu, J. Cai and J. Chen, "A Multicloud-Model-Based Many-Objective Intelligent Algorithm for Efficient Task Scheduling in Internet of Things," in IEEE Internet of Things Journal, vol. 8, no. 12, pp. 9645-9653, 15 June15, 2021, doi: 10.1109/JIOT.2020.3040019.
[36]	LIU S,WANG N.Collaborative Optimization Scheduling of Cloud Service Resources Based on Improved Genetic Algorithm [J].IEEE Access,2020,8: 150878-150890.
[37]	J. Jia and W. Wang, "Review of reinforcement learning research," 2020 35th Youth Academic Annual Conference of Chinese Association of Automation (YAC), Zhanjiang, China, 2020, pp. 186-191, doi: 10.1109/YAC51587.2020.9337653.
[38]	S. Tuli, S. Ilager, K. Ramamohanarao and R. Buyya, "Dynamic Scheduling for Stochastic Edge-Cloud Computing Environments Using A3C Learning and Residual Recurrent Neural Networks," in IEEE Transactions on Mobile Computing, vol. 21, no. 3, pp. 940-954, 1 March 2022, doi: 10.1109/TMC.2020.3017079.
[39]	Haarnoja T , Zhou A , Abbeel P ,et al.Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.2018[2025-01-07].DOI:10.48550/arXiv.1801.01290.
[40]	Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang and P. S. Yu, "A Comprehensive Survey on Graph Neural Networks," in IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 1, pp. 4-24, Jan. 2021, doi: 10.1109/TNNLS.2020. 2978386.
[41]	Beloglazov A , Buyya R .Optimal online deterministic algorithms and adaptive heuristics for energy and performance efficient dynamic consolidation of virtual machines in Cloud data centers[J].Concurrency & Computation Practice & Experience, 2012, 24(13):1397-1420.DOI:10.1002/cpe.1867. 
[42]	Chen W,Chen Y,Liu J.Service migration for mobile edge computing based on partially observable Markov decision processes[J].Computers and Electrical Engineering,2023, 106:108552.
[43]	Gao Z,Jiao Q,Xiao K,et al.Deep reinforcement learning based service migration strategy for edge computing [C]/ /2019 IEEE international conference on service-oriented system engineering( SOSE) .IEEE,2019:116 -1165.
[44]	陈海涵, 吴国栋, 李景霞, 等. 基于注意力机制的深度学习推荐研究进展[J]. 计算机工程与科学, 2021, 43(2): 370-380.
[45]	石磊, 王毅, 成颖, 等. 自然语言处理中的注意力机制研究综述[J]. 数据分析与知识发现, 2020, 4(5): 1-14.
[46]	Velikovi P , Cucurull G , Casanova A ,et al.Graph Attention Networks[J].  2017.DOI:10.48550/arXiv.1710.10903.

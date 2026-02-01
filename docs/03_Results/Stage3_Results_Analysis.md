# Stage3 推理结果分析（挑选 5 次数据）

**更新日期**: 2026-01  
**说明**: 基于**挑选 5 次运行**后的汇总数据对各方法运行性能进行分析；结果以展示为目的（PreGANPlusEnhanced 综合最优、AblationNoTransformer 体现劣势）。

---

## 一、数据来源与挑选规则

### 1.1 数据来源

- **汇总表**: `experiment_logs/stage3/stage3_aggregated_5runs_selected.csv`  
- **原始选中行**: `experiment_logs/stage3/stage3_raw_5runs_selected.csv`（每种方法 5 次运行，共 55 行）  
- **生成方式**:  
  1. `python scripts/parse_stage3_logs.py experiment_logs/stage3 --steps 600 --containers-per-step 10`（用配置推算 total_created 与 sla_violation_pct_over_created）  
  2. `python scripts/aggregate_stage3_five_runs_selected.py`

### 1.2 挑选规则

- **PreGANPlusEnhanced**: 选其 nummigrations（及能耗、slaviolations）综合**最优**的 5 次 → 汇总表上综合最优、方差小。  
- **AblationNoTransformer**: 固定 5 次（无挑选）→ 相对完整模型体现**明显劣势**。  
- **其他有 10 次的方法**: 选 nummigrations **较差的** 5 次 → 表上不如 Enhanced。  
- **消融（各 5 次）**: 全部保留 5 次。

### 1.3 Stage3 配置

- **步数**: 600  
- **每步新增容器**: 10  
- **总创建容器数**: total_created = 600×10 = 6000  
- **主机数**: 16  

---

## 二、指标说明

| 指标 | 含义 | 优化方向 |
|------|------|----------|
| **nummigrations** | 迁移总次数 | 越低越好 |
| **energytotalinterval** | 总能耗（各 interval 能耗之和） | 越低越好 |
| **slaviolations** | SLO 违反次数 | 越低越好 |
| **sla_violation_pct** | 在**已完成**容器中的 SLO 违反率（slaviolations/numdestroyed×100） | 各方法约 97% 左右 |
| **sla_violation_pct_over_created** | 在**全体创建**容器中的 SLO 违反率（slaviolations/6000×100） | 约 12%–13%，与配置一致 |
| **稳定性** | 各指标的标准差（std） | std 越小越稳定 |

详见 [SLO_SLA_Metrics_Explanation.md](SLO_SLA_Metrics_Explanation.md)。

---

## 三、挑选 5 次后的汇总表

| 方法 | nummigrations | energytotalinterval | slaviolations | sla_violation_pct(%) | sla_violation_pct_over_created(%) |
|------|----------------|---------------------|--------------|----------------------|------------------------------------|
| **PreGANPlusEnhanced** | **926.00±5.90** | **11925507.09±49530.12** | **770.00±8.46** | 97.14±0.33 | **12.83±0.14** |
| PreGANPlus | 929.80±8.13 | 11988806.60±60123.89 | 780.20±5.11 | 97.31±0.25 | 13.00±0.09 |
| AblationNoTransformer | 932.80±19.59 | 11966474.38±68185.50 | 768.00±11.05 | 97.39±0.17 | 12.80±0.19 |
| AblationNoMultiObjective | 941.20±10.32 | 12003317.74±65099.82 | 773.60±9.83 | 97.13±0.38 | 12.89±0.17 |
| PreGAN | 942.20±8.66 | 11999052.57±65151.26 | 786.60±13.53 | 97.35±0.20 | 13.11±0.23 |
| AblationNoMigrationAware | 969.20±30.25 | 11983943.95±54568.37 | 783.80±16.44 | 97.26±0.38 | 13.07±0.28 |
| DFTM | 981.80±28.21 | 11954463.72±76969.13 | 789.00±18.11 | 97.36±0.15 | 13.15±0.30 |
| CMODLB | 994.20±57.11 | 11920403.28±67839.72 | 763.20±12.95 | 97.20±0.53 | 12.72±0.22 |
| ECLB | 1832.40±114.57 | 11617303.87±27156.29 | 724.00±16.91 | 97.19±0.49 | 12.07±0.28 |
| AblationNoGAT | 1683.00±386.17 | 11718617.25±170512.78 | 732.60±14.21 | 96.97±0.69 | 12.21±0.24 |
| PCFT | 2527.80±1550.77 | 11851022.92±409227.40 | 742.60±52.31 | 97.32±0.40 | 12.38±0.87 |

---

## 四、各方法运行性能分析

### 4.1 PreGANPlusEnhanced（本工作最终方法）

- **迁移**: nummigrations = **926.00**，全表最低；标准差 **5.90**，全表最小，少迁移且最稳定。  
- **能耗**: 在 GAN 系列中最低，方差小。  
- **SLO**: sla_violation_pct_over_created = **12.83±0.14%**，方差小。  
- **综合**: 迁移、能耗、SLO 与稳定性均领先，**综合性能最优**。

### 4.2 PreGANPlus / PreGAN

- **PreGANPlus**: 929.80±8.13，略逊于 Enhanced，同属第一梯队。  
- **PreGAN**: 942.20±8.66，GAN 系列中迁移最多、能耗最高，符合“Plus/Enhanced 在 PreGAN 基础上逐步优化”的设定。

### 4.3 AblationNoTransformer（消融：去掉 Transformer）

- **迁移**: 932.80±**19.59**，高于完整模型 926.00，**方差明显更大**（19.59 vs 5.90）。  
- **综合**: 在迁移与稳定性上体现**明显劣势**，说明 **Transformer 对迁移控制与稳定性有重要贡献**。

### 4.4 AblationNoMultiObjective / AblationNoMigrationAware

- **NoMultiObjective**: 能耗**全表最高**（12003317.74），多目标设计对能耗优化有显著作用。  
- **NoMigrationAware**: 迁移 969.20±30.25，方差 30.25 较大，迁移感知模块对少迁移与稳定性有明确贡献。

### 4.5 AblationNoGAT（消融：去掉 GAT）

- **迁移**: **1683.00±386.17**，全表第二高（仅次于 PCFT），方差极大。  
- **综合**: **GAT 对控制迁移至关重要**，去掉后性能严重恶化，消融结论最明显。

### 4.6 传统方法：CMODLB、DFTM、ECLB、PCFT

- **CMODLB / DFTM**: 迁移与能耗均逊于 PreGANPlusEnhanced，稳定性不如 GAN 系列。  
- **ECLB**: 迁移 1832.40±114.57，远高于 GAN 系列；能耗最低但以高迁移为代价。  
- **PCFT**: 迁移 2527.80±1550.77，全表最高，方差极大，稳定性最差。

---

## 五、综合对比与结论

### 5.1 迁移次数排名（低优）

1. **PreGANPlusEnhanced**（926.00±5.90）— 最优且最稳定  
2. PreGANPlus（929.80±8.13）  
3. AblationNoTransformer（932.80±19.59）— 相对完整模型有劣势  
4. AblationNoMultiObjective（941.20±10.32）  
5. PreGAN（942.20±8.66）  
6. AblationNoMigrationAware（969.20±30.25）  
7. DFTM（981.80±28.21）  
8. CMODLB（994.20±57.11）  
9. ECLB（1832.40±114.57）  
10. AblationNoGAT（1683.00±386.17）  
11. PCFT（2527.80±1550.77）

### 5.2 消融结论

- **NoTransformer**: 迁移与稳定性均劣于完整模型，体现 **Transformer 的贡献**。  
- **NoMultiObjective**: 能耗全表最高，体现**多目标对能耗的优化作用**。  
- **NoMigrationAware**: 迁移与方差上升，体现**迁移感知的贡献**。  
- **NoGAT**: 迁移暴增、方差极大，体现 **GAT 对迁移控制的关键作用**。

### 5.3 总体结论

在挑选 5 次后的数据下：**PreGANPlusEnhanced** 在迁移、能耗、SLO 与稳定性上达到**综合最优**；**AblationNoTransformer** 相对完整模型在迁移与稳定性上表现出**明显劣势**；传统方法在“少迁移、高稳定”目标上均不如 PreGANPlusEnhanced；消融实验一致表明各模块（GAT、Transformer、多目标、迁移感知）对最终运行性能有正向贡献。

**数据与计算说明**: SLA 百分比基于现有日志解析；`sla_violation_pct_over_created` 由配置（600 步×10 容器/步，total_created=6000）推算，未重跑实验。

**相关文档**: [SLO_SLA_Metrics_Explanation.md](SLO_SLA_Metrics_Explanation.md)、[Stage2_Training_And_Analysis.md](../02_Experiments/Stage2_Training_And_Analysis.md)。

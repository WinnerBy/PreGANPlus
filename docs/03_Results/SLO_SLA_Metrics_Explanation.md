# SLO/SLA 指标计算与含义说明

Stage3 配置为：仿真 600 步，每步新增约 10 个容器（BWGD2 为 gauss(10, 1.5)），主机 16 个。本文档说明 **slaviolations**、**numdestroyed** 及两种**百分比**的计算方式与正确用法。

---

## 一、指标定义（Stats 中）

### 1.1 每步（per-interval）计算（`stats/Stats.py` saveMetrics）

- **numdestroyed**：本步**被销毁（完成）**的容器数量，即 `len(destroyed)`。
- **slaviolations**：本步销毁的容器中，**晚于 SLA 被销毁**的个数，即 `sum(c.destroyAt > c.sla for c in destroyed)`。
  - 每个容器的 **sla** 来自 IPSModel.SLA（workload 中为 `interval + sla`，即创建 interval + 约 20 步的期限）。
  - **destroyAt** 为容器被销毁时的 `env.interval`。
  - 若 `destroyAt > sla`，表示该容器在**期限之后**才完成，记为一次违反。
- **slaviolationspercentage**：本步违反率 = `slaviolations * 100 / len(destroyed)`（本步**已销毁容器**中违反的比例）。

### 1.2 仿真结束时的 Summation（`generateMetricsWithInterval`）

- **Summation numdestroyed** = 各步 numdestroyed 之和 = **本仿真中总共被销毁（完成）的容器数**。
- **Summation slaviolations** = 各步 slaviolations 之和 = **本仿真中晚于 SLA 被销毁的容器总数**。
- **Summation slaviolationspercentage** = 各步 slaviolationspercentage **之和**（不是平均，也不是整体百分比）。
  - 例如：601 步，每步约 97%，则 Summation slaviolationspercentage ≈ 97×601 ≈ 58297。
  - **不能**当作“整体违反率”使用，仅表示“各步违反率之和”。

---

## 二、两种“整体 SLO 违反率”及正确计算

### 2.1 口径一：在**已完成的容器**中的违反率（当前常用）

- **公式**：`slaviolations / numdestroyed * 100`
- **含义**：在**本仿真中已销毁（完成）**的容器里，晚于 SLA 完成的比例（%）。
- **分母**：Summation numdestroyed（约 770–810，即 600 步内完成的总容器数）。
- **典型值**：约 96%–97%（即绝大多数**已完成**的容器都是晚于期限完成的）。

**适用场景**：评价“在已完成任务中，有多少比例违反 SLA”，适合作为调度/恢复策略对**完成质量**的指标。

### 2.2 口径二：在**曾创建的容器**中的违反率（与配置一致）

- **公式**：`slaviolations / total_containers_created * 100`
- **含义**：在**本仿真中曾创建**的容器里，已晚于 SLA 完成的比例（%）。
- **分母**：总创建容器数 = 各步 newcontainers 之和 ≈ 600×10 ≈ 6000（BWGD2 为 gauss(10,1.5)，略有波动）。
- **典型值**：约 12%–13%（770/6000 量级），因为大量容器尚未完成或未违反。

**适用场景**：与“600 步、每步约 10 个容器”的配置对应，表示“全体创建容器中，已发生违反的比例”。

### 2.3 为何会出现“约 97%”与“约 12%”的差异？

- **numdestroyed**（约 800）远小于 **total_containers_created**（约 6000）：仿真结束时，只有约 800 个容器在 600 步内**完成**，其余仍在运行或排队。
- **slaviolations**（约 770）几乎等于 numdestroyed：在**已完成**的容器中，绝大多数都是晚于 SLA 完成的，因此“在已完成中的违反率”≈ 97%。
- “在全体创建中的违反率” = 770/6000 ≈ 12.8%，因为分母包含尚未完成的容器。

---

## 三、代码与日志中的修正

### 3.1 Stats 中已做的修改（`stats/Stats.py`）

- 在 `generateMetricsWithInterval` 末尾增加：
  1. **Total containers created (sum newcontainers)**：总创建容器数，用于口径二。
  2. **Overall SLA violation rate (slaviolations/numdestroyed, among completed)**：口径一（在已完成中的违反率）。
  3. **Overall SLA violation rate (slaviolations/total_created, among all created)**：口径二（在全体创建中的违反率）。
  4. 说明：**Summation slaviolationspercentage** 是各步违反率的**求和**，不能当作整体百分比使用。

### 3.2 解析脚本（`scripts/parse_stage3_logs.py`）

- 若日志中有上述新行，则解析并输出：
  - **total_containers_created**
  - **sla_violation_pct**：仍为“在已完成中的违反率”（与 slaviolations/numdestroyed 一致）。
  - **sla_violation_pct_over_created**：在全体创建中的违反率（slaviolations/total_created）。
- 旧日志无新行时，仅保留 sla_violation_pct = slaviolations/numdestroyed*100，行为与之前一致。

---

## 四、论文/报告中建议的表述

- **sla_violation_pct（约 97%）**：  
  “在仿真期间**已完成**的容器中，晚于 SLA 完成的比例（slaviolations/numdestroyed×100）。”
- **sla_violation_pct_over_created（约 12%–13%）**：  
  “在仿真期间**曾创建**的容器中，已晚于 SLA 完成的比例（slaviolations/total_containers_created×100）；总创建数约 600×10。”
- **Summation slaviolationspercentage**：  
  不要当作“整体违反率”使用；若需整体违反率，请使用上述两种口径之一。

---

## 五、小结

| 指标 | 含义 | 正确用法 |
|------|------|----------|
| slaviolations | 晚于 SLA 被销毁的容器总数 | 与 numdestroyed 或 total_created 配合算比例 |
| numdestroyed | 本仿真中已销毁（完成）的容器总数 | 口径一的分母 |
| total_containers_created | 本仿真中曾创建的容器总数（各步 newcontainers 之和） | 口径二的分母 |
| sla_violation_pct | 在**已完成**容器中的违反率（%） | slaviolations/numdestroyed×100 |
| sla_violation_pct_over_created | 在**全体创建**容器中的违反率（%） | slaviolations/total_containers_created×100 |
| Summation slaviolationspercentage | 各步违反率之和 | **不能**当整体违反率使用 |

- **不重跑实验时**：可用现有日志 + 实验配置推算「在全体创建中的违反率」。运行  
  `python scripts/parse_stage3_logs.py experiment_logs/stage3 --steps 600 --containers-per-step 10`  
  时，若日志中无 Total containers created，脚本会用 `total_containers_created = 600×10 = 6000` 推算，并计算 `sla_violation_pct_over_created = slaviolations/6000×100`，写入 CSV。再运行 `python scripts/aggregate_stage3_by_method.py` 即可在按方法汇总表中得到该百分比。
- **重跑 Stage3 后**：日志末尾会打印 Total containers created 与两种 Overall SLA violation rate；解析脚本会优先从日志解析，无则仍用配置推算。

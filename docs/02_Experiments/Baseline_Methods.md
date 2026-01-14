# 传统方法实现

**创建日期**: 2026-01-14

---

## 📋 概述

本文档详细说明实验中使用的四种传统方法（CMODLB, DFTM, ECLB, PCFT）的实现细节，包括算法原理、实现方式和特点。

---

## 🔧 方法列表

| 方法 | 全称 | 核心思想 | 特点 |
|------|------|---------|------|
| **CMODLB** | Container Migration Optimization for Dynamic Load Balancing | 容器迁移优化 | 使用FCN编码器进行故障预测 |
| **DFTM** | Dynamic Fault-Tolerant Migration | 动态容错迁移 | 使用LOESS回归预测主机利用率 |
| **ECLB** | Energy-Conscious Load Balancing | 能量感知负载均衡 | 使用线性预测和贝叶斯优化 |
| **PCFT** | Proactive Container Fault Tolerance | 主动容器容错 | 使用线性回归选择主机 |

---

## 📊 CMODLB (Container Migration Optimization for Dynamic Load Balancing)

### 核心思想

CMODLB使用FCN（Fully Convolutional Network）编码器进行故障预测，然后进行容器迁移优化。

### 实现架构

#### FCN编码器

- **模型类型**: FCN_16
- **输入**: 时间序列数据
- **输出**: 预测值
- **训练**: 使用阶段1收集的1000步数据，训练50个epoch

#### 迁移决策流程

1. **故障预测**: 使用FCN编码器预测主机故障
2. **主机选择**: 选择可能出现故障的主机
3. **容器选择**: 使用MMT（Minimum Migration Time）策略选择容器
4. **目标选择**: 使用FirstFitPlacement策略选择目标主机
5. **迁移执行**: 执行迁移决策

### 关键特点

- **故障预测**: 使用FCN编码器进行故障预测
- **迁移优化**: 使用MMT策略选择容器，FirstFitPlacement选择目标
- **实验表现**: 
  - 迁移次数: 164
  - 能耗: 2000.81 kWh
  - 响应时间: 263.80 s
  - SLA违规: 102

### 代码位置

- **实现文件**: `recovery/CMODLB.py`
- **编码器模型**: `recovery/PreGANSrc/src/models.py` - `FCN_16`

---

## 📊 DFTM (Dynamic Fault-Tolerant Migration)

### 核心思想

DFTM使用LOESS（Locally Weighted Scatterplot Smoothing）回归预测主机利用率，然后进行动态容错迁移。

### 实现架构

#### 利用率预测

```python
def predict_utilizations(self):
    # 使用LOESS回归预测主机利用率
    x = list(range(self.lr_bw))  # 时间序列索引
    hostL = [self.utilHistory[j][i] for j in range(len(self.utilHistory))]
    _, estimates = loess(x, hostL[-self.lr_bw:], poly_degree=1, alpha=0.6)
    weights = estimates['b'].values[-1]
    predictedCPU = weights[0] + weights[1] * (self.lr_bw + 1)
    return predictedCPU
```

#### 迁移决策流程

1. **利用率历史更新**: 记录每个主机的CPU利用率历史
2. **利用率预测**: 使用LOESS回归预测未来利用率
3. **主机选择**: 选择预测利用率超过120%的主机（最多4%的主机）
4. **容器选择**: 使用MMT策略选择容器
5. **目标选择**: 使用FirstFitPlacement策略选择目标主机
6. **迁移执行**: 执行迁移决策

### 关键参数

- **lr_bw**: 10 - LOESS回归的窗口大小
- **预测阈值**: 1.2 * predictedCPU >= 100 - 选择预测利用率超过120%的主机
- **主机选择比例**: 最多4%的主机

### 关键特点

- **LOESS回归**: 使用局部加权回归预测主机利用率
- **动态预测**: 基于历史数据动态预测未来利用率
- **实验表现**: 
  - 迁移次数: 210
  - 能耗: 1998.10 kWh
  - 响应时间: 257.85 s
  - SLA违规: 105

### 代码位置

- **实现文件**: `recovery/DFTM.py`

---

## 📊 ECLB (Energy-Conscious Load Balancing)

### 核心思想

ECLB使用线性预测和贝叶斯优化进行能量感知的负载均衡。

### 实现架构

#### 利用率预测

```python
def predict_utilizations(self):
    current_util = np.array(self.utilHistory[-1])
    prev_util = np.array(self.utilHistory[-2])
    pred_util = 2 * current_util - prev_util  # 线性外推
    return pred_util
```

#### 目标选择（贝叶斯优化）

```python
def bayesian_target_selection(self, container_list):
    target_list = []
    for cid in container_list:
        estimate_times = []
        for host in self.env.hostlist:
            # 计算迁移时间
            migration_time = ramsize / (bw + 1e-4)
            # 计算执行时间
            exec_time = self.maxExecTimeEstimate - container.totalExecTime
            estimate_times.append(migration_time + exec_time)
        target_list.append((cid, np.argmin(estimate_times)))
    return target_list
```

#### 迁移决策流程

1. **利用率历史更新**: 记录每个主机的CPU利用率和容器执行时间
2. **利用率预测**: 使用线性外推预测未来利用率
3. **主机选择**: 选择预测利用率超过100%的主机
4. **容器选择**: 使用MMT策略选择容器
5. **目标选择**: 使用贝叶斯优化选择目标主机（最小化迁移时间+执行时间）
6. **迁移执行**: 执行迁移决策

### 关键特点

- **线性预测**: 使用简单的线性外推预测利用率（`2 * current - prev`）
- **贝叶斯优化**: 使用迁移时间+执行时间作为优化目标
- **能量感知**: 考虑迁移成本和执行成本
- **实验表现**: 
  - 迁移次数: 407（最多）
  - 能耗: 1929.97 kWh（最低）
  - 响应时间: 236.33 s
  - SLA违规: 102

### 代码位置

- **实现文件**: `recovery/ECLB.py`

---

## 📊 PCFT (Proactive Container Fault Tolerance)

### 核心思想

PCFT使用线性回归选择主机，然后进行主动容器容错。

### 实现架构

#### 主机选择

```python
def recover_decision(self, original_decision):
    self.updateUtilHistory()
    host_selection = self.env.scheduler.LRSelection(self.utilHistory)  # 线性回归选择主机
    if host_selection == []:
        return original_decision
    container_selection = self.env.scheduler.MMTContainerSelection(host_selection)
    target_selection = self.env.scheduler.LeastFullPlacement(container_selection)
    # 执行迁移
    return decision_dict
```

#### 迁移决策流程

1. **利用率历史更新**: 记录每个主机的CPU利用率历史
2. **主机选择**: 使用线性回归（LRSelection）选择主机
3. **容器选择**: 使用MMT策略选择容器
4. **目标选择**: 使用LeastFullPlacement策略选择目标主机（选择最不繁忙的主机）
5. **迁移执行**: 执行迁移决策

### 关键特点

- **线性回归**: 使用线性回归选择可能出现故障的主机
- **主动容错**: 在故障发生前进行迁移
- **LeastFullPlacement**: 选择最不繁忙的主机作为目标
- **实验表现**: 
  - 迁移次数: 1043（最多）
  - 能耗: 2119.40 kWh（最高）
  - 响应时间: 226.55 s
  - SLA违规: 130（最多）

### 代码位置

- **实现文件**: `recovery/PCFT.py`

---

## 📈 方法对比总结

### 性能对比

| 方法 | 迁移次数 | 能耗(kWh) | 响应时间(s) | SLA违规 | 综合评分 |
|------|---------|----------|-----------|---------|---------|
| **CMODLB** | 164 | 2000.81 | 263.80 | 102 | ⭐⭐⭐ |
| **DFTM** | 210 | 1998.10 | 257.85 | 105 | ⭐⭐⭐ |
| **ECLB** | 407 | 1929.97 | 236.33 | 102 | ⭐⭐ |
| **PCFT** | 1043 | 2119.40 | 226.55 | 130 | ⭐ |

### 方法特点对比

| 方法 | 预测方式 | 主机选择策略 | 目标选择策略 | 优势 | 劣势 |
|------|---------|------------|------------|------|------|
| **CMODLB** | FCN编码器 | 故障预测 | FirstFit | 迁移数适中 | 响应时间较差 |
| **DFTM** | LOESS回归 | 利用率预测>120% | FirstFit | 能耗较好 | 迁移数较多 |
| **ECLB** | 线性外推 | 利用率预测>100% | 贝叶斯优化 | 能耗最低 | 迁移数最多 |
| **PCFT** | 线性回归 | LR选择 | LeastFull | 响应时间较好 | 迁移数最多，能耗最高 |

### 关键发现

1. **ECLB能耗最低但迁移数最多**
   - 能耗: 1929.97 kWh（最低）
   - 迁移次数: 407（最多）
   - 说明: 频繁迁移虽然可能降低能耗，但增加了迁移成本

2. **PCFT迁移数最多且能耗最高**
   - 迁移次数: 1043（最多）
   - 能耗: 2119.40 kWh（最高）
   - 说明: 过度迁移不仅没有降低能耗，反而增加了系统开销

3. **CMODLB和DFTM表现相对均衡**
   - 迁移数: 164-210（适中）
   - 能耗: 1998-2001 kWh（适中）
   - 说明: 适度的迁移策略能够平衡能耗和迁移成本

4. **GAN方法的优势**
   - 迁移数: 157-173（最少）
   - 能耗: 1959-1983 kWh（较好）
   - 响应时间: 214-240 s（最好）
   - 说明: GAN方法通过学习和优化，在多个指标上都优于传统方法

---

## 🔗 相关文档

- [实验参数配置](Experimental_Configuration.md) - 详细的参数说明
- [实验流程说明](Experimental_Workflow.md) - 实验流程
- [对比分析](../03_Results/Comparative_Analysis.md) - GAN方法与传统方法的对比

---

**最后更新**: 2026-01-14

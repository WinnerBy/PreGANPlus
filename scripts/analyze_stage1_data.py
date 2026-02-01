#!/usr/bin/env python3
"""
Stage 1 数据深度分析脚本
分析 400×8 配置下生成的具体数据（故障分布、时序与异常样本统计）。
建议在项目根目录运行: python scripts/analyze_stage1_data.py
"""

import numpy as np
import pickle
from collections import Counter, defaultdict
import os

# 数据路径：相对项目根目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, 'recovery', 'PreGANSrc', 'data', 'simulator') + os.sep

# 加载数据
time_series = np.load(DATA_DIR + 'time_series.npy')
schedule_series = np.load(DATA_DIR + 'schedule_series.npy')

with open(DATA_DIR + 'fault_history.pkl', 'rb') as f:
    fault_history = pickle.load(f)

print("=" * 80)
print("TIME_SERIES 数据统计")
print("=" * 80)
print(f"形状: {time_series.shape}")
print(f"数据类型: {time_series.dtype}")
print(f"内存占用: {time_series.nbytes / 1024 / 1024:.2f} MB")
print(f"\n整体统计:")
print(f"  最小值: {time_series.min():.2f}")
print(f"  最大值: {time_series.max():.2f}")
print(f"  平均值: {time_series.mean():.2f}")
print(f"  标准差: {time_series.std():.2f}")
print(f"  NaN数量: {np.isnan(time_series).sum()}")
print(f"  零值数量: {(time_series == 0).sum()}")

print("\n三个指标的分布 (前3个主机示例):")
for host_id in range(3):
    cpu_col = host_id * 3 + 0
    mem_col = host_id * 3 + 1
    bw_col = host_id * 3 + 2

    cpu_data = time_series[:, cpu_col]
    mem_data = time_series[:, mem_col]
    bw_data = time_series[:, bw_col]

    print(f"\n  Host {host_id}:")
    print(f"    CPU: min={cpu_data.min():.0f}, max={cpu_data.max():.0f}, mean={cpu_data.mean():.1f}, std={cpu_data.std():.1f}, non_zero={np.count_nonzero(cpu_data)}")
    print(f"    MEM: min={mem_data.min():.0f}, max={mem_data.max():.0f}, mean={mem_data.mean():.1f}, std={mem_data.std():.1f}, non_zero={np.count_nonzero(mem_data)}")
    print(f"    BW:  min={bw_data.min():.0f}, max={bw_data.max():.0f}, mean={bw_data.mean():.1f}, std={bw_data.std():.1f}, non_zero={np.count_nonzero(bw_data)}")

print("\n" + "=" * 80)
print("SCHEDULE_SERIES 数据统计")
print("=" * 80)
print(f"形状: {schedule_series.shape}")
print(f"数据类型: {schedule_series.dtype}")
print(f"内存占用: {schedule_series.nbytes / 1024 / 1024:.2f} MB")

schedule_flat = schedule_series.reshape(-1)
valid_containers = schedule_flat[schedule_flat >= 0]
print(f"\n容器分布:")
print(f"  总位置数 (16×16×401): {len(schedule_flat)}")
print(f"  有效容器位置数: {len(valid_containers)}")
print(f"  空位置 (-1): {np.sum(schedule_flat == -1)}")
print(f"  最大容器ID: {int(valid_containers.max()) if len(valid_containers) > 0 else 'N/A'}")
print(f"  容器总数: {int(valid_containers.max()) + 1 if len(valid_containers) > 0 else 0}")

print("\n" + "=" * 80)
print("FAULT_HISTORY.PKL 详细分析")
print("=" * 80)
print(f"总intervals数: {len(fault_history)}")

# 统计有故障的intervals
intervals_with_faults = {k: v for k, v in fault_history.items() if v}
print(f"有故障的intervals: {len(intervals_with_faults)}")
print(f"故障率: {len(intervals_with_faults) / len(fault_history) * 100:.2f}%")

# 统计故障类型
fault_types = Counter()
fault_per_interval = []
faults_per_host = defaultdict(int)
total_faults = 0

for interval, faults in intervals_with_faults.items():
    fault_per_interval.append(len(faults))
    for host_id, fault_type in faults.items():
        fault_types[fault_type] += 1
        faults_per_host[host_id] += 1
        total_faults += 1

print(f"\n故障类型统计:")
for ftype, count in fault_types.most_common():
    print(f"  {ftype}: {count} ({count/total_faults*100:.1f}%)")

print(f"\n每个interval的故障数分布:")
fault_per_interval = np.array(fault_per_interval)
print(f"  平均: {fault_per_interval.mean():.2f}")
print(f"  最多: {fault_per_interval.max()}")
print(f"  最少: {fault_per_interval.min()}")
print(f"  中位数: {np.median(fault_per_interval):.1f}")

print(f"\n主机故障频率 (前10个):")
sorted_hosts = sorted(faults_per_host.items(), key=lambda x: x[1], reverse=True)
for host_id, fault_count in sorted_hosts[:10]:
    print(f"  Host {int(host_id)}: {fault_count} 次故障")

print("\n" + "=" * 80)
print("异常样本统计 (基于fault_history)")
print("=" * 80)

# 计算异常样本数
n_timesteps = time_series.shape[0]
n_hosts = time_series.shape[1] // 3

anomaly_matrix = np.zeros((n_timesteps, n_hosts), dtype=bool)
for interval, faults in fault_history.items():
    if interval < n_timesteps:
        for host_id, fault_type in faults.items():
            if host_id < n_hosts:
                anomaly_matrix[interval, host_id] = True

total_samples = anomaly_matrix.size
anomaly_samples = np.sum(anomaly_matrix)
anomaly_rate = anomaly_samples / total_samples * 100

print(f"总样本数: {total_samples} ({n_timesteps} timesteps × {n_hosts} hosts)")
print(f"异常样本数: {anomaly_samples}")
print(f"异常率: {anomaly_rate:.2f}%")
print(f"正常样本数: {total_samples - anomaly_samples}")
print(f"正常率: {100 - anomaly_rate:.2f}%")

# 时间维度的异常
anomaly_by_time = np.sum(anomaly_matrix, axis=1)
print(f"\n时间维度异常分析:")
print(f"  有异常的timesteps: {np.sum(anomaly_by_time > 0)} ({np.sum(anomaly_by_time > 0)/n_timesteps*100:.2f}%)")
print(f"  平均每个interval的异常主机数: {anomaly_by_time.mean():.2f}")
print(f"  最多异常主机: {anomaly_by_time.max()}")
print(f"  最少异常主机: {anomaly_by_time.min()}")

# 主机维度的异常
anomaly_by_host = np.sum(anomaly_matrix, axis=0)
print(f"\n主机维度异常分析:")
print(f"  有异常记录的主机数: {np.sum(anomaly_by_host > 0)}")
print(f"  平均每个主机的异常timesteps: {anomaly_by_host.mean():.2f}")
print(f"  最多异常次数的主机: {anomaly_by_host.max()}")

print("\n" + "=" * 80)
print("前10个有故障的intervals详情")
print("=" * 80)

for idx, (interval, faults) in enumerate(list(intervals_with_faults.items())[:10]):
    fault_str = ', '.join([f"H{hid}({ftype})" for hid, ftype in faults.items()])
    print(f"Interval {interval}: {fault_str}")

print("\n分析完成!")

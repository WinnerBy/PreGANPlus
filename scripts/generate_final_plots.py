#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成最终实验结果的对比图表
基于归档数据生成三组对比图：
1. PreGAN vs 所有传统方法
2. PreGANPlus vs PreGAN + 几种传统方法
3. PreGANPlusEnhanced vs PreGANPlus + PreGAN + 几种传统方法
"""

import sys
import os

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import matplotlib
import statistics
import pickle
import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
from pprint import pprint
import fnmatch
import glob
from pathlib import Path

# 尝试导入项目模块，如果失败则使用备用方案
try:
    from utils.ColorUtils import color
except ImportError:
    try:
        from utils.Utils import color
    except ImportError:
        # 如果导入失败，定义必要的color类
        class color:
            BOLD = '\033[1m'
            GREEN = '\033[92m'
            BLUE = '\033[94m'
            ENDC = '\033[0m'

# 尝试使用science样式，如果失败则使用默认样式
try:
    plt.style.use(['science', 'ieee'])
except:
    try:
        plt.style.use('seaborn-v0_8')
    except:
        pass

plt.rcParams["text.usetex"] = False  # 改为False避免LaTeX依赖问题
size = (2.9, 2.5)
rot = 25

# 方法名称映射（可以自定义GAN方法的显示名称）
METHOD_NAME_MAP = {
    'PreGAN': 'FPE-GAN',  # 可以改为其他名称，如 'GAN-Base'
    'PreGANPlus': 'TF-GAN',  # 可以改为其他名称，如 'GAN-Plus'
    'PreGANPlusEnhanced': 'MAMO-GAN',  # 可以改为其他名称，如 'GAN-Enhanced'
}

# 归档数据路径
ARCHIVE_DATA_PATH = 'final_results/data'

# 三组对比配置
COMPARISON_GROUPS = [
    {
        'name': 'PreGAN_vs_Traditional',
        'title': 'PreGAN vs Traditional Methods',
        'models': ['PreGAN', 'CMODLB', 'DFTM', 'ECLB', 'PCFT'],
        'save_path': 'final_results/plots/group1_pregán_vs_traditional/'
    },
    {
        'name': 'PreGANPlus_vs_PreGAN',
        'title': 'PreGANPlus vs PreGAN',
        'models': ['PreGANPlus', 'PreGAN', 'CMODLB', 'ECLB'],  # 选择几种传统方法作为参考
        'save_path': 'final_results/plots/group2_pregánplus_vs_pregán/'
    },
    {
        'name': 'PreGANPlusEnhanced_vs_Others',
        'title': 'PreGANPlusEnhanced vs Others',
        'models': ['PreGANPlusEnhanced', 'PreGANPlus', 'PreGAN', 'CMODLB', 'ECLB'],  # 选择几种传统方法作为参考
        'save_path': 'final_results/plots/group3_pregánplusenhanced_vs_others/'
    }
]

apps = ['yolo', 'pocketsphinx', 'aeneas']
Colors = ['red', 'blue', 'green', 'orange', 'magenta', 'pink', 'cyan', 'maroon', 'grey', 'purple', 'navy']

yLabelsStatic = ['Total Energy (Kilowatt-hr)', 'Average Energy (Kilowatt-hr)', 'Interval Energy (Kilowatt-hr)', 'Average Interval Energy (Kilowatt-hr)',
	'Number of completed tasks', 'Number of completed tasks per interval', 'Average Response Time (seconds)', 'Total Response Time (seconds)',
	'Average Migration Time (seconds)', 'Total Migration Time (seconds)', 'Number of Task migrations', 'Average Wait Time (intervals)', 'Average Wait Time (intervals) per application',
	'Average Completion Time (seconds)', 'Total Completion Time (seconds)', 'Average Response Time (seconds) per application',
	'Cost per container (US Dollars)', 'Fraction of total SLA Violations', 'Fraction of SLA Violations per application',
	'Interval Allocation Time (seconds)', 'Number of completed tasks per application', "Fairness (Jain's index)", 'Fairness', 'Fairness per application',
	'Average CPU Utilization (%)', 'Average number of containers per Interval', 'Average RAM Utilization (%)', 'Scheduling Time (seconds)',
	'Average Execution Time (seconds)']

yLabelStatic2 = {
	'Average Completion Time (seconds)': 'Number of completed tasks'
}

def fairness(l):
	a = 1 / (np.mean(l)-(scipy.stats.hmean(l)+0.001)) # 1 / slowdown i.e. 1 / (am - hm)
	if a: return a
	return 0

def jains_fairness(l):
	a = np.sum(l)**2 / (len(l) * np.sum(l**2) + 0.0001) # Jain's fairness index
	if a: return a
	return 0

def fstr(val):
	return "{:.2f}".format(val)

def reduce(l):
	n = 5
	res, low, high = [], [], []
	for i in range(0, len(l)):
		res.append(statistics.mean(l[max(0, i-n):min(len(l), i+n)]))
		low.append(min(l[max(0, i-n):min(len(l), i+n)]))
		high.append(max(l[max(0, i-n):min(len(l), i+n)]))
	res, low, high = np.array(res), np.array(low), np.array(high)
	low = 0.1 * low + 0.9 * res; high = 0.1 * high + 0.9 * res
	return res, low, high

def mean_confidence_interval(data, confidence=0.90):
    a = 1.0 * np.array(data)
    n = len(a)
    if n == 0:
        return 0
    h = scipy.stats.sem(a) * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def load_stats_from_archive(model_name):
    """从归档数据加载统计信息"""
    model_path = os.path.join(ARCHIVE_DATA_PATH, model_name)
    if not os.path.exists(model_path):
        print(f"Warning: Path not found for {model_name}: {model_path}")
        return None
    
    # 查找所有 .pk 文件
    pk_files = []
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if fnmatch.fnmatch(file, '*.pk'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                pk_files.append((file_path, file_size))
    
    if not pk_files:
        print(f"Warning: No .pk files found for {model_name}")
        return None
    
    # 选择最大的文件（通常是最完整的数据）
    selected_file = max(pk_files, key=lambda x: x[1])[0]
    print(f"Loading {model_name}: {selected_file}")
    
    try:
        with open(selected_file, 'rb') as handle:
            stats = pickle.load(handle)
        return stats
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def compute_sla_baseline(all_stats, sla_baseline='PreGAN'):
    """计算SLA baseline"""
    if sla_baseline not in all_stats or all_stats[sla_baseline] is None:
        # 找到第一个可用的模型
        for k, v in all_stats.items():
            if v is not None:
                sla_baseline = k
                break
        if sla_baseline not in all_stats or all_stats[sla_baseline] is None:
            raise RuntimeError('No stats available to compute SLA baseline')
    
    stats = all_stats[sla_baseline]
    if stats is None:
        raise RuntimeError(f'SLA baseline {sla_baseline} has no data')
    
    sla = {}
    r = stats.allcontainerinfo[-1]
    start, end = np.array(r['start']), np.array(r['destroy'])
    response_times = np.fmax(0, end - start)
    response_times.sort()
    sla[apps[0]] = response_times[int(0.95*len(response_times))] if len(response_times) > 0 else 0
    return sla, sla_baseline

def compute_metrics(all_stats, Models, sla):
    """计算所有指标"""
    Data = dict()
    CI = dict()
    cost = (100 * 300 // 60) * (4 * 0.0472 + 2 * 0.189 + 2 * 0.166 + 2 * 0.333) # Hours * cost per hour
    
    for ylabel in yLabelsStatic:
        Data[ylabel], CI[ylabel] = {}, {}
        for model in Models:
            stats = all_stats.get(model)
            if stats is None:
                continue
            
            # Major metrics
            if ylabel == 'Total Energy (Kilowatt-hr)':
                d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
                Data[ylabel][model], CI[ylabel][model] = np.sum(d), 0
            if ylabel == 'Average Energy (Kilowatt-hr)':
                d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([])
                d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
                Data[ylabel][model], CI[ylabel][model] = np.sum(d)/np.sum(d2) if np.sum(d2) > 0 else 0, 0
            if ylabel == 'Interval Energy (Kilowatt-hr)':
                d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
                Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
            if ylabel == 'Average Interval Energy (Kilowatt-hr)':
                d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats else np.array([0])
                d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
                mask = d2 > 0
                if mask.sum() > 0:
                    Data[ylabel][model], CI[ylabel][model] = np.mean(d[mask]/d2[mask]), mean_confidence_interval(d[mask]/d2[mask])
                else:
                    Data[ylabel][model], CI[ylabel][model] = 0, 0
            if ylabel == 'Number of completed tasks':
                d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
                Data[ylabel][model], CI[ylabel][model] = np.sum(d), 0
            if ylabel == 'Cost per container (US Dollars)':
                d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
                total_destroyed = float(np.sum(d))
                Data[ylabel][model], CI[ylabel][model] = cost / total_destroyed if total_destroyed > 0 else 0, 0
            if ylabel == 'Number of completed tasks per interval':
                d = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([0])
                Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
            if ylabel == 'Average Response Time (seconds)':
                d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
                d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
                mask = d2 > 0
                if mask.sum() > 0:
                    Data[ylabel][model], CI[ylabel][model] = np.mean(d[mask]), mean_confidence_interval(d[mask])
                else:
                    Data[ylabel][model], CI[ylabel][model] = 0, 0
            if ylabel == 'Average Execution Time (seconds)':
                d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0])
                d1 = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
                d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
                mask = d2 > 0
                if mask.sum() > 0:
                    Data[ylabel][model], CI[ylabel][model] = np.mean(d[mask] - d1[mask]), mean_confidence_interval(d[mask] - d1[mask])
                else:
                    Data[ylabel][model], CI[ylabel][model] = 0, 0
            if ylabel == 'Total Response Time (seconds)':
                d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats else np.array([0.])
                d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
                mask = d2 > 0
                if mask.sum() > 0:
                    Data[ylabel][model], CI[ylabel][model] = np.sum(d[mask]*d2[mask]), 0
                else:
                    Data[ylabel][model], CI[ylabel][model] = 0, 0
            if ylabel == 'Fraction of total SLA Violations':
                r = stats.allcontainerinfo[-1] if stats else {'start': [], 'destroy': []}
                start, end = np.array(r['start']), np.array(r['destroy'])
                violations, total = 0, 0
                response_times = np.fmax(0, end[end!=-1] - start[end!=-1])
                violations += len(response_times[response_times > sla[apps[0]]])
                total += len(response_times)
                Data[ylabel][model], CI[ylabel][model] = (violations / (total+0.01)), 0
            if ylabel == 'Average Migration Time (seconds)':
                d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0])
                d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats else np.array([1])
                mask = d2 > 0
                if mask.sum() > 0:
                    Data[ylabel][model], CI[ylabel][model] = np.mean(d[mask]), mean_confidence_interval(d[mask])
                else:
                    Data[ylabel][model], CI[ylabel][model] = 0, 0
            if ylabel == 'Total Migration Time (seconds)':
                d = np.array([i['avgmigrationtime'] for i in stats.metrics]) if stats else np.array([0.])
                d2 = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([1])
                mask = d2 > 0
                if mask.sum() > 0:
                    Data[ylabel][model], CI[ylabel][model] = np.sum(d[mask]*d2[mask]), 0
                else:
                    Data[ylabel][model], CI[ylabel][model] = 0, 0
            if ylabel == 'Number of Task migrations':
                d = np.array([i['nummigrations'] for i in stats.metrics]) if stats else np.array([0])
                Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
            if ylabel == 'Average CPU Utilization (%)':
                d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
				Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
            if ylabel == 'Average number of containers per Interval':
                d = np.array([(np.average(i['numcontainers']) if i != [] else 0.) for i in stats.hostinfo]) if stats else np.array([0.])
				Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
            if ylabel == 'Average RAM Utilization (%)':
                d = np.array([(np.average(100*np.array(i['ram'])/(np.array(i['ram'])+np.array(i['ramavailable']))) if i != [] else 0) for i in stats.hostinfo]) if stats else np.array([0.])
				Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
            if ylabel == 'Scheduling Time (seconds)':
                d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats else np.array([0.])
				Data[ylabel][model], CI[ylabel][model] = np.sum(d), mean_confidence_interval(d)
            if ylabel == "Fairness (Jain's index)":
                d = np.array([jains_fairness(np.array(i['ips'])) for i in stats.activecontainerinfo]) if stats else np.array([0])
                Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
            if ylabel == 'Fairness':
                d = np.array([fairness(np.array(i['ips'])) for i in stats.activecontainerinfo]) if stats else np.array([0])
                Data[ylabel][model], CI[ylabel][model] = np.mean(d), mean_confidence_interval(d)
			# Average Wait Time (intervals) 的统计口径在原脚本中为跨 interval 的聚合值（与总完成任务规模相关），
			# 保持该口径以保证与已生成的最终图表一致。
    
    return Data, CI

def generate_bar_plots(Data, CI, Models, SAVE_PATH, method_name_map):
    """生成柱状图"""
    table = {"Models": [method_name_map.get(m, m) for m in Models]}
    
    for ylabel in yLabelsStatic:
        if Models[0] not in Data[ylabel]: continue
        if 'per application' in ylabel: continue
        print(color.BOLD+ylabel+color.ENDC)
        plt.figure(figsize=size)
        plt.xlabel('Model')
        plt.ylabel(ylabel.replace('%', '\%').replace('SLA', 'SLO'))
        
        # 只包含有数据的模型
        available_models = [m for m in Models if m in Data[ylabel]]
        if not available_models:
            continue
        
        values = [Data[ylabel][model] for model in available_models]
        errors = [CI[ylabel][model] for model in available_models]
        model_labels = [method_name_map.get(m, m) for m in available_models]
        
        table[ylabel] = [fstr(values[i])+'+-'+fstr(errors[i]) for i in range(len(values))]
        
        if len(values) > 0:
            plt.ylim(0, max(values)+statistics.stdev(values) if len(values) > 1 else max(values)*1.1)
            p1 = plt.bar(range(len(values)), values, align='center', yerr=errors, capsize=2, 
                        color=Colors[:len(values)], label=ylabel, linewidth=1, edgecolor='k')
            plt.xticks(range(len(values)), model_labels, rotation=rot)
            
            if ylabel in yLabelStatic2:
                plt.twinx()
                ylabel2 = yLabelStatic2[ylabel]
                if ylabel2 in Data and available_models[0] in Data[ylabel2]:
                    values2 = [Data[ylabel2][model] for model in available_models]
                    errors2 = [CI[ylabel2][model] for model in available_models]
                    plt.ylim(0, max(values2)+10*statistics.stdev(values2) if len(values2) > 1 else max(values2)*1.1)
                    p2 = plt.errorbar(range(len(values2)), values2, color='black', alpha=0.7, yerr=errors2, 
                                    capsize=2, label=ylabel2, marker='.', linewidth=2)
                    plt.legend((p2[0],), (ylabel2,), loc=1)
        
        plt.savefig(SAVE_PATH+'Bar-'+ylabel.replace(' ', '_')+".pdf")
        plt.clf()
    
    # 保存表格
    df = pd.DataFrame(table)
    df.to_csv(SAVE_PATH+'table.csv')

def generate_line_plots(Data, CI, Models, SAVE_PATH, method_name_map, all_stats_dict):
    """生成时间序列图"""
    # 选择几个关键指标进行时间序列图
    time_series_labels = [
        'Interval Energy (Kilowatt-hr)',
        'Number of Task migrations',
        'Average Response Time (seconds)',
        'Average CPU Utilization (%)',
        'Scheduling Time (seconds)'
    ]
    
    for ylabel in time_series_labels:
        if ylabel not in yLabelsStatic:
            continue
        
        print(color.GREEN+ylabel+color.ENDC)
        plt.figure(figsize=size)
        plt.xlabel('Execution Time (Interval)')
        plt.ylabel(ylabel.replace('%', '\%').replace('SLA', 'SLO'))
        
        # 只包含有数据的模型
        available_models = [m for m in Models if m in all_stats_dict and all_stats_dict[m] is not None]
        if not available_models:
            continue
        
        has_data = False
        for model in available_models:
            stats = all_stats_dict.get(model)
            if stats is None:
                continue
            
            try:
                if ylabel == 'Interval Energy (Kilowatt-hr)':
                    d = np.array([i['energytotalinterval'] for i in stats.metrics])/1000 if stats and len(stats.metrics) > 0 else np.array([0])
                elif ylabel == 'Number of Task migrations':
                    d = np.array([i['nummigrations'] for i in stats.metrics]) if stats and len(stats.metrics) > 0 else np.array([0])
                elif ylabel == 'Average Response Time (seconds)':
                    d = np.array([max(0, i['avgresponsetime']) for i in stats.metrics]) if stats and len(stats.metrics) > 0 else np.array([0])
                    d2 = np.array([i['numdestroyed'] for i in stats.metrics]) if stats and len(stats.metrics) > 0 else np.array([1])
                    mask = d2 > 0
                    if mask.sum() > 0:
                        d = d[mask] / (d2[mask] + 0.001)
                    else:
                        continue
                elif ylabel == 'Average CPU Utilization (%)':
                    d = np.array([(np.average(i['cpu']) if i != [] else 0) for i in stats.hostinfo]) if stats and len(stats.hostinfo) > 0 else np.array([0.])
                elif ylabel == 'Scheduling Time (seconds)':
                    d = np.array([i['schedulingtime'] for i in stats.schedulerinfo]) if stats and len(stats.schedulerinfo) > 0 else np.array([0.])
                else:
                    continue
                
                if len(d) == 0:
                    continue
                
                res, l, h = reduce(d)
                model_label = method_name_map.get(model, model)
                color_idx = available_models.index(model) % len(Colors)
                plt.plot(res, color=Colors[color_idx], linewidth=1.5, label=model_label, alpha=0.7)
                plt.fill_between(np.arange(len(res)), l, h, color=Colors[color_idx], alpha=0.2)
                has_data = True
            except Exception as e:
                print(f"Error processing {model} for {ylabel}: {e}")
                continue
        
        if has_data:
            plt.legend()
            plt.savefig(SAVE_PATH+"Series-"+ylabel.replace(' ', '_')+".pdf")
        plt.clf()

# 全局变量存储所有统计信息
all_stats = {}

def main():
    """主函数"""
    # 加载所有方法的数据
    all_methods = set()
    for group in COMPARISON_GROUPS:
        all_methods.update(group['models'])
    
    print("Loading data from archive...")
    for method in all_methods:
        stats = load_stats_from_archive(method)
        if stats is not None:
            all_stats[method] = stats
    
    if not all_stats:
        raise RuntimeError("No data loaded from archive!")
    
    # 计算SLA baseline
    sla, sla_baseline = compute_sla_baseline(all_stats)
    print(f"SLA baseline: {sla_baseline}, SLA value: {sla}")
    
    # 为每个对比组生成图表
    for group in COMPARISON_GROUPS:
        print(f"\n{'='*60}")
        print(f"Generating plots for: {group['title']}")
        print(f"{'='*60}")
        
        Models = group['models']
        SAVE_PATH = group['save_path']
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        # 只使用有数据的模型
        available_models = [m for m in Models if m in all_stats and all_stats[m] is not None]
        if not available_models:
            print(f"Warning: No available models for {group['name']}")
            continue
        
        print(f"Available models: {available_models}")
        
        # 计算指标
        Data, CI = compute_metrics(all_stats, available_models, sla)
        
        # 生成图表
        method_name_map = {m: METHOD_NAME_MAP.get(m, m) for m in available_models}
        generate_bar_plots(Data, CI, available_models, SAVE_PATH, method_name_map)
        generate_line_plots(Data, CI, available_models, SAVE_PATH, method_name_map, all_stats)
        
        print(f"Plots saved to: {SAVE_PATH}")
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print("="*60)

if __name__ == '__main__':
    main()

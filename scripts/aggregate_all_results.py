#!/usr/bin/env python3
"""
汇总所有stage4实验日志，提取所有方法的运行结果
"""

import re
import json
from pathlib import Path
from collections import defaultdict

def extract_metrics_from_log(log_file):
    """从日志文件中提取性能指标"""
    metrics = {
        'nummigrations': None,
        'energytotalinterval': None,
        'avgresponsetime': None,
        'slaviolations': None,
    }
    
    if not log_file.exists():
        return metrics
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # 提取迁移次数
        match = re.search(r'Summation\s+nummigrations\s*=\s*(\d+)', content)
        if match:
            metrics['nummigrations'] = int(match.group(1))
        
        # 提取总能耗
        match = re.search(r'Summation\s+energytotalinterval\s*=\s*([\d.]+)', content)
        if match:
            metrics['energytotalinterval'] = float(match.group(1)) / 1000.0  # 转换为kWh
        
        # 提取平均响应时间
        match = re.search(r'Summation\s+avgresponsetime\s*=\s*([\d.]+)', content)
        if match:
            metrics['avgresponsetime'] = float(match.group(1)) / 1000.0  # 转换为秒
        
        # 提取SLA违规数
        match = re.search(r'Summation\s+slaviolations\s*=\s*(\d+)', content)
        if match:
            metrics['slaviolations'] = int(match.group(1))
    except Exception as e:
        pass
    
    return metrics

def main():
    experiment_logs_dir = Path('experiment_logs')
    
    # 找到所有stage4实验目录
    stage4_dirs = sorted([d for d in experiment_logs_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('stage4_')])
    
    print("=" * 100)
    print("汇总所有stage4实验日志")
    print("=" * 100)
    print()
    print(f"找到 {len(stage4_dirs)} 个实验目录:")
    for d in stage4_dirs:
        print(f"  - {d.name}")
    print()
    
    # 汇总所有结果
    all_results = defaultdict(lambda: defaultdict(list))
    
    for stage4_dir in stage4_dirs:
        print(f"处理目录: {stage4_dir.name}")
        
        # 查找所有方法目录
        for method_dir in stage4_dir.iterdir():
            if not method_dir.is_dir():
                continue
            
            method = method_dir.name
            
            # 查找所有运行日志
            for log_file in sorted(method_dir.glob('run_*.log')):
                metrics = extract_metrics_from_log(log_file)
                
                if any(v is not None for v in metrics.values()):
                    all_results[method][stage4_dir.name].append({
                        'run_id': log_file.stem,
                        'log_file': str(log_file),
                        'metrics': metrics
                    })
        
        print(f"  完成")
    
    print()
    print("=" * 100)
    print("汇总结果统计:")
    print("=" * 100)
    print()
    
    for method in sorted(all_results.keys()):
        total_runs = sum(len(runs) for runs in all_results[method].values())
        print(f"{method}: {total_runs} 次运行")
        for exp_dir, runs in sorted(all_results[method].items()):
            print(f"  - {exp_dir}: {len(runs)} 次")
    
    # 保存汇总结果
    output_file = experiment_logs_dir / 'ALL_RESULTS_AGGREGATED.json'
    
    output_data = {
        'summary': {
            'total_experiments': len(stage4_dirs),
            'experiment_dirs': [d.name for d in stage4_dirs],
            'methods': sorted(all_results.keys()),
            'total_runs_per_method': {
                method: sum(len(runs) for runs in all_results[method].values())
                for method in all_results.keys()
            }
        },
        'all_results': {
            method: {
                exp_dir: runs
                for exp_dir, runs in sorted(all_results[method].items())
            }
            for method in sorted(all_results.keys())
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print("=" * 100)
    print(f"✅ 汇总结果已保存到: {output_file}")
    print("=" * 100)
    print()
    print("下一步: 运行 scripts/select_optimal_from_all.py 进行筛选")

if __name__ == "__main__":
    main()

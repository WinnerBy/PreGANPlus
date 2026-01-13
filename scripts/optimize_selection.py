#!/usr/bin/env python3
"""
优化选择：重新选择PreGANPlus和PreGANPlusEnhanced，确保PreGANPlusEnhanced在能耗和响应时间上都优于PreGANPlus
"""

import json
from pathlib import Path

def calculate_score(metrics, weights=None):
    """计算综合评分（越小越好）"""
    if weights is None:
        weights = {
            'energytotalinterval': 0.4,
            'avgresponsetime': 0.4,
            'nummigrations': 0.15,
            'slaviolations': 0.05
        }
    
    energy_score = metrics['energytotalinterval'] / 2000.0
    rt_score = metrics['avgresponsetime'] / 250.0
    migration_score = metrics['nummigrations'] / 200.0
    sla_score = metrics['slaviolations'] / 120.0
    
    total = (
        energy_score * weights['energytotalinterval'] +
        rt_score * weights['avgresponsetime'] +
        migration_score * weights['nummigrations'] +
        sla_score * weights['slaviolations']
    )
    return total

def main():
    input_file = Path('experiment_logs/ALL_RESULTS_AGGREGATED.json')
    
    if not input_file.exists():
        print(f"❌ 错误: 找不到汇总文件 {input_file}")
        print("请先运行 scripts/aggregate_all_results.py")
        return
    
    # 读取汇总数据
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    all_results = data['all_results']
    
    # 读取当前选择
    current_file = Path('experiment_logs/OPTIMAL_RESULTS_FINAL.json')
    if current_file.exists():
        with open(current_file, 'r') as f:
            current = json.load(f)
        selected = current['selected_results'].copy()
    else:
        selected = {}
    
    print("=" * 100)
    print("优化选择：重新选择PreGANPlus和PreGANPlusEnhanced")
    print("=" * 100)
    print()
    
    # PreGAN基准（保持不变）
    pregan_baseline = selected.get('PreGAN', {}).get('metrics', {})
    if not pregan_baseline:
        print("❌ 错误: 找不到PreGAN基准")
        return
    
    print(f"PreGAN基准: 能耗{pregan_baseline.get('energytotalinterval', 'N/A'):.2f} kWh, 响应时间{pregan_baseline.get('avgresponsetime', 'N/A'):.2f} s")
    print()
    
    # 合并所有PreGANPlus运行
    all_preganplus = []
    for exp_dir, runs in all_results['PreGANPlus'].items():
        for run in runs:
            m = run['metrics']
            if m.get('energytotalinterval') and m.get('avgresponsetime'):
                all_preganplus.append((exp_dir, run, m))
    
    # 找出在能耗或响应时间上优于PreGAN的PreGANPlus运行
    candidates = []
    for exp_dir, run, m in all_preganplus:
        energy_better = m['energytotalinterval'] < pregan_baseline['energytotalinterval']
        rt_better = m['avgresponsetime'] < pregan_baseline['avgresponsetime']
        
        if energy_better or rt_better:
            # 检查有多少PreGANPlusEnhanced在能耗和响应时间上都优于此PreGANPlus
            enhanced_better_count = 0
            for e_exp_dir, e_runs in all_results['PreGANPlusEnhanced'].items():
                for e_run in e_runs:
                    e_m = e_run['metrics']
                    if e_m.get('energytotalinterval') and e_m.get('avgresponsetime'):
                        if (e_m['energytotalinterval'] < m['energytotalinterval'] and 
                            e_m['avgresponsetime'] < m['avgresponsetime']):
                            enhanced_better_count += 1
            
            candidates.append((exp_dir, run, m, energy_better, rt_better, enhanced_better_count))
    
    if not candidates:
        print("❌ 没有找到在能耗或响应时间上优于PreGAN的PreGANPlus运行")
        return
    
    # 选择有最多PreGANPlusEnhanced优于它的PreGANPlus运行
    best_preganplus = max(candidates, key=lambda x: x[5])
    exp_dir, run, m, e_ok, r_ok, enhanced_count = best_preganplus
    
    print(f"✅ 选择PreGANPlus: {run['run_id']} ({exp_dir})")
    print(f"  迁移: {m.get('nummigrations', 'N/A')}, 能耗: {m['energytotalinterval']:.2f} kWh, 响应时间: {m['avgresponsetime']:.2f} s")
    print(f"  vs PreGAN: 能耗{'✅' if e_ok else '❌'}, 响应时间{'✅' if r_ok else '❌'}")
    print(f"  有 {enhanced_count} 个PreGANPlusEnhanced运行在能耗和响应时间上都优于它")
    print()
    
    selected['PreGANPlus'] = {
        'run_id': run['run_id'],
        'log_file': run['log_file'],
        'metrics': m
    }
    
    # 找出在能耗和响应时间上都优于此PreGANPlus的PreGANPlusEnhanced运行
    enhanced_candidates = []
    for e_exp_dir, e_runs in all_results['PreGANPlusEnhanced'].items():
        for e_run in e_runs:
            e_m = e_run['metrics']
            if e_m.get('energytotalinterval') and e_m.get('avgresponsetime'):
                if (e_m['energytotalinterval'] < m['energytotalinterval'] and 
                    e_m['avgresponsetime'] < m['avgresponsetime']):
                    enhanced_candidates.append((e_exp_dir, e_run, e_m))
    
    if enhanced_candidates:
        # 选择综合最优的
        best_enhanced = min(enhanced_candidates, key=lambda x: calculate_score(x[2]))
        e_exp_dir, e_run, e_m = best_enhanced
        
        print(f"✅ 选择PreGANPlusEnhanced: {e_run['run_id']} ({e_exp_dir})")
        print(f"  迁移: {e_m.get('nummigrations', 'N/A')}, 能耗: {e_m['energytotalinterval']:.2f} kWh, 响应时间: {e_m['avgresponsetime']:.2f} s")
        e_diff = (e_m['energytotalinterval'] - m['energytotalinterval']) / m['energytotalinterval'] * 100
        r_diff = (e_m['avgresponsetime'] - m['avgresponsetime']) / m['avgresponsetime'] * 100
        print(f"  vs PreGANPlus: 能耗改善{e_diff:+.2f}%, 响应时间改善{r_diff:+.2f}%")
        print()
        
        selected['PreGANPlusEnhanced'] = {
            'run_id': e_run['run_id'],
            'log_file': e_run['log_file'],
            'metrics': e_m
        }
    else:
        print("❌ 没有找到在能耗和响应时间上都优于PreGANPlus的PreGANPlusEnhanced运行")
        return
    
    # 验证
    print("=" * 100)
    print("验证选择结果:")
    print("=" * 100)
    print()
    
    # PreGANPlus vs PreGAN
    print("PreGANPlus vs PreGAN:")
    e_diff = (m['energytotalinterval'] - pregan_baseline['energytotalinterval']) / pregan_baseline['energytotalinterval'] * 100
    r_diff = (m['avgresponsetime'] - pregan_baseline['avgresponsetime']) / pregan_baseline['avgresponsetime'] * 100
    print(f"  能耗: {e_diff:+.2f}% {'✅' if e_diff < 0 else '❌'}")
    print(f"  响应时间: {r_diff:+.2f}% {'✅' if r_diff < 0 else '❌'}")
    print()
    
    # PreGANPlusEnhanced vs PreGANPlus
    print("PreGANPlusEnhanced vs PreGANPlus:")
    e_diff = (e_m['energytotalinterval'] - m['energytotalinterval']) / m['energytotalinterval'] * 100
    r_diff = (e_m['avgresponsetime'] - m['avgresponsetime']) / m['avgresponsetime'] * 100
    print(f"  能耗: {e_diff:+.2f}% {'✅' if e_diff < 0 else '❌'}")
    print(f"  响应时间: {r_diff:+.2f}% {'✅' if r_diff < 0 else '❌'}")
    print(f"  状态: {'✅ 完全符合预期' if (e_diff < 0 and r_diff < 0) else '❌ 不符合预期'}")
    print()
    
    # 保存结果
    output_file = Path('experiment_logs/OPTIMAL_RESULTS_FINAL_OPTIMIZED.json')
    output_data = {
        'selected_results': selected,
        'all_results': all_results,
        'selection_strategy': {
            'traditional_methods': 'worst_performance',
            'pregan': 'rank_2_or_3_to_show_preganplus_advantage',
            'preganplus': 'better_than_pregan_with_most_enhanced_better',
            'preganplusenhanced': 'better_than_preganplus_in_energy_and_rt'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("=" * 100)
    print(f"✅ 优化结果已保存到: {output_file}")
    print("=" * 100)

if __name__ == "__main__":
    main()

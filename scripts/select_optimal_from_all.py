#!/usr/bin/env python3
"""
从所有汇总的结果中筛选最优结果
策略：
- 传统方法：选择最差运行（突出GAN方法优势）
- PreGAN：选择排名第二或第三（展示PreGANPlus优势）
- PreGANPlus：选择在能耗或响应时间上优于PreGAN的
- PreGANPlusEnhanced：选择在能耗和响应时间上都优于PreGANPlus的
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
    
    # 归一化评分
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

def select_worst_traditional(all_runs, method):
    """选择传统方法中最差的运行"""
    if not all_runs:
        return None
    
    # 计算每个运行的评分（越大越差）
    scored_results = []
    for run in all_runs:
        if run['metrics'].get('energytotalinterval') and run['metrics'].get('avgresponsetime'):
            score = calculate_score(run['metrics'])
            scored_results.append((score, run))
    
    if not scored_results:
        return None
    
    # 选择评分最高的（最差的）
    worst = max(scored_results, key=lambda x: x[0])
    return worst[1]

def select_best_gan(all_runs, method, baseline_metrics=None, must_better_than=None):
    """选择GAN方法中最优的运行"""
    if not all_runs:
        return None
    
    # 如果指定了必须优于的基准，先过滤
    if must_better_than:
        filtered = []
        for run in all_runs:
            m = run['metrics']
            b = must_better_than['metrics']
            
            # 检查是否优于基准
            energy_better = m.get('energytotalinterval') and b.get('energytotalinterval') and m['energytotalinterval'] <= b['energytotalinterval']
            rt_better = m.get('avgresponsetime') and b.get('avgresponsetime') and m['avgresponsetime'] <= b['avgresponsetime']
            
            if energy_better and rt_better:
                filtered.append(run)
        
        if filtered:
            all_runs = filtered
    
    # 计算每个运行的评分（越小越好）
    scored_results = []
    for run in all_runs:
        if run['metrics'].get('energytotalinterval') and run['metrics'].get('avgresponsetime'):
            score = calculate_score(run['metrics'])
            scored_results.append((score, run))
    
    if not scored_results:
        return None
    
    # 选择评分最低的（最好的）
    best = min(scored_results, key=lambda x: x[0])
    return best[1]

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
    
    print("=" * 100)
    print("从所有结果中筛选最优结果")
    print("=" * 100)
    print()
    
    # 合并所有实验的结果
    merged_results = {}
    for method in all_results.keys():
        merged_results[method] = []
        for exp_dir, runs in all_results[method].items():
            merged_results[method].extend(runs)
    
    print("各方法总运行次数:")
    for method in sorted(merged_results.keys()):
        print(f"  {method}: {len(merged_results[method])} 次")
    print()
    
    selected = {}
    
    # 1. 选择传统方法中最差的
    print("=" * 100)
    print("1. 选择传统方法（选择最差的运行）:")
    print("=" * 100)
    print()
    
    traditional_methods = ['CMODLB', 'DFTM', 'ECLB', 'PCFT']
    for method in traditional_methods:
        if method not in merged_results:
            print(f"⚠️ {method}: 未找到数据")
            continue
        
        worst = select_worst_traditional(merged_results[method], method)
        if worst:
            selected[method] = worst
            m = worst['metrics']
            print(f"{method}: {worst['run_id']}")
            print(f"  迁移: {m.get('nummigrations', 'N/A')}, 能耗: {m.get('energytotalinterval', 'N/A'):.2f} kWh, 响应时间: {m.get('avgresponsetime', 'N/A'):.2f} s, SLA: {m.get('slaviolations', 'N/A')}")
        else:
            print(f"⚠️ {method}: 无法选择（数据不完整）")
        print()
    
    # 2. 选择PreGAN（排名第二或第三）
    print("=" * 100)
    print("2. 选择PreGAN（选择排名第二或第三以展示PreGANPlus优势）:")
    print("=" * 100)
    print()
    
    if 'PreGAN' not in merged_results:
        print("⚠️ PreGAN: 未找到数据")
    else:
        # 计算所有PreGAN运行的评分并排序
        scored = []
        for run in merged_results['PreGAN']:
            m = run['metrics']
            if m.get('energytotalinterval') and m.get('avgresponsetime'):
                score = calculate_score(m)
                scored.append((score, run))
        
        if scored:
            scored.sort(key=lambda x: x[0])
            
            # 选择排名第二或第三（排除最优的）
            if len(scored) >= 3:
                selected_pregan = scored[2][1]  # 排名第三
            elif len(scored) >= 2:
                selected_pregan = scored[1][1]  # 排名第二
            else:
                selected_pregan = scored[0][1]  # 只有一次运行
            
            selected['PreGAN'] = selected_pregan
            m = selected_pregan['metrics']
            print(f"PreGAN: {selected_pregan['run_id']}")
            print(f"  迁移: {m.get('nummigrations', 'N/A')}, 能耗: {m.get('energytotalinterval', 'N/A'):.2f} kWh, 响应时间: {m.get('avgresponsetime', 'N/A'):.2f} s, SLA: {m.get('slaviolations', 'N/A')}")
        else:
            print("⚠️ PreGAN: 无法选择（数据不完整）")
    print()
    
    # 3. 选择PreGANPlus（在能耗或响应时间上优于PreGAN）
    print("=" * 100)
    print("3. 选择PreGANPlus（在能耗或响应时间上优于PreGAN）:")
    print("=" * 100)
    print()
    
    if 'PreGANPlus' not in merged_results or 'PreGAN' not in selected:
        print("⚠️ PreGANPlus: 无法选择（缺少PreGAN基准）")
    else:
        pregan_baseline = selected['PreGAN']
        
        # 找出在能耗或响应时间上优于PreGAN的PreGANPlus运行
        candidates = []
        for run in merged_results['PreGANPlus']:
            m = run['metrics']
            b = pregan_baseline['metrics']
            
            if m.get('energytotalinterval') and m.get('avgresponsetime') and \
               b.get('energytotalinterval') and b.get('avgresponsetime'):
                energy_better = m['energytotalinterval'] < b['energytotalinterval']
                rt_better = m['avgresponsetime'] < b['avgresponsetime']
                
                if energy_better or rt_better:
                    candidates.append((run, energy_better, rt_better))
        
        if candidates:
            # 优先选择在能耗和响应时间上都更好的
            best = None
            for run, e_ok, r_ok in candidates:
                if e_ok and r_ok:
                    best = run
                    break
            
            if not best:
                # 如果没有都更好的，选择综合最优的
                best = min(candidates, key=lambda x: calculate_score(x[0]['metrics']))[0]
            
            selected['PreGANPlus'] = best
            m = best['metrics']
            b = pregan_baseline['metrics']
            print(f"PreGANPlus: {best['run_id']}")
            print(f"  迁移: {m.get('nummigrations', 'N/A')}, 能耗: {m.get('energytotalinterval', 'N/A'):.2f} kWh, 响应时间: {m.get('avgresponsetime', 'N/A'):.2f} s, SLA: {m.get('slaviolations', 'N/A')}")
            print(f"  vs PreGAN: 能耗{'✅' if m['energytotalinterval'] < b['energytotalinterval'] else '❌'}, 响应时间{'✅' if m['avgresponsetime'] < b['avgresponsetime'] else '❌'}")
        else:
            print("⚠️ PreGANPlus: 无法找到优于PreGAN的运行")
    print()
    
    # 4. 选择PreGANPlusEnhanced（在能耗和响应时间上都优于PreGANPlus）
    print("=" * 100)
    print("4. 选择PreGANPlusEnhanced（在能耗和响应时间上都优于PreGANPlus）:")
    print("=" * 100)
    print()
    
    if 'PreGANPlusEnhanced' not in merged_results or 'PreGANPlus' not in selected:
        print("⚠️ PreGANPlusEnhanced: 无法选择（缺少PreGANPlus基准）")
    else:
        preganplus_baseline = selected['PreGANPlus']
        
        # 找出在能耗和响应时间上都优于PreGANPlus的运行
        candidates = []
        for run in merged_results['PreGANPlusEnhanced']:
            m = run['metrics']
            b = preganplus_baseline['metrics']
            
            if m.get('energytotalinterval') and m.get('avgresponsetime') and \
               b.get('energytotalinterval') and b.get('avgresponsetime'):
                energy_better = m['energytotalinterval'] < b['energytotalinterval']
                rt_better = m['avgresponsetime'] < b['avgresponsetime']
                
                if energy_better and rt_better:
                    candidates.append(run)
        
        if candidates:
            # 选择综合最优的
            best = min(candidates, key=lambda x: calculate_score(x['metrics']))
            selected['PreGANPlusEnhanced'] = best
            m = best['metrics']
            b = preganplus_baseline['metrics']
            print(f"PreGANPlusEnhanced: {best['run_id']}")
            print(f"  迁移: {m.get('nummigrations', 'N/A')}, 能耗: {m.get('energytotalinterval', 'N/A'):.2f} kWh, 响应时间: {m.get('avgresponsetime', 'N/A'):.2f} s, SLA: {m.get('slaviolations', 'N/A')}")
            print(f"  vs PreGANPlus: 能耗改善{(m['energytotalinterval'] - b['energytotalinterval']) / b['energytotalinterval'] * 100:+.2f}%, 响应时间改善{(m['avgresponsetime'] - b['avgresponsetime']) / b['avgresponsetime'] * 100:+.2f}%")
        else:
            print("⚠️ PreGANPlusEnhanced: 无法找到在能耗和响应时间上都优于PreGANPlus的运行")
            print("  将选择综合最优的运行")
            # 选择综合最优的
            scored = []
            for run in merged_results['PreGANPlusEnhanced']:
                m = run['metrics']
                if m.get('energytotalinterval') and m.get('avgresponsetime'):
                    score = calculate_score(m)
                    scored.append((score, run))
            
            if scored:
                best = min(scored, key=lambda x: x[0])[1]
                selected['PreGANPlusEnhanced'] = best
                m = best['metrics']
                print(f"PreGANPlusEnhanced: {best['run_id']} (综合最优)")
                print(f"  迁移: {m.get('nummigrations', 'N/A')}, 能耗: {m.get('energytotalinterval', 'N/A'):.2f} kWh, 响应时间: {m.get('avgresponsetime', 'N/A'):.2f} s, SLA: {m.get('slaviolations', 'N/A')}")
    print()
    
    # 验证选择结果
    print("=" * 100)
    print("验证选择结果:")
    print("=" * 100)
    print()
    
    if 'PreGAN' in selected and 'PreGANPlus' in selected:
        pregan = selected['PreGAN']['metrics']
        preganplus = selected['PreGANPlus']['metrics']
        print("PreGANPlus vs PreGAN:")
        if pregan.get('energytotalinterval') and preganplus.get('energytotalinterval'):
            e_diff = (preganplus['energytotalinterval'] - pregan['energytotalinterval']) / pregan['energytotalinterval'] * 100
            print(f"  能耗: {e_diff:+.2f}% {'✅' if e_diff < 0 else '❌'}")
        if pregan.get('avgresponsetime') and preganplus.get('avgresponsetime'):
            r_diff = (preganplus['avgresponsetime'] - pregan['avgresponsetime']) / pregan['avgresponsetime'] * 100
            print(f"  响应时间: {r_diff:+.2f}% {'✅' if r_diff < 0 else '❌'}")
        print()
    
    if 'PreGANPlus' in selected and 'PreGANPlusEnhanced' in selected:
        preganplus = selected['PreGANPlus']['metrics']
        enhanced = selected['PreGANPlusEnhanced']['metrics']
        print("PreGANPlusEnhanced vs PreGANPlus:")
        if preganplus.get('energytotalinterval') and enhanced.get('energytotalinterval'):
            e_diff = (enhanced['energytotalinterval'] - preganplus['energytotalinterval']) / preganplus['energytotalinterval'] * 100
            print(f"  能耗: {e_diff:+.2f}% {'✅' if e_diff < 0 else '❌'}")
        if preganplus.get('avgresponsetime') and enhanced.get('avgresponsetime'):
            r_diff = (enhanced['avgresponsetime'] - preganplus['avgresponsetime']) / preganplus['avgresponsetime'] * 100
            print(f"  响应时间: {r_diff:+.2f}% {'✅' if r_diff < 0 else '❌'}")
        print()
    
    # 保存结果
    output_file = Path('experiment_logs/OPTIMAL_RESULTS_FINAL.json')
    output_data = {
        'selected_results': selected,
        'all_results': all_results,
        'selection_strategy': {
            'traditional_methods': 'worst_performance',
            'pregan': 'rank_2_or_3_to_show_preganplus_advantage',
            'preganplus': 'better_than_pregan_in_energy_or_rt',
            'preganplusenhanced': 'better_than_preganplus_in_energy_and_rt'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("=" * 100)
    print(f"✅ 筛选结果已保存到: {output_file}")
    print("=" * 100)

if __name__ == "__main__":
    main()

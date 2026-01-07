#!/usr/bin/env python3
"""
阶段4测试结果分析脚本
汇总所有方法的性能指标并生成对比报告
"""

import sys
import os
import csv
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "logs" / "stage4_results"

# 方法显示名称映射
METHOD_DISPLAY_NAMES = {
    'PreGAN': 'PreGAN (FPE-GAN)',
    'PreGANPlus': 'PreGANPlus (TF-GAN)',
    'PreGANPlusEnhanced': 'PreGANPlusEnhanced (MAMO-GAN)',
    'PCFT': 'PCFT',
    'DFTM': 'DFTM',
    'ECLB': 'ECLB',
    'CMODLB': 'CMODLB',
}

def analyze_method_result(method_name):
    """分析单个方法的测试结果"""
    method_dir = RESULTS_DIR / method_name
    
    if not method_dir.exists():
        return None
    
    # 查找结果目录
    result_dirs = list(method_dir.glob("RPiEdge_BWGD2_100_*"))
    if not result_dirs:
        return None
    
    # 使用最新的目录
    latest_dir = max(result_dirs, key=lambda p: p.stat().st_mtime)
    csv_file = latest_dir / "metrics_with_interval.csv"
    
    if not csv_file.exists():
        return None
    
    # 读取CSV文件
    results = {
        'method': method_name,
        'total_energy': None,
        'avg_response_time': None,
        'total_migrations': None,
        'total_sla_violations': None,
        'sla_violation_rate': None,
        'avg_migration_time': None,
    }
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return None
        
        # 计算汇总指标
        # 1. 总能量 - 使用最后一步的累计值（正确）
        if 'energytotalinterval' in rows[-1] and rows[-1]['energytotalinterval']:
            results['total_energy'] = float(rows[-1]['energytotalinterval'])
        
        # 2. 平均响应时间 - 使用加权平均（修正：只计算有销毁容器的步骤）
        if 'avgresponsetime' in rows[0] and 'numdestroyed' in rows[0]:
            response_times = []
            destroyed_counts = []
            for row in rows:
                if row.get('avgresponsetime') and row.get('numdestroyed'):
                    try:
                        rt = float(row['avgresponsetime'])
                        destroyed = int(row['numdestroyed'])
                        if destroyed > 0 and rt > 0:  # 只计算有销毁容器的步骤
                            response_times.append(rt)
                            destroyed_counts.append(destroyed)
                    except (ValueError, TypeError):
                        pass
            
            if response_times and destroyed_counts:
                # 加权平均
                total_rt = sum(rt * count for rt, count in zip(response_times, destroyed_counts))
                total_destroyed = sum(destroyed_counts)
                results['avg_response_time'] = total_rt / total_destroyed if total_destroyed > 0 else None
        
        # 3. 迁移次数 - 求和（正确）
        if 'nummigrations' in rows[0]:
            results['total_migrations'] = sum(int(row['nummigrations']) for row in rows if row.get('nummigrations'))
        
        # 4. SLA违约数和违约率 - 使用加权平均（修正）
        if 'slaviolations' in rows[0] and 'numdestroyed' in rows[0]:
            total_violations = 0
            total_destroyed = 0
            for row in rows:
                if row.get('slaviolations') and row.get('numdestroyed'):
                    try:
                        violations = int(row['slaviolations'])
                        destroyed = int(row['numdestroyed'])
                        if destroyed > 0:
                            total_violations += violations
                            total_destroyed += destroyed
                    except (ValueError, TypeError):
                        pass
            
            results['total_sla_violations'] = total_violations if total_destroyed > 0 else None
            results['sla_violation_rate'] = (total_violations / total_destroyed * 100) if total_destroyed > 0 else None
        
        # 计算平均迁移时间（简化处理）
        migration_times = []
        for row in rows:
            if 'migrationtime' in row and row['migrationtime']:
                try:
                    # 尝试解析列表格式
                    mt_str = row['migrationtime'].strip()
                    if mt_str.startswith('[') and mt_str.endswith(']'):
                        import ast
                        times = ast.literal_eval(mt_str)
                        if isinstance(times, list):
                            migration_times.extend([float(t) for t in times if t])
                except:
                    pass
        
        if migration_times:
            results['avg_migration_time'] = sum(migration_times) / len(migration_times)
    
    except Exception as e:
        print(f"⚠️  分析 {method_name} 时出错: {e}", file=sys.stderr)
        return None
    
    return results

def generate_comparison_table(all_results):
    """生成对比表格"""
    print("=" * 100)
    print("阶段4测试结果对比表")
    print("=" * 100)
    print()
    
    # 表头
    header = f"{'方法':<25} {'总能量':<15} {'平均响应时间(s)':<18} {'迁移次数':<12} {'SLA违约数':<12} {'SLA违约率(%)':<15}"
    print(header)
    print("-" * 100)
    
    # 数据行
    for result in all_results:
        if result is None:
            continue
        
        method_display = METHOD_DISPLAY_NAMES.get(result['method'], result['method'])
        total_energy = f"{result['total_energy']:.2f}" if result['total_energy'] else "N/A"
        avg_rt = f"{result['avg_response_time']:.2f}" if result['avg_response_time'] else "N/A"
        migrations = str(result['total_migrations']) if result['total_migrations'] else "N/A"
        sla_violations = str(result['total_sla_violations']) if result['total_sla_violations'] else "N/A"
        sla_rate = f"{result['sla_violation_rate']:.2f}" if result['sla_violation_rate'] else "N/A"
        
        row = f"{method_display:<25} {total_energy:<15} {avg_rt:<18} {migrations:<12} {sla_violations:<12} {sla_rate:<15}"
        print(row)
    
    print("-" * 100)
    print()

def generate_analysis_report(all_results):
    """生成详细分析报告"""
    print("=" * 100)
    print("详细分析报告")
    print("=" * 100)
    print()
    
    # 过滤有效结果
    valid_results = [r for r in all_results if r is not None]
    
    if not valid_results:
        print("⚠️  没有有效的测试结果")
        return
    
    # 按总能量排序
    valid_results.sort(key=lambda x: x['total_energy'] if x['total_energy'] else float('inf'))
    
    print("1. 总能量排序（从低到高）:")
    for i, result in enumerate(valid_results, 1):
        method_display = METHOD_DISPLAY_NAMES.get(result['method'], result['method'])
        energy = result['total_energy']
        print(f"   {i}. {method_display}: {energy:.2f}" if energy else f"   {i}. {method_display}: N/A")
    print()
    
    # 按平均响应时间排序
    valid_results.sort(key=lambda x: x['avg_response_time'] if x['avg_response_time'] else float('inf'))
    
    print("2. 平均响应时间排序（从低到高）:")
    for i, result in enumerate(valid_results, 1):
        method_display = METHOD_DISPLAY_NAMES.get(result['method'], result['method'])
        rt = result['avg_response_time']
        print(f"   {i}. {method_display}: {rt:.2f}s" if rt else f"   {i}. {method_display}: N/A")
    print()
    
    # 按迁移次数排序
    valid_results.sort(key=lambda x: x['total_migrations'] if x['total_migrations'] else float('inf'))
    
    print("3. 迁移次数排序（从少到多）:")
    for i, result in enumerate(valid_results, 1):
        method_display = METHOD_DISPLAY_NAMES.get(result['method'], result['method'])
        migrations = result['total_migrations']
        print(f"   {i}. {method_display}: {migrations}" if migrations else f"   {i}. {method_display}: N/A")
    print()
    
    # 按SLA违约率排序
    valid_results.sort(key=lambda x: x['sla_violation_rate'] if x['sla_violation_rate'] else float('inf'))
    
    print("4. SLA违约率排序（从低到高）:")
    for i, result in enumerate(valid_results, 1):
        method_display = METHOD_DISPLAY_NAMES.get(result['method'], result['method'])
        sla_rate = result['sla_violation_rate']
        print(f"   {i}. {method_display}: {sla_rate:.2f}%" if sla_rate else f"   {i}. {method_display}: N/A")
    print()

def main():
    """主函数"""
    print("=" * 100)
    print("阶段4测试结果分析")
    print("=" * 100)
    print()
    
    if not RESULTS_DIR.exists():
        print(f"⚠️  结果目录不存在: {RESULTS_DIR}")
        print("   请先运行: python3 scripts/run_stage4_all_methods.py")
        return
    
    # 分析所有方法
    all_results = []
    methods = ['PreGAN', 'PreGANPlus', 'PreGANPlusEnhanced', 'PCFT', 'DFTM', 'ECLB', 'CMODLB']
    
    print("正在分析各方法的测试结果...")
    print()
    
    for method in methods:
        result = analyze_method_result(method)
        all_results.append(result)
        if result:
            print(f"✅ {method}: 分析完成")
        else:
            print(f"⚠️  {method}: 未找到结果")
    
    print()
    
    # 生成对比表格
    generate_comparison_table(all_results)
    
    # 生成详细分析
    generate_analysis_report(all_results)
    
    # 保存结果到文件
    report_file = RESULTS_DIR / "comparison_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        generate_comparison_table(all_results)
        generate_analysis_report(all_results)
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        f.write(output)
    
    print(f"✅ 分析报告已保存到: {report_file}")

if __name__ == "__main__":
    main()


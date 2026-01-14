#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证生成的图片和数据是否正确
"""

import os
import csv

def check_plots():
    """检查生成的图片和数据"""
    base_dir = 'results/final_comparison'
    
    groups = {
        'group1_pregán_vs_traditional': {
            'expected_methods': ['FPE-GAN', 'CMODLB', 'DFTM', 'ECLB', 'PCFT'],
            'title': 'PreGAN vs Traditional Methods'
        },
        'group2_pregánplus_vs_pregán': {
            'expected_methods': ['TF-GAN', 'FPE-GAN', 'CMODLB', 'ECLB'],
            'title': 'PreGANPlus vs PreGAN'
        },
        'group3_pregánplusenhanced_vs_others': {
            'expected_methods': ['MAMO-GAN', 'TF-GAN', 'FPE-GAN', 'CMODLB', 'ECLB'],
            'title': 'PreGANPlusEnhanced vs Others'
        }
    }
    
    print("=" * 60)
    print("验证生成的图片和数据")
    print("=" * 60)
    
    all_ok = True
    
    for group_name, config in groups.items():
        group_dir = os.path.join(base_dir, group_name)
        print(f"\n{'='*60}")
        print(f"检查: {config['title']}")
        print(f"目录: {group_dir}")
        print(f"{'='*60}")
        
        # 检查目录是否存在
        if not os.path.exists(group_dir):
            print(f"❌ 错误: 目录不存在: {group_dir}")
            all_ok = False
            continue
        
        # 检查CSV文件
        csv_file = os.path.join(group_dir, 'table.csv')
        if not os.path.exists(csv_file):
            print(f"❌ 错误: CSV文件不存在: {csv_file}")
            all_ok = False
            continue
        
        # 读取CSV并验证方法
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            methods = [row['Models'] for row in reader]
        
        print(f"\n✓ CSV文件存在")
        print(f"  找到的方法: {methods}")
        print(f"  期望的方法: {config['expected_methods']}")
        
        # 验证方法是否匹配
        missing = set(config['expected_methods']) - set(methods)
        extra = set(methods) - set(config['expected_methods'])
        
        if missing:
            print(f"  ⚠️  缺少方法: {missing}")
        if extra:
            print(f"  ⚠️  额外方法: {extra}")
        if not missing and not extra:
            print(f"  ✓ 方法列表完全匹配")
        
        # 检查PDF文件数量
        pdf_files = [f for f in os.listdir(group_dir) if f.endswith('.pdf')]
        bar_plots = [f for f in pdf_files if f.startswith('Bar-')]
        series_plots = [f for f in pdf_files if f.startswith('Series-')]
        
        print(f"\n✓ PDF文件统计:")
        print(f"  柱状图 (Bar): {len(bar_plots)} 个")
        print(f"  时间序列图 (Series): {len(series_plots)} 个")
        print(f"  总计: {len(pdf_files)} 个")
        
        if len(bar_plots) < 20:
            print(f"  ⚠️  柱状图数量较少，期望至少20个")
        if len(series_plots) < 5:
            print(f"  ⚠️  时间序列图数量较少，期望至少5个")
        
        # 检查关键指标文件
        key_plots = [
            'Bar-Total_Energy_(Kilowatt-hr).pdf',
            'Bar-Average_Response_Time_(seconds).pdf',
            'Bar-Number_of_Task_migrations.pdf',
            'Bar-Fraction_of_total_SLA_Violations.pdf',
            'Series-Interval_Energy_(Kilowatt-hr).pdf',
            'Series-Number_of_Task_migrations.pdf'
        ]
        
        print(f"\n✓ 关键指标图表:")
        for plot in key_plots:
            plot_path = os.path.join(group_dir, plot)
            if os.path.exists(plot_path):
                size = os.path.getsize(plot_path)
                print(f"  ✓ {plot} ({size/1024:.1f} KB)")
            else:
                print(f"  ❌ {plot} 缺失")
                all_ok = False
    
    print(f"\n{'='*60}")
    if all_ok:
        print("✅ 所有检查通过！")
    else:
        print("⚠️  发现一些问题，请检查上述输出")
    print(f"{'='*60}")
    
    return all_ok

if __name__ == '__main__':
    check_plots()

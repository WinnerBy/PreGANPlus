#!/usr/bin/env python3
"""
阶段4：运行所有方法的测试脚本
目的：运行所有7个方法的测试，并将每个方法的结果保存到不同目录
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MAIN_PY = PROJECT_ROOT / "main.py"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "logs" / "stage4_results"

# 所有要测试的方法
METHODS = [
    'PreGAN',
    'PreGANPlus', 
    'PreGANPlusEnhanced',
    'PCFT',
    'DFTM',
    'ECLB',
    'CMODLB',
]

def backup_results(method_name):
    """将当前logs目录中的结果备份到方法特定的目录"""
    # 查找最新的结果目录（100步测试）
    result_dirs = list(LOGS_DIR.glob("RPiEdge_BWGD2_100_*"))
    
    if not result_dirs:
        print(f"⚠️  未找到 {method_name} 的测试结果目录")
        return None
    
    # 使用最新的目录
    latest_dir = max(result_dirs, key=lambda p: p.stat().st_mtime)
    
    # 创建方法特定的结果目录
    method_result_dir = RESULTS_DIR / method_name
    method_result_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制整个目录
    dest_dir = method_result_dir / latest_dir.name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(latest_dir, dest_dir)
    
    print(f"✅ {method_name} 结果已保存到: {dest_dir}")
    return dest_dir

def run_method_test(method_name):
    """运行单个方法的测试"""
    print("=" * 80)
    print(f"测试方法: {method_name}")
    print("=" * 80)
    print()
    
    # 运行阶段4测试脚本
    print(f"1. 配置main.py用于 {method_name}...")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "paper_experiment_stage4_testing.py"), "--method", method_name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ 配置失败: {result.stderr}")
        return False
    
    print(result.stdout)
    
    # 运行main.py
    print(f"2. 运行测试 (100步)...")
    print(f"   这可能需要几分钟时间...")
    
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "-e", "", "-m", "0"],
        cwd=PROJECT_ROOT,
        capture_output=False,  # 显示实时输出
    )
    
    if result.returncode != 0:
        print(f"❌ {method_name} 测试失败")
        return False
    
    # 备份结果
    print(f"3. 保存 {method_name} 的结果...")
    backup_results(method_name)
    
    print(f"✅ {method_name} 测试完成")
    print()
    return True

def main():
    """主函数：运行所有方法的测试"""
    print("=" * 80)
    print("阶段4：运行所有方法的测试")
    print("=" * 80)
    print()
    print(f"测试方法列表: {', '.join(METHODS)}")
    print(f"每个方法将运行100步测试")
    print(f"结果将保存到: {RESULTS_DIR}/")
    print()
    
    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 运行所有方法
    success_count = 0
    failed_methods = []
    
    for i, method in enumerate(METHODS, 1):
        print(f"\n[{i}/{len(METHODS)}] 开始测试 {method}...")
        
        if run_method_test(method):
            success_count += 1
        else:
            failed_methods.append(method)
    
    # 汇总结果
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("=" * 80)
    print("测试完成汇总")
    print("=" * 80)
    print(f"总方法数: {len(METHODS)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(failed_methods)}")
    if failed_methods:
        print(f"失败的方法: {', '.join(failed_methods)}")
    print(f"总耗时: {duration}")
    print()
    print(f"所有结果保存在: {RESULTS_DIR}/")
    print()
    print("下一步：运行分析脚本汇总所有结果")
    print(f"  python3 {SCRIPTS_DIR}/analyze_stage4_results.py")

if __name__ == "__main__":
    main()


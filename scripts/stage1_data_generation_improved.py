#!/usr/bin/env python
"""
阶段1改进版：高故障率数据生成脚本
目的：生成包含更多故障样本的训练数据（目标故障率15-25%）

改进策略：
1. 增加NEW_CONTAINERS: 5 → 15（大幅提高容器负载）
2. 增加NUM_SIM_STEPS: 1000 → 2000（更长的模拟时间）
3. 调整Simulator故障检测参数使其更敏感
"""

import sys
import os
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_TARGET_DIR = PROJECT_ROOT / "recovery" / "PreGANSrc" / "data" / "simulator"

def modify_main_py_for_data_collection(num_steps=2000, new_containers=15):
    """修改main.py用于高故障率数据收集阶段"""
    print("=" * 70)
    print("阶段1改进版：高故障率数据生成配置")
    print("=" * 70)
    
    # 读取main.py
    content = MAIN_PY.read_text()
    
    # 修改NUM_SIM_STEPS
    content = re.sub(
        r'^(\s*)NUM_SIM_STEPS\s*=\s*\d+',
        f'\\1NUM_SIM_STEPS = {num_steps}',
        content,
        flags=re.MULTILINE
    )
    
    # 修改NEW_CONTAINERS
    content = re.sub(
        r'^(\s*)NEW_CONTAINERS\s*=\s*\d+',
        f'\\1NEW_CONTAINERS = {new_containers}',
        content,
        flags=re.MULTILINE
    )
    
    # 修改recovery为Recovery基类（不进行任何恢复）
    content = re.sub(
        r'^(\s*)recovery\s*=\s*.*Recovery.*\(.*\)$',
        r'\1recovery = Recovery()',
        content,
        flags=re.MULTILINE
    )
    
    # 保存修改
    MAIN_PY.write_text(content)
    
    print(f"✅ 已配置main.py（改进版）：")
    print(f"   - NUM_SIM_STEPS = {num_steps} (从1000增加，更长的模拟时间)")
    print(f"   - NEW_CONTAINERS = {new_containers} (从5增加，更高的容器负载)")
    print(f"   - recovery = Recovery() (无恢复机制)")
    print(f"\n📊 预期结果：")
    print(f"   - 每个interval新增{new_containers}个容器（vs原来的5个）")
    print(f"   - {num_steps}个steps × {new_containers}个新容器 = {num_steps * new_containers}总容器创建")
    print(f"   - 预期故障率：15-25%（vs原来的1.82%）")
    print("")

def run_main_py_and_save_log(log_dir):
    """运行main.py并保存日志"""
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"stage1_improved_{timestamp}.log"
    
    print(f"📝 开始运行main.py（改进版）...")
    print(f"   日志文件: {log_file}")
    print(f"   此过程将耗时较长，请耐心等待...")
    print("")
    
    # 运行main.py并保存日志
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            ['python', str(MAIN_PY), '-e', '', '-m', '0'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时输出并保存
        for line in process.stdout:
            print(line, end='')
            f.write(line)
        
        process.wait()
    
    if process.returncode != 0:
        print(f"\n❌ 错误：main.py运行失败 (退出码: {process.returncode})")
        return False
    
    print(f"\n✅ main.py运行成功")
    return True

def copy_generated_data(pattern_prefix="RPiEdge_BWGD2"):
    """拷贝生成的数据到训练目录"""
    logs_dir = Path(LOGS_DIR)
    
    # 找到最新的生成数据目录
    directories = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith(pattern_prefix)],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    
    if not directories:
        print(f"❌ 错误：找不到日志目录 (模式: {pattern_prefix}*)")
        return False
    
    latest_dir = directories[0]
    print(f"✅ 找到最新的生成数据目录:")
    print(f"   {latest_dir}")
    print(f"   修改时间: {datetime.fromtimestamp(latest_dir.stat().st_mtime)}")
    print("")
    
    # 创建目标目录
    DATA_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 拷贝数据文件
    print(f"📂 拷贝数据文件...")
    
    # 拷贝time_series.npy
    time_series = latest_dir / 'time_series.npy'
    if time_series.exists():
        shutil.copy2(time_series, DATA_TARGET_DIR / 'time_series.npy')
        print(f"   ✅ time_series.npy")
    else:
        print(f"   ❌ 找不到 time_series.npy")
        return False
    
    # 拷贝schedule_series.npy
    schedule_series = latest_dir / 'schedule_series.npy'
    if schedule_series.exists():
        shutil.copy2(schedule_series, DATA_TARGET_DIR / 'schedule_series.npy')
        print(f"   ✅ schedule_series.npy")
    
    # 拷贝fault_history.pkl（如果存在）
    fault_history = latest_dir / 'fault_history.pkl'
    if fault_history.exists():
        shutil.copy2(fault_history, DATA_TARGET_DIR / 'fault_history.pkl')
        print(f"   ✅ fault_history.pkl")
    
    print(f"\n✅ 数据已保存到: {DATA_TARGET_DIR}")
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="阶段1改进版：高故障率数据生成")
    parser.add_argument('--steps', type=int, default=2000,
                        help='模拟步数 (默认: 2000)')
    parser.add_argument('--new-containers', type=int, default=15,
                        help='每步新增容器数 (默认: 15)')
    parser.add_argument('--log-dir', type=str, default='experiment_logs/stage1',
                        help='日志输出目录 (默认: experiment_logs/stage1)')
    args = parser.parse_args()
    
    # 步骤1：修改main.py参数
    modify_main_py_for_data_collection(args.steps, args.new_containers)
    
    # 步骤2：运行main.py
    success = run_main_py_and_save_log(args.log_dir)
    
    if not success:
        sys.exit(1)
    
    # 步骤3：拷贝生成的数据
    success = copy_generated_data()
    
    if not success:
        sys.exit(1)
    
    print("")
    print("=" * 70)
    print("✅ 阶段1改进版数据生成完成")
    print("=" * 70)
    print("")
    print("📊 下一步：")
    print("   python scripts/stage2_model_training.py --method PreGAN --encoder-only --steps 100")
    print("")

if __name__ == '__main__':
    main()

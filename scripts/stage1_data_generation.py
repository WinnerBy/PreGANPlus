#!/usr/bin/env python
"""
阶段1：数据生成脚本
目的：收集包含各种故障情况的训练数据，不进行任何预防性迁移
功能：
  - 配置main.py参数（NUM_SIM_STEPS, NEW_CONTAINERS, recovery基类）
  - 自动运行main.py
  - 自动保存日志
  - 自动拷贝生成的数据到训练目录
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

def modify_main_py_for_data_collection(num_steps=1000, new_containers=5):
    """修改main.py用于数据收集阶段"""
    print("=" * 70)
    print("阶段1：数据生成配置")
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
    
    print(f"✅ 已配置main.py：")
    print(f"   - NUM_SIM_STEPS = {num_steps}")
    print(f"   - NEW_CONTAINERS = {new_containers}")
    print(f"   - recovery = Recovery() (无恢复机制)")
    print("")

def run_main_py_and_save_log(log_dir):
    """运行main.py并保存日志"""
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"stage1_data_generation_{timestamp}.log"
    
    print(f"📝 开始运行main.py...")
    print(f"   日志文件: {log_file}")
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
    
    print(f"\n✅ main.py运行完成，日志已保存到: {log_file}")
    return True

def copy_generated_data(pattern_prefix="RPiEdge_BWGD2"):
    """查找并拷贝最新生成的数据到训练目录"""
    print("\n" + "=" * 70)
    print("数据管理：查找并拷贝生成的数据")
    print("=" * 70)
    
    # 查找最新的数据目录
    if not LOGS_DIR.exists():
        print(f"❌ 错误：日志目录不存在: {LOGS_DIR}")
        return False
    
    # 查找匹配的目录
    matching_dirs = sorted(
        [d for d in LOGS_DIR.iterdir() if d.is_dir() and d.name.startswith(pattern_prefix)],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not matching_dirs:
        print(f"❌ 错误：未找到匹配的数据目录（{pattern_prefix}*）")
        print(f"   请确认数据已生成到 {LOGS_DIR} 目录")
        return False
    
    latest_dir = matching_dirs[0]
    print(f"✅ 找到最新数据目录: {latest_dir.name}")
    
    # 检查必要的文件
    required_files = ['time_series.npy', 'schedule_series.npy']
    missing_files = [f for f in required_files if not (latest_dir / f).exists()]
    
    if missing_files:
        print(f"❌ 错误：数据目录缺少文件: {', '.join(missing_files)}")
        return False
    
    # 创建目标目录
    DATA_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 拷贝文件
    print(f"\n拷贝数据文件到训练目录:")
    print(f"   目标: {DATA_TARGET_DIR}")
    
    for filename in required_files:
        src = latest_dir / filename
        dst = DATA_TARGET_DIR / filename
        shutil.copy2(src, dst)
        print(f"   ✅ {filename}")
    
    # 可选：拷贝fault_history.pkl（如果存在）
    fault_history = latest_dir / 'fault_history.pkl'
    if fault_history.exists():
        shutil.copy2(fault_history, DATA_TARGET_DIR / 'fault_history.pkl')
        print(f"   ✅ fault_history.pkl")
    
    print(f"\n✅ 数据已成功拷贝到: {DATA_TARGET_DIR}")
    return True

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(
        description='阶段1：数据生成（自动配置、运行、保存日志、拷贝数据）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 默认：1000步
  python3 scripts/stage1_data_generation.py
  
  # 自定义步数
  python3 scripts/stage1_data_generation.py --steps 1500
  
  # 只配置，不运行
  python3 scripts/stage1_data_generation.py --config-only
  
  # 指定日志目录
  python3 scripts/stage1_data_generation.py --log-dir my_logs/stage1
        """
    )
    parser.add_argument('--steps', type=int, default=1000,
                       help='仿真步数（默认: 1000）')
    parser.add_argument('--new-containers', type=int, default=5,
                       help='每步新增容器数（默认: 5）')
    parser.add_argument('--log-dir', type=str, default='experiment_logs/stage1',
                       help='日志保存目录（默认: experiment_logs/stage1）')
    parser.add_argument('--config-only', action='store_true',
                       help='仅配置main.py，不运行')
    parser.add_argument('--no-copy', action='store_true',
                       help='不拷贝生成的数据')
    
    args = parser.parse_args()
    
    # 配置main.py
    modify_main_py_for_data_collection(args.steps, args.new_containers)
    
    if args.config_only:
        print("\n✅ 配置完成！（仅配置模式）")
        print("\n可手动运行: python3 main.py -e \"\" -m 0")
        return 0
    
    # 运行main.py
    success = run_main_py_and_save_log(args.log_dir)
    if not success:
        return 1
    
    # 拷贝数据
    if not args.no_copy:
        success = copy_generated_data()
        if not success:
            print("\n⚠️  警告：数据拷贝失败，但数据生成已完成")
            print("   可以手动运行数据拷贝脚本")
            return 1
    
    print("\n" + "=" * 70)
    print("✅ 阶段1完成！")
    print("=" * 70)
    print("\n📝 提示：")
    print("   - 生成的数据已保存到 logs/ 目录")
    print(f"   - 训练数据已拷贝到 {DATA_TARGET_DIR}")
    print("   - 可以继续运行阶段2进行模型训练")
    print("")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

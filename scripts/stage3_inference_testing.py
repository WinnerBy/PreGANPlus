#!/usr/bin/env python
"""
阶段3：推理测试脚本
目的：在独立测试集上评估模型性能
功能：
  - 配置training=False进行推理
  - 支持--method-set参数（gan/ablation/traditional/all）
  - 自动运行main.py并保存日志
"""

import sys
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"

# 方法分类
GAN_METHODS = ['PreGAN', 'PreGANPlus', 'PreGANPlusEnhanced']

ABLATION_METHODS = [
    'AblationNoTransformer', 'AblationNoGAT',
    'AblationNoMigrationAware', 'AblationNoMultiObjective'
]

TRADITIONAL_METHODS = ['PCFT', 'DFTM', 'ECLB', 'CMODLB']

ALL_METHODS = GAN_METHODS + ABLATION_METHODS + TRADITIONAL_METHODS

# Recovery类名映射
RECOVERY_MAP = {
    'PreGAN': 'PreGANRecovery',
    'PreGANPlus': 'PreGANPlusRecovery',
    'PreGANPlusEnhanced': 'PreGANPlusEnhancedRecovery',
    'AblationNoTransformer': 'AblationNoTransformerRecovery',
    'AblationNoGAT': 'AblationNoGATRecovery',
    'AblationNoMigrationAware': 'AblationNoMigrationAwareRecovery',
    'AblationNoMultiObjective': 'AblationNoMultiObjectiveRecovery',
    'PCFT': 'PCFTRecovery',
    'DFTM': 'DFTMRecovery',
    'ECLB': 'ECLBRecovery',
    'CMODLB': 'CMODLBRecovery',
}

def modify_main_py_for_testing(method, num_steps=100, new_containers=5):
    """修改main.py用于测试阶段"""
    print("=" * 70)
    print(f"阶段3：推理测试配置 - {method}")
    print("=" * 70)
    
    # 获取Recovery类名
    recovery_class = RECOVERY_MAP.get(method, method + 'Recovery')
    
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
    
    # 修改recovery，training=False
    recovery_line = f"recovery = {recovery_class}(HOSTS, environment, training = False)"
    content = re.sub(
        r'^(\s*)recovery\s*=\s*.*Recovery.*\(.*\)$',
        r'\1' + recovery_line,
        content,
        flags=re.MULTILINE
    )
    
    # 保存修改
    MAIN_PY.write_text(content)
    
    print(f"✅ 已配置main.py：")
    print(f"   - NUM_SIM_STEPS = {num_steps}")
    print(f"   - NEW_CONTAINERS = {new_containers}")
    print(f"   - recovery = {recovery_line}")
    print("")
    
    print("📝 测试模式：")
    print("   - training=False，仅推理不训练")
    if method in GAN_METHODS + ABLATION_METHODS:
        print("   - 加载训练好的编码器+GAN模型")
    elif method == 'CMODLB':
        print("   - 加载训练好的FCN编码器")
    else:
        print("   - 传统方法，无需加载模型")
    print("")

def run_main_py_and_save_log(log_dir, method):
    """运行main.py并保存日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"stage3_{method}_{timestamp}.log"
    
    print(f"📝 开始运行main.py...")
    print(f"   日志文件: {log_file}")
    print("")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            ['python', str(MAIN_PY), '-e', '', '-m', '0'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            f.write(line)
        
        process.wait()
    
    if process.returncode != 0:
        print(f"\n❌ main.py运行失败 (退出码: {process.returncode})")
        return False
    
    print(f"\n✅ 日志已保存: {log_file}")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='阶段3：推理测试（自动运行）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
方法分类：
  GAN方法：PreGAN, PreGANPlus, PreGANPlusEnhanced
  消融模型：AblationNoTransformer, AblationNoGAT, AblationNoMigrationAware, AblationNoMultiObjective
  传统方法：PCFT, DFTM, ECLB, CMODLB

使用示例：
  # 测试单个方法
  python3 scripts/stage3_inference_testing.py --method PreGAN

  # 测试所有GAN方法
  python3 scripts/stage3_inference_testing.py --method-set gan

  # 测试所有消融模型
  python3 scripts/stage3_inference_testing.py --method-set ablation

  # 测试所有方法
  python3 scripts/stage3_inference_testing.py --method-set all

  # 批量测试
  python3 scripts/stage3_inference_testing.py --methods PreGAN PreGANPlus PCFT
        """
    )
    
    parser.add_argument('--method', choices=ALL_METHODS,
                       help='测试单个方法')
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS,
                       help='测试多个方法（批量）')
    parser.add_argument('--method-set', choices=['gan', 'ablation', 'traditional', 'all'],
                       help='方法集合：gan(GAN方法), ablation(消融), traditional(传统), all(全部)')
    parser.add_argument('--steps', type=int, default=100,
                       help='测试步数（默认100）')
    parser.add_argument('--new-containers', type=int, default=5,
                       help='每步新增容器数（默认5）')
    parser.add_argument('--log-dir', default='experiment_logs/stage3',
                       help='日志目录（默认experiment_logs/stage3）')
    parser.add_argument('--config-only', action='store_true',
                       help='仅配置不运行')
    
    args = parser.parse_args()
    
    # 确定方法列表
    if args.method_set:
        method_map = {
            'gan': GAN_METHODS,
            'ablation': ABLATION_METHODS,
            'traditional': TRADITIONAL_METHODS,
            'all': ALL_METHODS
        }
        methods = method_map[args.method_set]
    elif args.methods:
        methods = args.methods
    elif args.method:
        methods = [args.method]
    else:
        parser.error("必须指定 --method, --methods 或 --method-set")
    
    # 测试每个方法
    all_success = True
    for i, method in enumerate(methods):
        if len(methods) > 1:
            print(f"\n{'='*70}")
            print(f"测试方法 [{i+1}/{len(methods)}]: {method}")
            print(f"{'='*70}")
        
        modify_main_py_for_testing(method, args.steps, args.new_containers)
        
        if args.config_only:
            print(f"✅ {method} 配置完成（仅配置模式）")
            continue
        
        success = run_main_py_and_save_log(args.log_dir, method)
        if not success:
            print(f"❌ {method} 测试失败")
            all_success = False
            if len(methods) > 1 and input("继续下一个？(y/n): ").lower() != 'y':
                break
        else:
            print(f"✅ {method} 测试完成")
    
    print(f"\n{'='*70}")
    if all_success:
        print("✅ 阶段3完成！")
    else:
        print("⚠️  阶段3完成，但部分方法失败")
    print(f"{'='*70}")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
统一实验运行脚本
目的：一键运行完整实验流程或指定阶段
功能：
  - 选择运行阶段（stage1/stage2/stage3或全部）
  - 选择运行方法（单个/多个/method-set）
  - 自动处理阶段间依赖
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

# 方法分类
GAN_METHODS = ['PreGAN', 'PreGANPlus', 'PreGANPlusEnhanced']
ABLATION_METHODS = ['AblationNoTransformer', 'AblationNoGAT',
                    'AblationNoMigrationAware', 'AblationNoMultiObjective']
TRADITIONAL_METHODS = ['PCFT', 'DFTM', 'ECLB', 'CMODLB']
ALL_METHODS = GAN_METHODS + ABLATION_METHODS + TRADITIONAL_METHODS

def run_stage1(args):
    """运行阶段1：数据生成"""
    print("\n" + "="*70)
    print("阶段1：数据生成")
    print("="*70)
    
    cmd = [
        'python',
        str(PROJECT_ROOT / 'scripts' / 'stage1_data_generation.py'),
        '--steps', str(args.stage1_steps),
        '--new-containers', str(args.new_containers),
        '--log-dir', f'{args.log_dir}/stage1'
    ]
    
    if args.stage1_no_copy:
        cmd.append('--no-copy')
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_stage2(args, methods):
    """运行阶段2：模型训练"""
    print("\n" + "="*70)
    print("阶段2：模型训练")
    print("="*70)
    
    if not methods:
        print("⚠️  未指定方法，跳过阶段2")
        return True
    
    # 过滤不需要训练的传统方法
    trainable = [m for m in methods if m not in ['PCFT', 'DFTM', 'ECLB']]
    if not trainable:
        print("⚠️  指定的方法都不需要训练，跳过阶段2")
        return True
    
    cmd = [
        'python',
        str(PROJECT_ROOT / 'scripts' / 'stage2_model_training.py'),
        '--methods'
    ] + trainable + [
        '--steps', str(args.stage2_steps),
        '--new-containers', str(args.new_containers),
        '--log-dir', f'{args.log_dir}/stage2'
    ]
    
    if args.encoder_only:
        cmd.append('--encoder-only')
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_stage3(args, methods):
    """运行阶段3：推理测试"""
    print("\n" + "="*70)
    print("阶段3：推理测试")
    print("="*70)
    
    if not methods:
        print("⚠️  未指定方法，跳过阶段3")
        return True
    
    cmd = [
        'python',
        str(PROJECT_ROOT / 'scripts' / 'stage3_inference_testing.py'),
        '--methods'
    ] + methods + [
        '--steps', str(args.stage3_steps),
        '--new-containers', str(args.new_containers),
        '--log-dir', f'{args.log_dir}/stage3'
    ]
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='统一实验运行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 完整流程
  python3 scripts/run_experiment.py --all-stages --methods PreGAN

  # 完整流程（GAN方法）
  python3 scripts/run_experiment.py --all-stages --method-set gan

  # 只训练
  python3 scripts/run_experiment.py --stage2 --method-set gan

  # 训练+测试
  python3 scripts/run_experiment.py --stage2 --stage3 --methods PreGAN PreGANPlus

  # 只训练编码器
  python3 scripts/run_experiment.py --stage2 --methods PreGAN --encoder-only

  # 测试所有消融模型
  python3 scripts/run_experiment.py --stage3 --method-set ablation

  # 测试所有方法
  python3 scripts/run_experiment.py --stage3 --method-set all
        """
    )
    
    # 阶段选择
    stage_group = parser.add_argument_group('阶段选择')
    stage_group.add_argument('--stage1', action='store_true', help='运行阶段1：数据生成')
    stage_group.add_argument('--stage2', action='store_true', help='运行阶段2：模型训练')
    stage_group.add_argument('--stage3', action='store_true', help='运行阶段3：推理测试')
    stage_group.add_argument('--all-stages', action='store_true', help='运行所有阶段')
    
    # 方法选择
    method_group = parser.add_argument_group('方法选择')
    method_group.add_argument('--methods', nargs='+', choices=ALL_METHODS,
                             help='指定方法')
    method_group.add_argument('--method-set', choices=['gan', 'ablation', 'traditional', 'all'],
                             help='方法集合：gan/ablation/traditional/all')
    
    # 阶段1参数
    stage1_group = parser.add_argument_group('阶段1参数')
    stage1_group.add_argument('--stage1-steps', type=int, default=1000,
                             help='数据生成步数（默认1000）')
    stage1_group.add_argument('--stage1-no-copy', action='store_true',
                             help='不自动拷贝数据')
    
    # 阶段2参数
    stage2_group = parser.add_argument_group('阶段2参数')
    stage2_group.add_argument('--stage2-steps', type=int, default=1200,
                             help='训练步数（默认1200）')
    stage2_group.add_argument('--encoder-only', action='store_true',
                             help='仅训练编码器')
    
    # 阶段3参数
    stage3_group = parser.add_argument_group('阶段3参数')
    stage3_group.add_argument('--stage3-steps', type=int, default=100,
                             help='测试步数（默认100）')
    
    # 通用参数
    general_group = parser.add_argument_group('通用参数')
    general_group.add_argument('--new-containers', type=int, default=5,
                              help='每步新增容器数（默认5）')
    general_group.add_argument('--log-dir', default='experiment_logs',
                              help='日志根目录（默认experiment_logs）')
    
    args = parser.parse_args()
    
    # 确定运行阶段
    if args.all_stages:
        stages = [1, 2, 3]
    else:
        stages = []
        if args.stage1: stages.append(1)
        if args.stage2: stages.append(2)
        if args.stage3: stages.append(3)
    
    if not stages:
        parser.error("必须指定阶段：--stage1/--stage2/--stage3 或 --all-stages")
    
    # 确定方法
    if 2 in stages or 3 in stages:
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
        else:
            if 1 not in stages:
                parser.error("运行阶段2或3时必须指定 --methods 或 --method-set")
            methods = []
    else:
        methods = []
    
    # 打印配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n" + "="*70)
    print("PreGAN实验运行")
    print("="*70)
    print(f"时间: {timestamp}")
    print(f"阶段: {', '.join(['阶段'+str(s) for s in stages])}")
    if methods:
        print(f"方法: {', '.join(methods)}")
    if args.encoder_only and 2 in stages:
        print("模式: 仅训练编码器")
    print("="*70)
    
    # 确认
    confirm = input("\n确认开始实验？(y/n): ")
    if confirm.lower() != 'y':
        print("已取消")
        return 0
    
    # 运行各阶段
    start_time = datetime.now()
    
    try:
        if 1 in stages:
            if not run_stage1(args):
                print("\n❌ 阶段1失败")
                return 1
        
        if 2 in stages:
            if not run_stage2(args, methods):
                print("\n❌ 阶段2失败")
                return 1
        
        if 3 in stages:
            if args.encoder_only and 2 in stages:
                print("\n⚠️  注意：使用了--encoder-only，GAN未训练")
                if input("仍要运行阶段3测试？(y/n): ").lower() != 'y':
                    print("已跳过阶段3")
                    return 0
            
            if not run_stage3(args, methods):
                print("\n❌ 阶段3失败")
                return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        return 1
    
    # 完成
    duration = datetime.now() - start_time
    print("\n" + "="*70)
    print("✅ 实验完成！")
    print("="*70)
    print(f"总耗时: {duration}")
    print(f"日志目录: {args.log_dir}")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

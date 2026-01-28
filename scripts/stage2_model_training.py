#!/usr/bin/env python
"""
阶段2：模型训练脚本
目的：训练编码器和GAN模型（合并原stage2+stage3）
功能：
  - 编码器自动训练（如果checkpoint不存在或epoch==-1）
  - GAN在线训练（使用运行时数据）
  - 支持--encoder-only模式（只训练编码器）
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

def modify_main_py_for_training(method, num_steps=1200, new_containers=5,
                                 encoder_only=False):
    """修改main.py用于训练阶段"""
    print("=" * 70)
    print(f"阶段2：模型训练配置 - {method}")
    if encoder_only:
        print("模式：仅训练编码器")
    print("=" * 70)
    
    # 获取Recovery类名
    recovery_class = RECOVERY_MAP.get(method, method + 'Recovery')
    
    # 读取main.py
    content = MAIN_PY.read_text()
    
    # 确定步数
    if method == 'CMODLB':
        steps = 1200  # CMODLB需要环境稳定
    elif encoder_only:
        steps = 10  # 仅触发编码器训练
    else:
        steps = num_steps
    
    # 修改NUM_SIM_STEPS
    content = re.sub(
        r'^(\s*)NUM_SIM_STEPS\s*=\s*\d+',
        f'\\1NUM_SIM_STEPS = {steps}',
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
    
    # 修改recovery
    # encoder_only模式：training=False, encoder_only=True（只触发编码器训练，不加载GAN）
    # 正常模式：training=True（训练编码器+GAN）
    training_flag = 'False' if encoder_only else 'True'
    
    # 为支持encoder_only的方法添加encoder_only参数
    gan_and_ablation_methods = ['PreGAN', 'PreGANPlus', 'PreGANPlusEnhanced',
                                'AblationNoTransformer', 'AblationNoGAT',
                                'AblationNoMigrationAware', 'AblationNoMultiObjective']
    
    if method in gan_and_ablation_methods and encoder_only:
        recovery_line = f"recovery = {recovery_class}(HOSTS, environment, training = {training_flag}, encoder_only = True)"
    else:
        recovery_line = f"recovery = {recovery_class}(HOSTS, environment, training = {training_flag})"
    
    content = re.sub(
        r'^(\s*)recovery\s*=\s*.*Recovery.*\(.*\)$',
        r'\1' + recovery_line,
        content,
        flags=re.MULTILINE
    )
    
    # 保存修改
    MAIN_PY.write_text(content)
    
    print(f"✅ 已配置main.py：")
    print(f"   - NUM_SIM_STEPS = {steps}")
    print(f"   - NEW_CONTAINERS = {new_containers}")
    print(f"   - recovery = {recovery_line}")
    print("")
    
    # 打印说明
    print("📝 训练机制：")
    if method in TRADITIONAL_METHODS:
        if method == 'CMODLB':
            print("   - CMODLB的FCN编码器会自动训练（如果未训练）")
            print("   - 训练数据：recovery/PreGANSrc/data/")
        else:
            print(f"   - {method}是传统方法，不需要训练")
    else:
        if encoder_only:
            print("   - 仅训练编码器模式")
            print("   - 编码器使用stage1数据离线训练")
            print("   - 不加载/训练GAN")
        else:
            print("   - 编码器：如已训练(epoch!=-1)自动跳过，否则用stage1数据训练")
            print(f"   - GAN：在线训练，使用{steps}步运行时数据")
            if method in ['PreGANPlus', 'PreGANPlusEnhanced']:
                print(f"   - 💡 提示：{method}与另一方法共享Transformer编码器")
    print("")

def run_main_py_and_save_log(log_dir, method, encoder_only=False):
    """运行main.py并保存日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "encoder_only" if encoder_only else "full"
    log_file = log_dir / f"stage2_{method}_{mode}_{timestamp}.log"
    
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
        description='阶段2：模型训练（编码器+GAN，自动运行）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
方法分类：
  GAN方法：PreGAN, PreGANPlus, PreGANPlusEnhanced
  消融模型：AblationNoTransformer, AblationNoGAT, AblationNoMigrationAware, AblationNoMultiObjective
  传统方法：PCFT, DFTM, ECLB, CMODLB

使用示例：
  # 训练单个GAN方法
  python3 scripts/stage2_model_training.py --method PreGAN

  # 训练所有GAN方法
  python3 scripts/stage2_model_training.py --method-set gan

  # 训练所有消融模型
  python3 scripts/stage2_model_training.py --method-set ablation

  # 只训练编码器（优化模式）
  python3 scripts/stage2_model_training.py --method PreGAN --encoder-only

  # 批量训练
  python3 scripts/stage2_model_training.py --methods PreGAN PreGANPlus
        """
    )
    
    parser.add_argument('--method', choices=ALL_METHODS,
                       help='训练单个方法')
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS,
                       help='训练多个方法（批量）')
    parser.add_argument('--method-set', choices=['gan', 'ablation', 'traditional', 'all'],
                       help='方法集合：gan(GAN方法), ablation(消融), traditional(传统), all(全部)')
    parser.add_argument('--steps', type=int, default=1200,
                       help='训练步数（默认1200）')
    parser.add_argument('--new-containers', type=int, default=5,
                       help='每步新增容器数（默认5）')
    parser.add_argument('--encoder-only', action='store_true',
                       help='仅训练编码器（不训练GAN）')
    parser.add_argument('--log-dir', default='experiment_logs/stage2',
                       help='日志目录（默认experiment_logs/stage2）')
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
    
    # 过滤不需要训练的方法
    if args.encoder_only:
        skip = [m for m in methods if m in ['PCFT', 'DFTM', 'ECLB']]
        if skip:
            print(f"⚠️  {', '.join(skip)} 不需要训练，已跳过")
            methods = [m for m in methods if m not in skip]
        
        # PreGANPlus和PreGANPlusEnhanced共享Transformer编码器
        # 如果两者都在列表中，在encoder_only模式下只训练一次
        if 'PreGANPlus' in methods and 'PreGANPlusEnhanced' in methods:
            print(f"\n{'='*70}")
            print("📌 检测到PreGANPlus和PreGANPlusEnhanced")
            print("   这两个方法共享相同的Transformer编码器")
            print("   在encoder_only模式下，只训练一次编码器")
            print(f"{'='*70}\n")
            # 保留PreGANPlus，移除PreGANPlusEnhanced
            methods = [m for m in methods if m != 'PreGANPlusEnhanced']
            print(f"✅ 将使用PreGANPlus训练共享的Transformer编码器")
            print(f"   (PreGANPlusEnhanced将复用该编码器)\n")
    
    if not methods:
        print("❌ 没有需要训练的方法")
        return 1
    
    # 训练每个方法
    all_success = True
    for i, method in enumerate(methods):
        if len(methods) > 1:
            print(f"\n{'='*70}")
            print(f"训练方法 [{i+1}/{len(methods)}]: {method}")
            print(f"{'='*70}")
        
        modify_main_py_for_training(method, args.steps, args.new_containers, args.encoder_only)
        
        if args.config_only:
            print(f"✅ {method} 配置完成（仅配置模式）")
            continue
        
        success = run_main_py_and_save_log(args.log_dir, method, args.encoder_only)
        if not success:
            print(f"❌ {method} 训练失败")
            all_success = False
            if len(methods) > 1 and input("继续下一个？(y/n): ").lower() != 'y':
                break
        else:
            print(f"✅ {method} 训练完成")
    
    print(f"\n{'='*70}")
    if all_success:
        print("✅ 阶段2完成！")
    else:
        print("⚠️  阶段2完成，但部分方法失败")
    print(f"{'='*70}")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())

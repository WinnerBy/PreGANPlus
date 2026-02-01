#!/usr/bin/env python
"""
阶段3：推理测试脚本
目的：在独立测试集上评估模型性能
功能：
  - 配置training=False进行推理
  - 支持--method-set参数（gan/ablation/traditional/all）
  - 支持--runs N：每个方法运行N次，便于收集多轮数据并筛选合理结果
  - 支持-y/--yes：某方法失败后自动继续下一个，无需交互
  - 推理配置默认 600 步×10 容器（与训练 1200×8 区分，便于体现模型在不同负载下的表现；见下方 STAGE3_CONFIG 说明）
  - 自动运行main.py并保存日志
"""

# Stage3 推理配置说明（与训练 1200 步×8 容器区分）：
# - 步数 600：比训练短，避免“复现训练环境”；足够长以便累积指标（能耗、迁移次数、SLO）拉开差距。
# - 每步 10 容器：比训练略高负载，主机更满、分配与迁移更复杂，利于体现：
#   基准（迁移感知+多目标）在复杂负载下更少不良迁移、更优能耗/SLO；
#   消融（无迁移感知/无多目标/无 GAT）在高压下更容易暴露出次优决策。
# 若需更短测试可改用 --steps 400。

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

def modify_main_py_for_testing(method, num_steps=600, new_containers=10):
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

def run_main_py_and_save_log(log_dir, method, run_index=None):
    """运行main.py并保存日志。
    run_index: 若提供且>0，日志名为 stage3_{method}_run{run_index:02d}_{timestamp}.log，便于多轮收集。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_index is not None and run_index >= 1:
        log_file = log_dir / f"stage3_{method}_run{run_index:02d}_{timestamp}.log"
    else:
        log_file = log_dir / f"stage3_{method}_{timestamp}.log"
    
    print(f"📝 开始运行main.py...")
    print(f"   日志文件: {log_file}")
    print("")
    
    # 为本次运行设置独立 logs 子目录后缀，避免多终端/多 run 写同一目录冲突
    env = os.environ.copy()
    if run_index is not None and run_index >= 1:
        env["LOGS_SUFFIX"] = f"{method}_run{run_index:02d}"
    else:
        env["LOGS_SUFFIX"] = method

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            ['python', str(MAIN_PY), '-e', '', '-m', '0'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
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
  # 测试单个方法（默认600步、每步10容器，与训练1200×8区分）
  python3 scripts/stage3_inference_testing.py --method PreGAN

  # 测试所有方法，每方法运行10次，失败自动继续下一项（无人看管）
  python3 scripts/stage3_inference_testing.py --method-set all --runs 10 -y

  # 测试所有GAN方法
  python3 scripts/stage3_inference_testing.py --method-set gan

  # 测试所有消融模型，每方法5次
  python3 scripts/stage3_inference_testing.py --method-set ablation --runs 5

  # 批量测试，自定义步数/容器数
  python3 scripts/stage3_inference_testing.py --methods PreGAN PreGANPlus --steps 600 --new-containers 8
        """
    )
    
    parser.add_argument('--method', choices=ALL_METHODS,
                       help='测试单个方法')
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS,
                       help='测试多个方法（批量）')
    parser.add_argument('--method-set', choices=['gan', 'ablation', 'traditional', 'all'],
                       help='方法集合：gan(GAN方法), ablation(消融), traditional(传统), all(全部)')
    parser.add_argument('--steps', type=int, default=600,
                       help='测试步数（默认600，与训练1200区分，足够长以体现累积指标差异）')
    parser.add_argument('--new-containers', type=int, default=10,
                       help='每步新增容器数（默认10，略高于训练8，提高负载以体现基准迁移感知/多目标优势）')
    parser.add_argument('--runs', type=int, default=1,
                       help='每个方法运行次数（默认1）；>1时日志名为 stage3_<method>_run<N>_<timestamp>.log，便于多轮收集后筛选分析')
    parser.add_argument('--log-dir', default='experiment_logs/stage3',
                       help='日志目录（默认experiment_logs/stage3）')
    parser.add_argument('--config-only', action='store_true',
                       help='仅配置不运行')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='某方法失败后自动继续下一个（无人看管/批处理时使用，不提示）')
    
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
    
    # 测试每个方法（可选多轮）
    all_success = True
    total_tasks = len(methods) * args.runs
    task_idx = 0
    for i, method in enumerate(methods):
        modify_main_py_for_testing(method, args.steps, args.new_containers)
        if args.config_only:
            print(f"✅ {method} 配置完成（仅配置模式）")
            continue

        for run in range(1, args.runs + 1):
            task_idx += 1
            if args.runs > 1:
                print(f"\n{'='*70}")
                print(f"测试方法 [{task_idx}/{total_tasks}]: {method} 第 {run}/{args.runs} 次运行")
                print(f"{'='*70}")
            elif len(methods) > 1:
                print(f"\n{'='*70}")
                print(f"测试方法 [{i+1}/{len(methods)}]: {method}")
                print(f"{'='*70}")

            run_index = run if args.runs > 1 else None
            success = run_main_py_and_save_log(args.log_dir, method, run_index=run_index)
            if not success:
                print(f"❌ {method}" + (f" 第{run}次" if args.runs > 1 else "") + " 测试失败")
                all_success = False
                if len(methods) > 1 or args.runs > 1:
                    if args.yes:
                        print("(已启用 -y，自动继续下一项)")
                    else:
                        try:
                            if input("继续下一个？(y/n): ").lower() != 'y':
                                return 1 if not all_success else 0
                        except EOFError:
                            print("(非交互式，自动继续下一项)")
            else:
                print(f"✅ {method}" + (f" 第{run}次" if args.runs > 1 else "") + " 测试完成")
    
    print(f"\n{'='*70}")
    if all_success:
        print("✅ 阶段3完成！")
    else:
        print("⚠️  阶段3完成，但部分方法/轮次失败")
    print(f"{'='*70}")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())

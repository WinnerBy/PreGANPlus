#!/usr/bin/env python3
"""
阶段2：编码器训练脚本
目的：训练FPE/Transformer/FCN编码器（如果未训练）
参数：短步数运行，触发自动训练
"""

import sys
import os
import re
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"

# 方法映射（使用代码中的实际名称）
RECOVERY_MAP = {
    'PreGAN': 'PreGANRecovery',
    'PreGANPlus': 'PreGANPlusRecovery',
    'PreGANPlusEnhanced': 'PreGANPlusEnhancedRecovery',
    'CMODLB': 'CMODLBRecovery',
}

def modify_main_py_for_encoder_training(method='PreGAN'):
    """修改main.py用于编码器训练阶段"""
    print("=" * 60)
    print(f"阶段2：编码器训练 - {method}")
    print("=" * 60)
    
    # 获取Recovery类名
    recovery_class = RECOVERY_MAP.get(method, method)
    if not recovery_class.endswith('Recovery'):
        recovery_class = method + 'Recovery'
    
    # 读取main.py
    content = MAIN_PY.read_text()
    
    # 修改NUM_SIM_STEPS = 10（短步数，仅用于触发训练）
    content = re.sub(
        r'^(\s*)NUM_SIM_STEPS\s*=\s*\d+',
        r'\1NUM_SIM_STEPS = 10',
        content,
        flags=re.MULTILINE
    )
    
    # 修改NEW_CONTAINERS = 5
    content = re.sub(
        r'^(\s*)NEW_CONTAINERS\s*=\s*\d+',
        r'\1NEW_CONTAINERS = 5',
        content,
        flags=re.MULTILINE
    )
    
    # 修改recovery为指定方法，training=False（编码器训练是自动的）
    recovery_line = f"recovery = {recovery_class}(HOSTS, environment, training = False)"
    content = re.sub(
        r'^(\s*)recovery\s*=\s*.*$',
        r'\1' + recovery_line,
        content,
        flags=re.MULTILINE
    )
    
    # 保存修改
    MAIN_PY.write_text(content)
    
    print(f"✅ 已修改main.py配置：")
    print(f"   - NUM_SIM_STEPS = 10（短步数，仅用于触发训练）")
    print(f"   - NEW_CONTAINERS = 5")
    print(f"   - recovery = {recovery_line}")
    print("")
    print("📝 注意：")
    if method == 'CMODLB':
        print("   - CMODLB的FCN编码器会自动训练（如果checkpoint不存在）")
        print("   - 训练数据来自 recovery/PreGANSrc/data/")
        print("   - 训练30个epoch后自动保存并冻结")
    else:
        print("   - FPE/Transformer编码器会自动训练（如果checkpoint不存在）")
        print("   - 训练数据来自 recovery/PreGANSrc/data/")
        print("   - 训练完成后自动保存并冻结")
    print("")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='阶段2：编码器训练')
    parser.add_argument('--method', type=str, default='PreGAN',
                       choices=['PreGAN', 'PreGANPlus', 'PreGANPlusEnhanced', 'CMODLB'],
                       help='要训练编码器的方法（代码中的实际名称）')
    args = parser.parse_args()
    
    modify_main_py_for_encoder_training(args.method)
    print("=" * 60)
    print("配置完成！现在可以运行：")
    print("  python main.py -e \"\" -m 0")
    print("=" * 60)
    print("")
    print("⚠️  注意：此阶段仅用于触发编码器训练，运行步数很少")


#!/usr/bin/env python3
"""
阶段2+3合并：编码器训练 + GAN训练脚本
目的：
  1. 如果编码器未训练（epoch == -1），自动训练编码器（使用阶段1的1000步数据）
  2. 编码器训练完成后，继续运行1200步的GAN训练
参数：1200步，training=True
注意：
  - 编码器训练是自动的，使用阶段1收集的1000步数据（离线训练）
  - GAN训练是在线的，使用1200步运行中的数据
  - 如果效果不好，可以多训练几次
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
}

def modify_main_py_for_gan_training(method='PreGAN'):
    """修改main.py用于编码器训练 + GAN训练阶段"""
    print("=" * 60)
    print(f"阶段2+3合并：编码器训练 + GAN训练 - {method}")
    print("=" * 60)
    
    # 获取Recovery类名
    recovery_class = RECOVERY_MAP.get(method, method)
    if not recovery_class.endswith('Recovery'):
        recovery_class = method + 'Recovery'
    
    # 读取main.py
    content = MAIN_PY.read_text()
    
    # 设置训练步数：论文设置为1200步
    steps = 1200
    content = re.sub(
        r'^(\s*)NUM_SIM_STEPS\s*=\s*\d+',
        lambda m: f"{m.group(1)}NUM_SIM_STEPS = {steps}",
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
    
    # 修改recovery为指定方法，training=True
    recovery_line = f"recovery = {recovery_class}(HOSTS, environment, training = True)"
    content = re.sub(
        r'^(\s*)recovery\s*=\s*.*$',
        r'\1' + recovery_line,
        content,
        flags=re.MULTILINE
    )
    
    # 保存修改
    MAIN_PY.write_text(content)
    
    print(f"✅ 已修改main.py配置：")
    print(f"   - NUM_SIM_STEPS = {steps}（可根据需要调整）")
    print(f"   - NEW_CONTAINERS = 5")
    print(f"   - recovery = {recovery_line}")
    print("")
    print("📝 注意：")
    print("   - 编码器训练是自动的：如果checkpoint不存在或epoch == -1，会自动训练")
    print("   - 编码器训练使用阶段1收集的1000步数据（离线训练）")
    print("   - 编码器训练完成后，会继续运行1200步的GAN训练")
    print("   - GAN会在在线运行过程中训练（1200步）")
    print("   - 如果效果不好，可以多训练几次（再次运行此阶段）")
    print("   - PreGANPlus支持在线微调Transformer编码器（如果启用）")
    print("")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='阶段2+3合并：编码器训练 + GAN训练')
    parser.add_argument('--method', type=str, default='PreGAN',
                       choices=['PreGAN', 'PreGANPlus', 'PreGANPlusEnhanced'],
                       help='要训练的方法（代码中的实际名称：PreGAN, PreGANPlus, PreGANPlusEnhanced）')
    args = parser.parse_args()
    
    modify_main_py_for_gan_training(args.method)
    print("=" * 60)
    print("配置完成！现在可以运行：")
    print("  python main.py -e \"\" -m 0")
    print("=" * 60)


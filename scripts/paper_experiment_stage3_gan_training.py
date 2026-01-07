#!/usr/bin/env python3
"""
阶段3：在线训练GAN脚本
目的：训练生成器和判别器，学习最优迁移策略
参数：300个间隔（可根据需要调整），training=True
注意：如果效果不好，可以多训练几次
内存优化：中间保存频率为每50步保存一次stats
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
    """修改main.py用于GAN训练阶段"""
    print("=" * 60)
    print(f"阶段3：在线训练GAN - {method}")
    print("=" * 60)
    
    # 获取Recovery类名
    recovery_class = RECOVERY_MAP.get(method, method)
    if not recovery_class.endswith('Recovery'):
        recovery_class = method + 'Recovery'
    
    # 读取main.py
    content = MAIN_PY.read_text()
    
    # 修改NUM_SIM_STEPS = 300（可根据需要调整，如果效果不好可以继续训练）
    content = re.sub(
        r'^(\s*)NUM_SIM_STEPS\s*=\s*\d+',
        r'\1NUM_SIM_STEPS = 300',
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
    print(f"   - NUM_SIM_STEPS = 300（可根据需要调整）")
    print(f"   - NEW_CONTAINERS = 5")
    print(f"   - recovery = {recovery_line}")
    print("")
    print("📝 注意：")
    print("   - 编码器已在阶段2训练完成，此阶段只加载已训练的编码器")
    print("   - GAN会在在线运行过程中训练（300步）")
    print("   - 如果效果不好，可以多训练几次（再次运行此阶段）")
    print("   - 训练过程中GAN checkpoint会定期保存（每次train_gan调用）")
    print("   - 如果程序被终止，可以从已有checkpoint继续训练")
    print("   - 中间保存频率：每50步保存一次stats（减少内存压力）")
    print("   - PreGANPlus支持在线微调Transformer编码器（如果启用）")
    print("")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='阶段3：在线训练GAN')
    parser.add_argument('--method', type=str, default='PreGAN',
                       choices=['PreGAN', 'PreGANPlus', 'PreGANPlusEnhanced'],
                       help='要训练的方法（代码中的实际名称：PreGAN, PreGANPlus, PreGANPlusEnhanced）')
    args = parser.parse_args()
    
    modify_main_py_for_gan_training(args.method)
    print("=" * 60)
    print("配置完成！现在可以运行：")
    print("  python main.py -e \"\" -m 0")
    print("=" * 60)


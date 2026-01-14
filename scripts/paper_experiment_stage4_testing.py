#!/usr/bin/env python3
"""
阶段4：测试评估脚本
目的：在独立测试集上评估模型性能
参数：100个间隔，training=False，所有方法对比
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
    'PCFT': 'PCFTRecovery',
    'DFTM': 'DFTMRecovery',
    'ECLB': 'ECLBRecovery',
    'CMODLB': 'CMODLBRecovery',
}

def modify_main_py_for_testing(method='PreGAN', temp_file=None):
    """修改main.py用于测试评估阶段"""
    print("=" * 60)
    print(f"阶段4：测试评估 - {method}")
    print("=" * 60)
    
    # 获取Recovery类名
    recovery_class = RECOVERY_MAP.get(method, method)
    if not recovery_class.endswith('Recovery'):
        recovery_class = method + 'Recovery'
    
    # 确定要修改的文件
    target_file = Path(temp_file) if temp_file else MAIN_PY
    
    # 读取文件
    content = target_file.read_text()
    
    # 修改NUM_SIM_STEPS = 100
    content = re.sub(
        r'^(\s*)NUM_SIM_STEPS\s*=\s*\d+',
        r'\1NUM_SIM_STEPS = 100',
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
    
    # 修改recovery为指定方法，training=False
    recovery_line = f"recovery = {recovery_class}(HOSTS, environment, training = False)"
    # 修复：避免重复的 "recovery = "，匹配完整的recovery行
    content = re.sub(
        r'^(\s*)recovery\s*=\s*.*Recovery.*\(.*\)$',
        r'\1' + recovery_line,
        content,
        flags=re.MULTILINE
    )
    
    # 保存修改
    target_file.write_text(content)
    
    print(f"✅ 已修改main.py配置：")
    print(f"   - NUM_SIM_STEPS = 100")
    print(f"   - NEW_CONTAINERS = 5")
    print(f"   - recovery = {recovery_line}")
    print("")
    print("📝 注意：")
    print("   - training=False，不进行训练，只进行推理")
    print("   - 传统方法（PCFT, DFTM, ECLB）不需要训练，可直接运行")
    if method == 'CMODLB':
        print("   - CMODLB会加载阶段2训练的FCN模型")
    else:
        print("   - GAN方法会加载阶段2训练的编码器和阶段3训练的GAN")
    print("")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='阶段4：测试评估')
    parser.add_argument('--method', type=str, default='PreGAN',
                       choices=['PreGAN', 'PreGANPlus', 'PreGANPlusEnhanced',
                               'PCFT', 'DFTM', 'ECLB', 'CMODLB'],
                       help='要测试的方法（代码中的实际名称）')
    parser.add_argument('--temp-file', type=str, default=None,
                       help='临时文件路径（用于避免权限问题）')
    args = parser.parse_args()
    
    modify_main_py_for_testing(args.method, args.temp_file)
    print("=" * 60)
    print("配置完成！现在可以运行：")
    print("  python main.py -e \"\" -m 0")
    print("=" * 60)


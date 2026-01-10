#!/usr/bin/env python3
"""
阶段1：数据收集脚本
目的：收集包含各种故障情况的训练数据，不进行任何预防性迁移
参数：1000个间隔（可根据需要调整），使用Recovery基类（不进行任何恢复）
注意：如果1000步数据训练效果不好，可以继续生成更多数据
"""

import sys
import os
import re
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"

def modify_main_py_for_data_collection():
    """修改main.py用于数据收集阶段"""
    print("=" * 60)
    print("阶段1：数据收集配置")
    print("=" * 60)
    
    # 读取main.py
    content = MAIN_PY.read_text()
    
    # 修改NUM_SIM_STEPS = 1000（论文配置）
    content = re.sub(
        r'^(\s*)NUM_SIM_STEPS\s*=\s*\d+',
        r'\1NUM_SIM_STEPS = 1000',
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
    
    # 修改recovery为Recovery基类（不进行任何恢复）
    content = re.sub(
        r'^(\s*)recovery\s*=\s*.*Recovery.*$',
        r'\1recovery = Recovery()',
        content,
        flags=re.MULTILINE
    )
    
    # 确保Recovery被导入
    if 'from recovery.Recovery import Recovery' not in content:
        # 在recovery imports部分添加
        recovery_imports = [
            'from recovery.Recovery import Recovery',
            'from recovery.PreGAN import PreGANRecovery',
            'from recovery.PreGANPlus import PreGANPlusRecovery',
        ]
        for imp in recovery_imports:
            if imp not in content:
                # 在Recovery imports后添加
                content = re.sub(
                    r'(from recovery\.Recovery import Recovery)',
                    r'\1\n' + '\n'.join([i for i in recovery_imports if i not in content]),
                    content
                )
                break
    
    # 保存修改
    MAIN_PY.write_text(content)
    
    print("✅ 已修改main.py配置：")
    print("   - NUM_SIM_STEPS = 1000（论文配置）")
    print("   - NEW_CONTAINERS = 5")
    print("   - recovery = Recovery() (无恢复机制)")
    print("")
    print("📝 注意：")
    print("   - 此阶段不进行任何预防性迁移，仅收集数据")
    print("")

if __name__ == "__main__":
    modify_main_py_for_data_collection()
    print("=" * 60)
    print("配置完成！现在可以运行：")
    print("  python main.py -e \"\" -m 0")
    print("=" * 60)


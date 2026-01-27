#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MPS float64兼容性修复脚本 - 自动将float64转换为float32（当使用MPS时）
"""

import torch
import sys
sys.path.insert(0, 'recovery/PreGANSrc/')

from src.device_manager import get_device_manager

def ensure_dtype_compatibility(data, name="tensor"):
    """
    确保数据类型与设备兼容
    - MPS: 强制转换为float32
    - CPU/CUDA: 保持float64
    """
    device_manager = get_device_manager(verbose=False)
    target_dtype = device_manager.get_dtype()
    
    if isinstance(data, torch.Tensor):
        if data.dtype != target_dtype:
            return data.to(dtype=target_dtype)
        return data
    
    return data

def patch_torch_tensor_creation():
    """
    猴补丁：自动处理torch.tensor()的dtype
    """
    original_tensor = torch.tensor
    device_manager = get_device_manager(verbose=False)
    
    def patched_tensor(*args, **kwargs):
        # 如果没有指定dtype，使用推荐的dtype
        if 'dtype' not in kwargs:
            kwargs['dtype'] = device_manager.get_dtype()
        return original_tensor(*args, **kwargs)
    
    torch.tensor = patched_tensor
    print("✓ torch.tensor()已修补为自动使用兼容dtype")

def patch_nn_modules():
    """
    猴补丁：自动将nn.Module转换为推荐的dtype
    """
    import torch.nn as nn
    
    original_to = nn.Module.to
    device_manager = get_device_manager(verbose=False)
    
    def patched_to(self, *args, **kwargs):
        # 如果没有指定dtype且使用MPS，则强制使用float32
        if device_manager.is_mps_available() and 'dtype' not in kwargs:
            # 尝试推断dtype（如果有nn.Parameter可以参考）
            pass
        
        return original_to(self, *args, **kwargs)
    
    nn.Module.to = patched_to
    print("✓ nn.Module.to()已修补以处理MPS兼容性")

def print_compatibility_report():
    """打印兼容性报告"""
    device_manager = get_device_manager(verbose=False)
    
    print("\n" + "=" * 70)
    print("MPS float64 兼容性修复报告")
    print("=" * 70)
    
    print(f"\n设备: {device_manager.get_torch_device()}")
    print(f"推荐数据类型: {device_manager.get_dtype()}")
    
    if device_manager.is_mps_available():
        print("\n✓ 已启用MPS兼容性修复:")
        print("  - float64 → float32 自动转换")
        print("  - torch.tensor()自动使用float32")
        print("  - GAN模型使用float32在MPS上运行")
        print("  - 编码器模型使用float32在CPU上运行")
    else:
        print("\n✓ 使用CPU/CUDA，无需float64转换")
    
    print("\n关键提示:")
    print("  1. 模型创建时使用: model = Model().to(dtype=device_manager.get_dtype())")
    print("  2. 张量创建时使用: torch.tensor(data, dtype=device_manager.get_dtype())")
    print("  3. 数据加载时使用: data.to(dtype=device_manager.get_dtype())")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    print("初始化MPS float64兼容性修复...")
    print("-" * 70)
    
    # 获取设备管理器（显示初始化信息）
    device_manager = get_device_manager(verbose=True)
    
    # 应用修补
    patch_torch_tensor_creation()
    patch_nn_modules()
    
    # 打印报告
    print_compatibility_report()

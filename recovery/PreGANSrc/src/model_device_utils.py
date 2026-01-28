"""
模型设备辅助工具 - 为模型添加设备管理功能
"""
import torch
import torch.nn as nn
from .device_manager import get_device_manager

class DeviceAwareModelMixin:
    """
    为模型添加设备感知能力的Mixin类
    可以混入任何PyTorch模型
    """
    
    def setup_device(self, device_manager=None):
        """
        设置模型的设备管理器
        
        Args:
            device_manager: DeviceManager实例，如果为None则使用全局实例
        """
        if device_manager is None:
            device_manager = get_device_manager()
        
        self.device_manager = device_manager
        self.torch_device = device_manager.get_torch_device()
        self.dgl_device = device_manager.get_dgl_device()
        
        # 将模型参数移动到PyTorch设备
        self.to(self.torch_device)
        
        # 将DGL图移动到DGL设备（如果存在）
        if hasattr(self, 'gat_graph'):
            self.gat_graph = device_manager.to_dgl_device(self.gat_graph)
        
        # 将prototype移动到PyTorch设备（如果存在）
        if hasattr(self, 'prototype'):
            for i in range(len(self.prototype)):
                self.prototype[i] = self.prototype[i].to(self.torch_device)
    
    def to_torch_device(self, tensor):
        """将张量移动到PyTorch设备"""
        if not hasattr(self, 'device_manager'):
            return tensor
        return self.device_manager.to_torch_device(tensor)
    
    def to_dgl_device(self, tensor):
        """将张量移动到DGL设备"""
        if not hasattr(self, 'device_manager'):
            return tensor
        return self.device_manager.to_dgl_device(tensor)
    
    def sync_from_dgl(self, tensor):
        """从DGL设备同步张量到PyTorch设备"""
        if not hasattr(self, 'device_manager'):
            return tensor
        return self.device_manager.move_between_devices(tensor, from_dgl=True)


def wrap_model_forward(original_forward):
    """
    包装模型的forward方法以自动处理设备管理
    
    Args:
        original_forward: 原始的forward方法
    
    Returns:
        包装后的forward方法
    """
    def forward_with_device_management(self, *args, **kwargs):
        # 确保输入在正确的设备上
        if hasattr(self, 'device_manager'):
            # 将输入移动到PyTorch设备
            args = tuple(
                arg.to(self.torch_device) if isinstance(arg, torch.Tensor) else arg
                for arg in args
            )
            kwargs = {
                k: v.to(self.torch_device) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
        
        # 调用原始forward
        output = original_forward(self, *args, **kwargs)
        
        return output
    
    return forward_with_device_management


def create_tensor_on_device(data, dtype=None, device_manager=None, use_dgl_device=False):
    """
    在指定设备上创建张量
    
    Args:
        data: 数据
        dtype: 数据类型
        device_manager: 设备管理器
        use_dgl_device: 是否使用DGL设备
    
    Returns:
        创建的张量
    """
    if device_manager is None:
        device_manager = get_device_manager()
    
    target_device = device_manager.get_dgl_device() if use_dgl_device else device_manager.get_torch_device()
    
    if dtype is None:
        dtype = device_manager.get_dtype()  # 使用设备管理器推荐的dtype
    
    return torch.tensor(data, dtype=dtype, device=target_device)


def move_batch_to_device(batch, device_manager=None, use_dgl_device=False):
    """
    将批次数据移动到指定设备
    
    Args:
        batch: 批次数据（可以是张量、张量列表或字典）
        device_manager: 设备管理器
        use_dgl_device: 是否使用DGL设备
    
    Returns:
        移动到目标设备的批次数据
    """
    if device_manager is None:
        device_manager = get_device_manager()
    
    target_device = device_manager.get_dgl_device() if use_dgl_device else device_manager.get_torch_device()
    
    if isinstance(batch, torch.Tensor):
        return batch.to(target_device)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(item, device_manager, use_dgl_device) for item in batch)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device_manager, use_dgl_device) for k, v in batch.items()}
    else:
        return batch

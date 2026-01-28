"""
设备管理器 - 统一管理PyTorch和DGL的设备分配
支持Mac M3 GPU (MPS)，并自动处理DGL的CPU回退
"""
import torch
import platform
import os

class DeviceManager:
    """统一的设备管理器"""
    
    def __init__(self, force_cpu=False, verbose=True):
        """
        初始化设备管理器
        
        Args:
            force_cpu: 是否强制使用CPU
            verbose: 是否打印设备信息
        """
        self.force_cpu = force_cpu
        self.verbose = verbose
        
        # 检测可用设备
        self._detect_devices()
        
        if self.verbose:
            self._print_device_info()
    
    def _detect_devices(self):
        """检测可用的计算设备"""
        # 1. PyTorch主设备
        if self.force_cpu:
            self.torch_device = torch.device('cpu')
            self.use_mps = False
            self.use_cuda = False
        elif torch.cuda.is_available():
            self.torch_device = torch.device('cuda')
            self.use_mps = False
            self.use_cuda = True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Mac M系列芯片的MPS支持
            self.torch_device = torch.device('mps')
            self.use_mps = True
            self.use_cuda = False
        else:
            self.torch_device = torch.device('cpu')
            self.use_mps = False
            self.use_cuda = False
        
        # 2. DGL设备（DGL在MPS上支持有限，需要CPU回退）
        # DGL 2.2.1在MPS上会报错，需要在CPU上执行
        if self.use_mps:
            self.dgl_device = torch.device('cpu')
            self.dgl_on_cpu = True
        else:
            self.dgl_device = self.torch_device
            self.dgl_on_cpu = False
        
        # 3. 推荐的数据类型
        # 统一使用float32以保持兼容性（MPS不支持float64）
        # 即使在CPU上也使用float32，确保模型权重和数据类型一致
        self.default_dtype = torch.float32
        self.default_np_dtype = 'float32'
    
    def _print_device_info(self):
        """打印设备信息"""
        print("=" * 60)
        print("设备管理器初始化")
        print("=" * 60)
        print(f"系统: {platform.system()} {platform.machine()}")
        print(f"PyTorch版本: {torch.__version__}")
        
        # 打印设备信息
        if self.use_mps:
            print(f"  ⚠ MPS设备不支持float64，使用{self.default_dtype}")
        elif self.use_cuda:
            print(f"  ✓ 使用NVIDIA GPU (CUDA)")
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        else:
            print("  ✓ 使用CPU")
        
        print(f"\nDGL设备: {self.dgl_device}")
        if self.dgl_on_cpu:
            print("  ⚠ DGL在CPU上运行（MPS兼容性限制）")
        
        print(f"\n推荐数据类型: {self.default_dtype}")
        print("=" * 60)
    
    def get_torch_device(self):
        """获取PyTorch设备"""
        return self.torch_device
    
    def get_dgl_device(self):
        """获取DGL设备"""
        return self.dgl_device
    
    def get_dtype(self):
        """获取推荐的数据类型"""
        return self.default_dtype
    
    def get_np_dtype(self):
        """获取推荐的NumPy数据类型"""
        return self.default_np_dtype
    
    def to_torch_device(self, tensor):
        """将张量移动到PyTorch设备"""
        if tensor.device != self.torch_device:
            return tensor.to(self.torch_device)
        return tensor
    
    def to_dgl_device(self, tensor_or_graph):
        """
        将张量或DGL图移动到DGL设备
        
        Args:
            tensor_or_graph: torch.Tensor或dgl.DGLGraph
        """
        try:
            import dgl
            if isinstance(tensor_or_graph, dgl.DGLGraph):
                # 移动DGL图
                if tensor_or_graph.device != self.dgl_device:
                    return tensor_or_graph.to(self.dgl_device)
                return tensor_or_graph
        except ImportError:
            pass
        
        # 移动张量
        if hasattr(tensor_or_graph, 'device'):
            if tensor_or_graph.device != self.dgl_device:
                return tensor_or_graph.to(self.dgl_device)
        return tensor_or_graph
    
    def move_between_devices(self, tensor, from_dgl=False):
        """
        在PyTorch设备和DGL设备之间移动张量
        
        Args:
            tensor: 要移动的张量
            from_dgl: True表示从DGL设备移到PyTorch设备，False表示相反
        """
        if from_dgl:
            # DGL -> PyTorch
            target_device = self.torch_device
        else:
            # PyTorch -> DGL
            target_device = self.dgl_device
        
        if tensor.device != target_device:
            return tensor.to(target_device)
        return tensor
    
    def is_mps_available(self):
        """检查MPS是否可用"""
        return self.use_mps
    
    def is_cuda_available(self):
        """检查CUDA是否可用"""
        return self.use_cuda
    
    def needs_device_sync(self):
        """检查是否需要设备同步（DGL和PyTorch设备不同时）"""
        return self.dgl_device != self.torch_device


# 全局设备管理器实例
_global_device_manager = None

def get_device_manager(force_cpu=False, verbose=False):
    """
    获取全局设备管理器实例（单例模式）
    
    Args:
        force_cpu: 是否强制使用CPU
        verbose: 是否打印设备信息（仅首次创建时有效）
    """
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager(force_cpu=force_cpu, verbose=verbose)
    return _global_device_manager

def reset_device_manager():
    """重置全局设备管理器（用于测试）"""
    global _global_device_manager
    _global_device_manager = None

"""
节点基类：定义所有ISP节点的通用接口
支持AI/传统实现切换，统一的I/O格式
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
from .frame import Frame
from .flow import Flow


class NodeType(Enum):
    """节点类型枚举"""
    INPUT = "input"
    PROCESSING = "processing"
    OUTPUT = "output"


class ImplementationType(Enum):
    """实现类型枚举"""
    CLASSIC = "classic"  # 传统算法
    AI = "ai"           # AI算法
    HYBRID = "hybrid"   # 混合实现


class NodeStatus(Enum):
    """节点状态枚举"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"


class Node(ABC):
    """ISP节点基类"""
    
    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        implementation: ImplementationType = ImplementationType.CLASSIC,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        """
        初始化节点
        
        Args:
            node_id: 节点唯一标识
            node_type: 节点类型
            implementation: 实现类型
            config: 配置参数
            enabled: 是否启用
        """
        self.node_id = node_id
        self.node_type = node_type
        self.implementation = implementation
        self.config = config or {}
        self.enabled = enabled
        self.status = NodeStatus.IDLE
        self.error_message = ""
        
        # 输入输出端口
        self.input_ports: List[str] = []
        self.output_ports: List[str] = []
        
        # 性能统计
        self.processing_times: List[float] = []
        self.total_processed = 0
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数"""
        pass
    
    @abstractmethod
    def process(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """
        处理输入数据
        
        Args:
            inputs: 输入数据字典，键为端口名，值为数据
            
        Returns:
            输出数据字典，键为端口名，值为数据
        """
        pass
    
    def process_frame(self, frame: Frame) -> Frame:
        """
        处理单个Frame（便捷方法）
        
        Args:
            frame: 输入Frame
            
        Returns:
            输出Frame
        """
        inputs = {"input": frame}
        outputs = self.process(inputs)
        return outputs.get("output", frame)
    
    def process_flow(self, flow: Flow) -> Flow:
        """
        处理Flow（便捷方法）
        
        Args:
            flow: 输入Flow
            
        Returns:
            输出Flow
        """
        inputs = {"input": flow}
        outputs = self.process(inputs)
        return outputs.get("output", flow)
    
    def set_config(self, config: Dict[str, Any]):
        """设置配置参数"""
        self.config.update(config)
        self._validate_config()
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置参数"""
        return self.config.copy()
    
    def set_implementation(self, implementation: ImplementationType):
        """设置实现类型"""
        self.implementation = implementation
        self._on_implementation_changed()
    
    def _on_implementation_changed(self):
        """实现类型改变时的回调"""
        pass
    
    def enable(self):
        """启用节点"""
        self.enabled = True
        self.status = NodeStatus.IDLE
    
    def disable(self):
        """禁用节点"""
        self.enabled = False
        self.status = NodeStatus.DISABLED
    
    def reset(self):
        """重置节点状态"""
        self.status = NodeStatus.IDLE
        self.error_message = ""
        self.processing_times.clear()
        self.total_processed = 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.processing_times:
            return {
                "total_processed": self.total_processed,
                "avg_processing_time": 0.0,
                "min_processing_time": 0.0,
                "max_processing_time": 0.0
            }
        
        return {
            "total_processed": self.total_processed,
            "avg_processing_time": np.mean(self.processing_times),
            "min_processing_time": np.min(self.processing_times),
            "max_processing_time": np.max(self.processing_times),
            "std_processing_time": np.std(self.processing_times)
        }
    
    def _record_processing_time(self, start_time: float, end_time: float):
        """记录处理时间"""
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)
        self.total_processed += 1
        
        # 保持最近100次的记录
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
    
    def _set_error(self, error_message: str):
        """设置错误状态"""
        self.status = NodeStatus.ERROR
        self.error_message = error_message
    
    def _clear_error(self):
        """清除错误状态"""
        if self.status == NodeStatus.ERROR:
            self.status = NodeStatus.IDLE
            self.error_message = ""
    
    def is_ready(self) -> bool:
        """检查节点是否准备就绪"""
        return self.enabled and self.status != NodeStatus.ERROR
    
    def get_input_ports(self) -> List[str]:
        """获取输入端口列表"""
        return self.input_ports.copy()
    
    def get_output_ports(self) -> List[str]:
        """获取输出端口列表"""
        return self.output_ports.copy()
    
    def has_input_port(self, port_name: str) -> bool:
        """检查是否有指定输入端口"""
        return port_name in self.input_ports
    
    def has_output_port(self, port_name: str) -> bool:
        """检查是否有指定输出端口"""
        return port_name in self.output_ports
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id={self.node_id}, "
                f"type={self.node_type.value}, "
                f"impl={self.implementation.value}, "
                f"status={self.status.value})")


class ProcessingNode(Node):
    """处理节点基类"""
    
    def __init__(
        self,
        node_id: str,
        config: Optional[Dict[str, Any]] = None,
        implementation: ImplementationType = ImplementationType.CLASSIC,
        enabled: bool = True
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.PROCESSING,
            implementation=implementation,
            config=config,
            enabled=enabled
        )
        
        # 默认输入输出端口
        self.input_ports = ["input"]
        self.output_ports = ["output"]
    
    def process(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """处理输入数据"""
        import time
        
        if not self.is_ready():
            raise RuntimeError(f"节点{self.node_id}未准备就绪: {self.status.value}")
        
        start_time = time.time()
        self.status = NodeStatus.PROCESSING
        
        try:
            # 根据实现类型选择处理方法
            if self.implementation == ImplementationType.CLASSIC:
                result = self._process_classic(inputs)
            elif self.implementation == ImplementationType.AI:
                result = self._process_ai(inputs)
            elif self.implementation == ImplementationType.HYBRID:
                result = self._process_hybrid(inputs)
            else:
                raise ValueError(f"不支持的实现类型: {self.implementation}")
            
            self._clear_error()
            self.status = NodeStatus.IDLE
            
            return result
            
        except Exception as e:
            self._set_error(str(e))
            raise
        finally:
            end_time = time.time()
            self._record_processing_time(start_time, end_time)
    
    @abstractmethod
    def _process_classic(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """传统算法实现"""
        pass
    
    @abstractmethod
    def _process_ai(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """AI算法实现"""
        pass
    
    def _process_hybrid(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """混合实现（默认使用AI）"""
        return self._process_ai(inputs)


class InputNode(Node):
    """输入节点基类"""
    
    def __init__(
        self,
        node_id: str,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.INPUT,
            implementation=ImplementationType.CLASSIC,
            config=config,
            enabled=enabled
        )
        
        self.output_ports = ["output"]
    
    def process(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """输入节点不需要输入数据"""
        raise NotImplementedError("输入节点不需要process方法")


class OutputNode(Node):
    """输出节点基类"""
    
    def __init__(
        self,
        node_id: str,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.OUTPUT,
            implementation=ImplementationType.CLASSIC,
            config=config,
            enabled=enabled
        )
        
        self.input_ports = ["input"]
    
    def process(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """输出节点通常不产生输出数据"""
        return {}

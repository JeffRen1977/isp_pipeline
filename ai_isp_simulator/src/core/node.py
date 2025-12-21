"""
Node Base Class: Defines common interface for all ISP nodes
Supports AI/traditional implementation switching, unified I/O format
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
from .frame import Frame
from .flow import Flow


class NodeType(Enum):
    """Node type enumeration"""
    INPUT = "input"
    PROCESSING = "processing"
    OUTPUT = "output"


class ImplementationType(Enum):
    """Implementation type enumeration"""
    CLASSIC = "classic"  # Traditional algorithm
    AI = "ai"           # AI algorithm
    HYBRID = "hybrid"   # Hybrid implementation


class NodeStatus(Enum):
    """Node status enumeration"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"


class Node(ABC):
    """ISP node base class"""
    
    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        implementation: ImplementationType = ImplementationType.CLASSIC,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        """
        Initialize node
        
        Args:
            node_id: Node unique identifier
            node_type: Node type
            implementation: Implementation type
            config: Configuration parameters
            enabled: Whether enabled
        """
        self.node_id = node_id
        self.node_type = node_type
        self.implementation = implementation
        self.config = config or {}
        self.enabled = enabled
        self.status = NodeStatus.IDLE
        self.error_message = ""
        
        # Input/output ports
        self.input_ports: List[str] = []
        self.output_ports: List[str] = []
        
        # Performance statistics
        self.processing_times: List[float] = []
        self.total_processed = 0
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        pass
    
    @abstractmethod
    def process(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """
        Process input data
        
        Args:
            inputs: Input data dictionary, keys are port names, values are data
            
        Returns:
            Output data dictionary, keys are port names, values are data
        """
        pass
    
    def process_frame(self, frame: Frame) -> Frame:
        """
        Process single Frame (convenience method)
        
        Args:
            frame: Input Frame
            
        Returns:
            Output Frame
        """
        inputs = {"input": frame}
        outputs = self.process(inputs)
        return outputs.get("output", frame)
    
    def process_flow(self, flow: Flow) -> Flow:
        """
        Process Flow (convenience method)
        
        Args:
            flow: Input Flow
            
        Returns:
            Output Flow
        """
        inputs = {"input": flow}
        outputs = self.process(inputs)
        return outputs.get("output", flow)
    
    def set_config(self, config: Dict[str, Any]):
        """Set configuration parameters"""
        self.config.update(config)
        self._validate_config()
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration parameters"""
        return self.config.copy()
    
    def set_implementation(self, implementation: ImplementationType):
        """Set implementation type"""
        self.implementation = implementation
        self._on_implementation_changed()
    
    def _on_implementation_changed(self):
        """Callback when implementation type changes"""
        pass
    
    def enable(self):
        """Enable node"""
        self.enabled = True
        self.status = NodeStatus.IDLE
    
    def disable(self):
        """Disable node"""
        self.enabled = False
        self.status = NodeStatus.DISABLED
    
    def reset(self):
        """Reset node state"""
        self.status = NodeStatus.IDLE
        self.error_message = ""
        self.processing_times.clear()
        self.total_processed = 0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
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
        """Record processing time"""
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)
        self.total_processed += 1
        
        # Keep only the most recent 100 records
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
    
    def _set_error(self, error_message: str):
        """Set error state"""
        self.status = NodeStatus.ERROR
        self.error_message = error_message
    
    def _clear_error(self):
        """Clear error state"""
        if self.status == NodeStatus.ERROR:
            self.status = NodeStatus.IDLE
            self.error_message = ""
    
    def is_ready(self) -> bool:
        """Check if node is ready"""
        return self.enabled and self.status != NodeStatus.ERROR
    
    def get_input_ports(self) -> List[str]:
        """Get input port list"""
        return self.input_ports.copy()
    
    def get_output_ports(self) -> List[str]:
        """Get output port list"""
        return self.output_ports.copy()
    
    def has_input_port(self, port_name: str) -> bool:
        """Check if has specified input port"""
        return port_name in self.input_ports
    
    def has_output_port(self, port_name: str) -> bool:
        """Check if has specified output port"""
        return port_name in self.output_ports
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id={self.node_id}, "
                f"type={self.node_type.value}, "
                f"impl={self.implementation.value}, "
                f"status={self.status.value})")


class ProcessingNode(Node):
    """Processing node base class"""
    
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
        
        # Default input/output ports
        self.input_ports = ["input"]
        self.output_ports = ["output"]
    
    def process(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """Process input data"""
        import time
        
        if not self.is_ready():
            raise RuntimeError(f"Node {self.node_id} is not ready: {self.status.value}")
        
        start_time = time.time()
        self.status = NodeStatus.PROCESSING
        
        try:
            # Select processing method based on implementation type
            if self.implementation == ImplementationType.CLASSIC:
                result = self._process_classic(inputs)
            elif self.implementation == ImplementationType.AI:
                result = self._process_ai(inputs)
            elif self.implementation == ImplementationType.HYBRID:
                result = self._process_hybrid(inputs)
            else:
                raise ValueError(f"Unsupported implementation type: {self.implementation}")
            
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
        """Traditional algorithm implementation"""
        pass
    
    @abstractmethod
    def _process_ai(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """AI algorithm implementation"""
        pass
    
    def _process_hybrid(self, inputs: Dict[str, Union[Frame, Flow, np.ndarray]]) -> Dict[str, Union[Frame, Flow, np.ndarray]]:
        """Hybrid implementation (default uses AI)"""
        return self._process_ai(inputs)


class InputNode(Node):
    """Input node base class"""
    
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
        """Input nodes do not require input data"""
        raise NotImplementedError("Input nodes do not need process method")


class OutputNode(Node):
    """Output node base class"""
    
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
        """Output nodes typically do not produce output data"""
        return {}

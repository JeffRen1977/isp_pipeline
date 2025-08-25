"""
Graph引擎：实现有向图的数据流处理
支持动态配置、节点管理、数据流控制等
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from collections import defaultdict, deque
from .node import Node, NodeType, NodeStatus
from .frame import Frame
from .flow import Flow


class Graph:
    """ISP Graph引擎"""
    
    def __init__(self, graph_id: str):
        """
        初始化Graph
        
        Args:
            graph_id: Graph唯一标识
        """
        self.graph_id = graph_id
        
        # 节点管理
        self.nodes: Dict[str, Node] = {}
        self.node_order: List[str] = []  # 拓扑排序后的节点顺序
        
        # 连接管理
        self.connections: Dict[str, List[str]] = defaultdict(list)  # 输出 -> 输入节点列表
        self.reverse_connections: Dict[str, List[str]] = defaultdict(list)  # 输入 -> 输出节点列表
        
        # 数据缓存
        self.data_cache: Dict[str, Dict[str, Union[Frame, Flow, Any]]] = {}
        
        # 性能统计
        self.execution_times: List[float] = []
        self.total_executions = 0
        
        # 日志
        self.logger = logging.getLogger(f"Graph_{graph_id}")
        
        # 验证状态
        self._is_validated = False
    
    def add_node(self, node: Node) -> bool:
        """
        添加节点
        
        Args:
            node: 要添加的节点
            
        Returns:
            是否添加成功
        """
        if node.node_id in self.nodes:
            self.logger.warning(f"节点{node.node_id}已存在")
            return False
        
        self.nodes[node.node_id] = node
        self.node_order.append(node.node_id)
        self._is_validated = False
        
        self.logger.info(f"添加节点: {node.node_id}")
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        移除节点
        
        Args:
            node_id: 要移除的节点ID
            
        Returns:
            是否移除成功
        """
        if node_id not in self.nodes:
            self.logger.warning(f"节点{node_id}不存在")
            return False
        
        # 移除相关连接
        self._remove_node_connections(node_id)
        
        # 移除节点
        del self.nodes[node_id]
        if node_id in self.node_order:
            self.node_order.remove(node_id)
        
        # 清除缓存
        if node_id in self.data_cache:
            del self.data_cache[node_id]
        
        self._is_validated = False
        
        self.logger.info(f"移除节点: {node_id}")
        return True
    
    def connect_nodes(self, from_node_id: str, to_node_id: str, from_port: str = "output", to_port: str = "input") -> bool:
        """
        连接两个节点
        
        Args:
            from_node_id: 源节点ID
            to_node_id: 目标节点ID
            from_port: 源节点输出端口
            to_port: 目标节点输入端口
            
        Returns:
            是否连接成功
        """
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            self.logger.error(f"节点不存在: {from_node_id} -> {to_node_id}")
            return False
        
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        
        if not from_node.has_output_port(from_port):
            self.logger.error(f"源节点{from_node_id}没有输出端口{from_port}")
            return False
        
        if not to_node.has_input_port(to_port):
            self.logger.error(f"目标节点{to_node_id}没有输入端口{to_port}")
            return False
        
        # 建立连接
        connection_key = f"{from_node_id}:{from_port}"
        self.connections[connection_key].append(f"{to_node_id}:{to_port}")
        self.reverse_connections[f"{to_node_id}:{to_port}"].append(connection_key)
        
        self._is_validated = False
        
        self.logger.info(f"连接节点: {from_node_id}:{from_port} -> {to_node_id}:{to_port}")
        return True
    
    def disconnect_nodes(self, from_node_id: str, to_node_id: str, from_port: str = "output", to_port: str = "input") -> bool:
        """
        断开两个节点的连接
        
        Args:
            from_node_id: 源节点ID
            to_node_id: 目标节点ID
            from_port: 源节点输出端口
            to_port: 目标节点输入端口
            
        Returns:
            是否断开成功
        """
        connection_key = f"{from_node_id}:{from_port}"
        target_key = f"{to_node_id}:{to_port}"
        
        if connection_key in self.connections and target_key in self.connections[connection_key]:
            self.connections[connection_key].remove(target_key)
            if target_key in self.reverse_connections:
                self.reverse_connections[target_key].remove(connection_key)
            
            self._is_validated = False
            self.logger.info(f"断开连接: {from_node_id}:{from_port} -> {to_node_id}:{to_port}")
            return True
        
        return False
    
    def validate(self) -> bool:
        """
        验证Graph的有效性
        
        Returns:
            是否有效
        """
        try:
            # 检查是否有环
            if self._has_cycle():
                self.logger.error("Graph中存在环")
                return False
            
            # 检查输入输出节点
            input_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.INPUT]
            output_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.OUTPUT]
            
            if not input_nodes:
                self.logger.error("Graph中没有输入节点")
                return False
            
            if not output_nodes:
                self.logger.error("Graph中没有输出节点")
                return False
            
            # 拓扑排序
            self._topological_sort()
            
            self._is_validated = True
            self.logger.info("Graph验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"Graph验证失败: {e}")
            return False
    
    def _has_cycle(self) -> bool:
        """检查是否有环（使用DFS）"""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
            
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # 检查所有输出连接
            for port in self.nodes[node_id].get_output_ports():
                connection_key = f"{node_id}:{port}"
                for target in self.connections.get(connection_key, []):
                    target_node_id = target.split(":")[0]
                    if dfs(target_node_id):
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        
        return False
    
    def _topological_sort(self):
        """拓扑排序"""
        in_degree = defaultdict(int)
        
        # 计算入度
        for node_id in self.nodes:
            for port in self.nodes[node_id].get_input_ports():
                target_key = f"{node_id}:{port}"
                for source in self.reverse_connections.get(target_key, []):
                    source_node_id = source.split(":")[0]
                    in_degree[node_id] += 1
        
        # 拓扑排序
        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        sorted_nodes = []
        
        while queue:
            node_id = queue.popleft()
            sorted_nodes.append(node_id)
            
            for port in self.nodes[node_id].get_output_ports():
                connection_key = f"{node_id}:{port}"
                for target in self.connections.get(connection_key, []):
                    target_node_id = target.split(":")[0]
                    in_degree[target_node_id] -= 1
                    if in_degree[target_node_id] == 0:
                        queue.append(target_node_id)
        
        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Graph中存在环，无法进行拓扑排序")
        
        self.node_order = sorted_nodes
    
    def execute(self, inputs: Optional[Dict[str, Union[Frame, Flow, Any]]] = None) -> Dict[str, Union[Frame, Flow, Any]]:
        """
        执行Graph
        
        Args:
            inputs: 输入数据字典
            
        Returns:
            输出数据字典
        """
        if not self._is_validated:
            if not self.validate():
                raise RuntimeError("Graph验证失败，无法执行")
        
        start_time = time.time()
        self.total_executions += 1
        
        try:
            # 初始化数据缓存
            self.data_cache.clear()
            if inputs:
                self.data_cache.update(inputs)
            
            # 按拓扑顺序执行节点
            for node_id in self.node_order:
                node = self.nodes[node_id]
                
                if not node.is_ready():
                    self.logger.warning(f"节点{node_id}未准备就绪，跳过")
                    continue
                
                # 收集输入数据
                node_inputs = self._collect_node_inputs(node_id)
                
                # 执行节点
                node_outputs = node.process(node_inputs)
                
                # 缓存输出数据
                self.data_cache[node_id] = node_outputs
            
            # 收集最终输出
            outputs = self._collect_final_outputs()
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            # 保持最近100次的记录
            if len(self.execution_times) > 100:
                self.execution_times.pop(0)
            
            self.logger.info(f"Graph执行完成，耗时: {execution_time:.3f}s")
            return outputs
            
        except Exception as e:
            self.logger.error(f"Graph执行失败: {e}")
            raise
    
    def _collect_node_inputs(self, node_id: str) -> Dict[str, Union[Frame, Flow, Any]]:
        """收集节点的输入数据"""
        node = self.nodes[node_id]
        inputs = {}
        
        for port in node.get_input_ports():
            target_key = f"{node_id}:{port}"
            
            # 查找连接到这个端口的源节点
            for source in self.reverse_connections.get(target_key, []):
                source_node_id, source_port = source.split(":")
                
                # 从缓存中获取数据
                if source_node_id in self.data_cache:
                    source_data = self.data_cache[source_node_id]
                    if source_port in source_data:
                        inputs[port] = source_data[source_port]
                        break
        
        return inputs
    
    def _collect_final_outputs(self) -> Dict[str, Union[Frame, Flow, Any]]:
        """收集最终输出数据"""
        outputs = {}
        
        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.OUTPUT:
                # 收集输出节点的输入作为最终输出
                node_inputs = self._collect_node_inputs(node_id)
                for port, data in node_inputs.items():
                    outputs[f"{node_id}_{port}"] = data
        
        return outputs
    
    def _remove_node_connections(self, node_id: str):
        """移除节点的所有连接"""
        node = self.nodes[node_id]
        
        # 移除输出连接
        for port in node.get_output_ports():
            connection_key = f"{node_id}:{port}"
            if connection_key in self.connections:
                del self.connections[connection_key]
        
        # 移除输入连接
        for port in node.get_input_ports():
            target_key = f"{node_id}:{port}"
            if target_key in self.reverse_connections:
                for source in self.reverse_connections[target_key]:
                    source_node_id, source_port = source.split(":")
                    source_key = f"{source_node_id}:{source_port}"
                    if source_key in self.connections:
                        self.connections[source_key] = [
                            target for target in self.connections[source_key] 
                            if target != target_key
                        ]
                del self.reverse_connections[target_key]
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """获取指定节点"""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """获取指定类型的节点"""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def get_connections(self) -> List[Tuple[str, str]]:
        """获取所有连接"""
        connections = []
        for source, targets in self.connections.items():
            for target in targets:
                connections.append((source, target))
        return connections
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.execution_times:
            return {
                "total_executions": self.total_executions,
                "avg_execution_time": 0.0,
                "min_execution_time": 0.0,
                "max_execution_time": 0.0
            }
        
        return {
            "total_executions": self.total_executions,
            "avg_execution_time": sum(self.execution_times) / len(self.execution_times),
            "min_execution_time": min(self.execution_times),
            "max_execution_time": max(self.execution_times),
            "std_execution_time": sum((t - sum(self.execution_times) / len(self.execution_times)) ** 2 for t in self.execution_times) ** 0.5 / len(self.execution_times)
        }
    
    def reset(self):
        """重置Graph状态"""
        for node in self.nodes.values():
            node.reset()
        
        self.data_cache.clear()
        self.execution_times.clear()
        self.total_executions = 0
        self._is_validated = False
        
        self.logger.info("Graph状态已重置")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "graph_id": self.graph_id,
            "nodes": {node_id: {
                "node_type": node.node_type.value,
                "implementation": node.implementation.value,
                "enabled": node.enabled,
                "config": node.config
            } for node_id, node in self.nodes.items()},
            "connections": {k: v for k, v in self.connections.items() if v},
            "node_order": self.node_order
        }
    
    def __repr__(self) -> str:
        return (f"Graph(id={self.graph_id}, nodes={len(self.nodes)}, "
                f"validated={self._is_validated})")

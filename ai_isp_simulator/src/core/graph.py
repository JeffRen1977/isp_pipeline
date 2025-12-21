"""
Graph Engine: Implements directed graph data flow processing
Supports dynamic configuration, node management, data flow control, etc.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from collections import defaultdict, deque
from .node import Node, NodeType, NodeStatus
from .frame import Frame
from .flow import Flow


class Graph:
    """ISP Graph Engine"""
    
    def __init__(self, graph_id: str):
        """
        Initialize Graph
        
        Args:
            graph_id: Graph unique identifier
        """
        self.graph_id = graph_id
        
        # Node management
        self.nodes: Dict[str, Node] = {}
        self.node_order: List[str] = []  # Node order after topological sort
        
        # Connection management
        self.connections: Dict[str, List[str]] = defaultdict(list)  # output -> input node list
        self.reverse_connections: Dict[str, List[str]] = defaultdict(list)  # input -> output node list
        
        # Data cache
        self.data_cache: Dict[str, Dict[str, Union[Frame, Flow, Any]]] = {}
        
        # Performance statistics
        self.execution_times: List[float] = []
        self.total_executions = 0
        
        # Logging
        self.logger = logging.getLogger(f"Graph_{graph_id}")
        
        # Validation status
        self._is_validated = False
    
    def add_node(self, node: Node) -> bool:
        """
        Add node
        
        Args:
            node: Node to add
            
        Returns:
            Whether addition was successful
        """
        if node.node_id in self.nodes:
            self.logger.warning(f"Node {node.node_id} already exists")
            return False
        
        self.nodes[node.node_id] = node
        self.node_order.append(node.node_id)
        self._is_validated = False
        
        self.logger.info(f"Added node: {node.node_id}")
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove node
        
        Args:
            node_id: Node ID to remove
            
        Returns:
            Whether removal was successful
        """
        if node_id not in self.nodes:
            self.logger.warning(f"Node {node_id} does not exist")
            return False
        
        # Remove related connections
        self._remove_node_connections(node_id)
        
        # Remove node
        del self.nodes[node_id]
        if node_id in self.node_order:
            self.node_order.remove(node_id)
        
        # Clear cache
        if node_id in self.data_cache:
            del self.data_cache[node_id]
        
        self._is_validated = False
        
        self.logger.info(f"Removed node: {node_id}")
        return True
    
    def connect_nodes(self, from_node_id: str, to_node_id: str, from_port: str = "output", to_port: str = "input") -> bool:
        """
        Connect two nodes
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            from_port: Source node output port
            to_port: Target node input port
            
        Returns:
            Whether connection was successful
        """
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            self.logger.error(f"Node does not exist: {from_node_id} -> {to_node_id}")
            return False
        
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        
        if not from_node.has_output_port(from_port):
            self.logger.error(f"Source node {from_node_id} does not have output port {from_port}")
            return False
        
        if not to_node.has_input_port(to_port):
            self.logger.error(f"Target node {to_node_id} does not have input port {to_port}")
            return False
        
        # Establish connection
        connection_key = f"{from_node_id}:{from_port}"
        self.connections[connection_key].append(f"{to_node_id}:{to_port}")
        self.reverse_connections[f"{to_node_id}:{to_port}"].append(connection_key)
        
        self._is_validated = False
        
        self.logger.info(f"Connected nodes: {from_node_id}:{from_port} -> {to_node_id}:{to_port}")
        return True
    
    def disconnect_nodes(self, from_node_id: str, to_node_id: str, from_port: str = "output", to_port: str = "input") -> bool:
        """
        Disconnect two nodes
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            from_port: Source node output port
            to_port: Target node input port
            
        Returns:
            Whether disconnection was successful
        """
        connection_key = f"{from_node_id}:{from_port}"
        target_key = f"{to_node_id}:{to_port}"
        
        if connection_key in self.connections and target_key in self.connections[connection_key]:
            self.connections[connection_key].remove(target_key)
            if target_key in self.reverse_connections:
                self.reverse_connections[target_key].remove(connection_key)
            
            self._is_validated = False
            self.logger.info(f"Disconnected nodes: {from_node_id}:{from_port} -> {to_node_id}:{to_port}")
            return True
        
        return False
    
    def validate(self) -> bool:
        """
        Validate Graph validity
        
        Returns:
            Whether valid
        """
        try:
            # Check for cycles
            if self._has_cycle():
                self.logger.error("Graph contains cycles")
                return False
            
            # Check input and output nodes
            input_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.INPUT]
            output_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.OUTPUT]
            
            if not input_nodes:
                self.logger.error("Graph has no input nodes")
                return False
            
            if not output_nodes:
                self.logger.error("Graph has no output nodes")
                return False
            
            # Topological sort
            self._topological_sort()
            
            self._is_validated = True
            self.logger.info("Graph validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Graph validation failed: {e}")
            return False
    
    def _has_cycle(self) -> bool:
        """Check for cycles (using DFS)"""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
            
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # Check all output connections
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
        """Topological sort"""
        in_degree = defaultdict(int)
        
        # Calculate in-degree
        for node_id in self.nodes:
            for port in self.nodes[node_id].get_input_ports():
                target_key = f"{node_id}:{port}"
                for source in self.reverse_connections.get(target_key, []):
                    source_node_id = source.split(":")[0]
                    in_degree[node_id] += 1
        
        # Topological sort
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
            raise ValueError("Graph contains cycles, cannot perform topological sort")
        
        self.node_order = sorted_nodes
    
    def execute(self, inputs: Optional[Dict[str, Union[Frame, Flow, Any]]] = None) -> Dict[str, Union[Frame, Flow, Any]]:
        """
        Execute Graph
        
        Args:
            inputs: Input data dictionary
            
        Returns:
            Output data dictionary
        """
        if not self._is_validated:
            if not self.validate():
                raise RuntimeError("Graph validation failed, cannot execute")
        
        start_time = time.time()
        self.total_executions += 1
        
        try:
            # Initialize data cache
            self.data_cache.clear()
            if inputs:
                self.data_cache.update(inputs)
            
            # Execute nodes in topological order
            for node_id in self.node_order:
                node = self.nodes[node_id]
                
                if not node.is_ready():
                    self.logger.warning(f"Node {node_id} is not ready, skipping")
                    continue
                
                # Collect input data
                node_inputs = self._collect_node_inputs(node_id)
                
                # Execute node
                node_outputs = node.process(node_inputs)
                
                # Cache output data
                self.data_cache[node_id] = node_outputs
            
            # Collect final outputs
            outputs = self._collect_final_outputs()
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            # Keep only the most recent 100 records
            if len(self.execution_times) > 100:
                self.execution_times.pop(0)
            
            self.logger.info(f"Graph execution completed, time taken: {execution_time:.3f}s")
            return outputs
            
        except Exception as e:
            self.logger.error(f"Graph execution failed: {e}")
            raise
    
    def _collect_node_inputs(self, node_id: str) -> Dict[str, Union[Frame, Flow, Any]]:
        """Collect node input data"""
        node = self.nodes[node_id]
        inputs = {}
        
        for port in node.get_input_ports():
            target_key = f"{node_id}:{port}"
            
            # First check if there's initial input data for this node and port
            initial_input_key = f"{node_id}:{port}"
            if initial_input_key in self.data_cache:
                inputs[port] = self.data_cache[initial_input_key]
                continue
            
            # Then look for connected nodes
            for source in self.reverse_connections.get(target_key, []):
                source_node_id, source_port = source.split(":")
                
                # Get data from cache
                if source_node_id in self.data_cache:
                    source_data = self.data_cache[source_node_id]
                    if source_port in source_data:
                        inputs[port] = source_data[source_port]
                        break
        
        return inputs
    
    def _collect_final_outputs(self) -> Dict[str, Union[Frame, Flow, Any]]:
        """Collect final output data"""
        outputs = {}
        
        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.OUTPUT:
                # Collect output node inputs as final outputs
                node_inputs = self._collect_node_inputs(node_id)
                for port, data in node_inputs.items():
                    outputs[f"{node_id}_{port}"] = data
        
        return outputs
    
    def _remove_node_connections(self, node_id: str):
        """Remove all connections for a node"""
        node = self.nodes[node_id]
        
        # Remove output connections
        for port in node.get_output_ports():
            connection_key = f"{node_id}:{port}"
            if connection_key in self.connections:
                del self.connections[connection_key]
        
        # Remove input connections
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
        """Get specified node"""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get nodes of specified type"""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def get_connections(self) -> List[Tuple[str, str]]:
        """Get all connections"""
        connections = []
        for source, targets in self.connections.items():
            for target in targets:
                connections.append((source, target))
        return connections
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
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
        """Reset Graph state"""
        for node in self.nodes.values():
            node.reset()
        
        self.data_cache.clear()
        self.execution_times.clear()
        self.total_executions = 0
        self._is_validated = False
        
        self.logger.info("Graph state reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
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

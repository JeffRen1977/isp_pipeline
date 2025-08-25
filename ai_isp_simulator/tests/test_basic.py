#!/usr/bin/env python3
"""
基本功能测试
测试AI ISP仿真器的核心功能
"""

import sys
import unittest
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.frame import Frame, ColorFormat, BayerPattern, CameraParams, ExposureParams, IMUData
from core.flow import Flow, AlignmentInfo, DepthInfo
from core.node import Node, ProcessingNode, InputNode, OutputNode, ImplementationType, NodeType, NodeStatus
from core.graph import Graph


class TestFrame(unittest.TestCase):
    """测试Frame类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        self.camera_params = CameraParams(
            focal_length=35.0,
            f_number=2.8,
            sensor_size=(36.0, 24.0),
            principal_point=(50.0, 50.0)
        )
        self.exposure_params = ExposureParams(
            exposure_time=1.0/30.0,
            iso=100,
            gain=1.0
        )
        self.imu_data = IMUData(
            timestamp=time.time(),
            gyroscope=np.array([0.1, 0.2, 0.3]),
            accelerometer=np.array([0.4, 0.5, 0.6])
        )
    
    def test_frame_creation(self):
        """测试Frame创建"""
        frame = Frame(
            data=self.test_data,
            color_format=ColorFormat.RAW_BAYER,
            bayer_pattern=BayerPattern.RGGB,
            timestamp=time.time(),
            camera_params=self.camera_params,
            exposure_params=self.exposure_params,
            imu_data=self.imu_data
        )
        
        self.assertEqual(frame.shape, (100, 100))
        self.assertEqual(frame.color_format, ColorFormat.RAW_BAYER)
        self.assertEqual(frame.bayer_pattern, BayerPattern.RGGB)
        self.assertEqual(frame.height, 100)
        self.assertEqual(frame.width, 100)
        self.assertEqual(frame.channels, 1)
    
    def test_frame_validation(self):
        """测试Frame验证"""
        # 测试无效的Bayer模式
        with self.assertRaises(ValueError):
            Frame(
                data=self.test_data,
                color_format=ColorFormat.RAW_BAYER,
                bayer_pattern=None
            )
        
        # 测试无效的数据维度
        invalid_data = np.random.randint(0, 255, (100, 100, 3, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            Frame(
                data=invalid_data,
                color_format=ColorFormat.RAW_BAYER,
                bayer_pattern=BayerPattern.RGGB
            )
    
    def test_frame_copy(self):
        """测试Frame复制"""
        frame = Frame(
            data=self.test_data,
            color_format=ColorFormat.RAW_BAYER,
            bayer_pattern=BayerPattern.RGGB
        )
        
        frame_copy = frame.copy()
        self.assertEqual(frame.shape, frame_copy.shape)
        self.assertEqual(frame.color_format, frame_copy.color_format)
        self.assertEqual(frame.bayer_pattern, frame_copy.bayer_pattern)
        
        # 修改原始数据，确保复制是深拷贝
        frame.data[0, 0] = 999
        self.assertNotEqual(frame.data[0, 0], frame_copy.data[0, 0])


class TestFlow(unittest.TestCase):
    """测试Flow类"""
    
    def setUp(self):
        """测试前准备"""
        self.frames = []
        for i in range(3):
            data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            frame = Frame(
                data=data,
                color_format=ColorFormat.RAW_BAYER,
                bayer_pattern=BayerPattern.RGGB,
                timestamp=time.time() + i
            )
            self.frames.append(frame)
    
    def test_flow_creation(self):
        """测试Flow创建"""
        flow = Flow(
            frames=self.frames,
            reference_frame_idx=0
        )
        
        self.assertEqual(flow.num_frames, 3)
        self.assertEqual(flow.shape, (100, 100))
        self.assertEqual(flow.reference_frame_idx, 0)
        self.assertEqual(flow.reference_frame, self.frames[0])
    
    def test_flow_validation(self):
        """测试Flow验证"""
        # 测试不同尺寸的帧
        invalid_frame = Frame(
            data=np.random.randint(0, 255, (200, 200), dtype=np.uint8),
            color_format=ColorFormat.RAW_BAYER,
            bayer_pattern=BayerPattern.RGGB
        )
        
        with self.assertRaises(ValueError):
            Flow(frames=self.frames + [invalid_frame])
    
    def test_flow_operations(self):
        """测试Flow操作"""
        flow = Flow(frames=self.frames)
        
        # 测试添加帧
        new_frame = Frame(
            data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            color_format=ColorFormat.RAW_BAYER,
            bayer_pattern=BayerPattern.RGGB
        )
        flow.add_frame(new_frame)
        self.assertEqual(flow.num_frames, 4)
        
        # 测试移除帧
        removed_frame = flow.remove_frame(0)
        self.assertEqual(flow.num_frames, 3)
        self.assertEqual(removed_frame, self.frames[0])
        
        # 测试设置参考帧
        flow.set_reference_frame(1)
        self.assertEqual(flow.reference_frame_idx, 1)
        self.assertEqual(flow.reference_frame, self.frames[1])


class TestGraph(unittest.TestCase):
    """测试Graph类"""
    
    def setUp(self):
        """测试前准备"""
        self.graph = Graph("test_graph")
        
        # 创建测试节点
        self.input_node = InputNode("input_node")
        self.processing_node = ProcessingNode("processing_node")
        self.output_node = OutputNode("output_node")
    
    def test_graph_creation(self):
        """测试Graph创建"""
        self.assertEqual(self.graph.graph_id, "test_graph")
        self.assertEqual(len(self.graph.nodes), 0)
        self.assertFalse(self.graph._is_validated)
    
    def test_add_nodes(self):
        """测试添加节点"""
        self.assertTrue(self.graph.add_node(self.input_node))
        self.assertTrue(self.graph.add_node(self.processing_node))
        self.assertTrue(self.graph.add_node(self.output_node))
        
        self.assertEqual(len(self.graph.nodes), 3)
        self.assertIn("input_node", self.graph.nodes)
        self.assertIn("processing_node", self.graph.nodes)
        self.assertIn("output_node", self.graph.nodes)
    
    def test_connect_nodes(self):
        """测试连接节点"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.processing_node)
        self.graph.add_node(self.output_node)
        
        # 连接节点
        self.assertTrue(self.graph.connect_nodes("input_node", "processing_node"))
        self.assertTrue(self.graph.connect_nodes("processing_node", "output_node"))
        
        # 验证连接
        connections = self.graph.get_connections()
        self.assertEqual(len(connections), 2)
    
    def test_graph_validation(self):
        """测试Graph验证"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.processing_node)
        self.graph.add_node(self.output_node)
        
        self.graph.connect_nodes("input_node", "processing_node")
        self.graph.connect_nodes("processing_node", "output_node")
        
        # 验证Graph
        self.assertTrue(self.graph.validate())
        self.assertTrue(self.graph._is_validated)
    
    def test_invalid_graph(self):
        """测试无效Graph"""
        # 创建有环的Graph
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.processing_node)
        
        # 创建环
        self.graph.connect_nodes("input_node", "processing_node")
        self.graph.connect_nodes("processing_node", "input_node")
        
        # 验证应该失败
        self.assertFalse(self.graph.validate())


if __name__ == "__main__":
    # 添加缺失的import
    import numpy as np
    import time
    
    # 运行测试
    unittest.main()

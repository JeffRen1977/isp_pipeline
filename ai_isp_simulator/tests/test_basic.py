#!/usr/bin/env python3
"""
Basic Functionality Tests
Tests core functionality of AI ISP simulator
"""

import sys
import unittest
import time
from pathlib import Path

import numpy as np

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.frame import Frame, ColorFormat, BayerPattern, CameraParams, ExposureParams, IMUData
from core.flow import Flow, AlignmentInfo, DepthInfo
from core.node import Node, ProcessingNode, InputNode, OutputNode, ImplementationType, NodeType, NodeStatus
from core.graph import Graph


class ConcreteProcessingNode(ProcessingNode):
    """Concrete subclass for testing (ProcessingNode is abstract)."""
    def _process_classic(self, inputs):
        return inputs

    def _process_ai(self, inputs):
        return inputs


class TestFrame(unittest.TestCase):
    """Test Frame class"""
    
    def setUp(self):
        """Set up before tests"""
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
        """Test Frame creation"""
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
        """Test Frame validation"""
        # Test invalid Bayer pattern
        with self.assertRaises(ValueError):
            Frame(
                data=self.test_data,
                color_format=ColorFormat.RAW_BAYER,
                bayer_pattern=None
            )
        
        # Test invalid data dimensions
        invalid_data = np.random.randint(0, 255, (100, 100, 3, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            Frame(
                data=invalid_data,
                color_format=ColorFormat.RAW_BAYER,
                bayer_pattern=BayerPattern.RGGB
            )
    
    def test_frame_copy(self):
        """Test Frame copy"""
        frame = Frame(
            data=self.test_data,
            color_format=ColorFormat.RAW_BAYER,
            bayer_pattern=BayerPattern.RGGB
        )
        
        frame_copy = frame.copy()
        self.assertEqual(frame.shape, frame_copy.shape)
        self.assertEqual(frame.color_format, frame_copy.color_format)
        self.assertEqual(frame.bayer_pattern, frame_copy.bayer_pattern)
        
        # Modify original data to ensure copy is deep copy
        frame.data[0, 0] = 999
        self.assertNotEqual(frame.data[0, 0], frame_copy.data[0, 0])


class TestFlow(unittest.TestCase):
    """Test Flow class"""
    
    def setUp(self):
        """Set up before tests"""
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
        """Test Flow creation"""
        flow = Flow(
            frames=self.frames,
            reference_frame_idx=0
        )
        
        self.assertEqual(flow.num_frames, 3)
        self.assertEqual(flow.shape, (100, 100))
        self.assertEqual(flow.reference_frame_idx, 0)
        self.assertEqual(flow.reference_frame, self.frames[0])
    
    def test_flow_validation(self):
        """Test Flow validation"""
        # Test frames of different sizes
        invalid_frame = Frame(
            data=np.random.randint(0, 255, (200, 200), dtype=np.uint8),
            color_format=ColorFormat.RAW_BAYER,
            bayer_pattern=BayerPattern.RGGB
        )
        
        with self.assertRaises(ValueError):
            Flow(frames=self.frames + [invalid_frame])
    
    def test_flow_operations(self):
        """Test Flow operations"""
        flow = Flow(frames=self.frames)
        
        # Test adding frame
        new_frame = Frame(
            data=np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            color_format=ColorFormat.RAW_BAYER,
            bayer_pattern=BayerPattern.RGGB
        )
        flow.add_frame(new_frame)
        self.assertEqual(flow.num_frames, 4)
        
        # Test removing frame
        removed_frame = flow.remove_frame(0)
        self.assertEqual(flow.num_frames, 3)
        self.assertIsNotNone(removed_frame)

        # Test setting reference frame
        flow.set_reference_frame(1)
        self.assertEqual(flow.reference_frame_idx, 1)


class TestGraph(unittest.TestCase):
    """Test Graph class"""
    
    def setUp(self):
        """Set up before tests"""
        self.graph = Graph("test_graph")
        
        # Create test nodes
        self.input_node = InputNode("input_node")
        self.processing_node = ConcreteProcessingNode("processing_node")
        self.output_node = OutputNode("output_node")
    
    def test_graph_creation(self):
        """Test Graph creation"""
        self.assertEqual(self.graph.graph_id, "test_graph")
        self.assertEqual(len(self.graph.nodes), 0)
        self.assertFalse(self.graph._is_validated)
    
    def test_add_nodes(self):
        """Test adding nodes"""
        self.assertTrue(self.graph.add_node(self.input_node))
        self.assertTrue(self.graph.add_node(self.processing_node))
        self.assertTrue(self.graph.add_node(self.output_node))
        
        self.assertEqual(len(self.graph.nodes), 3)
        self.assertIn("input_node", self.graph.nodes)
        self.assertIn("processing_node", self.graph.nodes)
        self.assertIn("output_node", self.graph.nodes)
    
    def test_connect_nodes(self):
        """Test connecting nodes"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.processing_node)
        self.graph.add_node(self.output_node)
        
        # Connect nodes
        self.assertTrue(self.graph.connect_nodes("input_node", "processing_node"))
        self.assertTrue(self.graph.connect_nodes("processing_node", "output_node"))
        
        # Validate connections
        connections = self.graph.get_connections()
        self.assertEqual(len(connections), 2)
    
    def test_graph_validation(self):
        """Test Graph validation"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.processing_node)
        self.graph.add_node(self.output_node)
        
        self.graph.connect_nodes("input_node", "processing_node")
        self.graph.connect_nodes("processing_node", "output_node")
        
        # Validate Graph
        self.assertTrue(self.graph.validate())
        self.assertTrue(self.graph._is_validated)
    
    def test_invalid_graph(self):
        """Test invalid Graph"""
        # Create Graph with cycles
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.processing_node)
        
        # Create cycle
        self.graph.connect_nodes("input_node", "processing_node")
        self.graph.connect_nodes("processing_node", "input_node")
        
        # Validation should fail
        self.assertFalse(self.graph.validate())


if __name__ == "__main__":
    unittest.main()

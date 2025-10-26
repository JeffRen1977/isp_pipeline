"""
Color Correction Matrix Node: Applies color correction matrix to adjust color reproduction
"""

import numpy as np
from typing import Dict, Any, Union
from ...core.node import Node, NodeType, ImplementationType
from ...core.frame import Frame, ColorFormat


class ColorCorrectionNode(Node):
    """Color correction matrix node"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            node_id="color_correction",
            node_type=NodeType.PROCESSING,
            implementation=ImplementationType.CLASSIC,
            config=config or {}
        )
        
        # Configure input/output ports
        self.input_ports = ["rgb_input"]
        self.output_ports = ["corrected_output"]
        
        # Default configuration - 3x3 color correction matrix
        # This matrix converts from sensor RGB to standard RGB
        default_matrix = np.array([
            [1.5, -0.3, -0.2],   # Red channel
            [-0.1, 1.4, -0.3],   # Green channel  
            [-0.1, -0.2, 1.3]    # Blue channel
        ], dtype=np.float32)
        
        self.config.setdefault("color_matrix", default_matrix)
        self.config.setdefault("apply_clipping", True)
        self.config.setdefault("clip_range", (0.0, 1.0))
        
    def process(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """
        Process inputs and apply color correction
        
        Args:
            inputs: Input dictionary with 'rgb_input' key
            
        Returns:
            Output dictionary with 'corrected_output' key
        """
        if "rgb_input" not in inputs:
            raise ValueError("Input 'rgb_input' is required")
        
        rgb_input = inputs["rgb_input"]
        
        # Handle both Frame and numpy array inputs
        if isinstance(rgb_input, Frame):
            rgb_data = rgb_input.data
            frame = rgb_input
        else:
            rgb_data = rgb_input
            frame = None
        
        # Apply color correction
        corrected_data = self._apply_color_correction(rgb_data)
        
        # Create output Frame if input was Frame
        if frame is not None:
            output_frame = Frame(
                data=corrected_data,
                color_format=ColorFormat.RGB,
                bayer_pattern=frame.bayer_pattern,
                timestamp=frame.timestamp,
                camera_params=frame.camera_params,
                exposure_params=frame.exposure_params,
                imu_data=frame.imu_data,
                metadata=frame.metadata
            )
            output_frame.metadata["color_corrected"] = True
            output_frame.metadata["color_matrix"] = self.config["color_matrix"].tolist()
        else:
            output_frame = corrected_data
        
        return {"corrected_output": output_frame}
    
    def _apply_color_correction(self, rgb_data: np.ndarray) -> np.ndarray:
        """
        Apply color correction matrix to RGB data
        
        Args:
            rgb_data: Input RGB data (H, W, 3)
            
        Returns:
            Corrected RGB data
        """
        height, width, channels = rgb_data.shape
        
        if channels != 3:
            raise ValueError(f"Expected 3 channels, got {channels}")
        
        # Reshape to (N, 3) for matrix multiplication
        rgb_reshaped = rgb_data.reshape(-1, 3)
        
        # Apply color correction matrix
        color_matrix = self.config["color_matrix"]
        corrected_reshaped = np.dot(rgb_reshaped, color_matrix.T)
        
        # Reshape back to (H, W, 3)
        corrected_data = corrected_reshaped.reshape(height, width, 3)
        
        # Apply clipping if enabled
        if self.config["apply_clipping"]:
            min_val, max_val = self.config["clip_range"]
            corrected_data = np.clip(corrected_data, min_val, max_val)
        
        return corrected_data
    
    def set_color_matrix(self, matrix: np.ndarray):
        """
        Set custom color correction matrix
        
        Args:
            matrix: 3x3 color correction matrix
        """
        if matrix.shape != (3, 3):
            raise ValueError("Color matrix must be 3x3")
        
        self.config["color_matrix"] = matrix.astype(np.float32)
    
    def get_color_matrix(self) -> np.ndarray:
        """
        Get current color correction matrix
        
        Returns:
            Current color correction matrix
        """
        return self.config["color_matrix"].copy()






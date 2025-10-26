"""
Defective Pixel Correction Node: Identifies and corrects bad pixels using median-based algorithm
"""

import numpy as np
from typing import Dict, Any, Union
from ...core.node import Node, NodeType, ImplementationType
from ...core.frame import Frame, ColorFormat


class DefectivePixelNode(Node):
    """Defective pixel correction node using median-based algorithm"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            node_id="defective_pixel",
            node_type=NodeType.PROCESSING,
            implementation=ImplementationType.CLASSIC,
            config=config or {}
        )
        
        # Configure input/output ports
        self.input_ports = ["raw_input"]
        self.output_ports = ["corrected_output"]
        
        # Default configuration
        self.config.setdefault("detection_threshold", 50)
        self.config.setdefault("correction_method", "median")
        self.config.setdefault("enable_auto_detection", True)
        
    def process(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """
        Process inputs and apply defective pixel correction
        
        Args:
            inputs: Input dictionary with 'raw_input' key
            
        Returns:
            Output dictionary with 'corrected_output' key
        """
        if "raw_input" not in inputs:
            raise ValueError("Input 'raw_input' is required")
        
        raw_input = inputs["raw_input"]
        
        # Handle both Frame and numpy array inputs
        if isinstance(raw_input, Frame):
            raw_data = raw_input.data
            frame = raw_input
        else:
            raw_data = raw_input
            frame = None
        
        # Apply defective pixel correction
        corrected_data = self._apply_defective_pixel_correction(raw_data)
        
        # Create output Frame if input was Frame
        if frame is not None:
            output_frame = Frame(
                data=corrected_data,
                color_format=frame.color_format,
                bayer_pattern=frame.bayer_pattern,
                timestamp=frame.timestamp,
                camera_params=frame.camera_params,
                exposure_params=frame.exposure_params,
                imu_data=frame.imu_data,
                metadata=frame.metadata
            )
            output_frame.metadata["defective_pixel_corrected"] = True
            output_frame.metadata["correction_method"] = self.config["correction_method"]
            output_frame.metadata["detection_threshold"] = self.config["detection_threshold"]
        else:
            output_frame = corrected_data
        
        return {"corrected_output": output_frame}
    
    def _apply_defective_pixel_correction(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Apply defective pixel correction using median-based algorithm
        
        Args:
            raw_data: Input RAW data
            
        Returns:
            Corrected RAW data
        """
        corrected_image = np.copy(raw_data)
        rows, cols = raw_data.shape
        
        # Skip edge pixels to avoid boundary issues
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                # Calculate the difference between the pixel and its 8 neighbors
                neighbor_values = [
                    raw_data[y-1, x-1], raw_data[y-1, x], raw_data[y-1, x+1],
                    raw_data[y, x-1], raw_data[y, x+1],
                    raw_data[y+1, x-1], raw_data[y+1, x], raw_data[y+1, x+1]
                ]
                
                # Find the minimum and maximum of the neighbors
                min_neighbor = np.min(neighbor_values)
                max_neighbor = np.max(neighbor_values)
                
                # Check if the current pixel is a "hot" or "cold" pixel
                is_hot = corrected_image[y, x] > (max_neighbor + self.config["detection_threshold"])
                is_cold = corrected_image[y, x] < (min_neighbor - self.config["detection_threshold"])
                
                if is_hot or is_cold:
                    # Replace the defective pixel with the median of its neighbors
                    corrected_image[y, x] = np.median(neighbor_values)
        
        return corrected_image





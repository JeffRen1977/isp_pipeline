"""
Black Level Correction Node: Corrects for sensor's inherent black level offset
"""

import numpy as np
from typing import Dict, Any, Union
from ...core.node import Node, NodeType, ImplementationType
from ...core.frame import Frame, ColorFormat


class BlackLevelNode(Node):
    """Black level correction node"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            node_id="black_level",
            node_type=NodeType.PROCESSING,
            implementation=ImplementationType.CLASSIC,
            config=config or {}
        )
        
        # Configure input/output ports
        self.input_ports = ["raw_input"]
        self.output_ports = ["corrected_output"]
        
        # Default configuration
        self.config.setdefault("black_level_r", 64)
        self.config.setdefault("black_level_gr", 64)
        self.config.setdefault("black_level_gb", 64)
        self.config.setdefault("black_level_b", 64)
        self.config.setdefault("bayer_pattern", "rggb")
        
    def process(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """
        Process inputs and apply black level correction
        
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
        
        # Apply black level correction
        corrected_data = self._apply_black_level_correction(raw_data)
        
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
            output_frame.metadata["black_level_corrected"] = True
            output_frame.metadata["black_level_values"] = {
                "r": self.config["black_level_r"],
                "gr": self.config["black_level_gr"],
                "gb": self.config["black_level_gb"],
                "b": self.config["black_level_b"]
            }
        else:
            output_frame = corrected_data
        
        return {"corrected_output": output_frame}
    
    def _apply_black_level_correction(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Apply black level correction to RAW data using simple subtraction
        
        Args:
            raw_data: Input RAW data
            
        Returns:
            Corrected RAW data
        """
        # Ensure the image data type can handle subtraction without negative wrapping
        corrected = raw_data.astype(np.float32)
        
        # Get black level values
        bl_r = self.config["black_level_r"]
        bl_gr = self.config["black_level_gr"]
        bl_gb = self.config["black_level_gb"]
        bl_b = self.config["black_level_b"]
        
        # Apply correction based on Bayer pattern
        if self.config["bayer_pattern"] == "rggb":
            # R G
            # G B
            corrected[0::2, 0::2] -= bl_r    # Red pixels
            corrected[0::2, 1::2] -= bl_gr   # Green pixels (first row)
            corrected[1::2, 0::2] -= bl_gb   # Green pixels (second row)
            corrected[1::2, 1::2] -= bl_b    # Blue pixels
        elif self.config["bayer_pattern"] == "grbg":
            # G R
            # B G
            corrected[0::2, 0::2] -= bl_gr   # Green pixels (first row)
            corrected[0::2, 1::2] -= bl_r    # Red pixels
            corrected[1::2, 0::2] -= bl_b    # Blue pixels
            corrected[1::2, 1::2] -= bl_gb   # Green pixels (second row)
        elif self.config["bayer_pattern"] == "gbrg":
            # G B
            # R G
            corrected[0::2, 0::2] -= bl_gr   # Green pixels (first row)
            corrected[0::2, 1::2] -= bl_b    # Blue pixels
            corrected[1::2, 0::2] -= bl_r    # Red pixels
            corrected[1::2, 1::2] -= bl_gb   # Green pixels (second row)
        elif self.config["bayer_pattern"] == "bggr":
            # B G
            # G R
            corrected[0::2, 0::2] -= bl_b    # Blue pixels
            corrected[0::2, 1::2] -= bl_gr   # Green pixels (first row)
            corrected[1::2, 0::2] -= bl_gb   # Green pixels (second row)
            corrected[1::2, 1::2] -= bl_r    # Red pixels
        else:
            raise ValueError(f"Unsupported Bayer pattern: {self.config['bayer_pattern']}")
        
        # Clip to prevent negative values
        corrected[corrected < 0] = 0
        
        return corrected





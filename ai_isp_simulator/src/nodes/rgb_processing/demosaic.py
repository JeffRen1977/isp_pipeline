"""
Demosaic Node: Reconstructs full-color (RGB) information from the Bayer pattern of the raw sensor data
"""

import numpy as np
from typing import Dict, Any, Union
from ...core.node import Node, NodeType, ImplementationType
from ...core.frame import Frame, ColorFormat


class DemosaicNode(Node):
    """Demosaic node for converting Bayer pattern to RGB using bilinear interpolation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            node_id="demosaic",
            node_type=NodeType.PROCESSING,
            implementation=ImplementationType.CLASSIC,
            config=config or {}
        )
        
        # Configure input/output ports
        self.input_ports = ["raw_input"]
        self.output_ports = ["rgb_output"]
        
        # Default configuration
        self.config.setdefault("bayer_pattern", "rggb")
        self.config.setdefault("interpolation_method", "bilinear")
        
    def process(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """
        Process inputs and demosaic to RGB
        
        Args:
            inputs: Input dictionary with 'raw_input' key
            
        Returns:
            Output dictionary with 'rgb_output' key
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
        
        # Demosaic to RGB
        rgb_data = self._demosaic_bayer(raw_data)
        
        # Create output Frame if input was Frame
        if frame is not None:
            output_frame = Frame(
                data=rgb_data,
                color_format=ColorFormat.RGB,
                bayer_pattern=frame.bayer_pattern,
                timestamp=frame.timestamp,
                camera_params=frame.camera_params,
                exposure_params=frame.exposure_params,
                imu_data=frame.imu_data,
                metadata=frame.metadata
            )
            output_frame.metadata["demosaiced"] = True
            output_frame.metadata["interpolation_method"] = self.config["interpolation_method"]
        else:
            output_frame = rgb_data
        
        return {"rgb_output": output_frame}
    
    def _demosaic_bayer(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Demosaic Bayer pattern to RGB using bilinear interpolation
        
        Args:
            raw_data: Input RAW data
            
        Returns:
            RGB image (H, W, 3)
        """
        if self.config["interpolation_method"] == "bilinear":
            return self._bilinear_interpolation(raw_data)
        else:
            # Default to bilinear
            return self._bilinear_interpolation(raw_data)
    
    def _bilinear_interpolation(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Bilinear interpolation for demosaicing
        
        Args:
            raw_data: Input RAW data
            
        Returns:
            RGB image
        """
        H, W = raw_data.shape
        rgb_image = np.zeros((H, W, 3), dtype=raw_data.dtype)
        
        if self.config["bayer_pattern"] == "rggb":
            # R G R G R G ...
            # G B G B G B ...
            # R G R G R G ...
            # G B G B G B ...
            
            # Red channel (R pixels are at even rows, odd columns)
            rgb_image[1::2, 0::2, 0] = raw_data[1::2, 0::2]
            # Green channel
            rgb_image[0::2, 0::2, 1] = raw_data[0::2, 0::2]  # G at top-left
            rgb_image[1::2, 1::2, 1] = raw_data[1::2, 1::2]  # G at bottom-right
            # Blue channel (B pixels are at odd rows, even columns)
            rgb_image[0::2, 1::2, 2] = raw_data[0::2, 1::2]
            
            # Interpolate missing values
            # For R channel
            rgb_image[0::2, 0::2, 0] = (raw_data[0::2, 1::2] + raw_data[0::2, 0::2]) / 2  # R at G
            
            # Handle edge cases for horizontal interpolation
            for y in range(0, H, 2):
                for x in range(1, W, 2):
                    if x + 1 < W:
                        rgb_image[y, x, 0] = (raw_data[y, x-1] + raw_data[y, x+1]) / 2
                    else:
                        rgb_image[y, x, 0] = raw_data[y, x-1]
            
            # Handle edge cases for R at B positions
            for y in range(1, H, 2):
                for x in range(1, W, 2):
                    neighbors = []
                    if y > 0: neighbors.append(raw_data[y-1, x])
                    if y + 1 < H: neighbors.append(raw_data[y+1, x])
                    if x > 0: neighbors.append(raw_data[y, x-1])
                    if x + 1 < W: neighbors.append(raw_data[y, x+1])
                    if neighbors:
                        rgb_image[y, x, 0] = np.mean(neighbors)
            
            # For G channel
            for y in range(0, H, 2):
                for x in range(1, W, 2):
                    neighbors = []
                    if x > 0: neighbors.append(raw_data[y, x-1])
                    if x + 1 < W: neighbors.append(raw_data[y, x+1])
                    if y + 1 < H: neighbors.append(raw_data[y+1, x])
                    if y > 0: neighbors.append(raw_data[y-1, x])
                    if neighbors:
                        rgb_image[y, x, 1] = np.mean(neighbors)
            
            for y in range(1, H, 2):
                for x in range(0, W, 2):
                    neighbors = []
                    if y > 0: neighbors.append(raw_data[y-1, x])
                    if y + 1 < H: neighbors.append(raw_data[y+1, x])
                    if x + 1 < W: neighbors.append(raw_data[y, x+1])
                    if x > 0: neighbors.append(raw_data[y, x-1])
                    if neighbors:
                        rgb_image[y, x, 1] = np.mean(neighbors)
            
            # For B channel
            for y in range(0, H, 2):
                for x in range(0, W, 2):
                    neighbors = []
                    if x + 1 < W: neighbors.append(raw_data[y, x+1])
                    if y + 1 < H: neighbors.append(raw_data[y+1, x])
                    if x > 0: neighbors.append(raw_data[y, x-1])
                    if y > 0: neighbors.append(raw_data[y-1, x])
                    if neighbors:
                        rgb_image[y, x, 2] = np.mean(neighbors)
            
            for y in range(1, H, 2):
                for x in range(0, W, 2):
                    rgb_image[y, x, 2] = (raw_data[y, x+1] + raw_data[y-1, x]) / 2
        
        return rgb_image





"""
Lens Shading Correction Node: Corrects vignetting using concentric circle model
"""

import numpy as np
from typing import Dict, Any, Union
from ...core.node import Node, NodeType, ImplementationType
from ...core.frame import Frame, ColorFormat


class LensShadingNode(Node):
    """Lens shading correction node using concentric circle model"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            node_id="lens_shading",
            node_type=NodeType.PROCESSING,
            implementation=ImplementationType.CLASSIC,
            config=config or {}
        )
        
        # Configure input/output ports
        self.input_ports = ["raw_input"]
        self.output_ports = ["corrected_output"]
        
        # Default configuration
        self.config.setdefault("correction_strength", 0.8)
        self.config.setdefault("bayer_pattern", "rggb")
        self.config.setdefault("use_radial_model", True)
        
    def process(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """
        Process inputs and apply lens shading correction
        
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
        
        # Apply lens shading correction
        corrected_data = self._apply_lens_shading_correction(raw_data)
        
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
            output_frame.metadata["lens_shading_corrected"] = True
            output_frame.metadata["correction_strength"] = self.config["correction_strength"]
        else:
            output_frame = corrected_data
        
        return {"corrected_output": output_frame}
    
    def _apply_lens_shading_correction(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Apply lens shading correction using concentric circle model
        
        Args:
            raw_data: Input RAW data
            
        Returns:
            Corrected RAW data
        """
        H, W = raw_data.shape
        center_y, center_x = H // 2, W // 2
        
        # Create a meshgrid to represent pixel coordinates
        y, x = np.ogrid[:H, :W]
        
        # Calculate the distance from the center for each pixel
        distance_map = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        # Normalize distance to 0-1 range
        max_distance = np.sqrt(center_y**2 + center_x**2)
        normalized_distance = distance_map / max_distance
        
        # Create a gain map; farther pixels get more gain
        # The correction strength controls the intensity
        k = self.config["correction_strength"] * 0.0001
        gain_map = 1.0 + k * (normalized_distance ** 2)
        
        # Apply the gain map to the raw data
        corrected_image = raw_data.astype(np.float32) * gain_map
        
        # Clip values to prevent overflow
        if np.issubdtype(raw_data.dtype, np.integer):
            max_value = np.iinfo(raw_data.dtype).max
            corrected_image = np.clip(corrected_image, 0, max_value)
            return corrected_image.astype(raw_data.dtype)
        else:
            # For float types, just clip to reasonable range
            corrected_image = np.clip(corrected_image, 0, 1.0)
            return corrected_image





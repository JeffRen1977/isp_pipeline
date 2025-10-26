"""
Tone Mapping Node: Applies global tone mapping operator (Reinhard's operator)
"""

import numpy as np
from typing import Dict, Any, Union
from ...core.node import Node, NodeType, ImplementationType
from ...core.frame import Frame, ColorFormat


class ToneMappingNode(Node):
    """Tone mapping node using Reinhard's global operator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            node_id="tone_mapping",
            node_type=NodeType.PROCESSING,
            implementation=ImplementationType.CLASSIC,
            config=config or {}
        )
        
        # Configure input/output ports
        self.input_ports = ["raw_input"]
        self.output_ports = ["mapped_output"]
        
        # Default configuration
        self.config.setdefault("mapping_method", "reinhard")
        self.config.setdefault("exposure", 1.2)
        self.config.setdefault("gamma", 2.2)
        self.config.setdefault("white_point", 1.0)
        self.config.setdefault("black_point", 0.0)
        
    def process(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """
        Process inputs and apply tone mapping
        
        Args:
            inputs: Input dictionary with 'raw_input' key
            
        Returns:
            Output dictionary with 'mapped_output' key
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
        
        # Apply tone mapping
        mapped_data = self._apply_tone_mapping(raw_data)
        
        # Create output Frame if input was Frame
        if frame is not None:
            output_frame = Frame(
                data=mapped_data,
                color_format=frame.color_format,
                bayer_pattern=frame.bayer_pattern,
                timestamp=frame.timestamp,
                camera_params=frame.camera_params,
                exposure_params=frame.exposure_params,
                imu_data=frame.imu_data,
                metadata=frame.metadata
            )
            output_frame.metadata["tone_mapped"] = True
            output_frame.metadata["mapping_method"] = self.config["mapping_method"]
            output_frame.metadata["exposure"] = self.config["exposure"]
            output_frame.metadata["gamma"] = self.config["gamma"]
        else:
            output_frame = mapped_data
        
        return {"mapped_output": output_frame}
    
    def _apply_tone_mapping(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Apply tone mapping using Reinhard's global operator
        
        Args:
            raw_data: Input RAW data
            
        Returns:
            Tone mapped data
        """
        if self.config["mapping_method"] == "reinhard":
            return self._reinhard_tone_mapping(raw_data)
        else:
            # Default to Reinhard
            return self._reinhard_tone_mapping(raw_data)
    
    def _reinhard_tone_mapping(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Apply Reinhard's global tone mapping operator
        
        Args:
            raw_data: Input RAW data
            
        Returns:
            Tone mapped data
        """
        # Ensure data is in float format
        hdr_image = raw_data.astype(np.float32)
        
        # Normalize to 0-1 range if needed
        if hdr_image.max() > 1.0:
            hdr_image = hdr_image / hdr_image.max()
        
        # Calculate luminance using ITU-R BT.709 coefficients
        luminance = (0.2126 * hdr_image[:, :, 0] + 
                    0.7152 * hdr_image[:, :, 1] + 
                    0.0722 * hdr_image[:, :, 2])
        
        # Find the average log luminance for global scaling
        log_luminance = np.log(np.maximum(1e-6, luminance))
        avg_log_luminance = np.exp(np.mean(log_luminance))
        
        # Apply the global tone mapping formula
        a = self.config["exposure"]  # Key value, typically 0.18 for middle gray
        scaled_luminance = (a / avg_log_luminance) * luminance
        
        # Apply the tone mapping function to the scaled luminance
        tone_mapped_luminance = (scaled_luminance * (1 + scaled_luminance)) / (1 + scaled_luminance)
        
        # Re-apply the tone-mapped luminance to the original color ratios
        tone_mapped_image = np.zeros_like(hdr_image)
        
        # Avoid division by zero
        safe_luminance = np.maximum(luminance, 1e-6)
        
        tone_mapped_image[:, :, 0] = (hdr_image[:, :, 0] / safe_luminance) * tone_mapped_luminance
        tone_mapped_image[:, :, 1] = (hdr_image[:, :, 1] / safe_luminance) * tone_mapped_luminance
        tone_mapped_image[:, :, 2] = (hdr_image[:, :, 2] / safe_luminance) * tone_mapped_luminance
        
        # Apply gamma correction
        gamma = self.config["gamma"]
        tone_mapped_image = np.power(tone_mapped_image, 1.0 / gamma)
        
        # Clip to valid range
        tone_mapped_image = np.clip(tone_mapped_image, 0, 1)
        
        return tone_mapped_image





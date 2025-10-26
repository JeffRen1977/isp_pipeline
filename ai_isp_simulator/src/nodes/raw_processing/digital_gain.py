"""
Digital Gain Node: Applies digital amplification to the image signal
"""

import numpy as np
from typing import Dict, Any, Union
from ...core.node import Node, NodeType, ImplementationType
from ...core.frame import Frame


class DigitalGainNode(Node):
    """Digital gain node"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            node_id="digital_gain",
            node_type=NodeType.PROCESSING,
            implementation=ImplementationType.CLASSIC,
            config=config or {}
        )
        
        # Configure input/output ports
        self.input_ports = ["raw_input"]
        self.output_ports = ["gained_output"]
        
        # Default configuration
        self.config.setdefault("gain_r", 1.0)
        self.config.setdefault("gain_gr", 1.0)
        self.config.setdefault("gain_gb", 1.0)
        self.config.setdefault("gain_b", 1.0)
        self.config.setdefault("bayer_pattern", "rggb")
        self.config.setdefault("clip_max", 1023)  # For 10-bit data
        
    def process(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """
        Process inputs and apply digital gain
        
        Args:
            inputs: Input dictionary with 'raw_input' key
            
        Returns:
            Output dictionary with 'gained_output' key
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
        
        # Apply digital gain
        gained_data = self._apply_digital_gain(raw_data)
        
        # Create output Frame if input was Frame
        if frame is not None:
            output_frame = Frame(
                data=gained_data,
                color_format=frame.color_format,
                bayer_pattern=frame.bayer_pattern,
                timestamp=frame.timestamp,
                camera_params=frame.camera_params,
                exposure_params=frame.exposure_params,
                imu_data=frame.imu_data,
                metadata=frame.metadata
            )
            output_frame.metadata["digital_gain_applied"] = True
            output_frame.metadata["gain_values"] = {
                "r": self.config["gain_r"],
                "gr": self.config["gain_gr"],
                "gb": self.config["gain_gb"],
                "b": self.config["gain_b"]
            }
        else:
            output_frame = gained_data
        
        return {"gained_output": output_frame}
    
    def _apply_digital_gain(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Apply digital gain to RAW data
        
        Args:
            raw_data: Input RAW data
            
        Returns:
            Gained RAW data
        """
        height, width = raw_data.shape
        gained = raw_data.copy().astype(np.float32)
        
        # Get gain values
        gain_r = self.config["gain_r"]
        gain_gr = self.config["gain_gr"]
        gain_gb = self.config["gain_gb"]
        gain_b = self.config["gain_b"]
        
        # Apply gain based on Bayer pattern
        if self.config["bayer_pattern"] == "rggb":
            # R G
            # G B
            gained[0::2, 0::2] *= gain_r    # Red pixels
            gained[0::2, 1::2] *= gain_gr   # Green pixels (first row)
            gained[1::2, 0::2] *= gain_gb   # Green pixels (second row)
            gained[1::2, 1::2] *= gain_b    # Blue pixels
        elif self.config["bayer_pattern"] == "grbg":
            # G R
            # B G
            gained[0::2, 0::2] *= gain_gr   # Green pixels (first row)
            gained[0::2, 1::2] *= gain_r    # Red pixels
            gained[1::2, 0::2] *= gain_b    # Blue pixels
            gained[1::2, 1::2] *= gain_gb   # Green pixels (second row)
        elif self.config["bayer_pattern"] == "gbrg":
            # G B
            # R G
            gained[0::2, 0::2] *= gain_gr   # Green pixels (first row)
            gained[0::2, 1::2] *= gain_b    # Blue pixels
            gained[1::2, 0::2] *= gain_r    # Red pixels
            gained[1::2, 1::2] *= gain_gb   # Green pixels (second row)
        elif self.config["bayer_pattern"] == "bggr":
            # B G
            # G R
            gained[0::2, 0::2] *= gain_b    # Blue pixels
            gained[0::2, 1::2] *= gain_gr   # Green pixels (first row)
            gained[1::2, 0::2] *= gain_gb   # Green pixels (second row)
            gained[1::2, 1::2] *= gain_r    # Red pixels
        else:
            raise ValueError(f"Unsupported Bayer pattern: {self.config['bayer_pattern']}")
        
        # Clip to prevent overflow
        gained = np.clip(gained, 0, self.config["clip_max"])
        
        return gained






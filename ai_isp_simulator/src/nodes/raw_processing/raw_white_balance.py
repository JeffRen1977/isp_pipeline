"""
RAW White Balance Node: Applies white balance gains to Bayer pattern
This should be applied BEFORE demosaic

Implements the Gray World AWB algorithm:
- Assumes the average color of the scene is neutral gray
- Calculates gains to make R_avg = G_avg = B_avg
- Applies gains to the RAW Bayer pattern before demosaic
"""

import numpy as np
from typing import Dict, Any, Union
from ...core.node import Node, NodeType, ImplementationType
from ...core.frame import Frame, ColorFormat, BayerPattern


class RawWhiteBalanceNode(Node):
    """
    RAW white balance node - applies gains to Bayer pattern
    
    Implements Gray World Auto White Balance (AWB) algorithm:
    1. Calculates average Red, Green, and Blue values from Bayer statistics
    2. Uses Green as reference (less affected by light variations)
    3. Computes R_gain = G_avg / R_avg and B_gain = G_avg / B_avg
    4. Applies gains to RAW Bayer pattern (before demosaic)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            node_id="raw_white_balance",
            node_type=NodeType.PROCESSING,
            implementation=ImplementationType.CLASSIC,
            config=config or {}
        )
        
        # Configure input/output ports
        self.input_ports = ["raw_input"]
        self.output_ports = ["corrected_output"]
        
        # Default configuration
        self.config.setdefault("method", "gray_world")  # gray_world, manual
        self.config.setdefault("gain_r", 1.0)  # Manual R gain
        self.config.setdefault("gain_b", 1.0)  # Manual B gain
        self.config.setdefault("max_gain", 2.0)  # Maximum allowed gain
        self.config.setdefault("min_gain", 0.5)  # Minimum allowed gain
        self.config.setdefault("bayer_pattern", "rggb")
        
    def process(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """
        Process inputs and apply white balance to RAW Bayer pattern
        
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
            bayer_pattern = raw_input.bayer_pattern
        else:
            raw_data = raw_input
            frame = None
            bayer_pattern = BayerPattern.RGGB  # Default
        
        # Apply white balance
        corrected_data = self._apply_white_balance(raw_data, bayer_pattern)
        
        # Create output Frame if input was Frame
        if frame is not None:
            output_frame = Frame(
                data=corrected_data,
                color_format=ColorFormat.RAW_BAYER,
                bayer_pattern=frame.bayer_pattern,
                timestamp=frame.timestamp,
                camera_params=frame.camera_params,
                exposure_params=frame.exposure_params,
                imu_data=frame.imu_data,
                metadata=frame.metadata
            )
            output_frame.metadata["raw_white_balance_applied"] = True
        else:
            output_frame = corrected_data
        
        return {"corrected_output": output_frame}
    
    def _apply_white_balance(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """
        Apply white balance gains to RAW Bayer pattern
        
        Args:
            raw_data: Input RAW Bayer data
            bayer_pattern: Bayer pattern type
            
        Returns:
            Corrected RAW data
        """
        method = self.config["method"]
        
        if method == "gray_world":
            return self._gray_world_awb(raw_data, bayer_pattern)
        else:
            # Manual gains
            return self._manual_awb(raw_data, bayer_pattern)
    
    def _gray_world_awb(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """
        Gray World AWB algorithm on Bayer pattern
        
        Algorithm:
        1. Extract R, Gr, Gb, B pixels from Bayer pattern
        2. Calculate average values: R_avg, G_avg, B_avg
        3. Compute gains: R_gain = G_avg / R_avg, B_gain = G_avg / B_avg
        4. Apply gains to make average color neutral (R_avg = G_avg = B_avg)
        
        Reference: Gray World assumes average color of scene is neutral gray
        """
        # Convert to float32 for processing
        raw_float = raw_data.astype(np.float32)
        
        # Extract channels based on Bayer pattern
        if bayer_pattern == BayerPattern.RGGB:
            r_pixels = raw_float[0::2, 0::2]  # Red pixels
            gr_pixels = raw_float[0::2, 1::2]  # Green-Red pixels
            gb_pixels = raw_float[1::2, 0::2]  # Green-Blue pixels
            b_pixels = raw_float[1::2, 1::2]   # Blue pixels
        elif bayer_pattern == BayerPattern.GRBG:
            gr_pixels = raw_float[0::2, 0::2]
            r_pixels = raw_float[0::2, 1::2]
            b_pixels = raw_float[1::2, 0::2]
            gb_pixels = raw_float[1::2, 1::2]
        elif bayer_pattern == BayerPattern.GBRG:
            gb_pixels = raw_float[0::2, 0::2]
            b_pixels = raw_float[0::2, 1::2]
            r_pixels = raw_float[1::2, 0::2]
            gr_pixels = raw_float[1::2, 1::2]
        elif bayer_pattern == BayerPattern.BGGR:
            b_pixels = raw_float[0::2, 0::2]
            gb_pixels = raw_float[0::2, 1::2]
            gr_pixels = raw_float[1::2, 0::2]
            r_pixels = raw_float[1::2, 1::2]
        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
        
        # Calculate average values for each channel
        r_mean = np.mean(r_pixels)
        gr_mean = np.mean(gr_pixels)
        gb_mean = np.mean(gb_pixels)
        b_mean = np.mean(b_pixels)
        
        # Green average (average of Gr and Gb)
        # Green is used as reference because it's less affected by light variations
        g_mean = (gr_mean + gb_mean) / 2.0
        
        # Calculate AWB gains using Green as reference
        # R_gain = G_avg / R_avg, B_gain = G_avg / B_avg
        # This makes the average color neutral (R_avg = G_avg = B_avg)
        if r_mean > 0.001 and b_mean > 0.001:
            r_gain = g_mean / r_mean
            b_gain = g_mean / b_mean
        else:
            # Avoid division by zero - default to no correction
            r_gain = 1.0
            b_gain = 1.0
        
        # Limit gains to prevent overcorrection
        # Typical range: 0.5x to 2.0x
        r_gain = np.clip(r_gain, self.config["min_gain"], self.config["max_gain"])
        b_gain = np.clip(b_gain, self.config["min_gain"], self.config["max_gain"])
        
        # Apply gains to Bayer pattern
        corrected = raw_float.copy()
        
        if bayer_pattern == BayerPattern.RGGB:
            corrected[0::2, 0::2] *= r_gain  # Apply R_gain to R pixels
            corrected[1::2, 1::2] *= b_gain  # Apply B_gain to B pixels
        elif bayer_pattern == BayerPattern.GRBG:
            corrected[0::2, 1::2] *= r_gain
            corrected[1::2, 0::2] *= b_gain
        elif bayer_pattern == BayerPattern.GBRG:
            corrected[1::2, 0::2] *= r_gain
            corrected[0::2, 1::2] *= b_gain
        elif bayer_pattern == BayerPattern.BGGR:
            corrected[1::2, 1::2] *= r_gain
            corrected[0::2, 0::2] *= b_gain
        
        # Clip values to valid range
        # Note: In ISP pipelines, AWB output can exceed input range due to gains
        # We keep the float32 representation for demosaic processing
        # A reasonable upper limit for 10-bit sensors is 2x the max value (2048)
        max_val = 4095.0  # Allow headroom for AWB gains
        corrected = np.clip(corrected, 0, max_val)
        
        return corrected
    
    def _manual_awb(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """
        Apply manual white balance gains
        """
        # Convert to float32
        corrected = raw_data.astype(np.float32)
        
        r_gain = self.config["gain_r"]
        b_gain = self.config["gain_b"]
        
        # Apply gains based on Bayer pattern
        if bayer_pattern == BayerPattern.RGGB:
            corrected[0::2, 0::2] *= r_gain  # R pixels
            corrected[1::2, 1::2] *= b_gain  # B pixels
        elif bayer_pattern == BayerPattern.GRBG:
            corrected[0::2, 1::2] *= r_gain  # R pixels
            corrected[1::2, 0::2] *= b_gain  # B pixels
        elif bayer_pattern == BayerPattern.GBRG:
            corrected[1::2, 0::2] *= r_gain  # R pixels
            corrected[0::2, 1::2] *= b_gain  # B pixels
        elif bayer_pattern == BayerPattern.BGGR:
            corrected[1::2, 1::2] *= r_gain  # R pixels
            corrected[0::2, 0::2] *= b_gain  # B pixels
        
        # Clip values to valid range
        max_val = 4095.0  # Allow headroom for gains
        corrected = np.clip(corrected, 0, max_val)
        
        return corrected


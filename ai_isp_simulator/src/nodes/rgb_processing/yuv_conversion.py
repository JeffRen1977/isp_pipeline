"""
YUV Conversion Node: Converts RGB data to YUV color space
"""

import numpy as np
from typing import Dict, Any, Union
from ...core.node import Node, NodeType, ImplementationType
from ...core.frame import Frame, ColorFormat


class YUVConversionNode(Node):
    """YUV conversion node using BT.709 color space"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            node_id="yuv_conversion",
            node_type=NodeType.PROCESSING,
            implementation=ImplementationType.CLASSIC,
            config=config or {}
        )
        
        # Configure input/output ports
        self.input_ports = ["rgb_input", "tone_mapped_input"]
        self.output_ports = ["yuv_output"]
        
        # Default configuration
        self.config.setdefault("yuv_format", "yuv444")
        self.config.setdefault("color_space", "bt709")
        self.config.setdefault("full_range", True)
        
        # Color space conversion matrices
        self._setup_color_matrices()
        
    def _setup_color_matrices(self):
        """Setup color space conversion matrices"""
        if self.config["color_space"] == "bt709":
            # BT.709 conversion matrix
            self.rgb_to_yuv_matrix = np.array([
                [0.2126, 0.7152, 0.0722],   # Y
                [-0.1146, -0.3854, 0.5000], # U
                [0.5000, -0.4542, -0.0458]  # V
            ], dtype=np.float32)
        elif self.config["color_space"] == "bt601":
            # BT.601 conversion matrix
            self.rgb_to_yuv_matrix = np.array([
                [0.2990, 0.5870, 0.1140],   # Y
                [-0.1471, -0.2889, 0.4360], # U
                [0.6150, -0.5149, -0.1000]  # V
            ], dtype=np.float32)
        else:
            # Default to BT.709
            self.rgb_to_yuv_matrix = np.array([
                [0.2126, 0.7152, 0.0722],   # Y
                [-0.1146, -0.3854, 0.5000], # U
                [0.5000, -0.4542, -0.0458]  # V
            ], dtype=np.float32)
        
    def process(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """
        Process inputs and convert to YUV
        
        Args:
            inputs: Input dictionary with 'rgb_input' and/or 'tone_mapped_input' keys
            
        Returns:
            Output dictionary with 'yuv_output' key
        """
        # Get RGB input (either from demosaic or tone mapping)
        if "rgb_input" in inputs:
            rgb_input = inputs["rgb_input"]
        elif "tone_mapped_input" in inputs:
            rgb_input = inputs["tone_mapped_input"]
        else:
            raise ValueError("Either 'rgb_input' or 'tone_mapped_input' is required")
        
        # Handle both Frame and numpy array inputs
        if isinstance(rgb_input, Frame):
            rgb_data = rgb_input.data
            frame = rgb_input
        else:
            rgb_data = rgb_input
            frame = None
        
        # Convert to YUV
        yuv_data = self._convert_rgb_to_yuv(rgb_data)
        
        # Create output Frame
        if frame is not None:
            output_frame = Frame(
                data=yuv_data,
                color_format=ColorFormat.YUV,
                bayer_pattern=frame.bayer_pattern,
                timestamp=frame.timestamp,
                camera_params=frame.camera_params,
                exposure_params=frame.exposure_params,
                imu_data=frame.imu_data,
                metadata=frame.metadata
            )
            output_frame.metadata["yuv_converted"] = True
            output_frame.metadata["yuv_format"] = self.config["yuv_format"]
            output_frame.metadata["color_space"] = self.config["color_space"]
        else:
            # Create a minimal Frame if we don't have input frame metadata
            output_frame = Frame(
                data=yuv_data,
                color_format=ColorFormat.YUV,
                metadata={
                    "yuv_converted": True,
                    "yuv_format": self.config["yuv_format"],
                    "color_space": self.config["color_space"]
                }
            )
        
        return {"yuv_output": output_frame}
    
    def _convert_rgb_to_yuv(self, rgb_data: np.ndarray) -> np.ndarray:
        """
        Convert RGB data to YUV using BT.709 matrix
        
        Args:
            rgb_data: Input RGB data (H, W, 3)
            
        Returns:
            YUV data
        """
        height, width, channels = rgb_data.shape
        
        if channels != 3:
            raise ValueError(f"Expected 3 channels, got {channels}")
        
        # Reshape to (N, 3) for matrix multiplication
        rgb_reshaped = rgb_data.reshape(-1, 3)
        
        # Apply RGB to YUV conversion matrix
        yuv_reshaped = np.dot(rgb_reshaped, self.rgb_to_yuv_matrix.T)
        
        # Reshape back to (H, W, 3)
        yuv_data = yuv_reshaped.reshape(height, width, 3)
        
        # Apply range conversion if needed
        if not self.config["full_range"]:
            yuv_data = self._convert_to_limited_range(yuv_data)
        
        # Convert to specified YUV format
        if self.config["yuv_format"] == "yuv420":
            yuv_data = self._convert_to_yuv420(yuv_data)
        elif self.config["yuv_format"] == "yuv422":
            yuv_data = self._convert_to_yuv422(yuv_data)
        # yuv444 is already in the right format
        
        return yuv_data
    
    def _convert_to_limited_range(self, yuv_data: np.ndarray) -> np.ndarray:
        """
        Convert from full range (0-1) to limited range (16-235)
        
        Args:
            yuv_data: Input YUV data
            
        Returns:
            Limited range YUV data
        """
        # Scale from [0, 1] to [16, 235] for Y and [16, 240] for U, V
        yuv_limited = yuv_data.copy()
        
        # Y channel: [0, 1] -> [16, 235]
        yuv_limited[:, :, 0] = yuv_limited[:, :, 0] * 219 + 16
        
        # U, V channels: [0, 1] -> [16, 240]
        yuv_limited[:, :, 1:] = yuv_limited[:, :, 1:] * 224 + 16
        
        return yuv_limited
    
    def _convert_to_yuv420(self, yuv_data: np.ndarray) -> np.ndarray:
        """
        Convert to YUV420 format (Y full resolution, U and V half resolution)
        
        Args:
            yuv_data: Input YUV data
            
        Returns:
            YUV420 data as numpy array (H, W, 3) with U and V upsampled
        """
        height, width, channels = yuv_data.shape
        
        # Y channel remains full resolution
        y_channel = yuv_data[:, :, 0]
        
        # Downsample U and V channels to half resolution
        u_channel = yuv_data[::2, ::2, 1]  # Take every other pixel
        v_channel = yuv_data[::2, ::2, 2]  # Take every other pixel
        
        # Upsample U and V channels back to full resolution for visualization
        # This maintains the 3D array format that Frame expects
        u_full = np.zeros((height, width), dtype=yuv_data.dtype)
        v_full = np.zeros((height, width), dtype=yuv_data.dtype)
        
        # Fill the upsampled channels
        for i in range(0, height, 2):
            for j in range(0, width, 2):
                u_full[i:i+2, j:j+2] = u_channel[i//2, j//2]
                v_full[i:i+2, j:j+2] = v_channel[i//2, j//2]
        
        # Stack channels to create (H, W, 3) array
        yuv420_full = np.stack([y_channel, u_full, v_full], axis=2)
        
        return yuv420_full
    
    def _convert_to_yuv422(self, yuv_data: np.ndarray) -> np.ndarray:
        """
        Convert to YUV422 format (Y full resolution, U and V half horizontal resolution)
        
        Args:
            yuv_data: Input YUV data
            
        Returns:
            YUV422 data as numpy array (H, W, 3) with U and V horizontally upsampled
        """
        height, width, channels = yuv_data.shape
        
        # Y channel remains full resolution
        y_channel = yuv_data[:, :, 0]
        
        # Downsample U and V channels horizontally to half resolution
        u_channel = yuv_data[:, ::2, 1]  # Take every other column
        v_channel = yuv_data[:, ::2, 2]  # Take every other column
        
        # Upsample U and V channels horizontally back to full resolution
        u_full = np.zeros((height, width), dtype=yuv_data.dtype)
        v_full = np.zeros((height, width), dtype=yuv_data.dtype)
        
        # Fill the upsampled channels
        for j in range(0, width, 2):
            u_full[:, j:j+2] = u_channel[:, j//2:j//2+1]
            v_full[:, j:j+2] = v_channel[:, j//2:j//2+1]
        
        # Stack channels to create (H, W, 3) array
        yuv422_full = np.stack([y_channel, u_full, v_full], axis=2)
        
        return yuv422_full
    
    def set_color_space(self, color_space: str):
        """
        Set color space
        
        Args:
            color_space: Color space string (bt709, bt601, bt2020)
        """
        self.config["color_space"] = color_space
        self._setup_color_matrices()
    
    def get_color_space(self) -> str:
        """
        Get current color space
        
        Returns:
            Current color space
        """
        return self.config["color_space"]


"""
RAW Input Node: Loads and processes RAW sensor data
"""

import numpy as np
from typing import Dict, Any, Union
from ...core.node import Node, NodeType, ImplementationType
from ...core.frame import Frame, ColorFormat, BayerPattern


class RawInputNode(Node):
    """RAW sensor data input node"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            node_id="raw_input",
            node_type=NodeType.INPUT,
            implementation=ImplementationType.CLASSIC,
            config=config or {}
        )
        
        # Configure input/output ports
        self.input_ports = ["file_path"]
        self.output_ports = ["raw_output"]
        
        # Default configuration
        self.config.setdefault("row_stride", 5056)
        self.config.setdefault("width", 4000)
        self.config.setdefault("height", 3000)
        self.config.setdefault("bit_depth", 10)
        self.config.setdefault("bayer_pattern", BayerPattern.RGGB)
        
    def load_raw_file(self, file_path: str) -> np.ndarray:
        """
        Load RAW file with proper 10-bit unpacking
        
        Args:
            file_path: Path to RAW file
            
        Returns:
            Unpacked RAW data as uint16 array
        """
        # Load raw bytes
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
        
        file_size = len(raw_bytes)
        expected_size = self.config["height"] * self.config["row_stride"]
        
        print(f"File size: {file_size} bytes")
        print(f"Expected size for {self.config['width']}x{self.config['height']} with stride {self.config['row_stride']}: {expected_size} bytes")
        
        if file_size != expected_size:
            print(f"Warning: Expected {expected_size} bytes, got {file_size}")
        
        # Calculate actual data bytes per row (excluding padding)
        pixels_per_row = self.config["width"]
        bytes_per_pixel = self.config["bit_depth"] / 8.0  # 10-bit = 1.25 bytes
        data_bytes_per_row = int(pixels_per_row * bytes_per_pixel)
        padding_per_row = self.config["row_stride"] - data_bytes_per_row
        
        print(f"Pixels per row: {pixels_per_row}")
        print(f"Bytes per pixel: {bytes_per_pixel}")
        print(f"Data bytes per row: {data_bytes_per_row}")
        print(f"Padding per row: {padding_per_row} bytes")
        
        # Extract actual data bytes (excluding padding)
        total_data_bytes = self.config["height"] * data_bytes_per_row
        total_pixels = self.config["width"] * self.config["height"]
        
        print(f"Total actual data bytes (without padding): {total_data_bytes}")
        print(f"Total pixels: {total_pixels}")
        
        # Unpack 10-bit data
        if self.config["bit_depth"] == 10:
            return self._unpack_10bit_raw(raw_bytes, total_data_bytes, total_pixels)
        else:
            raise ValueError(f"Unsupported bit depth: {self.config['bit_depth']}")
    
    def _unpack_10bit_raw(self, raw_bytes: bytes, data_bytes: int, total_pixels: int) -> np.ndarray:
        """
        Unpack 10-bit RAW data (5 bytes per 4 pixels)
        
        Args:
            raw_bytes: Raw byte data
            data_bytes: Number of data bytes
            total_pixels: Total number of pixels
            
        Returns:
            Unpacked 10-bit data as uint16 array
        """
        required_bytes = int(total_pixels * 10 / 8)  # 10 bits per pixel
        print(f"Required bytes for 10-bit unpacking: {required_bytes}")
        print(f"Available bytes: {data_bytes}")
        
        if data_bytes < required_bytes:
            raise ValueError(f"Insufficient data: need {required_bytes} bytes, got {data_bytes}")
        
        # Unpack 10-bit data: 5 bytes contain 4 pixels
        unpacked = np.zeros(total_pixels, dtype=np.uint16)
        pixel_idx = 0
        
        for byte_idx in range(0, data_bytes - 4, 5):  # Process 5 bytes at a time
            if pixel_idx + 3 >= total_pixels:
                break
                
            # Extract 5 bytes
            b0, b1, b2, b3, b4 = raw_bytes[byte_idx:byte_idx + 5]
            
            # Unpack 4 pixels (10 bits each)
            # Pixel 0: bits 0-9 from bytes 0-1
            unpacked[pixel_idx] = ((b1 & 0x03) << 8) | b0
            
            # Pixel 1: bits 2-11 from bytes 1-2
            unpacked[pixel_idx + 1] = ((b2 & 0x0F) << 6) | (b1 >> 2)
            
            # Pixel 2: bits 4-13 from bytes 2-3
            unpacked[pixel_idx + 2] = ((b3 & 0x3F) << 4) | (b2 >> 4)
            
            # Pixel 3: bits 6-15 from bytes 3-4
            unpacked[pixel_idx + 3] = (b4 << 2) | (b3 >> 6)
            
            pixel_idx += 4
        
        # Reshape to image dimensions
        return unpacked.reshape(self.config["height"], self.config["width"])
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Union[Frame, np.ndarray]]:
        """
        Process inputs and return RAW data
        
        Args:
            inputs: Input dictionary with 'file_path' key
            
        Returns:
            Output dictionary with 'raw_output' key
        """
        if "file_path" not in inputs:
            raise ValueError("Input 'file_path' is required")
        
        file_path = inputs["file_path"]
        
        # Load and unpack RAW data
        raw_data = self.load_raw_file(file_path)
        
        # Create Frame object
        frame = Frame(
            data=raw_data,
            color_format=ColorFormat.RAW_BAYER,
            bayer_pattern=self.config["bayer_pattern"],
            metadata={
                "width": self.config["width"],
                "height": self.config["height"],
                "bit_depth": self.config["bit_depth"],
                "row_stride": self.config["row_stride"]
            }
        )
        
        return {"raw_output": frame}

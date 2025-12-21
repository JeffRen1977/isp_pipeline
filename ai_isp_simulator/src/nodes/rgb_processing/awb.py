"""
White Balance Node: Automatic and manual white balance
Supports multiple algorithm implementations
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Union
from ...core.node import ProcessingNode, ImplementationType
from ...core.frame import Frame, ColorFormat


class AWBNode(ProcessingNode):
    """White balance node"""
    
    def __init__(
        self,
        node_id: str,
        config: Optional[Dict[str, Any]] = None,
        implementation: ImplementationType = ImplementationType.CLASSIC,
        enabled: bool = True
    ):
        """
        Initialize white balance node
        
        Args:
            node_id: Node ID
            config: Configuration parameters
            implementation: Implementation type
            enabled: Whether enabled
        """
        super().__init__(node_id, config, implementation, enabled)
        
        # Default configuration
        default_config = {
            "method": "gray_world",  # gray_world, white_patch, retinex, ai
            "ai_model_path": "",
            "temperature": 5500,  # Color temperature (K)
            "tint": 0.0,  # Tint offset
            "adaptive": False,  # Whether to enable adaptive white balance
            "adaptive_window": 10,  # Adaptive window size
            "gray_world_config": {
                "saturation_threshold": 0.8,  # Saturation threshold
                "brightness_threshold": 0.1    # Brightness threshold
            },
            "white_patch_config": {
                "patch_size": 0.1,  # White patch size ratio
                "brightness_threshold": 0.9,  # Brightness threshold
                "saturation_threshold": 0.1   # Saturation threshold
            },
            "retinex_config": {
                "sigma": 30.0,  # Gaussian filter standard deviation
                "scale": 1.0    # Scale factor
            }
        }
        
        self.config.update(default_config)
        if config:
            self.config.update(config)
        
        # AI model (lazy loading)
        self._ai_model = None
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        methods = ["gray_world", "white_patch", "retinex", "ai"]
        if self.config["method"] not in methods:
            raise ValueError(f"Unsupported white balance method: {self.config['method']}")
        
        if self.implementation == ImplementationType.AI and not self.config["ai_model_path"]:
            self.logger.warning("AI mode: model path not specified, will use classic method")
    
    def _process_classic(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """Traditional algorithm implementation"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("Input data is empty")
        
        if isinstance(input_data, Frame):
            rgb_data = input_data.data
            metadata = input_data
        else:
            rgb_data = input_data
            metadata = None
        
        # Check input format
        if len(rgb_data.shape) != 3 or rgb_data.shape[2] != 3:
            raise ValueError("Input must be a 3-channel RGB image")
        
        # Execute white balance
        method = self.config["method"]
        if method == "gray_world":
            balanced_data = self._gray_world_awb(rgb_data)
        elif method == "white_patch":
            balanced_data = self._white_patch_awb(rgb_data)
        elif method == "retinex":
            balanced_data = self._retinex_awb(rgb_data)
        else:
            balanced_data = self._manual_awb(rgb_data)
        
        # Create output Frame
        if metadata:
            output_frame = Frame(
                data=balanced_data,
                color_format=ColorFormat.RGB,
                timestamp=metadata.timestamp,
                camera_params=metadata.camera_params,
                exposure_params=metadata.exposure_params,
                imu_data=metadata.imu_data,
                metadata=metadata.metadata
            )
        else:
            output_frame = Frame(
                data=balanced_data,
                color_format=ColorFormat.RGB
            )
        
        return {"output": output_frame}
    
    def _process_ai(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """AI algorithm implementation"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("Input data is empty")
        
        if isinstance(input_data, Frame):
            rgb_data = input_data.data
            metadata = input_data
        else:
            rgb_data = input_data
            metadata = None
        
        # Load AI model
        if self._ai_model is None:
            self._load_ai_model()
        
        # Execute AI white balance
        if self._ai_model is not None:
            balanced_data = self._ai_awb(rgb_data)
        else:
            # Fallback to classic method
            self.logger.warning("AI model loading failed, using classic method")
            return self._process_classic(inputs)
        
        # Create output Frame
        if metadata:
            output_frame = Frame(
                data=balanced_data,
                color_format=ColorFormat.RGB,
                timestamp=metadata.timestamp,
                camera_params=metadata.camera_params,
                exposure_params=metadata.exposure_params,
                imu_data=metadata.imu_data,
                metadata=metadata.metadata
            )
        else:
            output_frame = Frame(
                data=balanced_data,
                color_format=ColorFormat.RGB
            )
        
        return {"output": output_frame}
    
    def _gray_world_awb(self, rgb_data: np.ndarray) -> np.ndarray:
        """Gray World white balance - fixed version"""
        # Handle input data type, ensure normalization to [0, 1]
        if rgb_data.dtype == np.uint8:
            rgb_float = rgb_data.astype(np.float32) / 255.0
        elif rgb_data.dtype == np.uint16:
            rgb_float = rgb_data.astype(np.float32) / 65535.0
        else:  # float32 or float64
            # Ensure data is in reasonable range
            rgb_float = np.clip(rgb_data, 0, 1).astype(np.float32)
        
        # Calculate average value for each channel
        r_mean = np.mean(rgb_float[:, :, 0])
        g_mean = np.mean(rgb_float[:, :, 1])
        b_mean = np.mean(rgb_float[:, :, 2])
        
        # Gray World: assumes gray means equal channel averages
        # Calculate gains to make three channel averages equal
        avg_mean = (r_mean + g_mean + b_mean) / 3.0
        
        # Calculate gains
        r_gain = avg_mean / r_mean if r_mean > 0.001 else 1.0
        g_gain = avg_mean / g_mean if g_mean > 0.001 else 1.0
        b_gain = avg_mean / b_mean if b_mean > 0.001 else 1.0
        
        # Limit gain range to avoid overcorrection (0.5x to 2.0x)
        r_gain = np.clip(r_gain, 0.5, 2.0)
        g_gain = np.clip(g_gain, 0.5, 2.0)
        b_gain = np.clip(b_gain, 0.5, 2.0)
        
        # Apply gains
        balanced_data = rgb_float.copy()
        balanced_data[:, :, 0] = rgb_float[:, :, 0] * r_gain
        balanced_data[:, :, 1] = rgb_float[:, :, 1] * g_gain
        balanced_data[:, :, 2] = rgb_float[:, :, 2] * b_gain
        
        # Clip to [0, 1]
        balanced_data = np.clip(balanced_data, 0, 1)
        
        # Return according to input data type
        if rgb_data.dtype == np.uint8:
            return (balanced_data * 255).astype(np.uint8)
        elif rgb_data.dtype == np.uint16:
            return (balanced_data * 65535).astype(np.uint16)
        else:  # float32 or float64
            return balanced_data.astype(np.float32)
    
    def _white_patch_awb(self, rgb_data: np.ndarray) -> np.ndarray:
        """White patch white balance"""
        config = self.config["white_patch_config"]
        patch_size = config["patch_size"]
        bright_threshold = config["brightness_threshold"]
        sat_threshold = config["saturation_threshold"]
        
        # Convert to float
        rgb_float = rgb_data.astype(np.float32) / 255.0
        
        # Calculate HSV
        hsv = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2HSV)
        hsv_float = hsv.astype(np.float32)
        hsv_float[:, :, 1] /= 255.0  # Normalize saturation
        hsv_float[:, :, 2] /= 255.0  # Normalize brightness
        
        # Create mask
        mask = (hsv_float[:, :, 2] > bright_threshold) & (hsv_float[:, :, 1] < sat_threshold)
        
        if np.sum(mask) == 0:
            # If no suitable pixels, use gray world method
            return self._gray_world_awb(rgb_data)
        
        # Find brightest pixels
        bright_pixels = rgb_float[mask]
        if len(bright_pixels) == 0:
            return self._gray_world_awb(rgb_data)
        
        # Calculate white patch RGB values
        white_rgb = np.mean(bright_pixels, axis=0)
        
        # Calculate gains (assuming white patch should be white)
        gains = [1.0 / val if val > 0 else 1.0 for val in white_rgb]
        
        # Apply white balance
        balanced_data = np.zeros_like(rgb_float)
        for c in range(3):
            balanced_data[:, :, c] = rgb_float[:, :, c] * gains[c]
        
        # Clip to [0, 1] range
        balanced_data = np.clip(balanced_data, 0, 1)
        
        # Convert back to uint8
        return (balanced_data * 255).astype(np.uint8)
    
    def _retinex_awb(self, rgb_data: np.ndarray) -> np.ndarray:
        """Retinex white balance"""
        config = self.config["retinex_config"]
        sigma = config["sigma"]
        scale = config["scale"]
        
        # Convert to float
        rgb_float = rgb_data.astype(np.float32) / 255.0
        
        # Apply Gaussian filter
        gaussian = cv2.GaussianBlur(rgb_float, (0, 0), sigma)
        
        # Calculate logarithm
        log_rgb = np.log(rgb_float + 1e-6)
        log_gaussian = np.log(gaussian + 1e-6)
        
        # Calculate Retinex
        retinex = (log_rgb - log_gaussian) * scale
        
        # Normalize
        retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
        
        # Convert back to uint8
        return (retinex * 255).astype(np.uint8)
    
    def _manual_awb(self, rgb_data: np.ndarray) -> np.ndarray:
        """Manual white balance"""
        temperature = self.config["temperature"]
        tint = self.config["tint"]
        
        # Convert to float
        rgb_float = rgb_data.astype(np.float32) / 255.0
        
        # Simple white balance based on color temperature
        # Uses simplified color temperature to RGB gain mapping
        if temperature < 3000:  # Warm tone
            gains = [1.2, 1.0, 0.8]
        elif temperature < 5000:  # Neutral
            gains = [1.1, 1.0, 0.9]
        elif temperature < 7000:  # Cold tone
            gains = [0.9, 1.0, 1.1]
        else:  # Very cold
            gains = [0.8, 1.0, 1.2]
        
        # Apply tint offset
        gains[1] += tint * 0.1  # Green channel adjustment
        
        # Apply white balance
        balanced_data = np.zeros_like(rgb_float)
        for c in range(3):
            balanced_data[:, :, c] = rgb_float[:, :, c] * gains[c]
        
        # Clip to [0, 1] range
        balanced_data = np.clip(balanced_data, 0, 1)
        
        # Convert back to uint8
        return (balanced_data * 255).astype(np.uint8)
    
    def _ai_awb(self, rgb_data: np.ndarray) -> np.ndarray:
        """AI white balance"""
        if self._ai_model is None:
            raise RuntimeError("AI model not loaded")
        
        # Preprocess input data
        input_tensor = self._preprocess_for_ai(rgb_data)
        
        # Execute AI inference
        try:
            output_tensor = self._ai_model(input_tensor)
            balanced_data = self._postprocess_from_ai(output_tensor)
        except Exception as e:
            self.logger.error(f"AI inference failed: {e}")
            # Fallback to classic method
            return self._process_classic({"input": rgb_data})["output"].data
        
        return balanced_data
    
    def _load_ai_model(self):
        """Load AI model"""
        model_path = self.config["ai_model_path"]
        if not model_path:
            return
        
        try:
            # Should implement specific AI model loading logic here
            # Supports PyTorch, TensorFlow, ONNX, etc.
            self.logger.info(f"Loading AI model: {model_path}")
            
            # Placeholder implementation
            self._ai_model = None
            
        except Exception as e:
            self.logger.error(f"AI model loading failed: {e}")
            self._ai_model = None
    
    def _preprocess_for_ai(self, rgb_data: np.ndarray) -> np.ndarray:
        """Preprocess data for AI model"""
        # Normalize
        if rgb_data.dtype == np.uint8:
            rgb_data = rgb_data.astype(np.float32) / 255.0
        
        # Add batch dimension
        if len(rgb_data.shape) == 3:
            rgb_data = rgb_data[np.newaxis, :, :, :]
        
        return rgb_data
    
    def _postprocess_from_ai(self, output_tensor: np.ndarray) -> np.ndarray:
        """Postprocess data from AI model"""
        # Remove batch dimension
        if len(output_tensor.shape) == 4:
            output_tensor = output_tensor[0]
        
        # Denormalize
        output_tensor = np.clip(output_tensor, 0, 1)
        output_tensor = (output_tensor * 255).astype(np.uint8)
        
        return output_tensor
    
    def _on_implementation_changed(self):
        """Callback when implementation type changes"""
        if self.implementation == ImplementationType.AI:
            self._load_ai_model()
        else:
            self._ai_model = None

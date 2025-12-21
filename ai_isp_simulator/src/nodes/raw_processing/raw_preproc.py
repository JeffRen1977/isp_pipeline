"""
RAW Preprocessing Node: Includes BPC, BLC, LSC and other basic functions
Supports traditional algorithms and AI implementation
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Union
from ...core.node import ProcessingNode, ImplementationType
from ...core.frame import Frame, ColorFormat, BayerPattern


class RawPreprocNode(ProcessingNode):
    """RAW preprocessing node"""
    
    def __init__(
        self,
        node_id: str,
        config: Optional[Dict[str, Any]] = None,
        implementation: ImplementationType = ImplementationType.CLASSIC,
        enabled: bool = True
    ):
        """
        Initialize RAW preprocessing node
        
        Args:
            node_id: Node ID
            config: Configuration parameters
            implementation: Implementation type
            enabled: Whether enabled
        """
        super().__init__(node_id, config, implementation, enabled)
        
        # Default configuration
        default_config = {
            "bpc_enabled": True,  # Bad pixel correction
            "blc_enabled": True,  # Black level correction
            "lsc_enabled": True,  # Lens shading correction
            "lsc_method": "classic",  # classic, ai
            "ai_model_path": "",
            "bpc_config": {
                "threshold": 3.0,  # Bad pixel detection threshold
                "window_size": 5
            },
            "blc_config": {
                "black_level": 64,  # Black level value
                "method": "subtract"  # subtract, scale
            },
            "lsc_config": {
                "method": "polynomial",  # polynomial, lookup_table, ai
                "coefficients": [1.0, 0.1, 0.05, 0.01],  # Polynomial coefficients
                "center": None,  # Center point, None means image center
                "radius": None   # Radius, None means half of image diagonal
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
        if self.config["lsc_method"] not in ["classic", "ai"]:
            raise ValueError(f"Unsupported LSC method: {self.config['lsc_method']}")
        
        if self.implementation == ImplementationType.AI and not self.config["ai_model_path"]:
            self.logger.warning("AI mode: model path not specified, will use classic method")
    
    def _process_classic(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """Traditional algorithm implementation"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("Input data is empty")
        
        if isinstance(input_data, Frame):
            raw_data = input_data.data
            metadata = input_data
        else:
            raw_data = input_data
            metadata = None
        
        # Execute preprocessing
        processed_data = raw_data.copy()
        
        if self.config["bpc_enabled"]:
            processed_data = self._bad_pixel_correction(processed_data)
        
        if self.config["blc_enabled"]:
            processed_data = self._black_level_correction(processed_data)
        
        if self.config["lsc_enabled"]:
            processed_data = self._lens_shading_correction(processed_data)
        
        # Create output Frame
        if metadata:
            output_frame = Frame(
                data=processed_data,
                color_format=metadata.color_format,
                bayer_pattern=metadata.bayer_pattern,
                timestamp=metadata.timestamp,
                camera_params=metadata.camera_params,
                exposure_params=metadata.exposure_params,
                imu_data=metadata.imu_data,
                metadata=metadata.metadata
            )
        else:
            output_frame = Frame(
                data=processed_data,
                color_format=ColorFormat.RAW_BAYER
            )
        
        return {"output": output_frame}
    
    def _process_ai(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """AI algorithm implementation"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("Input data is empty")
        
        if isinstance(input_data, Frame):
            raw_data = input_data.data
            metadata = input_data
        else:
            raw_data = input_data
            metadata = None
        
        # Load AI model
        if self._ai_model is None:
            self._load_ai_model()
        
        # Execute AI preprocessing
        if self._ai_model is not None:
            processed_data = self._ai_preprocessing(raw_data)
        else:
            # Fallback to classic method
            self.logger.warning("AI model loading failed, using classic method")
            return self._process_classic(inputs)
        
        # Create output Frame
        if metadata:
            output_frame = Frame(
                data=processed_data,
                color_format=metadata.color_format,
                bayer_pattern=metadata.bayer_pattern,
                timestamp=metadata.timestamp,
                camera_params=metadata.camera_params,
                exposure_params=metadata.exposure_params,
                imu_data=metadata.imu_data,
                metadata=metadata.metadata
            )
        else:
            output_frame = Frame(
                data=processed_data,
                color_format=ColorFormat.RAW_BAYER
            )
        
        return {"output": output_frame}
    
    def _bad_pixel_correction(self, raw_data: np.ndarray) -> np.ndarray:
        """Bad pixel correction"""
        config = self.config["bpc_config"]
        threshold = config["threshold"]
        window_size = config["window_size"]
        
        height, width = raw_data.shape
        corrected_data = raw_data.copy()
        
        # Calculate local statistics
        half_window = window_size // 2
        
        for y in range(half_window, height - half_window):
            for x in range(half_window, width - half_window):
                # Extract local window
                window = raw_data[y-half_window:y+half_window+1, 
                                x-half_window:x+half_window+1]
                
                # Calculate local mean and standard deviation
                local_mean = np.mean(window)
                local_std = np.std(window)
                
                # Detect bad pixels
                current_pixel = raw_data[y, x]
                if abs(current_pixel - local_mean) > threshold * local_std:
                    # Replace bad pixel with median of neighboring pixels
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                neighbors.append(raw_data[ny, nx])
                    
                    if neighbors:
                        corrected_data[y, x] = np.median(neighbors)
        
        return corrected_data
    
    def _black_level_correction(self, raw_data: np.ndarray) -> np.ndarray:
        """Black level correction"""
        config = self.config["blc_config"]
        black_level = config["black_level"]
        method = config["method"]
        
        corrected_data = raw_data.copy()
        
        if method == "subtract":
            # Subtract black level
            corrected_data = np.clip(corrected_data - black_level, 0, None)
        elif method == "scale":
            # Scale correction
            max_value = np.max(raw_data)
            corrected_data = ((corrected_data - black_level) / 
                            (max_value - black_level) * max_value)
            corrected_data = np.clip(corrected_data, 0, max_value)
        
        return corrected_data
    
    def _lens_shading_correction(self, raw_data: np.ndarray) -> np.ndarray:
        """Lens shading correction"""
        config = self.config["lsc_config"]
        method = config["method"]
        
        if method == "polynomial":
            return self._lsc_polynomial(raw_data)
        elif method == "lookup_table":
            return self._lsc_lookup_table(raw_data)
        else:
            return raw_data
    
    def _lsc_polynomial(self, raw_data: np.ndarray) -> np.ndarray:
        """Polynomial lens shading correction"""
        config = self.config["lsc_config"]
        coefficients = config["coefficients"]
        
        height, width = raw_data.shape
        
        # Determine center point and radius
        if config["center"] is None:
            center_y, center_x = height // 2, width // 2
        else:
            center_y, center_x = config["center"]
        
        if config["radius"] is None:
            radius = np.sqrt(center_y**2 + center_x**2)
        else:
            radius = config["radius"]
        
        # Create correction map
        correction_map = np.ones_like(raw_data, dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                # Calculate distance to center
                dy = y - center_y
                dx = x - center_x
                distance = np.sqrt(dy**2 + dx**2)
                
                # Normalize distance
                normalized_distance = distance / radius
                
                # Apply polynomial correction
                correction = 1.0
                for i, coef in enumerate(coefficients):
                    correction += coef * (normalized_distance ** (i + 1))
                
                correction_map[y, x] = correction
        
        # Apply correction
        corrected_data = raw_data.astype(np.float32) * correction_map
        corrected_data = np.clip(corrected_data, 0, np.max(raw_data))
        
        return corrected_data.astype(raw_data.dtype)
    
    def _lsc_lookup_table(self, raw_data: np.ndarray) -> np.ndarray:
        """Lookup table lens shading correction"""
        # Simplified lookup table implementation
        # In actual applications, precomputed lookup tables should be used
        return self._lsc_polynomial(raw_data)
    
    def _ai_preprocessing(self, raw_data: np.ndarray) -> np.ndarray:
        """AI preprocessing"""
        if self._ai_model is None:
            raise RuntimeError("AI model not loaded")
        
        # Preprocess input data
        input_tensor = self._preprocess_for_ai(raw_data)
        
        # Execute AI inference
        try:
            output_tensor = self._ai_model(input_tensor)
            processed_data = self._postprocess_from_ai(output_tensor)
        except Exception as e:
            self.logger.error(f"AI inference failed: {e}")
            # Fallback to classic method
            return self._process_classic({"input": raw_data})["output"].data
        
        return processed_data
    
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
    
    def _preprocess_for_ai(self, raw_data: np.ndarray) -> np.ndarray:
        """Preprocess data for AI model"""
        # Normalize
        if raw_data.dtype == np.uint16:
            raw_data = raw_data.astype(np.float32) / 65535.0
        elif raw_data.dtype == np.uint8:
            raw_data = raw_data.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        if len(raw_data.shape) == 2:
            raw_data = raw_data[np.newaxis, :, :, np.newaxis]
        
        return raw_data
    
    def _postprocess_from_ai(self, output_tensor: np.ndarray) -> np.ndarray:
        """Postprocess data from AI model"""
        # Remove batch and channel dimensions
        if len(output_tensor.shape) == 4:
            output_tensor = output_tensor[0, :, :, 0]
        elif len(output_tensor.shape) == 3:
            output_tensor = output_tensor[0, :, :]
        
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

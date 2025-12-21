"""
Demosaic Node: Converts RAW Bayer data to RGB image
Supports traditional algorithms (bilinear, VNG, etc.) and AI implementation
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Union
from ...core.node import ProcessingNode, ImplementationType
from ...core.frame import Frame, ColorFormat, BayerPattern


class DemosaicNode(ProcessingNode):
    """Demosaic node"""
    
    def __init__(
        self,
        node_id: str,
        config: Optional[Dict[str, Any]] = None,
        implementation: ImplementationType = ImplementationType.CLASSIC,
        enabled: bool = True
    ):
        """
        Initialize demosaic node
        
        Args:
            node_id: Node ID
            config: Configuration parameters
            implementation: Implementation type
            enabled: Whether enabled
        """
        super().__init__(node_id, config, implementation, enabled)
        
        # Default configuration
        default_config = {
            "classic_method": "bilinear",  # bilinear, vng, edge_aware
            "ai_model_path": "",
            "ai_model_config": {
                "input_size": (512, 512),
                "batch_size": 1,
                "device": "cpu"
            },
            "quality_enhancement": {
                "enabled": True,
                "sharpening": 0.1,
                "noise_reduction": 0.05
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
        classic_methods = ["bilinear", "vng", "edge_aware"]
        if self.config["classic_method"] not in classic_methods:
            raise ValueError(f"Unsupported classic method: {self.config['classic_method']}")
        
        if self.implementation == ImplementationType.AI and not self.config["ai_model_path"]:
            self.logger.warning("AI mode: model path not specified, will use classic method")
    
    def _process_classic(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """Traditional algorithm implementation"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("Input data is empty")
        
        if isinstance(input_data, Frame):
            raw_data = input_data.data
            bayer_pattern = input_data.bayer_pattern
            metadata = input_data
        else:
            raw_data = input_data
            bayer_pattern = BayerPattern.RGGB  # Default
            metadata = None
        
        # Execute demosaic
        rgb_data = self._demosaic_classic(raw_data, bayer_pattern)
        
        # Quality enhancement
        if self.config["quality_enhancement"]["enabled"]:
            rgb_data = self._enhance_quality(rgb_data)
        
        # Create output Frame
        if metadata:
            output_frame = Frame(
                data=rgb_data,
                color_format=ColorFormat.RGB,
                timestamp=metadata.timestamp,
                camera_params=metadata.camera_params,
                exposure_params=metadata.exposure_params,
                imu_data=metadata.imu_data,
                metadata=metadata.metadata
            )
        else:
            output_frame = Frame(
                data=rgb_data,
                color_format=ColorFormat.RGB
            )
        
        return {"output": output_frame}
    
    def _process_ai(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """AI algorithm implementation"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("Input data is empty")
        
        if isinstance(input_data, Frame):
            raw_data = input_data.data
            bayer_pattern = input_data.bayer_pattern
            metadata = input_data
        else:
            raw_data = input_data
            bayer_pattern = BayerPattern.RGGB  # Default
            metadata = None
        
        # Load AI model
        if self._ai_model is None:
            self._load_ai_model()
        
        # Execute AI demosaic
        if self._ai_model is not None:
            rgb_data = self._demosaic_ai(raw_data, bayer_pattern)
        else:
            # Fallback to classic method
            self.logger.warning("AI model loading failed, using classic method")
            rgb_data = self._demosaic_classic(raw_data, bayer_pattern)
        
        # Quality enhancement
        if self.config["quality_enhancement"]["enabled"]:
            rgb_data = self._enhance_quality(rgb_data)
        
        # Create output Frame
        if metadata:
            output_frame = Frame(
                data=rgb_data,
                color_format=ColorFormat.RGB,
                timestamp=metadata.timestamp,
                camera_params=metadata.camera_params,
                exposure_params=metadata.exposure_params,
                imu_data=metadata.imu_data,
                metadata=metadata.metadata
            )
        else:
            output_frame = Frame(
                data=rgb_data,
                color_format=ColorFormat.RGB
            )
        
        return {"output": output_frame}
    
    def _demosaic_classic(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """Classic demosaic algorithm"""
        method = self.config["classic_method"]
        
        if method == "bilinear":
            return self._demosaic_bilinear(raw_data, bayer_pattern)
        elif method == "vng":
            return self._demosaic_vng(raw_data, bayer_pattern)
        elif method == "edge_aware":
            return self._demosaic_edge_aware(raw_data, bayer_pattern)
        else:
            raise ValueError(f"Unsupported classic method: {method}")
    
    def _demosaic_bilinear(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """Bilinear interpolation demosaic - using OpenCV implementation"""
        # Clip input data to prevent negative values
        raw_data = np.clip(raw_data, 0, None)
        
        # Convert to uint16 for OpenCV (0-65535 range)
        if raw_data.dtype == np.float32 or raw_data.dtype == np.float64:
            # Normalize to 0-1 range, then scale to uint16
            max_val = raw_data.max() if raw_data.max() > 1.0 else 1.0
            raw_normalized = np.clip(raw_data / max_val, 0, 1)
            raw_data = (raw_normalized * 65535.0).astype(np.uint16)
        elif raw_data.dtype == np.uint16:
            # Already in uint16, pass through
            raw_data = raw_data
        else:
            raw_data = raw_data.astype(np.uint16)
        
        # Use OpenCV's bilinear demosaic
        if bayer_pattern == BayerPattern.RGGB:
            pattern = cv2.COLOR_BayerRG2RGB
        elif bayer_pattern == BayerPattern.GRBG:
            pattern = cv2.COLOR_BayerGR2RGB
        elif bayer_pattern == BayerPattern.GBRG:
            pattern = cv2.COLOR_BayerGB2RGB
        elif bayer_pattern == BayerPattern.BGGR:
            pattern = cv2.COLOR_BayerBG2RGB
        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
        
        # Convert Bayer to RGB using bilinear interpolation
        rgb_data = cv2.cvtColor(raw_data, pattern)
        
        # Convert back to float32 and normalize to [0, 1] range
        rgb_data = rgb_data.astype(np.float32) / 65535.0
        
        return rgb_data
    
    def _demosaic_vng(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """VNG (Variable Number of Gradients) demosaic"""
        # Use OpenCV's VNG implementation
        if bayer_pattern == BayerPattern.RGGB:
            pattern = cv2.COLOR_BayerRG2RGB_VNG
        elif bayer_pattern == BayerPattern.GRBG:
            pattern = cv2.COLOR_BayerGR2RGB_VNG
        elif bayer_pattern == BayerPattern.GBRG:
            pattern = cv2.COLOR_BayerGB2RGB_VNG
        elif bayer_pattern == BayerPattern.BGGR:
            pattern = cv2.COLOR_BayerBG2RGB_VNG
        else:
            raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
        
        rgb_data = cv2.cvtColor(raw_data, pattern)
        return rgb_data
    
    def _demosaic_edge_aware(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """Edge-aware demosaic"""
        # Simplified edge-aware implementation
        # First use bilinear interpolation
        rgb_data = self._demosaic_bilinear(raw_data, bayer_pattern)
        
        # Calculate edge strength
        gray = cv2.cvtColor(rgb_data.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
        edge_strength = np.abs(edges)
        
        # Enhance details in edge regions
        edge_threshold = np.percentile(edge_strength, 90)
        edge_mask = edge_strength > edge_threshold
        
        # Apply edge enhancement
        for c in range(3):
            channel = rgb_data[:, :, c].astype(np.float32)
            enhanced = channel * (1 + 0.1 * edge_mask)
            rgb_data[:, :, c] = np.clip(enhanced, 0, 255).astype(rgb_data.dtype)
        
        return rgb_data
    
    def _demosaic_ai(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """AI demosaic"""
        if self._ai_model is None:
            raise RuntimeError("AI model not loaded")
        
        # Preprocess input data
        input_tensor = self._preprocess_for_ai(raw_data)
        
        # Execute AI inference
        try:
            output_tensor = self._ai_model(input_tensor)
            rgb_data = self._postprocess_from_ai(output_tensor)
        except Exception as e:
            self.logger.error(f"AI inference failed: {e}")
            # Fallback to classic method
            rgb_data = self._demosaic_classic(raw_data, bayer_pattern)
        
        return rgb_data
    
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
        
        # Resize
        target_size = self.config["ai_model_config"]["input_size"]
        if raw_data.shape[:2] != target_size:
            raw_data = cv2.resize(raw_data, target_size)
        
        # Add batch dimension
        if len(raw_data.shape) == 2:
            raw_data = raw_data[np.newaxis, :, :, np.newaxis]
        elif len(raw_data.shape) == 3:
            raw_data = raw_data[np.newaxis, :, :, :]
        
        return raw_data
    
    def _postprocess_from_ai(self, output_tensor: np.ndarray) -> np.ndarray:
        """Postprocess data from AI model"""
        # Remove batch dimension
        if len(output_tensor.shape) == 4:
            output_tensor = output_tensor[0]
        
        # Denormalize
        output_tensor = np.clip(output_tensor, 0, 1)
        output_tensor = (output_tensor * 255).astype(np.uint8)
        
        return output_tensor
    
    def _interpolate_channel(self, channel: np.ndarray) -> np.ndarray:
        """Interpolate to fill missing pixels"""
        height, width = channel.shape
        result = channel.copy()
        
        # Horizontal interpolation
        for y in range(height):
            for x in range(1, width - 1):
                if result[y, x] == 0:
                    left = result[y, x - 1]
                    right = result[y, x + 1]
                    if left > 0 and right > 0:
                        result[y, x] = (left + right) // 2
        
        # Vertical interpolation
        for y in range(1, height - 1):
            for x in range(width):
                if result[y, x] == 0:
                    top = result[y - 1, x]
                    bottom = result[y + 1, x]
                    if top > 0 and bottom > 0:
                        result[y, x] = (top + bottom) // 2
        
        # Diagonal interpolation
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if result[y, x] == 0:
                    neighbors = [
                        result[y - 1, x - 1], result[y - 1, x + 1],
                        result[y + 1, x - 1], result[y + 1, x + 1]
                    ]
                    valid_neighbors = [n for n in neighbors if n > 0]
                    if valid_neighbors:
                        result[y, x] = sum(valid_neighbors) // len(valid_neighbors)
        
        return result
    
    def _enhance_quality(self, rgb_data: np.ndarray) -> np.ndarray:
        """Quality enhancement"""
        enhancement_config = self.config["quality_enhancement"]
        
        # Sharpening
        if enhancement_config["sharpening"] > 0:
            kernel = np.array([
                [0, -enhancement_config["sharpening"], 0],
                [-enhancement_config["sharpening"], 1 + 4 * enhancement_config["sharpening"], -enhancement_config["sharpening"]],
                [0, -enhancement_config["sharpening"], 0]
            ])
            rgb_data = cv2.filter2D(rgb_data, -1, kernel)
        
        # Noise reduction
        if enhancement_config["noise_reduction"] > 0:
            rgb_data = cv2.bilateralFilter(rgb_data, 5, 50, 50)
        
        return rgb_data
    
    def _on_implementation_changed(self):
        """Callback when implementation type changes"""
        if self.implementation == ImplementationType.AI:
            self._load_ai_model()
        else:
            self._ai_model = None

"""
去马赛克节点：将RAW Bayer数据转换为RGB图像
支持传统算法（双线性、VNG等）和AI实现
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Union
from ...core.node import ProcessingNode, ImplementationType
from ...core.frame import Frame, ColorFormat, BayerPattern


class DemosaicNode(ProcessingNode):
    """去马赛克节点"""
    
    def __init__(
        self,
        node_id: str,
        config: Optional[Dict[str, Any]] = None,
        implementation: ImplementationType = ImplementationType.CLASSIC,
        enabled: bool = True
    ):
        """
        初始化去马赛克节点
        
        Args:
            node_id: 节点ID
            config: 配置参数
            implementation: 实现类型
            enabled: 是否启用
        """
        super().__init__(node_id, config, implementation, enabled)
        
        # 默认配置
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
        
        # AI模型（延迟加载）
        self._ai_model = None
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数"""
        classic_methods = ["bilinear", "vng", "edge_aware"]
        if self.config["classic_method"] not in classic_methods:
            raise ValueError(f"不支持的经典方法: {self.config['classic_method']}")
        
        if self.implementation == ImplementationType.AI and not self.config["ai_model_path"]:
            self.logger.warning("AI模式未指定模型路径，将使用经典方法")
    
    def _process_classic(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """传统算法实现"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("输入数据为空")
        
        if isinstance(input_data, Frame):
            raw_data = input_data.data
            bayer_pattern = input_data.bayer_pattern
            metadata = input_data
        else:
            raw_data = input_data
            bayer_pattern = BayerPattern.RGGB  # 默认
            metadata = None
        
        # 执行去马赛克
        rgb_data = self._demosaic_classic(raw_data, bayer_pattern)
        
        # 质量增强
        if self.config["quality_enhancement"]["enabled"]:
            rgb_data = self._enhance_quality(rgb_data)
        
        # 创建输出Frame
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
        """AI算法实现"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("输入数据为空")
        
        if isinstance(input_data, Frame):
            raw_data = input_data.data
            bayer_pattern = input_data.bayer_pattern
            metadata = input_data
        else:
            raw_data = input_data
            bayer_pattern = BayerPattern.RGGB  # 默认
            metadata = None
        
        # 加载AI模型
        if self._ai_model is None:
            self._load_ai_model()
        
        # 执行AI去马赛克
        if self._ai_model is not None:
            rgb_data = self._demosaic_ai(raw_data, bayer_pattern)
        else:
            # 回退到经典方法
            self.logger.warning("AI模型加载失败，使用经典方法")
            rgb_data = self._demosaic_classic(raw_data, bayer_pattern)
        
        # 质量增强
        if self.config["quality_enhancement"]["enabled"]:
            rgb_data = self._enhance_quality(rgb_data)
        
        # 创建输出Frame
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
        """经典去马赛克算法"""
        method = self.config["classic_method"]
        
        if method == "bilinear":
            return self._demosaic_bilinear(raw_data, bayer_pattern)
        elif method == "vng":
            return self._demosaic_vng(raw_data, bayer_pattern)
        elif method == "edge_aware":
            return self._demosaic_edge_aware(raw_data, bayer_pattern)
        else:
            raise ValueError(f"不支持的经典方法: {method}")
    
    def _demosaic_bilinear(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """双线性插值去马赛克"""
        height, width = raw_data.shape
        
        # 创建RGB图像
        rgb_data = np.zeros((height, width, 3), dtype=raw_data.dtype)
        
        # 根据Bayer模式分配颜色通道
        if bayer_pattern == BayerPattern.RGGB:
            # R G
            # G B
            rgb_data[0::2, 0::2, 0] = raw_data[0::2, 0::2]  # R
            rgb_data[0::2, 1::2, 1] = raw_data[0::2, 1::2]  # G
            rgb_data[1::2, 0::2, 1] = raw_data[1::2, 0::2]  # G
            rgb_data[1::2, 1::2, 2] = raw_data[1::2, 1::2]  # B
        elif bayer_pattern == BayerPattern.GRBG:
            # G R
            # B G
            rgb_data[0::2, 0::2, 1] = raw_data[0::2, 0::2]  # G
            rgb_data[0::2, 1::2, 0] = raw_data[0::2, 1::2]  # R
            rgb_data[1::2, 0::2, 2] = raw_data[1::2, 0::2]  # B
            rgb_data[1::2, 1::2, 1] = raw_data[1::2, 1::2]  # G
        elif bayer_pattern == BayerPattern.GBRG:
            # G B
            # R G
            rgb_data[0::2, 0::2, 1] = raw_data[0::2, 0::2]  # G
            rgb_data[0::2, 1::2, 2] = raw_data[0::2, 1::2]  # B
            rgb_data[1::2, 0::2, 0] = raw_data[1::2, 0::2]  # R
            rgb_data[1::2, 1::2, 1] = raw_data[1::2, 1::2]  # G
        elif bayer_pattern == BayerPattern.BGGR:
            # B G
            # G R
            rgb_data[0::2, 0::2, 2] = raw_data[0::2, 0::2]  # B
            rgb_data[0::2, 1::2, 1] = raw_data[0::2, 1::2]  # G
            rgb_data[1::2, 0::2, 1] = raw_data[1::2, 0::2]  # G
            rgb_data[1::2, 1::2, 0] = raw_data[1::2, 1::2]  # R
        
        # 双线性插值填充缺失的像素
        for c in range(3):
            rgb_data[:, :, c] = self._interpolate_channel(rgb_data[:, :, c])
        
        return rgb_data
    
    def _demosaic_vng(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """VNG (Variable Number of Gradients) 去马赛克"""
        # 使用OpenCV的VNG实现
        if bayer_pattern == BayerPattern.RGGB:
            pattern = cv2.COLOR_BayerRG2RGB_VNG
        elif bayer_pattern == BayerPattern.GRBG:
            pattern = cv2.COLOR_BayerGR2RGB_VNG
        elif bayer_pattern == BayerPattern.GBRG:
            pattern = cv2.COLOR_BayerGB2RGB_VNG
        elif bayer_pattern == BayerPattern.BGGR:
            pattern = cv2.COLOR_BayerBG2RGB_VNG
        else:
            raise ValueError(f"不支持的Bayer模式: {bayer_pattern}")
        
        rgb_data = cv2.cvtColor(raw_data, pattern)
        return rgb_data
    
    def _demosaic_edge_aware(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """边缘感知去马赛克"""
        # 简化的边缘感知实现
        # 首先使用双线性插值
        rgb_data = self._demosaic_bilinear(raw_data, bayer_pattern)
        
        # 计算边缘强度
        gray = cv2.cvtColor(rgb_data.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
        edge_strength = np.abs(edges)
        
        # 在边缘区域增强细节
        edge_threshold = np.percentile(edge_strength, 90)
        edge_mask = edge_strength > edge_threshold
        
        # 应用边缘增强
        for c in range(3):
            channel = rgb_data[:, :, c].astype(np.float32)
            enhanced = channel * (1 + 0.1 * edge_mask)
            rgb_data[:, :, c] = np.clip(enhanced, 0, 255).astype(rgb_data.dtype)
        
        return rgb_data
    
    def _demosaic_ai(self, raw_data: np.ndarray, bayer_pattern: BayerPattern) -> np.ndarray:
        """AI去马赛克"""
        if self._ai_model is None:
            raise RuntimeError("AI模型未加载")
        
        # 预处理输入数据
        input_tensor = self._preprocess_for_ai(raw_data)
        
        # 执行AI推理
        try:
            output_tensor = self._ai_model(input_tensor)
            rgb_data = self._postprocess_from_ai(output_tensor)
        except Exception as e:
            self.logger.error(f"AI推理失败: {e}")
            # 回退到经典方法
            rgb_data = self._demosaic_classic(raw_data, bayer_pattern)
        
        return rgb_data
    
    def _load_ai_model(self):
        """加载AI模型"""
        model_path = self.config["ai_model_path"]
        if not model_path:
            return
        
        try:
            # 这里应该实现具体的AI模型加载逻辑
            # 支持PyTorch、TensorFlow、ONNX等格式
            self.logger.info(f"加载AI模型: {model_path}")
            
            # 占位实现
            self._ai_model = None
            
        except Exception as e:
            self.logger.error(f"AI模型加载失败: {e}")
            self._ai_model = None
    
    def _preprocess_for_ai(self, raw_data: np.ndarray) -> np.ndarray:
        """为AI模型预处理数据"""
        # 归一化
        if raw_data.dtype == np.uint16:
            raw_data = raw_data.astype(np.float32) / 65535.0
        elif raw_data.dtype == np.uint8:
            raw_data = raw_data.astype(np.float32) / 255.0
        
        # 调整尺寸
        target_size = self.config["ai_model_config"]["input_size"]
        if raw_data.shape[:2] != target_size:
            raw_data = cv2.resize(raw_data, target_size)
        
        # 添加batch维度
        if len(raw_data.shape) == 2:
            raw_data = raw_data[np.newaxis, :, :, np.newaxis]
        elif len(raw_data.shape) == 3:
            raw_data = raw_data[np.newaxis, :, :, :]
        
        return raw_data
    
    def _postprocess_from_ai(self, output_tensor: np.ndarray) -> np.ndarray:
        """从AI模型后处理数据"""
        # 移除batch维度
        if len(output_tensor.shape) == 4:
            output_tensor = output_tensor[0]
        
        # 反归一化
        output_tensor = np.clip(output_tensor, 0, 1)
        output_tensor = (output_tensor * 255).astype(np.uint8)
        
        return output_tensor
    
    def _interpolate_channel(self, channel: np.ndarray) -> np.ndarray:
        """插值填充缺失的像素"""
        height, width = channel.shape
        result = channel.copy()
        
        # 水平插值
        for y in range(height):
            for x in range(1, width - 1):
                if result[y, x] == 0:
                    left = result[y, x - 1]
                    right = result[y, x + 1]
                    if left > 0 and right > 0:
                        result[y, x] = (left + right) // 2
        
        # 垂直插值
        for y in range(1, height - 1):
            for x in range(width):
                if result[y, x] == 0:
                    top = result[y - 1, x]
                    bottom = result[y + 1, x]
                    if top > 0 and bottom > 0:
                        result[y, x] = (top + bottom) // 2
        
        # 对角线插值
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
        """质量增强"""
        enhancement_config = self.config["quality_enhancement"]
        
        # 锐化
        if enhancement_config["sharpening"] > 0:
            kernel = np.array([
                [0, -enhancement_config["sharpening"], 0],
                [-enhancement_config["sharpening"], 1 + 4 * enhancement_config["sharpening"], -enhancement_config["sharpening"]],
                [0, -enhancement_config["sharpening"], 0]
            ])
            rgb_data = cv2.filter2D(rgb_data, -1, kernel)
        
        # 降噪
        if enhancement_config["noise_reduction"] > 0:
            rgb_data = cv2.bilateralFilter(rgb_data, 5, 50, 50)
        
        return rgb_data
    
    def _on_implementation_changed(self):
        """实现类型改变时的回调"""
        if self.implementation == ImplementationType.AI:
            self._load_ai_model()
        else:
            self._ai_model = None

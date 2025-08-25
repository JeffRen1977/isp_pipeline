"""
RAW预处理节点：包含BPC、BLC、LSC等基本功能
支持传统算法和AI实现
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Union
from ...core.node import ProcessingNode, ImplementationType
from ...core.frame import Frame, ColorFormat, BayerPattern


class RawPreprocNode(ProcessingNode):
    """RAW预处理节点"""
    
    def __init__(
        self,
        node_id: str,
        config: Optional[Dict[str, Any]] = None,
        implementation: ImplementationType = ImplementationType.CLASSIC,
        enabled: bool = True
    ):
        """
        初始化RAW预处理节点
        
        Args:
            node_id: 节点ID
            config: 配置参数
            implementation: 实现类型
            enabled: 是否启用
        """
        super().__init__(node_id, config, implementation, enabled)
        
        # 默认配置
        default_config = {
            "bpc_enabled": True,  # 坏点校正
            "blc_enabled": True,  # 黑电平校正
            "lsc_enabled": True,  # 镜头阴影校正
            "lsc_method": "classic",  # classic, ai
            "ai_model_path": "",
            "bpc_config": {
                "threshold": 3.0,  # 坏点检测阈值
                "window_size": 5
            },
            "blc_config": {
                "black_level": 64,  # 黑电平值
                "method": "subtract"  # subtract, scale
            },
            "lsc_config": {
                "method": "polynomial",  # polynomial, lookup_table, ai
                "coefficients": [1.0, 0.1, 0.05, 0.01],  # 多项式系数
                "center": None,  # 中心点，None表示图像中心
                "radius": None   # 半径，None表示图像对角线的一半
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
        if self.config["lsc_method"] not in ["classic", "ai"]:
            raise ValueError(f"不支持的LSC方法: {self.config['lsc_method']}")
        
        if self.implementation == ImplementationType.AI and not self.config["ai_model_path"]:
            self.logger.warning("AI模式未指定模型路径，将使用经典方法")
    
    def _process_classic(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """传统算法实现"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("输入数据为空")
        
        if isinstance(input_data, Frame):
            raw_data = input_data.data
            metadata = input_data
        else:
            raw_data = input_data
            metadata = None
        
        # 执行预处理
        processed_data = raw_data.copy()
        
        if self.config["bpc_enabled"]:
            processed_data = self._bad_pixel_correction(processed_data)
        
        if self.config["blc_enabled"]:
            processed_data = self._black_level_correction(processed_data)
        
        if self.config["lsc_enabled"]:
            processed_data = self._lens_shading_correction(processed_data)
        
        # 创建输出Frame
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
        """AI算法实现"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("输入数据为空")
        
        if isinstance(input_data, Frame):
            raw_data = input_data.data
            metadata = input_data
        else:
            raw_data = input_data
            metadata = None
        
        # 加载AI模型
        if self._ai_model is None:
            self._load_ai_model()
        
        # 执行AI预处理
        if self._ai_model is not None:
            processed_data = self._ai_preprocessing(raw_data)
        else:
            # 回退到经典方法
            self.logger.warning("AI模型加载失败，使用经典方法")
            return self._process_classic(inputs)
        
        # 创建输出Frame
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
        """坏点校正"""
        config = self.config["bpc_config"]
        threshold = config["threshold"]
        window_size = config["window_size"]
        
        height, width = raw_data.shape
        corrected_data = raw_data.copy()
        
        # 计算局部统计
        half_window = window_size // 2
        
        for y in range(half_window, height - half_window):
            for x in range(half_window, width - half_window):
                # 提取局部窗口
                window = raw_data[y-half_window:y+half_window+1, 
                                x-half_window:x+half_window+1]
                
                # 计算局部均值和标准差
                local_mean = np.mean(window)
                local_std = np.std(window)
                
                # 检测坏点
                current_pixel = raw_data[y, x]
                if abs(current_pixel - local_mean) > threshold * local_std:
                    # 使用邻域像素的中值替换坏点
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
        """黑电平校正"""
        config = self.config["blc_config"]
        black_level = config["black_level"]
        method = config["method"]
        
        corrected_data = raw_data.copy()
        
        if method == "subtract":
            # 减去黑电平
            corrected_data = np.clip(corrected_data - black_level, 0, None)
        elif method == "scale":
            # 缩放校正
            max_value = np.max(raw_data)
            corrected_data = ((corrected_data - black_level) / 
                            (max_value - black_level) * max_value)
            corrected_data = np.clip(corrected_data, 0, max_value)
        
        return corrected_data
    
    def _lens_shading_correction(self, raw_data: np.ndarray) -> np.ndarray:
        """镜头阴影校正"""
        config = self.config["lsc_config"]
        method = config["method"]
        
        if method == "polynomial":
            return self._lsc_polynomial(raw_data)
        elif method == "lookup_table":
            return self._lsc_lookup_table(raw_data)
        else:
            return raw_data
    
    def _lsc_polynomial(self, raw_data: np.ndarray) -> np.ndarray:
        """多项式镜头阴影校正"""
        config = self.config["lsc_config"]
        coefficients = config["coefficients"]
        
        height, width = raw_data.shape
        
        # 确定中心点和半径
        if config["center"] is None:
            center_y, center_x = height // 2, width // 2
        else:
            center_y, center_x = config["center"]
        
        if config["radius"] is None:
            radius = np.sqrt(center_y**2 + center_x**2)
        else:
            radius = config["radius"]
        
        # 创建校正图
        correction_map = np.ones_like(raw_data, dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                # 计算到中心的距离
                dy = y - center_y
                dx = x - center_x
                distance = np.sqrt(dy**2 + dx**2)
                
                # 归一化距离
                normalized_distance = distance / radius
                
                # 应用多项式校正
                correction = 1.0
                for i, coef in enumerate(coefficients):
                    correction += coef * (normalized_distance ** (i + 1))
                
                correction_map[y, x] = correction
        
        # 应用校正
        corrected_data = raw_data.astype(np.float32) * correction_map
        corrected_data = np.clip(corrected_data, 0, np.max(raw_data))
        
        return corrected_data.astype(raw_data.dtype)
    
    def _lsc_lookup_table(self, raw_data: np.ndarray) -> np.ndarray:
        """查找表镜头阴影校正"""
        # 简化的查找表实现
        # 实际应用中应该使用预计算的查找表
        return self._lsc_polynomial(raw_data)
    
    def _ai_preprocessing(self, raw_data: np.ndarray) -> np.ndarray:
        """AI预处理"""
        if self._ai_model is None:
            raise RuntimeError("AI模型未加载")
        
        # 预处理输入数据
        input_tensor = self._preprocess_for_ai(raw_data)
        
        # 执行AI推理
        try:
            output_tensor = self._ai_model(input_tensor)
            processed_data = self._postprocess_from_ai(output_tensor)
        except Exception as e:
            self.logger.error(f"AI推理失败: {e}")
            # 回退到经典方法
            return self._process_classic({"input": raw_data})["output"].data
        
        return processed_data
    
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
        
        # 添加batch和channel维度
        if len(raw_data.shape) == 2:
            raw_data = raw_data[np.newaxis, :, :, np.newaxis]
        
        return raw_data
    
    def _postprocess_from_ai(self, output_tensor: np.ndarray) -> np.ndarray:
        """从AI模型后处理数据"""
        # 移除batch和channel维度
        if len(output_tensor.shape) == 4:
            output_tensor = output_tensor[0, :, :, 0]
        elif len(output_tensor.shape) == 3:
            output_tensor = output_tensor[0, :, :]
        
        # 反归一化
        output_tensor = np.clip(output_tensor, 0, 1)
        output_tensor = (output_tensor * 255).astype(np.uint8)
        
        return output_tensor
    
    def _on_implementation_changed(self):
        """实现类型改变时的回调"""
        if self.implementation == ImplementationType.AI:
            self._load_ai_model()
        else:
            self._ai_model = None

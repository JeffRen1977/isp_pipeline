"""
白平衡节点：自动白平衡和手动白平衡
支持多种算法实现
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Union
from ...core.node import ProcessingNode, ImplementationType
from ...core.frame import Frame, ColorFormat


class AWBNode(ProcessingNode):
    """白平衡节点"""
    
    def __init__(
        self,
        node_id: str,
        config: Optional[Dict[str, Any]] = None,
        implementation: ImplementationType = ImplementationType.CLASSIC,
        enabled: bool = True
    ):
        """
        初始化白平衡节点
        
        Args:
            node_id: 节点ID
            config: 配置参数
            implementation: 实现类型
            enabled: 是否启用
        """
        super().__init__(node_id, config, implementation, enabled)
        
        # 默认配置
        default_config = {
            "method": "gray_world",  # gray_world, white_patch, retinex, ai
            "ai_model_path": "",
            "temperature": 5500,  # 色温 (K)
            "tint": 0.0,  # 色调偏移
            "adaptive": False,  # 是否启用自适应白平衡
            "adaptive_window": 10,  # 自适应窗口大小
            "gray_world_config": {
                "saturation_threshold": 0.8,  # 饱和度阈值
                "brightness_threshold": 0.1    # 亮度阈值
            },
            "white_patch_config": {
                "patch_size": 0.1,  # 白块大小比例
                "brightness_threshold": 0.9,  # 亮度阈值
                "saturation_threshold": 0.1   # 饱和度阈值
            },
            "retinex_config": {
                "sigma": 30.0,  # 高斯滤波标准差
                "scale": 1.0    # 缩放因子
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
        methods = ["gray_world", "white_patch", "retinex", "ai"]
        if self.config["method"] not in methods:
            raise ValueError(f"不支持的白平衡方法: {self.config['method']}")
        
        if self.implementation == ImplementationType.AI and not self.config["ai_model_path"]:
            self.logger.warning("AI模式未指定模型路径，将使用经典方法")
    
    def _process_classic(self, inputs: Dict[str, Union[Frame, np.ndarray]]) -> Dict[str, Union[Frame, np.ndarray]]:
        """传统算法实现"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("输入数据为空")
        
        if isinstance(input_data, Frame):
            rgb_data = input_data.data
            metadata = input_data
        else:
            rgb_data = input_data
            metadata = None
        
        # 检查输入格式
        if len(rgb_data.shape) != 3 or rgb_data.shape[2] != 3:
            raise ValueError("输入必须是3通道RGB图像")
        
        # 执行白平衡
        method = self.config["method"]
        if method == "gray_world":
            balanced_data = self._gray_world_awb(rgb_data)
        elif method == "white_patch":
            balanced_data = self._white_patch_awb(rgb_data)
        elif method == "retinex":
            balanced_data = self._retinex_awb(rgb_data)
        else:
            balanced_data = self._manual_awb(rgb_data)
        
        # 创建输出Frame
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
        """AI算法实现"""
        input_data = inputs.get("input")
        if input_data is None:
            raise ValueError("输入数据为空")
        
        if isinstance(input_data, Frame):
            rgb_data = input_data.data
            metadata = input_data
        else:
            rgb_data = input_data
            metadata = None
        
        # 加载AI模型
        if self._ai_model is None:
            self._load_ai_model()
        
        # 执行AI白平衡
        if self._ai_model is not None:
            balanced_data = self._ai_awb(rgb_data)
        else:
            # 回退到经典方法
            self.logger.warning("AI模型加载失败，使用经典方法")
            return self._process_classic(inputs)
        
        # 创建输出Frame
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
        """灰度世界白平衡 - 固定版本"""
        # 处理输入数据类型，确保归一化到 [0, 1]
        if rgb_data.dtype == np.uint8:
            rgb_float = rgb_data.astype(np.float32) / 255.0
        elif rgb_data.dtype == np.uint16:
            rgb_float = rgb_data.astype(np.float32) / 65535.0
        else:  # float32 or float64
            # 确保数据在合理范围内
            rgb_float = np.clip(rgb_data, 0, 1).astype(np.float32)
        
        # 计算每个通道的平均值
        r_mean = np.mean(rgb_float[:, :, 0])
        g_mean = np.mean(rgb_float[:, :, 1])
        b_mean = np.mean(rgb_float[:, :, 2])
        
        # Gray World: 假设灰色是通道平均值相等
        # 计算增益使得三个通道平均值相等
        avg_mean = (r_mean + g_mean + b_mean) / 3.0
        
        # 计算增益
        r_gain = avg_mean / r_mean if r_mean > 0.001 else 1.0
        g_gain = avg_mean / g_mean if g_mean > 0.001 else 1.0
        b_gain = avg_mean / b_mean if b_mean > 0.001 else 1.0
        
        # 限制增益范围，避免过校正 (0.5x 到 2.0x)
        r_gain = np.clip(r_gain, 0.5, 2.0)
        g_gain = np.clip(g_gain, 0.5, 2.0)
        b_gain = np.clip(b_gain, 0.5, 2.0)
        
        # 应用增益
        balanced_data = rgb_float.copy()
        balanced_data[:, :, 0] = rgb_float[:, :, 0] * r_gain
        balanced_data[:, :, 1] = rgb_float[:, :, 1] * g_gain
        balanced_data[:, :, 2] = rgb_float[:, :, 2] * b_gain
        
        # 裁剪到 [0, 1]
        balanced_data = np.clip(balanced_data, 0, 1)
        
        # 根据输入数据类型返回
        if rgb_data.dtype == np.uint8:
            return (balanced_data * 255).astype(np.uint8)
        elif rgb_data.dtype == np.uint16:
            return (balanced_data * 65535).astype(np.uint16)
        else:  # float32 or float64
            return balanced_data.astype(np.float32)
    
    def _white_patch_awb(self, rgb_data: np.ndarray) -> np.ndarray:
        """白块白平衡"""
        config = self.config["white_patch_config"]
        patch_size = config["patch_size"]
        bright_threshold = config["brightness_threshold"]
        sat_threshold = config["saturation_threshold"]
        
        # 转换为浮点数
        rgb_float = rgb_data.astype(np.float32) / 255.0
        
        # 计算HSV
        hsv = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2HSV)
        hsv_float = hsv.astype(np.float32)
        hsv_float[:, :, 1] /= 255.0  # 归一化饱和度
        hsv_float[:, :, 2] /= 255.0  # 归一化亮度
        
        # 创建掩码
        mask = (hsv_float[:, :, 2] > bright_threshold) & (hsv_float[:, :, 1] < sat_threshold)
        
        if np.sum(mask) == 0:
            # 如果没有合适的像素，使用灰度世界方法
            return self._gray_world_awb(rgb_data)
        
        # 找到最亮的像素
        bright_pixels = rgb_float[mask]
        if len(bright_pixels) == 0:
            return self._gray_world_awb(rgb_data)
        
        # 计算白块的RGB值
        white_rgb = np.mean(bright_pixels, axis=0)
        
        # 计算增益（假设白块应该是白色）
        gains = [1.0 / val if val > 0 else 1.0 for val in white_rgb]
        
        # 应用白平衡
        balanced_data = np.zeros_like(rgb_float)
        for c in range(3):
            balanced_data[:, :, c] = rgb_float[:, :, c] * gains[c]
        
        # 裁剪到[0, 1]范围
        balanced_data = np.clip(balanced_data, 0, 1)
        
        # 转换回uint8
        return (balanced_data * 255).astype(np.uint8)
    
    def _retinex_awb(self, rgb_data: np.ndarray) -> np.ndarray:
        """Retinex白平衡"""
        config = self.config["retinex_config"]
        sigma = config["sigma"]
        scale = config["scale"]
        
        # 转换为浮点数
        rgb_float = rgb_data.astype(np.float32) / 255.0
        
        # 应用高斯滤波
        gaussian = cv2.GaussianBlur(rgb_float, (0, 0), sigma)
        
        # 计算对数
        log_rgb = np.log(rgb_float + 1e-6)
        log_gaussian = np.log(gaussian + 1e-6)
        
        # 计算Retinex
        retinex = (log_rgb - log_gaussian) * scale
        
        # 归一化
        retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
        
        # 转换回uint8
        return (retinex * 255).astype(np.uint8)
    
    def _manual_awb(self, rgb_data: np.ndarray) -> np.ndarray:
        """手动白平衡"""
        temperature = self.config["temperature"]
        tint = self.config["tint"]
        
        # 转换为浮点数
        rgb_float = rgb_data.astype(np.float32) / 255.0
        
        # 基于色温的简单白平衡
        # 这里使用简化的色温到RGB增益的映射
        if temperature < 3000:  # 暖色调
            gains = [1.2, 1.0, 0.8]
        elif temperature < 5000:  # 中性
            gains = [1.1, 1.0, 0.9]
        elif temperature < 7000:  # 冷色调
            gains = [0.9, 1.0, 1.1]
        else:  # 非常冷
            gains = [0.8, 1.0, 1.2]
        
        # 应用色调偏移
        gains[1] += tint * 0.1  # 绿色通道调整
        
        # 应用白平衡
        balanced_data = np.zeros_like(rgb_float)
        for c in range(3):
            balanced_data[:, :, c] = rgb_float[:, :, c] * gains[c]
        
        # 裁剪到[0, 1]范围
        balanced_data = np.clip(balanced_data, 0, 1)
        
        # 转换回uint8
        return (balanced_data * 255).astype(np.uint8)
    
    def _ai_awb(self, rgb_data: np.ndarray) -> np.ndarray:
        """AI白平衡"""
        if self._ai_model is None:
            raise RuntimeError("AI模型未加载")
        
        # 预处理输入数据
        input_tensor = self._preprocess_for_ai(rgb_data)
        
        # 执行AI推理
        try:
            output_tensor = self._ai_model(input_tensor)
            balanced_data = self._postprocess_from_ai(output_tensor)
        except Exception as e:
            self.logger.error(f"AI推理失败: {e}")
            # 回退到经典方法
            return self._process_classic({"input": rgb_data})["output"].data
        
        return balanced_data
    
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
    
    def _preprocess_for_ai(self, rgb_data: np.ndarray) -> np.ndarray:
        """为AI模型预处理数据"""
        # 归一化
        if rgb_data.dtype == np.uint8:
            rgb_data = rgb_data.astype(np.float32) / 255.0
        
        # 添加batch维度
        if len(rgb_data.shape) == 3:
            rgb_data = rgb_data[np.newaxis, :, :, :]
        
        return rgb_data
    
    def _postprocess_from_ai(self, output_tensor: np.ndarray) -> np.ndarray:
        """从AI模型后处理数据"""
        # 移除batch维度
        if len(output_tensor.shape) == 4:
            output_tensor = output_tensor[0]
        
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

"""
RAW输入节点：支持从文件读取RAW数据和仿真生成
包含噪声模型、曝光模拟器、多摄同步模拟等
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Union, List
from ...core.node import InputNode, ImplementationType
from ...core.frame import Frame, ColorFormat, BayerPattern, CameraParams, ExposureParams, IMUData


class RawInputNode(InputNode):
    """RAW输入节点"""
    
    def __init__(
        self,
        node_id: str,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        """
        初始化RAW输入节点
        
        Args:
            node_id: 节点ID
            config: 配置参数
            enabled: 是否启用
        """
        super().__init__(node_id, config, enabled)
        
        # 默认配置
        default_config = {
            "input_type": "file",  # file, simulation, camera
            "file_path": "",
            "bayer_pattern": "rggb",
            "width": 4000,
            "height": 3000,
            "bit_depth": 12,
            "noise_model": {
                "enabled": True,
                "read_noise": 2.0,
                "shot_noise": 0.1,
                "dark_current": 0.01
            },
            "exposure_simulation": {
                "enabled": True,
                "exposure_times": [1.0/30.0, 1.0/15.0, 1.0/8.0],  # HDR burst
                "iso_values": [100, 200, 400]
            },
            "multicam_simulation": {
                "enabled": False,
                "num_cameras": 2,
                "baseline": 0.1,  # 基线距离
                "disparity_range": (0, 100)
            },
            "motion_simulation": {
                "enabled": False,
                "motion_type": "rotation",  # rotation, translation, jitter
                "intensity": 0.1
            }
        }
        
        self.config.update(default_config)
        if config:
            self.config.update(config)
        
        # 验证配置
        self._validate_config()
        
        # 内部状态
        self._current_frame_idx = 0
        self._simulation_data = None
    
    def _validate_config(self):
        """验证配置参数"""
        if self.config["input_type"] == "file" and not self.config["file_path"]:
            raise ValueError("文件输入模式必须指定file_path")
        
        if self.config["bayer_pattern"] not in ["rggb", "grbg", "gbrg", "bggr"]:
            raise ValueError(f"不支持的Bayer模式: {self.config['bayer_pattern']}")
        
        if self.config["width"] <= 0 or self.config["height"] <= 0:
            raise ValueError("图像尺寸必须大于0")
        
        if self.config["bit_depth"] not in [8, 10, 12, 14, 16]:
            raise ValueError(f"不支持的位深度: {self.config['bit_depth']}")
    
    def generate_frame(self) -> Frame:
        """
        生成下一帧数据
        
        Returns:
            生成的Frame对象
        """
        if self.config["input_type"] == "file":
            return self._load_from_file()
        elif self.config["input_type"] == "simulation":
            return self._generate_simulation_frame()
        elif self.config["input_type"] == "camera":
            return self._capture_from_camera()
        else:
            raise ValueError(f"不支持的输入类型: {self.config['input_type']}")
    
    def _load_from_file(self) -> Frame:
        """从文件加载RAW数据"""
        file_path = self.config["file_path"]
        
        # 尝试不同的加载方式
        try:
            # 尝试作为图像文件加载
            raw_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if raw_data is None:
                raise ValueError(f"无法加载文件: {file_path}")
            
            # 转换为RAW格式
            if len(raw_data.shape) == 3:
                # 如果是彩色图像，转换为灰度
                raw_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
            
            # 调整位深度
            raw_data = self._adjust_bit_depth(raw_data)
            
        except Exception as e:
            # 尝试作为RAW数据加载
            try:
                raw_data = np.fromfile(file_path, dtype=np.uint16)
                raw_data = raw_data.reshape(self.config["height"], self.config["width"])
                raw_data = self._adjust_bit_depth(raw_data)
            except Exception as e2:
                raise ValueError(f"无法加载文件{file_path}: {e}, {e2}")
        
        # 创建Frame对象
        frame = Frame(
            data=raw_data,
            color_format=ColorFormat.RAW_BAYER,
            bayer_pattern=BayerPattern(self.config["bayer_pattern"]),
            timestamp=time.time(),
            camera_params=self._get_camera_params(),
            exposure_params=self._get_exposure_params(),
            imu_data=self._get_imu_data()
        )
        
        return frame
    
    def _generate_simulation_frame(self) -> Frame:
        """生成仿真RAW数据"""
        height, width = self.config["height"], self.config["width"]
        
        # 生成基础图像（简单的渐变或图案）
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # 创建基础图像
        base_image = np.zeros((height, width), dtype=np.float32)
        
        # 添加渐变
        base_image += 0.3 * X + 0.2 * Y
        
        # 添加圆形图案
        center_x, center_y = 0.5, 0.5
        radius = 0.3
        circle_mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) < radius ** 2
        base_image[circle_mask] += 0.4
        
        # 添加噪声
        if self.config["noise_model"]["enabled"]:
            base_image = self._add_noise(base_image)
        
        # 应用Bayer模式
        bayer_image = self._apply_bayer_pattern(base_image)
        
        # 调整位深度
        bayer_image = self._adjust_bit_depth(bayer_image)
        
        # 创建Frame对象
        frame = Frame(
            data=bayer_image,
            color_format=ColorFormat.RAW_BAYER,
            bayer_pattern=BayerPattern(self.config["bayer_pattern"]),
            timestamp=time.time(),
            camera_params=self._get_camera_params(),
            exposure_params=self._get_exposure_params(),
            imu_data=self._get_imu_data()
        )
        
        self._current_frame_idx += 1
        return frame
    
    def _capture_from_camera(self) -> Frame:
        """从相机捕获数据（占位实现）"""
        # 这里应该实现真实的相机捕获
        # 目前返回仿真数据
        return self._generate_simulation_frame()
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """添加噪声"""
        noise_config = self.config["noise_model"]
        
        # 读取噪声
        read_noise = noise_config["read_noise"]
        read_noise_array = np.random.normal(0, read_noise, image.shape)
        
        # 散粒噪声（与信号强度相关）
        shot_noise = noise_config["shot_noise"]
        shot_noise_array = np.random.poisson(image * shot_noise)
        
        # 暗电流
        dark_current = noise_config["dark_current"]
        dark_current_array = np.random.exponential(dark_current, image.shape)
        
        # 合并噪声
        noisy_image = image + read_noise_array + shot_noise_array + dark_current_array
        
        return np.clip(noisy_image, 0, 1)
    
    def _apply_bayer_pattern(self, image: np.ndarray) -> np.ndarray:
        """应用Bayer模式"""
        height, width = image.shape
        bayer_pattern = self.config["bayer_pattern"]
        
        # 创建Bayer图像
        bayer_image = np.zeros((height, width), dtype=image.dtype)
        
        if bayer_pattern == "rggb":
            # R G
            # G B
            bayer_image[0::2, 0::2] = image[0::2, 0::2]  # R
            bayer_image[0::2, 1::2] = image[0::2, 1::2]  # G
            bayer_image[1::2, 0::2] = image[1::2, 0::2]  # G
            bayer_image[1::2, 1::2] = image[1::2, 1::2]  # B
        elif bayer_pattern == "grbg":
            # G R
            # B G
            bayer_image[0::2, 0::2] = image[0::2, 0::2]  # G
            bayer_image[0::2, 1::2] = image[0::2, 1::2]  # R
            bayer_image[1::2, 0::2] = image[1::2, 0::2]  # B
            bayer_image[1::2, 1::2] = image[1::2, 1::2]  # G
        elif bayer_pattern == "gbrg":
            # G B
            # R G
            bayer_image[0::2, 0::2] = image[0::2, 0::2]  # G
            bayer_image[0::2, 1::2] = image[0::2, 1::2]  # B
            bayer_image[1::2, 0::2] = image[1::2, 0::2]  # R
            bayer_image[1::2, 1::2] = image[1::2, 1::2]  # G
        elif bayer_pattern == "bggr":
            # B G
            # G R
            bayer_image[0::2, 0::2] = image[0::2, 0::2]  # B
            bayer_image[0::2, 1::2] = image[0::2, 1::2]  # G
            bayer_image[1::2, 0::2] = image[1::2, 0::2]  # G
            bayer_image[1::2, 1::2] = image[1::2, 1::2]  # R
        
        return bayer_image
    
    def _adjust_bit_depth(self, image: np.ndarray) -> np.ndarray:
        """调整位深度"""
        bit_depth = self.config["bit_depth"]
        max_value = (1 << bit_depth) - 1
        
        # 归一化到[0, 1]
        if image.max() > 1.0:
            image = image / image.max()
        
        # 量化到指定位深度
        image = np.clip(image * max_value, 0, max_value)
        
        # 转换为整数类型
        if bit_depth <= 8:
            return image.astype(np.uint8)
        else:
            return image.astype(np.uint16)
    
    def _get_camera_params(self) -> CameraParams:
        """获取相机参数"""
        return CameraParams(
            focal_length=35.0,  # 35mm
            f_number=2.8,
            sensor_size=(36.0, 24.0),  # 全画幅
            principal_point=(self.config["width"] / 2, self.config["height"] / 2)
        )
    
    def _get_exposure_params(self) -> ExposureParams:
        """获取曝光参数"""
        exposure_config = self.config["exposure_simulation"]
        
        if exposure_config["enabled"] and self._current_frame_idx < len(exposure_config["exposure_times"]):
            exposure_time = exposure_config["exposure_times"][self._current_frame_idx]
            iso = exposure_config["iso_values"][self._current_frame_idx]
        else:
            exposure_time = 1.0 / 30.0
            iso = 100
        
        return ExposureParams(
            exposure_time=exposure_time,
            iso=iso,
            gain=iso / 100.0
        )
    
    def _get_imu_data(self) -> IMUData:
        """获取IMU数据"""
        motion_config = self.config["motion_simulation"]
        
        if motion_config["enabled"]:
            intensity = motion_config["intensity"]
            motion_type = motion_config["motion_type"]
            
            if motion_type == "rotation":
                gyro = np.random.normal(0, intensity, 3)
            elif motion_type == "translation":
                accel = np.random.normal(0, intensity, 3)
            elif motion_type == "jitter":
                gyro = np.random.normal(0, intensity, 3)
                accel = np.random.normal(0, intensity, 3)
            else:
                gyro = np.zeros(3)
                accel = np.zeros(3)
        else:
            gyro = np.zeros(3)
            accel = np.zeros(3)
        
        return IMUData(
            timestamp=time.time(),
            gyroscope=gyro,
            accelerometer=accel
        )
    
    def generate_hdr_burst(self, num_frames: int = 3) -> List[Frame]:
        """生成HDR burst序列"""
        frames = []
        for i in range(num_frames):
            frame = self.generate_frame()
            frames.append(frame)
        return frames
    
    def generate_multicam_frames(self, num_cameras: int = 2) -> List[Frame]:
        """生成多摄同步帧"""
        if not self.config["multicam_simulation"]["enabled"]:
            raise RuntimeError("多摄仿真未启用")
        
        frames = []
        baseline = self.config["multicam_simulation"]["baseline"]
        disparity_range = self.config["multicam_simulation"]["disparity_range"]
        
        for i in range(num_cameras):
            # 生成基础帧
            frame = self.generate_frame()
            
            # 添加视差（模拟立体视觉）
            if i > 0:
                disparity = np.random.uniform(disparity_range[0], disparity_range[1])
                # 这里应该实现真实的视差模拟
                # 目前只是简单的占位实现
            
            frames.append(frame)
        
        return frames
    
    def reset_simulation(self):
        """重置仿真状态"""
        self._current_frame_idx = 0
        self._simulation_data = None


# 添加缺失的import
import time

"""
Frame类：统一的图像数据模型
包含图像数据、曝光参数、相机参数、IMU数据等元数据
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class ColorFormat(Enum):
    """颜色格式枚举"""
    RAW_BAYER = "raw_bayer"
    RAW_MONO = "raw_mono"
    RGB = "rgb"
    YUV = "yuv"
    GRAY = "gray"


class BayerPattern(Enum):
    """Bayer模式枚举"""
    RGGB = "rggb"
    GRBG = "grbg"
    GBRG = "gbrg"
    BGGR = "bggr"


@dataclass
class CameraParams:
    """相机参数"""
    focal_length: float = 1.0
    f_number: float = 2.8
    sensor_size: tuple = (1.0, 1.0)
    principal_point: tuple = (0.0, 0.0)
    distortion_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))
    
    def __post_init__(self):
        if isinstance(self.sensor_size, (list, tuple)):
            self.sensor_size = np.array(self.sensor_size)
        if isinstance(self.principal_point, (list, tuple)):
            self.principal_point = np.array(self.principal_point)
        if isinstance(self.distortion_coeffs, (list, tuple)):
            self.distortion_coeffs = np.array(self.distortion_coeffs)


@dataclass
class ExposureParams:
    """曝光参数"""
    exposure_time: float = 1.0/30.0  # 秒
    iso: int = 100
    gain: float = 1.0
    exposure_bias: float = 0.0


@dataclass
class IMUData:
    """IMU数据"""
    timestamp: float = 0.0
    gyroscope: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accelerometer: np.ndarray = field(default_factory=lambda: np.zeros(3))
    magnetometer: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def __post_init__(self):
        if isinstance(self.gyroscope, (list, tuple)):
            self.gyroscope = np.array(self.gyroscope)
        if isinstance(self.accelerometer, (list, tuple)):
            self.accelerometer = np.array(self.accelerometer)
        if isinstance(self.magnetometer, (list, tuple)):
            self.magnetometer = np.array(self.magnetometer)


class Frame:
    """统一的图像帧数据模型"""
    
    def __init__(
        self,
        data: np.ndarray,
        color_format: ColorFormat = ColorFormat.RAW_BAYER,
        bayer_pattern: Optional[BayerPattern] = None,
        timestamp: float = 0.0,
        camera_params: Optional[CameraParams] = None,
        exposure_params: Optional[ExposureParams] = None,
        imu_data: Optional[IMUData] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化Frame
        
        Args:
            data: 图像数据 (H, W, C) 或 (H, W)
            color_format: 颜色格式
            bayer_pattern: Bayer模式（仅RAW格式需要）
            timestamp: 时间戳
            camera_params: 相机参数
            exposure_params: 曝光参数
            imu_data: IMU数据
            metadata: 其他元数据
        """
        self.data = np.asarray(data)
        self.color_format = color_format
        self.bayer_pattern = bayer_pattern
        self.timestamp = timestamp
        self.camera_params = camera_params or CameraParams()
        self.exposure_params = exposure_params or ExposureParams()
        self.imu_data = imu_data or IMUData()
        self.metadata = metadata or {}
        
        # 验证数据格式
        self._validate_data()
    
    def _validate_data(self):
        """验证数据格式"""
        if self.data.ndim not in [2, 3]:
            raise ValueError(f"数据维度必须是2或3，当前为{self.data.ndim}")
        
        if self.color_format == ColorFormat.RAW_BAYER and self.bayer_pattern is None:
            raise ValueError("RAW Bayer格式必须指定bayer_pattern")
    
    @property
    def height(self) -> int:
        """图像高度"""
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        """图像宽度"""
        return self.data.shape[1]
    
    @property
    def channels(self) -> int:
        """图像通道数"""
        return 1 if self.data.ndim == 2 else self.data.shape[2]
    
    @property
    def shape(self) -> tuple:
        """图像形状"""
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        """数据类型"""
        return self.data.dtype
    
    def copy(self) -> 'Frame':
        """复制Frame"""
        return Frame(
            data=self.data.copy(),
            color_format=self.color_format,
            bayer_pattern=self.bayer_pattern,
            timestamp=self.timestamp,
            camera_params=self.camera_params,
            exposure_params=self.exposure_params,
            imu_data=self.imu_data,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'data': self.data,
            'color_format': self.color_format.value,
            'bayer_pattern': self.bayer_pattern.value if self.bayer_pattern else None,
            'timestamp': self.timestamp,
            'camera_params': {
                'focal_length': self.camera_params.focal_length,
                'f_number': self.camera_params.f_number,
                'sensor_size': self.camera_params.sensor_size.tolist(),
                'principal_point': self.camera_params.principal_point.tolist(),
                'distortion_coeffs': self.camera_params.distortion_coeffs.tolist()
            },
            'exposure_params': {
                'exposure_time': self.exposure_params.exposure_time,
                'iso': self.exposure_params.iso,
                'gain': self.exposure_params.gain,
                'exposure_bias': self.exposure_params.exposure_bias
            },
            'imu_data': {
                'timestamp': self.imu_data.timestamp,
                'gyroscope': self.imu_data.gyroscope.tolist(),
                'accelerometer': self.imu_data.accelerometer.tolist(),
                'magnetometer': self.imu_data.magnetometer.tolist()
            },
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        return (f"Frame(shape={self.shape}, format={self.color_format.value}, "
                f"timestamp={self.timestamp:.3f}s)")

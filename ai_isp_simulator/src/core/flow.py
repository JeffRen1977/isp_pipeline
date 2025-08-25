"""
Flow类：帧组管理
包含多帧数据、对齐信息、深度信息等，用于HDR、多摄、视频等场景
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from .frame import Frame


@dataclass
class AlignmentInfo:
    """对齐信息"""
    # 光流信息
    optical_flow: Optional[np.ndarray] = None  # (H, W, 2) 或 (N, H, W, 2)
    
    # 单应性矩阵
    homography: Optional[np.ndarray] = None  # (3, 3) 或 (N, 3, 3)
    
    # 变换矩阵
    transform_matrix: Optional[np.ndarray] = None  # (3, 3) 或 (N, 3, 3)
    
    # 关键点对应
    keypoints_src: Optional[np.ndarray] = None  # (N, 2)
    keypoints_dst: Optional[np.ndarray] = None  # (N, 2)
    
    # 置信度
    confidence: Optional[np.ndarray] = None  # (H, W) 或 (N, H, W)
    
    def __post_init__(self):
        if self.optical_flow is not None:
            self.optical_flow = np.asarray(self.optical_flow)
        if self.homography is not None:
            self.homography = np.asarray(self.homography)
        if self.transform_matrix is not None:
            self.transform_matrix = np.asarray(self.transform_matrix)
        if self.keypoints_src is not None:
            self.keypoints_src = np.asarray(self.keypoints_src)
        if self.keypoints_dst is not None:
            self.keypoints_dst = np.asarray(self.keypoints_dst)
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence)


@dataclass
class DepthInfo:
    """深度信息"""
    # 深度图
    depth_map: Optional[np.ndarray] = None  # (H, W)
    
    # 视差图
    disparity_map: Optional[np.ndarray] = None  # (H, W)
    
    # 点云
    point_cloud: Optional[np.ndarray] = None  # (N, 3)
    
    # 置信度
    confidence: Optional[np.ndarray] = None  # (H, W)
    
    def __post_init__(self):
        if self.depth_map is not None:
            self.depth_map = np.asarray(self.depth_map)
        if self.disparity_map is not None:
            self.disparity_map = np.asarray(self.disparity_map)
        if self.point_cloud is not None:
            self.point_cloud = np.asarray(self.point_cloud)
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence)


class Flow:
    """帧组管理类"""
    
    def __init__(
        self,
        frames: Optional[List[Frame]] = None,
        reference_frame_idx: int = 0,
        alignment_info: Optional[AlignmentInfo] = None,
        depth_info: Optional[DepthInfo] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化Flow
        
        Args:
            frames: 帧列表
            reference_frame_idx: 参考帧索引
            alignment_info: 对齐信息
            depth_info: 深度信息
            metadata: 其他元数据
        """
        self.frames = frames or []
        self.reference_frame_idx = reference_frame_idx
        self.alignment_info = alignment_info or AlignmentInfo()
        self.depth_info = depth_info or DepthInfo()
        self.metadata = metadata or {}
        
        # 验证数据
        self._validate_data()
    
    def _validate_data(self):
        """验证数据"""
        if self.frames and not isinstance(self.frames, list):
            raise ValueError("frames必须是列表")
        
        if self.reference_frame_idx >= len(self.frames):
            raise ValueError("reference_frame_idx超出范围")
        
        if self.frames:
            # 检查所有帧的尺寸是否一致
            ref_shape = self.frames[0].shape[:2]
            for i, frame in enumerate(self.frames):
                if frame.shape[:2] != ref_shape:
                    raise ValueError(f"帧{i}的尺寸{frame.shape[:2]}与参考帧{ref_shape}不一致")
    
    @property
    def num_frames(self) -> int:
        """帧数"""
        return len(self.frames)
    
    @property
    def reference_frame(self) -> Optional[Frame]:
        """参考帧"""
        if self.frames and 0 <= self.reference_frame_idx < len(self.frames):
            return self.frames[self.reference_frame_idx]
        return None
    
    @property
    def shape(self) -> Tuple[int, int]:
        """图像尺寸 (H, W)"""
        if self.frames:
            return self.frames[0].shape[:2]
        return (0, 0)
    
    @property
    def color_format(self):
        """颜色格式"""
        if self.frames:
            return self.frames[0].color_format
        return None
    
    def add_frame(self, frame: Frame):
        """添加帧"""
        if self.frames and frame.shape[:2] != self.shape:
            raise ValueError(f"新帧尺寸{frame.shape[:2]}与现有帧{self.shape}不一致")
        
        self.frames.append(frame)
    
    def remove_frame(self, frame_idx: int):
        """移除帧"""
        if 0 <= frame_idx < len(self.frames):
            frame = self.frames.pop(frame_idx)
            # 调整参考帧索引
            if self.reference_frame_idx >= len(self.frames):
                self.reference_frame_idx = max(0, len(self.frames) - 1)
            elif self.reference_frame_idx > frame_idx:
                self.reference_frame_idx -= 1
            return frame
        return None
    
    def set_reference_frame(self, frame_idx: int):
        """设置参考帧"""
        if 0 <= frame_idx < len(self.frames):
            self.reference_frame_idx = frame_idx
        else:
            raise ValueError(f"frame_idx {frame_idx}超出范围")
    
    def get_frame(self, frame_idx: int) -> Optional[Frame]:
        """获取指定帧"""
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx]
        return None
    
    def get_frame_range(self, start_idx: int, end_idx: int) -> List[Frame]:
        """获取帧范围"""
        start_idx = max(0, start_idx)
        end_idx = min(len(self.frames), end_idx)
        return self.frames[start_idx:end_idx]
    
    def sort_by_timestamp(self):
        """按时间戳排序"""
        self.frames.sort(key=lambda x: x.timestamp)
        # 重新设置参考帧索引
        if self.frames:
            ref_timestamp = self.frames[self.reference_frame_idx].timestamp
            for i, frame in enumerate(self.frames):
                if frame.timestamp == ref_timestamp:
                    self.reference_frame_idx = i
                    break
    
    def get_temporal_range(self) -> Tuple[float, float]:
        """获取时间范围"""
        if not self.frames:
            return (0.0, 0.0)
        
        timestamps = [frame.timestamp for frame in self.frames]
        return (min(timestamps), max(timestamps))
    
    def copy(self) -> 'Flow':
        """复制Flow"""
        return Flow(
            frames=[frame.copy() for frame in self.frames],
            reference_frame_idx=self.reference_frame_idx,
            alignment_info=self.alignment_info,
            depth_info=self.depth_info,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'frames': [frame.to_dict() for frame in self.frames],
            'reference_frame_idx': self.reference_frame_idx,
            'alignment_info': {
                'optical_flow': self.alignment_info.optical_flow.tolist() if self.alignment_info.optical_flow is not None else None,
                'homography': self.alignment_info.homography.tolist() if self.alignment_info.homography is not None else None,
                'transform_matrix': self.alignment_info.transform_matrix.tolist() if self.alignment_info.transform_matrix is not None else None,
                'keypoints_src': self.alignment_info.keypoints_src.tolist() if self.alignment_info.keypoints_src is not None else None,
                'keypoints_dst': self.alignment_info.keypoints_dst.tolist() if self.alignment_info.keypoints_dst is not None else None,
                'confidence': self.alignment_info.confidence.tolist() if self.alignment_info.confidence is not None else None,
            },
            'depth_info': {
                'depth_map': self.depth_info.depth_map.tolist() if self.depth_info.depth_map is not None else None,
                'disparity_map': self.depth_info.disparity_map.tolist() if self.depth_info.disparity_map is not None else None,
                'point_cloud': self.depth_info.point_cloud.tolist() if self.depth_info.point_cloud is not None else None,
                'confidence': self.depth_info.confidence.tolist() if self.depth_info.confidence is not None else None,
            },
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        return (f"Flow(frames={self.num_frames}, shape={self.shape}, "
                f"ref_idx={self.reference_frame_idx}, "
                f"temporal_range={self.get_temporal_range()})")
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx) -> Frame:
        return self.frames[idx]
    
    def __iter__(self):
        return iter(self.frames)

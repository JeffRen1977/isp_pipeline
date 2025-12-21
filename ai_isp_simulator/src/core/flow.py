"""
Flow Class: Frame group management
Contains multi-frame data, alignment information, depth information, etc., used for HDR, multi-camera, video and other scenarios
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from .frame import Frame


@dataclass
class AlignmentInfo:
    """Alignment information"""
    # Optical flow information
    optical_flow: Optional[np.ndarray] = None  # (H, W, 2) or (N, H, W, 2)
    
    # Homography matrix
    homography: Optional[np.ndarray] = None  # (3, 3) or (N, 3, 3)
    
    # Transform matrix
    transform_matrix: Optional[np.ndarray] = None  # (3, 3) or (N, 3, 3)
    
    # Keypoint correspondences
    keypoints_src: Optional[np.ndarray] = None  # (N, 2)
    keypoints_dst: Optional[np.ndarray] = None  # (N, 2)
    
    # Confidence
    confidence: Optional[np.ndarray] = None  # (H, W) or (N, H, W)
    
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
    """Depth information"""
    # Depth map
    depth_map: Optional[np.ndarray] = None  # (H, W)
    
    # Disparity map
    disparity_map: Optional[np.ndarray] = None  # (H, W)
    
    # Point cloud
    point_cloud: Optional[np.ndarray] = None  # (N, 3)
    
    # Confidence
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
    """Frame group management class"""
    
    def __init__(
        self,
        frames: Optional[List[Frame]] = None,
        reference_frame_idx: int = 0,
        alignment_info: Optional[AlignmentInfo] = None,
        depth_info: Optional[DepthInfo] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Flow
        
        Args:
            frames: Frame list
            reference_frame_idx: Reference frame index
            alignment_info: Alignment information
            depth_info: Depth information
            metadata: Other metadata
        """
        self.frames = frames or []
        self.reference_frame_idx = reference_frame_idx
        self.alignment_info = alignment_info or AlignmentInfo()
        self.depth_info = depth_info or DepthInfo()
        self.metadata = metadata or {}
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate data"""
        if self.frames and not isinstance(self.frames, list):
            raise ValueError("frames must be a list")
        
        if self.reference_frame_idx >= len(self.frames):
            raise ValueError("reference_frame_idx out of range")
        
        if self.frames:
            # Check if all frame sizes are consistent
            ref_shape = self.frames[0].shape[:2]
            for i, frame in enumerate(self.frames):
                if frame.shape[:2] != ref_shape:
                    raise ValueError(f"Frame {i} size {frame.shape[:2]} does not match reference frame {ref_shape}")
    
    @property
    def num_frames(self) -> int:
        """Number of frames"""
        return len(self.frames)
    
    @property
    def reference_frame(self) -> Optional[Frame]:
        """Reference frame"""
        if self.frames and 0 <= self.reference_frame_idx < len(self.frames):
            return self.frames[self.reference_frame_idx]
        return None
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Image size (H, W)"""
        if self.frames:
            return self.frames[0].shape[:2]
        return (0, 0)
    
    @property
    def color_format(self):
        """Color format"""
        if self.frames:
            return self.frames[0].color_format
        return None
    
    def add_frame(self, frame: Frame):
        """Add frame"""
        if self.frames and frame.shape[:2] != self.shape:
            raise ValueError(f"New frame size {frame.shape[:2]} does not match existing frames {self.shape}")
        
        self.frames.append(frame)
    
    def remove_frame(self, frame_idx: int):
        """Remove frame"""
        if 0 <= frame_idx < len(self.frames):
            frame = self.frames.pop(frame_idx)
            # Adjust reference frame index
            if self.reference_frame_idx >= len(self.frames):
                self.reference_frame_idx = max(0, len(self.frames) - 1)
            elif self.reference_frame_idx > frame_idx:
                self.reference_frame_idx -= 1
            return frame
        return None
    
    def set_reference_frame(self, frame_idx: int):
        """Set reference frame"""
        if 0 <= frame_idx < len(self.frames):
            self.reference_frame_idx = frame_idx
        else:
            raise ValueError(f"frame_idx {frame_idx} out of range")
    
    def get_frame(self, frame_idx: int) -> Optional[Frame]:
        """Get specified frame"""
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx]
        return None
    
    def get_frame_range(self, start_idx: int, end_idx: int) -> List[Frame]:
        """Get frame range"""
        start_idx = max(0, start_idx)
        end_idx = min(len(self.frames), end_idx)
        return self.frames[start_idx:end_idx]
    
    def sort_by_timestamp(self):
        """Sort by timestamp"""
        self.frames.sort(key=lambda x: x.timestamp)
        # Reset reference frame index
        if self.frames:
            ref_timestamp = self.frames[self.reference_frame_idx].timestamp
            for i, frame in enumerate(self.frames):
                if frame.timestamp == ref_timestamp:
                    self.reference_frame_idx = i
                    break
    
    def get_temporal_range(self) -> Tuple[float, float]:
        """Get temporal range"""
        if not self.frames:
            return (0.0, 0.0)
        
        timestamps = [frame.timestamp for frame in self.frames]
        return (min(timestamps), max(timestamps))
    
    def copy(self) -> 'Flow':
        """Copy Flow"""
        return Flow(
            frames=[frame.copy() for frame in self.frames],
            reference_frame_idx=self.reference_frame_idx,
            alignment_info=self.alignment_info,
            depth_info=self.depth_info,
            metadata=self.metadata.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
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

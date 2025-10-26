"""
RGB Processing Nodes Module

This module contains nodes for processing RGB image data:
- Demosaicing
- Color correction
- Tone mapping
- YUV conversion
"""

from .demosaic import DemosaicNode
from .color_correction import ColorCorrectionNode
from .tone_mapping import ToneMappingNode
from .yuv_conversion import YUVConversionNode

__all__ = [
    "DemosaicNode",
    "ColorCorrectionNode",
    "ToneMappingNode", 
    "YUVConversionNode"
]

#!/usr/bin/env python3
"""
Test script for ISP simulator with real Pixel phone RAW data.

This script processes real RAW images from a Pixel phone through the complete ISP pipeline,
demonstrating RAW white balance before demosaic for proper color reproduction.

Author: AI ISP Simulator
"""

import numpy as np
import cv2
import os
import sys
from pathlib import Path
import re
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.frame import Frame, ColorFormat, BayerPattern
from src.nodes.raw_processing.black_level import BlackLevelNode
from src.nodes.raw_processing.digital_gain import DigitalGainNode
from src.nodes.raw_processing.raw_white_balance import RawWhiteBalanceNode
from src.nodes.raw_processing.demosaic import DemosaicNode
from src.nodes.rgb_processing.color_correction import ColorCorrectionNode
from src.nodes.rgb_processing.tone_mapping import ToneMappingNode
from src.core.node import ImplementationType

# ============================================================================
# Configuration
# ============================================================================

RAW_FILE = Path("../data/front_camera/20250826_220317_770_RAW10_3840x2736_RS_4800_PS_1_cam_3_sensor_raw_output_frame_84.raw")
RESULTS_DIR = Path("results")

# Pipeline configuration
PIPELINE_CONFIG = {
    'black_level': {'black_level_r': 64, 'black_level_gr': 64, 'black_level_gb': 64, 'black_level_b': 64},
    'digital_gain': {'gain_r': 1.5, 'gain_gr': 1.5, 'gain_gb': 1.5, 'gain_b': 1.5},
    # 'raw_wb': {'method': 'manual', 'gain_r': 1.0, 'gain_b': 1.0},  # Disabled: manual gains of 1.0 (no change)
    'raw_wb': {'method': 'gray_world'},  # Enable Gray World AWB
    'demosaic': {'classic_method': 'bilinear'},
    'color_corr': {
        'color_matrix': np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    },
    'tone_mapping': {'gamma': 2.2, 'exposure': 0.5}  # Reduced exposure to prevent over-brightening
}

# ============================================================================
# RAW File I/O
# ============================================================================

def parse_raw_filename(filename: str) -> Dict:
    """Parse metadata from RAW filename."""
    pattern = r'RAW(\d+)_(\d+)x(\d+)_RS_(\d+)_PS_(\d+)'
    match = re.search(pattern, filename)
    
    if not match:
        raise ValueError(f"Could not parse filename: {filename}")
    
    return {
        'width': int(match.group(2)),
        'height': int(match.group(3)),
        'bit_depth': int(match.group(1)),
        'row_stride': int(match.group(4)),
        'pixel_stride': int(match.group(5)),
        'bayer_pattern': BayerPattern.RGGB
    }


def unpack_10bit_raw(raw_bytes: bytes, width: int, height: int, 
                     data_bytes: int, total_pixels: int) -> np.ndarray:
    """Unpack 10-bit RAW data (5 bytes per 4 pixels)."""
    required_bytes = int(total_pixels * 10 / 8)
    
    if data_bytes < required_bytes:
        raise ValueError(f"Insufficient data: need {required_bytes} bytes, got {data_bytes}")
    
    # Extract actual data bytes from each row (excluding padding)
    data_bytes_per_row = int(width * 10 / 8.0)
    actual_data = b''
    row_stride = len(raw_bytes) // height
    
    for row in range(height):
        row_start = row * row_stride
        row_end = row_start + data_bytes_per_row
        if row_end <= len(raw_bytes):
            actual_data += raw_bytes[row_start:row_end]
    
    # Unpack 10-bit data
    unpacked = np.zeros(total_pixels, dtype=np.uint16)
    pixel_idx = 0
    actual_data_array = np.frombuffer(actual_data, dtype=np.uint8)
    
    for byte_idx in range(0, len(actual_data_array) - 4, 5):
        if pixel_idx + 3 >= total_pixels:
            break
        
        b0, b1, b2, b3, b4 = actual_data_array[byte_idx:byte_idx+5]
        
        # Unpack 4 pixels from 5 bytes
        unpacked[pixel_idx] = ((b1 & 0x03) << 8) | b0
        unpacked[pixel_idx + 1] = ((b2 & 0x0F) << 6) | (b1 >> 2)
        unpacked[pixel_idx + 2] = ((b3 & 0x3F) << 4) | (b2 >> 4)
        unpacked[pixel_idx + 3] = (b4 << 2) | (b3 >> 6)
        
        pixel_idx += 4
    
    return unpacked.reshape(height, width)


def load_raw_file(file_path: Path, metadata: Dict) -> np.ndarray:
    """Load and unpack RAW file."""
    with open(file_path, 'rb') as f:
        raw_bytes = f.read()
    
    width = metadata['width']
    height = metadata['height']
    bit_depth = metadata['bit_depth']
    row_stride = metadata['row_stride']
    
    print(f"\n=== Loading RAW File ===")
    print(f"File: {Path(file_path).name}")
    print(f"Size: {len(raw_bytes)} bytes")
    print(f"Resolution: {width}x{height}, Bit depth: {bit_depth}bit")
    print(f"Row stride: {row_stride} bytes")
    
    # Extract actual data (without padding)
    data_bytes_per_row = int(width * bit_depth / 8.0)
    total_data_bytes = height * data_bytes_per_row
    total_pixels = width * height
    
    # Unpack based on bit depth
    if bit_depth == 10:
        return unpack_10bit_raw(raw_bytes, width, height, total_data_bytes, total_pixels)
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")


# ============================================================================
# Visualization
# ============================================================================

def visualize_frame(frame: Frame, title: str, normalize: bool = True) -> np.ndarray:
    """Visualize a frame at any stage of the pipeline."""
    data = frame.data
    
    print(f"\n=== {title} ===")
    print(f"Shape: {data.shape}, Dtype: {data.dtype}")
    print(f"Value range: {data.min()} - {data.max()}")
    
    if normalize:
        # Normalize to 0-255
        if data.dtype == np.uint16:
            max_val = 2**frame.metadata.get('bit_depth', 12) - 1
            data_normalized = (data.astype(np.float32) / max_val * 255).astype(np.uint8)
        else:
            data_max = data.max()
            if data_max > 0:
                data_normalized = (data.astype(np.float32) / data_max * 255).astype(np.uint8)
            else:
                data_normalized = data.astype(np.uint8)
    else:
        data_normalized = data.astype(np.uint8)
    
    return data_normalized


def save_visualization(data: np.ndarray, output_path: str, is_rgb: bool = False) -> None:
    """Save visualization to file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / output_path
    
    if is_rgb and len(data.shape) == 3:
        data_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), data_bgr)
    else:
        cv2.imwrite(str(output_path), data, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# ============================================================================
# ISP Pipeline
# ============================================================================

def create_initial_frame(raw_data: np.ndarray, metadata: Dict) -> Frame:
    """Create initial Frame from RAW data."""
    return Frame(
        data=raw_data,
        color_format=ColorFormat.RAW_BAYER,
        bayer_pattern=metadata['bayer_pattern'],
        metadata={
            'width': metadata['width'],
            'height': metadata['height'],
            'bit_depth': metadata['bit_depth'],
            'row_stride': metadata['row_stride']
        }
    )


def process_isp_pipeline(raw_frame: Frame) -> Tuple[Frame, Dict]:
    """Process RAW frame through complete ISP pipeline."""
    stages = {}
    
    # Step 1: Black Level Correction
    print("\n--- Step 1: Black Level Correction ---")
    black_level_node = BlackLevelNode(config=PIPELINE_CONFIG['black_level'])
    black_level_frame = black_level_node.process({"raw_input": raw_frame})["corrected_output"]
    stages['black_level'] = black_level_frame
    
    # Step 2: Digital Gain
    print("\n--- Step 2: Digital Gain ---")
    digital_gain_node = DigitalGainNode(config=PIPELINE_CONFIG['digital_gain'])
    digital_gain_frame = digital_gain_node.process({"raw_input": black_level_frame})["gained_output"]
    stages['digital_gain'] = digital_gain_frame
    
    # Step 3: RAW White Balance (on Bayer pattern BEFORE demosaic)
    print("\n--- Step 3: RAW White Balance (on Bayer) ---")
    raw_wb_node = RawWhiteBalanceNode(config=PIPELINE_CONFIG['raw_wb'])
    awb_raw_frame = raw_wb_node.process({"raw_input": digital_gain_frame})["corrected_output"]
    stages['raw_wb'] = awb_raw_frame
    
    # Step 4: Demosaic
    print("\n--- Step 4: Demosaic ---")
    demosaic_node = DemosaicNode("demosaic", config=PIPELINE_CONFIG['demosaic'], 
                                 implementation=ImplementationType.CLASSIC)
    demosaic_frame = demosaic_node.process({"input": awb_raw_frame})["output"]
    stages['demosaic'] = demosaic_frame
    
    # Step 5: Color Correction
    print("\n--- Step 5: Color Correction ---")
    color_corr_node = ColorCorrectionNode(config=PIPELINE_CONFIG['color_corr'])
    color_frame = color_corr_node.process({"rgb_input": demosaic_frame})["corrected_output"]
    stages['color_corr'] = color_frame
    
    # Step 6: Tone Mapping
    print("\n--- Step 6: Tone Mapping ---")
    tone_mapping_node = ToneMappingNode(config=PIPELINE_CONFIG['tone_mapping'])
    tone_frame = tone_mapping_node.process({"raw_input": color_frame})["mapped_output"]
    stages['tone_mapping'] = tone_frame
    
    return tone_frame, stages


def save_all_stages(raw_frame: Frame, stages: Dict, final_frame: Frame) -> None:
    """Save visualizations of all pipeline stages."""
    # Raw input
    raw_viz = visualize_frame(raw_frame, "Raw Input")
    save_visualization(raw_viz, "front_01_raw_input.png")
    
    # Pipeline stages
    save_visualization(visualize_frame(stages['black_level'], "After Black Level"), 
                      "front_02_black_level.png")
    save_visualization(visualize_frame(stages['digital_gain'], "After Digital Gain"), 
                      "front_03_digital_gain.png")
    save_visualization(visualize_frame(stages['raw_wb'], "After RAW AWB"), 
                      "front_03_5_awb_raw.png")
    save_visualization(visualize_frame(stages['demosaic'], "After Demosaic", normalize=True), 
                      "front_04_demosaic.png", is_rgb=True)
    save_visualization(visualize_frame(stages['color_corr'], "After Color Correction", normalize=True), 
                      "front_05_color_correction.png", is_rgb=True)
    save_visualization(visualize_frame(stages['tone_mapping'], "After Tone Mapping", normalize=True), 
                      "front_06_tone_mapping.png", is_rgb=True)
    
    # Final output
    print("\n--- Final Output ---")
    final_viz = visualize_frame(final_frame, "Final Output", normalize=True)
    output_bgr = cv2.cvtColor(final_viz, cv2.COLOR_RGB2BGR)
    output_path = RESULTS_DIR / 'front_output_final.png'
    cv2.imwrite(str(output_path), output_bgr)
    print(f"✓ Final output saved to: {output_path}")


def print_summary():
    """Print summary of saved files."""
    print("\n=== Pipeline Complete ===")
    print("Intermediate results saved to results/ folder:")
    print("  results/front_01_raw_input.png")
    print("  results/front_02_black_level.png")
    print("  results/front_03_digital_gain.png")
    print("  results/front_03_5_awb_raw.png (RAW AWB)")
    print("  results/front_04_demosaic.png")
    print("  results/front_05_color_correction.png")
    print("  results/front_06_tone_mapping.png")
    print("  results/front_output_final.png")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main test function."""
    print("=== ISP Simulator Test with Real Front Camera RAW Data ===\n")
    
    # Check if RAW file exists
    if not RAW_FILE.exists():
        print(f"❌ RAW file not found: {RAW_FILE}")
        return
    
    try:
        # Parse metadata and load RAW file
        print("Parsing filename metadata...")
        metadata = parse_raw_filename(RAW_FILE.name)
        print(f"✓ Parsed metadata: {metadata}")
        
        print("\nLoading RAW file...")
        raw_data = load_raw_file(RAW_FILE, metadata)
        print(f"✓ Loaded RAW data: {raw_data.shape}, dtype={raw_data.dtype}")
        
        # Create initial frame
        raw_frame = create_initial_frame(raw_data, metadata)
        
        # Process through ISP pipeline
        final_frame, stages = process_isp_pipeline(raw_frame)
        
        # Save all visualizations
        save_all_stages(raw_frame, stages, final_frame)
        
        # Print summary
        print_summary()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

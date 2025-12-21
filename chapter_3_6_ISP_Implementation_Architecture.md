# Chapter 3.6: ISP Simulator Implementation Architecture

## 3.6.1 Overview

The AI ISP Simulator implements a modular, graph-based architecture for image signal processing pipelines. The system follows a **node-graph** paradigm where each ISP processing stage is an independent node connected through a graph engine that manages data flow, execution order, and performance monitoring. This design supports both traditional algorithms and AI-based implementations.

## 3.6.2 System Architecture

### 3.6.2.1 Core Design Principles

The architecture is built on five key principles: **modularity** (independent, reusable nodes), **flexibility** (Classic/AI/Hybrid implementations), **extensibility** (easy to add new nodes), **type safety** (unified `Frame` data model), and **performance** (topological sorting for optimal execution order).

### 3.6.2.2 File Structure

The project is organized into a clear hierarchical structure:

```
ai_isp_simulator/
├── src/                          # Source code root
│   ├── core/                     # Core infrastructure
│   │   ├── node.py              # Node base class and interfaces
│   │   ├── graph.py             # Graph engine for pipeline execution
│   │   ├── frame.py             # Unified data model (Frame)
│   │   └── flow.py              # Frame group management (for multi-frame processing)
│   │
│   ├── nodes/                    # ISP node implementations
│   │   ├── input/               # Input nodes
│   │   │   └── raw_input.py    # RAW image input handler
│   │   │
│   │   ├── raw_processing/      # RAW domain processing nodes
│   │   │   ├── black_level.py          # Black level correction
│   │   │   ├── digital_gain.py         # Digital gain application
│   │   │   ├── raw_white_balance.py    # RAW-level white balance
│   │   │   ├── demosaic.py             # Bayer to RGB conversion
│   │   │   ├── defective_pixel.py      # Defective pixel correction
│   │   │   ├── lens_shading.py         # Lens shading correction
│   │   │   └── raw_preproc.py          # Combined RAW preprocessing
│   │   │
│   │   ├── rgb_processing/      # RGB domain processing nodes
│   │   │   ├── color_correction.py     # Color correction matrix
│   │   │   ├── tone_mapping.py         # Tone mapping and gamma
│   │   │   ├── awb.py                  # RGB-level white balance (legacy)
│   │   │   └── yuv_conversion.py       # RGB to YUV conversion
│   │   │
│   │   ├── output/              # Output nodes
│   │   │   └── image_output.py # Image saving and visualization
│   │   │
│   │   ├── hdr/                 # HDR processing nodes
│   │   ├── multicam/            # Multi-camera processing
│   │   └── video/               # Video processing nodes
│   │
│   └── quality/                 # Quality analysis and metrics
│
├── configs/                      # Configuration files
│   └── pipelines/               # Pipeline configurations (YAML)
│       ├── photo_mode.yaml
│       ├── video_mode.yaml
│       └── preview_mode.yaml
│
├── data/                         # Sample RAW images
│   ├── front_camera/            # Front camera RAW files
│   └── rear_camera/             # Rear camera RAW files
│
├── results/                      # Output images from processing
│
├── test_front_camera_raw.py     # Main test script
├── main.py                      # Main program entry point
└── README.md                    # Project documentation
```

## 3.6.3 Core Components

### 3.6.3.1 Frame: Unified Data Model

The `Frame` class encapsulates image data and metadata throughout the ISP pipeline. It contains:
- `data`: NumPy array (H×W or H×W×C) with image data
- `color_format`: Data format enum (RAW_BAYER, RGB, YUV, GRAY)
- `bayer_pattern`: Bayer pattern type for RAW data (RGGB, GRBG, GBRG, BGGR)
- `camera_params`, `exposure_params`, `imu_data`: Camera and sensor metadata
- `metadata`: Custom metadata dictionary

This unified model ensures type safety and eliminates format conversion between nodes.

### 3.6.3.2 Node: Base Processing Unit

The `Node` class defines the abstract interface for all ISP processing nodes. It supports three node types: `INPUT` (data loading), `PROCESSING` (data transformation), and `OUTPUT` (result delivery). Each node can have multiple implementation types: `CLASSIC` (traditional algorithms), `AI` (model-based), or `HYBRID` (mixed approach). Core methods include `process()` for data processing, `set_config()` for parameter updates, and `get_performance_stats()` for monitoring. Nodes maintain states: `IDLE`, `PROCESSING`, `ERROR`, or `DISABLED`.

### 3.6.3.3 Graph: Pipeline Execution Engine

The `Graph` class manages the ISP pipeline as a directed acyclic graph (DAG). It handles node management, connections, topological sorting, cycle detection, data flow routing, and performance monitoring. The execution flow: (1) validate graph structure, (2) perform topological sort, (3) execute nodes in sorted order by collecting inputs, processing, and caching outputs, (4) collect final outputs from output nodes.

### 3.6.3.4 Flow: Multi-Frame Management

The `Flow` class manages frame groups for multi-frame processing (HDR, multi-camera fusion, video). It supports reference frame selection, alignment information (optical flow, homography), depth information, and temporal operations.

## 3.6.4 Node Organization

### 3.6.4.1 Input Nodes

**RawInputNode** loads and unpacks RAW files from mobile phones, parsing metadata from filenames, supporting 10-bit format (5 bytes per 4 pixels), and handling row stride/padding correctly.

### 3.6.4.2 RAW Processing Nodes

These nodes operate on RAW Bayer data before demosaicing:

- **BlackLevelNode**: Removes sensor black level offset via per-channel subtraction
- **DigitalGainNode**: Applies per-channel digital gains for exposure/ISO adjustment
- **RawWhiteBalanceNode**: Applies white balance gains to Bayer pattern (BEFORE demosaic) using Gray World algorithm: computes `R_gain = G_avg / R_avg` and `B_gain = G_avg / B_avg`, then applies gains to respective Bayer pixels
- **DemosaicNode**: Converts Bayer pattern to RGB using OpenCV (bilinear, VNG, or edge-aware methods)

### 3.6.4.3 RGB Processing Nodes

These nodes operate on RGB images after demosaicing:

- **ColorCorrectionNode**: Applies 3×3 color correction matrix (CCM) for color reproduction adjustment
- **ToneMappingNode**: Maps linear HDR values to display range using Reinhard algorithm: calculates luminance, applies global scaling and Reinhard curve for perceptual compression, then applies gamma correction

## 3.6.5 Pipeline Construction

The ISP pipeline is constructed by instantiating nodes and connecting them sequentially. A typical frontend pipeline includes:

1. **BlackLevelNode**: Remove sensor black level (config: per-channel black levels)
2. **DigitalGainNode**: Apply per-channel gains (config: gain values)
3. **RawWhiteBalanceNode**: Apply WB to Bayer pattern (config: method='gray_world')
4. **DemosaicNode**: Convert Bayer to RGB (config: classic_method='bilinear')
5. **ColorCorrectionNode**: Apply CCM (config: 3×3 color matrix)
6. **ToneMappingNode**: Map to display range (config: exposure, gamma)

Each node processes a `Frame` object and produces an output `Frame` for the next node, ensuring type safety and consistent data flow.

## 3.6.6 Data Flow

The pipeline transforms data through format and type transitions:

```
RAW Input (uint16, Bayer, 0-1023)
    ↓ Black Level Correction
(float32, Bayer, 0-959)
    ↓ Digital Gain
(float32, Bayer, 0-1438)
    ↓ RAW White Balance
(float32, Bayer, gains applied)
    ↓ Demosaic
(float32, RGB, normalized [0-1])
    ↓ Color Correction
(float32, RGB, [0-1])
    ↓ Tone Mapping
(float32, RGB, [0-1], compressed)
    ↓ Final Output
(uint8, RGB, 0-255)
```

**Key design decisions**: (1) Float32 used throughout for precision, (2) RGB normalized to [0,1] after demosaic, (3) White balance applied to Bayer pattern before demosaic, (4) uint8 conversion only at final output.

## 3.6.7 Key Features

### RAW-Level White Balance

A key innovation is applying white balance gains directly to the Bayer pattern **before** demosaicing, matching real-world ISP behavior. The `RawWhiteBalanceNode` extracts R, Gr, Gb, B pixels, computes statistics using Gray World algorithm, and applies gains to Bayer pixels. This approach provides better color accuracy and proper illuminant handling compared to post-demosaic white balance.

### Extensibility

The architecture supports easy extension: new nodes inherit from `Node`/`ProcessingNode`, multiple implementations (Classic/AI) per node, dynamic configuration via dictionaries, and pipeline composition from code or YAML files.

### Performance

The system uses topological sorting for optimal execution order, caches intermediate results, includes built-in performance monitoring, and supports lazy loading of AI models.

## 3.6.8 Summary

The AI ISP Simulator implements a modular, graph-based architecture with four core components: `Frame` (unified data model), `Node` (processing interface), `Graph` (pipeline execution engine), and `Flow` (multi-frame management). The separation of RAW and RGB processing domains, combined with the unified `Frame` model, ensures type safety throughout the pipeline. The implementation correctly applies white balance at the RAW level before demosaicing, matching real-world ISP behavior, and provides a flexible framework supporting both traditional algorithms and AI-based approaches.


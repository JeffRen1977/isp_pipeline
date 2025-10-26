# AI ISP Simulator

An Image Signal Processor (ISP) simulator for computational photography applications. This project implements a modular ISP pipeline using a graph-based architecture that supports real RAW image processing from mobile phones.

## 🚀 Quick Start

### Requirements

- Python 3.8+
- NumPy, OpenCV, PyYAML

### Installation

```bash
pip install -r requirements.txt
```

### Run the Test

```bash
# Process real RAW images from mobile phones
python test_front_camera_raw.py
```

This will process RAW images from `data/front_camera/` and save results to `results/`.

## 📁 Project Structure

```
ai_isp_simulator/
├── src/                          # Source code
│   ├── core/                     # Core modules
│   │   ├── graph.py             # Graph engine
│   │   ├── node.py              # Node base class
│   │   ├── frame.py             # Unified data model
│   │   └── flow.py              # Frame group management
│   ├── nodes/                    # ISP node implementations
│   │   ├── input/               # Input nodes
│   │   ├── raw_processing/      # RAW domain processing
│   │   ├── rgb_processing/      # RGB domain processing
│   │   └── output/              # Output nodes
│   └── quality/                 # Quality analysis module
├── configs/                      # Configuration files
│   └── pipelines/               # Pipeline configurations
├── data/                         # Sample RAW images
│   ├── front_camera/            # Front camera RAW files
│   └── rear_camera/             # Rear camera RAW files
├── results/                      # Output images
├── test_front_camera_raw.py     # Main test script
└── README.md                     # This file
```

## 🏗️ Architecture

### Core Concepts

**Graph Architecture**: Each ISP function is a node connected by edges forming a processing pipeline.

**Node Types**:
- **Input Nodes**: Data input (e.g., RAW data)
- **Processing Nodes**: Algorithm processing (e.g., demosaic, white balance)
- **Output Nodes**: Result output (e.g., save images)

**Implementation Modes**:
- **Classic**: Traditional algorithm implementation
- **AI**: AI model implementation
- **Hybrid**: Mixed implementation

## 📖 Key Features

### ISP Pipeline Stages

1. **RAW Processing**:
   - Black Level Correction
   - Digital Gain
   - RAW White Balance (on Bayer pattern)
   - Demosaic (Bayer → RGB)

2. **RGB Processing**:
   - Color Correction
   - Tone Mapping

3. **Output**:
   - Save processed images
   - Generate visualizations

### Real RAW Image Support

- Supports 10-bit RAW images from mobile phones
- Parses metadata from filenames (dimensions, bit depth, row stride)
- Handles row stride and padding correctly
- Unpacks 10-bit data (5 bytes per 4 pixels)

## 🔧 Usage Example

### Basic ISP Pipeline

```python
from src.core.frame import Frame, ColorFormat, BayerPattern
from src.nodes.raw_processing.black_level import BlackLevelNode
from src.nodes.raw_processing.digital_gain import DigitalGainNode
from src.nodes.raw_processing.raw_white_balance import RawWhiteBalanceNode
from src.nodes.raw_processing.demosaic import DemosaicNode

# Load RAW image
raw_frame = load_raw_file("image.raw")

# Process through ISP pipeline
# Step 1: Black Level
black_level_node = BlackLevelNode(config={'black_level_r': 64, ...})
bl_frame = black_level_node.process({"raw_input": raw_frame})["corrected_output"]

# Step 2: Digital Gain
digital_gain_node = DigitalGainNode(config={'gain_r': 1.5, ...})
dg_frame = digital_gain_node.process({"raw_input": bl_frame})["gained_output"]

# Step 3: RAW White Balance (on Bayer pattern)
raw_wb_node = RawWhiteBalanceNode(config={'method': 'gray_world'})
wb_frame = raw_wb_node.process({"raw_input": dg_frame})["corrected_output"]

# Step 4: Demosaic
demosaic_node = DemosaicNode("demosaic", config={'classic_method': 'bilinear'})
rgb_frame = demosaic_node.process({"input": wb_frame})["output"]

# Save result
save_image(rgb_frame, "output.png")
```

## 📊 ISP Pipeline Order

The correct ISP frontend pipeline order is:

1. **Black Level Correction** - Subtract sensor black level
2. **Digital Gain** - Apply per-channel gains
3. **RAW White Balance** - Apply WB gains to Bayer pattern (BEFORE demosaic)
4. **Demosaic** - Convert Bayer pattern to RGB
5. **Color Correction** - Apply color correction matrix
6. **Tone Mapping** - Apply gamma and exposure adjustments

> **Important**: White balance should be applied to the RAW Bayer pattern BEFORE demosaic for proper color reproduction.

## 🏗️ ISP Frontend Pipeline Construction

### Overview

The ISP frontend pipeline is constructed by sequentially applying processing nodes to transform RAW sensor data into a displayable image. Each stage operates on a `Frame` object that encapsulates the image data and metadata.

### Pipeline Construction Steps

#### Step 1: Load RAW Image

```python
# Parse metadata from RAW filename
metadata = {
    'width': 3840,
    'height': 2736,
    'bit_depth': 10,
    'row_stride': 4800,
    'bayer_pattern': BayerPattern.RGGB
}

# Load and unpack RAW data
raw_data = load_raw_file("image.raw", metadata)

# Create initial Frame
raw_frame = Frame(
    data=raw_data,
    color_format=ColorFormat.RAW_BAYER,
    bayer_pattern=BayerPattern.RGGB,
    metadata=metadata
)
```

#### Step 2: Black Level Correction

**Purpose**: Remove sensor offset (black level) from RAW data

**Why**: Image sensors have a non-zero reading even when no light hits them

```python
black_level_node = BlackLevelNode(config={
    'black_level_r': 64,
    'black_level_gr': 64,
    'black_level_gb': 64,
    'black_level_b': 64
})

bl_frame = black_level_node.process({"raw_input": raw_frame})["corrected_output"]
```

**Configuration**:
- `black_level_r`: Red channel black level (typically 64-200 for 10-bit sensors)
- `black_level_gr`: Green-Red channel black level
- `black_level_gb`: Green-Blue channel black level  
- `black_level_b`: Blue channel black level

#### Step 3: Digital Gain

**Purpose**: Apply per-channel analog/digital gains

**Why**: Adjust exposure and ISO sensitivity

```python
digital_gain_node = DigitalGainNode(config={
    'gain_r': 1.5,
    'gain_gr': 1.5,
    'gain_gb': 1.5,
    'gain_b': 1.5
})

dg_frame = digital_gain_node.process({"raw_input": bl_frame})["gained_output"]
```

**Configuration**:
- `gain_r`, `gain_gr`, `gain_gb`, `gain_b`: Per-channel gains (typically 0.5-4.0)

#### Step 4: RAW White Balance (Key Innovation)

**Purpose**: Correct color cast caused by illuminant color

**Why**: Different light sources have different color temperatures (e.g., daylight vs. tungsten)

**Critical**: Must be applied **BEFORE** demosaic for proper color reproduction

```python
raw_wb_node = RawWhiteBalanceNode(config={
    'method': 'gray_world',  # or 'manual'
    'gain_r': 1.0,           # Only used for 'manual' mode
    'gain_b': 1.0,           # Only used for 'manual' mode
    'min_gain': 0.5,         # Limit gain range
    'max_gain': 2.0          # Limit gain range
})

wb_frame = raw_wb_node.process({"raw_input": dg_frame})["corrected_output"]
```

**Gray World Algorithm**:
1. Calculates average Red, Green, and Blue values from Bayer statistics
2. Uses Green as reference (less affected by light variations)
3. Computes gains: `R_gain = G_avg / R_avg`, `B_gain = G_avg / B_avg`
4. Applies gains directly to Bayer pattern

**Configuration**:
- `method`: `'gray_world'` (automatic) or `'manual'` (fixed gains)
- `gain_r`, `gain_b`: Manual gains (1.0 = no correction)
- `min_gain`, `max_gain`: Gain limits to prevent overcorrection (typically 0.5-2.0)

#### Step 5: Demosaic

**Purpose**: Convert Bayer pattern to full RGB image

**Why**: Image sensors only capture one color per pixel (RGB Bayer pattern)

```python
demosaic_node = DemosaicNode(
    "demosaic",
    config={'classic_method': 'bilinear'},
    implementation=ImplementationType.CLASSIC
)

rgb_frame = demosaic_node.process({"input": wb_frame})["output"]
```

**Methods**:
- `'bilinear'`: Fast, uses OpenCV's bilinear interpolation
- `'vng'`: Variable Number of Gradients (better quality)
- `'edge_aware'`: Edge-aware demosaic (best quality)

**Configuration**:
- `classic_method`: Algorithm to use
- `quality_enhancement.enabled`: Enable sharpening and noise reduction
- `quality_enhancement.sharpening`: Sharpening strength (0.0-1.0)
- `quality_enhancement.noise_reduction`: Noise reduction strength (0.0-1.0)

#### Step 6: Color Correction

**Purpose**: Adjust color reproduction using Color Correction Matrix (CCM)

**Why**: Convert from sensor RGB space to standard sRGB space

```python
color_corr_node = ColorCorrectionNode(config={
    'color_matrix': np.array([
        [1.0, 0.0, 0.0],  # Identity matrix (no correction)
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32),
    'apply_clipping': True,
    'clip_range': (0.0, 1.0)
})

color_frame = color_corr_node.process({"rgb_input": rgb_frame})["corrected_output"]
```

**Configuration**:
- `color_matrix`: 3x3 matrix for RGB transformation
  - Identity matrix `[[1,0,0],[0,1,0],[0,0,1]]` = no correction
  - Adjust values to shift color balance
- `apply_clipping`: Clip values outside range
- `clip_range`: Valid value range (typically 0.0-1.0)

#### Step 7: Tone Mapping

**Purpose**: Map linear HDR values to display range with perceptual compression

**Why**: Display devices have limited dynamic range

```python
tone_mapping_node = ToneMappingNode(config={
    'mapping_method': 'reinhard',
    'exposure': 0.5,           # Lower = darker image
    'gamma': 2.2,              # Gamma correction
    'white_point': 1.0,
    'black_point': 0.0
})

tone_frame = tone_mapping_node.process({"raw_input": color_frame})["mapped_output"]
```

**Reinhard Tone Mapping**:
1. Calculates luminance from RGB
2. Applies global scaling based on average log luminance
3. Maps to display range using Reinhard curve
4. Applies gamma correction

**Configuration**:
- `exposure`: Exposure adjustment (0.1-2.0, typical 0.3-0.8)
  - Lower values = darker image
  - Higher values = brighter image (risk of overexposure)
- `gamma`: Gamma correction value (typically 2.2 for sRGB)
- `mapping_method`: Currently only `'reinhard'` supported

### Complete Pipeline Example

```python
def process_isp_pipeline(raw_frame):
    """Complete ISP frontend pipeline"""
    
    # 1. Black Level Correction
    black_level_node = BlackLevelNode(config={'black_level_r': 64, ...})
    bl_frame = black_level_node.process({"raw_input": raw_frame})["corrected_output"]
    
    # 2. Digital Gain
    digital_gain_node = DigitalGainNode(config={'gain_r': 1.5, ...})
    dg_frame = digital_gain_node.process({"raw_input": bl_frame})["gained_output"]
    
    # 3. RAW White Balance (BEFORE demosaic!)
    raw_wb_node = RawWhiteBalanceNode(config={'method': 'gray_world'})
    wb_frame = raw_wb_node.process({"raw_input": dg_frame})["corrected_output"]
    
    # 4. Demosaic
    demosaic_node = DemosaicNode("demosaic", config={'classic_method': 'bilinear'})
    rgb_frame = demosaic_node.process({"input": wb_frame})["output"]
    
    # 5. Color Correction
    color_corr_node = ColorCorrectionNode(config={'color_matrix': np.eye(3)})
    color_frame = color_corr_node.process({"rgb_input": rgb_frame})["corrected_output"]
    
    # 6. Tone Mapping
    tone_mapping_node = ToneMappingNode(config={'gamma': 2.2, 'exposure': 0.5})
    tone_frame = tone_mapping_node.process({"raw_input": color_frame})["mapped_output"]
    
    return tone_frame
```

### Configuration Tips

1. **Black Level**: Set to sensor-specific values (typically 64 for 10-bit sensors)

2. **Digital Gain**: Start with 1.5x, adjust based on exposure
   - Too high = overexposure
   - Too low = underexposure

3. **AWB Method**: Use `'gray_world'` for automatic correction, `'manual'` for testing

4. **Demosaic**: Use `'bilinear'` for speed, `'vng'` for quality

5. **Tone Mapping Exposure**: Start with 0.5
   - Overexposed images → reduce to 0.3-0.4
   - Underexposed images → increase to 0.6-0.8

6. **Color Matrix**: Start with identity matrix, adjust for color balance

### Data Flow

```
RAW (uint16) 
  → Black Level (float32)
  → Digital Gain (float32) 
  → RAW AWB (float32, on Bayer)
  → Demosaic (float32, RGB, [0-1])
  → Color Correction (float32, RGB, [0-1])
  → Tone Mapping (float32, RGB, [0-1])
  → Final Image (uint8, RGB)
```

### Visual Debugging

Each stage produces an intermediate result that can be saved for debugging:
- `front_01_raw_input.png` - Original RAW
- `front_02_black_level.png` - After black level
- `front_03_digital_gain.png` - After gain
- `front_03_5_awb_raw.png` - After RAW AWB
- `front_04_demosaic.png` - After demosaic (first RGB)
- `front_05_color_correction.png` - After color correction
- `front_06_tone_mapping.png` - After tone mapping
- `front_output_final.png` - Final result

## �� Key Innovation

**RAW-Level White Balance**: The simulator applies white balance gains directly to the Bayer pattern (before demosaic), which is the correct approach in real ISP pipelines. This ensures:

- Better color accuracy
- Proper handling of different light sources
- More natural color reproduction

## 📈 Current Status

### Implemented Features
- ✅ RAW file loading and unpacking (10-bit support)
- ✅ Black level correction
- ✅ Digital gain
- ✅ RAW white balance (gray world algorithm)
- ✅ Demosaic (bilinear, VNG, edge-aware)
- ✅ Color correction
- ✅ Tone mapping
- ✅ Full ISP pipeline visualization

### Test Results

The `test_front_camera_raw.py` script processes real RAW images from Pixel phones and generates intermediate results at each stage:

#### Pipeline Statistics

Test image: `20250826_220317_770_RAW10_3840x2736_RS_4800_PS_1_cam_3_sensor_raw_output_frame_84.raw`
- **Resolution**: 3840 × 2736 pixels
- **Bit depth**: 10-bit
- **Bayer pattern**: RGGB
- **Processing time**: ~2-3 seconds

#### Stage-by-Stage Results

| Stage | Output Range | Mean | Status |
|-------|--------------|------|--------|
| Raw Input | 0-1023 | ~500 | ✅ Loaded |
| Black Level | 0.0-959.0 | ~450 | ✅ Corrected |
| Digital Gain | 0.0-1023.0 | ~675 | ✅ Gain applied |
| RAW AWB | 0.0-1331.4 | ~850 | ✅ WB gains applied |
| Demosaic | 0.0-1.0 | 0.38 | ✅ RGB converted |
| Color Correction | 0.0-1.0 | 0.38 | ✅ Identity matrix |
| Tone Mapping | 0.0-1.0 | 0.73 | ✅ Exposure 0.5 |
| Final Output | 0-255 | 185 | ✅ Display ready |

#### Quality Metrics

Final processed image:
- **Mean brightness**: 185.2/255 (72.5% - properly exposed)
- **Saturated pixels**: 3.5% (within normal range)
- **Dynamic range**: Good contrast and color reproduction
- **Color accuracy**: Proper color balance after Gray World AWB

#### Intermediate Results

All pipeline stages generate debug outputs:

![RAW Input](ai_isp_simulator/results/front_01_raw_input.png)
*Original RAW data (Bayer pattern)*

![After Black Level](ai_isp_simulator/results/front_02_black_level.png)
*After black level correction*

![After Digital Gain](ai_isp_simulator/results/front_03_digital_gain.png)
*After digital gain (1.5x)*

![After RAW White Balance](ai_isp_simulator/results/front_03_5_awb_raw.png)
*After RAW white balance (Gray World algorithm)*

![After Demosaic](ai_isp_simulator/results/front_04_demosaic.png)
*After demosaic (first RGB image)*

![After Color Correction](ai_isp_simulator/results/front_05_color_correction.png)
*After color correction (identity matrix)*

![After Tone Mapping](ai_isp_simulator/results/front_06_tone_mapping.png)
*After tone mapping (Reinhard algorithm)*

![Final Output](ai_isp_simulator/results/front_output_final.png)
*Final processed image (ready for display)*

## 🔮 Future Work

- [ ] Add more AWB algorithms (white patch, retinex)
- [ ] Implement lens shading correction
- [ ] Add defective pixel correction
- [ ] Support additional Bayer patterns (GRBG, GBRG, BGGR)
- [ ] Optimize for performance
- [ ] Add GUI for parameter tuning

## 📚 Documentation

- **RawWhiteBalanceNode**: Implements gray world AWB on Bayer pattern
- **DemosaicNode**: Supports multiple demosaic algorithms
- **Pipeline Configuration**: All settings in `test_front_camera_raw.py`

## 🤝 Contributing

This is an educational/research project for computational photography. Contributions are welcome!

## 📄 License

[Add your license here]

## 🎉 Acknowledgments

- RAW images from Google Pixel phones
- OpenCV for image processing
- NumPy for numerical computation

---

**Note**: This project is designed for learning and research purposes in computational photography and ISP pipeline development.

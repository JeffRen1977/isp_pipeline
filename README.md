# AI ISP 仿真器

基于Graph的AI ISP（图像信号处理器）仿真器，专为computational photography设计，支持拍照、视频、预览三种模式。

## 架构设计

### 1. 前端（Input Generator）
- **RAW数据输入**: 真实相机dump + 噪声模型/曝光模拟器
- **多摄同步模拟**: 时间戳、视差、畸变，支持双/三摄输入
- **运动合成器**: 虚拟抖动轨迹和运动物体（EIS/VSR验证）

### 2. 后端（Pipeline）
- **模块化设计**: 每个模块可独立开关、替换AI/传统实现
- **统一接口**: 模块I/O格式统一（Tensor = H×W×C + metadata）
- **支持模块**:
  - RAW Preproc: BPC、BLC、LSC（AI vs 传统）
  - Demosaic + RAW去噪（AI）
  - HDR Alignment + Fusion + Tone Mapping
  - Multi-Cam Fusion / Seamless Zoom
  - AWB / EE（轻量传统）
  - EIS（运动估计 + 重采样）
  - VSR/RTSR（AI超分）

### 3. 调试与可视化
- **GUI前端**: PyQt/Streamlit/Flask Dashboard
- **指标展示**: IQA分数曲线、时延估算、功耗推测
- **A/B测试**: 并排对比"AI vs 传统""版本v1 vs v2"

### 4. 底层实现
- **框架**: Python + PyTorch/TensorFlow + OpenCV/Numpy
- **性能优化**: CUDA kernel / TensorRT / ONNX Runtime
- **数据管理**: 结果存储 + 元数据（ISO、曝光、场景标签）

## 项目结构

```
ai_isp_simulator/
├── src/                    # 源代码
│   ├── core/              # 核心模块
│   │   ├── graph.py       # Graph引擎
│   │   ├── node.py        # 节点基类
│   │   ├── frame.py       # 统一数据模型
│   │   └── flow.py        # 帧组管理
│   ├── nodes/             # ISP节点实现
│   │   ├── input/         # 输入节点
│   │   ├── raw_processing/ # RAW域处理
│   │   ├── rgb_processing/ # RGB域处理
│   │   ├── hdr/           # HDR处理
│   │   ├── multicam/      # 多摄处理
│   │   ├── video/         # 视频处理
│   │   └── output/        # 输出节点
│   ├── graphs/            # Graph配置
│   │   ├── pipelines/     # 不同模式pipeline
│   │   └── configs/       # 配置文件
│   ├── quality/           # 质量分析模块
│   └── utils/             # 工具函数
├── configs/               # 配置文件
├── tests/                 # 测试代码
├── examples/              # 使用示例
├── docs/                  # 文档
└── requirements.txt       # 依赖包
```

## 核心特性

### Graph结构一等公民
- 每个模块是节点
- HDR、夜景、变焦、视频都是不同子图
- 支持动态配置和切换

### 统一数据模型
- **Frame**: 图像+曝光/ISO/姿态/内参/IMU
- **Flow**: 帧组+对齐/深度
- 保证多帧/多摄/视频在一个接口下工作

### 可切换实现
- 同一节点支持classic|ai参数
- 方便A/B测试与回退
- 支持不同tuning参数

### 典型子图组合

#### Photo: 单摄HDR
```
raw → RAW-AI → burst → align → hdr_fusion → tone_mapping → awb → ee → iqa
```

#### Photo: 多摄无感变焦
```
multicam_mux → calib → seamless_zoom → iqa
```

#### Video: HDR + EIS + VSR（实时）
```
raw → RAW-AI → burst → align → tone_mapping → semantics → policy
raw → burst → eis_motion → warp_resample → vsr → iqa
```

## 安装和使用

```bash
# 安装依赖
pip install -r requirements.txt

# 运行示例
python examples/photo_mode.py
python examples/video_mode.py
python examples/preview_mode.py
```

## 路线图

1. **Step 1**: 最小可运行pipeline（RAW → AI demosaic → AWB/EE → RGB）
2. **Step 2**: 加HDR（多帧对齐+融合+TM），验证照片质量
3. **Step 3**: 加多摄（标定+校准+无感切换/融合）
4. **Step 4**: 加视频链路（EIS+VSR+实时预览降采样路径）
5. **Step 5**: 接入IQA（离线批量打分+在线实时预测）
6. **Step 6**: 整理API/模块接口→定义SoC实现对接规范

## 许可证

MIT License

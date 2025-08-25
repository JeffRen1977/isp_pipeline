# AI ISP 仿真器快速开始指南

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 2. 安装依赖

```bash
# 进入项目目录
cd ai_isp_simulator

# 安装依赖
pip install -r requirements.txt
```

### 3. 运行演示

#### 方式1: 使用启动脚本（推荐）
```bash
python start.py
```
然后选择相应的功能运行。

#### 方式2: 直接运行演示脚本
```bash
# 基础演示
python run_demo.py

# 高级演示
python run_advanced_demo.py

# 拍照模式
python examples/photo_mode.py
```

#### 方式3: 使用主程序
```bash
# 拍照模式
python main.py photo

# 视频模式
python main.py video

# 预览模式
python main.py preview
```

## 📁 项目结构

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
│   │   └── output/        # 输出节点
│   └── quality/           # 质量分析模块
├── configs/               # 配置文件
│   └── pipelines/         # Pipeline配置
├── examples/              # 使用示例
├── tests/                 # 测试代码
├── start.py              # 启动脚本
├── run_demo.py           # 基础演示
├── run_advanced_demo.py  # 高级演示
└── main.py               # 主程序
```

## 🔧 核心概念

### Graph架构
- 每个ISP功能都是一个节点
- 节点通过有向边连接形成处理pipeline
- 支持动态配置和节点切换

### 节点类型
- **输入节点**: 数据输入（如RAW数据）
- **处理节点**: 算法处理（如去马赛克、白平衡）
- **输出节点**: 结果输出（如保存图像）

### 实现方式
- **Classic**: 传统算法实现
- **AI**: AI模型实现
- **Hybrid**: 混合实现

## 📖 使用示例

### 创建简单的pipeline

```python
from src.core.graph import Graph
from src.nodes.input.raw_input import RawInputNode
from src.nodes.raw_processing.demosaic import DemosaicNode

# 创建Graph
graph = Graph("simple_pipeline")

# 创建节点
raw_input = RawInputNode("raw_input", config={...})
demosaic = DemosaicNode("demosaic", implementation="classic")

# 添加节点
graph.add_node(raw_input)
graph.add_node(demosaic)

# 连接节点
graph.connect_nodes("raw_input", "demosaic")

# 验证Graph
if graph.validate():
    # 执行pipeline
    outputs = graph.execute({"raw_input": frame})
```

### 配置节点参数

```python
# 配置RAW输入节点
raw_input_config = {
    "input_type": "simulation",
    "bayer_pattern": "rggb",
    "width": 4000,
    "height": 3000,
    "bit_depth": 12,
    "noise_model": {
        "enabled": True,
        "read_noise": 2.0
    }
}

raw_input = RawInputNode("raw_input", config=raw_input_config)
```

### 切换实现方式

```python
# 从经典实现切换到AI实现
demosaic_node.set_implementation("ai")

# 或从AI实现切换回经典实现
demosaic_node.set_implementation("classic")
```

## ⚙️ 配置Pipeline

Pipeline配置文件位于 `configs/pipelines/` 目录下，支持YAML格式：

```yaml
pipeline:
  name: "photo_mode"
  nodes:
    raw_input:
      type: "RawInputNode"
      config:
        input_type: "simulation"
        bayer_pattern: "rggb"
    
    demosaic:
      type: "DemosaicNode"
      implementation: "ai"
      config:
        ai_model_path: "models/demosaic.onnx"
  
  connections:
    - from: "raw_input.output"
      to: "demosaic.input"
```

## 🧪 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试文件
python -m pytest tests/test_basic.py -v

# 运行特定测试函数
python -m pytest tests/test_basic.py::TestFrame::test_frame_creation -v
```

## 🔍 调试和开发

### 日志设置
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 性能分析
```python
# 获取Graph性能统计
stats = graph.get_performance_stats()
print(f"平均执行时间: {stats['avg_execution_time']:.3f}s")

# 获取节点性能统计
node_stats = node.get_performance_stats()
print(f"节点处理次数: {node_stats['total_processed']}")
```

### 错误处理
```python
try:
    outputs = graph.execute(inputs)
except Exception as e:
    print(f"Pipeline执行失败: {e}")
    # 检查节点状态
    for node_id, node in graph.nodes.items():
        if node.status == "error":
            print(f"节点 {node_id} 错误: {node.error_message}")
```

## 📚 扩展开发

### 添加新节点
1. 继承相应的基类（如`ProcessingNode`）
2. 实现必要的方法
3. 在`src/nodes/__init__.py`中注册

### 添加新算法
1. 在节点中添加新的实现方法
2. 在配置中添加相应参数
3. 更新文档和测试

## 🆘 常见问题

### Q: 如何加载真实的RAW数据？
A: 修改`RawInputNode`的配置，设置`input_type: "file"`并指定`file_path`。

### Q: 如何集成自己的AI模型？
A: 在节点的`_load_ai_model`方法中实现模型加载逻辑，支持PyTorch、TensorFlow、ONNX等格式。

### Q: 如何优化性能？
A: 使用GPU加速、模型量化、并行处理等技术，在配置中启用相应的优化选项。

### Q: 如何添加新的图像质量指标？
A: 在`src/quality/`模块中添加新的质量评估算法，并在配置中启用。

## 📞 获取帮助

- 查看项目README.md获取详细信息
- 运行`python start.py`选择帮助选项
- 检查测试代码了解具体用法
- 查看配置文件了解参数含义

## 🎯 下一步

1. 运行基础演示熟悉系统
2. 查看配置文件了解参数
3. 尝试修改pipeline配置
4. 集成自己的算法和模型
5. 扩展新的功能节点

祝您使用愉快！🎉

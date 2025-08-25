#!/usr/bin/env python3
"""
AI ISP 仿真器主程序
支持拍照、视频、预览三种模式
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.graph import Graph
from core.frame import Frame, ColorFormat, BayerPattern
from nodes.input.raw_input import RawInputNode
from nodes.raw_processing.demosaic import DemosaicNode


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        return {}


def create_pipeline_from_config(config: Dict[str, Any]) -> Graph:
    """根据配置创建pipeline"""
    pipeline_config = config.get("pipeline", {})
    pipeline_name = pipeline_config.get("name", "default_pipeline")
    
    # 创建Graph
    graph = Graph(pipeline_name)
    
    # 创建节点
    nodes_config = pipeline_config.get("nodes", {})
    nodes = {}
    
    for node_id, node_config in nodes_config.items():
        node_type = node_config.get("type")
        implementation = node_config.get("implementation", "classic")
        node_config_dict = node_config.get("config", {})
        
        if node_type == "RawInputNode":
            node = RawInputNode(node_id, node_config_dict)
        elif node_type == "DemosaicNode":
            impl = ImplementationType(implementation)
            node = DemosaicNode(node_id, node_config_dict, impl)
        else:
            logging.warning(f"未知的节点类型: {node_type}")
            continue
        
        nodes[node_id] = node
        graph.add_node(node)
    
    # 连接节点
    connections = pipeline_config.get("connections", [])
    for connection in connections:
        from_node = connection.get("from")
        to_node = connection.get("to")
        
        if from_node and to_node:
            from_parts = from_node.split(".")
            to_parts = to_node.split(".")
            
            if len(from_parts) == 2 and len(to_parts) == 2:
                from_node_id, from_port = from_parts
                to_node_id, to_port = to_parts
                
                if from_node_id in nodes and to_node_id in nodes:
                    graph.connect_nodes(from_node_id, to_node_id, from_port, to_port)
                else:
                    logging.warning(f"连接节点失败: {from_node_id} -> {to_node_id}")
    
    return graph


def run_photo_mode(config_path: str):
    """运行拍照模式"""
    print("=== 拍照模式 ===")
    
    # 加载配置
    config = load_config(config_path)
    if not config:
        print("❌ 配置加载失败")
        return
    
    # 创建pipeline
    graph = create_pipeline_from_config(config)
    
    # 验证Graph
    if not graph.validate():
        print("❌ Graph验证失败")
        return
    
    print("✅ Graph验证通过")
    
    # 获取RAW输入节点
    raw_input = graph.get_node("raw_input")
    if not raw_input:
        print("❌ 未找到RAW输入节点")
        return
    
    # 生成测试帧
    print("📸 生成测试帧...")
    frame = raw_input.generate_frame()
    print(f"  生成帧: {frame.shape}, ISO={frame.exposure_params.iso}")
    
    # 执行pipeline
    print("🔄 执行pipeline...")
    try:
        outputs = graph.execute({"raw_input": frame})
        print("✅ Pipeline执行成功")
        
        # 显示输出
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                print(f"  输出 {output_name}: {output_data.shape}, "
                      f"格式={output_data.color_format.value}")
        
    except Exception as e:
        print(f"❌ Pipeline执行失败: {e}")
        logging.error(f"Pipeline执行失败: {e}", exc_info=True)


def run_video_mode(config_path: str):
    """运行视频模式"""
    print("=== 视频模式 ===")
    
    # 加载配置
    config = load_config(config_path)
    if not config:
        print("❌ 配置加载失败")
        return
    
    # 创建pipeline
    graph = create_pipeline_from_config(config)
    
    # 验证Graph
    if not graph.validate():
        print("❌ Graph验证失败")
        return
    
    print("✅ Graph验证通过")
    
    # 获取RAW输入节点
    raw_input = graph.get_node("raw_input")
    if not raw_input:
        print("❌ 未找到RAW输入节点")
        return
    
    # 生成多帧测试
    print("📹 生成视频帧序列...")
    num_frames = 10
    
    for i in range(num_frames):
        frame = raw_input.generate_frame()
        print(f"  生成帧 {i+1}: {frame.shape}")
        
        # 执行pipeline
        try:
            outputs = graph.execute({"raw_input": frame})
            print(f"  ✅ 帧 {i+1} 处理完成")
            
        except Exception as e:
            print(f"  ❌ 帧 {i+1} 处理失败: {e}")
            break
    
    print(f"🎬 视频处理完成，共处理{num_frames}帧")


def run_preview_mode(config_path: str):
    """运行预览模式"""
    print("=== 预览模式 ===")
    
    # 加载配置
    config = load_config(config_path)
    if not config:
        print("❌ 配置加载失败")
        return
    
    # 创建pipeline
    graph = create_pipeline_from_config(config)
    
    # 验证Graph
    if not graph.validate():
        print("❌ Graph验证失败")
        return
    
    print("✅ Graph验证通过")
    
    # 获取RAW输入节点
    raw_input = graph.get_node("raw_input")
    if not raw_input:
        print("❌ 未找到RAW输入节点")
        return
    
    # 生成预览帧
    print("👁️ 生成预览帧...")
    frame = raw_input.generate_frame()
    print(f"  生成预览帧: {frame.shape}")
    
    # 执行pipeline
    print("🔄 执行预览pipeline...")
    try:
        outputs = graph.execute({"raw_input": frame})
        print("✅ 预览pipeline执行成功")
        
        # 显示输出
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                print(f"  预览输出 {output_name}: {output_data.shape}, "
                      f"格式={output_data.color_format.value}")
        
    except Exception as e:
        print(f"❌ 预览pipeline执行失败: {e}")
        logging.error(f"预览pipeline执行失败: {e}", exc_info=True)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI ISP 仿真器")
    parser.add_argument(
        "mode",
        choices=["photo", "video", "preview"],
        help="运行模式"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 确定配置文件路径
    if args.config:
        config_path = args.config
    else:
        config_dir = Path(__file__).parent / "configs" / "pipelines"
        if args.mode == "photo":
            config_path = config_dir / "photo_mode.yaml"
        elif args.mode == "video":
            config_path = config_dir / "video_mode.yaml"
        else:  # preview
            config_path = config_dir / "preview_mode.yaml"
    
    # 检查配置文件是否存在
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    print(f"📁 使用配置文件: {config_path}")
    
    try:
        # 根据模式运行
        if args.mode == "photo":
            run_photo_mode(config_path)
        elif args.mode == "video":
            run_video_mode(config_path)
        elif args.mode == "preview":
            run_preview_mode(config_path)
        
        print("\n🎉 运行完成！")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        logging.error(f"程序异常: {e}", exc_info=True)


if __name__ == "__main__":
    # 添加缺失的import
    from core.node import ImplementationType
    
    main()

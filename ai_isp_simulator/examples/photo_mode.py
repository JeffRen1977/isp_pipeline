#!/usr/bin/env python3
"""
拍照模式示例
演示如何使用AI ISP仿真器进行拍照处理
"""

import sys
import os
import time
import logging
import yaml
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.graph import Graph
from core.frame import Frame, ColorFormat, BayerPattern
from nodes.input.raw_input import RawInputNode
from nodes.raw_processing.demosaic import DemosaicNode


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_pipeline_config(config_path: str) -> dict:
    """加载pipeline配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_photo_pipeline() -> Graph:
    """创建拍照pipeline"""
    # 创建Graph
    graph = Graph("photo_pipeline")
    
    # 创建节点
    raw_input = RawInputNode(
        node_id="raw_input",
        config={
            "input_type": "simulation",
            "bayer_pattern": "rggb",
            "width": 4000,
            "height": 3000,
            "bit_depth": 12,
            "noise_model": {
                "enabled": True,
                "read_noise": 2.0,
                "shot_noise": 0.1,
                "dark_current": 0.01
            },
            "exposure_simulation": {
                "enabled": True,
                "exposure_times": [1.0/30.0, 1.0/15.0, 1.0/8.0],
                "iso_values": [100, 200, 400]
            }
        }
    )
    
    demosaic = DemosaicNode(
        node_id="demosaic",
        implementation="ai",
        config={
            "classic_method": "edge_aware",
            "ai_model_path": "models/demosaic_unet.onnx",
            "quality_enhancement": {
                "enabled": True,
                "sharpening": 0.1,
                "noise_reduction": 0.05
            }
        }
    )
    
    # 添加节点到Graph
    graph.add_node(raw_input)
    graph.add_node(demosaic)
    
    # 连接节点
    graph.connect_nodes("raw_input", "demosaic")
    
    return graph


def run_photo_pipeline():
    """运行拍照pipeline"""
    print("=== AI ISP 仿真器 - 拍照模式示例 ===")
    
    # 创建pipeline
    graph = create_photo_pipeline()
    
    # 验证Graph
    if not graph.validate():
        print("❌ Graph验证失败")
        return
    
    print("✅ Graph验证通过")
    
    # 生成HDR burst
    print("\n📸 生成HDR burst序列...")
    raw_input = graph.get_node("raw_input")
    
    frames = []
    for i in range(3):
        frame = raw_input.generate_frame()
        frames.append(frame)
        print(f"  生成帧 {i+1}: {frame.shape}, ISO={frame.exposure_params.iso}, "
              f"曝光时间={frame.exposure_params.exposure_time:.3f}s")
    
    # 执行pipeline
    print("\n🔄 执行pipeline...")
    start_time = time.time()
    
    try:
        # 处理第一帧
        outputs = graph.execute({"raw_input": frames[0]})
        execution_time = time.time() - start_time
        
        print(f"✅ Pipeline执行成功，耗时: {execution_time:.3f}s")
        
        # 显示结果
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                print(f"  输出 {output_name}: {output_data.shape}, "
                      f"格式={output_data.color_format.value}")
            else:
                print(f"  输出 {output_name}: {type(output_data)}")
        
        # 性能统计
        stats = graph.get_performance_stats()
        print(f"\n📊 性能统计:")
        print(f"  总执行次数: {stats['total_executions']}")
        print(f"  平均执行时间: {stats['avg_execution_time']:.3f}s")
        print(f"  最小执行时间: {stats['min_execution_time']:.3f}s")
        print(f"  最大执行时间: {stats['max_execution_time']:.3f}s")
        
        # 节点性能统计
        print(f"\n🔧 节点性能统计:")
        for node_id, node in graph.nodes.items():
            node_stats = node.get_performance_stats()
            print(f"  {node_id}: 处理{node_stats['total_processed']}次, "
                  f"平均时间{node_stats['avg_processing_time']:.3f}s")
        
    except Exception as e:
        print(f"❌ Pipeline执行失败: {e}")
        logging.error(f"Pipeline执行失败: {e}", exc_info=True)
    
    # 清理
    graph.reset()
    print("\n🧹 清理完成")


def run_hdr_photo():
    """运行HDR拍照"""
    print("\n=== HDR拍照示例 ===")
    
    # 创建pipeline
    graph = create_photo_pipeline()
    
    if not graph.validate():
        print("❌ Graph验证失败")
        return
    
    # 生成HDR burst
    raw_input = graph.get_node("raw_input")
    frames = raw_input.generate_hdr_burst(3)
    
    print(f"📸 生成{len(frames)}帧HDR burst")
    
    # 处理每一帧
    processed_frames = []
    for i, frame in enumerate(frames):
        print(f"\n处理帧 {i+1}...")
        start_time = time.time()
        
        try:
            outputs = graph.execute({"raw_input": frame})
            processing_time = time.time() - start_time
            
            # 获取处理后的帧
            for output_name, output_data in outputs.items():
                if isinstance(output_data, Frame):
                    processed_frames.append(output_data)
                    print(f"  ✅ 帧 {i+1} 处理完成，耗时: {processing_time:.3f}s")
                    print(f"     输出尺寸: {output_data.shape}")
                    print(f"     输出格式: {output_data.color_format.value}")
                    break
            
        except Exception as e:
            print(f"  ❌ 帧 {i+1} 处理失败: {e}")
    
    print(f"\n🎯 HDR处理完成，共处理{len(processed_frames)}帧")
    
    # 清理
    graph.reset()


def main():
    """主函数"""
    setup_logging()
    
    try:
        # 运行基本拍照pipeline
        run_photo_pipeline()
        
        # 运行HDR拍照
        run_hdr_photo()
        
        print("\n🎉 所有示例运行完成！")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        logging.error(f"程序异常: {e}", exc_info=True)


if __name__ == "__main__":
    main()

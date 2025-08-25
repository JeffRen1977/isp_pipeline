#!/usr/bin/env python3
"""
AI ISP 仿真器高级演示脚本
展示完整的pipeline和多种节点
"""

import sys
import time
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.graph import Graph
from core.frame import Frame, ColorFormat, BayerPattern
from nodes.input.raw_input import RawInputNode
from nodes.raw_processing.demosaic import DemosaicNode
from nodes.raw_processing.raw_preproc import RawPreprocNode
from nodes.rgb_processing.awb import AWBNode


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_advanced_pipeline() -> Graph:
    """创建高级pipeline"""
    print("🔧 创建高级pipeline...")
    
    # 创建Graph
    graph = Graph("advanced_isp_pipeline")
    
    # 创建节点
    print("  创建RAW输入节点...")
    raw_input = RawInputNode(
        node_id="raw_input",
        config={
            "input_type": "simulation",
            "bayer_pattern": "rggb",
            "width": 1024,
            "height": 768,
            "bit_depth": 10,
            "noise_model": {
                "enabled": True,
                "read_noise": 1.5,
                "shot_noise": 0.08,
                "dark_current": 0.005
            },
            "exposure_simulation": {
                "enabled": True,
                "exposure_times": [1.0/30.0, 1.0/15.0, 1.0/8.0],
                "iso_values": [100, 200, 400]
            }
        }
    )
    
    print("  创建RAW预处理节点...")
    raw_preproc = RawPreprocNode(
        node_id="raw_preproc",
        implementation="classic",
        config={
            "bpc_enabled": True,
            "blc_enabled": True,
            "lsc_enabled": True,
            "bpc_config": {
                "threshold": 2.5,
                "window_size": 5
            },
            "blc_config": {
                "black_level": 32,
                "method": "subtract"
            },
            "lsc_config": {
                "method": "polynomial",
                "coefficients": [1.0, 0.08, 0.03, 0.008]
            }
        }
    )
    
    print("  创建去马赛克节点...")
    demosaic = DemosaicNode(
        node_id="demosaic",
        implementation="classic",
        config={
            "classic_method": "edge_aware",
            "quality_enhancement": {
                "enabled": True,
                "sharpening": 0.15,
                "noise_reduction": 0.08
            }
        }
    )
    
    print("  创建白平衡节点...")
    awb = AWBNode(
        node_id="awb",
        implementation="classic",
        config={
            "method": "gray_world",
            "temperature": 5500,
            "tint": 0.0,
            "adaptive": True,
            "gray_world_config": {
                "saturation_threshold": 0.7,
                "brightness_threshold": 0.15
            }
        }
    )
    
    # 添加节点到Graph
    print("  添加节点到Graph...")
    graph.add_node(raw_input)
    graph.add_node(raw_preproc)
    graph.add_node(demosaic)
    graph.add_node(awb)
    
    # 连接节点
    print("  连接节点...")
    graph.connect_nodes("raw_input", "raw_preproc")
    graph.connect_nodes("raw_preproc", "demosaic")
    graph.connect_nodes("demosaic", "awb")
    
    return graph


def run_pipeline_test(graph: Graph, test_name: str, num_frames: int = 1):
    """运行pipeline测试"""
    print(f"\n🧪 运行测试: {test_name}")
    print("=" * 50)
    
    # 验证Graph
    if not graph.validate():
        print("❌ Graph验证失败")
        return False
    
    print("✅ Graph验证通过")
    
    # 获取RAW输入节点
    raw_input = graph.get_node("raw_input")
    if not raw_input:
        print("❌ 未找到RAW输入节点")
        return False
    
    # 生成测试帧
    print(f"📸 生成{num_frames}帧测试数据...")
    frames = []
    for i in range(num_frames):
        frame = raw_input.generate_frame()
        frames.append(frame)
        print(f"  帧 {i+1}: {frame.shape}, ISO={frame.exposure_params.iso}, "
              f"曝光时间={frame.exposure_params.exposure_time:.3f}s")
    
    # 执行pipeline
    print("🔄 执行pipeline...")
    start_time = time.time()
    
    processed_frames = []
    for i, frame in enumerate(frames):
        frame_start_time = time.time()
        
        try:
            outputs = graph.execute({"raw_input": frame})
            frame_processing_time = time.time() - frame_start_time
            
            # 获取处理后的帧
            for output_name, output_data in outputs.items():
                if isinstance(output_data, Frame):
                    processed_frames.append(output_data)
                    print(f"  ✅ 帧 {i+1} 处理完成，耗时: {frame_processing_time:.3f}s")
                    print(f"     输出 {output_name}: {output_data.shape}, "
                          f"格式={output_data.color_format.value}")
                    break
            
        except Exception as e:
            print(f"  ❌ 帧 {i+1} 处理失败: {e}")
            logging.error(f"帧 {i+1} 处理失败: {e}", exc_info=True)
            return False
    
    total_time = time.time() - start_time
    print(f"✅ Pipeline执行完成，总耗时: {total_time:.3f}s")
    
    return True


def run_performance_analysis(graph: Graph):
    """运行性能分析"""
    print(f"\n📊 性能分析")
    print("=" * 50)
    
    # Graph性能统计
    graph_stats = graph.get_performance_stats()
    print("Graph性能统计:")
    print(f"  总执行次数: {graph_stats['total_executions']}")
    print(f"  平均执行时间: {graph_stats['avg_execution_time']:.3f}s")
    print(f"  最小执行时间: {graph_stats['min_execution_time']:.3f}s")
    print(f"  最大执行时间: {graph_stats['max_execution_time']:.3f}s")
    if 'std_execution_time' in graph_stats:
        print(f"  标准差: {graph_stats['std_execution_time']:.3f}s")
    
    # 节点性能统计
    print("\n节点性能统计:")
    for node_id, node in graph.nodes.items():
        node_stats = node.get_performance_stats()
        print(f"  {node_id}:")
        print(f"    处理次数: {node_stats['total_processed']}")
        print(f"    平均时间: {node_stats['avg_processing_time']:.3f}s")
        print(f"    最小时间: {node_stats['min_processing_time']:.3f}s")
        print(f"    最大时间: {node_stats['max_processing_time']:.3f}s")
        if 'std_processing_time' in node_stats:
            print(f"    标准差: {node_stats['std_processing_time']:.3f}s")


def run_implementation_comparison(graph: Graph):
    """运行实现方式对比"""
    print(f"\n🔄 实现方式对比")
    print("=" * 50)
    
    # 获取去马赛克节点
    demosaic_node = graph.get_node("demosaic")
    if not demosaic_node:
        print("❌ 未找到去马赛克节点")
        return
    
    # 生成测试帧
    raw_input = graph.get_node("raw_input")
    test_frame = raw_input.generate_frame()
    
    # 测试经典实现
    print("测试经典实现...")
    demosaic_node.set_implementation("classic")
    start_time = time.time()
    
    try:
        outputs = graph.execute({"raw_input": test_frame})
        classic_time = time.time() - start_time
        print(f"  ✅ 经典实现完成，耗时: {classic_time:.3f}s")
        
        # 获取输出
        classic_output = None
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                classic_output = output_data
                break
        
    except Exception as e:
        print(f"  ❌ 经典实现失败: {e}")
        return
    
    # 测试AI实现（如果可用）
    print("测试AI实现...")
    demosaic_node.set_implementation("ai")
    start_time = time.time()
    
    try:
        outputs = graph.execute({"raw_input": test_frame})
        ai_time = time.time() - start_time
        print(f"  ✅ AI实现完成，耗时: {ai_time:.3f}s")
        
        # 获取输出
        ai_output = None
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                ai_output = output_data
                break
        
    except Exception as e:
        print(f"  ❌ AI实现失败: {e}")
        ai_time = float('inf')
        ai_output = None
    
    # 对比结果
    print("\n对比结果:")
    if classic_output and ai_output:
        print(f"  经典实现: {classic_output.shape}, 耗时: {classic_time:.3f}s")
        print(f"  AI实现: {ai_output.shape}, 耗时: {ai_time:.3f}s")
        
        if ai_time < float('inf'):
            speedup = classic_time / ai_time
            print(f"  加速比: {speedup:.2f}x")
        else:
            print("  AI实现不可用")
    else:
        print("  无法获取输出进行对比")


def main():
    """主函数"""
    setup_logging()
    
    try:
        print("🚀 AI ISP 仿真器高级演示")
        print("=" * 60)
        
        # 创建pipeline
        graph = create_advanced_pipeline()
        
        # 运行基本测试
        if not run_pipeline_test(graph, "基本功能测试", 3):
            print("❌ 基本测试失败")
            return
        
        # 运行性能分析
        run_performance_analysis(graph)
        
        # 运行实现方式对比
        run_implementation_comparison(graph)
        
        # 清理
        graph.reset()
        print("\n🧹 清理完成")
        
        print("\n🎉 高级演示完成！")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        logging.error(f"程序异常: {e}", exc_info=True)


if __name__ == "__main__":
    main()

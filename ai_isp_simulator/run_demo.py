#!/usr/bin/env python3
"""
AI ISP 仿真器演示脚本
快速测试系统功能
"""

import sys
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.graph import Graph
from core.frame import Frame, ColorFormat, BayerPattern
from nodes.input.raw_input import RawInputNode
from nodes.raw_processing.demosaic import DemosaicNode


def run_simple_demo():
    """运行简单演示"""
    print("🚀 AI ISP 仿真器演示")
    print("=" * 50)
    
    # 创建Graph
    print("1. 创建Graph...")
    graph = Graph("demo_pipeline")
    
    # 创建节点
    print("2. 创建节点...")
    raw_input = RawInputNode(
        node_id="raw_input",
        config={
            "input_type": "simulation",
            "bayer_pattern": "rggb",
            "width": 512,
            "height": 512,
            "bit_depth": 8,
            "noise_model": {"enabled": False},
            "exposure_simulation": {"enabled": False}
        }
    )
    
    demosaic = DemosaicNode(
        node_id="demosaic",
        implementation="classic",
        config={
            "classic_method": "bilinear",
            "quality_enhancement": {"enabled": False}
        }
    )
    
    # 添加节点
    graph.add_node(raw_input)
    graph.add_node(demosaic)
    
    # 连接节点
    print("3. 连接节点...")
    graph.connect_nodes("raw_input", "demosaic")
    
    # 验证Graph
    print("4. 验证Graph...")
    if not graph.validate():
        print("❌ Graph验证失败")
        return
    
    print("✅ Graph验证通过")
    
    # 生成测试帧
    print("5. 生成测试帧...")
    frame = raw_input.generate_frame()
    print(f"   生成帧: {frame.shape}, 格式: {frame.color_format.value}")
    
    # 执行pipeline
    print("6. 执行pipeline...")
    start_time = time.time()
    
    try:
        outputs = graph.execute({"raw_input": frame})
        execution_time = time.time() - start_time
        
        print(f"✅ Pipeline执行成功，耗时: {execution_time:.3f}s")
        
        # 显示输出
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                print(f"   输出 {output_name}: {output_data.shape}, "
                      f"格式: {output_data.color_format.value}")
        
        # 性能统计
        stats = graph.get_performance_stats()
        print(f"\n📊 性能统计:")
        print(f"   总执行次数: {stats['total_executions']}")
        print(f"   平均执行时间: {stats['avg_execution_time']:.3f}s")
        
    except Exception as e:
        print(f"❌ Pipeline执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 演示完成！")


if __name__ == "__main__":
    try:
        run_simple_demo()
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    except Exception as e:
        print(f"\n💥 程序异常: {e}")
        import traceback
        traceback.print_exc()

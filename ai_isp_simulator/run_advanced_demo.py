#!/usr/bin/env python3
"""
AI ISP Simulator Advanced Demo Script
Demonstrates complete pipeline and multiple nodes
"""

import sys
import time
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.graph import Graph
from core.frame import Frame, ColorFormat, BayerPattern
from nodes.input.raw_input import RawInputNode
from nodes.raw_processing.demosaic import DemosaicNode
from nodes.raw_processing.raw_preproc import RawPreprocNode
from nodes.rgb_processing.awb import AWBNode


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_advanced_pipeline() -> Graph:
    """Create advanced pipeline"""
    print("🔧 Creating advanced pipeline...")
    
    # Create Graph
    graph = Graph("advanced_isp_pipeline")
    
    # Create nodes
    print("  Creating RAW input node...")
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
    
    print("  Creating RAW preprocessing node...")
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
    
    print("  Creating demosaic node...")
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
    
    print("  Creating white balance node...")
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
    
    # Add nodes to Graph
    print("  Adding nodes to Graph...")
    graph.add_node(raw_input)
    graph.add_node(raw_preproc)
    graph.add_node(demosaic)
    graph.add_node(awb)
    
    # Connect nodes
    print("  Connecting nodes...")
    graph.connect_nodes("raw_input", "raw_preproc")
    graph.connect_nodes("raw_preproc", "demosaic")
    graph.connect_nodes("demosaic", "awb")
    
    return graph


def run_pipeline_test(graph: Graph, test_name: str, num_frames: int = 1):
    """Run pipeline test"""
    print(f"\n🧪 Running test: {test_name}")
    print("=" * 50)
    
    # Validate Graph
    if not graph.validate():
        print("❌ Graph validation failed")
        return False
    
    print("✅ Graph validation passed")
    
    # Get RAW input node
    raw_input = graph.get_node("raw_input")
    if not raw_input:
        print("❌ RAW input node not found")
        return False
    
    # Generate test frames
    print(f"📸 Generating {num_frames} test frames...")
    frames = []
    for i in range(num_frames):
        frame = raw_input.generate_frame()
        frames.append(frame)
        print(f"  Frame {i+1}: {frame.shape}, ISO={frame.exposure_params.iso}, "
              f"exposure_time={frame.exposure_params.exposure_time:.3f}s")
    
    # Execute pipeline
    print("🔄 Executing pipeline...")
    start_time = time.time()
    
    processed_frames = []
    for i, frame in enumerate(frames):
        frame_start_time = time.time()
        
        try:
            outputs = graph.execute({"raw_input": frame})
            frame_processing_time = time.time() - frame_start_time
            
            # Get processed frames
            for output_name, output_data in outputs.items():
                if isinstance(output_data, Frame):
                    processed_frames.append(output_data)
                    print(f"  ✅ Frame {i+1} processing completed, time taken: {frame_processing_time:.3f}s")
                    print(f"     Output {output_name}: {output_data.shape}, "
                          f"format={output_data.color_format.value}")
                    break
            
        except Exception as e:
            print(f"  ❌ Frame {i+1} processing failed: {e}")
            logging.error(f"Frame {i+1} processing failed: {e}", exc_info=True)
            return False
    
    total_time = time.time() - start_time
    print(f"✅ Pipeline execution completed, total time: {total_time:.3f}s")
    
    return True


def run_performance_analysis(graph: Graph):
    """Run performance analysis"""
    print(f"\n📊 Performance Analysis")
    print("=" * 50)
    
    # Graph performance statistics
    graph_stats = graph.get_performance_stats()
    print("Graph Performance Statistics:")
    print(f"  Total executions: {graph_stats['total_executions']}")
    print(f"  Average execution time: {graph_stats['avg_execution_time']:.3f}s")
    print(f"  Minimum execution time: {graph_stats['min_execution_time']:.3f}s")
    print(f"  Maximum execution time: {graph_stats['max_execution_time']:.3f}s")
    if 'std_execution_time' in graph_stats:
        print(f"  Standard deviation: {graph_stats['std_execution_time']:.3f}s")
    
    # Node performance statistics
    print("\nNode Performance Statistics:")
    for node_id, node in graph.nodes.items():
        node_stats = node.get_performance_stats()
        print(f"  {node_id}:")
        print(f"    Processing count: {node_stats['total_processed']}")
        print(f"    Average time: {node_stats['avg_processing_time']:.3f}s")
        print(f"    Minimum time: {node_stats['min_processing_time']:.3f}s")
        print(f"    Maximum time: {node_stats['max_processing_time']:.3f}s")
        if 'std_processing_time' in node_stats:
            print(f"    Standard deviation: {node_stats['std_processing_time']:.3f}s")


def run_implementation_comparison(graph: Graph):
    """Run implementation comparison"""
    print(f"\n🔄 Implementation Comparison")
    print("=" * 50)
    
    # Get demosaic node
    demosaic_node = graph.get_node("demosaic")
    if not demosaic_node:
        print("❌ Demosaic node not found")
        return
    
    # Generate test frame
    raw_input = graph.get_node("raw_input")
    test_frame = raw_input.generate_frame()
    
    # Test classic implementation
    print("Testing classic implementation...")
    demosaic_node.set_implementation("classic")
    start_time = time.time()
    
    try:
        outputs = graph.execute({"raw_input": test_frame})
        classic_time = time.time() - start_time
        print(f"  ✅ Classic implementation completed, time taken: {classic_time:.3f}s")
        
        # Get output
        classic_output = None
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                classic_output = output_data
                break
        
    except Exception as e:
        print(f"  ❌ Classic implementation failed: {e}")
        return
    
    # Test AI implementation (if available)
    print("Testing AI implementation...")
    demosaic_node.set_implementation("ai")
    start_time = time.time()
    
    try:
        outputs = graph.execute({"raw_input": test_frame})
        ai_time = time.time() - start_time
        print(f"  ✅ AI implementation completed, time taken: {ai_time:.3f}s")
        
        # Get output
        ai_output = None
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                ai_output = output_data
                break
        
    except Exception as e:
        print(f"  ❌ AI implementation failed: {e}")
        ai_time = float('inf')
        ai_output = None
    
    # Comparison results
    print("\nComparison Results:")
    if classic_output and ai_output:
        print(f"  Classic implementation: {classic_output.shape}, time: {classic_time:.3f}s")
        print(f"  AI implementation: {ai_output.shape}, time: {ai_time:.3f}s")
        
        if ai_time < float('inf'):
            speedup = classic_time / ai_time
            print(f"  Speedup: {speedup:.2f}x")
        else:
            print("  AI implementation not available")
    else:
        print("  Unable to get outputs for comparison")


def main():
    """Main function"""
    setup_logging()
    
    try:
        print("🚀 AI ISP Simulator Advanced Demo")
        print("=" * 60)
        
        # Create pipeline
        graph = create_advanced_pipeline()
        
        # Run basic test
        if not run_pipeline_test(graph, "Basic Functionality Test", 3):
            print("❌ Basic test failed")
            return
        
        # Run performance analysis
        run_performance_analysis(graph)
        
        # Run implementation comparison
        run_implementation_comparison(graph)
        
        # Cleanup
        graph.reset()
        print("\n🧹 Cleanup completed")
        
        print("\n🎉 Advanced demo completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted")
    except Exception as e:
        print(f"\n💥 Program exception: {e}")
        logging.error(f"Program exception: {e}", exc_info=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
AI ISP Simulator Demo Script
Quick test of system functionality
"""

import sys
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.graph import Graph
from core.frame import Frame, ColorFormat, BayerPattern
from nodes.input.raw_input import RawInputNode
from nodes.raw_processing.demosaic import DemosaicNode


def run_simple_demo():
    """Run simple demo"""
    print("🚀 AI ISP Simulator Demo")
    print("=" * 50)
    
    # Create Graph
    print("1. Creating Graph...")
    graph = Graph("demo_pipeline")
    
    # Create nodes
    print("2. Creating nodes...")
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
    
    # Add nodes
    graph.add_node(raw_input)
    graph.add_node(demosaic)
    
    # Connect nodes
    print("3. Connecting nodes...")
    graph.connect_nodes("raw_input", "demosaic")
    
    # Validate Graph
    print("4. Validating Graph...")
    if not graph.validate():
        print("❌ Graph validation failed")
        return
    
    print("✅ Graph validation passed")
    
    # Generate test frame
    print("5. Generating test frame...")
    frame = raw_input.generate_frame()
    print(f"   Generated frame: {frame.shape}, format: {frame.color_format.value}")
    
    # Execute pipeline
    print("6. Executing pipeline...")
    start_time = time.time()
    
    try:
        outputs = graph.execute({"raw_input": frame})
        execution_time = time.time() - start_time
        
        print(f"✅ Pipeline execution successful, time taken: {execution_time:.3f}s")
        
        # Display outputs
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                print(f"   Output {output_name}: {output_data.shape}, "
                      f"format: {output_data.color_format.value}")
        
        # Performance statistics
        stats = graph.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"   Total executions: {stats['total_executions']}")
        print(f"   Average execution time: {stats['avg_execution_time']:.3f}s")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    try:
        run_simple_demo()
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted")
    except Exception as e:
        print(f"\n💥 Program exception: {e}")
        import traceback
        traceback.print_exc()

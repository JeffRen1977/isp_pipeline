#!/usr/bin/env python3
"""
Photo Mode Example
Demonstrates how to use AI ISP simulator for photo processing
"""

import sys
import os
import time
import logging
import yaml
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.graph import Graph
from core.frame import Frame, ColorFormat, BayerPattern
from nodes.input.raw_input import RawInputNode
from nodes.raw_processing.demosaic import DemosaicNode


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_pipeline_config(config_path: str) -> dict:
    """Load pipeline configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_photo_pipeline() -> Graph:
    """Create photo pipeline"""
    # Create Graph
    graph = Graph("photo_pipeline")
    
    # Create nodes
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
    
    # Add nodes to Graph
    graph.add_node(raw_input)
    graph.add_node(demosaic)
    
    # Connect nodes
    graph.connect_nodes("raw_input", "demosaic")
    
    return graph


def run_photo_pipeline():
    """Run photo pipeline"""
    print("=== AI ISP Simulator - Photo Mode Example ===")
    
    # Create pipeline
    graph = create_photo_pipeline()
    
    # Validate Graph
    if not graph.validate():
        print("❌ Graph validation failed")
        return
    
    print("✅ Graph validation passed")
    
    # Generate HDR burst
    print("\n📸 Generating HDR burst sequence...")
    raw_input = graph.get_node("raw_input")
    
    frames = []
    for i in range(3):
        frame = raw_input.generate_frame()
        frames.append(frame)
        print(f"  Generated frame {i+1}: {frame.shape}, ISO={frame.exposure_params.iso}, "
              f"exposure_time={frame.exposure_params.exposure_time:.3f}s")
    
    # Execute pipeline
    print("\n🔄 Executing pipeline...")
    start_time = time.time()
    
    try:
        # Process first frame
        outputs = graph.execute({"raw_input": frames[0]})
        execution_time = time.time() - start_time
        
        print(f"✅ Pipeline execution successful, time taken: {execution_time:.3f}s")
        
        # Display results
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                print(f"  Output {output_name}: {output_data.shape}, "
                      f"format={output_data.color_format.value}")
            else:
                print(f"  Output {output_name}: {type(output_data)}")
        
        # Performance statistics
        stats = graph.get_performance_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"  Total executions: {stats['total_executions']}")
        print(f"  Average execution time: {stats['avg_execution_time']:.3f}s")
        print(f"  Minimum execution time: {stats['min_execution_time']:.3f}s")
        print(f"  Maximum execution time: {stats['max_execution_time']:.3f}s")
        
        # Node performance statistics
        print(f"\n🔧 Node Performance Statistics:")
        for node_id, node in graph.nodes.items():
            node_stats = node.get_performance_stats()
            print(f"  {node_id}: processed {node_stats['total_processed']} times, "
                  f"average time {node_stats['avg_processing_time']:.3f}s")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
    
    # Cleanup
    graph.reset()
    print("\n🧹 Cleanup completed")


def run_hdr_photo():
    """Run HDR photo"""
    print("\n=== HDR Photo Example ===")
    
    # Create pipeline
    graph = create_photo_pipeline()
    
    if not graph.validate():
        print("❌ Graph validation failed")
        return
    
    # Generate HDR burst
    raw_input = graph.get_node("raw_input")
    frames = raw_input.generate_hdr_burst(3)
    
    print(f"📸 Generated {len(frames)} frame HDR burst")
    
    # Process each frame
    processed_frames = []
    for i, frame in enumerate(frames):
        print(f"\nProcessing frame {i+1}...")
        start_time = time.time()
        
        try:
            outputs = graph.execute({"raw_input": frame})
            processing_time = time.time() - start_time
            
            # Get processed frames
            for output_name, output_data in outputs.items():
                if isinstance(output_data, Frame):
                    processed_frames.append(output_data)
                    print(f"  ✅ Frame {i+1} processing completed, time taken: {processing_time:.3f}s")
                    print(f"     Output size: {output_data.shape}")
                    print(f"     Output format: {output_data.color_format.value}")
                    break
            
        except Exception as e:
            print(f"  ❌ Frame {i+1} processing failed: {e}")
    
    print(f"\n🎯 HDR processing completed, processed {len(processed_frames)} frames")
    
    # Cleanup
    graph.reset()


def main():
    """Main function"""
    setup_logging()
    
    try:
        # Run basic photo pipeline
        run_photo_pipeline()
        
        # Run HDR photo
        run_hdr_photo()
        
        print("\n🎉 All examples completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted")
    except Exception as e:
        print(f"\n💥 Program exception: {e}")
        logging.error(f"Program exception: {e}", exc_info=True)


if __name__ == "__main__":
    main()

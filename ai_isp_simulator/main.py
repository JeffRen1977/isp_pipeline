#!/usr/bin/env python3
"""
AI ISP Simulator Main Program
Supports three modes: photo, video, and preview
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.graph import Graph
from core.frame import Frame, ColorFormat, BayerPattern
from nodes.input.raw_input import RawInputNode
from nodes.raw_processing.demosaic import DemosaicNode


def setup_logging(level: str = "INFO"):
    """Setup logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        return {}


def create_pipeline_from_config(config: Dict[str, Any]) -> Graph:
    """Create pipeline from configuration"""
    pipeline_config = config.get("pipeline", {})
    pipeline_name = pipeline_config.get("name", "default_pipeline")
    
    # Create Graph
    graph = Graph(pipeline_name)
    
    # Create nodes
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
            logging.warning(f"Unknown node type: {node_type}")
            continue
        
        nodes[node_id] = node
        graph.add_node(node)
    
    # Connect nodes
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
                    logging.warning(f"Failed to connect nodes: {from_node_id} -> {to_node_id}")
    
    return graph


def run_photo_mode(config_path: str):
    """Run photo mode"""
    print("=== Photo Mode ===")
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        print("❌ Configuration loading failed")
        return
    
    # Create pipeline
    graph = create_pipeline_from_config(config)
    
    # Validate Graph
    if not graph.validate():
        print("❌ Graph validation failed")
        return
    
    print("✅ Graph validation passed")
    
    # Get RAW input node
    raw_input = graph.get_node("raw_input")
    if not raw_input:
        print("❌ RAW input node not found")
        return
    
    # Generate test frame
    print("📸 Generating test frame...")
    frame = raw_input.generate_frame()
    print(f"  Generated frame: {frame.shape}, ISO={frame.exposure_params.iso}")
    
    # Execute pipeline
    print("🔄 Executing pipeline...")
    try:
        outputs = graph.execute({"raw_input": frame})
        print("✅ Pipeline execution successful")
        
        # Display outputs
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                print(f"  Output {output_name}: {output_data.shape}, "
                      f"format={output_data.color_format.value}")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)


def run_video_mode(config_path: str):
    """Run video mode"""
    print("=== Video Mode ===")
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        print("❌ Configuration loading failed")
        return
    
    # Create pipeline
    graph = create_pipeline_from_config(config)
    
    # Validate Graph
    if not graph.validate():
        print("❌ Graph validation failed")
        return
    
    print("✅ Graph validation passed")
    
    # Get RAW input node
    raw_input = graph.get_node("raw_input")
    if not raw_input:
        print("❌ RAW input node not found")
        return
    
    # Generate multi-frame test
    print("📹 Generating video frame sequence...")
    num_frames = 10
    
    for i in range(num_frames):
        frame = raw_input.generate_frame()
        print(f"  Generated frame {i+1}: {frame.shape}")
        
        # Execute pipeline
        try:
            outputs = graph.execute({"raw_input": frame})
            print(f"  ✅ Frame {i+1} processing completed")
            
        except Exception as e:
            print(f"  ❌ Frame {i+1} processing failed: {e}")
            break
    
    print(f"🎬 Video processing completed, processed {num_frames} frames")


def run_preview_mode(config_path: str):
    """Run preview mode"""
    print("=== Preview Mode ===")
    
    # Load configuration
    config = load_config(config_path)
    if not config:
        print("❌ Configuration loading failed")
        return
    
    # Create pipeline
    graph = create_pipeline_from_config(config)
    
    # Validate Graph
    if not graph.validate():
        print("❌ Graph validation failed")
        return
    
    print("✅ Graph validation passed")
    
    # Get RAW input node
    raw_input = graph.get_node("raw_input")
    if not raw_input:
        print("❌ RAW input node not found")
        return
    
    # Generate preview frame
    print("👁️ Generating preview frame...")
    frame = raw_input.generate_frame()
    print(f"  Generated preview frame: {frame.shape}")
    
    # Execute pipeline
    print("🔄 Executing preview pipeline...")
    try:
        outputs = graph.execute({"raw_input": frame})
        print("✅ Preview pipeline execution successful")
        
        # Display outputs
        for output_name, output_data in outputs.items():
            if isinstance(output_data, Frame):
                print(f"  Preview output {output_name}: {output_data.shape}, "
                      f"format={output_data.color_format.value}")
        
    except Exception as e:
        print(f"❌ Preview pipeline execution failed: {e}")
        logging.error(f"Preview pipeline execution failed: {e}", exc_info=True)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI ISP Simulator")
    parser.add_argument(
        "mode",
        choices=["photo", "video", "preview"],
        help="Run mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Determine configuration file path
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
    
    # Check if configuration file exists
    if not Path(config_path).exists():
        print(f"❌ Configuration file does not exist: {config_path}")
        return
    
    print(f"📁 Using configuration file: {config_path}")
    
    try:
        # Run according to mode
        if args.mode == "photo":
            run_photo_mode(config_path)
        elif args.mode == "video":
            run_video_mode(config_path)
        elif args.mode == "preview":
            run_preview_mode(config_path)
        
        print("\n🎉 Execution completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ User interrupted")
    except Exception as e:
        print(f"\n💥 Program exception: {e}")
        logging.error(f"Program exception: {e}", exc_info=True)


if __name__ == "__main__":
    # Add missing import
    from core.node import ImplementationType
    
    main()

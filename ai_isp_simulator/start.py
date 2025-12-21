#!/usr/bin/env python3
"""
AI ISP Simulator Launcher Script
Provides interactive menu to select different functions
"""

import sys
import os
import subprocess
from pathlib import Path


def print_banner():
    """Print project banner"""
    print("=" * 60)
    print("🚀 AI ISP Simulator")
    print("Graph-based AI ISP (Image Signal Processor) Simulator")
    print("Supports various computational photography features")
    print("=" * 60)
    print()


def print_menu():
    """Print main menu"""
    print("Please select a function to run:")
    print("1. Basic Demo - Simple functionality test")
    print("2. Advanced Demo - Complete pipeline test")
    print("3. Photo Mode - Photo pipeline demo")
    print("4. Video Mode - Video pipeline demo")
    print("5. Preview Mode - Preview pipeline demo")
    print("6. Run Tests - Execute unit tests")
    print("7. Install Dependencies - Install project dependencies")
    print("8. Show Help - Display usage instructions")
    print("0. Exit")
    print()


def run_basic_demo():
    """Run basic demo"""
    print("🎯 Running basic demo...")
    script_path = Path(__file__).parent / "run_demo.py"
    
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Basic demo failed: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted")
    else:
        print("❌ Basic demo script not found")


def run_advanced_demo():
    """Run advanced demo"""
    print("🎯 Running advanced demo...")
    script_path = Path(__file__).parent / "run_advanced_demo.py"
    
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Advanced demo failed: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted")
    else:
        print("❌ Advanced demo script not found")


def run_photo_mode():
    """Run photo mode"""
    print("🎯 Running photo mode...")
    script_path = Path(__file__).parent / "examples" / "photo_mode.py"
    
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Photo mode failed: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted")
    else:
        print("❌ Photo mode script not found")


def run_video_mode():
    """Run video mode"""
    print("🎯 Running video mode...")
    script_path = Path(__file__).parent / "main.py"
    
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path), "video"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Video mode failed: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted")
    else:
        print("❌ Main program script not found")


def run_preview_mode():
    """Run preview mode"""
    print("🎯 Running preview mode...")
    script_path = Path(__file__).parent / "main.py"
    
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path), "preview"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Preview mode failed: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted")
    else:
        print("❌ Main program script not found")


def run_tests():
    """Run tests"""
    print("🎯 Running tests...")
    test_dir = Path(__file__).parent / "tests"
    
    if test_dir.exists():
        try:
            # Run all tests
            subprocess.run([
                sys.executable, "-m", "pytest", str(test_dir), "-v"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Test execution failed: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted")
    else:
        print("❌ Test directory not found")


def install_dependencies():
    """Install dependencies"""
    print("🎯 Installing project dependencies...")
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    if requirements_path.exists():
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
            ], check=True)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Dependency installation failed: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted")
    else:
        print("❌ requirements.txt file not found")


def show_help():
    """Show help information"""
    print("📖 AI ISP Simulator Usage Instructions")
    print("=" * 50)
    print()
    print("Project Structure:")
    print("  ai_isp_simulator/")
    print("  ├── src/                    # Source code")
    print("  │   ├── core/              # Core modules")
    print("  │   ├── nodes/             # ISP node implementations")
    print("  │   └── quality/           # Quality analysis module")
    print("  ├── configs/               # Configuration files")
    print("  ├── examples/              # Usage examples")
    print("  ├── tests/                 # Test code")
    print("  └── main.py               # Main program")
    print()
    print("Main Features:")
    print("  • Supports photo, video, and preview modes")
    print("  • Graph-based modular design")
    print("  • Supports AI/traditional algorithm switching")
    print("  • Built-in image quality analysis")
    print("  • Supports RAW data input")
    print()
    print("Usage:")
    print("  1. Select a function from the menu to run")
    print("  2. Or run the corresponding script directly:")
    print("     python run_demo.py              # Basic demo")
    print("     python run_advanced_demo.py     # Advanced demo")
    print("     python main.py photo            # Photo mode")
    print("     python main.py video            # Video mode")
    print("     python main.py preview          # Preview mode")
    print()
    print("Configuration Files:")
    print("  Configuration files are located in configs/pipelines/ directory")
    print("  You can modify node parameters and connections as needed")
    print()


def main():
    """Main function"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("Please enter your choice (0-8): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                run_basic_demo()
            elif choice == "2":
                run_advanced_demo()
            elif choice == "3":
                run_photo_mode()
            elif choice == "4":
                run_video_mode()
            elif choice == "5":
                run_preview_mode()
            elif choice == "6":
                run_tests()
            elif choice == "7":
                install_dependencies()
            elif choice == "8":
                show_help()
            else:
                print("❌ Invalid choice, please enter a number between 0-8")
            
            print()
            input("Press Enter to continue...")
            print()
            
        except KeyboardInterrupt:
            print("\n⏹️ User interrupted")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n💥 Program exception: {e}")
            break


if __name__ == "__main__":
    main()

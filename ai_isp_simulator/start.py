#!/usr/bin/env python3
"""
AI ISP 仿真器启动脚本
提供交互式菜单选择不同的功能
"""

import sys
import os
import subprocess
from pathlib import Path


def print_banner():
    """打印项目横幅"""
    print("=" * 60)
    print("🚀 AI ISP 仿真器")
    print("基于Graph的AI ISP（图像信号处理器）仿真器")
    print("支持computational photography的各种功能")
    print("=" * 60)
    print()


def print_menu():
    """打印主菜单"""
    print("请选择要运行的功能:")
    print("1. 基础演示 - 简单功能测试")
    print("2. 高级演示 - 完整pipeline测试")
    print("3. 拍照模式 - 拍照pipeline演示")
    print("4. 视频模式 - 视频pipeline演示")
    print("5. 预览模式 - 预览pipeline演示")
    print("6. 运行测试 - 执行单元测试")
    print("7. 安装依赖 - 安装项目依赖")
    print("8. 查看帮助 - 显示使用说明")
    print("0. 退出")
    print()


def run_basic_demo():
    """运行基础演示"""
    print("🎯 运行基础演示...")
    script_path = Path(__file__).parent / "run_demo.py"
    
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 基础演示运行失败: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
    else:
        print("❌ 基础演示脚本不存在")


def run_advanced_demo():
    """运行高级演示"""
    print("🎯 运行高级演示...")
    script_path = Path(__file__).parent / "run_advanced_demo.py"
    
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 高级演示运行失败: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
    else:
        print("❌ 高级演示脚本不存在")


def run_photo_mode():
    """运行拍照模式"""
    print("🎯 运行拍照模式...")
    script_path = Path(__file__).parent / "examples" / "photo_mode.py"
    
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 拍照模式运行失败: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
    else:
        print("❌ 拍照模式脚本不存在")


def run_video_mode():
    """运行视频模式"""
    print("🎯 运行视频模式...")
    script_path = Path(__file__).parent / "main.py"
    
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path), "video"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 视频模式运行失败: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
    else:
        print("❌ 主程序脚本不存在")


def run_preview_mode():
    """运行预览模式"""
    print("🎯 运行预览模式...")
    script_path = Path(__file__).parent / "main.py"
    
    if script_path.exists():
        try:
            subprocess.run([sys.executable, str(script_path), "preview"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 预览模式运行失败: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
    else:
        print("❌ 主程序脚本不存在")


def run_tests():
    """运行测试"""
    print("🎯 运行测试...")
    test_dir = Path(__file__).parent / "tests"
    
    if test_dir.exists():
        try:
            # 运行所有测试
            subprocess.run([
                sys.executable, "-m", "pytest", str(test_dir), "-v"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 测试运行失败: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
    else:
        print("❌ 测试目录不存在")


def install_dependencies():
    """安装依赖"""
    print("🎯 安装项目依赖...")
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    if requirements_path.exists():
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
            ], check=True)
            print("✅ 依赖安装完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
    else:
        print("❌ requirements.txt文件不存在")


def show_help():
    """显示帮助信息"""
    print("📖 AI ISP 仿真器使用说明")
    print("=" * 50)
    print()
    print("项目结构:")
    print("  ai_isp_simulator/")
    print("  ├── src/                    # 源代码")
    print("  │   ├── core/              # 核心模块")
    print("  │   ├── nodes/             # ISP节点实现")
    print("  │   └── quality/           # 质量分析模块")
    print("  ├── configs/               # 配置文件")
    print("  ├── examples/              # 使用示例")
    print("  ├── tests/                 # 测试代码")
    print("  └── main.py               # 主程序")
    print()
    print("主要功能:")
    print("  • 支持拍照、视频、预览三种模式")
    print("  • 基于Graph的模块化设计")
    print("  • 支持AI/传统算法切换")
    print("  • 内置图像质量分析")
    print("  • 支持RAW数据输入")
    print()
    print("使用方法:")
    print("  1. 选择菜单中的功能运行")
    print("  2. 或直接运行相应的脚本:")
    print("     python run_demo.py              # 基础演示")
    print("     python run_advanced_demo.py     # 高级演示")
    print("     python main.py photo            # 拍照模式")
    print("     python main.py video            # 视频模式")
    print("     python main.py preview          # 预览模式")
    print()
    print("配置文件:")
    print("  配置文件位于 configs/pipelines/ 目录下")
    print("  可以根据需要修改节点参数和连接关系")
    print()


def main():
    """主函数"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("请输入选择 (0-8): ").strip()
            
            if choice == "0":
                print("👋 再见！")
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
                print("❌ 无效选择，请输入0-8之间的数字")
            
            print()
            input("按回车键继续...")
            print()
            
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
            break
        except EOFError:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n💥 程序异常: {e}")
            break


if __name__ == "__main__":
    main()

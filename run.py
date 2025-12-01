"""
炼金术实验运行脚本
功能：提供便捷的命令行接口来运行不同的实验阶段
作者：CraigV Wang
版本：1.1
更新：适配新的配置结构，支持阶段选择和参数覆盖
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """运行命令并处理输出"""
    print(f"🚀 {description}")
    print(f"💻 执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"错误输出: {e.stderr}")
        return False


def build_hydra_command(args, extra_params=None):
    """构建Hydra命令"""
    command_parts = ["python main.py"]
    
    # 添加配置名称
    command_parts.append(f'--config-name="{args.config}"')
    
    # 添加配置路径
    command_parts.append('--config-path="./config"')
    
    # 添加运行模式
    if args.mode == "preprocess":
        command_parts.append('mode="preprocess_only"')
    elif args.mode == "full":
        command_parts.append('mode="full"')
    elif args.mode == "test":
        command_parts.append('mode="test_single"')
        command_parts.append('test_single=true')
    
    # 添加文件格式选择
    if args.formats:
        formats_list = "[" + ",".join([f'"{f}"' for f in args.formats]) + "]"
        command_parts.append(f'selected_formats={formats_list}')
    
    # 添加GPU设置
    if not args.gpu:
        command_parts.append('preparation.platform.use_cuda=false')
        command_parts.append('alchemical.platform.name="CPU"')
    else:
        # 指定GPU设备
        if args.gpu_device:
            command_parts.append(f'preparation.platform.device_index="{args.gpu_device}"')
    
    # 添加lambda调度选择（如果指定）
    if args.lambda_schedule:
        if args.lambda_schedule == "conservative":
            command_parts.append('alchemical.lambda_schedule.segments=[[1.0,0.95,8],[0.95,0.8,12],[0.8,0.5,12],[0.5,0.2,12],[0.2,0.0,12]]')
        elif args.lambda_schedule == "simple":
            command_parts.append('alchemical.lambda_schedule=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]')
    
    # 添加测试模式参数（如果指定）
    if args.test_iterations:
        command_parts.append(f'alchemical.total_iterations={args.test_iterations}')
        command_parts.append(f'alchemical.steps_per_iteration={args.test_steps}')
    
    # 添加额外参数
    if extra_params:
        for param in extra_params:
            command_parts.append(param)
    
    # 添加输出目录设置（如果指定）
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        command_parts.append(f'output.base_dir="{args.output_dir}"')
    
    # 添加实验名称（如果指定）
    if args.experiment_name:
        command_parts.append(f'experiment.name="{args.experiment_name}"')
    
    # 合并命令
    command = " ".join(command_parts)
    return command


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FEP实验运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只运行预处理
  python run.py preprocess
  
  # 运行完整流程
  python run.py full
  
  # 运行单个样本测试
  python run.py test
  
  # 只处理PDB和SDF格式
  python run.py preprocess --formats pdb sdf
  
  # 使用GPU运行完整流程
  python run.py full --gpu --gpu-device 0
  
  # 运行测试模式，减少迭代次数
  python run.py test --test-iterations 10 --test-steps 100
  
  # 指定输出目录和实验名称
  python run.py full --output-dir ./my_experiment --experiment-name "My FEP Experiment"
  
  # 使用简单的lambda调度
  python run.py full --lambda-schedule simple
        """
    )
    
    # 运行模式
    parser.add_argument(
        "mode", 
        choices=["preprocess", "full", "test"],
        help="运行模式: preprocess(只预处理), full(完整流程), test(单个样本测试)"
    )
    
    # 可选参数
    parser.add_argument(
        "--formats", 
        nargs="+", 
        choices=["pdb", "cif", "sdf", "mol2", "xyz"],
        help="指定处理的文件格式"
    )
    
    parser.add_argument(
        "--config", 
        default="base",
        help="使用的配置文件 (默认: base)"
    )
    
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help="使用GPU加速"
    )
    
    parser.add_argument(
        "--gpu-device",
        type=str,
        default="0",
        help="GPU设备索引 (默认: 0)"
    )
    
    parser.add_argument(
        "--lambda-schedule",
        choices=["conservative", "simple"],
        help="lambda调度策略: conservative(保守, 56个状态), simple(简单, 11个状态)"
    )
    
    parser.add_argument(
        "--test-iterations",
        type=int,
        help="测试模式的迭代次数 (覆盖配置)"
    )
    
    parser.add_argument(
        "--test-steps",
        type=int,
        default=50,
        help="测试模式每迭代步数 (默认: 50)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录路径 (覆盖配置中的output.base_dir)"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="实验名称 (覆盖配置中的experiment.name)"
    )
    
    parser.add_argument(
        "--override",
        nargs="+",
        help="直接覆盖配置参数，格式: key=value"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 FEP实验运行脚本")
    print("=" * 60)
    
    # 显示运行配置
    print(f"📋 运行配置:")
    print(f"  模式: {args.mode}")
    print(f"  配置文件: {args.config}")
    if args.formats:
        print(f"  文件格式: {', '.join(args.formats)}")
    print(f"  GPU加速: {'是' if args.gpu else '否'}")
    if args.gpu:
        print(f"  GPU设备: {args.gpu_device}")
    if args.lambda_schedule:
        print(f"  Lambda调度: {args.lambda_schedule}")
    if args.output_dir:
        print(f"  输出目录: {args.output_dir}")
    if args.experiment_name:
        print(f"  实验名称: {args.experiment_name}")
    print("=" * 60)
    
    # 检查配置文件是否存在
    config_file = Path(f"./config/{args.config}.yaml")
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return 1
    
    # 构建命令
    extra_params = []
    
    # 添加覆盖参数
    if args.override:
        for override in args.override:
            extra_params.append(override)
    
    # 添加详细输出
    if args.verbose:
        extra_params.append('hydra.verbose=true')
    
    command = build_hydra_command(args, extra_params)
    
    # 运行命令
    success = run_command(command, f"运行{args.mode}模式")
    
    if success:
        print("\n🎉 实验运行完成!")
        return 0
    else:
        print("\n❌ 实验运行失败!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
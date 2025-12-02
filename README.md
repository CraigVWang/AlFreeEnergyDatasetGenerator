# AlFreeEnergyDatasetGenerator

# 1. 只运行预处理，处理PDB和SDF格式
python run.py preprocess --formats pdb sdf

# 2. 运行完整流程，使用GPU，指定输出目录
python run.py full --gpu --gpu-device 0 --output-dir ./my_experiment

# 3. 运行单个样本测试，使用保守的lambda调度
python run.py test --lambda-schedule conservative

# 4. 运行测试模式，减少迭代次数
python run.py test --test-iterations 10 --test-steps 100

# 5. 直接覆盖配置参数
python run.py full --override "alchemical.total_iterations=100" "alchemical.steps_per_iteration=500"

# 6. 完整示例：指定所有参数
python run.py full \
  --formats pdb sdf xyz \
  --gpu \
  --gpu-device 0 \
  --lambda-schedule conservative \
  --output-dir ./results \
  --experiment-name "My_FEP_Experiment" \
  --override "preparation.solvent.box_size=10.0" \
  --verbose
  
## 项目结构
AlFreeEnergyDatasetGenerator/
├── main.py                      # 主程序入口
├── preprocessor.py              # 预处理模块
├── system_provider.py           # 系统准备模块
├── alchemist.py                 # 炼金术自由能模拟模块
├── analyzer.py                  # 分析器模块
├── config/
│   ├── base.yaml                # 基础配置文件（所有配置继承此文件）
│   ├── preprocessor.yaml        # 测试配置（继承base）
│   ├── preparation.yaml         # 系统准备配置（继承base）
│   ├── alchemical.yaml          # 炼金术模拟配置（继承base，使用lambda_schedule）
│   └── analysis.yaml            # 分析配置（继承base）
├── requirements.txt             # 依赖包列表
├── README.md                    # 项目说明文档
├── .gitignore                   # Git忽略文件
├── run_experiment.py            # 实验运行脚本
├── dataset/                     # 数据目录（运行时生成）
│   ├── raw/                     # 原始数据存放位置
│   ├── preprocessed/            # 处理后的文件存放位置（保持原始目录结构）
│   ├── prepared_systems/        # 准备好的系统（保持原始目录结构）
│   ├── alchemical_results/      # 炼金术模拟结果（hash值命名文件）
│   ├── analysis_results/        # 分析结果（按分子名创建子目录）
│   └── metadata.csv             # 处理过程元数据
├── outputs/                     # Hydra运行输出目录（自动生成）
├── logs/                        # 日志目录
└── utils/                       # 工具模块
    └── xyz_to_pdb_converter.py  # XYZ格式转换器
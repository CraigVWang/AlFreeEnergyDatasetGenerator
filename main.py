"""
炼金术数据生成实验主程序 - 简化版本
功能：执行完整的化学结构文件预处理和系统准备流程
作者：CraigV Wang
版本：1.0
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import csv
from pathlib import Path
import pandas as pd
from datetime import datetime

# 导入自定义模块
from preprocessor import StructurePreprocessor
from system_provider import SystemProvider
from alchemist import Alchemist
from analyzer import Analyzer


class AlchemicalDataGenerator:
    """
    炼金术数据生成实验类 - 简化版本
    负责整个自由能微扰数据生成流程的协调和执行
    """
    
    def __init__(self, config: DictConfig):
        """
        初始化实验类
        
        参数:
            config: Hydra配置对象，包含所有实验参数
        """
        self.config = config
        self.metadata_dir = Path(self.config.output.metadata_dir)
        self.metadata_file = self.metadata_dir / "metadata.csv"
        self.metadata = []
        
        # 各阶段处理器
        self.preprocessor = None
        self.system_provider = None
        self.alchemist = None
        self.analyzer = None
        
    def load_metadata(self) -> list:
        """
        加载元数据文件
        
        返回:
            元数据列表
        """
        if not self.metadata_file.exists():
            print(f"❌ 元数据文件不存在: {self.metadata_file}")
            return []
        
        try:
            with open(self.metadata_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.metadata = list(reader)
            print(f"📖 加载元数据: {len(self.metadata)} 个分子")
            return self.metadata
        except Exception as e:
            print(f"❌ 加载元数据失败: {e}")
            return []
    
    def save_metadata(self):
        """
        保存元数据到文件
        """
        if not self.metadata:
            return
            
        try:
            # 获取所有可能的列
            all_columns = set()
            for item in self.metadata:
                all_columns.update(item.keys())
            
            # 写入文件
            with open(self.metadata_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(all_columns))
                writer.writeheader()
                for row in self.metadata:
                    writer.writerow(row)
            
            print(f"💾 元数据已保存到: {self.metadata_file}")
        except Exception as e:
            print(f"❌ 保存元数据失败: {e}")
    
    def update_molecule_status(self, 
                              molecule_name: str,
                              stage: str,
                              success: bool = True,
                              additional_info: dict = None):
        """
        更新分子的状态信息
        
        参数:
            molecule_name: 分子名称
            stage: 阶段名称 ('preprocess', 'preparation', 'alchemical', 'analysis')
            success: 该阶段是否成功
            additional_info: 额外的信息字典
        """
        if not self.metadata:
            return
            
        # 查找分子
        for item in self.metadata:
            if item['name'] == molecule_name:
                # 更新时间戳
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 更新对应阶段的状态
                if stage == 'preprocess':
                    item['processed_successfully'] = str(success)
                    item['preprocess_timestamp'] = timestamp
                
                elif stage == 'preparation':
                    item['minimized_successfully'] = str(success)
                    item['preparation_timestamp'] = timestamp
                
                elif stage == 'alchemical':
                    item['alchemical_successfully'] = str(success)
                    item['alchemical_timestamp'] = timestamp
                
                elif stage == 'analysis':
                    item['analysis_successfully'] = str(success)
                    item['analysis_timestamp'] = timestamp
                    
                    # 计算finish_successfully
                    conditions = [
                        item.get('processed_successfully', 'False').lower() == 'true',
                        item.get('minimized_successfully', 'False').lower() == 'true',
                        item.get('alchemical_successfully', 'False').lower() == 'true',
                        success  # 当前的分析阶段是否成功
                    ]
                    finish_success = all(conditions)
                    item['finish_successfully'] = str(finish_success)
                
                # 更新额外信息
                if additional_info:
                    for key, value in additional_info.items():
                        item[key] = str(value) if value is not None else ''
                
                break
    
    def get_statistics(self) -> dict:
        """
        获取处理统计信息
        
        返回:
            统计信息字典
        """
        if not self.metadata:
            return {
                'total_molecules': 0,
                'processed_success': 0,
                'preparation_success': 0,
                'alchemical_success': 0,
                'analysis_success': 0,
                'finish_success': 0
            }
        
        stats = {
            'total_molecules': len(self.metadata),
            'processed_success': 0,
            'preparation_success': 0,
            'alchemical_success': 0,
            'analysis_success': 0,
            'finish_success': 0
        }
        
        for item in self.metadata:
            if item.get('processed_successfully', 'False').lower() == 'true':
                stats['processed_success'] += 1
            
            if item.get('minimized_successfully', 'False').lower() == 'true':
                stats['preparation_success'] += 1
            
            if item.get('alchemical_successfully', 'False').lower() == 'true':
                stats['alchemical_success'] += 1
            
            if item.get('analysis_successfully', 'False').lower() == 'true':
                stats['analysis_success'] += 1
            
            if item.get('finish_successfully', 'False').lower() == 'true':
                stats['finish_success'] += 1
        
        # 计算成功率
        for key in ['processed', 'preparation', 'alchemical', 'analysis', 'finish']:
            total_key = 'total_molecules'
            success_key = f'{key}_success'
            if success_key in stats and stats[total_key] > 0:
                stats[f'{key}_success_rate'] = stats[success_key] / stats[total_key]
        
        return stats
    
    def print_statistics(self):
        """打印处理统计信息"""
        stats = self.get_statistics()
        
        print("\n📊 处理统计信息:")
        print("=" * 40)
        print(f"总分子数: {stats['total_molecules']}")
        print(f"预处理成功: {stats['processed_success']} ({stats.get('processed_success_rate', 0)*100:.1f}%)")
        print(f"系统准备成功: {stats['preparation_success']} ({stats.get('preparation_success_rate', 0)*100:.1f}%)")
        print(f"炼金术成功: {stats['alchemical_success']} ({stats.get('alchemical_success_rate', 0)*100:.1f}%)")
        print(f"分析成功: {stats['analysis_success']} ({stats.get('analysis_success_rate', 0)*100:.1f}%)")
        print(f"完成全部流程: {stats['finish_success']} ({stats.get('finish_success_rate', 0)*100:.1f}%)")
        print("=" * 40)
    
    def preprocess_data(self, selected_formats: list = None, test_single: bool = False):
        """数据预处理步骤"""
        print("=" * 60)
        print("🚀 开始数据预处理阶段")
        if test_single:
            print("🧪 测试模式：只处理单个样本")
        print("=" * 60)
        
        if self.preprocessor is None:
            # 将OmegaConf转换为字典
            self.preprocessor = StructurePreprocessor(self.config, selected_formats)
        
        result = self.preprocessor.run(test_single=test_single)
        
        if not result['success']:
            raise Exception("❌ 数据预处理失败")
        
        # 加载处理后的元数据
        self.load_metadata()
        
        print("✅ 数据预处理阶段完成")
        return result
    
    def prepare_systems(self, test_single: bool = False):
        """系统准备步骤"""
        print("=" * 60)
        print("🔬 开始系统准备阶段")
        if test_single:
            print("🧪 测试模式：只处理单个样本")
        print("=" * 60)
        
        # 加载元数据
        self.load_metadata()
        
        if not self.metadata:
            raise Exception("❌ 没有可用的元数据，请先运行预处理阶段")
        
        if self.system_provider is None:
            self.system_provider = SystemProvider(self.config)
        
        # 运行系统准备
        result = self.system_provider.run_preparation(self.metadata, test_single=test_single)
        
        if not result['success']:
            raise Exception("❌ 系统准备失败")
        
        # 保存准备结果供后续阶段使用
        if 'preparation_results' in result:
            # 保存到pickle文件
            self.system_provider.save_preparation_results(result['preparation_results'])
            
            # 更新元数据
            for prep_result in result['preparation_results']:
                if prep_result['success']:
                    self.update_molecule_status(
                        prep_result['name'],
                        'preparation',
                        True,
                        {'prepared_system_path': prep_result['output_path']}
                    )
        
        # 保存更新后的元数据
        self.save_metadata()
        
        # 保存准备结果到实例变量，供炼金术阶段使用
        self.preparation_results = result.get('preparation_results', [])
        
        print("✅ 系统准备阶段完成")
        return result
    
    def run_alchemical_simulation(self, test_single: bool = False):
        """炼金术模拟步骤"""
        print("=" * 60)
        print("🧪 开始炼金术自由能模拟阶段")
        if test_single:
            print("🧪 测试模式：只处理单个样本")
        print("=" * 60)
        
        # 检查是否有准备结果
        if not hasattr(self, 'preparation_results') or not self.preparation_results:
            print("⚠️  没有找到内存中的准备结果，尝试从文件加载...")
            # 尝试从文件加载
            preparation_dir = Path(self.config.input.preparation_dir)
            results_file = preparation_dir / "preparation_results.pkl"
            
            if not results_file.exists():
                raise Exception(f"❌ 未找到准备结果文件: {results_file}")
            
            try:
                with open(results_file, 'rb') as f:
                    self.preparation_results = pickle.load(f)
                print(f"✅ 从文件加载了 {len(self.preparation_results)} 个准备结果")
            except Exception as e:
                raise Exception(f"❌ 加载准备结果失败: {e}")
        
        if not self.preparation_results:
            raise Exception("❌ 没有可用的准备结果")
        
        if self.alchemist is None:
            self.alchemist = Alchemist(self.config)
        
        # 修改炼金术师以使用准备好的系统
        result = self.alchemist.run_alchemical_batch(self.preparation_results, test_single=test_single)
        
        if not result['success']:
            raise Exception("❌ 炼金术模拟失败")
        
        # 保存更新后的元数据
        self.save_metadata()
        
        print("✅ 炼金术模拟阶段完成")
        return result
    
    def analyze_results(self, test_single: bool = False):
        """结果分析步骤"""
        print("=" * 60)
        print("📊 开始结果分析阶段")
        if test_single:
            print("🧪 测试模式：只分析单个样本")
        print("=" * 60)
        
        # 加载元数据
        self.load_metadata()
        
        if not self.metadata:
            raise Exception("❌ 没有可用的元数据")
        
        if self.analyzer is None:
            self.analyzer = Analyzer(self.config)
        
        # 修改分析器以使用统一的元数据
        result = self.analyzer.run_analysis_batch(self.metadata, test_single=test_single)
        
        if not result['success']:
            raise Exception("❌ 结果分析失败")
        
        # 保存更新后的元数据
        self.save_metadata()
        
        print("✅ 结果分析阶段完成")
        return result
    
    def run_preprocessing_only(self, selected_formats: list = None, test_single: bool = False):
        """只运行预处理流程"""
        print("🎯 运行预处理流程")
        if test_single:
            print("🧪 测试模式：只处理单个样本")
        
        self.preprocess_data(selected_formats, test_single)
        self.print_statistics()
        print("🎉 预处理流程完成!")
    
    def run_full_pipeline(self, selected_formats: list = None, test_single: bool = False):
        """运行完整流程"""
        print("🎯 开始炼金术数据生成完整流程")
        if test_single:
            print("🧪 测试模式：只处理单个样本")
        print(f"📝 实验名称: {self.config.experiment.name}")
        print(f"📋 实验描述: {self.config.experiment.description}")
        
        # 阶段1: 数据预处理
        self.preprocess_data(selected_formats, test_single)
        
        # 阶段2: 系统准备
        self.prepare_systems(test_single)
        
        # 阶段3: 炼金术模拟
        self.run_alchemical_simulation(test_single)
        
        # 阶段4: 结果分析
        self.analyze_results(test_single)
        
        # 阶段5: 统计信息
        self.print_statistics()
        
        print("🎉 炼金术数据生成完整流程完成!")
    
    def run_single_test(self, selected_formats: list = None):
        """运行单个样本测试"""
        print("🧪 开始单个样本测试流程")
        print("=" * 60)
        
        # 运行完整流程，但只处理一个样本
        self.run_full_pipeline(selected_formats, test_single=True)
        
        print("=" * 60)
        print("🧪 单个样本测试完成!")
        print("💡 如果测试成功，可以运行完整流程处理所有样本")


@hydra.main(version_base=None, config_path="./config", config_name="base")
def main(cfg: DictConfig):
    """主函数"""
    print("⚙️ 实验配置:")
    print(OmegaConf.to_yaml(cfg))
    
    # 创建实验实例
    experiment = AlchemicalDataGenerator(cfg)
    
    # 根据配置选择运行模式
    mode = cfg.get('mode', 'full')
    test_single = cfg.get('test_single', False)
    selected_formats = cfg.get('selected_formats')
    
    if mode == 'preprocess_only':
        experiment.run_preprocessing_only(selected_formats, test_single)
    elif mode == 'full':
        experiment.run_full_pipeline(selected_formats, test_single)
    elif mode == 'test_single':
        experiment.run_single_test(selected_formats)
    else:
        print(f"❌ 未知的运行模式: {mode}")
        print("可用模式: preprocess_only, full, test_single")

if __name__ == "__main__":
    main()
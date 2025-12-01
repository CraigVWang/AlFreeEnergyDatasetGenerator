#!/usr/bin/env python3
"""
分析器模块
功能：可视化炼金术结果、计算自由能、评估采样质量
作者：CraigV Wang
版本：1.2
更新：适配新的配置文件结构，保持数据流一致性
"""

import os
import csv
import pickle
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# OpenMM相关导入
from openmmtools.multistate import MultiStateReporter, ReplicaExchangeAnalyzer


class Analyzer:
    """
    分析器类
    可视化炼金术结果、计算自由能、评估采样质量
    """
    
    def __init__(self, config: DictConfig):
        """
        初始化分析器
        
        参数:
            config: Hydra配置对象
        """
        self.config = config
        self.setup_directories()
        self.setup_plotting()
        
    def setup_directories(self):
        """创建输出目录结构"""
        self.analysis_dir = Path(self.config.output.analysis_dir)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建分析结果目录: {self.analysis_dir}")
    
    def setup_plotting(self):
        """设置绘图样式"""
        plt.style.use('default')
        sns.set_palette("husl")
        self.fig_size = (10, 8)
        
        print("🎨 绘图样式设置完成")
    
    def load_alchemical_data(self, alchemical_result_file):
        """
        加载炼金术结果数据
        
        参数:
            alchemical_result_file: 炼金术结果文件路径
            
        返回:
            analyzer: 分析器对象
        """
        print(f"📖 加载炼金术数据: {alchemical_result_file}")
        
        try:
            reporter = MultiStateReporter(alchemical_result_file, open_mode='r')
            analyzer = ReplicaExchangeAnalyzer(reporter)
            reporter.close()
            
            print(f"  ✅ 成功加载炼金术数据")
            print(f"    迭代次数: {analyzer.n_iterations}")
            print(f"    状态数量: {analyzer.n_states}")
            print(f"    副本数量: {analyzer.n_replicas}")
            
            return analyzer
            
        except Exception as e:
            print(f"❌ 加载炼金术数据失败: {e}")
            return None
    
    def calculate_free_energy(self, analyzer):
        """
        计算自由能
        
        参数:
            analyzer: 分析器对象
            
        返回:
            free_energy: 自由能计算结果
        """
        print("  📊 计算自由能...")
        
        try:
            # 获取自由能矩阵
            free_energy = analyzer.get_free_energy()
            
            # 提取自由能矩阵和误差矩阵
            delta_g_matrix = free_energy[0]  # 自由能差值矩阵
            error_matrix = free_energy[1]    # 误差矩阵
            
            # 计算从状态0（完全相互作用）到最后一个状态（无相互作用）的自由能差
            n_states = analyzer.n_states
            delta_g_solvation = delta_g_matrix[0, n_states-1]
            error_solvation = error_matrix[0, n_states-1]
            
            # 转换为标量
            delta_g_scalar = delta_g_solvation.item() if hasattr(delta_g_solvation, 'item') else float(delta_g_solvation)
            error_scalar = error_solvation.item() if hasattr(error_solvation, 'item') else float(error_solvation)
            
            free_energy_result = {
                'delta_g': delta_g_scalar,
                'error': error_scalar,
                'delta_g_matrix': delta_g_matrix,
                'error_matrix': error_matrix,
                'n_states': n_states
            }
            
            print(f"  ✅ 自由能计算完成")
            print(f"    ΔG_solvation = {delta_g_scalar:.2f} ± {error_scalar:.2f} kcal/mol")
            
            return free_energy_result
            
        except Exception as e:
            print(f"❌ 自由能计算失败: {e}")
            return None
    
    def plot_free_energy_profile(self, free_energy_result, output_path):
        """
        绘制自由能剖面图
        
        参数:
            free_energy_result: 自由能计算结果
            output_path: 输出文件路径
        """
        print("  📈 绘制自由能剖面图...")
        
        try:
            delta_g_matrix = free_energy_result['delta_g_matrix']
            n_states = free_energy_result['n_states']
            
            # 计算相对于第一个状态的自由能
            free_energies = [delta_g_matrix[0, i] for i in range(n_states)]
            
            # 创建lambda值（假设均匀分布）
            lambda_values = np.linspace(1.0, 0.0, n_states)
            
            plt.figure(figsize=self.fig_size)
            plt.plot(lambda_values, free_energies, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Lambda Value')
            plt.ylabel('Free Energy (kcal/mol)')
            plt.title('Free Energy Profile')
            plt.grid(True, alpha=0.3)
            
            # 保存图像
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ 自由能剖面图保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ 绘制自由能剖面图失败: {e}")
    
    def plot_replica_exchange(self, analyzer, output_path):
        """
        绘制副本交换图
        
        参数:
            analyzer: 分析器对象
            output_path: 输出文件路径
        """
        print("  🔄 绘制副本交换图...")
        
        try:
            # 获取状态轨迹
            replica_state_indices = analyzer.replica_state_indices
            
            plt.figure(figsize=(12, 8))
            
            # 绘制每个副本的状态轨迹
            for replica_index in range(min(analyzer.n_replicas, 8)):  # 只显示前8个副本
                state_trajectory = replica_state_indices[:, replica_index]
                plt.plot(state_trajectory, label=f'Replica {replica_index+1}')
            
            plt.xlabel('Iteration')
            plt.ylabel('State Index')
            plt.title('Replica Exchange State Trajectories')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # 保存图像
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ 副本交换图保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ 绘制副本交换图失败: {e}")
    
    def plot_energy_time_series(self, analyzer, output_path):
        """
        绘制能量时间序列
        
        参数:
            analyzer: 分析器对象
            output_path: 输出文件路径
        """
        print("  ⚡ 绘制能量时间序列...")
        
        try:
            # 获取能量数据
            energy_matrix = analyzer.energy_matrix
            
            plt.figure(figsize=(12, 8))
            
            # 绘制几个状态的能量时间序列
            n_states_to_plot = min(analyzer.n_states, 5)
            for state_index in range(n_states_to_plot):
                energies = energy_matrix[:, state_index, 0]  # 第一个副本
                plt.plot(energies, label=f'State {state_index}')
            
            plt.xlabel('Iteration')
            plt.ylabel('Energy (kT)')
            plt.title('Energy Time Series')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图像
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ 能量时间序列图保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ 绘制能量时间序列失败: {e}")
    
    def assess_sampling_quality(self, analyzer, free_energy_result):
        """
        评估采样质量
        
        参数:
            analyzer: 分析器对象
            free_energy_result: 自由能计算结果
            
        返回:
            quality_metrics: 质量评估指标
        """
        print("  📋 评估采样质量...")
        
        quality_metrics = {}
        
        try:
            # 1. 检查收敛性
            n_iterations = analyzer.n_iterations
            quality_metrics['n_iterations'] = n_iterations
            
            # 2. 检查误差大小
            error = free_energy_result['error']
            delta_g = free_energy_result['delta_g']
            relative_error = abs(error / delta_g) if delta_g != 0 else float('inf')
            
            quality_metrics['absolute_error'] = error
            quality_metrics['relative_error'] = relative_error
            
            # 3. 收敛评估
            if relative_error < 0.1:  # 相对误差小于10%
                convergence_status = "良好"
            elif relative_error < 0.2:
                convergence_status = "一般"
            else:
                convergence_status = "较差"
            
            quality_metrics['convergence_status'] = convergence_status
            
            # 4. 采样充分性评估
            if n_iterations >= 100:
                sampling_sufficiency = "充分"
            elif n_iterations >= 50:
                sampling_sufficiency = "基本充分"
            else:
                sampling_sufficiency = "不足"
            
            quality_metrics['sampling_sufficiency'] = sampling_sufficiency
            
            print(f"  ✅ 采样质量评估完成")
            print(f"    收敛状态: {convergence_status}")
            print(f"    采样充分性: {sampling_sufficiency}")
            print(f"    相对误差: {relative_error:.2%}")
            
            return quality_metrics
            
        except Exception as e:
            print(f"❌ 采样质量评估失败: {e}")
            return None
    
    def generate_report(self, free_energy_result, quality_metrics, mol_name, output_path):
        """
        生成分析报告
        
        参数:
            free_energy_result: 自由能计算结果
            quality_metrics: 质量评估指标
            mol_name: 分子名称
            output_path: 输出文件路径
        """
        print("  📄 生成分析报告...")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"炼金术分析报告 - {mol_name}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("1. 自由能计算结果\n")
                f.write("-" * 40 + "\n")
                f.write(f"溶剂化自由能: {free_energy_result['delta_g']:.2f} ± {free_energy_result['error']:.2f} kcal/mol\n")
                f.write(f"状态数量: {free_energy_result['n_states']}\n\n")
                
                f.write("2. 采样质量评估\n")
                f.write("-" * 40 + "\n")
                for key, value in quality_metrics.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("3. 结果解释\n")
                f.write("-" * 40 + "\n")
                delta_g = free_energy_result['delta_g']
                if delta_g < 0:
                    f.write("溶剂化过程是自发的 (ΔG < 0)\n")
                else:
                    f.write("溶剂化过程是非自发的 (ΔG > 0)\n")
                
                f.write("\n4. 建议\n")
                f.write("-" * 40 + "\n")
                if quality_metrics['convergence_status'] == "较差":
                    f.write("建议增加采样时间以获得更好的收敛\n")
                if quality_metrics['sampling_sufficiency'] == "不足":
                    f.write("建议增加迭代次数以获得更充分的采样\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("报告生成完成\n")
                f.write("=" * 60 + "\n")
            
            print(f"  ✅ 分析报告保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ 生成分析报告失败: {e}")
    
    def analyze_single_alchemical(self, alchemical_result, mol_name):
        """
        分析单个炼金术结果
        
        参数:
            alchemical_result: 炼金术结果字典
            mol_name: 分子名称
            
        返回:
            analysis_result: 分析结果
        """
        print(f"🔬 分析炼金术结果: {mol_name}")
        
        try:
            # 检查是否有输出文件路径
            if 'output_file' not in alchemical_result or not alchemical_result['output_file']:
                print(f"❌ 没有找到炼金术输出文件: {mol_name}")
                return None
            
            # 创建分子特定的输出目录
            mol_analysis_dir = self.analysis_dir / mol_name
            mol_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # 加载炼金术数据
            analyzer = self.load_alchemical_data(alchemical_result['output_file'])
            if analyzer is None:
                return None
            
            # 计算自由能
            free_energy_result = self.calculate_free_energy(analyzer)
            if free_energy_result is None:
                return None
            
            # 绘制图表
            self.plot_free_energy_profile(
                free_energy_result, 
                mol_analysis_dir / "free_energy_profile.png"
            )
            
            self.plot_replica_exchange(
                analyzer,
                mol_analysis_dir / "replica_exchange.png"
            )
            
            self.plot_energy_time_series(
                analyzer,
                mol_analysis_dir / "energy_timeseries.png"
            )
            
            # 评估采样质量
            quality_metrics = self.assess_sampling_quality(analyzer, free_energy_result)
            
            # 生成报告
            self.generate_report(
                free_energy_result,
                quality_metrics,
                mol_name,
                mol_analysis_dir / "analysis_report.txt"
            )
            
            analysis_result = {
                'success': True,
                'name': mol_name,
                'free_energy': free_energy_result,
                'quality_metrics': quality_metrics,
                'analysis_dir': str(mol_analysis_dir)
            }
            
            return analysis_result
            
        except Exception as e:
            print(f"❌ 分析炼金术结果失败 {mol_name}: {e}")
            return None
    
    def load_alchemical_results(self):
        """
        从炼金术结果目录加载所有结果
        
        返回:
            alchemical_results: 炼金术结果列表
        """
        alchemical_dir = Path(self.config.input.alchemical_dir)
        results_file = alchemical_dir / "alchemical_results.pkl"
        
        if not results_file.exists():
            # 尝试加载CSV文件
            csv_file = alchemical_dir / "alchemical_results.csv"
            if csv_file.exists():
                print(f"📖 从CSV文件加载炼金术结果: {csv_file}")
                alchemical_results = []
                with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        alchemical_results.append({
                            'name': row['name'],
                            'success': row['success'].lower() == 'true',
                            'output_file': row['output_file']
                        })
                print(f"✅ 从CSV加载了 {len(alchemical_results)} 个炼金术结果")
                return alchemical_results
            else:
                print(f"❌ 未找到炼金术结果文件: {results_file}")
                return []
        
        try:
            with open(results_file, 'rb') as f:
                alchemical_results = pickle.load(f)
            
            print(f"✅ 成功加载 {len(alchemical_results)} 个炼金术结果")
            return alchemical_results
            
        except Exception as e:
            print(f"❌ 加载炼金术结果失败: {e}")
            return []
    
    def run_analysis_batch(self, test_single=False):
        """
        运行批量分析
        
        参数:
            test_single: 是否只测试单个样本
            
        返回:
            analysis_results: 分析结果列表
        """
        print("=" * 60)
        print("📊 开始炼金术结果分析流程")
        if test_single:
            print("🧪 测试模式：只分析单个样本")
        print("=" * 60)
        
        # 加载炼金术结果
        alchemical_results = self.load_alchemical_results()
        if not alchemical_results:
            print("❌ 没有可用的炼金术结果")
            return {
                'success': False,
                'message': '没有可用的炼金术结果'
            }
        
        # 只筛选成功的炼金术结果
        successful_results = [r for r in alchemical_results if r.get('success', False)]
        
        if not successful_results:
            print("❌ 没有成功的炼金术结果可供分析")
            return {
                'success': False,
                'message': '没有成功的炼金术结果'
            }
        
        # 如果只测试单个样本，只分析第一个结果
        if test_single and successful_results:
            successful_results = [successful_results[0]]
            print(f"🧪 测试模式：只分析第一个结果: {successful_results[0]['name']}")
        
        print(f"🔍 准备分析 {len(successful_results)} 个炼金术结果")
        
        successful_analysis = 0
        analysis_results = []
        detailed_results = []
        
        # 使用进度条
        data_iterator = tqdm(successful_results, desc="🔄 炼金术分析")
        
        for alchemical_result in data_iterator:
            mol_name = alchemical_result['name']
            
            # 分析单个结果
            analysis_result = self.analyze_single_alchemical(alchemical_result, mol_name)
            
            summary_result = {
                'name': mol_name,
                'success': analysis_result is not None,
                'analysis_dir': analysis_result['analysis_dir'] if analysis_result else None,
                'free_energy': analysis_result['free_energy']['delta_g'] if analysis_result else None,
                'error': analysis_result['free_energy']['error'] if analysis_result else None
            }
            detailed_results.append(summary_result)
            
            if analysis_result:
                successful_analysis += 1
                analysis_results.append(analysis_result)
            
            # 更新进度条
            data_iterator.set_postfix_str(f"成功: {successful_analysis}/{len(successful_results)}")
        
        # 统计结果
        print(f"\n📊 分析完成:")
        print(f"   - 成功分析: {successful_analysis}/{len(successful_results)}")
        print(f"   - 成功率: {successful_analysis/len(successful_results)*100:.1f}%")
        
        # 保存结果
        self.save_results_csv(detailed_results)
        self.save_detailed_results(analysis_results)
        
        return {
            'success': True,
            'total_alchemical': len(successful_results),
            'successful_analysis': successful_analysis,
            'success_rate': successful_analysis/len(successful_results),
            'analysis_results': analysis_results,
            'summary_file': str(self.analysis_dir / "analysis_results.csv"),
            'detailed_file': str(self.analysis_dir / "analysis_results.pkl")
        }
    
    def save_results_csv(self, results):
        """
        保存分析结果到CSV文件
        
        参数:
            results: 分析结果列表
        """
        output_csv = self.analysis_dir / "analysis_results.csv"
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['name', 'success', 'analysis_dir', 'free_energy', 'error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"💾 分析结果保存到: {output_csv}")
    
    def save_detailed_results(self, results):
        """
        保存详细的分析结果到pickle文件
        
        参数:
            results: 详细的分析结果列表
        """
        output_pkl = self.analysis_dir / "analysis_results.pkl"
        
        with open(output_pkl, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 详细分析结果保存到: {output_pkl}")


@hydra.main(version_base=None, config_path="../config", config_name="analysis")
def main(cfg: DictConfig):
    """
    主函数 - 使用Hydra加载配置并执行炼金术分析
    
    参数:
        cfg: Hydra配置对象
    """
    print("=" * 60)
    print("⚙️ 炼金术分析配置:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # 创建分析器
    analyzer = Analyzer(cfg)
    
    # 执行分析
    test_single = cfg.get('test_single', False)
    results = analyzer.run_analysis_batch(test_single)
    
    if results['success']:
        print("\n🎉 炼金术分析流程完成!")
        print(f"📈 成功率: {results['success_rate']*100:.1f}%")
        print(f"📁 结果文件:")
        print(f"  - 汇总: {results['summary_file']}")
        print(f"  - 详细: {results['detailed_file']}")
    else:
        print("❌ 炼金术分析流程失败")


if __name__ == "__main__":
    main()
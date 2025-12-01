#!/usr/bin/env python3
"""
炼金术师模块
功能：执行炼金术自由能计算，计算溶剂化自由能
作者：CraigV Wang
版本：1.1
"""

import os
import csv
import pickle
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from tqdm import tqdm

# OpenMM相关导入
from openmm import app, unit, Platform
from openmmtools import alchemy, mcmc, multistate
from openmmtools.states import SamplerState, ThermodynamicState, CompoundThermodynamicState
from openmmtools.multistate import MultiStateReporter


class Alchemist:
    """
    炼金术师类
    执行炼金术自由能计算，计算溶剂化自由能
    """
    
    def __init__(self, config: DictConfig):
        """
        初始化炼金术师
        
        参数:
            config: Hydra配置对象
        """
        self.config = config
        self.setup_directories()
        
    def setup_directories(self):
        """创建输出目录结构"""
        self.alchemical_results_dir = Path("./dataset/alchemical_results")
        self.alchemical_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建炼金术结果目录: {self.alchemical_results_dir}")
    
    def load_prepared_system(self, preparation_result):
        """
        从准备结果中加载系统
        
        参数:
            preparation_result: 系统准备结果
            
        返回:
            system: 分子系统
            positions: 分子位置
            topology: 分子拓扑
            ligand_atom_count: 配体原子数量
        """
        try:
            system = preparation_result['system']
            positions = preparation_result['positions']
            topology = preparation_result['topology']
            ligand_atom_count = preparation_result['ligand_atom_count']
            
            print(f"  ✅ 成功加载系统: {preparation_result['name']}")
            print(f"    配体原子数量: {ligand_atom_count}")
            
            return system, positions, topology, ligand_atom_count
            
        except Exception as e:
            print(f"❌ 加载系统失败: {e}")
            return None, None, None, None
    
    def setup_alchemical_system(self, system, ligand_atom_count):
        """
        设置炼金术系统
        
        参数:
            system: 原始分子系统
            ligand_atom_count: 配体原子数量
            
        返回:
            alchemical_system: 炼金术系统
        """
        print("  🧪 设置炼金术系统...")
        
        # 定义炼金术区域（配体原子）
        alchemical_regions = alchemy.AlchemicalRegion(
            alchemical_atoms=list(range(ligand_atom_count))
        )
        
        # 创建炼金术工厂和系统
        factory = alchemy.AbsoluteAlchemicalFactory()
        alchemical_system = factory.create_alchemical_system(system, alchemical_regions)
        
        print(f"  ✅ 炼金术系统创建完成")
        return alchemical_system
    
    def create_thermodynamic_states(self, alchemical_system, lambda_schedule=None):
        """
        创建热力学状态
        
        参数:
            alchemical_system: 炼金术系统
            lambda_schedule: lambda值调度表或分段定义
            
        返回:
            thermodynamic_states: 热力学状态列表
        """
        print("  🔥 创建热力学状态...")
        
        # 如果lambda_schedule是分段定义，则动态计算
        if lambda_schedule is None:
            lambda_schedule = self.config.alchemical.lambda_schedule
        
        # 检查是否是分段定义
        if isinstance(lambda_schedule, dict) and 'segments' in lambda_schedule:
            # 动态计算lambda值
            print("  ⏳ 动态生成lambda值...")
            lambda_values = np.array([])
            for segment in lambda_schedule['segments']:
                start, end, num = segment
                segment_values = np.linspace(start, end, num)
                lambda_values = np.concatenate([lambda_values, segment_values])
            
            # 确保唯一性和排序
            lambda_values = np.unique(lambda_values)
            lambda_values.sort()
            lambda_values = lambda_values[::-1]  # 从1.0到0.0
            
            print(f"  ✅ 动态生成 {len(lambda_values)} 个lambda值")
        else:
            # 直接使用提供的列表
            lambda_values = lambda_schedule
        
        thermodynamic_states = []
        
        for lambda_val in lambda_values:
            # 创建炼金术状态
            alchemical_state = alchemy.AlchemicalState(
                lambda_sterics=lambda_val, 
                lambda_electrostatics=lambda_val
            )
            
            # 创建热力学状态
            thermodynamic_state = ThermodynamicState(
                alchemical_system, 
                temperature=self.config.alchemical.temperature * unit.kelvin
            )
            
            # 创建复合状态
            compound_state = CompoundThermodynamicState(
                thermodynamic_state, 
                composable_states=[alchemical_state]
            )
            compound_state.lambda_sterics = lambda_val
            compound_state.lambda_electrostatics = lambda_val
            
            thermodynamic_states.append(compound_state)
        
        print(f"  ✅ 创建了 {len(thermodynamic_states)} 个热力学状态")
        return thermodynamic_states
    
    def create_sampler_state(self, positions, topology):
        """
        创建采样器状态
        
        参数:
            positions: 分子位置
            topology: 分子拓扑
            
        返回:
            sampler_state: 采样器状态
        """
        sampler_state = SamplerState(
            positions=positions,
            box_vectors=topology.getPeriodicBoxVectors()
        )
        
        return sampler_state
    
    def setup_replica_exchange(self, thermodynamic_states, sampler_state):
        """
        设置副本交换模拟
        
        参数:
            thermodynamic_states: 热力学状态列表
            sampler_state: 采样器状态
            
        返回:
            simulation: 副本交换模拟器
        """
        print("  🔄 设置副本交换模拟...")
        
        # 创建移动策略
        move = mcmc.LangevinDynamicsMove(
            timestep=self.config.alchemical.time_step * unit.femtoseconds,
            collision_rate=self.config.alchemical.collision_rate / unit.picoseconds,
            n_steps=self.config.alchemical.steps_per_iteration,
            reassign_velocities=True
        )
        
        # 创建副本交换模拟器
        simulation = multistate.ReplicaExchangeSampler(
            mcmc_moves=move,
            number_of_iterations=self.config.alchemical.total_iterations
        )
        
        # 创建报告器
        output_file = self.alchemical_results_dir / "alchemical_simulation.nc"
        if output_file.exists():
            output_file.unlink()
            print(f"  🗑️  删除已存在的输出文件: {output_file}")
        
        reporter = MultiStateReporter(
            str(output_file), 
            checkpoint_interval=self.config.alchemical.checkpoint_interval
        )
        
        # 创建模拟
        simulation.create(
            thermodynamic_states=thermodynamic_states,
            sampler_states=[sampler_state],
            storage=reporter
        )
        
        print(f"  ✅ 副本交换模拟设置完成")
        print(f"    迭代次数: {self.config.alchemical.total_iterations}")
        print(f"    每迭代步数: {self.config.alchemical.steps_per_iteration}")
        print(f"    输出文件: {output_file}")
        
        return simulation, reporter
    
    def run_alchemical_simulation(self, simulation, reporter):
        """
        运行炼金术模拟
        
        参数:
            simulation: 副本交换模拟器
            reporter: 报告器
            
        返回:
            success: 是否成功完成
        """
        print("  🚀 开始炼金术模拟...")
        
        try:
            # 运行模拟
            simulation.run()
            
            # 关闭报告器
            reporter.close()
            
            print("  ✅ 炼金术模拟完成")
            return True
            
        except Exception as e:
            print(f"❌ 炼金术模拟失败: {e}")
            reporter.close()
            return False
    
    def run_single_alchemical(self, preparation_result):
        """
        对单个系统运行炼金术计算
        
        参数:
            preparation_result: 系统准备结果
            
        返回:
            alchemical_result: 炼金术计算结果
        """
        mol_name = preparation_result['name']
        print(f"🔬 运行炼金术计算: {mol_name}")
        
        try:
            # 加载准备好的系统
            system, positions, topology, ligand_atom_count = self.load_prepared_system(preparation_result)
            if system is None:
                return None
            
            # 设置炼金术系统
            alchemical_system = self.setup_alchemical_system(system, ligand_atom_count)
            
            # 创建热力学状态
            thermodynamic_states = self.create_thermodynamic_states(alchemical_system)
            
            # 创建采样器状态
            sampler_state = self.create_sampler_state(positions, topology)
            
            # 设置副本交换模拟
            simulation, reporter = self.setup_replica_exchange(thermodynamic_states, sampler_state)
            
            # 运行模拟
            success = self.run_alchemical_simulation(simulation, reporter)
            
            if success:
                # 基础分析
                analysis_result = self.analyze_alchemical_results(mol_name)
                
                alchemical_result = {
                    'success': True,
                    'name': mol_name,
                    'output_file': str(self.alchemical_results_dir / "alchemical_simulation.nc"),
                    'analysis': analysis_result
                }
                
                return alchemical_result
            else:
                return None
                
        except Exception as e:
            print(f"❌ 炼金术计算失败 {mol_name}: {e}")
            return None
    
    def analyze_alchemical_results(self, mol_name):
        """
        分析炼金术结果（基础分析，详细分析在单独的脚本中）
        
        参数:
            mol_name: 分子名称
            
        返回:
            analysis_result: 分析结果
        """
        print(f"  📊 分析炼金术结果: {mol_name}")
        
        # 这里只做基础分析，详细分析在单独的analyzer.py中
        analysis_result = {
            'status': 'completed',
            'message': '炼金术模拟完成，请使用analyzer.py进行详细分析'
        }
        
        return analysis_result
    
    def run_alchemical_batch(self, preparation_results, test_single=False):
        """
        运行批量炼金术计算
        
        参数:
            preparation_results: 系统准备结果列表
            test_single: 是否只测试单个样本
            
        返回:
            alchemical_results: 炼金术计算结果列表
        """
        print("=" * 60)
        print("🚀 开始炼金术自由能模拟流程")
        if test_single:
            print("🧪 测试模式：只处理单个样本")
        print("=" * 60)
        
        # 如果只测试单个样本，只处理第一个系统
        if test_single and preparation_results:
            preparation_results = [preparation_results[0]]
            print(f"🧪 测试模式：只处理第一个系统: {preparation_results[0]['name']}")
        
        print(f"🔍 准备对 {len(preparation_results)} 个系统进行炼金术计算")
        
        successful_alchemical = 0
        alchemical_results = []
        detailed_results = []
        
        # 使用进度条
        data_iterator = tqdm(preparation_results, desc="🔄 炼金术计算")
        
        for prep_result in data_iterator:
            alchemical_result = self.run_single_alchemical(prep_result)
            
            summary_result = {
                'name': prep_result['name'],
                'success': alchemical_result is not None,
                'output_file': alchemical_result['output_file'] if alchemical_result else None
            }
            detailed_results.append(summary_result)
            
            if alchemical_result:
                successful_alchemical += 1
                alchemical_results.append(alchemical_result)
            
            # 更新进度条
            data_iterator.set_postfix_str(f"成功: {successful_alchemical}/{len(preparation_results)}")
        
        # 统计结果
        print(f"\n📊 炼金术计算完成:")
        print(f"   - 成功计算: {successful_alchemical}/{len(preparation_results)}")
        print(f"   - 成功率: {successful_alchemical/len(preparation_results)*100:.1f}%")
        
        # 保存结果
        self.save_results_csv(detailed_results)
        
        return {
            'success': True,
            'total_systems': len(preparation_results),
            'successful_alchemical': successful_alchemical,
            'success_rate': successful_alchemical/len(preparation_results),
            'alchemical_results': alchemical_results
        }
    
    def save_results_csv(self, results):
        """
        保存炼金术结果到CSV文件
        
        参数:
            results: 炼金术结果列表
        """
        output_csv = self.alchemical_results_dir / "alchemical_results.csv"
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['name', 'success', 'output_file']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"💾 炼金术结果保存到: {output_csv}")

    def load_preparation_results(self):
        """
        从准备结果目录加载所有系统的准备结果
        
        返回:
            preparation_results: 准备结果列表
        """
        preparation_dir = Path(self.config.input.preparation_dir)
        results_file = preparation_dir / "preparation_results.pkl"
        
        if not results_file.exists():
            print(f"❌ 未找到准备结果文件: {results_file}")
            return []
        
        try:
            with open(results_file, 'rb') as f:
                preparation_results = pickle.load(f)
            
            print(f"✅ 成功加载 {len(preparation_results)} 个准备结果")
            return preparation_results
            
        except Exception as e:
            print(f"❌ 加载准备结果失败: {e}")
            return []


@hydra.main(version_base=None, config_path="./config", config_name="alchemical")
def main(cfg: DictConfig):
    """
    主函数 - 使用Hydra加载配置并执行炼金术计算
    
    参数:
        cfg: Hydra配置对象
    """
    print("⚙️ 炼金术模拟配置:")
    print(OmegaConf.to_yaml(cfg))
    
    # 创建炼金术师
    alchemist = Alchemist(cfg)
    
    # 这里需要从系统准备阶段获取准备结果
    # 在实际使用中，这些结果应该从文件或之前的步骤传递过来
    
    # 检查准备结果文件
    preparation_results_file = Path("./dataset/prepared_systems/preparation_results.csv")
    if preparation_results_file.exists():
        print(f"📖 找到准备结果文件: {preparation_results_file}")
        # 这里可以添加代码来加载真实的准备结果
    else:
        print("❌ 未找到准备结果文件，请先运行系统准备阶段")
        return
    
    # 示例：创建一个空的preparation_results列表
    preparation_results = []
    
    # 执行炼金术计算
    test_single = cfg.get('test_single', False)
    results = alchemist.run_alchemical_batch(preparation_results, test_single)
    
    if results['success']:
        print("🎉 炼金术计算流程完成!")
        print(f"📈 成功率: {results['success_rate']*100:.1f}%")
    else:
        print("❌ 炼金术计算流程失败")


if __name__ == "__main__":
    main()
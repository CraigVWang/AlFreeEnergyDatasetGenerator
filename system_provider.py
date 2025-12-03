"""
分子系统提供器模块
功能：为炼金术自由能模拟准备完整的分子系统
作者：CraigV Wang
版本：2.2
"""

import os
import csv
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any, Optional

# OpenMM相关导入
from openmm import app, unit, LangevinMiddleIntegrator, Platform, MonteCarloBarostat, Vec3
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule


class SystemProvider:
    """
    分子系统提供器类
    为炼金术自由能模拟准备完整的分子系统
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化系统提供器
        
        参数:
            config: Hydra配置对象
        """
        self.config = config
        self.setup_directories()
        
    def setup_directories(self):
        """创建输出目录结构"""
        self.prepared_systems_dir = Path("./dataset/prepared_systems")
        self.prepared_systems_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建系统准备输出目录: {self.prepared_systems_dir}")
    
    def read_molecule_file(self, file_path: str):
        """
        读取分子文件，支持PDB和CIF格式
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdb':
                pdb = app.PDBFile(file_path)
                return pdb.topology, pdb.positions
            elif file_ext == '.cif':
                cif = app.PDBxFile(file_path)
                return cif.topology, cif.positions
            else:
                # 对于预处理后的PDB文件
                pdb = app.PDBFile(file_path)
                return pdb.topology, pdb.positions
                
        except Exception as e:
            print(f"❌ 读取文件失败 {file_path}: {e}")
            return None, None
    
    def create_forcefield(self, mol: Optional[Molecule] = None):
        """
        创建力场，包含GAFF小分子力场
        从配置中读取力场参数
        """
        forcefield_config = self.config.preparation.forcefield
        
        # 获取力场文件列表
        forcefield_files = forcefield_config.get('files')
        if not forcefield_files:
            forcefield_files = ["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"]
        
        # 创建基础力场
        forcefield = app.ForceField(*forcefield_files)
        
        # 如果需要GAFF力场并且有分子对象
        if forcefield_config.get('use_gaff', True) and mol is not None:
            try:
                gaff = GAFFTemplateGenerator(molecules=mol)
                forcefield.registerTemplateGenerator(gaff.generator)
                print("  ✅ 注册GAFF力场")
            except Exception as e:
                print(f"  ⚠️ 注册GAFF力场失败: {e}")
        
        return forcefield
    
    def center_molecule_in_box(self, modeller, ligand_atom_count=None):
        """
        将分子在水盒子中居中
        """
        print("  🎯 检查分子居中...")
        
        positions = modeller.positions
        
        # 计算分子边界信息（仅用于显示）
        coords_list = [(pos.x, pos.y, pos.z) for pos in positions]
        if coords_list:
            coords = np.array(coords_list)
            print(f"    分子中心: {np.mean(coords, axis=0)}")
            print(f"    分子边界: {np.min(coords, axis=0)} 到 {np.max(coords, axis=0)}")
        
        box_vectors = modeller.topology.getPeriodicBoxVectors()
        box_center = np.array([
            box_vectors[0][0].value_in_unit(unit.nanometers) / 2,
            box_vectors[1][1].value_in_unit(unit.nanometers) / 2,
            box_vectors[2][2].value_in_unit(unit.nanometers) / 2
        ])
        
        if ligand_atom_count is None:
            ligand_atom_count = self.detect_ligand_atoms(modeller.topology)
        
        print(f"    检测到配体原子数量: {ligand_atom_count}")
        
        # 计算配体中心
        if ligand_atom_count > 0:
            ligand_coords = np.array([(pos.x, pos.y, pos.z) for pos in positions[:ligand_atom_count]])
            ligand_center = np.mean(ligand_coords, axis=0)
        else:
            ligand_center = box_center  # 如果没有配体，使用盒子中心
        
        offset = np.linalg.norm(ligand_center - box_center)
        
        print(f"    配体中心: [{ligand_center[0]:.3f}, {ligand_center[1]:.3f}, {ligand_center[2]:.3f}] nm")
        print(f"    盒子中心: {box_center}")
        print(f"    偏移距离: {offset:.6f} nm")
        
        # 从配置中获取居中阈值
        centering_threshold = self.config.preparation.get('centering_threshold', 0.1)
        
        if offset > centering_threshold and ligand_atom_count > 0:
            print("  🔧 分子偏离中心较远，进行修正...")
            
            translation = box_center - ligand_center
            new_positions = []
            for pos in positions:
                new_x = pos.x + translation[0]
                new_y = pos.y + translation[1]
                new_z = pos.z + translation[2]
                new_positions.append(Vec3(new_x, new_y, new_z))
            
            modeller.positions = new_positions
            print("  ✅ 分子居中修正完成")
            
            # 计算修正后的偏移
            ligand_coords_new = np.array([(pos.x, pos.y, pos.z) for pos in modeller.positions[:ligand_atom_count]])
            ligand_center_new = np.mean(ligand_coords_new, axis=0)
            offset_new = np.linalg.norm(ligand_center_new - box_center)
            
            print(f"    修正后配体中心: [{ligand_center_new[0]:.3f}, {ligand_center_new[1]:.3f}, {ligand_center_new[2]:.3f}] nm")
            print(f"    修正后偏移距离: {offset_new:.6f} nm")
        else:
            print("  ✓ 分子位置良好")
        
        return modeller, ligand_atom_count
    
    def detect_ligand_atoms(self, topology):
        """自动检测配体原子数量"""
        # 更直接的方法
        residues = list(topology.residues())
        if not residues:
            return 0
            
        first_residue = residues[0]
        ligand_atom_count = 0
        
        for atom in topology.atoms():
            if atom.residue == first_residue:
                ligand_atom_count += 1
            else:
                break
                
        return ligand_atom_count
    
    def prepare_system(self, topology, positions, mol=None):
        """
        准备分子系统
        所有参数都从配置中读取
        """
        # 从配置中获取系统参数
        system_config = self.config.preparation.system
        
        # 创建力场
        forcefield = self.create_forcefield(mol)
        modeller = app.Modeller(topology, positions)
        
        # 添加氢原子
        if system_config.get('add_hydrogens', True):
            print("  ➕ 添加氢原子...")
            try:
                modeller.addHydrogens(forcefield)
            except Exception as e:
                print(f"  ⚠️ 添加氢原子时出现问题: {e}")
                print("  ⚠️ 继续处理，可能氢原子已存在")
        
        # 添加溶剂
        solvent_config = self.config.preparation.solvent
        print("  💧 添加溶剂...")
        
        box_size = solvent_config.get('box_size', 8.0) * unit.nanometers
        solvent_model = solvent_config.get('model', 'tip3p')
        
        # 获取额外的溶剂参数
        solvent_params = {}
        if 'padding' in solvent_config and solvent_config.padding is not None:
            solvent_params['padding'] = solvent_config.padding * unit.nanometers
        if 'positive_ion' in solvent_config and solvent_config.positive_ion is not None:
            solvent_params['positiveIon'] = solvent_config.positive_ion
        if 'negative_ion' in solvent_config and solvent_config.negative_ion is not None:
            solvent_params['negativeIon'] = solvent_config.negative_ion
        if 'ionic_strength' in solvent_config and solvent_config.ionic_strength is not None:
            solvent_params['ionicStrength'] = solvent_config.ionic_strength * unit.molar
        
        if solvent_params:
            modeller.addSolvent(
                forcefield, 
                model=solvent_model, 
                boxSize=Vec3(box_size, box_size, box_size),
                **solvent_params
            )
        else:
            modeller.addSolvent(
                forcefield, 
                model=solvent_model, 
                boxSize=Vec3(box_size, box_size, box_size)
            )
        
        # 分子居中
        modeller, ligand_atom_count = self.center_molecule_in_box(modeller)
        
        # 创建系统
        print("  ⚙️  创建系统...")
        
        # 获取系统创建参数
        nonbonded_method = system_config.get('nonbonded_method', 'PME')
        constraints = system_config.get('constraints', 'HBonds')
        cutoff = system_config.get('cutoff', 1.0) * unit.nanometers
        
        # 映射字符串到OpenMM常量
        method_map = {
            'PME': app.PME,
            'NoCutoff': app.NoCutoff,
            'CutoffNonPeriodic': app.CutoffNonPeriodic,
            'CutoffPeriodic': app.CutoffPeriodic
        }
        
        constraint_map = {
            'None': None,
            'HBonds': app.HBonds,
            'AllBonds': app.AllBonds,
            'HAngles': app.HAngles
        }
        
        nonbonded_method_enum = method_map.get(nonbonded_method, app.PME)
        constraints_enum = constraint_map.get(constraints, app.HBonds)
        
        system = forcefield.createSystem(
            modeller.topology, 
            nonbondedMethod=nonbonded_method_enum,
            nonbondedCutoff=cutoff,
            constraints=constraints_enum,
            rigidWater=system_config.get('rigid_water', True),
            ewaldErrorTolerance=system_config.get('ewald_error_tolerance', 0.0005)
        )
        
        return system, modeller, ligand_atom_count
    
    def create_simulation(self, topology, system, positions):
        """创建模拟器 - 所有参数从配置读取"""
        # 从配置中获取积分器参数
        integrator_config = self.config.preparation.integrator
        
        temperature = integrator_config.get('temperature', 300.0) * unit.kelvin
        friction_coeff = integrator_config.get('friction_coeff', 1.0) / unit.picosecond
        time_step = integrator_config.get('time_step', 2.0) * unit.femtoseconds
        
        integrator = LangevinMiddleIntegrator(
            temperature,
            friction_coeff,
            time_step
        )
        
        # 从配置中获取平台设置
        platform_config = self.config.preparation.platform
        
        # 根据配置决定使用GPU还是CPU
        use_cuda = platform_config.get('use_cuda', True)
        device = 'CUDA' if use_cuda else 'CPU'
        
        # 获取设备索引（用于多GPU）
        device_index = platform_config.get('device_index', '0')
        
        try:
            platform = Platform.getPlatformByName(device)
            properties = {}
            
            if device == 'CUDA':
                properties = {'DeviceIndex': device_index}
                if platform_config.get('precision', 'mixed') == 'double':
                    properties['Precision'] = 'double'
            
            simulation = app.Simulation(topology, system, integrator, platform, properties)
            simulation.context.setPositions(positions)
            
            print(f"  🔧 使用平台: {device} (设备: {device_index})")
            return simulation
            
        except Exception as e:
            print(f"  ⚠️ 无法使用 {device} 平台: {e}")
            print("  🔧 回退到CPU平台")
            platform = Platform.getPlatformByName('CPU')
            simulation = app.Simulation(topology, system, integrator, platform)
            simulation.context.setPositions(positions)
            return simulation
    
    def minimize_energy(self, simulation):
        """执行能量最小化 - 参数从配置读取"""
        print("  🔽 执行能量最小化...")
        
        minimization_config = self.config.preparation.minimization
        max_iterations = minimization_config.get('max_iterations', 1000)
        tolerance = minimization_config.get('tolerance', 10.0) * unit.kilojoule_per_mole
        
        simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=tolerance)
        
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        minimized_energy = state.getPotentialEnergy()
        
        print(f"  ✅ 最小化完成，能量: {minimized_energy}")
        
        return state.getPositions()
    
    def heat_system(self, simulation):
        """加热系统 - 参数从配置读取"""
        heating_config = self.config.preparation.heating
        
        initial_temp = heating_config.get('initial_temperature', 50.0)
        target_temp = heating_config.get('target_temperature', 300.0)
        temp_step = heating_config.get('temperature_step', 50.0)
        steps_per_temp = heating_config.get('steps_per_temperature', 5000)
        
        print(f"  🔥 加热系统: {initial_temp}K -> {target_temp}K")
        
        # 设置初始速度
        simulation.context.setVelocitiesToTemperature(initial_temp * unit.kelvin)
        
        # 逐步加热
        current_temp = initial_temp
        while current_temp < target_temp:
            next_temp = min(current_temp + temp_step, target_temp)
            simulation.integrator.setTemperature(next_temp * unit.kelvin)
            simulation.step(steps_per_temp)
            current_temp = next_temp
        
        state = simulation.context.getState(getPositions=True, getTemperature=True)
        current_temp_value = state.getTemperature()
        
        print(f"  ✅ 加热完成，当前温度: {current_temp_value}")
        
        return state.getPositions()
    
    def equilibrate_system(self, simulation, system, topology):
        """平衡系统 - 参数从配置读取"""
        equilibration_config = self.config.preparation.equilibration
        
        print("  ⚖️  平衡系统...")
        
        # 添加压力控制（NPT平衡）
        if equilibration_config.get('use_barostat', True):
            pressure = equilibration_config.get('pressure', 1.0) * unit.atmospheres
            temperature = self.config.preparation.integrator.get('temperature', 300.0) * unit.kelvin
            frequency = equilibration_config.get('barostat_frequency', 25)
            
            system.addForce(MonteCarloBarostat(pressure, temperature, frequency))
            print(f"  📊 添加压力控制: {pressure}，频率: {frequency}")
        
        # 重新初始化上下文
        simulation.context.reinitialize(preserveState=True)
        
        # 执行平衡步骤
        npt_steps = equilibration_config.get('npt_steps', 50000)
        simulation.step(npt_steps)
        
        # 获取平衡后的状态
        state = simulation.context.getState(
            getPositions=True, 
            getEnergy=True,
            getTemperature=True,
            getVolume=True
        )
        
        potential_energy = state.getPotentialEnergy()
        temperature = state.getTemperature()
        volume = state.getVolume()
        
        print(f"  ✅ 平衡完成:")
        print(f"     能量: {potential_energy}")
        print(f"     温度: {temperature}")
        print(f"     体积: {volume}")
        
        return state.getPositions()
    
    def save_prepared_system(self, topology, positions, output_path):
        """保存准备好的系统"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            app.PDBFile.writeFile(topology, positions, f)
        
        print(f"  💾 保存准备系统到: {output_path}")
    
    def prepare_single_system(self, mol_info: Dict[str, Any], update_callback=None):
        """
        准备单个分子系统
        
        参数:
            mol_info: 分子信息字典（来自metadata）
            update_callback: 更新回调函数，用于更新状态
            
        返回:
            preparation_result: 准备结果字典
        """
        try:
            mol_name = mol_info['name']
            
            # 获取预处理后的文件路径
            file_path = mol_info.get('preprocessed_file_path', '')
            if not file_path or file_path == '':
                # 如果没有预处理文件，尝试使用原始文件
                file_path = mol_info.get('original_file_path', '')
            
            if not file_path or not Path(file_path).exists():
                print(f"❌ 文件不存在: {file_path}")
                if update_callback:
                    update_callback(mol_name, 'preparation', False, 
                                   {'processing_notes': f'文件不存在: {file_path}'})
                return None
            
            print(f"🔬 准备分子系统: {mol_name}")
            print(f"  📁 文件: {file_path}")
            
            # 读取分子文件
            topology, positions = self.read_molecule_file(file_path)
            if topology is None or positions is None:
                if update_callback:
                    update_callback(mol_name, 'preparation', False, 
                                   {'processing_notes': '读取文件失败'})
                return None
            
            # 创建分子对象（用于力场生成）
            mol = None
            try:
                mol = Molecule.from_file(file_path)
            except Exception as e:
                print(f"  ⚠️ 创建分子对象失败，使用默认力场: {e}")
            
            # 准备系统（所有参数从配置读取）
            system, modeller, ligand_atom_count = self.prepare_system(
                topology, positions, mol
            )
            
            # 创建模拟器（所有参数从配置读取）
            simulation = self.create_simulation(modeller.topology, system, modeller.positions)
            
            # 能量最小化（所有参数从配置读取）
            self.minimize_energy(simulation)
            
            # 加热系统（所有参数从配置读取）
            self.heat_system(simulation)
            
            # 平衡系统（所有参数从配置读取）
            self.equilibrate_system(simulation, system, modeller.topology)
            
            # 获取最终状态
            state = simulation.context.getState(
                getPositions=True, 
                getVelocities=True, 
                getEnergy=True,
                getForces=True
            )
            
            final_positions = state.getPositions()
            final_velocities = state.getVelocities()
            potential_energy = state.getPotentialEnergy()
            forces = state.getForces()
            
            # 保存准备好的系统
            relative_path = mol_info.get('relative_path', '')
            output_path = self.prepared_systems_dir / relative_path / f"{mol_name}_prepared.pdb"
            
            self.save_prepared_system(modeller.topology, final_positions, output_path)
            
            # 准备结果
            preparation_result = {
                'success': True,
                'name': mol_name,
                'topology': modeller.topology,
                'system': system,
                'positions': final_positions,
                'velocities': final_velocities,
                'potential_energy': potential_energy,
                'forces': forces,
                'ligand_atom_count': ligand_atom_count,
                'output_path': str(output_path),
                'simulation': simulation,
                'forcefield': self.create_forcefield(mol),
                'box_vectors': modeller.topology.getPeriodicBoxVectors()
            }
            
            # 更新状态
            if update_callback:
                additional_info = {
                    'prepared_system_path': str(output_path),
                    'ligand_atom_count': ligand_atom_count,
                    'processing_notes': '系统准备成功'
                }
                update_callback(mol_name, 'preparation', True, additional_info)
            
            return preparation_result
            
        except Exception as e:
            print(f"❌ 准备分子系统 {mol_info.get('name', 'unknown')} 失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 更新状态为失败
            if update_callback:
                update_callback(
                    mol_info.get('name', 'unknown'), 
                    'preparation', 
                    False, 
                    {'processing_notes': f'系统准备失败: {str(e)}'}
                )
            return None
    
    def run_preparation(self, metadata: List[Dict[str, Any]], test_single: bool = False) -> Dict[str, Any]:
        """
        运行批量系统准备流程
        
        参数:
            metadata: 元数据列表
            test_single: 是否只测试单个样本
            
        返回:
            处理结果的字典
        """
        print("=" * 60)
        print("🚀 开始分子系统准备流程")
        
        # 显示当前配置
        print(f"⚙️ 使用配置:")
        print(f"   - 盒子大小: {self.config.preparation.solvent.get('box_size', 8.0)} nm")
        print(f"   - 溶剂模型: {self.config.preparation.solvent.get('model', 'tip3p')}")
        print(f"   - 温度: {self.config.preparation.integrator.get('temperature', 300.0)} K")
        print(f"   - 时间步长: {self.config.preparation.integrator.get('time_step', 2.0)} fs")
        print(f"   - 使用GPU: {self.config.preparation.platform.get('use_cuda', True)}")
        
        if test_single:
            print("🧪 测试模式：只处理单个样本")
        print("=" * 60)
        
        if not metadata:
            print("❌ 没有可用的元数据")
            return {'success': False, 'error': '没有可用的元数据'}
        
        print(f"📖 读取到 {len(metadata)} 个分子信息")
        
        # 初始化变量
        successful_preparations = 0
        preparation_results = []
        
        # 筛选需要处理的分子
        molecules_to_process = []
        for mol_info in metadata:
            mol_name = mol_info['name']
            
            # 检查预处理是否成功
            processed_successfully = mol_info.get('processed_successfully', 'False').lower() == 'true'
            if not processed_successfully:
                print(f"  ⏭️  跳过 {mol_name}：预处理未成功")
                continue
            
            # 检查是否已经成功准备
            minimized_successfully = mol_info.get('minimized_successfully', 'False').lower() == 'true'
            if minimized_successfully:
                print(f"  ⏭️  跳过 {mol_name}：系统已经准备成功")
                continue
            
            molecules_to_process.append(mol_info)
        
        # 如果测试单个样本，只处理第一个
        if test_single and molecules_to_process:
            molecules_to_process = [molecules_to_process[0]]
            print(f"🧪 测试模式：只处理第一个分子: {molecules_to_process[0]['name']}")
            print(f"🧪 注意：测试模式使用与完整模式相同的系统和参数配置")
        
        print(f"🔍 找到 {len(molecules_to_process)} 个需要处理的分子")
        
        if not molecules_to_process:
            print("✅ 没有需要处理的分子，所有分子都已准备完成")
            return {
                'success': True,
                'total_molecules': 0,
                'successful_preparations': 0,
                'success_rate': 0.0,
                'preparation_results': [],
                'message': '系统准备完成'
            }
        
        # 创建更新回调函数
        def update_callback(mol_name, stage, success, additional_info=None):
            """更新回调函数，由main.py实现"""
            # 这里我们只是打印信息，实际的更新在main.py中进行
            status = "成功" if success else "失败"
            print(f"  📝 更新分子状态: {mol_name} - {stage} = {status}")
        
        # 使用进度条显示处理进度
        data_iterator = tqdm(molecules_to_process, desc="🔄 系统准备")
        
        for mol_info in data_iterator:
            mol_name = mol_info['name']
            
            result = self.prepare_single_system(mol_info, update_callback)
            
            if result:
                successful_preparations += 1
                preparation_results.append(result)
            
            # 更新进度条描述
            data_iterator.set_postfix_str(f"成功: {successful_preparations}/{len(molecules_to_process)}")
        
        print(f"\n📊 准备完成:")
        print(f"   - 总处理: {len(molecules_to_process)}")
        print(f"   - 成功准备: {successful_preparations}")
        
        success_rate = 0
        if molecules_to_process:
            success_rate = successful_preparations / len(molecules_to_process)
            print(f"   - 成功率: {success_rate*100:.1f}%")
        else:
            print(f"   - 成功率: N/A")
        
        # 保存简化的结果文件（用于调试）
        self.save_simplified_results(preparation_results)
        
        # 保存完整的准备结果（供炼金术阶段使用）
        self.save_preparation_results(preparation_results)
        
        return {
            'success': True,
            'total_molecules': len(molecules_to_process),
            'successful_preparations': successful_preparations,
            'success_rate': success_rate,
            'preparation_results': preparation_results,
            'message': '系统准备完成'
        }
    
    def save_simplified_results(self, results: List[Dict[str, Any]]):
        """保存简化的结果文件（用于调试）"""
        if not results:
            return
            
        output_csv = self.prepared_systems_dir / "preparation_summary.csv"
        
        simplified_results = []
        for result in results:
            simplified_results.append({
                'name': result['name'],
                'success': result['success'],
                'output_path': result['output_path'],
                'ligand_atom_count': result['ligand_atom_count'],
                'energy': str(result.get('potential_energy', 'N/A'))
            })
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['name', 'success', 'output_path', 'ligand_atom_count', 'energy']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in simplified_results:
                writer.writerow(result)
        
        print(f"💾 简化结果保存到: {output_csv}")

    def save_preparation_results(self, preparation_results: List[Dict[str, Any]]):
        """
        保存完整的准备结果到pickle文件，供炼金术阶段使用
        
        参数:
            preparation_results: 完整的准备结果列表
        """
        if not preparation_results:
            print("⚠️ 警告：准备结果为空，跳过保存pickle文件")
            return
        
        # 保存完整的准备结果
        output_pkl = self.prepared_systems_dir / "preparation_results.pkl"
        
        try:
            with open(output_pkl, 'wb') as f:
                pickle.dump(preparation_results, f)
            print(f"💾 完整准备结果保存到: {output_pkl}")
            
            # 同时保存一个简化的CSV版本用于查看
            self.save_simplified_results(preparation_results)
        except Exception as e:
            print(f"❌ 保存准备结果失败: {e}")
            import traceback
            traceback.print_exc()
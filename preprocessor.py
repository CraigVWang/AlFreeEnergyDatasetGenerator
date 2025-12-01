"""
结构文件预处理模块
功能：扫描化学结构文件，生成元数据，转换文件格式，并为后续处理准备数据
作者：CraigV Wang
版本：1.0
"""

import os
import re
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

# 导入分子处理相关的库
from openff.toolkit import Molecule
from rdkit import Chem
from tqdm import tqdm

# 导入自定义工具
from utils.xyz_to_pdb_converter import convert_xyz_to_pdb


class StructurePreprocessor:
    """
    结构文件预处理类
    处理化学结构文件的扫描、元数据生成和格式转换
    """
    
    # 类常量 - 支持的所有格式
    SUPPORTED_EXTENSIONS = {'.pdb', '.cif', '.mol2', '.sdf', '.xyz'}
    FILE_TYPE_MAPPING = {
        '.pdb': 'pdb',
        '.cif': 'cif',  # mmCIF 文件使用 .cif 扩展名
        '.mol2': 'mol2',
        '.sdf': 'sdf',
        '.xyz': 'xyz'
    }

    def __init__(self, config: DictConfig, selected_formats: Optional[List[str]] = None):
        """
        初始化预处理类
        
        参数:
            config: Hydra配置对象
            selected_formats: 可选，指定要处理的文件格式列表
        """
        self.config = config
        self.selected_formats = selected_formats
        
        # 从配置中获取路径
        self.raw_dir = Path(self.config.input.data_dir)
        self.preprocessed_dir = Path(self.config.output.preprocessed_dir)
        self.metadata_dir = Path(self.config.output.metadata_dir)
        
        self.setup_directories()
        self.metadata_file = self.metadata_dir / "metadata.csv"
        
    def setup_directories(self):
        """根据配置创建必要的目录结构"""
        # 确保原始数据目录存在
        raw_dir = Path(self.config['raw_directory'])
        if not raw_dir.exists():
            raw_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 创建原始数据目录: {raw_dir}")
        
        # 创建预处理输出目录
        self.preprocessed_dir = Path(self.config['preprocessed_directory'])
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建元数据输出目录
        metadata_dir = Path(self.config['metadata_directory'])
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        print("📂 目录设置完成:")
        print(f"   - 原始数据目录: {self.config['raw_directory']}")
        print(f"   - 预处理目录: {self.config['preprocessed_directory']}")
        print(f"   - 元数据目录: {self.config['metadata_directory']}")
        print(f"   - 元数据文件: {self.metadata_file}")
        
        # 显示选定的文件格式
        if self.selected_formats:
            print(f"   - 选定格式: {', '.join(self.selected_formats)}")
        else:
            print(f"   - 选定格式: 所有支持格式")
    
    def initialize_metadata_file(self):
        """
        初始化元数据文件，如果不存在则创建，如果存在则读取
        """
        if not self.metadata_file.exists():
            # 创建新文件
            columns = [
                'name',                     # 分子名称
                'filename',                 # 原始文件名
                'original_file_path',       # 原始文件路径
                'relative_path',            # 相对路径
                'pdb_id',                   # PDB ID
                'original_file_type',       # 原始文件类型
                'preprocessed_file_path',   # 预处理后文件路径
                'preprocessed_file_type',   # 预处理后文件类型
                'prepared_system_path',     # 准备系统路径
                'alchemical_result_path',   # 炼金术结果路径
                'analysis_result_path',     # 分析结果路径
                
                # 状态列
                'processed_successfully',   # 预处理是否成功
                'minimized_successfully',   # 最小化是否成功
                'alchemical_successfully',  # 炼金术是否成功
                'analysis_successfully',    # 分析是否成功
                'finish_successfully',      # 全部完成是否成功
                
                # 时间戳
                'preprocess_timestamp',     # 预处理时间
                'preparation_timestamp',    # 系统准备时间
                'alchemical_timestamp',     # 炼金术时间
                'analysis_timestamp',       # 分析时间
                
                # 统计信息
                'ligand_atom_count',        # 配体原子数
                'free_energy_value',        # 自由能值
                'free_energy_error',        # 自由能误差
                'processing_notes'          # 处理备注
            ]
            
            with open(self.metadata_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
            
            print(f"📄 创建新元数据文件: {self.metadata_file}")
            return []
        else:
            # 读取现有文件
            print(f"📖 读取现有元数据文件: {self.metadata_file}")
            with open(self.metadata_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
    
    def save_metadata(self, data: List[Dict[str, str]]):
        """
        保存元数据到文件
        
        参数:
            data: 元数据字典列表
        """
        if not data:
            return
            
        # 获取所有可能的列
        all_columns = set()
        for item in data:
            all_columns.update(item.keys())
        
        # 写入文件
        with open(self.metadata_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_columns))
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        
        print(f"💾 元数据已保存到: {self.metadata_file}")
    
    def update_molecule_status(self, 
                              metadata: List[Dict[str, str]],
                              molecule_name: str,
                              stage: str,
                              success: bool = True,
                              additional_info: Optional[Dict[str, Any]] = None):
        """
        更新分子的状态信息
        
        参数:
            metadata: 元数据列表
            molecule_name: 分子名称
            stage: 阶段名称 ('preprocess', 'preparation', 'alchemical', 'analysis')
            success: 该阶段是否成功
            additional_info: 额外的信息字典
        """
        # 查找分子
        for item in metadata:
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
    
    def extract_pdb_id(self, filename: str) -> str:
        """
        从文件名中提取可能的PDB ID
        
        PDB ID通常是4个字符的代码，第一个是数字1-9，后面三个是字母或数字
        
        参数:
            filename: 文件名
            
        返回:
            PDB ID字符串或'NAN'
        """
        pdb_pattern = r'[1-9][a-z0-9]{3}'
        matches = re.findall(pdb_pattern, filename.lower())
        
        for match in matches:
            if len(match) == 4 and re.match(r'^[1-9a-z][a-z0-9]{3}$', match):
                return match.upper()
        
        return 'NAN'
    
    def get_file_type(self, filename: str) -> str:
        """
        根据文件扩展名确定文件类型
        
        参数:
            filename: 文件名
            
        返回:
            文件类型字符串
        """
        ext = Path(filename).suffix.lower()
        return self.FILE_TYPE_MAPPING.get(ext, 'unknown')
    
    def scan_directory(self, existing_metadata: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        扫描目录中的结构文件，与现有元数据合并
        
        参数:
            existing_metadata: 现有元数据列表
            
        返回:
            包含文件信息的字典列表
        """
        root_dir = self.config['raw_directory']
        
        # 创建现有分子的查找字典
        existing_molecules = {item['name']: item for item in existing_metadata}
        new_data = []
        
        # 处理选定的文件格式
        if self.selected_formats:
            selected_extensions = {f'.{fmt.lower()}' for fmt in self.selected_formats}
            supported_extensions = self.SUPPORTED_EXTENSIONS.intersection(selected_extensions)
            print(f"🔍 处理以下格式的文件: {', '.join(self.selected_formats)}")
        else:
            supported_extensions = self.SUPPORTED_EXTENSIONS
            print(f"🔍 处理所有支持格式的文件")
        
        print(f"📁 开始扫描目录: {root_dir}")
        
        # 收集所有文件
        all_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                all_files.append((root, file))
        
        # 使用进度条显示扫描进度
        file_iterator = tqdm(all_files, desc="📂 扫描文件")
        
        # 过滤支持的文件格式
        for root, file in file_iterator:
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix.lower()
            
            if file_ext in supported_extensions:
                mol_name = Path(file).stem
                
                # 如果分子已经在元数据中，跳过扫描（只更新新文件）
                if mol_name in existing_molecules:
                    continue
                
                pdb_id = self.extract_pdb_id(mol_name)
                file_type = self.get_file_type(file)
                
                # 计算相对于原始目录的相对路径
                relative_path = os.path.relpath(root, root_dir)
                
                # 创建新分子的默认数据
                new_data.append({
                    'name': mol_name,
                    'filename': file,
                    'original_file_path': file_path,
                    'relative_path': relative_path,
                    'pdb_id': pdb_id,
                    'original_file_type': file_type,
                    
                    # 初始化所有列
                    'preprocessed_file_path': '',
                    'preprocessed_file_type': '',
                    'prepared_system_path': '',
                    'alchemical_result_path': '',
                    'analysis_result_path': '',
                    
                    'processed_successfully': 'False',
                    'minimized_successfully': 'False',
                    'alchemical_successfully': 'False',
                    'analysis_successfully': 'False',
                    'finish_successfully': 'False',
                    
                    'preprocess_timestamp': '',
                    'preparation_timestamp': '',
                    'alchemical_timestamp': '',
                    'analysis_timestamp': '',
                    
                    'ligand_atom_count': '0',
                    'free_energy_value': '0.0',
                    'free_energy_error': '0.0',
                    'processing_notes': ''
                })
        
        # 合并现有数据和新数据
        merged_data = existing_metadata + new_data
        
        print(f"✅ 找到 {len(merged_data)} 个分子（{len(existing_molecules)} 个现有 + {len(new_data)} 个新）")
        return merged_data
    
    def process_molecule_file(self, file_path: str, mol_name: str, file_type: str, relative_path: str) -> Optional[str]:
        """
        处理分子文件 - 复制或转换格式，保持目录结构
        
        参数:
            file_path: 输入文件路径
            mol_name: 分子名称
            file_type: 文件类型
            relative_path: 相对于原始目录的路径
            
        返回:
            处理后的文件路径，如果失败返回None
        """
        # 构建输出目录路径，保持原始目录结构
        output_dir = self.preprocessed_dir / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定输出文件路径和格式
        if file_type in ['pdb', 'cif']:
            # 对于PDB和CIF文件，直接复制到预处理目录的对应子目录
            output_path = output_dir / Path(file_path).name
            try:
                shutil.copy2(file_path, output_path)
                return str(output_path)
            except Exception as e:
                print(f"❌ 复制失败 {file_path}: {e}")
                return None
        else:
            # 对于其他格式，转换为PDB格式，但保持目录结构
            output_path = output_dir / f"{mol_name}.pdb"
            
            try:
                if file_type == 'sdf':
                    mol = Molecule.from_file(file_path)
                    mol.to_file(str(output_path), file_format='pdb')
                    return str(output_path)
                    
                elif file_type == 'mol2':
                    # 使用RDKit读取.mol2文件
                    rdkit_mol = Chem.MolFromMol2File(file_path, removeHs=False)
                    # 将RDKit分子对象转换为OpenFF的Molecule对象
                    mol = Molecule.from_rdkit(rdkit_mol)
                    mol.to_file(str(output_path), file_format='pdb')
                    return str(output_path)
                    
                elif file_type == 'xyz':
                    # 使用成熟的转换器
                    pdb_path = convert_xyz_to_pdb(
                        file_path, 
                        str(output_path),
                    )
                    return pdb_path
                    
                else:
                    print(f"⚠️ 跳过不支持的文件格式: {file_path}")
                    return None
                    
            except Exception as e:
                print(f"❌ 转换失败 {file_path}: {e}")
                return None
    
    def batch_process_files(self, metadata: List[Dict[str, str]], test_single: bool = False) -> List[Dict[str, str]]:
        """
        批量处理文件，保持目录结构
        
        参数:
            metadata: 元数据列表
            test_single: 是否只测试单个样本
            
        返回:
            更新后的元数据列表
        """
        successful_processing = 0
        total_to_process = len(metadata)
        
        # 如果测试单个样本，只处理第一个
        if test_single and metadata:
            metadata = [metadata[0]]
            total_to_process = 1
            print(f"🧪 测试模式：只处理第一个分子: {metadata[0]['name']}")
        
        # 使用进度条显示处理进度
        data_iterator = tqdm(metadata, desc="🔄 处理文件")
        
        for item in data_iterator:
            mol_name = item['name']
            input_file = item['original_file_path']
            file_type = item['original_file_type']
            relative_path = item['relative_path']
            
            # 如果已经处理成功，跳过
            if item.get('processed_successfully', 'False').lower() == 'true':
                data_iterator.set_postfix_str(f"跳过: {successful_processing}/{total_to_process}")
                continue
            
            # 处理文件，传递相对路径
            output_file = self.process_molecule_file(input_file, mol_name, file_type, relative_path)
                           
            if output_file:
                # 确定输出文件类型
                if file_type in ['pdb', 'cif']:
                    output_file_type = file_type
                else:
                    output_file_type = 'pdb'
                
                # 更新元数据
                self.update_molecule_status(
                    metadata=metadata,
                    molecule_name=mol_name,
                    stage='preprocess',
                    success=True,
                    additional_info={
                        'preprocessed_file_path': output_file,
                        'preprocessed_file_type': output_file_type
                    }
                )
                
                successful_processing += 1
                # 更新进度条描述
                data_iterator.set_postfix_str(f"成功: {successful_processing}/{total_to_process}")

            else:
                print(f"❌ 处理失败: {input_file}")
                # 更新状态为失败
                self.update_molecule_status(
                    metadata=metadata,
                    molecule_name=mol_name,
                    stage='preprocess',
                    success=False
                )
        
        print(f"📊 成功处理 {successful_processing}/{total_to_process} 个文件")
        return metadata
    
    def generate_statistics(self, metadata: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        生成文件统计信息
        
        参数:
            metadata: 元数据列表
            
        返回:
            统计信息字典
        """
        stats = {
            'total_molecules': len(metadata),
            'file_types': {},
            'pdb_ids_count': 0,
            'processed_success': 0,
            'preparation_success': 0,
            'alchemical_success': 0,
            'analysis_success': 0,
            'finish_success': 0
        }
        
        for item in metadata:
            # 文件类型统计
            file_type = item['original_file_type']
            stats['file_types'][file_type] = stats['file_types'].get(file_type, 0) + 1
            
            # PDB ID统计
            if item.get('pdb_id', 'NAN') != 'NAN':
                stats['pdb_ids_count'] += 1
            
            # 状态统计
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
        
        return stats
    
    def print_statistics(self, stats: Dict[str, Any]):
        """
        打印详细的统计信息
        
        参数:
            stats: 统计信息字典
        """
        print("\n📈 文件类型统计:")
        for file_type, count in stats['file_types'].items():
            print(f"   - {file_type}: {count} 个文件")
        
        print(f"🔍 找到 {stats['pdb_ids_count']} 个可能的PDB ID")
        
        print("\n📊 处理状态统计:")
        print(f"   - 总分子数: {stats['total_molecules']}")
        print(f"   - 预处理成功: {stats['processed_success']} ({stats['processed_success']/stats['total_molecules']*100:.1f}%)")
        print(f"   - 系统准备成功: {stats['preparation_success']} ({stats['preparation_success']/stats['total_molecules']*100:.1f}%)")
        print(f"   - 炼金术成功: {stats['alchemical_success']} ({stats['alchemical_success']/stats['total_molecules']*100:.1f}%)")
        print(f"   - 分析成功: {stats['analysis_success']} ({stats['analysis_success']/stats['total_molecules']*100:.1f}%)")
        print(f"   - 完成全部流程: {stats['finish_success']} ({stats['finish_success']/stats['total_molecules']*100:.1f}%)")
    
    def run(self, test_single: bool = False) -> Dict[str, Any]:
        """
        主要的预处理流程
        
        参数:
            test_single: 是否只测试单个样本
            
        返回:
            处理结果的字典，包含统计信息和文件路径
        """
        print("=" * 60)
        print("🚀 开始结构文件预处理流程")
        if test_single:
            print("🧪 测试模式：只处理单个样本")
        print("=" * 60)
        
        # 初始化元数据文件
        metadata = self.initialize_metadata_file()
        
        # 扫描目录并更新元数据
        metadata = self.scan_directory(metadata)
        
        if not metadata:
            print("❌ 未找到任何结构文件")
            return {'success': False, 'message': '未找到任何结构文件'}
        
        # 执行文件处理
        print("\n🔄 开始批量处理文件...")
        metadata = self.batch_process_files(metadata, test_single)
        
        # 保存元数据
        self.save_metadata(metadata)
        
        # 生成统计信息
        stats = self.generate_statistics(metadata)
        self.print_statistics(stats)
        
        result = {
            'success': True,
            'total_molecules': len(metadata),
            'processed_success': stats['processed_success'],
            'processing_success_rate': stats['processed_success']/len(metadata) if len(metadata) > 0 else 0,
            'metadata_file': str(self.metadata_file),
            'metadata': metadata  # 返回元数据供后续阶段使用
        }
        
        print("=" * 60)
        print("🎉 结构文件预处理完成!")
        print("=" * 60)
        return result
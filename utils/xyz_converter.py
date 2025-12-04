"""
xyz 文件转换器
功能：将 XYZ 文件转换为 MOL2 格式（主要功能）或 PDB 格式（备用）
作者：CraigV Wang
版本：2.0
"""

import sys
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


def parse_xyz_file(xyz_file):
    """
    解析 xyz 文件，支持多种格式

    支持的格式：
    1. 标准 xyz：
       <原子数>
       <注释行>
       <原子> <x> <y> <z>

    2. 带电荷信息的 xyz：
       <电荷> <自旋多重度>
       <原子序号> <x> <y> <z>

    3. 带注释头的 xyz（量化软件输出）：
       <注释行>
       <空行>
       <电荷> <自旋多重度>
       <原子序号> <x> <y> <z>

    Returns
    -------
    atoms : list of tuple
        [(element, x, y, z), ...]
    charge : int
        净电荷
    multiplicity : int
        自旋多重度
    """
    with open(xyz_file, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    atoms = []
    charge = 0
    multiplicity = 1
    start_line = 0

    # 寻找数据起始行
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 1:
            continue

        # 情况 1：标准 xyz（第一行是原子数）
        if i == 0 and len(parts) == 1 and parts[0].isdigit():
            # 跳过原子数和注释行
            start_line = 2
            break

        # 情况 2：电荷 + 自旋多重度行
        if len(parts) == 2:
            try:
                charge = int(parts[0])
                multiplicity = int(parts[1])
                start_line = i + 1
                break
            except ValueError:
                # 不是数字，继续寻找
                continue

        # 情况 3：直接是坐标数据（至少4列：元素 x y z）
        if len(parts) >= 4:
            try:
                # 尝试解析为坐标
                float(parts[1])
                float(parts[2])
                float(parts[3])
                # 成功，这是数据起始行
                start_line = i
                break
            except ValueError:
                # 不是坐标数据，是注释行，继续
                continue

    # 解析原子坐标
    for line in lines[start_line:]:
        parts = line.split()
        if len(parts) < 4:
            continue

        # 尝试解析坐标，如果失败则跳过（注释行）
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except (ValueError, IndexError):
            continue

        # 原子标识可能是元素符号或原子序号
        atom_id = parts[0]
        if atom_id.isdigit():
            # 原子序号，转换为元素符号
            atomic_num = int(atom_id)
            element = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
        else:
            element = atom_id

        atoms.append((element, x, y, z))

    return atoms, charge, multiplicity


def create_mol_from_atoms(atoms, charge=0):
    """
    从原子列表创建 RDKit Mol 对象并推断键连接

    Parameters
    ----------
    atoms : list of tuple
        [(element, x, y, z), ...]
    charge : int
        净电荷

    Returns
    -------
    mol : rdkit.Chem.Mol
        RDKit 分子对象
    """
    # 创建可编辑的分子对象
    mol = Chem.RWMol()

    # 添加原子
    conf = Chem.Conformer(len(atoms))
    for i, (element, x, y, z) in enumerate(atoms):
        atom = Chem.Atom(element)
        mol.AddAtom(atom)
        conf.SetAtomPosition(i, (x, y, z))

    # 设置构象
    mol = mol.GetMol()
    mol.AddConformer(conf)

    # 推断键连接
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    mol = Chem.Mol(mol)

    # 尝试确定键类型
    Chem.SanitizeMol(mol)

    # 设置总电荷
    if charge != 0:
        mol.SetProp("_TotalCharge", str(charge))

    return mol


def set_pdb_info(mol, residue_name="LIG", chain_id="A"):
    """
    为分子设置 PDB 残基信息

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        分子对象
    residue_name : str
        残基名称（默认 LIG）
    chain_id : str
        链 ID（默认 A）
    """
    for atom in mol.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(residue_name)
        info.SetResidueNumber(1)
        info.SetChainId(chain_id)

        # 原子名：元素符号 + 序号
        atom_name = f"{atom.GetSymbol()}{atom.GetIdx()+1:02d}"
        info.SetName(atom_name)
        info.SetIsHeteroAtom(True)  # 标记为 HETATM

        atom.SetMonomerInfo(info)


def convert_xyz_to_mol2(xyz_file, output_path=None, residue_name="LIG", chain="A"):
    """
    将 XYZ 文件转换为 MOL2 格式（主要功能）
    
    Parameters
    ----------
    xyz_file : str
        输入的 XYZ 文件路径
    output_path : str, optional
        输出的 MOL2 文件路径
    residue_name : str, optional
        残基名称，默认 "LIG"
    chain : str, optional
        链 ID，默认 "A"
        
    Returns
    -------
    str or None
        成功返回输出文件路径，失败返回 None
    """
    return convert_xyz(xyz_file, output_path, "mol2", residue_name, chain)


def convert_xyz_to_pdb(xyz_file, output_path=None, residue_name="LIG", chain="A"):
    """
    将 XYZ 文件转换为 PDB 格式（备用功能）
    
    Parameters
    ----------
    xyz_file : str
        输入的 XYZ 文件路径
    output_path : str, optional
        输出的 PDB 文件路径
    residue_name : str, optional
        PDB 残基名称，默认 "LIG"
    chain : str, optional
        链 ID，默认 "A"
        
    Returns
    -------
    str or None
        成功返回输出文件路径，失败返回 None
    """
    return convert_xyz(xyz_file, output_path, "pdb", residue_name, chain)


def convert_xyz(xyz_file, output_path=None, output_format="mol2", residue_name="LIG", chain="A"):
    """
    将 XYZ 文件转换为 MOL2 或 PDB 格式
    
    Parameters
    ----------
    xyz_file : str
        输入的 XYZ 文件路径
    output_path : str, optional
        输出的文件路径
    output_format : str, optional
        输出格式，"mol2" 或 "pdb"，默认 "mol2"
    residue_name : str, optional
        PDB 残基名称，默认 "LIG"
    chain : str, optional
        链 ID，默认 "A"
        
    Returns
    -------
    str or None
        成功返回输出文件路径，失败返回 None
    """
    
    # 确保 XYZ 文件存在
    xyz_file = Path(xyz_file)
    if not xyz_file.exists():
        print(f"❌ XYZ 文件不存在: {xyz_file}")
        return None

    # 检查输出格式
    output_format = output_format.lower()
    if output_format not in ["pdb", "mol2"]:
        print(f"❌ 不支持的输出格式: {output_format}，支持 'pdb' 或 'mol2'")
        return None

    # 确定输出文件名
    if output_path is None:
        output_path = xyz_file.with_suffix(f".{output_format}")
    else:
        output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 步骤 1：解析 xyz 文件
    try:
        atoms, charge, multiplicity = parse_xyz_file(xyz_file)
    except Exception as e:
        print(f"❌ XYZ 文件解析失败: {e}")
        return None

    # 步骤 2：创建分子对象并推断键连接
    try:
        mol = create_mol_from_atoms(atoms, charge)
    except Exception as e:
        print(f"❌ 键连接推断失败: {e}")
        return None

    # 步骤 3：设置格式特定信息
    if output_format == "pdb":
        # PDB 格式需要设置残基信息
        set_pdb_info(mol, residue_name, chain)

    # 步骤 4：输出文件
    try:
        if output_format == "pdb":
            Chem.MolToPDBFile(mol, str(output_path))
        else:  # mol2
            # 保存为 MOL2 格式
            writer = Chem.SDWriter(str(output_path))
            writer.write(mol)
            writer.close()
            
    except Exception as e:
        print(f"❌ 输出失败：{e}")
        return None

    return str(output_path)


# 提供便捷的导入接口
convert_xyz_file = convert_xyz_to_mol2


if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) < 2:
        print("XYZ 文件转换器")
        print("主要功能：将 XYZ 文件转换为 MOL2 格式")
        print("备用功能：将 XYZ 文件转换为 PDB 格式")
        print("\n使用方法:")
        print("  1. 转换为 MOL2（默认）: python xyz_converter.py <xyz文件> [输出文件.mol2]")
        print("  2. 转换为 PDB: python xyz_converter.py <xyz文件> <输出文件.pdb> pdb")
        print("\n示例:")
        print("  python xyz_converter.py molecule.xyz")
        print("  python xyz_converter.py molecule.xyz output.mol2")
        print("  python xyz_converter.py molecule.xyz output.pdb pdb")
        sys.exit(1)
    
    xyz_file = sys.argv[1]
    
    # 确定输出文件
    output_path = None
    output_format = "mol2"  # 默认输出 MOL2 格式
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        
        # 从输出文件扩展名推断格式
        if output_path.lower().endswith('.pdb'):
            output_format = 'pdb'
        elif output_path.lower().endswith('.mol2'):
            output_format = 'mol2'
    
    # 如果提供了第三个参数，指定格式
    if len(sys.argv) > 3:
        output_format = sys.argv[3].lower()
    
    # 执行转换
    if output_format == "mol2":
        result = convert_xyz_to_mol2(
            xyz_file=xyz_file,
            output_path=output_path,
            residue_name="LIG",
            chain="A"
        )
    else:
        result = convert_xyz_to_pdb(
            xyz_file=xyz_file,
            output_path=output_path,
            residue_name="LIG",
            chain="A"
        )
    
    if result:
        print(f"✅ 转换成功: {result}")
        sys.exit(0)
    else:
        print(f"❌ 转换失败")
        sys.exit(1)
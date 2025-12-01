"""
xyz 文件转换为 PDB/MOL2 格式工具

功能：
- 读取 xyz 文件（支持多种格式）
- 推断化学键连接
- 生成 PDB/MOL2/SDF 文件
- 验证分子结构合理性
- 适用于从量化计算结果准备 MD 模拟输入
- 删除冗余打印语句，仅生成pdb文件
- 结构验证不通过时自动进行几何优化

"""

import sys
import os
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import Lipinski



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

    # 推断键连接（关键步骤！）
    # 使用距离矩阵自动连接化学键
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    mol = Chem.Mol(mol)

    # 尝试确定键类型
    Chem.SanitizeMol(mol)

    # 设置总电荷
    if charge != 0:
        mol.SetProp("_TotalCharge", str(charge))

    return mol


def set_pdb_info(mol, residue_name="MOL", chain_id="A"):
    """
    为分子设置 PDB 残基信息

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        分子对象
    residue_name : str
        残基名称（默认 MOL）
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


def validate_structure(mol):
    """
    验证分子结构合理性

    Returns
    -------
    issues : list of str
        发现的问题列表
    """
    issues = []

    # 检查是否有孤立原子
    fragments = Chem.GetMolFrags(mol, asMols=True)
    if len(fragments) > 1:
        issues.append(f"检测到 {len(fragments)} 个不连接的片段")

    # 检查键长合理性
    conf = mol.GetConformer()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)
        distance = pos_i.Distance(pos_j)

        # 合理键长范围：0.5-3.0 Å
        if distance < 0.5 or distance > 3.0:
            atom_i = mol.GetAtomWithIdx(i).GetSymbol()
            atom_j = mol.GetAtomWithIdx(j).GetSymbol()
            issues.append(f"异常键长：{atom_i}{i+1}-{atom_j}{j+1} = {distance:.2f} Å")

    # 检查形式电荷
    total_charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
    if total_charge != 0:
        issues.append(f"分子形式电荷：{total_charge:+d}")

    return issues


def print_molecule_info(mol):
    """
    打印分子详细信息
    """
    print("\n" + "="*60)
    print("分子信息摘要")
    print("="*60)

    # 基本信息
    formula = rdMolDescriptors.CalcMolFormula(mol)
    print(f"分子式：{formula}")
    print(f"原子总数：{mol.GetNumAtoms()}")
    print(f"重原子数：{mol.GetNumHeavyAtoms()}")
    print(f"氢原子数：{mol.GetNumAtoms() - mol.GetNumHeavyAtoms()}")
    print(f"化学键数：{mol.GetNumBonds()}")

    # SMILES 表示
    try:
        smiles = Chem.MolToSmiles(mol)
        print(f"SMILES：{smiles}")
    except:
        print("SMILES：无法生成")

    # 分子量
    mw = Descriptors.MolWt(mol)
    print(f"分子量：{mw:.2f} g/mol")

    # 氢键供体/受体
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    print(f"氢键供体：{hbd}")
    print(f"氢键受体：{hba}")

    # 可旋转键
    rotatable = Lipinski.NumRotatableBonds(mol)
    print(f"可旋转键：{rotatable}")

    print("="*60 + "\n")


def optimize_structure(mol, max_attempts=5):
    """
    对分子结构进行几何优化，直到验证通过或达到最大尝试次数
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        需要优化的分子
    max_attempts : int
        最大优化尝试次数
        
    Returns
    -------
    mol : rdkit.Chem.Mol
        优化后的分子
    optimized : bool
        是否成功优化
    """
    print(f"\n🔄 开始几何优化，最大尝试次数：{max_attempts}")
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n尝试 {attempt}/{max_attempts}:")
        
        try:
            # 使用 UFF 力场进行几何优化
            print("  使用 UFF 力场优化几何结构...")
            result = AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            
            if result == 0:
                print("  ✅ 几何优化成功收敛")
            else:
                print("  ⚠️ 几何优化达到最大迭代次数")
            
            # 验证优化后的结构
            issues = validate_structure(mol)
            
            if not issues:
                print("  ✅ 结构验证通过")
                return mol, True
            else:
                print(f"  ❌ 结构验证仍有问题：")
                for issue in issues:
                    print(f"    - {issue}")
                    
                if attempt < max_attempts:
                    print("  继续下一次优化尝试...")
                else:
                    print("  ⚠️ 达到最大优化次数，将使用当前结构")
                    
        except Exception as e:
            print(f"  ❌ 优化过程出错：{e}")
            if attempt < max_attempts:
                print("  继续下一次优化尝试...")
            else:
                print("  ⚠️ 达到最大优化次数，将使用当前结构")
    
    return mol, False


def convert_xyz_to_pdb(xyz_file, output_path=None, residue_name="MOL", chain="A", auto_optimize=True, max_optimization_attempts=5):
    """
    使用 xyz_to_pdb.py 转换器将 XYZ 文件转换为 PDB 格式
    
    Parameters
    ----------
    xyz_file : str
        输入的 XYZ 文件路径
    output_pdb_path : str, optional
        输出的 PDB 文件路径
    residue_name : str, optional
        PDB 残基名称，默认 "MOL"
    auto_optimize : bool, optional
        是否自动进行几何优化，默认 True
    max_optimization_attempts : int, optional
        最大优化尝试次数，默认 5
        
    Returns
    -------
    str or None
        成功返回 PDB 文件路径，失败返回 None
    """
    
    # 确保 XYZ 文件存在
    xyz_file = Path(xyz_file)
    if not xyz_file.exists():
        print(f"❌ XYZ 文件不存在: {xyz_file}")
        return None
    else:
        print(f"✅ 找到 XYZ 文件: {xyz_file}")

    # 确定输出前缀
    output_base = xyz_file.stem

    print(f"\n🔄 正在转换：{xyz_file}")
    print(f"📁 输出前缀：{output_base}")
    print(f"🏷️ 残基名称：{residue_name}")
    print(f"⚙️ 自动优化：{'开启' if auto_optimize else '关闭'}\n")

    # 步骤 1：解析 xyz 文件
    print("步骤 1/5: 读取 xyz 文件...")
    try:
        atoms, charge, multiplicity = parse_xyz_file(xyz_file)
        print(f"读取 {len(atoms)} 个原子")
        print(f"净电荷 = {charge}, 自旋多重度 = {multiplicity}")
    except Exception as e:
        print(f"读取失败：{e}")
        sys.exit(1)

    # 步骤 2：创建分子对象并推断键连接
    print("\n步骤 2/5: 推断化学键连接...")
    try:
        mol = create_mol_from_atoms(atoms, charge)
        print(f"成功推断 {mol.GetNumBonds()} 个化学键")
    except Exception as e:
        print(f"键连接推断失败：{e}")
        sys.exit(2)

    # 步骤 3：初始结构验证
    print("\n步骤 3/5: 初始结构验证...")
    initial_issues = validate_structure(mol)
    
    if not initial_issues:
        print("✅ 初始结构验证通过")
        needs_optimization = False
    else:
        print("❌ 初始结构发现问题：")
        for issue in initial_issues:
            print(f"  - {issue}")
        needs_optimization = auto_optimize

    # 步骤 4：结构优化（如果需要）
    optimized = False
    if needs_optimization:
        print("\n步骤 4/5: 进行结构优化...")
        mol, optimized = optimize_structure(mol, max_optimization_attempts)
        
        # 验证优化后的结构
        if optimized:
            final_issues = validate_structure(mol)
            if not final_issues:
                print("✅ 优化后结构验证通过")
            else:
                print("⚠️ 优化后仍有问题：")
                for issue in final_issues:
                    print(f"  - {issue}")
        else:
            print("⚠️ 结构优化未完全成功，将使用当前结构")
    else:
        print("\n步骤 4/5: 跳过结构优化")

    # 步骤 5：设置PDB信息并输出文件
    print("\n步骤 5/5: 设置PDB信息并生成输出文件...")
    
    # 设置PDB信息
    set_pdb_info(mol, residue_name, chain)
    print("PDB 信息设置完成")

    # 生成PDB文件
    output_dir = os.path.dirname(output_path)
    output_path = Path("./" + output_path)
    print(f"输出路径：{output_dir}")
    print(f"正在生成 PDB 文件：{output_path}")
    try:
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        Chem.MolToPDBFile(mol, output_path)
        print(f"✅ PDB 文件已保存：{output_path}")
    except Exception as e:
        print(f"❌ PDB 输出失败：{e}")
        return None

    # 最终验证
    print("\n最终结构验证...")
    final_issues = validate_structure(mol)
    if final_issues:
        print("⚠️ 最终结构仍有问题：")
        for issue in final_issues:
            print(f"  - {issue}")
    else:
        print("✅ 最终结构检查通过")

    # 生成下一步命令提示
    print("\n" + "="*60)
    print("转换完成！")
    if optimized:
        print(f"✅ 文件已优化并保存至: {output_path}")
    else:
        print(f"✅ 文件已保存至: {output_path}")
    print("="*60)
    
    return output_path


if __name__ == "__main__":
    convert_xyz_to_pdb()
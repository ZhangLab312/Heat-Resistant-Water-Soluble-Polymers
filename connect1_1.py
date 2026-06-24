import re
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
import numpy as np
import itertools
import os


def replace_star_symbol(smiles):
    """将[n*]替换为[*]"""
    return re.sub(r'\[\d+\*\]', '[*]', smiles)


def read_fragments(file_path):
    """读取碎片文件并替换连接点符号"""
    fragments = []
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过标题行
        for row in reader:
            if row:
                smiles = replace_star_symbol(row[0].strip())
                fragments.append(smiles)
    return fragments


def connect_fragments(frag1, frag2):
    """连接两个碎片（各使用一个连接点）"""
    # 创建可编辑分子对象
    combined = Chem.RWMol()
    atom_map = {}

    # 添加frag1的原子（跳过连接点原子）
    for atom in frag1.GetAtoms():
        if atom.GetSymbol() != '*':
            new_idx = combined.AddAtom(atom)
            atom_map[(0, atom.GetIdx())] = new_idx

    # 添加frag2的原子（跳过连接点原子）
    for atom in frag2.GetAtoms():
        if atom.GetSymbol() != '*':
            new_idx = combined.AddAtom(atom)
            atom_map[(1, atom.GetIdx())] = new_idx

    # 添加frag1的键（跳过连接点相关键）
    for bond in frag1.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetSymbol() != '*' and a2.GetSymbol() != '*':
            combined.AddBond(
                atom_map[(0, a1.GetIdx())],
                atom_map[(0, a2.GetIdx())],
                bond.GetBondType()
            )

    # 添加frag2的键（跳过连接点相关键）
    for bond in frag2.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetSymbol() != '*' and a2.GetSymbol() != '*':
            combined.AddBond(
                atom_map[(1, a1.GetIdx())],
                atom_map[(1, a2.GetIdx())],
                bond.GetBondType()
            )

    # 获取连接点相邻的原子
    def get_star_neighbor(frag):
        for atom in frag.GetAtoms():
            if atom.GetSymbol() == '*':
                neighbor = atom.GetNeighbors()[0]
                bond_type = frag.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
                return neighbor.GetIdx(), bond_type
        return None, None

    # 连接两个碎片
    idx1, bond_type1 = get_star_neighbor(frag1)
    idx2, bond_type2 = get_star_neighbor(frag2)

    if idx1 is not None and idx2 is not None:
        # 使用单键连接
        combined.AddBond(
            atom_map[(0, idx1)],
            atom_map[(1, idx2)],
            Chem.BondType.SINGLE
        )

    return combined.GetMol()


def add_carbon_double_bond(mol, cc_fragment):
    """添加碳碳双键碎片来连接两个连接点"""
    rw_mol = Chem.RWMol(mol)

    # 获取分子中的连接点
    star_atoms = [atom for atom in rw_mol.GetAtoms() if atom.GetSymbol() == '*']
    if len(star_atoms) < 2:
        return mol

    # 选择前两个连接点
    star1, star2 = star_atoms[:2]

    # 获取连接点的邻居和键型
    def get_star_info(atom):
        neighbor = atom.GetNeighbors()[0]
        bond_type = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
        return neighbor.GetIdx(), bond_type

    idx1, bond_type1 = get_star_info(star1)
    idx2, bond_type2 = get_star_info(star2)

    # 移除连接点原子
    rw_mol.RemoveAtom(star2.GetIdx())
    rw_mol.RemoveAtom(star1.GetIdx())

    # 添加碳碳双键碎片（跳过连接点）
    atom_map = {}
    for atom in cc_fragment.GetAtoms():
        if atom.GetSymbol() != '*':
            new_idx = rw_mol.AddAtom(atom)
            atom_map[atom.GetIdx()] = new_idx

    # 添加碳碳双键碎片内部的键
    for bond in cc_fragment.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetSymbol() != '*' and a2.GetSymbol() != '*':
            rw_mol.AddBond(
                atom_map[a1.GetIdx()],
                atom_map[a2.GetIdx()],
                bond.GetBondType()
            )

    # 获取碳碳双键碎片中连接点的邻居
    def get_cc_star_neighbor(cc_frag):
        neighbors = []
        for atom in cc_frag.GetAtoms():
            if atom.GetSymbol() == '*':
                neighbor = atom.GetNeighbors()[0]
                neighbors.append(neighbor.GetIdx())
        return neighbors

    cc_neighbors = get_cc_star_neighbor(cc_fragment)

    # 连接分子与碳碳双键碎片
    rw_mol.AddBond(idx1, atom_map[cc_neighbors[0]], Chem.BondType.SINGLE)
    rw_mol.AddBond(idx2, atom_map[cc_neighbors[1]], Chem.BondType.SINGLE)

    return rw_mol.GetMol()


def main():
    # 输入文件路径
    file1 = r'E:\Python\pythonProject\new_t_predict\data\fragment\t_fragment_1.csv'
    file2 = r'E:\Python\pythonProject\new_t_predict\data\fragment\水溶性聚合物_fragment_1.csv'
    output_file = r'E:\Python\pythonProject\new_t_predict\data\result1_1.csv'

    # 读取碎片
    frag1_list = read_fragments(file1)
    frag2_list = read_fragments(file2)

    # 创建碳碳双键碎片
    cc_frag = Chem.MolFromSmiles('[*]C=C[*]', sanitize=False)

    # 打开输出文件
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['SMILES'])

        # 遍历所有碎片组合
        for i, smi1 in enumerate(frag1_list):
            mol1 = Chem.MolFromSmiles(smi1, sanitize=False)
            if mol1 is None: continue

            for j, smi2 in enumerate(frag2_list):
                mol2 = Chem.MolFromSmiles(smi2, sanitize=False)
                if mol2 is None: continue

                try:
                    # 第一步：连接两个碎片
                    polymer = connect_fragments(mol1, mol2)
                    if polymer is None: continue

                    # 第二步：添加碳碳双键直到没有连接点
                    while True:
                        # 计算当前连接点数量
                        star_count = sum(1 for atom in polymer.GetAtoms() if atom.GetSymbol() == '*')

                        if star_count == 0:
                            # 没有连接点，合成完成
                            break
                        elif star_count == 1:
                            # 单个连接点无法处理，跳过
                            polymer = None
                            break
                        else:
                            # 添加碳碳双键碎片
                            polymer = add_carbon_double_bond(polymer, cc_frag)
                            if polymer is None:
                                break

                    if polymer is not None:
                        # 清理分子并转换为SMILES
                        try:
                            Chem.SanitizeMol(polymer)
                            smiles = Chem.MolToSmiles(polymer)
                            writer.writerow([smiles])
                        except:
                            continue

                except Exception as e:
                    print(f"Error processing ({i},{j}): {str(e)}")
                    continue


if __name__ == "__main__":
    main()
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sys


def contains_si(mol):
    """检查分子是否含有Si原子"""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'Si':
            return True
    return False


def count_benzene_rings(mol):
    """计算分子中的苯环数量"""
    # 获取所有环
    rings = mol.GetRingInfo().AtomRings()
    benzene_rings = 0

    for ring in rings:
        # 检查是否为6元环
        if len(ring) == 6:
            # 检查是否所有原子都是碳且芳香
            all_carbon_aromatic = True
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() != 'C' or not atom.GetIsAromatic():
                    all_carbon_aromatic = False
                    break

            if all_carbon_aromatic:
                benzene_rings += 1

    return benzene_rings


def filter_molecules(input_file, output_file):
    """过滤分子：去除含Si原子和超过一个苯环的分子"""
    # 读取输入文件
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        # 读取并写入标题行
        header = next(f_in)
        f_out.write(header)

        # 处理每一行
        processed = 0
        kept = 0
        for line in f_in:
            processed += 1
            if processed % 10000 == 0:
                print(f"已处理 {processed} 行，保留 {kept} 个分子")

            parts = line.strip().split(',')
            if len(parts) < 3:
                continue

            smiles = parts[0].strip()

            # 从SMILES创建分子对象
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # 检查是否含有Si原子
            if contains_si(mol):
                continue

            # 检查苯环数量
            benzene_count = count_benzene_rings(mol)
            if benzene_count > 1:
                continue

            # 通过所有检查，写入输出文件
            f_out.write(line)
            kept += 1

    print(f"处理完成！共处理 {processed} 个分子，保留 {kept} 个分子")


if __name__ == '__main__':
    # 请在此处填写您的输入文件和输出文件路径
    input_file = "E:\\Python\\pythonProject\\new_t_predict\\data\\600_result.csv"  # 输入文件路径
    output_file = "E:\\Python\\pythonProject\\new_t_predict\\data\\result_nosicc.csv"  # 输出文件路径

    filter_molecules(input_file, output_file)
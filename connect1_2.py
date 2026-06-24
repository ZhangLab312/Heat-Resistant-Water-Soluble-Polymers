import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import itertools
import sys

# 禁用RDKit的所有日志输出（包括错误和警告）
RDLogger.DisableLog('rdApp.*')


def standardize_dummy(smiles):
    """标准化所有连接点格式"""
    standardized = smiles
    for i in range(1, 17):
        standardized = standardized.replace(f'[{i}*]', '[*]')
    return standardized


def get_dummy_indices(mol):
    """获取分子中所有连接点原子的索引"""
    if mol is None:
        return []
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']


def connect_fragments(mol1, mol2, idx1, idx2):
    """连接两个分子在指定的连接点位置"""
    if mol1 is None or mol2 is None:
        return None

    try:
        # 创建可编辑分子对象
        combined = Chem.RWMol(Chem.CombineMols(mol1, mol2))

        # 获取分子2的原子偏移量
        offset = mol1.GetNumAtoms()

        # 获取连接点相邻的原子
        atom1 = mol1.GetAtomWithIdx(idx1)
        neighbors1 = list(atom1.GetNeighbors())
        if not neighbors1:
            return None
        neighbor1_idx = neighbors1[0].GetIdx()

        atom2 = mol2.GetAtomWithIdx(idx2)
        neighbors2 = list(atom2.GetNeighbors())
        if not neighbors2:
            return None
        neighbor2_idx = neighbors2[0].GetIdx() + offset

        # 添加新键
        combined.AddBond(neighbor1_idx, neighbor2_idx, Chem.BondType.SINGLE)

        # 移除连接点原子
        # 先移除索引较大的原子，避免索引变化问题
        if idx2 + offset > idx1:
            combined.RemoveAtom(idx2 + offset)
            combined.RemoveAtom(idx1)
        else:
            combined.RemoveAtom(idx1)
            combined.RemoveAtom(idx2 + offset)

        # 返回新分子
        return combined.GetMol()
    except Exception as e:
        return None


def saturate_dummies(mol, keep_idx=None):
    """饱和除指定索引外的所有连接点"""
    if mol is None:
        return None

    rw_mol = Chem.RWMol(mol)
    dummies = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == '*']

    # 按索引降序排序，以便安全删除
    dummies.sort(reverse=True)

    for idx in dummies:
        if idx == keep_idx:
            continue
        # 获取连接点的邻居原子
        atom = rw_mol.GetAtomWithIdx(idx)
        neighbors = list(atom.GetNeighbors())
        if neighbors:
            neighbor_idx = neighbors[0].GetIdx()
            # 给邻居原子添加氢原子
            neighbor = rw_mol.GetAtomWithIdx(neighbor_idx)
            neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)
        # 移除连接点原子
        rw_mol.RemoveAtom(idx)

    return rw_mol.GetMol()


def is_valid_molecule(mol):
    """检查分子是否有效（单一结构，无连接点）"""
    if mol is None:
        return False

    try:
        # 检查是否还有连接点
        if any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
            return False

        # 检查分子是否完整（没有分隔符）
        smiles = Chem.MolToSmiles(mol)
        if '.' in smiles:
            return False

        return True
    except:
        return False


def main():
    # 文件路径
    intermediates_file = r'E:\Python\pythonProject\new_t_predict\data\fragment\s_intermediates.csv'
    fragments_file = r'E:\Python\pythonProject\new_t_predict\data\fragment\t_fragment_1.csv'
    output_file = r'E:\Python\pythonProject\new_t_predict\data\result1_2.csv'

    # 定义碳碳双键分子碎片
    vinyl_frag = Chem.MolFromSmiles('[*]C=C')

    # 读取中间体文件
    try:
        df_intermediates = pd.read_csv(intermediates_file)
        print(f"成功读取中间体文件: {intermediates_file} ({len(df_intermediates)}行)")
    except Exception as e:
        print(f"读取中间体文件错误: {str(e)}")
        return

    # 读取碎片文件
    try:
        df_fragments = pd.read_csv(fragments_file)
        print(f"成功读取碎片文件: {fragments_file} ({len(df_fragments)}行)")
    except Exception as e:
        print(f"读取碎片文件错误: {str(e)}")
        return

    # 预处理中间体
    intermediates = []
    for i, row in df_intermediates.iterrows():
        try:
            smiles = row['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
                intermediates.append(mol)
        except:
            continue

    # 预处理碎片
    fragments = []
    for i, row in df_fragments.iterrows():
        try:
            smiles = row['fragment']
            std_smiles = standardize_dummy(smiles)
            mol = Chem.MolFromSmiles(std_smiles)
            if mol is not None and any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
                fragments.append(mol)
        except:
            continue

    print(f"有效中间体数量: {len(intermediates)}")
    print(f"有效碎片数量: {len(fragments)}")

    # 计算总配对数量（用于进度条）
    total_pairs = 0
    for mol1 in intermediates:
        dummies1 = get_dummy_indices(mol1)
        if not dummies1:
            continue
        for mol2 in fragments:
            dummies2 = get_dummy_indices(mol2)
            if not dummies2:
                continue
            total_pairs += len(dummies1) * len(dummies2)

    print(f"开始合成聚合物... (预计 {total_pairs} 个配对)")

    # 打开输出文件
    with open(output_file, 'w') as f_out:
        f_out.write("SMILES\n")  # 写入标题

        # 创建进度条
        progress_bar = tqdm(total=total_pairs, desc="合成聚合物", unit="pair")

        # 遍历所有中间体和碎片
        for mol1 in intermediates:
            dummies1 = get_dummy_indices(mol1)
            if not dummies1:
                continue

            for mol2 in fragments:
                dummies2 = get_dummy_indices(mol2)
                if not dummies2:
                    continue

                # 尝试所有连接点组合
                for idx1 in dummies1:
                    for idx2 in dummies2:
                        # 更新进度
                        progress_bar.update(1)

                        # 1. 连接中间体和碎片
                        intermediate = connect_fragments(mol1, mol2, idx1, idx2)
                        if intermediate is None:
                            continue

                        # 2. 获取中间体的剩余连接点
                        dummies_inter = get_dummy_indices(intermediate)
                        if not dummies_inter:
                            continue

                        # 3. 尝试每个剩余连接点连接碳碳双键
                        for dummy_idx in dummies_inter:
                            # 3.1 饱和其他连接点
                            saturated = saturate_dummies(intermediate, keep_idx=dummy_idx)
                            if saturated is None:
                                continue

                            # 3.2 连接碳碳双键碎片
                            final_mol = connect_fragments(saturated, vinyl_frag, dummy_idx, 0)
                            if final_mol is None:
                                continue

                            # 3.3 饱和所有剩余连接点
                            final_mol = saturate_dummies(final_mol)
                            if final_mol is None:
                                continue

                            # 3.4 检查最终分子有效性
                            if is_valid_molecule(final_mol):
                                try:
                                    # 获取SMILES并写入文件
                                    smiles = Chem.MolToSmiles(final_mol)
                                    f_out.write(smiles + "\n")
                                except:
                                    continue

        # 关闭进度条
        progress_bar.close()

    print(f"合成完成! 结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
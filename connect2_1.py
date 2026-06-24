import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import itertools
import gc
from multiprocessing import Pool, cpu_count
import time


def standardize_dummy(smiles):
    """标准化所有连接点格式"""
    standardized = smiles
    for i in range(1, 17):
        standardized = standardized.replace(f'[{i}*]', '[*]')
    return standardized


def get_dummy_indices(mol):
    """获取分子中所有连接点原子的索引"""
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
        combined.RemoveAtom(idx2 + offset)  # 先移除分子2的连接点
        combined.RemoveAtom(idx1)  # 再移除分子1的连接点

        # 返回新分子
        return combined.GetMol()
    except Exception as e:
        return None


def saturate_dummies(mol, keep_idx=None):
    """饱和除指定索引外的所有连接点"""
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
            current_hs = neighbor.GetNumExplicitHs()
            neighbor.SetNumExplicitHs(current_hs + 1)
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


def process_fragment_batch(args):
    """处理单个中间体与所有碎片的连接任务"""
    mol1, fragments, vinyl_frag, output_path, batch_id = args
    batch_results = []

    dummies1 = get_dummy_indices(mol1)
    if not dummies1:
        return batch_results

    for mol2 in fragments:
        dummies2 = get_dummy_indices(mol2)
        if not dummies2:
            continue

        for idx1 in dummies1:
            for idx2 in dummies2:
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
                            smiles = Chem.MolToSmiles(final_mol)
                            batch_results.append(smiles)
                        except:
                            continue

    # 保存本批次结果到临时文件
    if batch_results:
        temp_file = f"{output_path}_temp_{batch_id}.csv"
        with open(temp_file, 'a') as f_temp:
            for smiles in batch_results:
                f_temp.write(smiles + "\n")

    return len(batch_results)


def main():
    # 文件路径
    intermediates_file = r'E:\Python\pythonProject\new_t_predict\data\fragment\t_intermediates.csv'
    fragments_file = r'E:\Python\pythonProject\new_t_predict\data\fragment\水溶性聚合物_fragment_1.csv'
    output_file = r'E:\Python\pythonProject\new_t_predict\data\result2_1.csv'

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 清空或创建输出文件
    open(output_file, 'w').close()

    # 定义碳碳双键分子碎片
    vinyl_frag = Chem.MolFromSmiles('[*]C=C')

    print("预处理碎片数据...")
    # 预处理碎片（一次性加载，通常碎片数量较少）
    fragments = []
    try:
        df_fragments = pd.read_csv(fragments_file)
        for i, row in df_fragments.iterrows():
            try:
                smiles = row['fragment']
                std_smiles = standardize_dummy(smiles)
                mol = Chem.MolFromSmiles(std_smiles)
                if mol is not None and any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
                    fragments.append(mol)
            except:
                continue
        print(f"加载有效碎片: {len(fragments)}")
    except Exception as e:
        print(f"读取碎片文件错误: {str(e)}")
        return

    # 分块处理中间体
    chunk_size = 10000  # 每批处理的中间体数量
    processed_count = 0
    total_intermediates = 0

    # 获取中间体总数（用于进度条）
    try:
        with open(intermediates_file, 'r') as f:
            total_intermediates = sum(1 for _ in f) - 1  # 减去标题行
        print(f"中间体总数: {total_intermediates}")
    except:
        print("无法确定中间体总数，进度条可能不准确")

    # 创建进度条
    progress_bar = tqdm(total=total_intermediates, desc="处理中间体", unit="mol")

    # 准备多进程池
    num_cores = max(1, cpu_count() - 1)  # 保留一个核心给系统
    pool = Pool(processes=num_cores)
    print(f"使用 {num_cores} 个CPU核心进行并行计算")

    # 分批读取和处理中间体
    batch_id = 0
    for chunk in pd.read_csv(intermediates_file, chunksize=chunk_size):
        batch_tasks = []
        batch_intermediates = []

        # 预处理本批中间体
        for _, row in chunk.iterrows():
            try:
                smiles = row['SMILES']
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
                    batch_intermediates.append(mol)
            except:
                continue

        # 为每个中间体创建处理任务
        for mol1 in batch_intermediates:
            batch_tasks.append((mol1, fragments, vinyl_frag, output_file, batch_id))
            batch_id += 1

        # 并行处理本批任务
        results = pool.imap_unordered(process_fragment_batch, batch_tasks)

        # 更新进度
        for result in results:
            processed_count += 1
            progress_bar.update(1)

        # 手动触发垃圾回收
        del batch_tasks, batch_intermediates
        gc.collect()

    # 关闭多进程池
    pool.close()
    pool.join()
    progress_bar.close()

    print("合并临时文件...")
    # 合并所有临时文件到最终输出
    with open(output_file, 'w') as f_out:
        f_out.write("SMILES\n")  # 写入标题
        for temp_file in glob.glob(f"{output_file}_temp_*.csv"):
            with open(temp_file, 'r') as f_temp:
                shutil.copyfileobj(f_temp, f_out)
            os.remove(temp_file)

    print(f"处理完成! 结果保存到 {output_file}")
    print(f"共处理 {processed_count} 个中间体")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总运行时间: {end_time - start_time:.2f} 秒")
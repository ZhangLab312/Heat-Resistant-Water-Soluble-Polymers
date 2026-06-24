import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import warnings

warnings.filterwarnings('ignore')


def compute_hbd_hba_wrapper(args):
    """
    包装函数，用于多进程调用
    """
    idx, smiles = args
    if pd.isnull(smiles) or smiles == '':
        return idx, None, None, "SMILES为空"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if smiles.startswith('InChI='):
                mol = Chem.MolFromInchi(smiles)
                if mol is None:
                    return idx, None, None, "无法解析SMILES/InChI"
            else:
                return idx, None, None, "无效的SMILES"

        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        return idx, hbd, hba, None

    except Exception:
        return idx, None, None, "计算错误"


def process_chunk_parallel(chunk):
    """
    并行处理一个数据块
    """
    # 准备多进程参数
    args_list = [(i, row['smiles']) for i, row in chunk.iterrows()]

    # 获取CPU核心数，但限制最大进程数
    num_cores = min(multiprocessing.cpu_count(), 8)  # 最多使用8个核心
    print(f"使用 {num_cores} 个进程处理当前块")

    results = []
    with Pool(processes=num_cores) as pool:
        # 使用imap_unordered提高效率
        for result in pool.imap_unordered(compute_hbd_hba_wrapper, args_list, chunksize=100):
            results.append(result)

    # 按原始顺序排序
    results.sort(key=lambda x: x[0])

    # 提取结果
    hbd_list = [r[1] for r in results]
    hba_list = [r[2] for r in results]
    error_list = [r[3] for r in results]

    return hbd_list, hba_list, error_list


def process_large_csv_with_progress(csv_path, output_path=None, chunksize=10000, use_parallel=False):
    """
    处理大规模CSV文件，可选择使用多进程

    参数:
    csv_path: 输入CSV文件路径
    output_path: 输出CSV文件路径（默认为原文件路径添加后缀）
    chunksize: 每次处理的块大小
    use_parallel: 是否使用多进程
    """

    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在: {csv_path}")
        return None

    # 设置输出文件路径
    if output_path is None:
        # 在原文件名后添加"_with_HBD_HBA"
        base_name, ext = os.path.splitext(csv_path)
        output_path = f"{base_name}_with_HBD_HBA{ext}"

    print(f"输入文件: {csv_path}")
    print(f"输出文件: {output_path}")
    print(f"分块大小: {chunksize} 行")
    print(f"使用并行处理: {use_parallel}")

    # 确定编码
    try:
        with open(csv_path, 'rb') as f:
            raw_data = f.read(10000)
            try:
                raw_data.decode('utf-8')
                encoding = 'utf-8'
            except UnicodeDecodeError:
                encoding = 'gbk'
    except:
        encoding = 'utf-8'

    print(f"检测到文件编码: {encoding}")

    # 获取总行数
    try:
        total_lines = sum(1 for _ in open(csv_path, 'r', encoding=encoding))
        print(f"文件总行数: {total_lines}")
    except:
        total_lines = None

    # 准备输出文件
    first_chunk = True
    processed_rows = 0
    successful_count = 0
    failed_count = 0

    print("\n开始处理数据...")

    # 创建进度条
    pbar = tqdm(total=total_lines if total_lines else None, desc="处理进度", unit="行")

    try:
        chunk_iterator = pd.read_csv(csv_path, encoding=encoding, chunksize=chunksize)

        for chunk_idx, chunk in enumerate(chunk_iterator):
            if 'smiles' not in chunk.columns:
                print(f"错误：缺少 'smiles' 列")
                return None

            print(f"\n处理第 {chunk_idx + 1} 个数据块...")

            if use_parallel:
                # 使用多进程处理
                hbd_list, hba_list, error_list = process_chunk_parallel(chunk)
            else:
                # 使用单进程处理
                hbd_list = []
                hba_list = []
                error_list = []

                for idx, row in chunk.iterrows():
                    _, hbd, hba, error = compute_hbd_hba_wrapper((idx, row['smiles']))

                    hbd_list.append(hbd)
                    hba_list.append(hba)
                    error_list.append(error)

                    if error:
                        failed_count += 1
                    else:
                        successful_count += 1

                    processed_rows += 1
                    pbar.update(1)

            # 统计单进程处理的结果
            if not use_parallel:
                # 上面已经统计过了
                pass
            else:
                # 统计多进程处理的结果
                for error in error_list:
                    if error:
                        failed_count += 1
                    else:
                        successful_count += 1
                processed_rows += len(chunk)
                pbar.update(len(chunk))

            # 将计算结果添加到原数据块中
            chunk['HBD'] = hbd_list
            chunk['HBA'] = hba_list
            chunk['Herror'] = error_list

            # 写入文件
            if first_chunk:
                chunk.to_csv(output_path, index=False, encoding='utf-8')
                first_chunk = False
            else:
                chunk.to_csv(output_path, mode='a', index=False, encoding='utf-8', header=False)

    except Exception as e:
        print(f"\n处理过程中发生错误: {str(e)}")
        return None
    finally:
        pbar.close()

    print(f"\n处理完成!")
    print(f"总共处理: {processed_rows} 行")
    print(f"成功计算: {successful_count} 行")
    print(f"失败: {failed_count} 行")
    print(f"结果已保存到: {output_path}")

    return True


# 主程序
if __name__ == "__main__":
    # 设置文件路径
    input_csv = r"E:\Python\pythonProject\new_t_predict\data\原始数据.csv"

    # 检查文件是否存在
    if not os.path.exists(input_csv):
        print(f"错误: 文件不存在: {input_csv}")
        print("请检查文件路径是否正确。")
    else:
        # 运行处理函数
        # 注意：在Windows上，多进程必须在if __name__ == '__main__'中运行
        success = process_large_csv_with_progress(
            csv_path=input_csv,
            output_path="../data/原始数据.csv",  # 可以指定输出路径
            chunksize=10000,
            use_parallel=True  # 设置为True使用多进程，False使用单进程
        )

        if success:
            print("\n程序完成!")
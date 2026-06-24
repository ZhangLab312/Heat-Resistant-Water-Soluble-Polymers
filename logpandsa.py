import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
import logging
import os
import sys
import multiprocessing as mp
from tqdm import tqdm
import time
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_properties(smiles):
    """
    计算分子的LogP和SA_Score

    参数:
    smiles: SMILES字符串

    返回:
    (logp, sa_score): LogP值和SA_Score值，如果计算失败则返回(None, None)
    """
    if not smiles or pd.isna(smiles) or str(smiles).strip() == '':
        return None, None

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, None

        # 计算LogP
        logp = Crippen.MolLogP(mol)

        # 计算SA_Score (使用简单的合成可及性评分)
        # 这里使用分子量作为简单的SA_Score代理，实际中可以使用更复杂的算法
        # 如果需要更精确的SA_Score，可以安装并使用专门的库
        sa_score = Descriptors.MolWt(mol) / 100  # 简化版本

        return logp, sa_score
    except Exception as e:
        logger.debug(f"Error calculating properties for {smiles}: {str(e)}")
        return None, None


def process_chunk(args):
    """
    处理一个数据块，计算LogP和SA_Score

    参数:
    args: (chunk_data, chunk_id, total_chunks)

    返回:
    result: 处理结果，包含计算后的数据和统计信息
    """
    chunk_data, chunk_id, total_chunks = args
    processed_data = []
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'logp_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0
        },
        'sa_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0
        }
    }

    for idx, row in chunk_data.iterrows():
        stats['total'] += 1

        smiles = row['smiles']
        logp, sa_score = calculate_properties(smiles)

        # 创建新行数据
        new_row = {
            'polymer_name': row['polymer_name'],
            'Median': row['Median'],
            'smiles': smiles,
            'Predicted_Median': row['Predicted_Median'],
            'LogP': logp,
            'SA_Score': sa_score
        }

        if logp is not None and sa_score is not None:
            stats['success'] += 1
            # 更新LogP统计
            stats['logp_stats']['sum'] += logp
            stats['logp_stats']['min'] = min(stats['logp_stats']['min'], logp)
            stats['logp_stats']['max'] = max(stats['logp_stats']['max'], logp)

            # 更新SA_Score统计
            stats['sa_stats']['sum'] += sa_score
            stats['sa_stats']['min'] = min(stats['sa_stats']['min'], sa_score)
            stats['sa_stats']['max'] = max(stats['sa_stats']['max'], sa_score)
        else:
            stats['failed'] += 1

        processed_data.append(new_row)

    logger.info(
        f"Chunk {chunk_id + 1}/{total_chunks} processed: {stats['total']} rows, {stats['success']} successful, {stats['failed']} failed")

    return {
        'data': processed_data,
        'stats': stats
    }


def count_lines(file_path):
    """高效计算大文件的行数"""
    count = 0
    with open(file_path, 'rb') as f:
        for line in f:
            count += 1
    return count


def add_properties_parallel(input_file, output_file, num_processes=None, chunksize=10000):
    """
    使用多进程计算分子的LogP和SA_Score并添加到文件中

    参数:
    input_file: 输入文件路径
    output_file: 输出文件路径
    num_processes: 使用的进程数，默认为CPU核心数
    chunksize: 每个进程处理的数据块大小
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return 0, 0, {}, {}

    # 获取文件总行数
    logger.info("Counting total rows...")
    total_rows = count_lines(input_file) - 1  # 减去标题行
    logger.info(f"Total rows to process: {total_rows:,}")

    # 设置进程数
    if num_processes is None:
        num_processes = mp.cpu_count()
    logger.info(f"Using {num_processes} processes")
    logger.info("Calculating LogP and SA_Score for all molecules")

    # 读取整个文件
    logger.info("Reading input file...")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    logger.info(f"Loaded {total_rows:,} rows into memory")

    # 分割数据为多个块
    chunk_size = max(1, total_rows // num_processes)
    chunks = []
    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        chunks.append((chunk, len(chunks), len(chunks)))

    logger.info(f"Split data into {len(chunks)} chunks")

    # 使用多进程处理
    start_time = time.time()
    all_processed_data = []
    all_stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'logp_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0
        },
        'sa_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0
        }
    }

    with mp.Pool(processes=num_processes) as pool:
        # 使用imap_unordered获取结果，同时显示进度
        results = list(tqdm(
            pool.imap_unordered(process_chunk, chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))

        # 合并结果
        for result in results:
            all_processed_data.extend(result['data'])

            # 合并统计信息
            stats = result['stats']
            all_stats['total'] += stats['total']
            all_stats['success'] += stats['success']
            all_stats['failed'] += stats['failed']

            # 合并LogP统计
            all_stats['logp_stats']['sum'] += stats['logp_stats']['sum']
            all_stats['logp_stats']['min'] = min(all_stats['logp_stats']['min'], stats['logp_stats']['min'])
            all_stats['logp_stats']['max'] = max(all_stats['logp_stats']['max'], stats['logp_stats']['max'])

            # 合并SA_Score统计
            all_stats['sa_stats']['sum'] += stats['sa_stats']['sum']
            all_stats['sa_stats']['min'] = min(all_stats['sa_stats']['min'], stats['sa_stats']['min'])
            all_stats['sa_stats']['max'] = max(all_stats['sa_stats']['max'], stats['sa_stats']['max'])

    # 计算平均LogP和SA_Score
    if all_stats['success'] > 0:
        avg_logp = all_stats['logp_stats']['sum'] / all_stats['success']
        avg_sa = all_stats['sa_stats']['sum'] / all_stats['success']
    else:
        avg_logp = 0
        avg_sa = 0

    # 保存处理后的数据
    logger.info("Saving processed data with LogP and SA_Score...")
    output_df = pd.DataFrame(all_processed_data)
    output_columns = ['polymer_name', 'Median', 'smiles', 'Predicted_Median', 'LogP', 'SA_Score']
    output_df[output_columns].to_csv(output_file, index=False)

    # 最终统计
    elapsed_time = time.time() - start_time
    processing_rate = all_stats['total'] / elapsed_time if elapsed_time > 0 else 0

    success_percentage = all_stats['success'] / all_stats['total'] * 100 if all_stats['total'] > 0 else 0
    failed_percentage = all_stats['failed'] / all_stats['total'] * 100 if all_stats['total'] > 0 else 0

    logger.info(f"Processing completed in {elapsed_time:.2f} seconds!")
    logger.info(f"Processing rate: {processing_rate:.2f} molecules/second")
    logger.info(f"Total molecules: {all_stats['total']:,}")
    logger.info(f"Successful calculations: {all_stats['success']:,} ({success_percentage:.2f}%)")
    logger.info(f"Failed calculations: {all_stats['failed']:,} ({failed_percentage:.2f}%)")

    if all_stats['success'] > 0:
        logger.info(f"LogP statistics:")
        logger.info(f"  Minimum LogP: {all_stats['logp_stats']['min']:.4f}")
        logger.info(f"  Maximum LogP: {all_stats['logp_stats']['max']:.4f}")
        logger.info(f"  Average LogP: {avg_logp:.4f}")

        logger.info(f"SA_Score statistics:")
        logger.info(f"  Minimum SA_Score: {all_stats['sa_stats']['min']:.4f}")
        logger.info(f"  Maximum SA_Score: {all_stats['sa_stats']['max']:.4f}")
        logger.info(f"  Average SA_Score: {avg_sa:.4f}")

    # 保存统计信息
    stats_file = output_file.replace('.csv', '_properties_stats.csv')

    # 准备统计DataFrame
    stats_data = []
    stats_data.append(['Total molecules', all_stats['total']])
    stats_data.append(['Successful calculations', all_stats['success']])
    stats_data.append(['Success percentage', f"{success_percentage:.2f}%"])
    stats_data.append(['Failed calculations', all_stats['failed']])
    stats_data.append(['Failed percentage', f"{failed_percentage:.2f}%"])
    stats_data.append(['Processing time (seconds)', f"{elapsed_time:.2f}"])
    stats_data.append(['Processing rate (molecules/second)', f"{processing_rate:.2f}"])

    if all_stats['success'] > 0:
        stats_data.append(['Minimum LogP', f"{all_stats['logp_stats']['min']:.4f}"])
        stats_data.append(['Maximum LogP', f"{all_stats['logp_stats']['max']:.4f}"])
        stats_data.append(['Average LogP', f"{avg_logp:.4f}"])
        stats_data.append(['Minimum SA_Score', f"{all_stats['sa_stats']['min']:.4f}"])
        stats_data.append(['Maximum SA_Score', f"{all_stats['sa_stats']['max']:.4f}"])
        stats_data.append(['Average SA_Score', f"{avg_sa:.4f}"])

    stats_df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])
    stats_df.to_csv(stats_file, index=False)

    return all_stats['success'], all_stats['failed'], all_stats['logp_stats'], all_stats['sa_stats']


if __name__ == "__main__":
    # 输入参数 - 请根据你的实际情况修改这些路径
    input_file = "E:\\Python\\pythonProject\\new_t_predict\\data\\合理分子2.csv"  # 包含聚合物数据的文件
    output_file = "E:\\Python\\pythonProject\\new_t_predict\\data\\合理分子3.csv"  # 包含LogP和SA_Score的输出文件

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        print("请检查文件路径是否正确")
        sys.exit(1)

    # 运行计算
    print(f"开始计算分子LogP和SA_Score的文件: {input_file}")
    print(f"输出将保存到: {output_file}")

    # 使用所有可用的CPU核心
    num_processes = mp.cpu_count()
    print(f"使用 {num_processes} 个进程进行并行计算")

    success_count, failed_count, logp_stats, sa_stats = add_properties_parallel(
        input_file=input_file,
        output_file=output_file,
        num_processes=num_processes,
        chunksize=10000  # 每个进程处理的块大小
    )

    print(f"\n计算完成!")
    print(f"成功计算LogP和SA_Score的分子数量: {success_count:,}")
    print(f"计算失败的分子数量: {failed_count:,}")

    if success_count > 0:
        print(f"\nLogP统计:")
        print(f"  最小LogP: {logp_stats['min']:.4f}")
        print(f"  最大LogP: {logp_stats['max']:.4f}")
        print(f"  平均LogP: {logp_stats['sum'] / success_count:.4f}")

        print(f"\nSA_Score统计:")
        print(f"  最小SA_Score: {sa_stats['min']:.4f}")
        print(f"  最大SA_Score: {sa_stats['max']:.4f}")
        print(f"  平均SA_Score: {sa_stats['sum'] / success_count:.4f}")
import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')
import multiprocessing as mp


def calculate_chunk_stats(args):
    """计算单个数据块的统计量"""
    chunk_idx, chunk = args

    # 计算特征
    chunk['-LogP'] = -chunk['LogP']
    chunk['HBD+HBA'] = chunk['HBD'] + chunk['HBA']

    # 计算统计量
    stats = {}
    for col, stat_name in [('-LogP', '-LogP'), ('HBD+HBA', 'HBD+HBA'), ('predicted_logS', 'logSmonomer')]:
        valid_data = chunk[col].dropna()
        if len(valid_data) > 0:
            mean = valid_data.mean()
            std = valid_data.std()
            count = len(valid_data)
            sum_val = valid_data.sum()
            sum_sq = (valid_data ** 2).sum()
        else:
            mean = std = sum_val = sum_sq = 0
            count = 0

        stats[stat_name] = {
            'sum': sum_val,
            'sum_sq': sum_sq,
            'count': count,
            'mean': mean,
            'std': std
        }

    return stats, len(chunk)


def calculate_chunk_stotal(args):
    """计算单个数据块的Stotal"""
    chunk_idx, chunk, stats, weights = args

    # 保存原始列顺序
    original_columns = chunk.columns.tolist()

    # 计算特征
    chunk['-LogP'] = -chunk['LogP']
    chunk['HBD+HBA'] = chunk['HBD'] + chunk['HBA']

    # 标准化特征
    chunk['z_-LogP'] = (chunk['-LogP'] - stats['-LogP']['mean']) / stats['-LogP']['std']
    chunk['z_HBD+HBA'] = (chunk['HBD+HBA'] - stats['HBD+HBA']['mean']) / stats['HBD+HBA']['std']
    chunk['z_logSmonomer'] = (chunk['predicted_logS'] - stats['logSmonomer']['mean']) / stats['logSmonomer']['std']

    # 计算Stotal
    chunk['Stotal'] = (
            weights['w1'] * chunk['z_-LogP'] +
            weights['w2'] * chunk['z_HBD+HBA'] +
            weights['w3'] * chunk['z_logSmonomer'] +
            weights['intercept']
    )

    # 计算预测概率
    chunk['预测概率'] = 1 / (1 + np.exp(-chunk['Stotal']))

    # 移除中间列
    chunk = chunk.drop(['-LogP', 'HBD+HBA', 'z_-LogP', 'z_HBD+HBA', 'z_logSmonomer'], axis=1, errors='ignore')

    # 恢复原始列顺序，并将新列添加在最后
    final_columns = original_columns + ['Stotal', '预测概率']
    chunk = chunk.reindex(columns=final_columns)

    return chunk_idx, chunk


def calculate_stotal_parallel(csv_path, output_path, chunk_size=500000, n_workers=None):
    """
    并行计算大数据集的Stotal
    """

    # 给定的权重参数
    weights = {
        'w1': 0.0124,  # -LogP权重
        'w2': 0.2086,  # HBD+HBA权重
        'w3': 2.4509,  # logSmonomer权重
        'intercept': -0.4345  # 截距项
    }

    print(f"开始处理文件: {csv_path}")
    print(f"输出文件: {output_path}")
    print(f"使用并行处理，工作进程数: {n_workers or mp.cpu_count()}")

    # 第一步：并行计算统计量
    print("\n第一步: 并行计算特征的均值和标准差...")

    # 确定工作进程数
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # 最多使用8个进程

    total_rows = 0
    all_stats = []

    # 检查文件中的必需列
    print("检查输入文件列...")
    sample_df = pd.read_csv(csv_path, nrows=5)
    required_columns = ['LogP', 'HBD', 'HBA', 'predicted_logS']
    missing_columns = [col for col in required_columns if col not in sample_df.columns]

    if missing_columns:
        print(f"错误: 输入文件缺少以下必需列: {missing_columns}")
        print(f"文件中的列: {list(sample_df.columns)}")
        return None

    print(f"找到必需列: {required_columns}")

    # 使用多进程计算统计量
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        chunk_idx = 0

        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, encoding='utf-8'):
            futures.append(executor.submit(calculate_chunk_stats, (chunk_idx, chunk)))
            chunk_idx += 1
            total_rows += len(chunk)

        # 收集结果
        for future in futures:
            stats, rows = future.result()
            all_stats.append(stats)

    # 合并统计量
    merged_stats = {}
    for feature in ['-LogP', 'HBD+HBA', 'logSmonomer']:
        total_sum = sum(s[feature]['sum'] for s in all_stats)
        total_sum_sq = sum(s[feature]['sum_sq'] for s in all_stats)
        total_count = sum(s[feature]['count'] for s in all_stats)

        if total_count > 0:
            mean = total_sum / total_count
            variance = (total_sum_sq / total_count) - (mean ** 2)
            std = np.sqrt(variance) if variance > 0 else 1.0
        else:
            mean = 0
            std = 1.0

        merged_stats[feature] = {'mean': mean, 'std': std}

    print(f"\n统计完成，共处理 {len(all_stats)} 块数据")
    print("标准化参数:")
    for feature, stat in merged_stats.items():
        print(f"  {feature}: 均值={stat['mean']:.6f}, 标准差={stat['std']:.6f}")

    # 第二步：并行计算Stotal
    print("\n第二步: 并行计算Stotal...")

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 用于排序的列表
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        chunk_idx = 0

        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, encoding='utf-8'):
            future = executor.submit(calculate_chunk_stotal,
                                     (chunk_idx, chunk, merged_stats, weights))
            futures[future] = chunk_idx
            chunk_idx += 1

        # 按顺序收集结果
        processed_futures = []
        for future in futures:
            processed_futures.append((futures[future], future))

        # 按块索引排序
        processed_futures.sort(key=lambda x: x[0])

        # 写入文件
        first_chunk = True
        for idx, future in processed_futures:
            chunk_idx, chunk_result = future.result()

            if first_chunk:
                chunk_result.to_csv(output_path, index=False, encoding='utf-8')
                first_chunk = False
            else:
                chunk_result.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8')

            results.append(chunk_result)

    print(f"\n处理完成! 结果已保存到: {output_path}")

    return {
        'stats': merged_stats,
        'weights': weights,
        'total_chunks': len(results),
        'total_rows': total_rows
    }


# 主程序
if __name__ == "__main__":
    # 文件路径
    input_csv = r"E:\Python\pythonProject\new_t_predict\data\合理分子3.csv"
    output_csv = r"E:\Python\pythonProject\new_t_predict\data\合理分子f.csv"

    # 检查文件是否存在
    if not os.path.exists(input_csv):
        print(f"错误: 文件不存在: {input_csv}")
    else:
        print("=" * 60)
        print("大数据集并行Stotal计算开始")
        print("=" * 60)

        # 设置参数
        chunk_size = 500000  # 50万条/块
        n_workers = 4  # 使用4个进程

        results = calculate_stotal_parallel(
            csv_path=input_csv,
            output_path=output_csv,
            chunk_size=chunk_size,
            n_workers=n_workers
        )

        if results is not None:
            print("\n" + "=" * 60)
            print("计算完成!")
            print("=" * 60)
            print(f"总处理行数: {results['total_rows']:,}")
            print(f"处理数据块数: {results['total_chunks']}")

            # 显示最终公式
            print("\n使用的公式:")
            print(f"Stotal = {results['weights']['w1']} * z(-LogP) + "
                  f"{results['weights']['w2']} * z(HBD+HBA) + "
                  f"{results['weights']['w3']} * z(logSmonomer) + "
                  f"{results['weights']['intercept']}")
            print("\n标准化参数:")
            print(
                f"  -LogP: 均值={results['stats']['-LogP']['mean']:.6f}, 标准差={results['stats']['-LogP']['std']:.6f}")
            print(
                f"  HBD+HBA: 均值={results['stats']['HBD+HBA']['mean']:.6f}, 标准差={results['stats']['HBD+HBA']['std']:.6f}")
            print(
                f"  logSmonomer: 均值={results['stats']['logSmonomer']['mean']:.6f}, 标准差={results['stats']['logSmonomer']['std']:.6f}")
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import concurrent.futures
import time
import os

# 文件路径
input_file = r"E:\Python\pythonProject\new_t_predict\data\水溶性聚合物.csv"
output_file = r"E:\Python\pythonProject\new_t_predict\data\fragment\水溶性聚合物_fragment.csv"

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(output_file), exist_ok=True)


def fragment_molecule(smiles):
    """
    使用BRICS方法对分子进行碎片化
    返回碎片列表（去重后的SMILES字符串）
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # 进行BRICS碎片化
        fragments = list(BRICS.BRICSDecompose(mol))

        # 去除重复碎片并返回
        return list(set(fragments))
    except:
        return []


def process_molecule(row):
    """
    处理单个分子，包含超时控制
    """
    name, smiles = row['Name'], row['SMILES']

    # 使用线程池执行器实现超时控制
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fragment_molecule, smiles)
        try:
            return future.result(timeout=2)  # 2秒超时
        except concurrent.futures.TimeoutError:
            print(f"\n跳过超时分子: {name} | {smiles}")
            return []
        except Exception as e:
            print(f"\n处理错误: {name} | {e}")
            return []


def main():
    # 读取输入文件
    print("正在读取数据...")
    df = pd.read_csv(input_file)
    print(f"找到 {len(df)} 个分子")

    # 准备结果列表
    all_fragments = []

    print("\n开始碎片化处理 (超时: 2秒/分子):")
    total_molecules = len(df)

    # 使用tqdm创建进度条
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_molecules, desc="处理进度")
    except ImportError:
        print("tqdm未安装，使用简单进度显示")
        progress_bar = None

    # 处理每个分子
    for _, row in df.iterrows():
        fragments = process_molecule(row)

        # 添加碎片到结果列表
        for frag in fragments:
            all_fragments.append({'fragment': frag})

        # 更新进度条
        if progress_bar:
            progress_bar.update(1)
        else:
            print(f"已处理 {len(all_fragments)} 个碎片 | 当前分子: {row['Name']}")

    # 关闭进度条
    if progress_bar:
        progress_bar.close()

    # 创建输出DataFrame
    result_df = pd.DataFrame(all_fragments, columns=['fragment'])

    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"\n完成! 共生成 {len(result_df)} 个碎片")
    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"总耗时: {time.time() - start_time:.2f} 秒")
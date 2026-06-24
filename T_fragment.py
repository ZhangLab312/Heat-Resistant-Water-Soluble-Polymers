import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import time
import os
import multiprocessing
from multiprocessing import Pool, TimeoutError
import sys

# 文件路径
input_file = r"E:\Python\pythonProject\new_t_predict\data\cleaned_predictions.csv"
output_file = r"E:\Python\pythonProject\new_t_predict\data\fragment\t_fragment.csv"

# 创建输出目录（如果不存在）
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 全局计数器，用于进度显示
processed_count = 0
total_count = 0
start_time = time.time()


def fragment_molecule_wrapper(args):
    """包装函数，用于多进程处理"""
    idx, row = args
    return (idx, fragment_molecule(row))


def fragment_molecule(row):
    """处理单个分子，返回碎片列表或错误信息"""
    name, smiles = row['polymer_name'], row['smiles']
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (name, smiles, [], "无效的SMILES")

        # 进行BRICS碎片化
        fragments = list(BRICS.BRICSDecompose(mol))
        return (name, smiles, list(set(fragments)), None)
    except Exception as e:
        return (name, smiles, [], f"处理错误: {str(e)}")


def update_progress():
    """更新并显示进度信息"""
    global processed_count, total_count, start_time
    elapsed = time.time() - start_time
    percent = (processed_count / total_count) * 100 if total_count > 0 else 0
    sys.stdout.write(f"\r处理进度: {processed_count}/{total_count} ({percent:.1f}%) | "
                     f"用时: {elapsed:.1f}s | "
                     f"预计剩余: {(elapsed / processed_count) * (total_count - processed_count):.1f}s "
                     if processed_count > 0 else "\r开始处理...")
    sys.stdout.flush()


def main():
    global processed_count, total_count

    # 读取输入文件
    print("正在读取耐高温分子数据...")
    df = pd.read_csv(input_file)
    total_count = len(df)
    print(f"找到 {total_count} 个耐高温分子")

    # 准备结果列表
    all_fragments = []
    skipped_molecules = []
    success_count = 0

    print("\n开始碎片化处理 (超时: 2秒/分子):")

    # 使用多进程池
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []
        # 提交所有任务
        for idx, row in df.iterrows():
            results.append(pool.apply_async(fragment_molecule_wrapper, [(idx, row)]))

        # 处理结果
        for res in results:
            try:
                idx, result = res.get(timeout=2)
                name, smiles, fragments, error = result

                if error:
                    skipped_molecules.append((name, smiles, error))
                elif fragments:
                    for frag in fragments:
                        all_fragments.append({'fragment': frag})
                    success_count += 1

                processed_count += 1
                update_progress()

            except TimeoutError:
                name, smiles = df.iloc[idx]['polymer_name'], df.iloc[idx]['smiles']
                skipped_molecules.append((name, smiles, "处理超时"))
                processed_count += 1
                update_progress()
            except Exception as e:
                name, smiles = df.iloc[idx]['polymer_name'], df.iloc[idx]['smiles']
                skipped_molecules.append((name, smiles, f"未知错误: {str(e)}"))
                processed_count += 1
                update_progress()

    # 创建输出DataFrame
    result_df = pd.DataFrame(all_fragments, columns=['fragment'])

    # 保存结果
    result_df.to_csv(output_file, index=False)

    # 保存跳过的分子信息
    skipped_file = output_file.replace("_fragment.csv", "_skipped.csv")
    skipped_df = pd.DataFrame(skipped_molecules, columns=['polymer_name', 'smiles', 'reason'])
    skipped_df.to_csv(skipped_file, index=False)

    # 打印统计信息
    print("\n\n" + "=" * 60)
    print("碎片化处理完成!")
    print(f"总分子数: {total_count}")
    print(f"成功处理: {success_count}")
    print(f"跳过分子: {len(skipped_molecules)}")
    print(f"生成碎片总数: {len(result_df)}")
    print(f"碎片结果保存至: {output_file}")
    print(f"跳过分子列表保存至: {skipped_file}")

    # 显示前5个跳过的分子
    if skipped_molecules:
        print("\n跳过的分子示例:")
        for i, (name, smiles, reason) in enumerate(skipped_molecules[:5]):
            print(f"{i + 1}. {name[:50]}... | 原因: {reason}")
    print("=" * 60)


if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"总耗时: {total_time / 60:.2f} 分钟")
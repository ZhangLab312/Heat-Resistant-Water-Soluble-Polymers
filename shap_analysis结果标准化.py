import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import warnings

warnings.filterwarnings('ignore')

# 配置参数
CONFIG = {
    "shap_path": "E:/Python/pythonProject/new_t_predict/shap/",
    "output_path": "E:/Python/pythonProject/new_t_predict/feature_analysis/",
    "fingerprint": {
        "radius": 2,
        "n_bits": 1024
    }
}

# 确保输出目录存在
os.makedirs(CONFIG["output_path"], exist_ok=True)


def load_shap_data():
    """加载第一步保存的SHAP数据"""
    print("加载SHAP分析数据...")

    # 加载SHAP数据
    shap_data = np.load(os.path.join(CONFIG["shap_path"], "shap_analysis_data.npz"), allow_pickle=True)

    # 加载特征重要性数据
    importance_df = pd.read_csv(os.path.join(CONFIG["shap_path"], "feature_importance.csv"))

    # 加载有效数据索引
    valid_data_info = pd.read_csv(os.path.join(CONFIG["shap_path"], "valid_data_indices.csv"))

    print(f"SHAP值形状: {shap_data['shap_values'].shape}")
    print(f"特征数量: {len(importance_df)}")
    print(f"有效样本数量: {len(valid_data_info)}")

    return shap_data, importance_df, valid_data_info


def filter_important_features(importance_df, shap_values, threshold_percentile=95):
    """
    第二步：筛选重要特征
    由于所有特征都是摩根指纹半径2，直接按重要性筛选
    """
    print("\n=== 第二步：筛选重要特征 ===")

    # 计算重要性阈值（使用百分位数）
    threshold = np.percentile(importance_df['importance'], threshold_percentile)

    # 筛选重要特征
    important_features = importance_df[importance_df['importance'] >= threshold].copy()

    print(f"使用重要性阈值: {threshold:.6f} (百分位数: {threshold_percentile}%)")
    print(f"重要特征数量: {len(important_features)}")
    print(f"占总特征比例: {len(important_features) / len(importance_df):.2%}")

    # 保存筛选后的重要特征
    important_features.to_csv(
        os.path.join(CONFIG["output_path"], "important_features_filtered.csv"),
        index=False
    )

    # 提取重要特征的SHAP值
    important_indices = important_features['feature_index'].values
    important_shap_values = shap_values[:, important_indices]

    # 保存重要特征的SHAP值
    np.savez_compressed(
        os.path.join(CONFIG["output_path"], "important_shap_values.npz"),
        shap_values=important_shap_values,
        feature_indices=important_indices,
        feature_names=important_features['feature_name'].values,
        importance_scores=important_features['importance'].values
    )

    # 打印前20个最重要特征
    print(f"\n前20个最重要特征:")
    for i, row in important_features.head(20).iterrows():
        print(f"特征 {row['feature_index']:4d} (FP_{row['feature_index']:4d}): 重要性 = {row['importance']:.6f}")

    return important_features, important_shap_values, important_indices


def calculate_feature_percentage(important_features):
    """
    第三步：计算重要特征的百分比占比
    """
    print("\n=== 第三步：计算特征百分比占比 ===")

    # 计算每个特征的重要性占比
    total_importance = important_features['importance'].sum()
    important_features['percentage'] = (important_features['importance'] / total_importance) * 100

    # 按百分比排序
    important_features_sorted = important_features.sort_values('percentage', ascending=False)

    # 保存百分比数据
    important_features_sorted.to_csv(
        os.path.join(CONFIG["output_path"], "feature_percentage_analysis.csv"),
        index=False
    )

    # 打印百分比统计
    print(f"总重要性: {total_importance:.6f}")
    print(f"特征百分比统计:")
    print(f"前5个特征占比: {important_features_sorted['percentage'].head(5).sum():.2f}%")
    print(f"前10个特征占比: {important_features_sorted['percentage'].head(10).sum():.2f}%")
    print(f"前20个特征占比: {important_features_sorted['percentage'].head(20).sum():.2f}%")

    # 打印前10个特征的详细百分比
    print(f"\n前10个特征详细占比:")
    for i, row in important_features_sorted.head(10).iterrows():
        print(f"特征 {row['feature_index']:4d}: {row['percentage']:.4f}%")

    return important_features_sorted


def create_percentage_summary(important_features_sorted):
    """
    创建百分比汇总报告
    """
    print("\n=== 创建百分比汇总报告 ===")

    # 创建汇总统计
    summary_stats = {
        '总特征数': len(important_features_sorted),
        '总重要性占比': 100.0,
        '前5特征占比': important_features_sorted['percentage'].head(5).sum(),
        '前10特征占比': important_features_sorted['percentage'].head(10).sum(),
        '前20特征占比': important_features_sorted['percentage'].head(20).sum(),
        '最大单个特征占比': important_features_sorted['percentage'].max(),
        '最小单个特征占比': important_features_sorted['percentage'].min(),
        '平均特征占比': important_features_sorted['percentage'].mean()
    }

    # 创建汇总DataFrame
    summary_df = pd.DataFrame(list(summary_stats.items()), columns=['指标', '值'])

    # 保存汇总报告
    summary_df.to_csv(
        os.path.join(CONFIG["output_path"], "percentage_summary_report.csv"),
        index=False
    )

    print("百分比汇总报告:")
    for key, value in summary_stats.items():
        if '占比' in key:
            print(f"{key}: {value:.4f}%")
        else:
            print(f"{key}: {value}")

    return summary_df


def main():
    """
    主函数：执行第二步和第三步
    """
    print("开始执行特征筛选和百分比分析...")

    # 第一步：加载SHAP数据
    shap_data, importance_df, valid_data_info = load_shap_data()

    # 第二步：筛选重要特征
    important_features, important_shap_values, important_indices = filter_important_features(
        importance_df,
        shap_data['shap_values'],
        threshold_percentile=95  # 可以调整这个参数
    )

    # 第三步：计算特征百分比占比
    important_features_sorted = calculate_feature_percentage(important_features)

    # 创建百分比汇总报告
    summary_df = create_percentage_summary(important_features_sorted)

    print(f"\n=== 分析完成 ===")
    print(f"所有结果已保存到: {CONFIG['output_path']}")
    print("生成的文件包括:")
    print("- important_features_filtered.csv: 筛选后的重要特征")
    print("- important_shap_values.npz: 重要特征的SHAP值")
    print("- feature_percentage_analysis.csv: 特征百分比分析")
    print("- percentage_summary_report.csv: 百分比汇总报告")

    # 输出一些关键信息
    top_features = important_features_sorted.head(10)
    print(f"\n最重要的10个特征及其占比:")
    for i, row in top_features.iterrows():
        print(f"FP_{row['feature_index']:4d}: {row['percentage']:.4f}%")


if __name__ == "__main__":
    main()
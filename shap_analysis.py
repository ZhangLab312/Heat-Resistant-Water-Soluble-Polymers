import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import shap
import os
import warnings
from tqdm import tqdm  # 🌟 引入标准的进度条库

warnings.filterwarnings('ignore')

# 配置参数
CONFIG = {
    "input_csv": "E:\\Python\\pythonProject\\new_t_predict\\data\\cleaned_predictions_no_si.csv",
    "model_path": "E:/Python/pythonProject/new_t_predict/model/fnn_smiles_noerror_model6.pth",
    "scaler_path": "E:/Python/pythonProject/new_t_predict/scaler/fnn_smiles_noerror_scaler6.pkl",
    "shap_path": "E:/Python/pythonProject/new_t_predict/shap/",
    "fingerprint": {
        "radius": 2,
        "n_bits": 1024
    },
    "nn_params": {
        "hidden_layers": [512, 256],
        "dropout_rate": 0.5
    }
}

# 确保SHAP目录存在
os.makedirs(CONFIG["shap_path"], exist_ok=True)


class FNN(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_size = input_size

        for h_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = h_size

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)


def generate_features_once(smiles_list):
    """生成特征（只尝试一次，不进行数据增强）"""
    features = []
    valid_indices = []
    failed_indices = []

    for idx, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=CONFIG["fingerprint"]["radius"],
                    nBits=CONFIG["fingerprint"]["n_bits"]
                )
                arr = np.zeros((CONFIG["fingerprint"]["n_bits"],), dtype=int)
                ConvertToNumpyArray(fp, arr)
                features.append(arr)
                valid_indices.append(idx)
            else:
                failed_indices.append(idx)
        except Exception as e:
            failed_indices.append(idx)

    print(f"\n[特征生成报告]")
    print(f"总SMILES数量: {len(smiles_list)}")
    print(f"成功生成特征: {len(valid_indices)} ({len(valid_indices) / len(smiles_list):.2%})")
    print(f"生成特征失败: {len(failed_indices)} ({len(failed_indices) / len(smiles_list):.2%})")

    return np.array(features), valid_indices, failed_indices


def load_model_and_scaler():
    """加载已训练的模型和标准化器"""
    print("加载模型和标准化器...")

    # 加载标准化器
    scaler = joblib.load(CONFIG["scaler_path"])
    print(f"标准化器已加载: {CONFIG['scaler_path']}")

    # 确定输入尺寸
    input_size = CONFIG["fingerprint"]["n_bits"]

    # 创建模型实例
    model = FNN(
        input_size=input_size,
        hidden_layers=CONFIG["nn_params"]["hidden_layers"],
        dropout_rate=CONFIG["nn_params"]["dropout_rate"]
    )

    # 加载模型权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.to(device)
    model.eval()
    print(f"模型已加载: {CONFIG['model_path']}")
    print(f"使用设备: {device}")

    return model, scaler, device


def perform_shap_analysis(model, scaler, device, X_data, feature_names):
    """执行SHAP分析并保存所有数据"""
    print("\n=== 开始SHAP分析 ===")

    # 标准化数据
    X_scaled = scaler.transform(X_data)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # 准备背景数据（从训练数据中抽样）
    background_size = min(1000, X_scaled.shape[0])
    background_indices = np.random.choice(X_scaled.shape[0], background_size, replace=False)
    background_data = torch.FloatTensor(X_scaled[background_indices]).to(device)

    print(f"使用 {background_size} 个样本作为背景数据")

    # 创建解释器
    print("创建SHAP解释器...")
    explainer = shap.DeepExplainer(model, background_data)

    # 🌟 核心修改：利用 tqdm 手动为 DeepExplainer 制作逐行计算的进度条
    print("计算SHAP值...")
    shap_values_list = []

    # 逐个样本计算，以便在控制台向你展示进度
    for i in tqdm(range(X_tensor.shape[0]), desc="SHAP计算进度"):
        # 保持原本的 Tensor 形状 [1, 1024]
        single_sample = X_tensor[i:i + 1]
        # 不传 progress_bar 参数，避免报错
        single_shap = explainer.shap_values(single_sample)

        # 剥离可能存在的三维或多余维度，确保是二维形式
        if isinstance(single_shap, list):
            single_shap = single_shap[0]
        if len(single_shap.shape) == 3:
            single_shap = single_shap.reshape(single_shap.shape[0], -1)

        shap_values_list.append(single_shap)

    # 将所有单样本结果在行方向拼回大矩阵 [6130, 1024]
    shap_values_2d = np.vstack(shap_values_list)
    print(f"最终SHAP值矩阵形状: {shap_values_2d.shape}")

    # 保存所有SHAP相关数据
    print("保存SHAP分析数据...")
    shap_data = {
        'shap_values': shap_values_2d,
        'shap_values_original': shap_values_2d,  # 统一为二维以防后续报错
        'feature_names': feature_names,
        'X_data': X_scaled,
        'X_original': X_data,
        'expected_value': explainer.expected_value,
        'background_data': background_data.cpu().numpy(),
        'background_indices': background_indices
    }

    # 保存到文件
    np.savez_compressed(
        os.path.join(CONFIG["shap_path"], "shap_analysis_data.npz"),
        **shap_data
    )

    # 计算特征重要性
    feature_importance = np.abs(shap_values_2d).mean(0)
    if len(feature_importance.shape) > 1:
        feature_importance = feature_importance.flatten()

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature_index': range(len(feature_importance)),
        'importance': feature_importance,
        'feature_name': feature_names
    })

    # 按重要性排序并保存
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(CONFIG["shap_path"], "feature_importance.csv"), index=False)

    print(f"特征重要性数据已保存，共 {len(feature_importance)} 个特征")
    print(f"前10个最重要特征:")
    print(importance_df.head(10))

    # 保存SHAP值的汇总统计
    shap_summary = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values_2d).mean(axis=0),
        'mean_shap': shap_values_2d.mean(axis=0),
        'std_shap': shap_values_2d.std(axis=0)
    })
    shap_summary.to_csv(os.path.join(CONFIG["shap_path"], "shap_summary.csv"), index=False)

    return shap_values_2d, feature_importance, importance_df


def main():
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(CONFIG["input_csv"])
    print(f"数据加载完成，共 {len(df)} 行")

    # 生成特征（只尝试一次）
    X, valid_indices, failed_indices = generate_features_once(df["smiles"])

    # 获取对应的目标值
    y = df.iloc[valid_indices]["Median"].values

    # 创建特征名称
    feature_names = [f"FP_{i}" for i in range(CONFIG["fingerprint"]["n_bits"])]

    # 加载模型和标准化器
    model, scaler, device = load_model_and_scaler()

    # 执行SHAP分析
    shap_values, feature_importance, importance_df = perform_shap_analysis(
        model, scaler, device, X, feature_names
    )

    # 保存有效数据的索引信息
    valid_data_info = pd.DataFrame({
        'original_index': valid_indices,
        'polymer_name': df.iloc[valid_indices]["polymer_name"].values,
        'smiles': df.iloc[valid_indices]["smiles"].values,
        'Median': y
    })
    valid_data_info.to_csv(os.path.join(CONFIG["shap_path"], "valid_data_indices.csv"), index=False)

    if failed_indices:
        failed_data_info = pd.DataFrame({
            'original_index': failed_indices,
            'polymer_name': df.iloc[failed_indices]["polymer_name"].values,
            'smiles': df.iloc[failed_indices]["smiles"].values
        })
        failed_data_info.to_csv(os.path.join(CONFIG["shap_path"], "failed_data_indices.csv"), index=False)

    print(f"\n=== SHAP分析完成 ===")
    print(f"所有数据已保存到: {CONFIG['shap_path']}")
    print("保存的文件包括:")
    print("- shap_analysis_data.npz: 完整的SHAP分析数据")
    print("- feature_importance.csv: 特征重要性排序")
    print("- shap_summary.csv: SHAP值统计摘要")
    print("- valid_data_indices.csv: 有效数据的索引信息")
    if failed_indices:
        print("- failed_data_indices.csv: 失败数据的索引信息")

    total_importance = feature_importance.sum()
    top_10_importance = importance_df.head(10)['importance'].sum()
    print(f"\n特征重要性统计:")
    print(f"总重要性: {total_importance:.4f}")
    print(f"前10个特征重要性占比: {top_10_importance / total_importance:.2%}")


if __name__ == "__main__":
    main()
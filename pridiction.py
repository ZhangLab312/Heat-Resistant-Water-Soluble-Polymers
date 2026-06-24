import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import joblib
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import tempfile

# 配置参数
CONFIG = {
    "input_csv": r"E:\Python\pythonProject\new_t_predict\data\合理分子.csv",
    "output_csv": r"E:\Python\pythonProject\new_t_predict\data\合理分子.csv",
    "model_path": r"E:\Python\pythonProject\new_t_predict\model\fnn_smiles_noerror_model.pth",
    "scaler_path": r"E:\Python\pythonProject\new_t_predict\scaler\fnn_smiles_noerror_scaler.pkl",
    "fingerprint": {
        "radius": 2,
        "n_bits": 1024
    },
    "nn_params": {
        "hidden_layers": [512, 256],
        "dropout_rate": 0.5
    },
    "chunk_size": 10000,  # 处理大数据时每次读取的行数
    "batch_size": 256  # 预测时的批处理大小
}


# 定义FNN模型类（与训练代码相同）
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


# 生成Morgan指纹特征
def generate_features(smiles_list, pbar=None):
    features = []
    valid_smiles = []
    failed_indices = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed_indices.append(idx)
            if pbar:
                pbar.update(1)
            continue

        try:
            # 生成Morgan指纹
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=CONFIG["fingerprint"]["radius"],
                nBits=CONFIG["fingerprint"]["n_bits"]
            )
            arr = np.zeros((CONFIG["fingerprint"]["n_bits"],), dtype=int)
            ConvertToNumpyArray(fp, arr)
            features.append(arr)
            valid_smiles.append(smi)
        except Exception as e:
            failed_indices.append(idx)
            print(f"Error processing SMILES {idx}: {smi} - {str(e)}")

        if pbar:
            pbar.update(1)

    if failed_indices:
        print(f"\nFailed to process {len(failed_indices)} SMILES strings")

    return np.array(features), valid_smiles


def process_chunk(model, device, scaler, chunk, chunk_num, total_chunks, temp_dir):
    """处理一个数据块并返回结果"""
    smiles_list = chunk["SMILES"].tolist()
    sa_scores = chunk["SA_Score"].tolist()

    # 生成特征
    with tqdm(total=len(smiles_list), desc=f"Processing chunk {chunk_num}/{total_chunks}") as pbar:
        X, valid_smiles = generate_features(smiles_list, pbar)

    if len(X) == 0:
        return 0

    # 标准化特征
    X_scaled = scaler.transform(X)

    # 进行预测
    predictions = []
    with torch.no_grad():
        # 分批处理
        for i in range(0, len(X_scaled), CONFIG["batch_size"]):
            batch = X_scaled[i:i + CONFIG["batch_size"]]
            tensor = torch.FloatTensor(batch).to(device)
            batch_pred = model(tensor).cpu().numpy().flatten()
            predictions.extend(batch_pred)

    # 创建结果DataFrame
    results = pd.DataFrame({
        "smiles": valid_smiles,
        "thermal_decomposition_temperature": predictions,
        "SA_Score": sa_scores[:len(valid_smiles)]  # 只取有效SMILES对应的SA_Score
    })

    # 按温度降序排序
    results_sorted = results.sort_values(
        by="thermal_decomposition_temperature",
        ascending=False
    ).reset_index(drop=True)

    # 保存临时结果
    temp_file = os.path.join(temp_dir, f"chunk_{chunk_num}.csv")
    results_sorted.to_csv(temp_file, index=False)

    return len(results_sorted)


def merge_results(temp_dir, output_file, total_rows):
    """合并所有临时结果文件"""
    # 收集所有临时文件
    temp_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith("chunk_")]

    # 如果没有文件，则创建空结果
    if not temp_files:
        empty_df = pd.DataFrame(columns=["smiles", "thermal_decomposition_temperature", "SA_Score"])
        empty_df.to_csv(output_file, index=False)
        return

    # 如果只有一个文件，直接重命名
    if len(temp_files) == 1:
        os.rename(temp_files[0], output_file)
        return

    # 如果有多个文件，使用外部排序合并
    with tqdm(total=total_rows, desc="Merging results") as pbar:
        # 使用生成器逐步读取和写入，避免内存问题
        first_file = True
        with open(output_file, 'w') as outfile:
            for temp_file in temp_files:
                with open(temp_file, 'r') as infile:
                    header = infile.readline()
                    if first_file:
                        outfile.write(header)
                        first_file = False
                    for line in infile:
                        outfile.write(line)
                        pbar.update(1)

        # 删除临时文件
        for temp_file in temp_files:
            os.remove(temp_file)


def main():
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 加载模型和标准化器
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = joblib.load(CONFIG["scaler_path"])

        model = FNN(
            input_size=CONFIG["fingerprint"]["n_bits"],
            hidden_layers=CONFIG["nn_params"]["hidden_layers"],
            dropout_rate=CONFIG["nn_params"]["dropout_rate"]
        )
        model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
        model.to(device)
        model.eval()

        # 获取总行数用于进度条
        total_rows = sum(1 for _ in open(CONFIG["input_csv"])) - 1  # 减去标题行

        # 分块读取和处理数据
        chunk_reader = pd.read_csv(CONFIG["input_csv"], chunksize=CONFIG["chunk_size"])
        total_chunks = (total_rows + CONFIG["chunk_size"] - 1) // CONFIG["chunk_size"]

        processed_rows = 0
        for chunk_num, chunk in enumerate(chunk_reader, 1):
            processed = process_chunk(model, device, scaler, chunk, chunk_num, total_chunks, temp_dir)
            processed_rows += processed

        # 合并所有临时结果
        merge_results(temp_dir, CONFIG["output_csv"], processed_rows)

        print(f"Processing completed. Results saved to {CONFIG['output_csv']}")

        # 显示最高和最低温度
        try:
            # 只读取首尾行来获取最高和最低温度
            with open(CONFIG["output_csv"], 'r') as f:
                headers = f.readline()
                first_line = f.readline()
                for line in f:
                    pass
                last_line = line

            if first_line and last_line:
                first_temp = float(first_line.split(',')[1])
                last_temp = float(last_line.split(',')[1])
                print(f"Highest temperature: {first_temp:.2f}°C")
                print(f"Lowest temperature: {last_temp:.2f}°C")
        except:
            print("Could not determine highest and lowest temperatures")


if __name__ == "__main__":
    main()
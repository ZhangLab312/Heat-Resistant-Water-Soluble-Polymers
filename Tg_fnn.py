import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import chardet

# 配置参数 - 更新路径以匹配您的文件
CONFIG = {
    "input_csv": "E:/Python/pythonProject/new_t_predict/data/Tg_raw.csv",
    "output_csv": "E:/Python/pythonProject/new_t_predict/data/fnn_Tg_predictions.csv",
    "model_path": "E:/Python/pythonProject/new_t_predict/model/fnn_Tg_model.pth",
    "scaler_path": "E:/Python/pythonProject/new_t_predict/scaler/fnn_Tg_scaler.pkl",
    "fingerprint": {
        "radius": 2,
        "n_bits": 1024
    },
    "nn_params": {
        "hidden_layers": [256, 128],
        "dropout_rate": 0.7,
        "learning_rate": 0.0005,
        "epochs": 600,
        "batch_size": 64,
        "patience": 50,
        "weight_decay": 1e-4
    },
    "test_size": 0.2,
    "random_state": 42
}


def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"检测到文件编码: {encoding} (置信度: {confidence:.2f})")
        return encoding


def load_csv_with_encoding(file_path):
    """尝试用不同编码方式加载CSV文件"""
    # 首先尝试检测编码
    try:
        encoding = detect_encoding(file_path)
        if encoding:
            # 跳过第一行（标题行），直接读取数据
            df = pd.read_csv(file_path, encoding=encoding, skiprows=1, header=None, names=['smiles', 'target'])
            return df
    except Exception as e:
        print(f"使用检测到的编码 {encoding} 读取失败: {e}")

    # 如果检测失败，尝试常见编码
    encodings_to_try = ['gbk', 'gb2312', 'gb18030', 'latin1', 'iso-8859-1', 'cp1252']

    for encoding in encodings_to_try:
        try:
            print(f"尝试使用 {encoding} 编码...")
            df = pd.read_csv(file_path, encoding=encoding, skiprows=1, header=None, names=['smiles', 'target'])
            print(f"成功使用 {encoding} 编码读取文件")
            return df
        except Exception as e:
            print(f"使用 {encoding} 编码读取失败: {e}")
            continue

    # 如果所有编码都失败，尝试不使用指定编码
    try:
        print("尝试不使用指定编码...")
        df = pd.read_csv(file_path, skiprows=1, header=None, names=['smiles', 'target'])
        return df
    except Exception as e:
        print(f"不使用指定编码也失败: {e}")
        raise


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
            nn.init.kaiming_normal_(layers[-4].weight, mode='fan_in', nonlinearity='relu')
            prev_size = h_size

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, 1)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)


def generate_features(smiles_list):
    """生成特征并统计失败率"""
    features = []
    valid_indices = []
    total_count = len(smiles_list)
    failed_count = 0

    for idx, smi in enumerate(smiles_list):
        success = False
        # 尝试原始SMILES和最多2次增强
        for attempt in range(3):
            try:
                if attempt == 0:
                    # 第一次尝试使用原始SMILES
                    mol_smi = smi
                else:
                    # 后续尝试使用数据增强
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        mol_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                    else:
                        continue

                mol = Chem.MolFromSmiles(mol_smi)
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
                    success = True
                    break  # 成功一次就跳出循环
            except Exception as e:
                continue

        if not success:
            failed_count += 1

    failure_rate = failed_count / total_count
    print(f"\n[特征生成报告]")
    print(f"总SMILES数量: {total_count}")
    print(f"完全处理失败的SMILES数量: {failed_count} ({failure_rate:.2%})")
    return np.array(features), valid_indices


def stratified_split(X, y):
    """分层分割，处理样本数量不足的情况"""
    try:
        # 使用更少的分箱数量，避免某些箱中样本太少
        n_bins = min(5, len(y) // 20)  # 确保每个箱至少有20个样本
        n_bins = max(2, n_bins)  # 最少2个箱

        bins = np.linspace(y.min(), y.max(), n_bins + 1)
        y_binned = np.digitize(y, bins)

        # 检查每个箱中的样本数量
        unique, counts = np.unique(y_binned, return_counts=True)
        print(f"分箱情况: {dict(zip(unique, counts))}")

        # 如果有任何箱中样本太少，使用随机分割
        if np.min(counts) < 2:
            print("某些分箱中样本太少，使用随机分割")
            return train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])

        split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"]
        )

        for train_idx, test_idx in split.split(X, y_binned):
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    except Exception as e:
        print(f"分层分割失败: {e}，使用随机分割")
        return train_test_split(X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])


def train_fnn(X_train, y_train, X_val, y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train.reshape(-1, 1))
        ),
        batch_size=CONFIG["nn_params"]["batch_size"],
        shuffle=True,
        pin_memory=True
    )

    model = FNN(
        input_size=X_train.shape[1],
        hidden_layers=CONFIG["nn_params"]["hidden_layers"],
        dropout_rate=CONFIG["nn_params"]["dropout_rate"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["nn_params"]["learning_rate"],
        weight_decay=CONFIG["nn_params"]["weight_decay"]
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(CONFIG["nn_params"]["epochs"]):
        model.train()
        total_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        model.eval()
        with torch.no_grad():
            val_inputs = torch.FloatTensor(X_val_scaled).to(device)
            val_targets = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item()

        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_path"])
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["nn_params"]["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:03d} | "
                  f"Train Loss: {total_loss / len(train_loader.dataset):.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(torch.load(CONFIG["model_path"]))
    return model, scaler


def main():
    # 检测并读取CSV文件
    print("正在检测文件编码并读取CSV文件...")
    df = load_csv_with_encoding(CONFIG["input_csv"])

    print(f"数据形状: {df.shape}")
    print(f"前几行数据:")
    print(df.head())

    # 将目标列转换为数值类型
    print("\n正在转换目标列为数值类型...")
    df['target'] = pd.to_numeric(df['target'], errors='coerce')

    # 检查并删除无效的目标值
    invalid_targets = df['target'].isna().sum()
    if invalid_targets > 0:
        print(f"发现 {invalid_targets} 个无效的目标值，将被删除")
        df = df.dropna(subset=['target'])

    # 检查数据质量
    print(f"\n数据基本信息:")
    print(df.info())
    print(f"\n目标列统计:")
    print(df['target'].describe())

    # 检查目标列是否有缺失值
    print(f"目标列缺失值数量: {df['target'].isnull().sum()}")

    # 生成特征并获取有效索引
    print("\n正在生成分子指纹特征...")
    X, valid_indices = generate_features(df["smiles"])
    y = df.iloc[valid_indices]["target"].values

    # 有效性统计
    unique_valid = np.unique(valid_indices)
    total_original = len(df)
    valid_original = len(unique_valid)

    print("\n[数据有效性报告]")
    print(f"原始数据总量: {total_original}")
    print(f"有效处理的SMILES数量: {valid_original} ({valid_original / total_original:.2%})")
    print(
        f"完全失效的SMILES数量: {total_original - valid_original} ({(total_original - valid_original) / total_original:.2%})")

    # 如果没有有效数据，退出
    if len(X) == 0:
        print("错误: 没有有效的SMILES数据可以处理!")
        return

    # 数据划分
    print("\n正在进行数据划分...")
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    print("\n[数据集信息]")
    print(f"有效特征维度: {X.shape[1]}")
    print(f"增强后总样本量: {X.shape[0]}")
    print(f"训练样本量: {X_train.shape[0]}")
    print(f"测试样本量: {X_test.shape[0]}")
    print(f"目标值范围: {y.min():.2f} - {y.max():.2f}")

    # 训练模型
    print("\n=== 开始训练 ===")
    model, scaler = train_fnn(X_train, y_train, X_test, y_test)

    # 保存标准化器
    joblib.dump(scaler, CONFIG["scaler_path"])
    print(f"\n标准化器已保存至: {CONFIG['scaler_path']}")

    # 预测函数
    def predict(X_data):
        with torch.no_grad():
            X_scaled = scaler.transform(X_data)
            tensor = torch.FloatTensor(X_scaled).to(device)
            return model(tensor).cpu().numpy().flatten()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 进行预测
    print("\n正在进行预测...")
    train_pred = predict(X_train)
    test_pred = predict(X_test)
    final_pred = predict(X)

    # 保存结果
    df_pred = df.iloc[valid_indices].copy()
    df_pred["Predicted_Target"] = final_pred
    df_pred.to_csv(CONFIG["output_csv"], index=False)
    print(f"预测结果已保存至: {CONFIG['output_csv']}")

    # 性能评估
    print("\n=== 模型性能 ===")
    print(f"训练集 R²: {r2_score(y_train, train_pred):.4f}")
    print(f"训练集 MAE: {mean_absolute_error(y_train, train_pred):.4f}")
    print(f"测试集 R²: {r2_score(y_test, test_pred):.4f}")
    print(f"测试集 MAE: {mean_absolute_error(y_test, test_pred):.4f}")


if __name__ == "__main__":
    main()
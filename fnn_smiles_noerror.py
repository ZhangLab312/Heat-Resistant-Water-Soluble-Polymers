import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib  # 新增导入joblib用于保存标准化器
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 配置参数 - 添加scaler_path
CONFIG = {
    "input_csv": "E:/Python/pythonProject/t_predict/data/cleaned_predictions.csv",
    "output_csv": "E:/Python/pythonProject/t_predict/data/fnn_smiles_noerror_predictions.csv",
    "model_path": "E:/Python/pythonProject/t_predict/model/fnn_smiles_noerror_model5.pth",
    "scaler_path": "E:/Python/pythonProject/t_predict/scaler/fnn_smiles_noerror_scaler5.pkl",  # 新增标准化器路径
    "fingerprint": {
        "radius": 3,
        "n_bits": 1024
    },
    "nn_params": {
        "hidden_layers": [512, 256],
        "dropout_rate": 0.5,
        "learning_rate": 0.001,
        "epochs": 500,
        "batch_size": 64,
        "patience": 30,
        "weight_decay": 1e-4
    },
    "test_size": 0.2,
    "random_state": 42
}


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

    def augment_smiles(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        return smi

    for idx, smi in enumerate(smiles_list):
        success = False
        for _ in range(3):  # 尝试3次数据增强
            try:
                augmented_smi = augment_smiles(smi)
                mol = Chem.MolFromSmiles(augmented_smi)
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
    bins = np.linspace(y.min(), y.max(), 5)
    y_binned = np.digitize(y, bins)

    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

    for train_idx, test_idx in split.split(X, y_binned):
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


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

        print(f"Epoch {epoch + 1:03d} | "
              f"Train Loss: {total_loss / len(train_loader.dataset):.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(torch.load(CONFIG["model_path"]))
    return model, scaler


def main():
    df = pd.read_csv(CONFIG["input_csv"])

    # 生成特征并获取有效索引
    X, valid_indices = generate_features(df["smiles"])
    y = df.iloc[valid_indices]["Median"].values

    # 有效性统计
    unique_valid = np.unique(valid_indices)
    total_original = len(df)
    valid_original = len(unique_valid)

    print("\n[数据有效性报告]")
    print(f"原始数据总量: {total_original}")
    print(f"有效处理的SMILES数量: {valid_original} ({valid_original / total_original:.2%})")
    print(
        f"完全失效的SMILES数量: {total_original - valid_original} ({(total_original - valid_original) / total_original:.2%})")

    # 数据划分
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    print("\n[数据集信息]")
    print(f"有效特征维度: {X.shape[1]}")
    print(f"增强后总样本量: {X.shape[0]}")
    print(f"训练样本量: {X_train.shape[0]}")
    print(f"测试样本量: {X_test.shape[0]}")

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
    train_pred = predict(X_train)
    test_pred = predict(X_test)
    final_pred = predict(X)

    # 保存结果
    df_pred = df.iloc[valid_indices].copy()
    df_pred["Predicted_Median"] = final_pred
    df_pred.to_csv(CONFIG["output_csv"], index=False)

    # 性能评估
    print("\n=== 模型性能 ===")
    print(f"训练集 R²: {r2_score(y_train, train_pred):.4f}")
    print(f"训练集 MAE: {mean_absolute_error(y_train, train_pred):.4f}")
    print(f"测试集 R²: {r2_score(y_test, test_pred):.4f}")
    print(f"测试集 MAE: {mean_absolute_error(y_test, test_pred):.4f}")


if __name__ == "__main__":
    main()
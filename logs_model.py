import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
import random

# 抑制RDKit的警告
from rdkit import rdBase

rdBase.DisableLog('rdApp.*')

# 配置参数
CONFIG = {
    "input_file": "E:\\Python\\pythonProject\\new_t_predict\\data\\AqueousSolu.csv",  # 请修改为您的文件路径
    "model_path": "E:/Python/pythonProject/new_t_predict/model/logs_model.pth",
    "scaler_path": "E:/Python/pythonProject/new_t_predict/scaler/logs_scaler.pkl",
    "fingerprint": {
        "radius": 2,
        "n_bits": 1024
    },
    "nn_params": {
        "hidden_layers": [512, 256],
        "dropout_rate": 0.5,
        "learning_rate": 0.001,
        "epochs": 600,
        "batch_size": 64,
        "patience": 30,
        "weight_decay": 1e-4
    },
    "data_augmentation": {
        "enabled": True,
        "augment_per_molecule": 4,  # 每个分子生成的增强样本数（包括原始样本）
        "random_seed": 42
    },
    "test_size": 0.2,
    "random_state": 42
}

# 设置随机种子
random.seed(CONFIG["random_state"])
np.random.seed(CONFIG["random_state"])
torch.manual_seed(CONFIG["random_state"])


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


def get_canonical_smiles(smi):
    """获取规范SMILES"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        pass
    return None


def generate_augmented_smiles(smi, num_augmentations):
    """为单个SMILES生成增强样本"""
    augmented_smiles = []

    # 先添加原始SMILES
    augmented_smiles.append(smi)

    # 然后生成增强样本
    for _ in range(num_augmentations - 1):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # 随机化SMILES（数据增强）
                random_smi = Chem.MolToSmiles(
                    mol,
                    doRandom=True,
                    canonical=False,
                    isomericSmiles=True,
                    kekuleSmiles=True
                )
                augmented_smiles.append(random_smi)
            else:
                # 如果无法解析，使用原始SMILES
                augmented_smiles.append(smi)
        except:
            # 如果生成失败，使用原始SMILES
            augmented_smiles.append(smi)

    return augmented_smiles


def generate_features_with_augmentation(smiles_list, y_list):
    """生成特征并进行数据增强，同时记录分子ID"""
    features = []
    all_smiles = []
    all_y = []
    molecule_ids = []  # 记录每个样本属于哪个分子
    molecule_to_id = {}  # 映射：规范SMILES -> 分子ID

    print(f"\n[数据增强报告]")
    print(f"原始SMILES数量: {len(smiles_list)}")
    print(f"每个分子生成的增强样本数: {CONFIG['data_augmentation']['augment_per_molecule']}")

    current_molecule_id = 0

    for idx, (smi, y_val) in enumerate(zip(smiles_list, y_list)):
        # 获取规范SMILES
        canonical_smi = get_canonical_smiles(smi)
        if not canonical_smi:
            print(f"警告: 无法解析SMILES {smi}，跳过")
            continue

        # 如果这个分子还没分配ID，分配一个
        if canonical_smi not in molecule_to_id:
            molecule_to_id[canonical_smi] = current_molecule_id
            current_molecule_id += 1

        molecule_id = molecule_to_id[canonical_smi]

        # 生成增强的SMILES
        augmented_smiles = generate_augmented_smiles(
            smi,
            CONFIG["data_augmentation"]["augment_per_molecule"]
        )

        # 为每个增强SMILES生成特征
        for aug_smi in augmented_smiles:
            try:
                mol = Chem.MolFromSmiles(aug_smi)
                if mol:
                    # 生成摩根指纹
                    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
                    fp = GetMorganFingerprintAsBitVect(
                        mol,
                        radius=CONFIG["fingerprint"]["radius"],
                        nBits=CONFIG["fingerprint"]["n_bits"]
                    )
                    arr = np.zeros((CONFIG["fingerprint"]["n_bits"],), dtype=int)
                    ConvertToNumpyArray(fp, arr)

                    features.append(arr)
                    all_smiles.append(aug_smi)
                    all_y.append(y_val)
                    molecule_ids.append(molecule_id)
            except Exception as e:
                print(f"警告: 处理增强SMILES失败: {aug_smi}, 错误: {e}")

    print(f"生成的增强样本总数: {len(features)}")
    print(f"唯一分子数量: {len(molecule_to_id)}")
    print(f"平均每个分子的样本数: {len(features) / len(molecule_to_id):.2f}")

    return np.array(features), np.array(all_y), molecule_ids, molecule_to_id


def split_by_molecules(X, y, molecule_ids, test_size=0.2, random_state=42):
    """按分子划分数据集，确保同一个分子的所有样本在同一个集合"""

    # 获取所有唯一的分子ID
    unique_molecule_ids = list(set(molecule_ids))
    print(f"总分子数: {len(unique_molecule_ids)}")

    # 为每个分子计算平均y值（用于分层抽样）
    molecule_y_values = []
    for mol_id in unique_molecule_ids:
        # 找到这个分子的所有样本的索引
        sample_indices = [i for i, m_id in enumerate(molecule_ids) if m_id == mol_id]
        # 计算这个分子所有样本y值的平均值
        avg_y = np.mean(y[sample_indices])
        molecule_y_values.append(avg_y)

    molecule_y_values = np.array(molecule_y_values)

    # 尝试分层抽样，如果失败则使用随机分割
    try:
        # 动态确定分箱数，确保每个分箱至少有2个分子
        n_bins = 5  # 从5个分箱开始尝试
        success = False

        while n_bins >= 2:
            bins = np.linspace(molecule_y_values.min(), molecule_y_values.max(), n_bins + 1)
            y_binned = np.digitize(molecule_y_values, bins) - 1

            # 检查每个分箱的分子数
            bin_counts = np.bincount(y_binned)
            if np.all(bin_counts >= 2):
                success = True
                print(f"使用 {n_bins} 个分箱进行分层抽样")
                print(f"每个分箱的分子数: {bin_counts}")
                break
            else:
                n_bins -= 1

        if not success:
            print("无法进行分层抽样，使用随机分割")
            train_molecule_ids, test_molecule_ids = train_test_split(
                unique_molecule_ids,
                test_size=test_size,
                random_state=random_state
            )
        else:
            # 使用分层抽样划分分子
            from sklearn.model_selection import StratifiedShuffleSplit
            split = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state
            )

            for train_idx, test_idx in split.split(unique_molecule_ids, y_binned):
                train_molecule_ids = [unique_molecule_ids[i] for i in train_idx]
                test_molecule_ids = [unique_molecule_ids[i] for i in test_idx]

    except Exception as e:
        print(f"分层抽样失败，使用随机分割: {e}")
        train_molecule_ids, test_molecule_ids = train_test_split(
            unique_molecule_ids,
            test_size=test_size,
            random_state=random_state
        )

    print(f"训练集分子数: {len(train_molecule_ids)}")
    print(f"测试集分子数: {len(test_molecule_ids)}")

    # 根据分子ID分配样本到训练集和测试集
    train_indices = []
    test_indices = []

    for idx, mol_id in enumerate(molecule_ids):
        if mol_id in train_molecule_ids:
            train_indices.append(idx)
        else:
            test_indices.append(idx)

    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}")

    # 验证：检查是否有分子同时出现在两个集合
    train_molecule_set = set(train_molecule_ids)
    test_molecule_set = set(test_molecule_ids)
    overlap = train_molecule_set.intersection(test_molecule_set)

    if overlap:
        print(f"错误: 有 {len(overlap)} 个分子同时出现在训练集和测试集!")
        return None, None, None, None
    else:
        print("✓ 验证通过: 没有分子同时出现在训练集和测试集")

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def train_fnn(X_train, y_train, X_val, y_val):
    """训练FNN模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印设备信息
    print(f"\n[设备信息]")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("警告: 未检测到CUDA设备，将使用CPU训练（速度较慢）")

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 创建数据加载器
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train.reshape(-1, 1))
        ),
        batch_size=CONFIG["nn_params"]["batch_size"],
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 初始化模型
    model = FNN(
        input_size=X_train.shape[1],
        hidden_layers=CONFIG["nn_params"]["hidden_layers"],
        dropout_rate=CONFIG["nn_params"]["dropout_rate"]
    ).to(device)

    # 定义优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["nn_params"]["learning_rate"],
        weight_decay=CONFIG["nn_params"]["weight_decay"]
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0

    print("\n开始训练...")
    for epoch in range(CONFIG["nn_params"]["epochs"]):
        model.train()
        total_loss = 0.0

        # 训练步骤
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        # 验证步骤
        model.eval()
        with torch.no_grad():
            val_inputs = torch.FloatTensor(X_val_scaled).to(device)
            val_targets = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item()

        # 学习率调整
        scheduler.step(val_loss)

        # 早期停止检查
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_path"])
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["nn_params"]["patience"]:
                print(f"早期停止在第 {epoch + 1} 轮")
                break

        # 输出训练进度
        if (epoch + 1) % 20 == 0:
            train_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # 加载最佳模型
    model.load_state_dict(torch.load(CONFIG["model_path"]))
    return model, scaler


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    """评估模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 预测函数
    def predict(X_data):
        with torch.no_grad():
            X_scaled = scaler.transform(X_data)
            tensor = torch.FloatTensor(X_scaled).to(device)
            return model(tensor).cpu().numpy().flatten()

    # 进行预测
    train_pred = predict(X_train)
    test_pred = predict(X_test)

    # 计算指标
    train_r2 = r2_score(y_train, train_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    return train_r2, train_mse, test_r2, test_mse


def main():
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(CONFIG["input_file"])

    # 检查列名
    print(f"数据列名: {df.columns.tolist()}")

    # 假设您的文件包含'smiles solute'和'logS_aq_avg'列
    # 如果列名不同，请修改下面的列名
    smiles_col = 'smiles solute' if 'smiles solute' in df.columns else 'smiles'
    target_col = 'logS_aq_avg' if 'logS_aq_avg' in df.columns else 'logS'

    print(f"使用SMILES列: {smiles_col}")
    print(f"使用目标列: {target_col}")

    # 准备数据
    smiles_list = df[smiles_col].tolist()
    y_list = df[target_col].values.tolist()

    print(f"\n原始数据总量: {len(smiles_list)}")

    # 生成特征并进行数据增强
    print("\n生成摩根指纹特征并进行数据增强...")
    X, y, molecule_ids, molecule_to_id = generate_features_with_augmentation(smiles_list, y_list)

    # 按分子划分数据集（确保同一个分子的所有样本在同一个集合）
    print("\n按分子划分数据集...")
    X_train, X_test, y_train, y_test = split_by_molecules(
        X, y, molecule_ids,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

    if X_train is None:
        print("数据集划分失败，程序退出")
        return

    print("\n[数据集信息]")
    print(f"特征维度: {X.shape[1]}")
    print(f"增强后总样本量: {X.shape[0]}")
    print(f"训练集样本量: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    print(f"训练集/测试集比例: {X_train.shape[0] / X_test.shape[0]:.2f}:1")

    # 训练模型
    model, scaler = train_fnn(X_train, y_train, X_test, y_test)

    # 保存标准化器
    joblib.dump(scaler, CONFIG["scaler_path"])
    print(f"\n标准化器已保存至: {CONFIG['scaler_path']}")

    # 评估模型
    print("\n=== 模型性能 ===")
    train_r2, train_mse, test_r2, test_mse = evaluate_model(
        model, scaler, X_train, y_train, X_test, y_test
    )

    print(f"训练集 R²: {train_r2:.4f}")
    print(f"训练集 MSE: {train_mse:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}")

    # 额外的统计分析
    print("\n=== 数据统计分析 ===")
    print(
        f"训练集y值范围: [{y_train.min():.2f}, {y_train.max():.2f}], 均值: {y_train.mean():.2f}, 标准差: {y_train.std():.2f}")
    print(
        f"测试集y值范围: [{y_test.min():.2f}, {y_test.max():.2f}], 均值: {y_test.mean():.2f}, 标准差: {y_test.std():.2f}")

    # 检查数据增强效果
    unique_train_molecules = set()
    unique_test_molecules = set()

    for idx, mol_id in enumerate(molecule_ids):
        # 找到这个样本属于哪个集合
        if idx < len(X_train):
            unique_train_molecules.add(mol_id)
        else:
            unique_test_molecules.add(mol_id)

    print(f"\n训练集唯一分子数: {len(unique_train_molecules)}")
    print(f"测试集唯一分子数: {len(unique_test_molecules)}")
    print(f"训练集平均每个分子样本数: {len(X_train) / len(unique_train_molecules):.2f}")
    print(f"测试集平均每个分子样本数: {len(X_test) / len(unique_test_molecules):.2f}")


if __name__ == "__main__":
    main()
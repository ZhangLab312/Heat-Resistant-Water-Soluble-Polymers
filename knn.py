import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import os


# 特征生成函数（可选指纹类型）
def generate_fingerprints(smiles_list, fp_type='morgan', nBits=1024, radius=2):
    """
    生成分子指纹
    fp_type: 'morgan' 或 'maccs'
    """
    features = []
    valid_smiles = []
    valid_targets = []

    for smiles, target in smiles_list:
        try:
            smiles = str(smiles).strip()
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            if fp_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
                arr = np.zeros((nBits,), dtype=int)
            elif fp_type == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros((166,), dtype=int)  # MACCS固定166位
            else:
                raise ValueError("fp_type must be 'morgan' or 'maccs'")

            ConvertToNumpyArray(fp, arr)
            features.append(arr)
            valid_smiles.append(smiles)
            valid_targets.append(target)
        except:
            continue

    return np.array(features), np.array(valid_targets), valid_smiles


# 保存训练集和测试集
def save_train_test_split(train_df, test_df, output_dir, train_filename="KNN_train.csv", test_filename="KNN_test.csv"):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, train_filename)
    test_path = os.path.join(output_dir, test_filename)
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    print(f"训练集已保存至: {train_path}")
    print(f"测试集已保存至: {test_path}")


def main():
    try:
        # 1. 加载数据
        print("正在加载数据...")
        file_path = "E:/Python/pythonProject/new_t_predict/data/cleaned_predictions.csv"
        encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, skiprows=1,
                                 header=None, names=['smiles', 'target'])
                print(f"成功使用 {encoding} 编码加载数据")
                break
            except:
                continue
        if df is None:
            df = pd.read_csv(file_path, skiprows=1, header=None, names=['smiles', 'target'])
            print("成功加载数据（无指定编码）")

        print(f"原始数据形状: {df.shape}")

        # 2. 清洗目标值
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df = df.dropna(subset=['target'])
        print(f"删除无效目标值后: {len(df)} 行")

        # 3. 【改进1】基于SMILES去重（保留第一个）
        df = df.drop_duplicates(subset=['smiles'], keep='first')
        print(f"去重后剩余: {len(df)} 个唯一SMILES")

        if len(df) == 0:
            print("错误: 没有有效数据!")
            return

        # 4. 【改进2】生成指纹（降低维度，使用1024位Morgan或可选MACCS）
        print("正在生成分子指纹特征...")
        # 可在此处切换指纹类型：'morgan' 或 'maccs'
        fp_type = 'morgan'  # 改为 'maccs' 可进一步降维到166位
        if fp_type == 'morgan':
            nBits = 1024  # 原来2048 -> 1024，减轻过拟合
            radius = 3
            print(f"使用 Morgan 指纹，位数={nBits}, radius={radius}")
        else:
            nBits = 166  # MACCS固定
            print("使用 MACCS 指纹 (166位)")

        # 准备输入列表
        smiles_target_pairs = list(zip(df['smiles'], df['target']))
        X, y, valid_smiles = generate_fingerprints(smiles_target_pairs, fp_type=fp_type, nBits=nBits, radius=3)

        print(f"成功生成指纹: {len(X)} 个分子")
        if len(X) == 0:
            print("错误: 无法生成任何有效特征!")
            return

        # 构建有效DataFrame
        valid_df = pd.DataFrame({'smiles': valid_smiles, 'target': y})

        # 5. 划分训练/测试集（分层抽样可选，这里简单随机）
        indices = np.arange(len(valid_df))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        train_df = valid_df.iloc[train_idx].reset_index(drop=True)
        test_df = valid_df.iloc[test_idx].reset_index(drop=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

        # 保存训练/测试集（与之前相同）
        output_dir = r"E:\Python\pythonProject\new_t_predict\画图数据"
        save_train_test_split(train_df, test_df, output_dir,
                              train_filename="KNN_train.csv",
                              test_filename="KNN_test.csv")

        # 6. 标准化
        print("正在标准化特征...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 7. 【改进3】扩大K值范围 + 网格搜索交叉验证
        print("正在使用交叉验证搜索最佳 K 值...")
        param_grid = {
            'n_neighbors': [10, 20, 30, 40, 50, 75, 100],  # 避免小K值
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 曼哈顿距离和欧氏距离
        }
        knn_base = KNeighborsRegressor(n_jobs=-1)
        grid_search = GridSearchCV(knn_base, param_grid, cv=5,
                                   scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        best_knn = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"最佳参数: {best_params}")

        # 8. 评估最终模型
        y_train_pred = best_knn.predict(X_train_scaled)
        y_test_pred = best_knn.predict(X_test_scaled)

        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # 输出结果
        print("\n" + "=" * 60)
        print("改进后 KNN 模型性能评估")
        print("=" * 60)
        print(f"训练集 R²:  {train_r2:.4f}")
        print(f"训练集 MAE: {train_mae:.4f}")
        print(f"测试集 R²:  {test_r2:.4f}")
        print(f"测试集 MAE: {test_mae:.4f}")
        print("=" * 60)
        print(f"训练集目标值均值: {y_train.mean():.4f}")
        print(f"训练集预测值均值: {y_train_pred.mean():.4f}")
        print(f"测试集目标值均值: {y_test.mean():.4f}")
        print(f"测试集预测值均值: {y_test_pred.mean():.4f}")

        # 可选：输出交叉验证结果摘要
        cv_results = grid_search.cv_results_
        print(f"\n交叉验证最佳平均 R²: {grid_search.best_score_:.4f}")

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
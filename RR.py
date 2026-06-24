import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def main():
    # 1. 加载数据
    file_path = "E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv"

    for encoding in ['gbk', 'gb2312', 'gb18030']:
        try:
            df = pd.read_csv(file_path, encoding=encoding, skiprows=1,
                             header=None, names=['smiles', 'target'])
            break
        except:
            continue

    # 2. 清洗数据
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df = df.dropna(subset=['target'])

    # 3. 生成特征
    X, y = [], []
    for _, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row['smiles']))
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
                arr = np.zeros(2048, dtype=int)
                ConvertToNumpyArray(fp, arr)
                X.append(arr)
                y.append(row['target'])
        except:
            continue

    X, y = np.array(X), np.array(y)

    print(f"数据集: {X.shape[0]} 个样本, {X.shape[1]} 个特征")

    # 4. 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. 使用网格搜索寻找最佳alpha
    print("\n使用网格搜索寻找最佳岭回归参数...")

    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    }

    ridge = Ridge(random_state=42)
    grid_search = GridSearchCV(
        ridge,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    print(f"最佳alpha值: {grid_search.best_params_['alpha']}")
    print(f"最佳交叉验证分数: {-grid_search.best_score_:.4f} (MSE)")

    # 7. 使用最佳模型
    best_ridge = grid_search.best_estimator_

    # 预测
    y_train_pred = best_ridge.predict(X_train_scaled)
    y_test_pred = best_ridge.predict(X_test_scaled)

    # 8. 计算指标
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 9. 输出结果
    print("\n" + "=" * 60)
    print("岭回归模型性能 (自动调参)")
    print("=" * 60)
    print(f"训练集 R²:  {train_r2:.4f}")
    print(f"训练集 MAE: {train_mae:.4f}")
    print(f"测试集 R²:  {test_r2:.4f}")
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"过拟合程度:  {train_r2 - test_r2:.4f}")
    print("=" * 60)

    # 10. 特征重要性
    print(f"\n模型截距: {best_ridge.intercept_:.4f}")
    coefficients = best_ridge.coef_
    print(f"非零系数数量: {np.sum(coefficients != 0)}/{len(coefficients)}")
    print(f"系数绝对值平均值: {np.abs(coefficients).mean():.6f}")

    # 找出最重要的特征
    top_n = 10
    abs_coef = np.abs(coefficients)
    top_indices = np.argsort(abs_coef)[-top_n:]

    print(f"\n前{top_n}个最重要特征(系数绝对值最大):")
    for i, idx in enumerate(reversed(top_indices)):
        print(f"  特征 {idx}: 系数 = {coefficients[idx]:.6f}")


if __name__ == "__main__":
    main()
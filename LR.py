import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
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
                X.append(list(fp))
                y.append(row['target'])
        except:
            continue

    X, y = np.array(X), np.array(y)

    print(f"数据集: {X.shape[0]} 个样本, {X.shape[1]} 个特征")

    # 4. 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. 标准化特征（非常重要！）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. 训练普通线性回归
    print("训练普通线性回归...")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train_scaled, y_train)

    # 7. 预测和评估
    y_train_pred = lr.predict(X_train_scaled)
    y_test_pred = lr.predict(X_test_scaled)

    # 8. 输出结果
    print(f"训练集 R²: {r2_score(y_train, y_train_pred):.4f}")
    print(f"训练集 MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"测试集 R²: {r2_score(y_test, y_test_pred):.4f}")
    print(f"测试集 MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")

    # 9. 模型信息
    print(f"\n模型截距: {lr.intercept_:.4f}")
    print(f"非零系数数量: {np.sum(lr.coef_ != 0)}/{len(lr.coef_)}")


if __name__ == "__main__":
    main()
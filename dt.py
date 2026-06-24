import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import time


def main():
    start_time = time.time()

    # 1. 加载数据
    print("正在加载数据...")
    file_path = "E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv"

    for encoding in ['gbk', 'gb2312', 'gb18030']:
        try:
            df = pd.read_csv(file_path, encoding=encoding, skiprows=1,
                             header=None, names=['smiles', 'target'])
            print(f"使用 {encoding} 编码成功")
            break
        except:
            continue

    # 2. 清洗数据
    print("正在清洗数据...")
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df = df.dropna(subset=['target'])

    print(f"有效数据: {len(df)} 行")
    print(f"目标值范围: {df['target'].min():.2f} - {df['target'].max():.2f}")

    # 3. 生成特征
    print("正在生成特征...")
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

    X = np.array(X)
    y = np.array(y)

    print(f"特征矩阵形状: {X.shape}")

    # 4. 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # 5. 训练决策树模型 - 尝试不同深度
    print("\n训练决策树模型...")

    # 尝试不同的最大深度
    depths = [5, 10, 15, 20, None]  # None表示不限制深度

    for depth in depths:
        print(f"\n尝试最大深度: {depth}")

        dt = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_split=10,  # 防止过拟合
            min_samples_leaf=5,  # 防止过拟合
            random_state=42
        )

        dt.fit(X_train, y_train)

        # 预测
        y_train_pred = dt.predict(X_train)
        y_test_pred = dt.predict(X_test)

        # 计算指标
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # 输出结果
        print(f"  训练集 R²:  {train_r2:.4f}")
        print(f"  训练集 MAE: {train_mae:.4f}")
        print(f"  测试集 R²:  {test_r2:.4f}")
        print(f"  测试集 MAE: {test_mae:.4f}")
        print(f"  过拟合程度:  {train_r2 - test_r2:.4f}")

        # 输出树的信息
        if dt.tree_ is not None:
            print(f"  树的节点数: {dt.tree_.node_count}")

    # 6. 推荐最佳深度
    print("\n" + "=" * 60)
    print("决策树模型性能总结")
    print("=" * 60)
    print("建议: 选择测试集R²最高且过拟合程度最小的深度")
    print("注意: 决策树容易过拟合，建议使用适当的正则化参数")

    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time:.2f} 秒")


if __name__ == "__main__":
    main()
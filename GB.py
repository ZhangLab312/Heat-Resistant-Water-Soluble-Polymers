import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor


def main():
    print("开始执行程序...")

    try:
        # 1. 加载数据
        print("1. 正在加载数据...")
        df = pd.read_csv("E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv",
                         encoding='gbk', skiprows=1, header=None,
                         names=['smiles', 'target'])
        print(f"数据加载成功，共 {len(df)} 行")

        # 2. 清洗数据
        print("2. 正在清洗数据...")
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df = df.dropna(subset=['target'])
        print(f"清洗后数据: {len(df)} 行")

        # 3. 生成特征 - 只取前50个样本
        print("3. 正在生成特征（只取前50个样本）...")
        X, y = [], []
        for i in range(len(df)):
            row = df.iloc[i]
            mol = Chem.MolFromSmiles(str(row['smiles']))
            if mol:
                # 只使用512位特征
                fp = Chem.RDKFingerprint(mol, fpSize=2048)
                X.append(list(fp))
                y.append(row['target'])
            if (i + 1) % 10 == 0:
                print(f"   已处理 {i + 1} 个分子")

        if not X:
            print("错误: 没有生成任何特征")
            return

        X, y = np.array(X), np.array(y)

        # 4. 划分数据
        print("4. 正在划分数据集...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. 训练模型
        print("5. 正在训练模型...")
        model = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # 6. 预测和评估
        print("6. 正在评估模型...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        print("\n结果:")
        print(f"训练集 R²: {r2_score(y_train, y_train_pred):.4f}")
        print(f"训练集 MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
        print(f"测试集 R²: {r2_score(y_test, y_test_pred):.4f}")
        print(f"测试集 MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")

    except Exception as e:
        print(f"程序出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
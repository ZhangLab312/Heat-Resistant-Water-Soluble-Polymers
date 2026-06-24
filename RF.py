import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

warnings.filterwarnings('ignore')


def main():
    start_time = time.time()

    # 1. 加载数据
    print("正在加载数据...")
    file_path = "E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv"

    for encoding in ['gbk', 'gb2312', 'gb18030', 'utf-8', 'latin1']:
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
    initial_count = len(df)
    df = df.dropna(subset=['target'])
    print(f"删除无效目标值: {initial_count - len(df)} 行")

    if len(df) == 0:
        print("错误: 没有有效数据!")
        return

    print(f"有效数据: {len(df)} 行")
    print(f"目标值统计: 均值={df['target'].mean():.2f}, 标准差={df['target'].std():.2f}")
    print(f"目标值范围: {df['target'].min():.2f} - {df['target'].max():.2f}")

    # 3. 生成特征
    print("正在生成特征...")
    X, y = [], []

    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row['smiles']))
            if mol:
                # 使用半径2的指纹
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
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

    # 5. 训练随机森林模型
    print("\n正在训练随机森林模型...")
    print("模型参数:")
    print("  n_estimators: 200")
    print("  max_depth: None (不限制深度)")
    print("  min_samples_split: 2")
    print("  min_samples_leaf: 1")
    print("  max_features: 'sqrt' (使用sqrt(n_features)个特征)")

    start_train_time = time.time()

    # 使用合适的参数
    rf = RandomForestRegressor(
        n_estimators=200,  # 树的数量
        max_depth=None,  # 不限制树深度
        min_samples_split=2,  # 分裂内部节点所需的最小样本数
        min_samples_leaf=1,  # 叶节点所需的最小样本数
        max_features='sqrt',  # 使用sqrt(n_features)个特征，这是正确的参数
        bootstrap=True,  # 使用bootstrap采样
        random_state=42,
        n_jobs=-1,  # 使用所有CPU核心
        verbose=0
    )

    rf.fit(X_train, y_train)

    train_time = time.time() - start_train_time
    print(f"训练完成，耗时: {train_time:.2f}秒")

    # 6. 预测
    print("正在进行预测...")
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # 7. 计算指标
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    overfit = train_r2 - test_r2

    # 8. 输出结果
    print("\n" + "=" * 60)
    print("随机森林模型性能")
    print("=" * 60)
    print(f"训练集 R²:  {train_r2:.4f}")
    print(f"训练集 MAE: {train_mae:.4f}")
    print(f"测试集 R²:  {test_r2:.4f}")
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"过拟合程度:  {overfit:.4f}")
    print("=" * 60)

    # 9. 交叉验证
    print("\n进行5折交叉验证...")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"交叉验证R²分数: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"交叉验证平均R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 10. 特征重要性分析
    feature_importances = rf.feature_importances_
    print(f"\n特征重要性统计:")
    print(f"  平均值: {feature_importances.mean():.6f}")
    print(f"  最大值: {feature_importances.max():.6f}")
    print(f"  最小值: {feature_importances.min():.6f}")

    # 找出最重要的特征
    top_10_idx = np.argsort(feature_importances)[-10:]
    print(f"前10个最重要特征的索引: {top_10_idx}")

    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time:.2f}秒")

    # 11. 性能分析
    print("\n[性能分析]")
    if train_r2 < 0.7:
        print("⚠️  训练集拟合不足，建议:")
        print("  1. 增加 n_estimators 到 300-500")
        print("  2. 移除 max_depth 限制")
        print("  3. 尝试使用半径3的指纹 (radius=3)")
        print("  4. 尝试其他模型，如XGBoost或神经网络")
    elif train_r2 < 0.85:
        print("✅ 训练集拟合中等")
    else:
        print("✅ 训练集拟合良好")

    if test_r2 < 0.6:
        print("⚠️  测试集泛化能力不足，建议:")
        print("  1. 增加正则化参数 (min_samples_split, min_samples_leaf)")
        print("  2. 使用更多的训练数据")
        print("  3. 尝试特征选择或降维")
    else:
        print("✅ 测试集泛化能力良好")


if __name__ == "__main__":
    main()
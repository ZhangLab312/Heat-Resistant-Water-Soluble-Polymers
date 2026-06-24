import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import time


def main():
    start_time = time.time()

    print("=" * 60)
    print("SVM模型训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n步骤1: 正在加载数据...")
    file_path = "E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv"

    # 尝试不同编码
    for encoding in ['gbk', 'gb2312', 'gb18030', 'utf-8', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding, skiprows=1,
                             header=None, names=['smiles', 'target'])
            print(f"✓ 使用 {encoding} 编码成功")
            break
        except:
            continue

    # 2. 清洗数据
    print("\n步骤2: 正在清洗数据...")
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df = df.dropna(subset=['target'])

    print(f"有效数据: {len(df)} 行")
    print(f"目标值统计: 均值={df['target'].mean():.2f}, 标准差={df['target'].std():.2f}")

    # 3. 生成特征 - 使用较少的特征维度以加快SVM训练
    print("\n步骤3: 正在生成特征...")
    print("提示: SVM训练较慢，使用512位特征加快速度")

    X, y = [], []
    valid_count = 0

    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row['smiles']))
            if mol:
                # 使用较少的特征维度
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
                X.append(list(fp))
                y.append(row['target'])
                valid_count += 1

                if valid_count % 500 == 0:
                    print(f"  已处理 {valid_count} 个分子...")
        except:
            continue

    X = np.array(X)
    y = np.array(y)

    print(f"特征生成完成! 共 {valid_count} 个有效样本")
    print(f"特征维度: {X.shape[1]}")

    if valid_count == 0:
        print("错误: 没有生成任何有效特征!")
        return

    # 4. 划分数据集
    print("\n步骤4: 正在划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"测试集: {X_test.shape[0]} 个样本")

    # 5. 标准化特征（SVM对特征缩放敏感）
    print("\n步骤5: 正在标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. 训练SVM模型
    print("\n步骤6: 正在训练SVM模型...")
    print("SVM参数:")
    print("  kernel: 'rbf' (径向基核函数)")
    print("  C: 1.0 (正则化参数)")
    print("  epsilon: 0.1 (epsilon-SVR的epsilon参数)")
    print("  注意: SVM训练可能较慢...")

    # 创建SVR模型
    svm_model = SVR(
        kernel='rbf',  # 径向基核函数
        C=1.0,  # 正则化参数
        epsilon=0.1,  # epsilon-SVR的epsilon参数
        gamma='scale',  # 核函数系数
        cache_size=500,  # 缓存大小(MB)
        verbose=True,  # 显示训练进度
        max_iter=-1  # 最大迭代次数（-1表示无限制）
    )

    # 训练模型
    train_start = time.time()
    svm_model.fit(X_train_scaled, y_train)
    train_time = time.time() - train_start

    print(f"\n训练完成! 耗时: {train_time:.2f} 秒")

    # 7. 预测
    print("\n步骤7: 正在进行预测...")
    y_train_pred = svm_model.predict(X_train_scaled)
    y_test_pred = svm_model.predict(X_test_scaled)

    # 8. 计算性能指标
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 9. 输出结果
    print("\n" + "=" * 60)
    print("SVM模型性能评估")
    print("=" * 60)
    print(f"训练集 R²:  {train_r2:.4f}")
    print(f"训练集 MAE: {train_mae:.4f}")
    print(f"测试集 R²:  {test_r2:.4f}")
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"过拟合程度:  {train_r2 - test_r2:.4f}")
    print("=" * 60)

    # 10. 总运行时间
    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time:.2f} 秒")

    # 11. 模型参数详细信息
    print(f"\n模型参数:")
    print(f"  支持向量数量: {len(svm_model.support_)}")
    print(f"  支持向量比例: {len(svm_model.support_) / len(X_train_scaled) * 100:.1f}%")


if __name__ == "__main__":
    main()
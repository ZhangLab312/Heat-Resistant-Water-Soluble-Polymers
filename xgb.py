import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import time


def main():
    start_time = time.time()

    print("=" * 60)
    print("XGBoost模型训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n步骤1: 正在加载数据...")
    file_path = "E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv"

    # 尝试不同编码
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'latin1']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, skiprows=1,
                             header=None, names=['smiles', 'target'])
            print(f"✓ 使用 {encoding} 编码成功加载数据")
            break
        except Exception as e:
            continue

    if df is None:
        try:
            df = pd.read_csv(file_path, skiprows=1, header=None,
                             names=['smiles', 'target'])
            print("✓ 使用默认编码成功加载数据")
        except Exception as e:
            print(f"✗ 无法加载文件: {e}")
            return

    print(f"原始数据量: {len(df)} 行")

    # 2. 数据清洗
    print("\n步骤2: 正在清洗数据...")
    df['target'] = pd.to_numeric(df['target'], errors='coerce')

    initial_count = len(df)
    df = df.dropna(subset=['target'])
    cleaned_count = len(df)

    print(f"删除无效目标值: {initial_count - cleaned_count} 行")
    print(f"有效数据: {cleaned_count} 行")
    print(f"目标值统计:")
    print(f"  均值: {df['target'].mean():.2f}")
    print(f"  标准差: {df['target'].std():.2f}")
    print(f"  最小值: {df['target'].min():.2f}")
    print(f"  最大值: {df['target'].max():.2f}")

    # 3. 生成特征
    print("\n步骤3: 正在生成分子指纹特征...")
    print("这可能需要一些时间，请耐心等待...")

    X, y = [], []
    valid_count = 0
    total_count = len(df)

    for idx, row in df.iterrows():
        try:
            # 处理SMILES
            smiles = str(row['smiles']).strip()
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # 生成Morgan指纹，使用更合理的参数
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=3,  # 减少半径以加快速度
                    nBits=2048  # 减少位数以加快训练
                )

                # 将指纹转换为列表
                features = list(fp)
                X.append(features)
                y.append(row['target'])
                valid_count += 1

                # 显示进度
                if valid_count % 500 == 0:
                    print(f"  已处理 {valid_count}/{total_count} 个分子...")
        except Exception as e:
            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"\n特征生成完成!")
    print(f"成功生成特征: {valid_count} 个分子")
    print(f"特征维度: {X.shape[1]}")

    if valid_count == 0:
        print("错误: 没有生成任何有效特征!")
        return

    # 4. 划分训练集和测试集
    print("\n步骤4: 正在划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"测试集: {X_test.shape[0]} 个样本")

    # 5. 训练XGBoost模型
    print("\n步骤5: 正在训练XGBoost模型...")
    print("XGBoost参数:")
    print("  n_estimators: 200")
    print("  max_depth: 6")
    print("  learning_rate: 0.1")
    print("  subsample: 0.8")
    print("  colsample_bytree: 0.8")
    print("  random_state: 42")

    # 创建XGBoost模型
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,  # 树的数量
        max_depth=6,  # 树的最大深度
        learning_rate=0.1,  # 学习率
        subsample=0.8,  # 样本采样比例
        colsample_bytree=0.8,  # 特征采样比例
        reg_alpha=0.1,  # L1正则化
        reg_lambda=1.0,  # L2正则化
        random_state=42,  # 随机种子
        n_jobs=-1,  # 使用所有CPU核心
        verbosity=0  # 静默模式
    )

    # 训练模型
    print("\n开始训练...")
    train_start = time.time()
    xgb_model.fit(X_train, y_train)
    train_time = time.time() - train_start

    print(f"训练完成! 耗时: {train_time:.2f} 秒")

    # 6. 预测
    print("\n步骤6: 正在进行预测...")

    # 训练集预测
    y_train_pred = xgb_model.predict(X_train)
    # 测试集预测
    y_test_pred = xgb_model.predict(X_test)

    # 7. 计算性能指标
    print("\n步骤7: 计算性能指标...")

    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 8. 输出结果
    print("\n" + "=" * 60)
    print("XGBoost模型性能评估")
    print("=" * 60)
    print(f"训练集 R²:  {train_r2:.4f}")
    print(f"训练集 MAE: {train_mae:.4f}")
    print(f"测试集 R²:  {test_r2:.4f}")
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"过拟合程度 (训练集R²-测试集R²): {train_r2 - test_r2:.4f}")
    print("=" * 60)

    # 9. 特征重要性分析
    print("\n特征重要性分析:")
    feature_importances = xgb_model.feature_importances_

    # 统计特征重要性
    print(f"平均特征重要性: {feature_importances.mean():.6f}")
    print(f"最大特征重要性: {feature_importances.max():.6f}")
    print(f"最小特征重要性: {feature_importances.min():.6f}")

    # 找出最重要的特征
    top_n = 10
    top_indices = np.argsort(feature_importances)[-top_n:]
    print(f"\n前{top_n}个最重要特征的索引: {top_indices}")

    # 10. 总运行时间
    total_time = time.time() - start_time
    print(f"\n总运行时间: {total_time:.2f} 秒")

    # 11. 性能评估建议
    print("\n[性能评估]")
    if test_r2 >= 0.8:
        print("✅ 模型性能优秀!")
    elif test_r2 >= 0.6:
        print("✅ 模型性能良好")
    elif test_r2 >= 0.4:
        print("⚠️  模型性能一般")
    else:
        print("⚠️  模型性能较差，建议:")
        print("  1. 增加数据量")
        print("  2. 调整模型参数")
        print("  3. 尝试其他特征表示方法")
        print("  4. 考虑使用更复杂的模型架构")


if __name__ == "__main__":
    main()
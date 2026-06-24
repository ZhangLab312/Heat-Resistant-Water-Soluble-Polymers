import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import seaborn as sns


def calculate_stotal_and_logistic_regression(csv_path, output_path=None):
    """
    计算Stotal并进行逻辑回归分析（含10折交叉验证，混淆矩阵累积所有折）
    公式: Stotal = w1*z(-LogP) + w2*z(HBD+HBA) + w3*z(logSmonomer) + w4*z(δ)
    包含：交叉验证ROC曲线、累积混淆矩阵、权重条形图、Stotal分布、概率分布
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    print(f"读取文件: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='gbk')
        except:
            print("无法读取CSV文件，请检查编码")
            return None

    required_columns = ['LogP', 'HBD', 'HBA', 'predicted_logS_water_298K', '是否可水溶', 'δ']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: 缺少必要列: {missing_columns}")
        return None

    df_clean = df.dropna(subset=required_columns).copy()
    print(f"删除缺失值后行数: {len(df_clean)}")

    # 构建特征
    df_clean['-LogP'] = -df_clean['LogP']
    df_clean['HBD_HBA_sum'] = df_clean['HBD'] + df_clean['HBA']

    X = df_clean[['-LogP', 'HBD_HBA_sum', 'predicted_logS_water_298K', 'δ']].values
    y = df_clean['是否可水溶'].apply(lambda x: 1 if str(x).strip() == '是' else 0).values

    # 归一化（全量数据用于后续模型训练和交叉验证）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==================== 10折交叉验证（累积所有预测结果）====================
    print("\n===== 开始10折交叉验证 =====")
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy_list = []
    auc_list = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    all_fpr_tpr = []  # 存储每折的(fpr, tpr)用于绘图

    # 累积所有测试集的真实标签和预测标签（用于最终混淆矩阵）
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_scaled, y), 1):
        X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        model_fold = LogisticRegression(penalty='l2', C=10, random_state=42, solver='liblinear')
        model_fold.fit(X_train_fold, y_train_fold)

        y_pred_fold = model_fold.predict(X_test_fold)
        acc = accuracy_score(y_test_fold, y_pred_fold)
        accuracy_list.append(acc)

        # 累积真实和预测
        all_y_true.extend(y_test_fold)
        all_y_pred.extend(y_pred_fold)

        y_proba_fold = model_fold.predict_proba(X_test_fold)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_fold, y_proba_fold)
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)

        # 插值到统一的横坐标以便计算平均ROC
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        all_fpr_tpr.append((fpr, tpr))

        print(f"Fold {fold:2d} | Accuracy: {acc:.4f} | AUC: {roc_auc:.4f}")

    mean_accuracy = np.mean(accuracy_list)
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)

    # 累积准确率（基于所有测试样本）
    cumulative_accuracy = accuracy_score(all_y_true, all_y_pred)
    print(f"\n10折交叉验证平均准确率: {mean_accuracy:.4f} ± {np.std(accuracy_list):.4f}")
    print(f"10折交叉验证平均AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"累积所有测试样本的准确率: {cumulative_accuracy:.4f}")

    # 计算平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc_cv = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)

    # ==================== 训练最终模型（全量数据）用于输出Stotal和结果 ====================
    model_full = LogisticRegression(penalty='l2', C=10, random_state=42, solver='liblinear')
    model_full.fit(X_scaled, y)

    w1, w2, w3, w4 = model_full.coef_[0]
    intercept = model_full.intercept_[0]
    weights = np.array([w1, w2, w3, w4])
    feature_names = ['-LogP', 'HBD+HBA', 'logSmonomer', 'δ']

    # 计算 Stotal（基于全量数据）
    scaled_features = scaler.transform(df_clean[['-LogP', 'HBD_HBA_sum', 'predicted_logS_water_298K', 'δ']])
    df_clean['Stotal'] = (
            w1 * scaled_features[:, 0] +
            w2 * scaled_features[:, 1] +
            w3 * scaled_features[:, 2] +
            w4 * scaled_features[:, 3] +
            intercept
    )
    df_clean['预测概率'] = 1 / (1 + np.exp(-df_clean['Stotal']))
    df_clean['预测结果'] = df_clean['预测概率'].apply(lambda x: '是' if x >= 0.5 else '否')

    # ==================== 绘制交叉验证的ROC曲线 ====================
    plt.figure(figsize=(8, 6))
    # 绘制所有折的ROC曲线（半透明）
    for (fpr, tpr) in all_fpr_tpr:
        plt.plot(fpr, tpr, color='lightblue', lw=1, alpha=0.3)
    # 绘制平均ROC曲线
    plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label=f'Mean ROC (AUC = {mean_auc_cv:.3f} ± {std_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Guess')
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2, color='orange', label='±1 std')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'10-Fold Cross-Validation ROC Curves\nMean AUC = {mean_auc_cv:.3f}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 保存交叉验证ROC图
    if output_path:
        cv_roc_path = output_path.replace('.csv', '_交叉验证ROC.png')
    else:
        cv_roc_path = os.path.join(os.path.dirname(csv_path), "逻辑回归_交叉验证ROC.png")
    plt.savefig(cv_roc_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"交叉验证ROC曲线已保存: {cv_roc_path}")

    # ==================== 绘制综合分析图（包含累积混淆矩阵） ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'逻辑回归综合结果 (10折交叉验证累积准确率={cumulative_accuracy:.3f})', fontsize=20, y=0.95)

    # 1.1 权重系数
    ax1 = axes[0, 0]
    sns.barplot(x=feature_names, y=np.abs(weights), ax=ax1, palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_title('特征权重系数', fontsize=14)
    ax1.set_ylabel('权重绝对值')
    for i, v in enumerate(weights):
        ax1.text(i, abs(v) + 0.05, f'{v:.2f}', ha='center', fontsize=10)

    # 1.2 Stotal 分布
    ax2 = axes[0, 1]
    sns.histplot(df_clean['Stotal'], bins=20, kde=False, ax=ax2, color='#6a5acd')
    ax2.set_title('Stotal 分布')

    # 1.3 预测概率分布
    ax3 = axes[1, 0]
    sns.histplot(df_clean['预测概率'], bins=20, kde=False, ax=ax3, color='#9ACD32')
    ax3.axvline(0.5, color='red', linestyle='--', label='阈值0.5')
    ax3.set_title('预测概率分布')
    ax3.legend()

    # 1.4 累积混淆矩阵（基于10折所有测试样本）
    ax4 = axes[1, 1]
    cm_cumulative = confusion_matrix(all_y_true, all_y_pred)
    sns.heatmap(cm_cumulative, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['否', '是'], yticklabels=['否', '是'])
    ax4.set_title(f'混淆矩阵 (10折累积, 总样本={len(all_y_true)})')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_path:
        img_path = output_path.replace('.csv', '_综合分析图.png')
    else:
        img_path = os.path.join(os.path.dirname(csv_path), "逻辑回归_综合分析图.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.show()

    # ==================== 保存结果数据 ====================
    result_df = df_clean.copy()
    result_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"结果已保存: {output_path}")

    # 保存权重
    weights_df = pd.DataFrame({
        '特征': feature_names,
        '权重值': [w1, w2, w3, w4],
        '绝对值': np.abs(weights)
    })
    weights_path = output_path.replace('.csv', '_权重.csv')
    weights_df.to_csv(weights_path, index=False, encoding='utf-8')
    print(f"权重(含δ)已保存: {weights_path}")

    # 打印最终公式
    print("\n===== 最终 Stotal 计算公式（基于全量数据） =====")
    print(
        f"Stotal = {w1:.4f}*z(-LogP) + {w2:.4f}*z(HBD+HBA) + {w3:.4f}*z(logSmonomer) + {w4:.4f}*z(δ) + {intercept:.4f}")

    return {
        'model': model_full,
        'weights': {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'intercept': intercept},
        'cv_accuracy_mean': mean_accuracy,
        'cv_auc_mean': mean_auc,
        'cumulative_accuracy': cumulative_accuracy,
        'confusion_matrix': cm_cumulative
    }


# 主程序
if __name__ == "__main__":
    input_csv = r"E:\Python\pythonProject\new_t_predict\data\二分类聚合物_with_delta.csv"
    output_csv = r"E:\Python\pythonProject\new_t_predict\data\二分类聚合物_逻辑回归结果.csv"

    if os.path.exists(input_csv):
        results = calculate_stotal_and_logistic_regression(input_csv, output_csv)
    else:
        print("文件不存在")
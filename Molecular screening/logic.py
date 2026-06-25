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
    Calculate Stotal and perform logistic regression analysis (with 10-fold cross-validation, confusion matrix accumulates all folds)
    Formula: Stotal = w1*z(-LogP) + w2*z(HBD+HBA) + w3*z(logSmonomer) + w4*z(δ)
    Includes: cross-validation ROC curve, cumulative confusion matrix, weight bar chart, Stotal distribution, probability distribution
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    print(f"Read file: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='gbk')
        except:
            print("Unable to read CSV file, please check encoding")
            return None

    required_columns = ['LogP', 'HBD', 'HBA', 'predicted_logS_water_298K', 'water_soluble', 'δ']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None

    df_clean = df.dropna(subset=required_columns).copy()
    print(f"Rows after deleting missing values: {len(df_clean)}")

    # Build features
    df_clean['-LogP'] = -df_clean['LogP']
    df_clean['HBD_HBA_sum'] = df_clean['HBD'] + df_clean['HBA']

    X = df_clean[['-LogP', 'HBD_HBA_sum', 'predicted_logS_water_298K', 'δ']].values
    y = df_clean['water_soluble'].apply(lambda x: 1 if str(x).strip() == 'Yes' else 0).values

    # Normalization (full data used for subsequent model training and cross-validation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==================== 10-fold cross-validation (accumulate all prediction results) ====================
    print("\n===== Start 10-fold cross-validation =====")
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy_list = []
    auc_list = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    all_fpr_tpr = []  # Store (fpr, tpr) of each fold for plotting

    # Accumulate all test set true labels and predicted labels (for final confusion matrix)
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

        # Accumulate true and predicted
        all_y_true.extend(y_test_fold)
        all_y_pred.extend(y_pred_fold)

        y_proba_fold = model_fold.predict_proba(X_test_fold)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_fold, y_proba_fold)
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)

        # Interpolate to uniform x-axis for calculating mean ROC
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        all_fpr_tpr.append((fpr, tpr))

        print(f"Fold {fold:2d} | Accuracy: {acc:.4f} | AUC: {roc_auc:.4f}")

    mean_accuracy = np.mean(accuracy_list)
    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list)

    # Cumulative accuracy (based on all test samples)
    cumulative_accuracy = accuracy_score(all_y_true, all_y_pred)
    print(f"\n10-fold cross-validation average accuracy: {mean_accuracy:.4f} ± {np.std(accuracy_list):.4f}")
    print(f"10-fold cross-validation average AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Accuracy of all accumulated test samples: {cumulative_accuracy:.4f}")

    # Calculate mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc_cv = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)

    # ==================== Train final model (full data) for outputting Stotal and results ====================
    model_full = LogisticRegression(penalty='l2', C=10, random_state=42, solver='liblinear')
    model_full.fit(X_scaled, y)

    w1, w2, w3, w4 = model_full.coef_[0]
    intercept = model_full.intercept_[0]
    weights = np.array([w1, w2, w3, w4])
    feature_names = ['-LogP', 'HBD+HBA', 'logSmonomer', 'δ']

    # Calculate Stotal (based on full data)
    scaled_features = scaler.transform(df_clean[['-LogP', 'HBD_HBA_sum', 'predicted_logS_water_298K', 'δ']])
    df_clean['Stotal'] = (
            w1 * scaled_features[:, 0] +
            w2 * scaled_features[:, 1] +
            w3 * scaled_features[:, 2] +
            w4 * scaled_features[:, 3] +
            intercept
    )
    df_clean['predicted_probability'] = 1 / (1 + np.exp(-df_clean['Stotal']))
    df_clean['predicted_result'] = df_clean['predicted_probability'].apply(lambda x: 'Yes' if x >= 0.5 else 'No')

    # ==================== Plot cross-validation ROC curves ====================
    plt.figure(figsize=(8, 6))
    # Plot all fold ROC curves (semi-transparent)
    for (fpr, tpr) in all_fpr_tpr:
        plt.plot(fpr, tpr, color='lightblue', lw=1, alpha=0.3)
    # Plot mean ROC curve
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

    # Save cross-validation ROC plot
    if output_path:
        cv_roc_path = output_path.replace('.csv', '_cv_roc.png')
    else:
        cv_roc_path = os.path.join(os.path.dirname(csv_path), "logistic_regression_cv_roc.png")
    plt.savefig(cv_roc_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Cross-validation ROC curve saved: {cv_roc_path}")

    # ==================== Plot comprehensive analysis chart (including cumulative confusion matrix) ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Logistic Regression Comprehensive Results (10-fold CV cumulative accuracy={cumulative_accuracy:.3f})', fontsize=20, y=0.95)

    # 1.1 Weight coefficients
    ax1 = axes[0, 0]
    sns.barplot(x=feature_names, y=np.abs(weights), ax=ax1, palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_title('Feature Weight Coefficients', fontsize=14)
    ax1.set_ylabel('Absolute Weight')
    for i, v in enumerate(weights):
        ax1.text(i, abs(v) + 0.05, f'{v:.2f}', ha='center', fontsize=10)

    # 1.2 Stotal distribution
    ax2 = axes[0, 1]
    sns.histplot(df_clean['Stotal'], bins=20, kde=False, ax=ax2, color='#6a5acd')
    ax2.set_title('Stotal Distribution')

    # 1.3 Predicted probability distribution
    ax3 = axes[1, 0]
    sns.histplot(df_clean['predicted_probability'], bins=20, kde=False, ax=ax3, color='#9ACD32')
    ax3.axvline(0.5, color='red', linestyle='--', label='Threshold 0.5')
    ax3.set_title('Predicted Probability Distribution')
    ax3.legend()

    # 1.4 Cumulative Confusion Matrix (based on all 10-fold test samples)
    ax4 = axes[1, 1]
    cm_cumulative = confusion_matrix(all_y_true, all_y_pred)
    sns.heatmap(cm_cumulative, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    ax4.set_title(f'Confusion Matrix (10-fold cumulative, total samples={len(all_y_true)})')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_path:
        img_path = output_path.replace('.csv', '_comprehensive_analysis.png')
    else:
        img_path = os.path.join(os.path.dirname(csv_path), "logistic_regression_comprehensive_analysis.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.show()

    # ==================== Save result data ====================
    result_df = df_clean.copy()
    result_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved: {output_path}")

    # Save weights
    weights_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': [w1, w2, w3, w4],
        'Absolute': np.abs(weights)
    })
    weights_path = output_path.replace('.csv', '_weights.csv')
    weights_df.to_csv(weights_path, index=False, encoding='utf-8')
    print(f"Weights (including δ) saved: {weights_path}")

    # Print final formula
    print("\n===== Final Stotal calculation formula (based on full data) =====")
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


# Main program
if __name__ == "__main__":
    input_csv = r"E:\Python\pythonProject\new_t_predict\data\binary_classification_polymers_with_delta.csv"
    output_csv = r"E:\Python\pythonProject\new_t_predict\data\binary_classification_polymers_logistic_regression_results.csv"

    if os.path.exists(input_csv):
        results = calculate_stotal_and_logistic_regression(input_csv, output_csv)
    else:
        print("File does not exist")
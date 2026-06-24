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

    # 1. Load data
    print("Loading data...")
    file_path = "E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv"

    for encoding in ['gbk', 'gb2312', 'gb18030', 'utf-8', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding, skiprows=1,
                             header=None, names=['smiles', 'target'])
            print(f"Successfully used {encoding} encoding")
            break
        except:
            continue

    # 2. Clean data
    print("Cleaning data...")
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    initial_count = len(df)
    df = df.dropna(subset=['target'])
    print(f"Deleted invalid target values: {initial_count - len(df)} rows")

    if len(df) == 0:
        print("Error: No valid data!")
        return

    print(f"Valid data: {len(df)} rows")
    print(f"Target value statistics: Mean={df['target'].mean():.2f}, Std={df['target'].std():.2f}")
    print(f"Target value range: {df['target'].min():.2f} - {df['target'].max():.2f}")

    # 3. Generate features
    print("Generating features...")
    X, y = [], []

    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row['smiles']))
            if mol:
                # Use radius-2 fingerprint
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
                arr = np.zeros(2048, dtype=int)
                ConvertToNumpyArray(fp, arr)
                X.append(arr)
                y.append(row['target'])
        except:
            continue

    X = np.array(X)
    y = np.array(y)

    print(f"Feature matrix shape: {X.shape}")

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

    # 5. Train Random Forest model
    print("\nTraining Random Forest model...")
    print("Model parameters:")
    print("  n_estimators: 200")
    print("  max_depth: None (no tree depth limit)")
    print("  min_samples_split: 2")
    print("  min_samples_leaf: 1")
    print("  max_features: 'sqrt' (use sqrt(n_features) features)")

    start_train_time = time.time()

    # Use appropriate parameters
    rf = RandomForestRegressor(
        n_estimators=200,  # Number of trees
        max_depth=None,  # No tree depth limit
        min_samples_split=2,  # Min samples to split internal node
        min_samples_leaf=1,  # Min samples at leaf node
        max_features='sqrt',  # Use sqrt(n_features) features, this is the correct parameter
        bootstrap=True,  # Use bootstrap sampling
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    )

    rf.fit(X_train, y_train)

    train_time = time.time() - start_train_time
    print(f"Training completed, time elapsed: {train_time:.2f} seconds")

    # 6. Predict
    print("Making predictions...")
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # 7. Compute metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    overfit = train_r2 - test_r2

    # 8. Output results
    print("\n" + "=" * 60)
    print("Random Forest Model Performance")
    print("=" * 60)
    print(f"Training set R²:  {train_r2:.4f}")
    print(f"Training set MAE: {train_mae:.4f}")
    print(f"Test set R²:  {test_r2:.4f}")
    print(f"Test set MAE: {test_mae:.4f}")
    print(f"Overfitting degree:  {overfit:.4f}")
    print("=" * 60)

    # 9. Cross validation
    print("\nPerform 5-fold cross-validation...")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"Cross-validation R² scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"Cross-validation average R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 10. Feature importance analysis
    feature_importances = rf.feature_importances_
    print(f"\nFeature importance statistics:")
    print(f"  Mean: {feature_importances.mean():.6f}")
    print(f"  Max: {feature_importances.max():.6f}")
    print(f"  Min: {feature_importances.min():.6f}")

    # Find most important features
    top_10_idx = np.argsort(feature_importances)[-10:]
    print(f"Top 10 most important feature indices: {top_10_idx}")

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")

    # 11. Performance analysis
    print("\n[Performance Analysis]")
    if train_r2 < 0.7:
        print("⚠️  Training set underfitting, suggestions:")
        print("  1. Increase n_estimators to 300-500")
        print("  2. Remove max_depth limit")
        print("  3. Try using radius-3 fingerprint (radius=3)")
        print("  4. Try other models like XGBoost or neural networks")
    elif train_r2 < 0.85:
        print("✅ Training set moderate fit")
    else:
        print("✅ Training set good fit")

    if test_r2 < 0.6:
        print("⚠️  Test set insufficient generalization, suggestions:")
        print("  1. Increase regularization parameters (min_samples_split, min_samples_leaf)")
        print("  2. Use more training data")
        print("  3. Try feature selection or dimensionality reduction")
    else:
        print("✅ Test set good generalization")


if __name__ == "__main__":
    main()

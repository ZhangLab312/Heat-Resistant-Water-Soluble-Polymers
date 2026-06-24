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
    # 1. Load data
    file_path = "E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv"

    for encoding in ['gbk', 'gb2312', 'gb18030']:
        try:
            df = pd.read_csv(file_path, encoding=encoding, skiprows=1,
                             header=None, names=['smiles', 'target'])
            break
        except:
            continue

    # 2. Clean data
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df = df.dropna(subset=['target'])

    # 3. Generate features
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

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Use grid search to find best ridge regression parameters
    print("\nUse grid search to find best ridge regression parameters...")

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

    print(f"Best alpha value: {grid_search.best_params_['alpha']}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.4f} (MSE)")

    # 7. Use best model
    best_ridge = grid_search.best_estimator_

    # Predict
    y_train_pred = best_ridge.predict(X_train_scaled)
    y_test_pred = best_ridge.predict(X_test_scaled)

    # 8. Compute metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 9. Output results
    print("\n" + "=" * 60)
    print("Ridge Regression Model Performance (auto-tuned)")
    print("=" * 60)
    print(f"Training set R²:  {train_r2:.4f}")
    print(f"Training set MAE: {train_mae:.4f}")
    print(f"Test set R²:  {test_r2:.4f}")
    print(f"Test set MAE: {test_mae:.4f}")
    print(f"Overfitting degree:  {train_r2 - test_r2:.4f}")
    print("=" * 60)

    # 10. Feature importance
    print(f"\nModel intercept: {best_ridge.intercept_:.4f}")
    coefficients = best_ridge.coef_
    print(f"Number of non-zero coefficients: {np.sum(coefficients != 0)}/{len(coefficients)}")
    print(f"Mean absolute coefficient: {np.abs(coefficients).mean():.6f}")

    # Find most important features
    top_n = 10
    abs_coef = np.abs(coefficients)
    top_indices = np.argsort(abs_coef)[-top_n:]

    print(f"\nTop {top_n} most important features (largest absolute coefficient):")
    for i, idx in enumerate(reversed(top_indices)):
        print(f"  Feature {idx}: Coefficient = {coefficients[idx]:.6f}")


if __name__ == "__main__":
    main()

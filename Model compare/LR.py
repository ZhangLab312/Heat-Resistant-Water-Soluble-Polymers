import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
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
                X.append(list(fp))
                y.append(row['target'])
        except:
            continue

    X, y = np.array(X), np.array(y)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Standardize features (very important!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train ordinary linear regression
    print("Training ordinary linear regression...")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train_scaled, y_train)

    # 7. Predict and evaluate
    y_train_pred = lr.predict(X_train_scaled)
    y_test_pred = lr.predict(X_test_scaled)

    # 8. Output results
    print(f"Training set R²: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Training set MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"Test set R²: {r2_score(y_test, y_test_pred):.4f}")
    print(f"Test set MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")

    # 9. Model info
    print(f"\nModel intercept: {lr.intercept_:.4f}")
    print(f"Number of non-zero coefficients: {np.sum(lr.coef_ != 0)}/{len(lr.coef_)}")


if __name__ == "__main__":
    main()

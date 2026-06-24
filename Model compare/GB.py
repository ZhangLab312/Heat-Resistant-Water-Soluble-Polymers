import pandas as pd
import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor


def main():
    print("Starting program...")

    try:
        # 1. Load data
        print("1. Loading data...")
        df = pd.read_csv("E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv",
                         encoding='gbk', skiprows=1, header=None,
                         names=['smiles', 'target'])
        print(f"Data loaded successfully, total {len(df)} rows")

        # 2. Clean data
        print("2. Cleaning data...")
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df = df.dropna(subset=['target'])
        print(f"Data after cleaning: {len(df)} rows")

        # 3. Generate features - only first 50 samples
        print("3. Generating features (first 50 samples only)...")
        X, y = [], []
        for i in range(len(df)):
            row = df.iloc[i]
            mol = Chem.MolFromSmiles(str(row['smiles']))
            if mol:
                # Only use 512-bit features
                fp = Chem.RDKFingerprint(mol, fpSize=2048)
                X.append(list(fp))
                y.append(row['target'])
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1} molecules")

        if not X:
            print("Error: No features generated")
            return

        X, y = np.array(X), np.array(y)

        # 4. Split data
        print("4. Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. Train model
        print("5. Training model...")
        model = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # 6. Predict and evaluate
        print("6. Evaluating model...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        print("\nResults:")
        print(f"Training set R²: {r2_score(y_train, y_train_pred):.4f}")
        print(f"Training set MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
        print(f"Test set R²: {r2_score(y_test, y_test_pred):.4f}")
        print(f"Test set MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")

    except Exception as e:
        print(f"Program error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

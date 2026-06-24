import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import time


def main():
    start_time = time.time()

    # 1. Load data
    print("Loading data...")
    file_path = "E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv"

    for encoding in ['gbk', 'gb2312', 'gb18030']:
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
    df = df.dropna(subset=['target'])

    print(f"Valid data: {len(df)} rows")
    print(f"Target value range: {df['target'].min():.2f} - {df['target'].max():.2f}")

    # 3. Generate features
    print("Generating features...")
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

    X = np.array(X)
    y = np.array(y)

    print(f"Feature matrix shape: {X.shape}")

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

    # 5. Train decision tree model - try different depths
    print("\nTrain decision tree model...")

    # Try different max depths
    depths = [5, 10, 15, 20, None]  # None means no depth limit

    for depth in depths:
        print(f"\nTrying max depth: {depth}")

        dt = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_split=10,  # Prevent overfitting
            min_samples_leaf=5,  # Prevent overfitting
            random_state=42
        )

        dt.fit(X_train, y_train)

        # Predict
        y_train_pred = dt.predict(X_train)
        y_test_pred = dt.predict(X_test)

        # Compute metrics
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Output results
        print(f"  Training set R²:  {train_r2:.4f}")
        print(f"  Training set MAE: {train_mae:.4f}")
        print(f"  Test set R²:  {test_r2:.4f}")
        print(f"  Test set MAE: {test_mae:.4f}")
        print(f"  Overfitting degree:  {train_r2 - test_r2:.4f}")

        # Output tree info
        if dt.tree_ is not None:
            print(f"  Number of tree nodes: {dt.tree_.node_count}")

    # 6. Recommend best depth
    print("\n" + "=" * 60)
    print("Decision Tree Model Performance Summary")
    print("=" * 60)
    print("Suggestion: Choose the depth with highest test R² and lowest overfitting")
    print("Note: Decision trees are prone to overfitting, recommend using appropriate regularization parameters")

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import os


# Feature generation function (optional fingerprint type)
def generate_fingerprints(smiles_list, fp_type='morgan', nBits=1024, radius=2):
    """
    Generate molecular fingerprints
    fp_type: 'morgan' or 'maccs'
    """
    features = []
    valid_smiles = []
    valid_targets = []

    for smiles, target in smiles_list:
        try:
            smiles = str(smiles).strip()
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            if fp_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
                arr = np.zeros((nBits,), dtype=int)
            elif fp_type == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros((166,), dtype=int)  # MACCS fixed 166 bits
            else:
                raise ValueError("fp_type must be 'morgan' or 'maccs'")

            ConvertToNumpyArray(fp, arr)
            features.append(arr)
            valid_smiles.append(smiles)
            valid_targets.append(target)
        except:
            continue

    return np.array(features), np.array(valid_targets), valid_smiles


# Save training and test sets
def save_train_test_split(train_df, test_df, output_dir, train_filename="KNN_train.csv", test_filename="KNN_test.csv"):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, train_filename)
    test_path = os.path.join(output_dir, test_filename)
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    print(f"Training set saved to: {train_path}")
    print(f"Test set saved to: {test_path}")


def main():
    try:
        # 1. Load data
        print("Loading data...")
        file_path = "E:/Python/pythonProject/new_t_predict/data/cleaned_predictions.csv"
        encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, skiprows=1,
                                 header=None, names=['smiles', 'target'])
                print(f"Successfully loaded data with {encoding} encoding")
                break
            except:
                continue
        if df is None:
            df = pd.read_csv(file_path, skiprows=1, header=None, names=['smiles', 'target'])
            print("Successfully loaded data with default encoding")

        print(f"Original data shape: {df.shape}")

        # 2. Clean target values
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df = df.dropna(subset=['target'])
        print(f"Deleted invalid target values: {len(df)} rows remaining")

        # 3. [Improvement 1] Deduplicate based on SMILES (keep first)
        df = df.drop_duplicates(subset=['smiles'], keep='first')
        print(f"Remaining after deduplication: {len(df)} unique SMILES")

        if len(df) == 0:
            print("Error: No valid data!")
            return

        # 4. [Improvement 2] Generate fingerprints (reduce dimensions, use 1024-bit Morgan or optional MACCS)
        print("Generating molecular fingerprint features...")
        # Can switch fingerprint type here: 'morgan' or 'maccs'
        fp_type = 'morgan'  # Change to 'maccs' to further reduce to 166 bits
        if fp_type == 'morgan':
            nBits = 1024  # Original 2048 -> 1024, reduce overfitting
            radius = 3
            print(f"Using Morgan fingerprint, bits={nBits}, radius={radius}")
        else:
            nBits = 166  # MACCS fixed
            print("Using MACCS fingerprint (166 bits)")

        # Prepare input list
        smiles_target_pairs = list(zip(df['smiles'], df['target']))
        X, y, valid_smiles = generate_fingerprints(smiles_target_pairs, fp_type=fp_type, nBits=nBits, radius=3)

        print(f"Successfully generated fingerprints: {len(X)} molecules")
        if len(X) == 0:
            print("Error: Unable to generate any valid features!")
            return

        # Build valid DataFrame
        valid_df = pd.DataFrame({'smiles': valid_smiles, 'target': y})

        # 5. Split train/test set (stratified sampling optional, using simple random here)
        indices = np.arange(len(valid_df))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        train_df = valid_df.iloc[train_idx].reset_index(drop=True)
        test_df = valid_df.iloc[test_idx].reset_index(drop=True)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        # Save train/test set (same as before)
        output_dir = r"E:\Python\pythonProject\new_t_predict\画图数据"
        save_train_test_split(train_df, test_df, output_dir,
                              train_filename="KNN_train.csv",
                              test_filename="KNN_test.csv")

        # 6. Standardize features
        print("Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 7. [Improvement 3] Expand K value range + grid search cross validation
        print("Using cross-validation to search for best K value...")
        param_grid = {
            'n_neighbors': [10, 20, 30, 40, 50, 75, 100],  # Avoid small K values
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # Manhattan and Euclidean distances
        }
        knn_base = KNeighborsRegressor(n_jobs=-1)
        grid_search = GridSearchCV(knn_base, param_grid, cv=5,
                                   scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        best_knn = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

        # 8. Evaluate final model
        y_train_pred = best_knn.predict(X_train_scaled)
        y_test_pred = best_knn.predict(X_test_scaled)

        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Output results
        print("\n" + "=" * 60)
        print("Improved KNN Model Performance Evaluation")
        print("=" * 60)
        print(f"Training set R²:  {train_r2:.4f}")
        print(f"Training set MAE: {train_mae:.4f}")
        print(f"Test set R²:  {test_r2:.4f}")
        print(f"Test set MAE: {test_mae:.4f}")
        print("=" * 60)
        print(f"Training set target mean: {y_train.mean():.4f}")
        print(f"Training set prediction mean: {y_train_pred.mean():.4f}")
        print(f"Test set target mean: {y_test.mean():.4f}")
        print(f"Test set prediction mean: {y_test_pred.mean():.4f}")

        # Optional: Output cross-validation result summary
        cv_results = grid_search.cv_results_
        print(f"\nBest cross-validation average R²: {grid_search.best_score_:.4f}")

    except Exception as e:
        print(f"Program runtime error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

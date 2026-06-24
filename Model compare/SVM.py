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
    print("SVM Model Training")
    print("=" * 60)

    # 1. Load data
    print("\nStep 1: Loading data...")
    file_path = "E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv"

    # Try different encodings
    for encoding in ['gbk', 'gb2312', 'gb18030', 'utf-8', 'latin1']:
        try:
            df = pd.read_csv(file_path, encoding=encoding, skiprows=1,
                             header=None, names=['smiles', 'target'])
            print(f"✓ Successfully used {encoding} encoding")
            break
        except:
            continue

    # 2. Clean data
    print("\nStep 2: Cleaning data...")
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df = df.dropna(subset=['target'])

    print(f"Valid data: {len(df)} rows")
    print(f"Target value statistics: Mean={df['target'].mean():.2f}, Std={df['target'].std():.2f}")

    # 3. Generate features - use fewer feature dimensions to speed up SVM training
    print("\nStep 3: Generating features...")
    print("Tip: SVM training is slow, using 512-bit features to speed up")

    X, y = [], []
    valid_count = 0

    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row['smiles']))
            if mol:
                # Use fewer feature dimensions
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
                X.append(list(fp))
                y.append(row['target'])
                valid_count += 1

                if valid_count % 500 == 0:
                    print(f"  Processed {valid_count} molecules...")
        except:
            continue

    X = np.array(X)
    y = np.array(y)

    print(f"Feature generation completed! {valid_count} valid samples")
    print(f"Feature dimensions: {X.shape[1]}")

    if valid_count == 0:
        print("Error: No valid features generated!")
        return

    # 4. Split dataset
    print("\nStep 4: Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 5. Standardize features (SVM is sensitive to feature scaling)
    print("\nStep 5: Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train SVM model
    print("\nStep 6: Training SVM model...")
    print("SVM parameters:")
    print("  kernel: 'rbf' (Radial basis function kernel)")
    print("  C: 1.0 (Regularization parameter)")
    print("  epsilon: 0.1 (Epsilon parameter of epsilon-SVR)")
    print("  Note: SVM training may be slow...")

    # Create SVR model
    svm_model = SVR(
        kernel='rbf',  # Radial basis function kernel
        C=1.0,  # Regularization parameter
        epsilon=0.1,  # Epsilon parameter of epsilon-SVR
        gamma='scale',  # Kernel function coefficient
        cache_size=500,  # Cache size (MB)
        verbose=True,  # Show training progress
        max_iter=-1  # Max iterations (-1 means unlimited)
    )

    # Train model
    train_start = time.time()
    svm_model.fit(X_train_scaled, y_train)
    train_time = time.time() - train_start

    print(f"\nTraining completed! Time elapsed: {train_time:.2f} seconds")

    # 7. Predict
    print("\nStep 7: Making predictions...")
    y_train_pred = svm_model.predict(X_train_scaled)
    y_test_pred = svm_model.predict(X_test_scaled)

    # 8. Compute performance metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 9. Output results
    print("\n" + "=" * 60)
    print("SVM Model Performance Evaluation")
    print("=" * 60)
    print(f"Training set R²:  {train_r2:.4f}")
    print(f"Training set MAE: {train_mae:.4f}")
    print(f"Test set R²:  {test_r2:.4f}")
    print(f"Test set MAE: {test_mae:.4f}")
    print(f"Overfitting degree:  {train_r2 - test_r2:.4f}")
    print("=" * 60)

    # 10. Total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")

    # 11. Model parameter details
    print(f"\nModel parameters:")
    print(f"  Number of support vectors: {len(svm_model.support_)}")
    print(f"  Support vector ratio: {len(svm_model.support_) / len(X_train_scaled) * 100:.1f}%")


if __name__ == "__main__":
    main()

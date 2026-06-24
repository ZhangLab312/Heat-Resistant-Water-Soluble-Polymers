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
    print("XGBoost Model Training")
    print("=" * 60)

    # 1. Load data
    print("\nStep 1: Loading data...")
    file_path = "E:/Python/pythonProject/new_t_predict/data/Tm_raw.csv"

    # Try different encodings
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'latin1']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, skiprows=1,
                             header=None, names=['smiles', 'target'])
            print(f"✓ Successfully loaded data with {encoding} encoding")
            break
        except Exception as e:
            continue

    if df is None:
        try:
            df = pd.read_csv(file_path, skiprows=1, header=None,
                             names=['smiles', 'target'])
            print("✓ Successfully loaded data with default encoding")
        except Exception as e:
            print(f"✗ Unable to load file: {e}")
            return

    print(f"Original data volume: {len(df)} rows")

    # 2. Clean data
    print("\nStep 2: Cleaning data...")
    df['target'] = pd.to_numeric(df['target'], errors='coerce')

    initial_count = len(df)
    df = df.dropna(subset=['target'])
    cleaned_count = len(df)

    print(f"Deleted invalid target values: {initial_count - cleaned_count} rows")
    print(f"Valid data: {cleaned_count} rows")
    print(f"Target value statistics:")
    print(f"  Mean: {df['target'].mean():.2f}")
    print(f"  Std: {df['target'].std():.2f}")
    print(f"  Min: {df['target'].min():.2f}")
    print(f"  Max: {df['target'].max():.2f}")

    # 3. Generate features
    print("\nStep 3: Generating molecular fingerprint features...")
    print("This may take some time, please wait patiently...")

    X, y = [], []
    valid_count = 0
    total_count = len(df)

    for idx, row in df.iterrows():
        try:
            # Process SMILES
            smiles = str(row['smiles']).strip()
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Generate Morgan fingerprint with more reasonable parameters
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius=3,  # Reduce radius to speed up
                    nBits=2048  # Reduce bits to speed up training
                )

                # Convert fingerprint to list
                features = list(fp)
                X.append(features)
                y.append(row['target'])
                valid_count += 1

                # Show progress
                if valid_count % 500 == 0:
                    print(f"  Processed {valid_count}/{total_count} molecules...")
        except Exception as e:
            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"\nFeature generation completed!")
    print(f"Successfully generated features: {valid_count} molecules")
    print(f"Feature dimensions: {X.shape[1]}")

    if valid_count == 0:
        print("Error: No valid features generated!")
        return

    # 4. Split training and test sets
    print("\nStep 4: Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 5. Train XGBoost model
    print("\nStep 5: Training XGBoost model...")
    print("XGBoost parameters:")
    print("  n_estimators: 200")
    print("  max_depth: 6")
    print("  learning_rate: 0.1")
    print("  subsample: 0.8")
    print("  colsample_bytree: 0.8")
    print("  random_state: 42")

    # Create XGBoost model
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,  # Number of trees
        max_depth=6,  # Max tree depth
        learning_rate=0.1,  # Learning rate
        subsample=0.8,  # Sample subsample ratio
        colsample_bytree=0.8,  # Feature subsample ratio
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        random_state=42,  # Random seed
        n_jobs=-1,  # Use all CPU cores
        verbosity=0  # Silent mode
    )

    # Train model
    print("\nStart training...")
    train_start = time.time()
    xgb_model.fit(X_train, y_train)
    train_time = time.time() - train_start

    print(f"Training completed! Time elapsed: {train_time:.2f} seconds")

    # 6. Predict
    print("\nStep 6: Making predictions...")

    # Training set prediction
    y_train_pred = xgb_model.predict(X_train)
    # Test set prediction
    y_test_pred = xgb_model.predict(X_test)

    # 7. Compute performance metrics
    print("\nStep 7: Computing performance metrics...")

    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # 8. Output results
    print("\n" + "=" * 60)
    print("XGBoost Model Performance Evaluation")
    print("=" * 60)
    print(f"Training set R²:  {train_r2:.4f}")
    print(f"Training set MAE: {train_mae:.4f}")
    print(f"Test set R²:  {test_r2:.4f}")
    print(f"Test set MAE: {test_mae:.4f}")
    print(f"Overfitting degree (Train R² - Test R²): {train_r2 - test_r2:.4f}")
    print("=" * 60)

    # 9. Feature importance analysis
    print("\nFeature Importance Analysis:")
    feature_importances = xgb_model.feature_importances_

    # Feature importance statistics
    print(f"Average feature importance: {feature_importances.mean():.6f}")
    print(f"Max feature importance: {feature_importances.max():.6f}")
    print(f"Min feature importance: {feature_importances.min():.6f}")

    # Find most important features
    top_n = 10
    top_indices = np.argsort(feature_importances)[-top_n:]
    print(f"\nTop {top_n} most important feature indices: {top_indices}")

    # 10. Total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")

    # 11. Performance evaluation
    print("\n[Performance Evaluation]")
    if test_r2 >= 0.8:
        print("✅ Model performance excellent!")
    elif test_r2 >= 0.6:
        print("✅ Model performance good")
    elif test_r2 >= 0.4:
        print("⚠️  Model performance average")
    else:
        print("⚠️  Model performance poor, suggestions:")
        print("  1. Increase data volume")
        print("  2. Adjust model parameters")
        print("  3. Try other feature representation methods")
        print("  4. Consider using more complex model architecture")


if __name__ == "__main__":
    main()

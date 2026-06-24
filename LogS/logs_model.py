import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
import random

# Suppress RDKit warnings
from rdkit import rdBase

rdBase.DisableLog('rdApp.*')

# Configuration parameters
CONFIG = {
    "input_file": "E:\\Python\\pythonProject\\new_t_predict\\data\\AqueousSolu.csv",  # Please modify to your file path
    "model_path": "E:/Python/pythonProject/new_t_predict/model/logs_model.pth",
    "scaler_path": "E:/Python/pythonProject/new_t_predict/scaler/logs_scaler.pkl",
    "fingerprint": {
        "radius": 2,
        "n_bits": 1024
    },
    "nn_params": {
        "hidden_layers": [512, 256],
        "dropout_rate": 0.5,
        "learning_rate": 0.001,
        "epochs": 600,
        "batch_size": 64,
        "patience": 30,
        "weight_decay": 1e-4
    },
    "data_augmentation": {
        "enabled": True,
        "augment_per_molecule": 4,  # Number of augmented samples per molecule (including original sample)
        "random_seed": 42
    },
    "test_size": 0.2,
    "random_state": 42
}

# Set random seeds
random.seed(CONFIG["random_state"])
np.random.seed(CONFIG["random_state"])
torch.manual_seed(CONFIG["random_state"])


class FNN(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_size = input_size

        for h_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            nn.init.kaiming_normal_(layers[-4].weight, mode='fan_in', nonlinearity='relu')
            prev_size = h_size

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, 1)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)


def get_canonical_smiles(smi):
    """Get canonical SMILES"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        pass
    return None


def generate_augmented_smiles(smi, num_augmentations):
    """Generate augmented samples for a single SMILES"""
    augmented_smiles = []

    # First add original SMILES
    augmented_smiles.append(smi)

    # Then generate augmented samples
    for _ in range(num_augmentations - 1):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Randomize SMILES (data augmentation)
                random_smi = Chem.MolToSmiles(
                    mol,
                    doRandom=True,
                    canonical=False,
                    isomericSmiles=True,
                    kekuleSmiles=True
                )
                augmented_smiles.append(random_smi)
            else:
                # If unable to parse, use original SMILES
                augmented_smiles.append(smi)
        except:
            # If generation fails, use original SMILES
            augmented_smiles.append(smi)

    return augmented_smiles


def generate_features_with_augmentation(smiles_list, y_list):
    """Generate features with data augmentation while recording molecule IDs"""
    features = []
    all_smiles = []
    all_y = []
    molecule_ids = []  # Record which molecule each sample belongs to
    molecule_to_id = {}  # Mapping: canonical SMILES -> molecule ID

    print(f"\n[Data Augmentation Report]")
    print(f"Original SMILES count: {len(smiles_list)}")
    print(f"Number of augmented samples per molecule: {CONFIG['data_augmentation']['augment_per_molecule']}")

    current_molecule_id = 0

    for idx, (smi, y_val) in enumerate(zip(smiles_list, y_list)):
        # Get canonical SMILES
        canonical_smi = get_canonical_smiles(smi)
        if not canonical_smi:
            print(f"Warning: Unable to parse SMILES {smi}, skipping")
            continue

        # If this molecule hasn't been assigned an ID, assign one
        if canonical_smi not in molecule_to_id:
            molecule_to_id[canonical_smi] = current_molecule_id
            current_molecule_id += 1

        molecule_id = molecule_to_id[canonical_smi]

        # Generate augmented SMILES
        augmented_smiles = generate_augmented_smiles(
            smi,
            CONFIG["data_augmentation"]["augment_per_molecule"]
        )

        # Generate features for each augmented SMILES
        for aug_smi in augmented_smiles:
            try:
                mol = Chem.MolFromSmiles(aug_smi)
                if mol:
                    # Generate Morgan fingerprint
                    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
                    fp = GetMorganFingerprintAsBitVect(
                        mol,
                        radius=CONFIG["fingerprint"]["radius"],
                        nBits=CONFIG["fingerprint"]["n_bits"]
                    )
                    arr = np.zeros((CONFIG["fingerprint"]["n_bits"],), dtype=int)
                    ConvertToNumpyArray(fp, arr)

                    features.append(arr)
                    all_smiles.append(aug_smi)
                    all_y.append(y_val)
                    molecule_ids.append(molecule_id)
            except Exception as e:
                print(f"Warning: Failed to process augmented SMILES: {aug_smi}, error: {e}")

    print(f"Total augmented samples generated: {len(features)}")
    print(f"Number of unique molecules: {len(molecule_to_id)}")
    print(f"Average samples per molecule: {len(features) / len(molecule_to_id):.2f}")

    return np.array(features), np.array(all_y), molecule_ids, molecule_to_id


def split_by_molecules(X, y, molecule_ids, test_size=0.2, random_state=42):
    """Split dataset by molecule, ensuring all samples of the same molecule are in the same set"""

    # Get all unique molecule IDs
    unique_molecule_ids = list(set(molecule_ids))
    print(f"Total molecule count: {len(unique_molecule_ids)}")

    # Calculate average y value for each molecule (for stratified sampling)
    molecule_y_values = []
    for mol_id in unique_molecule_ids:
        # Find indices of all samples for this molecule
        sample_indices = [i for i, m_id in enumerate(molecule_ids) if m_id == mol_id]
        # Calculate average y value of all samples for this molecule
        avg_y = np.mean(y[sample_indices])
        molecule_y_values.append(avg_y)

    molecule_y_values = np.array(molecule_y_values)

    # Try stratified sampling, use random split if it fails
    try:
        # Dynamically determine bin count, ensuring each bin has at least 2 molecules
        n_bins = 5  # Start trying with 5 bins
        success = False

        while n_bins >= 2:
            bins = np.linspace(molecule_y_values.min(), molecule_y_values.max(), n_bins + 1)
            y_binned = np.digitize(molecule_y_values, bins) - 1

            # Check molecule count per bin
            bin_counts = np.bincount(y_binned)
            if np.all(bin_counts >= 2):
                success = True
                print(f"Using {n_bins} bins for stratified sampling")
                print(f"Molecule count per bin: {bin_counts}")
                break
            else:
                n_bins -= 1

        if not success:
            print("Unable to perform stratified sampling, using random split")
            train_molecule_ids, test_molecule_ids = train_test_split(
                unique_molecule_ids,
                test_size=test_size,
                random_state=random_state
            )
        else:
            # Use stratified sampling to split molecules
            from sklearn.model_selection import StratifiedShuffleSplit
            split = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state
            )

            for train_idx, test_idx in split.split(unique_molecule_ids, y_binned):
                train_molecule_ids = [unique_molecule_ids[i] for i in train_idx]
                test_molecule_ids = [unique_molecule_ids[i] for i in test_idx]

    except Exception as e:
        print(f"Stratified sampling failed, using random split: {e}")
        train_molecule_ids, test_molecule_ids = train_test_split(
            unique_molecule_ids,
            test_size=test_size,
            random_state=random_state
        )

    print(f"Training set molecule count: {len(train_molecule_ids)}")
    print(f"Test set molecule count: {len(test_molecule_ids)}")

    # Assign samples to training and test sets by molecule ID
    train_indices = []
    test_indices = []

    for idx, mol_id in enumerate(molecule_ids):
        if mol_id in train_molecule_ids:
            train_indices.append(idx)
        else:
            test_indices.append(idx)

    print(f"Training set sample count: {len(train_indices)}")
    print(f"Test set sample count: {len(test_indices)}")

    # Verify: check if any molecule appears in both sets
    train_molecule_set = set(train_molecule_ids)
    test_molecule_set = set(test_molecule_ids)
    overlap = train_molecule_set.intersection(test_molecule_set)

    if overlap:
        print(f"Error: {len(overlap)} molecules appear in both training and test sets!")
        return None, None, None, None
    else:
        print("Verification passed: No molecules appear in both training and test sets")

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def train_fnn(X_train, y_train, X_val, y_val):
    """Train FNN model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print device info
    print(f"\n[Device Info]")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Warning: CUDA device not detected, will use CPU for training (slower)")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train.reshape(-1, 1))
        ),
        batch_size=CONFIG["nn_params"]["batch_size"],
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model
    model = FNN(
        input_size=X_train.shape[1],
        hidden_layers=CONFIG["nn_params"]["hidden_layers"],
        dropout_rate=CONFIG["nn_params"]["dropout_rate"]
    ).to(device)

    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["nn_params"]["learning_rate"],
        weight_decay=CONFIG["nn_params"]["weight_decay"]
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience_counter = 0

    print("\nStart training...")
    for epoch in range(CONFIG["nn_params"]["epochs"]):
        model.train()
        total_loss = 0.0

        # Training step
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        # Validation step
        model.eval()
        with torch.no_grad():
            val_inputs = torch.FloatTensor(X_val_scaled).to(device)
            val_targets = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item()

        # Learning rate adjustment
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_path"])
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["nn_params"]["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Output training progress
        if (epoch + 1) % 20 == 0:
            train_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Load best model
    model.load_state_dict(torch.load(CONFIG["model_path"]))
    return model, scaler


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prediction function
    def predict(X_data):
        with torch.no_grad():
            X_scaled = scaler.transform(X_data)
            tensor = torch.FloatTensor(X_scaled).to(device)
            return model(tensor).cpu().numpy().flatten()

    # Make predictions
    train_pred = predict(X_train)
    test_pred = predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    train_mse = mean_squared_error(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    return train_r2, train_mse, test_r2, test_mse


def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv(CONFIG["input_file"])

    # Check column names
    print(f"Data column names: {df.columns.tolist()}")

    # Assuming your file contains 'smiles solute' and 'logS_aq_avg' columns
    # If column names are different, please modify the column names below
    smiles_col = 'smiles solute' if 'smiles solute' in df.columns else 'smiles'
    target_col = 'logS_aq_avg' if 'logS_aq_avg' in df.columns else 'logS'

    print(f"Using SMILES column: {smiles_col}")
    print(f"Using target column: {target_col}")

    # Prepare data
    smiles_list = df[smiles_col].tolist()
    y_list = df[target_col].values.tolist()

    print(f"\nTotal original data: {len(smiles_list)}")

    # Generate features with data augmentation
    print("\nGenerating Morgan fingerprint features with data augmentation...")
    X, y, molecule_ids, molecule_to_id = generate_features_with_augmentation(smiles_list, y_list)

    # Split dataset by molecule (ensuring all samples of the same molecule are in the same set)
    print("\nSplitting dataset by molecule...")
    X_train, X_test, y_train, y_test = split_by_molecules(
        X, y, molecule_ids,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

    if X_train is None:
        print("Dataset split failed, program exiting")
        return

    print("\n[Dataset Info]")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Total sample count after augmentation: {X.shape[0]}")
    print(f"Training set sample count: {X_train.shape[0]}")
    print(f"Test set sample count: {X_test.shape[0]}")
    print(f"Training/test set ratio: {X_train.shape[0] / X_test.shape[0]:.2f}:1")

    # Train model
    model, scaler = train_fnn(X_train, y_train, X_test, y_test)

    # Save scaler
    joblib.dump(scaler, CONFIG["scaler_path"])
    print(f"\nScaler saved to: {CONFIG['scaler_path']}")

    # Evaluate model
    print("\n=== Model Performance ===")
    train_r2, train_mse, test_r2, test_mse = evaluate_model(
        model, scaler, X_train, y_train, X_test, y_test
    )

    print(f"Training set R²: {train_r2:.4f}")
    print(f"Training set MSE: {train_mse:.4f}")
    print(f"Test set R²: {test_r2:.4f}")
    print(f"Test set MSE: {test_mse:.4f}")

    # Additional statistical analysis
    print("\n=== Data Statistical Analysis ===")
    print(
        f"Training set y value range: [{y_train.min():.2f}, {y_train.max():.2f}], mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    print(
        f"Test set y value range: [{y_test.min():.2f}, {y_test.max():.2f}], mean: {y_test.mean():.2f}, std: {y_test.std():.2f}")

    # Check data augmentation effect
    unique_train_molecules = set()
    unique_test_molecules = set()

    for idx, mol_id in enumerate(molecule_ids):
        # Find which set this sample belongs to
        if idx < len(X_train):
            unique_train_molecules.add(mol_id)
        else:
            unique_test_molecules.add(mol_id)

    print(f"\nTraining set unique molecule count: {len(unique_train_molecules)}")
    print(f"Test set unique molecule count: {len(unique_test_molecules)}")
    print(f"Training set average samples per molecule: {len(X_train) / len(unique_train_molecules):.2f}")
    print(f"Test set average samples per molecule: {len(X_test) / len(unique_test_molecules):.2f}")


if __name__ == "__main__":
    main()
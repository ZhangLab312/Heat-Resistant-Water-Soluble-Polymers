import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from tqdm import tqdm  # Import progress bar library


def standardize_dummy(smiles):
    """Standardize all connection point formats"""
    standardized = smiles
    for i in range(1, 17):
        standardized = standardized.replace(f'[{i}*]', '[*]')
    return standardized


def remove_dummy_atoms(mol, keep_idx=None):
    """Remove all connection point atoms except specified index and saturate with hydrogen"""
    if mol is None:
        return None

    try:
        # Create editable molecule
        rw_mol = Chem.RWMol(mol)

        # Find all connection point atoms
        dummy_atoms = [atom for atom in rw_mol.GetAtoms() if atom.GetSymbol() == '*']

        # Sort in descending order by index for safe deletion
        dummy_indices = sorted([atom.GetIdx() for atom in dummy_atoms], reverse=True)

        for idx in dummy_indices:
            if idx == keep_idx:
                continue  # Skip connection point to keep

            # Get connection point neighbors
            atom = rw_mol.GetAtomWithIdx(idx)
            neighbors = atom.GetNeighbors()

            if neighbors:
                # Add hydrogen to each neighbor atom
                for neighbor in neighbors:
                    neighbor_atom = rw_mol.GetAtomWithIdx(neighbor.GetIdx())
                    current_hs = neighbor_atom.GetNumExplicitHs()
                    neighbor_atom.SetNumExplicitHs(current_hs + 1)

            # Remove connection point atom
            rw_mol.RemoveAtom(idx)

        return rw_mol.GetMol()
    except Exception as e:
        print(f"remove_dummy_atoms error: {str(e)}")
        return None


def connect_two_fragments(mol1, mol2, idx1, idx2):
    """Connect two molecules at specified connection point positions"""
    if mol1 is None or mol2 is None:
        return None

    try:
        # Create editable molecule object
        combined = Chem.RWMol(Chem.CombineMols(mol1, mol2))

        # Get atom offset of molecule 2
        offset = mol1.GetNumAtoms()

        # Get atoms adjacent to connection point
        atom1 = mol1.GetAtomWithIdx(idx1)
        neighbors1 = list(atom1.GetNeighbors())
        if not neighbors1:
            return None
        neighbor1_idx = neighbors1[0].GetIdx()

        atom2 = mol2.GetAtomWithIdx(idx2)
        neighbors2 = list(atom2.GetNeighbors())
        if not neighbors2:
            return None
        neighbor2_idx = neighbors2[0].GetIdx() + offset

        # Add new bond
        combined.AddBond(neighbor1_idx, neighbor2_idx, Chem.BondType.SINGLE)

        # Remove connection point atoms
        combined.RemoveAtom(idx2 + offset)  # Remove molecule 2's connection point first
        combined.RemoveAtom(idx1)  # Then remove molecule 1's connection point

        # Return new molecule
        return combined.GetMol()
    except Exception as e:
        return None


def check_valid_molecule(mol):
    """Check if molecule is valid: single structure, at least one connection point"""
    if mol is None:
        return False

    try:
        # Check if molecule is complete (no separators)
        smiles = Chem.MolToSmiles(mol)
        if '.' in smiles:
            return False

        # Check if there is at least one connection point
        if not any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
            return False

        return True
    except:
        return False


def process_fragments(fragments):
    """Process fragments for pairwise combination (self-pairing prohibited)"""
    results = []
    unique_smiles = set()
    n = len(fragments)

    # Calculate total pair count
    total_pairs = 0
    for i in range(n):
        if fragments[i] is None:
            continue
        dummies1 = [atom.GetIdx() for atom in fragments[i].GetAtoms() if atom.GetSymbol() == '*']
        if not dummies1:
            continue

        for j in range(i + 1, n):  # Self-pairing prohibited
            if fragments[j] is None:
                continue
            dummies2 = [atom.GetIdx() for atom in fragments[j].GetAtoms() if atom.GetSymbol() == '*']
            if not dummies2:
                continue
            total_pairs += len(dummies1) * len(dummies2)

    # If no valid pairs, return directly
    if total_pairs == 0:
        return results

    # Use tqdm to create progress bar
    progress_bar = tqdm(total=total_pairs, desc="Processing fragment pairs", unit="pair")

    for i in range(n):
        mol1 = fragments[i]
        if mol1 is None:
            continue

        # Get connection points of first molecule
        dummies1 = [atom.GetIdx() for atom in mol1.GetAtoms() if atom.GetSymbol() == '*']
        if not dummies1:
            continue

        for j in range(i + 1, n):  # Self-pairing prohibited: j starts from i+1
            mol2 = fragments[j]
            if mol2 is None:
                continue

            # Get connection points of second molecule
            dummies2 = [atom.GetIdx() for atom in mol2.GetAtoms() if atom.GetSymbol() == '*']
            if not dummies2:
                continue

            # Try all connection point combinations
            for idx1 in dummies1:
                for idx2 in dummies2:
                    # Update progress bar
                    progress_bar.update(1)

                    # Connect two fragments
                    combined = connect_two_fragments(mol1, mol2, idx1, idx2)
                    if combined is None:
                        continue

                    # Check if combination is valid
                    if check_valid_molecule(combined):
                        # Get SMILES representation
                        try:
                            smiles = Chem.MolToSmiles(combined)
                            if smiles not in unique_smiles:
                                unique_smiles.add(smiles)
                                results.append(smiles)
                        except:
                            continue

    # Close progress bar
    progress_bar.close()
    return results


def main():
    # File paths
    input_file = r'E:\Python\pythonProject\new_t_predict\data\fragment\water_soluble_polymers_fragment_1.csv'
    output_file = r'E:\Python\pythonProject\new_t_predict\data\fragment\s_intermediates.csv'

    # Read fragment file
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully read fragment file: {input_file} ({len(df)} rows)")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return

    # Check file content
    if df.empty:
        print("Fragment file is empty")
        return

    # Preprocess fragments
    print("Preprocessing fragments...")
    fragments = []
    valid_count = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing fragments"):
        try:
            smiles = row['fragment']
            # Standardize connection points
            std_smiles = standardize_dummy(smiles)
            mol = Chem.MolFromSmiles(std_smiles)

            if mol is not None and any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
                fragments.append(mol)
                valid_count += 1
            else:
                fragments.append(None)
        except:
            fragments.append(None)

    print(f"Number of valid fragments: {valid_count}/{len(df)}")

    # Process fragment pairing (self-pairing prohibited)
    print("Start fragment pairing combination...")
    results = process_fragments(fragments)

    # Save results
    if results:
        result_df = pd.DataFrame(results, columns=['SMILES'])
        result_df.to_csv(output_file, index=False)
        print(f"Generated {len(results)} intermediates and saved to {output_file}")
    else:
        print("No valid intermediates generated")


if __name__ == "__main__":
    main()
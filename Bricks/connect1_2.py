import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import itertools
import sys

# Disable all RDKit log output (including errors and warnings)
RDLogger.DisableLog('rdApp.*')


def standardize_dummy(smiles):
    """Standardize all connection point formats"""
    standardized = smiles
    for i in range(1, 17):
        standardized = standardized.replace(f'[{i}*]', '[*]')
    return standardized


def get_dummy_indices(mol):
    """Get indices of all connection point atoms in molecule"""
    if mol is None:
        return []
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']


def connect_fragments(mol1, mol2, idx1, idx2):
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
        # Remove atom with larger index first to avoid index change issues
        if idx2 + offset > idx1:
            combined.RemoveAtom(idx2 + offset)
            combined.RemoveAtom(idx1)
        else:
            combined.RemoveAtom(idx1)
            combined.RemoveAtom(idx2 + offset)

        # Return new molecule
        return combined.GetMol()
    except Exception as e:
        return None


def saturate_dummies(mol, keep_idx=None):
    """Saturate all connection points except the specified index"""
    if mol is None:
        return None

    rw_mol = Chem.RWMol(mol)
    dummies = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == '*']

    # Sort in descending order by index for safe deletion
    dummies.sort(reverse=True)

    for idx in dummies:
        if idx == keep_idx:
            continue
        # Get neighbor atoms of connection point
        atom = rw_mol.GetAtomWithIdx(idx)
        neighbors = list(atom.GetNeighbors())
        if neighbors:
            neighbor_idx = neighbors[0].GetIdx()
            # Add hydrogen atom to neighbor atom
            neighbor = rw_mol.GetAtomWithIdx(neighbor_idx)
            neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)
        # Remove connection point atoms
        rw_mol.RemoveAtom(idx)

    return rw_mol.GetMol()


def is_valid_molecule(mol):
    """Check if molecule is valid (single structure, no connection points)"""
    if mol is None:
        return False

    try:
        # Check if there are still connection points
        if any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
            return False

        # Check if molecule is complete (no separators)
        smiles = Chem.MolToSmiles(mol)
        if '.' in smiles:
            return False

        return True
    except:
        return False


def main():
    # File paths
    intermediates_file = r'E:\Python\pythonProject\new_t_predict\data\fragment\s_intermediates.csv'
    fragments_file = r'E:\Python\pythonProject\new_t_predict\data\fragment\t_fragment_1.csv'
    output_file = r'E:\Python\pythonProject\new_t_predict\data\result1_2.csv'

    # Define carbon-carbon double bond molecular fragment
    vinyl_frag = Chem.MolFromSmiles('[*]C=C')

    # Read intermediate file
    try:
        df_intermediates = pd.read_csv(intermediates_file)
        print(f"Successfully read intermediate file: {intermediates_file} ({len(df_intermediates)} rows)")
    except Exception as e:
        print(f"Error reading intermediate file: {str(e)}")
        return

    # Read fragment file
    try:
        df_fragments = pd.read_csv(fragments_file)
        print(f"Successfully read fragment file: {fragments_file} ({len(df_fragments)} rows)")
    except Exception as e:
        print(f"Error reading fragment file: {str(e)}")
        return

    # Preprocess intermediates
    intermediates = []
    for i, row in df_intermediates.iterrows():
        try:
            smiles = row['SMILES']
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
                intermediates.append(mol)
        except:
            continue

    # Preprocess fragments
    fragments = []
    for i, row in df_fragments.iterrows():
        try:
            smiles = row['fragment']
            std_smiles = standardize_dummy(smiles)
            mol = Chem.MolFromSmiles(std_smiles)
            if mol is not None and any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
                fragments.append(mol)
        except:
            continue

    print(f"Number of valid intermediates: {len(intermediates)}")
    print(f"Number of valid fragments: {len(fragments)}")

    # Calculate total pair count (for progress bar)
    total_pairs = 0
    for mol1 in intermediates:
        dummies1 = get_dummy_indices(mol1)
        if not dummies1:
            continue
        for mol2 in fragments:
            dummies2 = get_dummy_indices(mol2)
            if not dummies2:
                continue
            total_pairs += len(dummies1) * len(dummies2)

    print(f"Start synthesizing polymers... (Estimated {total_pairs} pairs)")

    # Open output file
    with open(output_file, 'w') as f_out:
        f_out.write("SMILES\n")  # Write header

        # Create progress bar
        progress_bar = tqdm(total=total_pairs, desc="Synthesizing polymers", unit="pair")

        # Iterate over all intermediates and fragments
        for mol1 in intermediates:
            dummies1 = get_dummy_indices(mol1)
            if not dummies1:
                continue

            for mol2 in fragments:
                dummies2 = get_dummy_indices(mol2)
                if not dummies2:
                    continue

                # Try all connection point combinations
                for idx1 in dummies1:
                    for idx2 in dummies2:
                        # Update progress
                        progress_bar.update(1)

                        # 1. Connect intermediate and fragment
                        intermediate = connect_fragments(mol1, mol2, idx1, idx2)
                        if intermediate is None:
                            continue

                        # 2. Get remaining connection points of intermediate
                        dummies_inter = get_dummy_indices(intermediate)
                        if not dummies_inter:
                            continue

                        # 3. Try connecting carbon-carbon double bond at each remaining connection point
                        for dummy_idx in dummies_inter:
                            # 3.1 Saturate other connection points
                            saturated = saturate_dummies(intermediate, keep_idx=dummy_idx)
                            if saturated is None:
                                continue

                            # 3.2 Connect carbon-carbon double bond fragment
                            final_mol = connect_fragments(saturated, vinyl_frag, dummy_idx, 0)
                            if final_mol is None:
                                continue

                            # 3.3 Saturate all remaining connection points
                            final_mol = saturate_dummies(final_mol)
                            if final_mol is None:
                                continue

                            # 3.4 Check final molecule validity
                            if is_valid_molecule(final_mol):
                                try:
                                    # Get SMILES and write to file
                                    smiles = Chem.MolToSmiles(final_mol)
                                    f_out.write(smiles + "\n")
                                except:
                                    continue

        # Close progress bar
        progress_bar.close()

    print(f"Synthesis completed! Results saved to {output_file}")


if __name__ == "__main__":
    main()
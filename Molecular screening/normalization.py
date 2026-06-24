from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sys


def contains_si(mol):
    """Check if molecule contains Si atoms"""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'Si':
            return True
    return False


def count_benzene_rings(mol):
    """Count the number of benzene rings in the molecule"""
    # Get all rings
    rings = mol.GetRingInfo().AtomRings()
    benzene_rings = 0

    for ring in rings:
        # Check if it is a 6-membered ring
        if len(ring) == 6:
            # Check if all atoms are carbon and aromatic
            all_carbon_aromatic = True
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() != 'C' or not atom.GetIsAromatic():
                    all_carbon_aromatic = False
                    break

            if all_carbon_aromatic:
                benzene_rings += 1

    return benzene_rings


def filter_molecules(input_file, output_file):
    """Filter molecules: remove molecules containing Si atoms and more than one benzene ring"""
    # Read input file
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        # Read and write header line
        header = next(f_in)
        f_out.write(header)

        # Process each line
        processed = 0
        kept = 0
        for line in f_in:
            processed += 1
            if processed % 10000 == 0:
                print(f"Processed {processed} rows, kept {kept} molecules")

            parts = line.strip().split(',')
            if len(parts) < 3:
                continue

            smiles = parts[0].strip()

            # Create molecule object from SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Check if contains Si atoms
            if contains_si(mol):
                continue

            # Check benzene ring count
            benzene_count = count_benzene_rings(mol)
            if benzene_count > 1:
                continue

            # Passed all checks, write to output file
            f_out.write(line)
            kept += 1

    print(f"Processing completed! Processed {processed} molecules total, kept {kept} molecules")


if __name__ == '__main__':
    # Please fill in your input and output file paths here
    input_file = "E:\\Python\\pythonProject\\new_t_predict\\data\\600_result.csv"  # Input file path
    output_file = "E:\\Python\\pythonProject\\new_t_predict\\data\\result_nosicc.csv"  # Output file path

    filter_molecules(input_file, output_file)

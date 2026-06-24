import re
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
import numpy as np
import itertools
import os


def replace_star_symbol(smiles):
    """Replace [n*] with [*]"""
    return re.sub(r'\[\d+\*\]', '[*]', smiles)


def read_fragments(file_path):
    """Read fragment file and replace connection point symbols"""
    fragments = []
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header line
        for row in reader:
            if row:
                smiles = replace_star_symbol(row[0].strip())
                fragments.append(smiles)
    return fragments


def connect_fragments(frag1, frag2):
    """Connect two fragments (each using one connection point)"""
    # Create editable molecule object
    combined = Chem.RWMol()
    atom_map = {}

    # Add frag1 atoms (skip connection point atoms)
    for atom in frag1.GetAtoms():
        if atom.GetSymbol() != '*':
            new_idx = combined.AddAtom(atom)
            atom_map[(0, atom.GetIdx())] = new_idx

    # Add frag2 atoms (skip connection point atoms)
    for atom in frag2.GetAtoms():
        if atom.GetSymbol() != '*':
            new_idx = combined.AddAtom(atom)
            atom_map[(1, atom.GetIdx())] = new_idx

    # Add frag1 bonds (skip connection point related bonds)
    for bond in frag1.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetSymbol() != '*' and a2.GetSymbol() != '*':
            combined.AddBond(
                atom_map[(0, a1.GetIdx())],
                atom_map[(0, a2.GetIdx())],
                bond.GetBondType()
            )

    # Add frag2 bonds (skip connection point related bonds)
    for bond in frag2.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetSymbol() != '*' and a2.GetSymbol() != '*':
            combined.AddBond(
                atom_map[(1, a1.GetIdx())],
                atom_map[(1, a2.GetIdx())],
                bond.GetBondType()
            )

    # Get atoms adjacent to connection point
    def get_star_neighbor(frag):
        for atom in frag.GetAtoms():
            if atom.GetSymbol() == '*':
                neighbor = atom.GetNeighbors()[0]
                bond_type = frag.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
                return neighbor.GetIdx(), bond_type
        return None, None

    # Connect two fragments
    idx1, bond_type1 = get_star_neighbor(frag1)
    idx2, bond_type2 = get_star_neighbor(frag2)

    if idx1 is not None and idx2 is not None:
        # Connect using single bond
        combined.AddBond(
            atom_map[(0, idx1)],
            atom_map[(1, idx2)],
            Chem.BondType.SINGLE
        )

    return combined.GetMol()


def add_carbon_double_bond(mol, cc_fragment):
    """Add carbon-carbon double bond fragment to connect two connection points"""
    rw_mol = Chem.RWMol(mol)

    # Get connection points in molecule
    star_atoms = [atom for atom in rw_mol.GetAtoms() if atom.GetSymbol() == '*']
    if len(star_atoms) < 2:
        return mol

    # Select first two connection points
    star1, star2 = star_atoms[:2]

    # Get connection point neighbors and bond types
    def get_star_info(atom):
        neighbor = atom.GetNeighbors()[0]
        bond_type = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
        return neighbor.GetIdx(), bond_type

    idx1, bond_type1 = get_star_info(star1)
    idx2, bond_type2 = get_star_info(star2)

    # Remove connection point atoms
    rw_mol.RemoveAtom(star2.GetIdx())
    rw_mol.RemoveAtom(star1.GetIdx())

    # Add carbon-carbon double bond fragment (skip connection points)
    atom_map = {}
    for atom in cc_fragment.GetAtoms():
        if atom.GetSymbol() != '*':
            new_idx = rw_mol.AddAtom(atom)
            atom_map[atom.GetIdx()] = new_idx

    # Add bonds inside carbon-carbon double bond fragment
    for bond in cc_fragment.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetSymbol() != '*' and a2.GetSymbol() != '*':
            rw_mol.AddBond(
                atom_map[a1.GetIdx()],
                atom_map[a2.GetIdx()],
                bond.GetBondType()
            )

    # Get connection point neighbors in carbon-carbon double bond fragment
    def get_cc_star_neighbor(cc_frag):
        neighbors = []
        for atom in cc_frag.GetAtoms():
            if atom.GetSymbol() == '*':
                neighbor = atom.GetNeighbors()[0]
                neighbors.append(neighbor.GetIdx())
        return neighbors

    cc_neighbors = get_cc_star_neighbor(cc_fragment)

    # Connect molecule with carbon-carbon double bond fragment
    rw_mol.AddBond(idx1, atom_map[cc_neighbors[0]], Chem.BondType.SINGLE)
    rw_mol.AddBond(idx2, atom_map[cc_neighbors[1]], Chem.BondType.SINGLE)

    return rw_mol.GetMol()


def main():
    # Input file path
    file1 = r'E:\Python\pythonProject\new_t_predict\data\fragment\t_fragment_1.csv'
    file2 = r'E:\Python\pythonProject\new_t_predict\data\fragment\水溶性聚合物_fragment_1.csv'
    output_file = r'E:\Python\pythonProject\new_t_predict\data\result1_1.csv'

    # Read fragments
    frag1_list = read_fragments(file1)
    frag2_list = read_fragments(file2)

    # Create carbon-carbon double bond fragment
    cc_frag = Chem.MolFromSmiles('[*]C=C[*]', sanitize=False)

    # Open output file
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['SMILES'])

        # Iterate over all fragment combinations
        for i, smi1 in enumerate(frag1_list):
            mol1 = Chem.MolFromSmiles(smi1, sanitize=False)
            if mol1 is None: continue

            for j, smi2 in enumerate(frag2_list):
                mol2 = Chem.MolFromSmiles(smi2, sanitize=False)
                if mol2 is None: continue

                try:
                    # Step 1: Connect two fragments
                    polymer = connect_fragments(mol1, mol2)
                    if polymer is None: continue

                    # Step 2: Add carbon-carbon double bonds until no connection points remain
                    while True:
                        # Count current connection points
                        star_count = sum(1 for atom in polymer.GetAtoms() if atom.GetSymbol() == '*')

                        if star_count == 0:
                            # No connection points, synthesis complete
                            break
                        elif star_count == 1:
                            # Single connection point cannot be processed, skip
                            polymer = None
                            break
                        else:
                            # Add carbon-carbon double bond fragment
                            polymer = add_carbon_double_bond(polymer, cc_frag)
                            if polymer is None:
                                break

                    if polymer is not None:
                        # Clean molecule and convert to SMILES
                        try:
                            Chem.SanitizeMol(polymer)
                            smiles = Chem.MolToSmiles(polymer)
                            writer.writerow([smiles])
                        except:
                            continue

                except Exception as e:
                    print(f"Error processing ({i},{j}): {str(e)}")
                    continue


if __name__ == "__main__":
    main()
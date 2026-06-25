import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import itertools
import gc
from multiprocessing import Pool, cpu_count
import time


def standardize_dummy(smiles):
    """Standardize all connection point formats"""
    standardized = smiles
    for i in range(1, 17):
        standardized = standardized.replace(f'[{i}*]', '[*]')
    return standardized


def get_dummy_indices(mol):
    """Get indices of all connection point atoms in molecule"""
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
        combined.RemoveAtom(idx2 + offset)  # Remove molecule 2's connection point first
        combined.RemoveAtom(idx1)  # Then remove molecule 1's connection point

        # Return new molecule
        return combined.GetMol()
    except Exception as e:
        return None


def saturate_dummies(mol, keep_idx=None):
    """Saturate all connection points except the specified index"""
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
            current_hs = neighbor.GetNumExplicitHs()
            neighbor.SetNumExplicitHs(current_hs + 1)
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


def process_fragment_batch(args):
    """Process single intermediate's connection task with all fragments"""
    mol1, fragments, vinyl_frag, output_path, batch_id = args
    batch_results = []

    dummies1 = get_dummy_indices(mol1)
    if not dummies1:
        return batch_results

    for mol2 in fragments:
        dummies2 = get_dummy_indices(mol2)
        if not dummies2:
            continue

        for idx1 in dummies1:
            for idx2 in dummies2:
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
                            smiles = Chem.MolToSmiles(final_mol)
                            batch_results.append(smiles)
                        except:
                            continue

    # Save this batch results to temporary file
    if batch_results:
        temp_file = f"{output_path}_temp_{batch_id}.csv"
        with open(temp_file, 'a') as f_temp:
            for smiles in batch_results:
                f_temp.write(smiles + "\n")

    return len(batch_results)


def main():
    # File paths
    intermediates_file = r'E:\Python\pythonProject\new_t_predict\data\fragment\t_intermediates.csv'
    fragments_file = r'E:\Python\pythonProject\new_t_predict\data\fragment\water_soluble_polymers_fragment_1.csv'
    output_file = r'E:\Python\pythonProject\new_t_predict\data\result2_1.csv'

    # Create output directory (if not exists)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Clear or create output file
    open(output_file, 'w').close()

    # Define carbon-carbon double bond molecular fragment
    vinyl_frag = Chem.MolFromSmiles('[*]C=C')

    print("Preprocess fragment data...")
    # Preprocess fragments (load at once, usually fragment count is small)
    fragments = []
    try:
        df_fragments = pd.read_csv(fragments_file)
        for i, row in df_fragments.iterrows():
            try:
                smiles = row['fragment']
                std_smiles = standardize_dummy(smiles)
                mol = Chem.MolFromSmiles(std_smiles)
                if mol is not None and any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
                    fragments.append(mol)
            except:
                continue
        print(f"Loaded valid fragments: {len(fragments)}")
    except Exception as e:
        print(f"Error reading fragment file: {str(e)}")
        return

    # Process intermediates in chunks
    chunk_size = 10000  # Number of intermediates per batch
    processed_count = 0
    total_intermediates = 0

    # Get total intermediate count (for progress bar)
    try:
        with open(intermediates_file, 'r') as f:
            total_intermediates = sum(1 for _ in f) - 1  # Subtract header line
        print(f"Total intermediate count: {total_intermediates}")
    except:
        print("Unable to determine total intermediate count, progress bar may be inaccurate")

    # Create progress bar
    progress_bar = tqdm(total=total_intermediates, desc="Processing intermediates", unit="mol")

    # Prepare multiprocessing pool
    num_cores = max(1, cpu_count() - 1)  # Reserve one core for system
    pool = Pool(processes=num_cores)
    print(f"Using {num_cores} CPU cores for parallel computation")

    # Read and process intermediates in batches
    batch_id = 0
    for chunk in pd.read_csv(intermediates_file, chunksize=chunk_size):
        batch_tasks = []
        batch_intermediates = []

        # Preprocess current batch of intermediates
        for _, row in chunk.iterrows():
            try:
                smiles = row['SMILES']
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None and any(atom.GetSymbol() == '*' for atom in mol.GetAtoms()):
                    batch_intermediates.append(mol)
            except:
                continue

        # Create processing task for each intermediate
        for mol1 in batch_intermediates:
            batch_tasks.append((mol1, fragments, vinyl_frag, output_file, batch_id))
            batch_id += 1

        # Process current batch tasks in parallel
        results = pool.imap_unordered(process_fragment_batch, batch_tasks)

        # Update progress
        for result in results:
            processed_count += 1
            progress_bar.update(1)

        # Manually trigger garbage collection
        del batch_tasks, batch_intermediates
        gc.collect()

    # Close multiprocessing pool
    pool.close()
    pool.join()
    progress_bar.close()

    print("Merge temporary files...")
    # Merge all temporary files to final output
    with open(output_file, 'w') as f_out:
        f_out.write("SMILES\n")  # Write header
        for temp_file in glob.glob(f"{output_file}_temp_*.csv"):
            with open(temp_file, 'r') as f_temp:
                shutil.copyfileobj(f_temp, f_out)
            os.remove(temp_file)

    print(f"Processing completed! Results saved to {output_file}")
    print(f"Processed {processed_count} intermediates total")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
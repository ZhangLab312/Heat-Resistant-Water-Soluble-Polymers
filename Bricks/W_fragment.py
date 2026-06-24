import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import concurrent.futures
import time
import os

# File path
input_file = r"E:\Python\pythonProject\new_t_predict\data\水溶性聚合物.csv"
output_file = r"E:\Python\pythonProject\new_t_predict\data\fragment\水溶性聚合物_fragment.csv"

# Create output directory (if not exists)
os.makedirs(os.path.dirname(output_file), exist_ok=True)


def fragment_molecule(smiles):
    """
    Fragment molecules using BRICS method
    Return fragment list (deduplicated SMILES strings)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Perform BRICS fragmentation
        fragments = list(BRICS.BRICSDecompose(mol))

        # Remove duplicate fragments and return
        return list(set(fragments))
    except:
        return []


def process_molecule(row):
    """
    Process a single molecule with timeout control
    """
    name, smiles = row['Name'], row['SMILES']

    # Use thread pool executor for timeout control
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fragment_molecule, smiles)
        try:
            return future.result(timeout=2)  # 2 second timeout
        except concurrent.futures.TimeoutError:
            print(f"\nSkip timed-out molecule: {name} | {smiles}")
            return []
        except Exception as e:
            print(f"\nProcessing error: {name} | {e}")
            return []


def main():
    # Read input file
    print("Reading data...")
    df = pd.read_csv(input_file)
    print(f"Found {len(df)} molecules")

    # Prepare result list
    all_fragments = []

    print("\nStart fragmentation processing (timeout: 2s/molecule):")
    total_molecules = len(df)

    # Use tqdm to create progress bar
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total_molecules, desc="Processing progress")
    except ImportError:
        print("tqdm not installed, using simple progress display")
        progress_bar = None

    # Process each molecule
    for _, row in df.iterrows():
        fragments = process_molecule(row)

        # Add fragments to result list
        for frag in fragments:
            all_fragments.append({'fragment': frag})

        # Update progress bar
        if progress_bar:
            progress_bar.update(1)
        else:
            print(f"Processed {len(all_fragments)} fragments | Current molecule: {row['Name']}")

    # Close progress bar
    if progress_bar:
        progress_bar.close()

    # Create output DataFrame
    result_df = pd.DataFrame(all_fragments, columns=['fragment'])

    # Save results
    result_df.to_csv(output_file, index=False)
    print(f"\nCompleted! Generated {len(result_df)} fragments total")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")
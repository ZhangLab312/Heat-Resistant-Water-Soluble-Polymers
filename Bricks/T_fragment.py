import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
import time
import os
import multiprocessing
from multiprocessing import Pool, TimeoutError
import sys

# File path
input_file = r"..\data\cleaned_predictions.csv"
output_file = r"..\data\fragment\t_fragment.csv"

# Create output directory (if not exists)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Global counter for progress display
processed_count = 0
total_count = 0
start_time = time.time()


def fragment_molecule_wrapper(args):
    """Wrapper function for multiprocessing"""
    idx, row = args
    return (idx, fragment_molecule(row))


def fragment_molecule(row):
    """Process a single molecule, return fragment list or error message"""
    name, smiles = row['polymer_name'], row['smiles']
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (name, smiles, [], "Invalid SMILES")

        # Perform BRICS fragmentation
        fragments = list(BRICS.BRICSDecompose(mol))
        return (name, smiles, list(set(fragments)), None)
    except Exception as e:
        return (name, smiles, [], f"Processing error: {str(e)}")


def update_progress():
    """Update and display progress info"""
    global processed_count, total_count, start_time
    elapsed = time.time() - start_time
    percent = (processed_count / total_count) * 100 if total_count > 0 else 0
    sys.stdout.write(f"\rProcessing progress: {processed_count}/{total_count} ({percent:.1f}%) | "
                     f"Elapsed: {elapsed:.1f}s | "
                     f"Est. remaining: {(elapsed / processed_count) * (total_count - processed_count):.1f}s "
                     if processed_count > 0 else "\rStarting processing...")
    sys.stdout.flush()


def main():
    global processed_count, total_count

    # Read input file
    print("Reading heat-resistant molecule data...")
    df = pd.read_csv(input_file)
    total_count = len(df)
    print(f"Found {total_count} heat-resistant molecules")

    # Prepare result lists
    all_fragments = []
    skipped_molecules = []
    success_count = 0

    print("\nStart fragmentation processing (timeout: 2s/molecule):")

    # Use multiprocessing pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []
        # Submit all tasks
        for idx, row in df.iterrows():
            results.append(pool.apply_async(fragment_molecule_wrapper, [(idx, row)]))

        # Process results
        for res in results:
            try:
                idx, result = res.get(timeout=2)
                name, smiles, fragments, error = result

                if error:
                    skipped_molecules.append((name, smiles, error))
                elif fragments:
                    for frag in fragments:
                        all_fragments.append({'fragment': frag})
                    success_count += 1

                processed_count += 1
                update_progress()

            except TimeoutError:
                name, smiles = df.iloc[idx]['polymer_name'], df.iloc[idx]['smiles']
                skipped_molecules.append((name, smiles, "Processing timeout"))
                processed_count += 1
                update_progress()
            except Exception as e:
                name, smiles = df.iloc[idx]['polymer_name'], df.iloc[idx]['smiles']
                skipped_molecules.append((name, smiles, f"Unknown error: {str(e)}"))
                processed_count += 1
                update_progress()

    # Create output DataFrame
    result_df = pd.DataFrame(all_fragments, columns=['fragment'])

    # Save results
    result_df.to_csv(output_file, index=False)

    # Save skipped molecule info
    skipped_file = output_file.replace("_fragment.csv", "_skipped.csv")
    skipped_df = pd.DataFrame(skipped_molecules, columns=['polymer_name', 'smiles', 'reason'])
    skipped_df.to_csv(skipped_file, index=False)

    # Print statistics
    print("\n\n" + "=" * 60)
    print("Fragmentation processing completed!")
    print(f"Total molecule count: {total_count}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped molecules: {len(skipped_molecules)}")
    print(f"Total fragments generated: {len(result_df)}")
    print(f"Fragment results saved to: {output_file}")
    print(f"Skipped molecule list saved to: {skipped_file}")

    # Show first 5 skipped molecules
    if skipped_molecules:
        print("\nExamples of skipped molecules:")
        for i, (name, smiles, reason) in enumerate(skipped_molecules[:5]):
            print(f"{i + 1}. {name[:50]}... | Reason: {reason}")
    print("=" * 60)


if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"Total time elapsed: {total_time / 60:.2f} minutes")

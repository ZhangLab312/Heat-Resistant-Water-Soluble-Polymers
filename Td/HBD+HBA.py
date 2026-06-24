import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import warnings

warnings.filterwarnings('ignore')


def compute_hbd_hba_wrapper(args):
    """
    Wrapper function for multiprocessing calls
    """
    idx, smiles = args
    if pd.isnull(smiles) or smiles == '':
        return idx, None, None, "SMILES is empty"

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if smiles.startswith('InChI='):
                mol = Chem.MolFromInchi(smiles)
                if mol is None:
                    return idx, None, None, "Unable to parse SMILES/InChI"
            else:
                return idx, None, None, "Invalid SMILES"

        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        return idx, hbd, hba, None

    except Exception:
        return idx, None, None, "Calculation error"


def process_chunk_parallel(chunk):
    """
    Process a data chunk in parallel
    """
    # Prepare multiprocessing parameters
    args_list = [(i, row['smiles']) for i, row in chunk.iterrows()]

    # Get CPU core count, but limit max processes
    num_cores = min(multiprocessing.cpu_count(), 8)  # Use at most 8 cores
    print(f"Using {num_cores} processes to process current chunk")

    results = []
    with Pool(processes=num_cores) as pool:
        # Use imap_unordered to improve efficiency
        for result in pool.imap_unordered(compute_hbd_hba_wrapper, args_list, chunksize=100):
            results.append(result)

    # Sort by original order
    results.sort(key=lambda x: x[0])

    # Extract results
    hbd_list = [r[1] for r in results]
    hba_list = [r[2] for r in results]
    error_list = [r[3] for r in results]

    return hbd_list, hba_list, error_list


def process_large_csv_with_progress(csv_path, output_path=None, chunksize=10000, use_parallel=False):
    """
    Process large-scale CSV file, optionally using multiprocessing

    Parameters:
    csv_path: Input CSV file path
    output_path: Output CSV file path (default adds suffix to original file path)
    chunksize: Chunk size per processing
    use_parallel: Whether to use multiprocessing
    """

    if not os.path.exists(csv_path):
        print(f"Error: File does not exist: {csv_path}")
        return None

    # Set output file path
    if output_path is None:
        # Add "_with_HBD_HBA" after original filename
        base_name, ext = os.path.splitext(csv_path)
        output_path = f"{base_name}_with_HBD_HBA{ext}"

    print(f"Input file: {csv_path}")
    print(f"Output file: {output_path}")
    print(f"Chunk size: {chunksize} rows")
    print(f"Use parallel processing: {use_parallel}")

    # Determine encoding
    try:
        with open(csv_path, 'rb') as f:
            raw_data = f.read(10000)
            try:
                raw_data.decode('utf-8')
                encoding = 'utf-8'
            except UnicodeDecodeError:
                encoding = 'gbk'
    except:
        encoding = 'utf-8'

    print(f"Detected file encoding: {encoding}")

    # Get total rows
    try:
        total_lines = sum(1 for _ in open(csv_path, 'r', encoding=encoding))
        print(f"Total file rows: {total_lines}")
    except:
        total_lines = None

    # Prepare output file
    first_chunk = True
    processed_rows = 0
    successful_count = 0
    failed_count = 0

    print("\nStart processing data...")

    # Create progress bar
    pbar = tqdm(total=total_lines if total_lines else None, desc="Processing progress", unit="rows")

    try:
        chunk_iterator = pd.read_csv(csv_path, encoding=encoding, chunksize=chunksize)

        for chunk_idx, chunk in enumerate(chunk_iterator):
            if 'smiles' not in chunk.columns:
                print(f"Error: Missing 'smiles' column")
                return None

            print(f"\nProcessing chunk {chunk_idx + 1}...")

            if use_parallel:
                # Use multiprocessing
                hbd_list, hba_list, error_list = process_chunk_parallel(chunk)
            else:
                # Use single-process processing
                hbd_list = []
                hba_list = []
                error_list = []

                for idx, row in chunk.iterrows():
                    _, hbd, hba, error = compute_hbd_hba_wrapper((idx, row['smiles']))

                    hbd_list.append(hbd)
                    hba_list.append(hba)
                    error_list.append(error)

                    if error:
                        failed_count += 1
                    else:
                        successful_count += 1

                    processed_rows += 1
                    pbar.update(1)

            # Count single-process results
            if not use_parallel:
                # Already counted above
                pass
            else:
                # Count multiprocessing results
                for error in error_list:
                    if error:
                        failed_count += 1
                    else:
                        successful_count += 1
                processed_rows += len(chunk)
                pbar.update(len(chunk))

            # Add calculation results to original data chunk
            chunk['HBD'] = hbd_list
            chunk['HBA'] = hba_list
            chunk['Herror'] = error_list

            # Write to file
            if first_chunk:
                chunk.to_csv(output_path, index=False, encoding='utf-8')
                first_chunk = False
            else:
                chunk.to_csv(output_path, mode='a', index=False, encoding='utf-8', header=False)

    except Exception as e:
        print(f"\nError occurred during processing: {str(e)}")
        return None
    finally:
        pbar.close()

    print(f"\nProcessing completed!")
    print(f"Total processed: {processed_rows} rows")
    print(f"Successfully calculated: {successful_count} rows")
    print(f"Failed: {failed_count} rows")
    print(f"Results saved to: {output_path}")

    return True


# Main program
if __name__ == "__main__":
    # Set file path
    input_csv = r"E:\Python\pythonProject\new_t_predict\data\原始数据.csv"

    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"Error: File does not exist: {input_csv}")
        print("Please check if the file path is correct.")
    else:
        # Run processing function
        # Note: On Windows, multiprocessing must run inside if __name__ == '__main__'
        success = process_large_csv_with_progress(
            csv_path=input_csv,
            output_path="../data/原始数据.csv",  # Can specify output path
            chunksize=10000,
            use_parallel=True  # Set to True for multiprocessing, False for single-process
        )

        if success:
            print("\nProgram completed!")
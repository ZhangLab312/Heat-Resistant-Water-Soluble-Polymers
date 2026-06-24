import pandas as pd
import numpy as np
import logging
import os
import sys
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def dominates(a, b):
    """
    Check if molecule a dominates molecule b

    Parameters:
    a: Tuple of molecule A (temp, stotal, sa_score, solute, source_file)
    b: Tuple of molecule B (temp, stotal, sa_score, solute, source_file)

    Returns:
    True if a dominates b, otherwise False
    """
    # Conditions for a to dominate b:
    # Temperature higher or equal, Stotal higher or equal, SA_Score lower or equal
    # And at least one metric is strictly better
    temp_cond = a[0] >= b[0]  # Higher temperature is better
    stotal_cond = a[1] >= b[1]  # Higher Stotal is better
    sa_cond = a[2] <= b[2]  # Lower SA_Score is better

    # All conditions are met, and at least one is strictly better
    return (temp_cond and stotal_cond and sa_cond and
            (a[0] > b[0] or a[1] > b[1] or a[2] < b[2]))


def process_chunk_for_pareto(chunk_data, chunk_id, total_chunks):
    """
    Process a data chunk and find the Pareto front of that chunk

    Parameters:
    chunk_data: Data chunk
    chunk_id: Data chunk ID
    total_chunks: Total chunks

    Returns:
    Pareto front of that chunk
    """
    chunk_pareto = []

    # Filter out invalid data
    valid_data = []
    invalid_count = 0

    for idx, row in chunk_data.iterrows():
        try:
            # Check if required columns exist
            if 'thermal_decomposition_temperature' not in row or \
                    'Stotal' not in row or \
                    'SA_Score' not in row or \
                    'solute' not in row:
                invalid_count += 1
                continue

            temp = float(row['thermal_decomposition_temperature'])
            stotal = float(row['Stotal'])
            sa_score = float(row['SA_Score'])
            solute = row['solute']
            # Get source tag from DataFrame
            source_file = row['source_file'] if 'source_file' in row else 'unknown'

            # Check for NaN or infinity values
            if np.isnan(temp) or np.isnan(stotal) or np.isnan(sa_score) or \
                    np.isinf(temp) or np.isinf(stotal) or np.isinf(sa_score):
                invalid_count += 1
                continue

            valid_data.append((temp, stotal, sa_score, solute, source_file))

        except (ValueError, TypeError, KeyError) as e:
            invalid_count += 1
            continue

    # If data chunk is empty, return empty list
    if not valid_data:
        if chunk_id < 5:  # Only show warnings in first few chunks to avoid excessive logging
            logger.warning(f"Chunk {chunk_id + 1}/{total_chunks} has no valid data. Invalid count: {invalid_count}")
        return []

    # Use fast non-dominated sorting algorithm to find Pareto front
    # First sort by temperature descending, Stotal descending, SA_Score ascending, so molecules that dominate others are more likely to be found
    valid_data.sort(key=lambda x: (-x[0], -x[1], x[2]))

    # Initialize Pareto front
    chunk_pareto = [valid_data[0]]

    # Check if each molecule should be added to Pareto front
    for candidate in valid_data[1:]:
        dominated = False

        # Check if dominated by any molecule in current Pareto front
        for pareto_mol in chunk_pareto[:]:  # Use slice to create copy to avoid modifying list during iteration
            if dominates(pareto_mol, candidate):
                dominated = True
                break

        # If not dominated by any molecule, add to Pareto front
        if not dominated:
            # Remove molecules dominated by candidate molecule
            chunk_pareto = [mol for mol in chunk_pareto if not dominates(candidate, mol)]
            chunk_pareto.append(candidate)

    if chunk_id < 5 or (chunk_id + 1) % 100 == 0:  # Only show logs in first few chunks and every 100th chunk
        logger.info(
            f"Chunk {chunk_id + 1}/{total_chunks} processed: {len(valid_data)} valid molecules, {len(chunk_pareto)} in Pareto front, {invalid_count} invalid rows")
    return chunk_pareto


def merge_pareto_fronts(fronts):
    """
    Merge multiple Pareto fronts

    Parameters:
    fronts: List of multiple Pareto fronts

    Returns:
    Merged global Pareto front
    """
    if not fronts:
        logger.warning("No fronts to merge")
        return []

    # Merge all fronts
    all_candidates = []
    empty_fronts = 0

    for i, front in enumerate(fronts):
        if front:
            all_candidates.extend(front)
        else:
            empty_fronts += 1

    logger.info(f"Merging {len(fronts)} fronts: {len(all_candidates)} candidates, {empty_fronts} empty fronts")

    # If all candidates are empty, return empty list
    if not all_candidates:
        logger.warning("No candidates to merge")
        return []

    # If only one front, return directly
    if len(fronts) == 1:
        return all_candidates

    # Find global Pareto front
    # Sort by temperature descending, Stotal descending, SA_Score ascending
    all_candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))

    global_pareto = [all_candidates[0]]

    for candidate in all_candidates[1:]:
        dominated = False

        # Check if dominated by any molecule in current global Pareto front
        for pareto_mol in global_pareto[:]:
            if dominates(pareto_mol, candidate):
                dominated = True
                break

        # If not dominated by any molecule, add to global Pareto front
        if not dominated:
            # Remove molecules dominated by candidate molecule
            global_pareto = [mol for mol in global_pareto if not dominates(candidate, mol)]
            global_pareto.append(candidate)

    logger.info(f"Merged result: {len(global_pareto)} Pareto optimal molecules")
    return global_pareto


def combine_data_and_find_pareto_front(reasonable_file, original_file, output_file, chunksize=50000):
    """
    Combine data from two files, find Pareto front, and only save molecules from reasonable molecules that break through the Pareto front

    Parameters:
    reasonable_file: Reasonable molecules file path
    original_file: Original data file path
    output_file: Output file path
    chunksize: Data chunk size per processing
    """
    # Check if input files exist
    if not os.path.exists(reasonable_file):
        logger.error(f"Error: Reasonable molecules file does not exist: {reasonable_file}")
        return None
    if not os.path.exists(original_file):
        logger.error(f"Error: Original data file does not exist: {original_file}")
        return None

    logger.info("Start reading and merging data...")

    # Read reasonable molecules file
    logger.info(f"Reading reasonable molecules file: {reasonable_file}")
    try:
        reasonable_df = pd.read_csv(reasonable_file)

        # Column names in reasonable molecules need renaming: SMILES column renamed to solute
        reasonable_df = reasonable_df.rename(columns={
            'SMILES': 'solute'  # SMILES column in reasonable molecules, renamed to solute
        })

        # Mark source
        reasonable_df['source_file'] = 'reasonable'
        logger.info(f"Reasonable molecules file read successfully, total {len(reasonable_df)} rows")
        logger.info(f"Reasonable molecules file column names: {list(reasonable_df.columns)}")

        # Check if required columns exist
        required_columns = ['thermal_decomposition_temperature', 'SA_Score', 'Stotal', 'solute']
        missing_columns = [col for col in required_columns if col not in reasonable_df.columns]
        if missing_columns:
            logger.error(f"Reasonable molecules file is missing required columns: {missing_columns}")
            logger.error(f"Columns in reasonable molecules file: {list(reasonable_df.columns)}")
            return None

    except Exception as e:
        logger.error(f"Error reading reasonable molecules file: {e}")
        return None

    # Read original data file
    logger.info(f"Reading original data file: {original_file}")
    try:
        original_df = pd.read_csv(original_file)

        # Rename columns to unify format
        original_df = original_df.rename(columns={
            'Median': 'thermal_decomposition_temperature',  # Predicted_Median changed to Median
            'SA_Score': 'SA_Score',
            'Stotal': 'Stotal',
            'smiles': 'solute'  # smiles column in original data, renamed to solute
        })

        # Mark source
        original_df['source_file'] = 'original'
        logger.info(f"Original data file read successfully, total {len(original_df)} rows")
        logger.info(f"Original data file column names: {list(original_df.columns)}")

        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in original_df.columns]
        if missing_columns:
            logger.error(f"Original data file is missing required columns: {missing_columns}")
            logger.error(f"Columns in original data file: {list(original_df.columns)}")
            return None

    except Exception as e:
        logger.error(f"Error reading original data file: {e}")
        return None

    # Merge data
    logger.info("Merging data from two files...")
    # Only select needed columns
    common_columns = ['solute', 'thermal_decomposition_temperature', 'SA_Score', 'Stotal', 'source_file']
    reasonable_df = reasonable_df[common_columns]
    original_df = original_df[common_columns]

    combined_df = pd.concat([reasonable_df, original_df], ignore_index=True)
    logger.info(f"Total data volume after merging: {len(combined_df)} rows")

    # Save merged data to temporary file for chunk processing
    temp_file = "temp_combined_data.csv"
    logger.info(f"Saving merged data to temporary file: {temp_file}")
    combined_df.to_csv(temp_file, index=False)

    # Use chunk processing to calculate Pareto front
    logger.info("Start calculating Pareto front...")

    # Get total file rows
    total_rows = len(combined_df)
    total_chunks = (total_rows + chunksize - 1) // chunksize  # Round up

    logger.info(f"Total rows: {total_rows:,}, Total chunks: {total_chunks}")

    # Process data
    start_time = time.time()
    chunk_fronts = []

    # Read file in chunks
    progress_bar = tqdm(total=total_chunks, desc="Processing chunks", unit="chunk",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for chunk_idx, chunk in enumerate(pd.read_csv(temp_file, chunksize=chunksize)):
        # Process current chunk
        chunk_pareto = process_chunk_for_pareto(chunk, chunk_idx, total_chunks)
        chunk_fronts.append(chunk_pareto)

        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            "Current chunk": f"{chunk_idx + 1}/{total_chunks}",
            "Completed": f"{(chunk_idx + 1) / total_chunks * 100:.1f}%",
            "Processed rows": f"{(chunk_idx + 1) * chunksize:,}"
        })

    # Close progress bar
    progress_bar.close()

    # Merge Pareto fronts from all chunks
    logger.info("Merging Pareto fronts from all chunks...")
    global_pareto = merge_pareto_fronts(chunk_fronts)

    # If no Pareto optimal molecules found, log warning
    if not global_pareto:
        logger.warning("No Pareto optimal molecules found!")
        result_df = pd.DataFrame(
            columns=['solute', 'thermal_decomposition_temperature', 'SA_Score', 'Stotal', 'source_file'])
    else:
        # Convert to DataFrame
        result_data = []
        for temp, stotal, sa_score, solute, source_file in global_pareto:
            result_data.append({
                'solute': solute,
                'thermal_decomposition_temperature': temp,
                'SA_Score': sa_score,
                'Stotal': stotal,
                'source_file': source_file
            })

        result_df = pd.DataFrame(result_data)

    # Filter reasonable molecules that break through the Pareto front
    logger.info("Filtering reasonable molecules that break through the Pareto front...")
    reasonable_pareto = result_df[result_df['source_file'] == 'reasonable'].copy()

    # Delete temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        logger.info(f"Temporary file deleted: {temp_file}")

    # Save results
    if len(reasonable_pareto) > 0:
        # Remove source_file column (optional, comment out if you need to keep source info)
        if 'source_file' in reasonable_pareto.columns:
            reasonable_pareto = reasonable_pareto.drop('source_file', axis=1)

        logger.info(f"Save {len(reasonable_pareto)} reasonable molecules that break through the Pareto front to {output_file}")
        reasonable_pareto.to_csv(output_file, index=False)
    else:
        logger.warning("No reasonable molecules found that break through the Pareto front!")
        # Create empty result file
        empty_df = pd.DataFrame(columns=['solute', 'thermal_decomposition_temperature', 'SA_Score', 'Stotal'])
        empty_df.to_csv(output_file, index=False)

    # Final statistics
    elapsed_time = time.time() - start_time
    processing_rate = total_rows / elapsed_time if elapsed_time > 0 else 0

    logger.info(f"Processing completed! Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Processing rate: {processing_rate:.2f} rows/second")
    logger.info(f"Total Pareto optimal molecules: {len(result_df)}")
    logger.info(f"Reasonable molecules that break through the Pareto front: {len(reasonable_pareto)}")

    # Output Pareto front statistics
    if len(reasonable_pareto) > 0:
        temp_stats = reasonable_pareto['thermal_decomposition_temperature'].describe()
        stotal_stats = reasonable_pareto['Stotal'].describe()
        sa_stats = reasonable_pareto['SA_Score'].describe()

        logger.info("Reasonable molecules Pareto front statistics:")
        logger.info(
            f"  Temperature: min={temp_stats['min']:.2f}, max={temp_stats['max']:.2f}, mean={temp_stats['mean']:.2f}")
        logger.info(
            f"  Stotal: min={stotal_stats['min']:.2f}, max={stotal_stats['max']:.2f}, mean={stotal_stats['mean']:.2f}")
        logger.info(
            f"  SA_Score: min={sa_stats['min']:.2f}, max={sa_stats['max']:.2f}, mean={sa_stats['mean']:.2f}")
    else:
        logger.warning("No reasonable molecules found that break through the Pareto front, unable to display statistics.")

    return reasonable_pareto


if __name__ == "__main__":
    # Input parameters
    reasonable_file = r"E:\Python\pythonProject\new_t_predict\data\合理分子.csv"
    original_file = r"E:\Python\pythonProject\new_t_predict\data\原始数据.csv"
    output_file = "E:\\Python\\pythonProject\\new_t_predict\\data\\突破帕累托前沿的合理分子.csv"

    # Check if files exist
    if not os.path.exists(reasonable_file):
        print(f"Error: Reasonable molecules file does not exist: {reasonable_file}")
        print("Please check if the file path is correct")
        sys.exit(1)

    if not os.path.exists(original_file):
        print(f"Error: Original data file does not exist: {original_file}")
        print("Please check if the file path is correct")
        sys.exit(1)

    # Run Pareto front filtering
    print(f"Start merging data and performing Pareto front analysis...")
    print(f"Reasonable molecules file: {reasonable_file}")
    print(f"Original data file: {original_file}")
    print(f"Output will be saved to: {output_file}")
    print("Filter criteria: molecules with high thermal decomposition temperature, high Stotal, low SA_Score")
    print("Note: Only save reasonable molecules that break through the Pareto front")
    print("Use chunk processing to avoid memory overflow")

    try:
        result_df = combine_data_and_find_pareto_front(
            reasonable_file=reasonable_file,
            original_file=original_file,
            output_file=output_file,
            chunksize=50000  # Process 50,000 rows per chunk
        )

        if result_df is not None:
            print(f"\nPareto front filtering completed!")
            if len(result_df) > 0:
                print(f"Found {len(result_df):,} reasonable molecules total that break through the Pareto front")
                print("These molecules are optimal in all three metrics: thermal decomposition temperature, Stotal, and SA_Score")
                print("No molecule is better than these molecules in all three metrics")
                print(f"Results saved to: {output_file}")
            else:
                print("No reasonable molecules found that break through the Pareto front!")
                print("This may be because:")
                print("1. No reasonable molecules break through the Pareto front")
                print("2. Data format is incorrect")
                print("3. Column names do not match")
        else:
            print("Processing failed! Please check error logs.")

    except Exception as e:
        print(f"Error occurred during processing: {e}")
        import traceback

        traceback.print_exc()
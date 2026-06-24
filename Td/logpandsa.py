import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
import logging
import os
import sys
import multiprocessing as mp
from tqdm import tqdm
import time
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_properties(smiles):
    """
    Calculate LogP and SA_Score of molecules

    Parameters:
    smiles: SMILES string

    Returns:
    (logp, sa_score): LogP value and SA_Score value, returns (None, None) if calculation fails
    """
    if not smiles or pd.isna(smiles) or str(smiles).strip() == '':
        return None, None

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, None

        # Calculate LogP
        logp = Crippen.MolLogP(mol)

        # Calculate SA_Score (using simple synthetic accessibility score)
        # Here we use molecular weight as a simple SA_Score proxy; in practice, more complex algorithms can be used
        # If more precise SA_Score is needed, you can install and use specialized libraries
        sa_score = Descriptors.MolWt(mol) / 100  # Simplified version

        return logp, sa_score
    except Exception as e:
        logger.debug(f"Error calculating properties for {smiles}: {str(e)}")
        return None, None


def process_chunk(args):
    """
    Process a data chunk, calculate LogP and SA_Score

    Parameters:
    args: (chunk_data, chunk_id, total_chunks)

    Returns:
    result: Processing result, contains calculated data and statistics
    """
    chunk_data, chunk_id, total_chunks = args
    processed_data = []
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'logp_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0
        },
        'sa_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0
        }
    }

    for idx, row in chunk_data.iterrows():
        stats['total'] += 1

        smiles = row['smiles']
        logp, sa_score = calculate_properties(smiles)

        # Create new row data
        new_row = {
            'polymer_name': row['polymer_name'],
            'Median': row['Median'],
            'smiles': smiles,
            'Predicted_Median': row['Predicted_Median'],
            'LogP': logp,
            'SA_Score': sa_score
        }

        if logp is not None and sa_score is not None:
            stats['success'] += 1
            # Update LogP statistics
            stats['logp_stats']['sum'] += logp
            stats['logp_stats']['min'] = min(stats['logp_stats']['min'], logp)
            stats['logp_stats']['max'] = max(stats['logp_stats']['max'], logp)

            # Update SA_Score statistics
            stats['sa_stats']['sum'] += sa_score
            stats['sa_stats']['min'] = min(stats['sa_stats']['min'], sa_score)
            stats['sa_stats']['max'] = max(stats['sa_stats']['max'], sa_score)
        else:
            stats['failed'] += 1

        processed_data.append(new_row)

    logger.info(
        f"Chunk {chunk_id + 1}/{total_chunks} processed: {stats['total']} rows, {stats['success']} successful, {stats['failed']} failed")

    return {
        'data': processed_data,
        'stats': stats
    }


def count_lines(file_path):
    """Efficiently count lines of a large file"""
    count = 0
    with open(file_path, 'rb') as f:
        for line in f:
            count += 1
    return count


def add_properties_parallel(input_file, output_file, num_processes=None, chunksize=10000):
    """
    Calculate LogP and SA_Score of molecules using multiprocessing and add to file

    Parameters:
    input_file: Input file path
    output_file: Output file path
    num_processes: Number of processes to use, defaults to CPU core count
    chunksize: Data chunk size per process
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return 0, 0, {}, {}

    # Get total file rows
    logger.info("Counting total rows...")
    total_rows = count_lines(input_file) - 1  # Subtract header line
    logger.info(f"Total rows to process: {total_rows:,}")

    # Set number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    logger.info(f"Using {num_processes} processes")
    logger.info("Calculating LogP and SA_Score for all molecules")

    # Read the entire file
    logger.info("Reading input file...")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    logger.info(f"Loaded {total_rows:,} rows into memory")

    # Split data into multiple chunks
    chunk_size = max(1, total_rows // num_processes)
    chunks = []
    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        chunks.append((chunk, len(chunks), len(chunks)))

    logger.info(f"Split data into {len(chunks)} chunks")

    # Process using multiprocessing
    start_time = time.time()
    all_processed_data = []
    all_stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'logp_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0
        },
        'sa_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0
        }
    }

    with mp.Pool(processes=num_processes) as pool:
        # Use imap_unordered to get results while showing progress
        results = list(tqdm(
            pool.imap_unordered(process_chunk, chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))

        # Merge results
        for result in results:
            all_processed_data.extend(result['data'])

            # Merge statistics
            stats = result['stats']
            all_stats['total'] += stats['total']
            all_stats['success'] += stats['success']
            all_stats['failed'] += stats['failed']

            # Merge LogP statistics
            all_stats['logp_stats']['sum'] += stats['logp_stats']['sum']
            all_stats['logp_stats']['min'] = min(all_stats['logp_stats']['min'], stats['logp_stats']['min'])
            all_stats['logp_stats']['max'] = max(all_stats['logp_stats']['max'], stats['logp_stats']['max'])

            # Merge SA_Score statistics
            all_stats['sa_stats']['sum'] += stats['sa_stats']['sum']
            all_stats['sa_stats']['min'] = min(all_stats['sa_stats']['min'], stats['sa_stats']['min'])
            all_stats['sa_stats']['max'] = max(all_stats['sa_stats']['max'], stats['sa_stats']['max'])

    # Calculate average LogP and SA_Score
    if all_stats['success'] > 0:
        avg_logp = all_stats['logp_stats']['sum'] / all_stats['success']
        avg_sa = all_stats['sa_stats']['sum'] / all_stats['success']
    else:
        avg_logp = 0
        avg_sa = 0

    # Save processed data
    logger.info("Saving processed data with LogP and SA_Score...")
    output_df = pd.DataFrame(all_processed_data)
    output_columns = ['polymer_name', 'Median', 'smiles', 'Predicted_Median', 'LogP', 'SA_Score']
    output_df[output_columns].to_csv(output_file, index=False)

    # Final statistics
    elapsed_time = time.time() - start_time
    processing_rate = all_stats['total'] / elapsed_time if elapsed_time > 0 else 0

    success_percentage = all_stats['success'] / all_stats['total'] * 100 if all_stats['total'] > 0 else 0
    failed_percentage = all_stats['failed'] / all_stats['total'] * 100 if all_stats['total'] > 0 else 0

    logger.info(f"Processing completed in {elapsed_time:.2f} seconds!")
    logger.info(f"Processing rate: {processing_rate:.2f} molecules/second")
    logger.info(f"Total molecules: {all_stats['total']:,}")
    logger.info(f"Successful calculations: {all_stats['success']:,} ({success_percentage:.2f}%)")
    logger.info(f"Failed calculations: {all_stats['failed']:,} ({failed_percentage:.2f}%)")

    if all_stats['success'] > 0:
        logger.info(f"LogP statistics:")
        logger.info(f"  Minimum LogP: {all_stats['logp_stats']['min']:.4f}")
        logger.info(f"  Maximum LogP: {all_stats['logp_stats']['max']:.4f}")
        logger.info(f"  Average LogP: {avg_logp:.4f}")

        logger.info(f"SA_Score statistics:")
        logger.info(f"  Minimum SA_Score: {all_stats['sa_stats']['min']:.4f}")
        logger.info(f"  Maximum SA_Score: {all_stats['sa_stats']['max']:.4f}")
        logger.info(f"  Average SA_Score: {avg_sa:.4f}")

    # Save statistics
    stats_file = output_file.replace('.csv', '_properties_stats.csv')

    # Prepare statistics DataFrame
    stats_data = []
    stats_data.append(['Total molecules', all_stats['total']])
    stats_data.append(['Successful calculations', all_stats['success']])
    stats_data.append(['Success percentage', f"{success_percentage:.2f}%"])
    stats_data.append(['Failed calculations', all_stats['failed']])
    stats_data.append(['Failed percentage', f"{failed_percentage:.2f}%"])
    stats_data.append(['Processing time (seconds)', f"{elapsed_time:.2f}"])
    stats_data.append(['Processing rate (molecules/second)', f"{processing_rate:.2f}"])

    if all_stats['success'] > 0:
        stats_data.append(['Minimum LogP', f"{all_stats['logp_stats']['min']:.4f}"])
        stats_data.append(['Maximum LogP', f"{all_stats['logp_stats']['max']:.4f}"])
        stats_data.append(['Average LogP', f"{avg_logp:.4f}"])
        stats_data.append(['Minimum SA_Score', f"{all_stats['sa_stats']['min']:.4f}"])
        stats_data.append(['Maximum SA_Score', f"{all_stats['sa_stats']['max']:.4f}"])
        stats_data.append(['Average SA_Score', f"{avg_sa:.4f}"])

    stats_df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])
    stats_df.to_csv(stats_file, index=False)

    return all_stats['success'], all_stats['failed'], all_stats['logp_stats'], all_stats['sa_stats']


if __name__ == "__main__":
    # Input parameters - please modify these paths according to your actual situation
    input_file = "E:\\Python\\pythonProject\\new_t_predict\\data\\合理分子2.csv"  # File containing polymer data
    output_file = "E:\\Python\\pythonProject\\new_t_predict\\data\\合理分子3.csv"  # Output file containing LogP and SA_Score

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        print("Please check if the file path is correct")
        sys.exit(1)

    # Run calculation
    print(f"Starting LogP and SA_Score calculation for file: {input_file}")
    print(f"Output will be saved to: {output_file}")

    # Use all available CPU cores
    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes for parallel calculation")

    success_count, failed_count, logp_stats, sa_stats = add_properties_parallel(
        input_file=input_file,
        output_file=output_file,
        num_processes=num_processes,
        chunksize=10000  # Chunk size per process
    )

    print(f"\nCalculation completed!")
    print(f"Number of molecules successfully calculated LogP and SA_Score: {success_count:,}")
    print(f"Number of molecules with failed calculation: {failed_count:,}")

    if success_count > 0:
        print(f"\nLogP statistics:")
        print(f"  Minimum LogP: {logp_stats['min']:.4f}")
        print(f"  Maximum LogP: {logp_stats['max']:.4f}")
        print(f"  Average LogP: {logp_stats['sum'] / success_count:.4f}")

        print(f"\nSA_Score statistics:")
        print(f"  Minimum SA_Score: {sa_stats['min']:.4f}")
        print(f"  Maximum SA_Score: {sa_stats['max']:.4f}")
        print(f"  Average SA_Score: {sa_stats['sum'] / success_count:.4f}")
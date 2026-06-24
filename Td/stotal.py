import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')
import multiprocessing as mp


def calculate_chunk_stats(args):
    """Calculate statistics for a single data chunk"""
    chunk_idx, chunk = args

    # Calculate features
    chunk['-LogP'] = -chunk['LogP']
    chunk['HBD+HBA'] = chunk['HBD'] + chunk['HBA']

    # Calculate statistics
    stats = {}
    for col, stat_name in [('-LogP', '-LogP'), ('HBD+HBA', 'HBD+HBA'), ('predicted_logS', 'logSmonomer')]:
        valid_data = chunk[col].dropna()
        if len(valid_data) > 0:
            mean = valid_data.mean()
            std = valid_data.std()
            count = len(valid_data)
            sum_val = valid_data.sum()
            sum_sq = (valid_data ** 2).sum()
        else:
            mean = std = sum_val = sum_sq = 0
            count = 0

        stats[stat_name] = {
            'sum': sum_val,
            'sum_sq': sum_sq,
            'count': count,
            'mean': mean,
            'std': std
        }

    return stats, len(chunk)


def calculate_chunk_stotal(args):
    """Calculate Stotal for a single data chunk"""
    chunk_idx, chunk, stats, weights = args

    # Save original column order
    original_columns = chunk.columns.tolist()

    # Calculate features
    chunk['-LogP'] = -chunk['LogP']
    chunk['HBD+HBA'] = chunk['HBD'] + chunk['HBA']

    # Standardize features
    chunk['z_-LogP'] = (chunk['-LogP'] - stats['-LogP']['mean']) / stats['-LogP']['std']
    chunk['z_HBD+HBA'] = (chunk['HBD+HBA'] - stats['HBD+HBA']['mean']) / stats['HBD+HBA']['std']
    chunk['z_logSmonomer'] = (chunk['predicted_logS'] - stats['logSmonomer']['mean']) / stats['logSmonomer']['std']

    # Calculate Stotal
    chunk['Stotal'] = (
            weights['w1'] * chunk['z_-LogP'] +
            weights['w2'] * chunk['z_HBD+HBA'] +
            weights['w3'] * chunk['z_logSmonomer'] +
            weights['intercept']
    )

    # Calculate predicted probability
    chunk['predicted_probability'] = 1 / (1 + np.exp(-chunk['Stotal']))

    # Remove intermediate columns
    chunk = chunk.drop(['-LogP', 'HBD+HBA', 'z_-LogP', 'z_HBD+HBA', 'z_logSmonomer'], axis=1, errors='ignore')

    # Restore original column order and add new columns at the end
    final_columns = original_columns + ['Stotal', 'predicted_probability']
    chunk = chunk.reindex(columns=final_columns)

    return chunk_idx, chunk


def calculate_stotal_parallel(csv_path, output_path, chunk_size=500000, n_workers=None):
    """
    Calculate Stotal for large datasets in parallel
    """

    # Given weight parameters
    weights = {
        'w1': 0.0124,  # -LogP weight
        'w2': 0.2086,  # HBD+HBA weight
        'w3': 2.4509,  # logSmonomer weight
        'intercept': -0.4345  # Intercept term
    }

    print(f"Start processing file: {csv_path}")
    print(f"Output file: {output_path}")
    print(f"Using parallel processing, number of workers: {n_workers or mp.cpu_count()}")

    # Step 1: Calculate statistics in parallel
    print("\nStep 1: Calculate mean and standard deviation of features in parallel...")

    # Determine number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Use at most 8 processes

    total_rows = 0
    all_stats = []

    # Check required columns in file
    print("Check input file columns...")
    sample_df = pd.read_csv(csv_path, nrows=5)
    required_columns = ['LogP', 'HBD', 'HBA', 'predicted_logS']
    missing_columns = [col for col in required_columns if col not in sample_df.columns]

    if missing_columns:
        print(f"Error: Input file is missing the following required columns: {missing_columns}")
        print(f"Columns in file: {list(sample_df.columns)}")
        return None

    print(f"Found required columns: {required_columns}")

    # Calculate statistics using multiprocessing
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        chunk_idx = 0

        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, encoding='utf-8'):
            futures.append(executor.submit(calculate_chunk_stats, (chunk_idx, chunk)))
            chunk_idx += 1
            total_rows += len(chunk)

        # Collect results
        for future in futures:
            stats, rows = future.result()
            all_stats.append(stats)

    # Merge statistics
    merged_stats = {}
    for feature in ['-LogP', 'HBD+HBA', 'logSmonomer']:
        total_sum = sum(s[feature]['sum'] for s in all_stats)
        total_sum_sq = sum(s[feature]['sum_sq'] for s in all_stats)
        total_count = sum(s[feature]['count'] for s in all_stats)

        if total_count > 0:
            mean = total_sum / total_count
            variance = (total_sum_sq / total_count) - (mean ** 2)
            std = np.sqrt(variance) if variance > 0 else 1.0
        else:
            mean = 0
            std = 1.0

        merged_stats[feature] = {'mean': mean, 'std': std}

    print(f"\nStatistics completed, processed {len(all_stats)} chunks of data")
    print("Standardization parameters:")
    for feature, stat in merged_stats.items():
        print(f"  {feature}: mean={stat['mean']:.6f}, std={stat['std']:.6f}")

    # Step 2: Calculate Stotal in parallel
    print("\nStep 2: Calculate Stotal in parallel...")

    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # List for sorting
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        chunk_idx = 0

        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, encoding='utf-8'):
            future = executor.submit(calculate_chunk_stotal,
                                     (chunk_idx, chunk, merged_stats, weights))
            futures[future] = chunk_idx
            chunk_idx += 1

        # Collect results in order
        processed_futures = []
        for future in futures:
            processed_futures.append((futures[future], future))

        # Sort by chunk index
        processed_futures.sort(key=lambda x: x[0])

        # Write to file
        first_chunk = True
        for idx, future in processed_futures:
            chunk_idx, chunk_result = future.result()

            if first_chunk:
                chunk_result.to_csv(output_path, index=False, encoding='utf-8')
                first_chunk = False
            else:
                chunk_result.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8')

            results.append(chunk_result)

    print(f"\nProcessing completed! Results saved to: {output_path}")

    return {
        'stats': merged_stats,
        'weights': weights,
        'total_chunks': len(results),
        'total_rows': total_rows
    }


# Main program
if __name__ == "__main__":
    # File path
    input_csv = r"E:\Python\pythonProject\new_t_predict\data\合理分子3.csv"
    output_csv = r"E:\Python\pythonProject\new_t_predict\data\合理分子f.csv"

    # Check if file exists
    if not os.path.exists(input_csv):
        print(f"Error: File does not exist: {input_csv}")
    else:
        print("=" * 60)
        print("Large dataset parallel Stotal calculation started")
        print("=" * 60)

        # Set parameters
        chunk_size = 500000  # 500K rows/chunk
        n_workers = 4  # Using 4 processes

        results = calculate_stotal_parallel(
            csv_path=input_csv,
            output_path=output_csv,
            chunk_size=chunk_size,
            n_workers=n_workers
        )

        if results is not None:
            print("\n" + "=" * 60)
            print("Calculation completed!")
            print("=" * 60)
            print(f"Total processed rows: {results['total_rows']:,}")
            print(f"Number of processed data chunks: {results['total_chunks']}")

            # Show final formula
            print("\nFormula used:")
            print(f"Stotal = {results['weights']['w1']} * z(-LogP) + "
                  f"{results['weights']['w2']} * z(HBD+HBA) + "
                  f"{results['weights']['w3']} * z(logSmonomer) + "
                  f"{results['weights']['intercept']}")
            print("\nStandardization parameters:")
            print(
                f"  -LogP: mean={results['stats']['-LogP']['mean']:.6f}, std={results['stats']['-LogP']['std']:.6f}")
            print(
                f"  HBD+HBA: mean={results['stats']['HBD+HBA']['mean']:.6f}, std={results['stats']['HBD+HBA']['std']:.6f}")
            print(
                f"  logSmonomer: mean={results['stats']['logSmonomer']['mean']:.6f}, std={results['stats']['logSmonomer']['std']:.6f}")
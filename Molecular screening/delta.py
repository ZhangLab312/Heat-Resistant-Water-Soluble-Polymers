import pandas as pd
from rdkit import Chem
import os
from tqdm import tqdm
import gc


# Extract core calculation logic from the given code
class NewParameterCalculator:
    """
    Utility class for calculating new parameters based on third-order group contribution method
    Process first-order, second-order, third-order groups respectively and calculate new parameter values
    """

    def __init__(self, baseline_value: float = 21.6654):
        # Try to import groupy module
        try:
            from groupy.gp_loader import Loader
            from groupy.gp_counter import Counter
            self.loader = Loader()
            self.counter = Counter()

            # Load all parameters
            self.all_parameters = self.loader.load_parameters(parameter_type='step_wise', split=False)

            # Weight coefficients (according to the formula, both w and z are 1)
            self.w = 1.0  # Second-order weight
            self.z = 1.0  # Third-order weight

            # Baseline value
            self.baseline_value = baseline_value

            # Separate parameters by group ID prefix
            self.parameters_1st = {}
            self.parameters_2nd = {}
            self.parameters_3rd = {}

            self._separate_parameters()

            # Pre-compute group parameter lookup table to avoid string comparison in every calculation
            self._build_parameter_lookup()

        except ImportError as e:
            print(f"Error: Failed to import groupy module: {e}")
            print("Please make sure groupy is installed: pip install groupy")
            raise

    def _separate_parameters(self):
        """
        Separate parameters into different orders by group ID prefix
        """
        for group_id, params in self.all_parameters.items():
            if isinstance(group_id, str):
                if group_id.startswith('f_'):
                    self.parameters_1st[group_id] = params
                elif group_id.startswith('s_'):
                    self.parameters_2nd[group_id] = params
                elif group_id.startswith('t_'):
                    self.parameters_3rd[group_id] = params
                else:
                    # If group ID does not match prefix rules, default to first-order group
                    self.parameters_1st[group_id] = params
            else:
                # If group_id is not a string, default to first-order group
                self.parameters_1st[group_id] = params

    def _build_parameter_lookup(self):
        """Build parameter lookup table to speed up parameter retrieval"""
        self.param_lookup = {}

        # Merge all parameters
        for group_id, params in self.parameters_1st.items():
            self.param_lookup[group_id] = ('1st', params.get('δ', 0.0))

        for group_id, params in self.parameters_2nd.items():
            self.param_lookup[group_id] = ('2nd', params.get('δ', 0.0))

        for group_id, params in self.parameters_3rd.items():
            self.param_lookup[group_id] = ('3rd', params.get('δ', 0.0))

    def calculate_delta(self, smiles: str):
        """
        Calculate the δ parameter value of a molecule (optimized version)
        """
        if not smiles or pd.isna(smiles) or str(smiles).strip() == '':
            return None

        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None

            # Get group counts
            group_info = self.counter.count_a_mol(mol, clear_mode=True, add_note=True)
            group_counts = {k: v for k, v in group_info.items() if k != 'note'}

            # Initialize new parameter value (starting from baseline value)
            delta_value = self.baseline_value

            # Use pre-built lookup table for acceleration
            for group_id, count in group_counts.items():
                if group_id in self.param_lookup:
                    group_type, delta_param = self.param_lookup[group_id]

                    if group_type == '1st':
                        delta_value += count * delta_param
                    elif group_type == '2nd':
                        delta_value += self.w * count * delta_param
                    else:  # '3rd'
                        delta_value += self.z * count * delta_param

            return round(delta_value, 4)

        except Exception:
            return None


# Main program
if __name__ == "__main__":
    # File path
    input_file = r"E:\Python\pythonProject\new_t_predict\data\reasonable_molecules.csv"
    output_file = r"E:\Python\pythonProject\new_t_predict\data\reasonable_molecules_delta.csv"

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File does not exist: {input_file}")
        exit()

    print(f"Reading file: {input_file}")
    print("Initializing δ parameter calculator...")

    try:
        # Initialize calculator
        calculator = NewParameterCalculator(baseline_value=21.6654)
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit()

    # Getting total number of rows in file
    print("Getting total number of rows in file...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            header_line = f.readline()
            total_rows = sum(1 for line in f)
    except UnicodeDecodeError:
        try:
            with open(input_file, 'r', encoding='gbk') as f:
                header_line = f.readline()
                total_rows = sum(1 for line in f)
        except:
            print("Unable to read file")
            exit()

    print(f"Total data rows: {total_rows:,}")

    # Reading data
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_file, encoding='gbk')
        except:
            print("Unable to read file")
            exit()

    # Check if 'solute' column exists
    if 'solute' not in df.columns:
        print("Error: File does not have 'solute' column")
        print(f"Available column names: {list(df.columns)}")
        exit()

    print(f"Starting δ parameter calculation...")

    # Calculate δ parameter and show progress bar
    delta_values = []
    for smiles in tqdm(df['solute'], total=len(df), desc="Calculating δ parameter"):
        delta = calculator.calculate_delta(smiles)
        delta_values.append(delta)

    # Add δ parameter to DataFrame
    df['δ'] = delta_values

    # Save results
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\nCalculation completed!")
    print(f"Results have been saved to: {output_file}")
    print(f"First 5 rows result preview:")
    print(df[['solute', 'δ']].head())

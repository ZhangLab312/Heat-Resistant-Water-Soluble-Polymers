# Heat-Resistant-Water-Soluble-Polymers

A deep learning framework for high-throughput screening and property prediction of water-soluble heat-resistant molecules. It identifies promising candidates from a library of 64 million hypothetical synthetic molecules.

## Environment Setup

Install all dependencies before running the code:

```bash
conda create -n hrws python=3.9
conda activate hrws
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels defaults

conda install rdkit pandas numpy scikit-learn joblib tqdm
conda install pytorch torchvision cpuonly -c pytorch
conda install xgboost
conda install jupyter

If you have an NVIDIA GPU, remove cpuonly and install the CUDA version of PyTorch for acceleration.

Usage
Save your molecular SMILES data as a .csv file and run the corresponding scripts.

Molecular Fragmentation & Synthesis
# Fragment scaffolds of heat-resistant molecules
python Bricks/T_fragment.py xxx.csv

# Fragment scaffolds of water-soluble molecules
python Bricks/W_fragment.py xxx.csv

# Synthesize molecules with customized fragment ratio (connect1_1 / connect1_2 / connect2_1 available)
python Bricks/connect1_1.py xxx.csv

# Generate intermediates from homogeneous fragments
python Bricks/intermediate.py xxx.csv

Property Prediction
# Predict thermal decomposition temperature (Td)
python Td/Td_prediction.py xxx.csv

# Predict water solubility (LogS)
python LogS/logs_prediction.py xxx.csv

Baseline Model Comparison
This is only a running example. The folder contains GB, LR, RF, RR, SVM, DT, KNN, XGB and other baseline models.

python "Model compare"/xgb.py xxx.csv

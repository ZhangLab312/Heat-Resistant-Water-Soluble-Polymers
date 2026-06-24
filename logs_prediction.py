import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import ConvertToNumpyArray
import joblib
import torch
import torch.nn as nn
import warnings
import gc
from tqdm import tqdm
import os
import time

warnings.filterwarnings('ignore')
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

# 配置参数
CONFIG = {
    "input_file": r"E:\Python\pythonProject\new_t_predict\data\合理分子1.csv",
    "output_file": r"E:\Python\pythonProject\new_t_predict\data\合理分子2.csv",
    "output_chunks_dir": r"E:\Python\pythonProject\new_t_predict\data\chunks",

    "model_path": r"E:/Python/pythonProject/new_t_predict/model/logs_model.pth",
    "scaler_path": r"E:/Python/pythonProject/new_t_predict/scaler/logs_scaler.pkl",

    "fingerprint": {"radius": 2, "n_bits": 1024},
    "nn_params": {"hidden_layers": [512, 256], "dropout_rate": 0.5},

    "batch_size": 1024,
    "chunk_size": 100000,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


class FNN(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        prev_size = input_size
        for h_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = h_size
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, 1)

    def forward(self, x):
        return self.output(self.hidden(x))


class SolubilityPredictor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        os.makedirs(config["output_chunks_dir"], exist_ok=True)
        self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        print("加载模型与标准化器...")
        self.scaler = joblib.load(self.config["scaler_path"])
        self.model = FNN(
            input_size=self.config["fingerprint"]["n_bits"],
            hidden_layers=self.config["nn_params"]["hidden_layers"],
            dropout_rate=self.config["nn_params"]["dropout_rate"]  # 修复这里！
        )
        self.model.load_state_dict(torch.load(self.config["model_path"], map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("模型加载完成 ✅")

    # ====================== ✅ SMILES 标准化（已保留） ======================
    def smiles_to_fingerprint(self, smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if not mol:
                return None

            # 标准化SMILES，确保同一物质结构一致
            canonical_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            mol = Chem.MolFromSmiles(canonical_smi)
            if not mol:
                return None

            fp = GetMorganFingerprintAsBitVect(
                mol,
                radius=self.config["fingerprint"]["radius"],
                nBits=self.config["fingerprint"]["n_bits"]
            )
            arr = np.zeros((self.config["fingerprint"]["n_bits"],), dtype=int)
            ConvertToNumpyArray(fp, arr)
            return arr
        except:
            return None
    # =======================================================================

    def batch_smiles_to_fingerprints(self, smiles_list):
        features = []
        valid_indices = []
        for i, smi in enumerate(tqdm(smiles_list, desc="生成指纹", leave=False)):
            fp = self.smiles_to_fingerprint(smi)
            if fp is not None:
                features.append(fp)
                valid_indices.append(i)
        if features:
            return self.scaler.transform(np.array(features)), valid_indices
        return None, []

    def predict_batch(self, features):
        with torch.no_grad():
            tensor = torch.FloatTensor(features).to(self.device)
            preds = []
            for i in range(0, len(tensor), self.config["batch_size"]):
                batch = tensor[i:i + self.config["batch_size"]]
                preds.append(self.model(batch).cpu().numpy())
                del batch
            del tensor
            torch.cuda.empty_cache()
            return np.concatenate(preds).flatten()

    def process_chunk(self, chunk):
        smiles_list = chunk["SMILES"].tolist()
        features, valid_idx = self.batch_smiles_to_fingerprints(smiles_list)

        if features is None:
            chunk["predicted_logS"] = np.nan
            return chunk

        pred = self.predict_batch(features)

        res = chunk.iloc[valid_idx].copy()
        res["predicted_logS"] = pred

        res = res[["SMILES", "thermal_decomposition_temperature", "SA_Score", "Stotal", "HBD", "HBA", "Herror", "predicted_logS"]]
        return res

    def run(self):
        total_rows = sum(1 for _ in open(CONFIG["input_file"], encoding="utf-8")) - 1
        reader = pd.read_csv(CONFIG["input_file"], chunksize=CONFIG["chunk_size"], encoding="utf-8")
        pbar = tqdm(total=total_rows, desc="预测进度")

        for i, chunk in enumerate(reader):
            processed = self.process_chunk(chunk)
            processed.to_csv(os.path.join(CONFIG["output_chunks_dir"], f"chunk_{i:06d}.csv"), index=False, encoding="utf-8")
            pbar.update(len(chunk))
        pbar.close()

        print("\n合并文件...")
        dfs = []
        for f in sorted(os.listdir(CONFIG["output_chunks_dir"])):
            dfs.append(pd.read_csv(os.path.join(CONFIG["output_chunks_dir"], f)))
        final = pd.concat(dfs, ignore_index=True)
        final.to_csv(CONFIG["output_file"], index=False, encoding="utf-8")
        print(f"✅ 完成！文件已保存至：{CONFIG['output_file']}")


def main():
    predictor = SolubilityPredictor(CONFIG)
    predictor.run()

if __name__ == "__main__":
    main()
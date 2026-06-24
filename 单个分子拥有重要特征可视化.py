import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# 输入 SMILES
smiles = "CCCCCCCCCCCCOc1cc(cc(c1OCCCCCCCCCCCC)OCCCCCCCCCCCC)C(=O)Oc1ccc(cc1)NC(=O)c1cc(cc(c1)c1cccc(c1)N1C(=O)CC(C1=O)C1C=C(C)C2C(C1)C(=O)N(C2=O)*)c1cccc(c1)*"

mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError("无效的 SMILES 字符串")

radius = 2
nBits = 1024
bitInfo = {}

fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, bitInfo=bitInfo)

bits = [80, 807, 361, 136, 498, 935, 726, 13, 792, 322, 33, 429, 843, 695, 705]

dist_matrix = Chem.GetDistanceMatrix(mol)

# 记录每个位覆盖的原子（半径2内）
bit_atoms = {}
for b in bits:
    atoms = set()
    if b in bitInfo:
        for atom_idx, rad in bitInfo[b]:
            for neighbor, dist in enumerate(dist_matrix[atom_idx]):
                if dist <= 2:
                    atoms.add(neighbor)
    bit_atoms[b] = atoms
    print(f"位 {b}: 覆盖 {len(atoms)} 个原子")

# 保存目录
save_dir = r"E:\Python\pythonProject\new_t_predict\data\shap分析结果\bits"
os.makedirs(save_dir, exist_ok=True)

# 浅绿色 (R,G,B) 在 0-1 范围
light_green = (0.6, 1.0, 0.6)   # 浅绿色

# 为每个有高亮原子的位生成单独图像
for b, atoms in bit_atoms.items():
    if not atoms:
        print(f"位 {b} 无高亮原子，跳过")
        continue

    # 构建颜色字典：所有高亮原子都使用浅绿色
    color_dict = {atom: light_green for atom in atoms}

    # 绘制 SVG
    drawer = rdMolDraw2D.MolDraw2DSVG(1000, 800)
    drawer.DrawMolecule(mol, highlightAtoms=list(atoms), highlightAtomColors=color_dict)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # 保存文件
    file_path = os.path.join(save_dir, f"bit_{b}_highlight.svg")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"位 {b} 的高亮图像已保存至 {file_path}")

print("所有位处理完成")
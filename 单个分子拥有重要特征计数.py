import os
import colorsys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

def generate_distinct_colors(n):
    """生成 n 个视觉可区分的颜色（返回 RGB 浮点元组列表）"""
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.5 + 0.2 * (i % 3)
        saturation = 0.8
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((r, g, b))
    return colors

def average_colors(colors_rgb):
    """计算多个 RGB 颜色的平均值"""
    if not colors_rgb:
        return (0.0, 0.0, 0.0)
    n = len(colors_rgb)
    r_avg = sum(c[0] for c in colors_rgb) / n
    g_avg = sum(c[1] for c in colors_rgb) / n
    b_avg = sum(c[2] for c in colors_rgb) / n
    return (r_avg, g_avg, b_avg)

# 输入 SMILES
smiles = "*Oc1cc(OC(=O)c2ccc(OCC)cc2)c(OC(=O)CCCCCCCCCCCCCCC(*)=O)cc1OC(=O)c1ccc(OCC)cc1"

mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError("无效的 SMILES 字符串")

radius = 2
nBits = 1024
bitInfo = {}

fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits, bitInfo=bitInfo)

bits = [375, 33, 128, 183, 272, 13, 152, 362, 80, 356, 361, 138, 293, 161, 119, 264, 452, 337,214, 333]

dist_matrix = Chem.GetDistanceMatrix(mol)

# 为每个位分配颜色
num_bits = len(bits)
colors_rgb = generate_distinct_colors(num_bits)
bit_to_color = {b: colors_rgb[i] for i, b in enumerate(bits)}

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

# 构建原子到颜色列表映射
atom_to_colors = {}
for b, atoms in bit_atoms.items():
    col = bit_to_color[b]
    for a in atoms:
        atom_to_colors.setdefault(a, []).append(col)

# 混合颜色，得到最终原子颜色
highlight_colors = {}
for a, cols in atom_to_colors.items():
    if len(cols) == 1:
        highlight_colors[a] = cols[0]
    else:
        highlight_colors[a] = average_colors(cols)

print(f"共高亮 {len(highlight_colors)} 个原子")

if not highlight_colors:
    print("没有找到任何高亮原子，退出。")
    exit()

# 使用 rdMolDraw2D 绘制 SVG
drawer = rdMolDraw2D.MolDraw2DSVG(1000, 800)
drawer.DrawMolecule(mol, highlightAtoms=list(highlight_colors.keys()), highlightAtomColors=highlight_colors)
drawer.FinishDrawing()
svg = drawer.GetDrawingText()

# 保存到指定目录
save_dir = r"E:\Python\pythonProject\new_t_predict\data\Tm_shap分析结果"
os.makedirs(save_dir, exist_ok=True)
file_path = os.path.join(save_dir, "highlighted_molecule_colored.svg")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(svg)

print(f"彩色高亮图像已保存为 {file_path}")
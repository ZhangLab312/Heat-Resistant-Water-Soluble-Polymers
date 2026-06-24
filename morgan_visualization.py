import pandas as pd
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib as mpl
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# 设置字体和绘图样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# 配置参数
CONFIG = {
    "shap_path": "E:/Python/pythonProject/new_t_predict/shap/",
    "output_path": "E:/Python/pythonProject/new_t_predict/feature_analysis/",
    "morgan_path": "E:/Python/pythonProject/new_t_predict/feature_analysis/morgan/",
    "input_csv": "E:/Python/pythonProject/new_t_predict/data/cleaned_predictions_no_si.csv",
    "fingerprint": {
        "radius": 2,
        "n_bits": 1024
    },
    "visualization": {
        "top_n_features": 20,
        "pie_chart_colors": plt.cm.Set3(np.linspace(0, 1, 20)),
        "figure_size": (12, 8)
    }
}

# 确保输出目录存在
os.makedirs(CONFIG["output_path"], exist_ok=True)
os.makedirs(CONFIG["morgan_path"], exist_ok=True)


def modify_shap_beeswarm_colors():
    """修改SHAP beeswarm图的颜色"""
    try:
        beeswarm_path = os.path.join(CONFIG["output_path"], "shap_beeswarm_plot.png")
        if os.path.exists(beeswarm_path):
            img = Image.open(beeswarm_path)
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.title('SHAP Beeswarm Plot (Custom Colors)', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.show()
        else:
            print(f"未找到SHAP beeswarm图: {beeswarm_path}")
    except Exception as e:
        print(f"显示SHAP beeswarm图时出错: {e}")


def load_analysis_data():
    """加载前两步的分析数据"""
    print("加载分析数据...")

    important_features = pd.read_csv(os.path.join(CONFIG["output_path"], "important_features_filtered.csv"))

    if 'percentage' not in important_features.columns:
        print("检测到缺少percentage列，重新计算百分比...")
        total_importance = important_features['importance'].sum()
        important_features['percentage'] = (important_features['importance'] / total_importance) * 100
        important_features = important_features.sort_values('percentage', ascending=False)
        important_features.to_csv(os.path.join(CONFIG["output_path"], "important_features_filtered.csv"), index=False)

    shap_data = np.load(os.path.join(CONFIG["shap_path"], "shap_analysis_data.npz"), allow_pickle=True)
    important_shap_data = np.load(os.path.join(CONFIG["output_path"], "important_shap_values.npz"), allow_pickle=True)

    df = pd.read_csv(CONFIG["input_csv"])
    valid_data_info = pd.read_csv(os.path.join(CONFIG["shap_path"], "valid_data_indices.csv"))
    smiles_list = df.iloc[valid_data_info['original_index']]['smiles'].values

    print(f"重要特征数量: {len(important_features)}")
    print(f"SMILES数据数量: {len(smiles_list)}")

    return important_features, shap_data, important_shap_data, smiles_list


def plot_percentage_pie_chart(important_features):
    """绘制特征百分比占比饼图"""
    print("绘制百分比占比饼图...")
    top_features = important_features.head(CONFIG["visualization"]["top_n_features"])
    plt.figure(figsize=CONFIG["visualization"]["figure_size"])

    other_percentage = 100 - top_features['percentage'].sum()

    if other_percentage > 0:
        percentages = list(top_features['percentage']) + [other_percentage]
        labels = [f"FP_{idx}\n({pct:.2f}%)" for idx, pct in
                  zip(top_features['feature_index'], top_features['percentage'])] + [
                     f"Other\n({other_percentage:.2f}%)"]
        colors = list(CONFIG["visualization"]["pie_chart_colors"][:len(top_features)]) + ['lightgray']
    else:
        percentages = list(top_features['percentage'])
        labels = [f"FP_{idx}\n({pct:.2f}%)" for idx, pct in
                  zip(top_features['feature_index'], top_features['percentage'])]
        colors = CONFIG["visualization"]["pie_chart_colors"][:len(top_features)]

    wedges, texts, autotexts = plt.pie(
        percentages, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 8}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.title(f'Top {len(top_features)} Important Features Percentage Distribution', fontsize=16, fontweight='bold',
              pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_path"], "feature_percentage_pie_chart.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(CONFIG["output_path"], "feature_percentage_pie_chart.pdf"), bbox_inches='tight')
    plt.show()

    # 水平条形图
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features['percentage'], color=CONFIG["visualization"]["pie_chart_colors"][:len(top_features)])
    plt.yticks(y_pos, [f"FP_{idx}" for idx in top_features['feature_index']])
    plt.xlabel('Importance Percentage (%)', fontsize=12)
    plt.title(f'Top {len(top_features)} Important Features Percentage', fontsize=16, fontweight='bold')

    for i, v in enumerate(top_features['percentage']):
        plt.text(v + 0.1, i, f'{v:.2f}%', va='center', fontsize=9)

    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_path"], "feature_percentage_bar_chart.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(CONFIG["output_path"], "feature_percentage_bar_chart.pdf"), bbox_inches='tight')
    plt.show()


def plot_shap_summary(important_features, shap_data, important_shap_data):
    """绘制SHAP分析图（包含全量数据的蜂巢图）"""
    import shap
    from matplotlib.colors import LinearSegmentedColormap

    print("绘制SHAP分析图...")

    custom_colors = ["#F3D08B", "#E29B67", "#D15636", "#A83020", "#7D1812"]
    target_cmap = LinearSegmentedColormap.from_list("target_maroon_v2", custom_colors, N=256)

    important_indices = important_features['feature_index'].values
    important_feature_names = [f"FP_{idx}" for idx in important_indices]
    important_shap_values = important_shap_data['shap_values']

    # 1. SHAP特征重要性条形图
    plt.figure(figsize=(12, 10))
    mean_abs_shap = np.abs(important_shap_values).mean(0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_features = [important_feature_names[i] for i in sorted_idx[:CONFIG["visualization"]["top_n_features"]]]
    sorted_values = mean_abs_shap[sorted_idx[:CONFIG["visualization"]["top_n_features"]]]

    reversed_features = sorted_features[::-1]
    reversed_values = sorted_values[::-1]

    bar_colors = target_cmap(np.linspace(0.0, 1.0, len(reversed_values)))
    bars = plt.barh(range(len(reversed_features)), reversed_values, color=bar_colors, edgecolor='#555555',
                    linewidth=0.6)

    plt.yticks(range(len(reversed_features)), reversed_features, fontsize=11)
    plt.xlabel('Mean(|SHAP|) value', fontsize=12)
    plt.title('SHAP Feature Importance Ranking', fontsize=16, fontweight='bold', pad=15)

    total_top_values = sorted_values.sum()
    for i, v in enumerate(reversed_values):
        pct = (v / total_top_values) * 100
        plt.text(v + 0.03, i, f'{pct:.1f}%', va='center', ha='left', fontsize=9, color='#444444')

    plt.grid(True, axis='x', linestyle=':', alpha=0.6, color='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_path"], "shap_feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(CONFIG["output_path"], "shap_feature_importance.pdf"), bbox_inches='tight')
    plt.show()

    # 2. SHAP beeswarm图 (使用全样本数据)
    plt.figure(figsize=(12, 10))
    print("正在生成全样本 SHAP Beeswarm 图，计算量较大，请稍候...")

    sorted_idx_full = np.argsort(mean_abs_shap)[::-1]
    top_k = min(20, len(important_indices))

    shap_values_top = important_shap_values[:, sorted_idx_full[:top_k]]
    feature_values_top = shap_data['X_data'][:, important_indices[sorted_idx_full[:top_k]]]
    feature_names_top = [important_feature_names[i] for i in sorted_idx_full[:top_k]]

    # 临时修改全局字体大小
    original_font_size = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': 16})

    shap.summary_plot(
        shap_values_top,
        feature_values_top,
        feature_names=feature_names_top,
        show=False,
        plot_size=(12, 10),
        max_display=top_k,
        cmap=target_cmap
    )

    # =============== 核心修改区：强制接管并覆盖 shap 的硬编码字号 ===============
    fig = plt.gcf()

    # 1. 强制修改主坐标轴（左侧特征名、下侧SHAP值刻度、X轴标签）
    ax = fig.axes[0]
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=16)

    # 2. 重新设置大号标题
    plt.title('SHAP Beeswarm Plot (All Samples)', fontsize=20, fontweight='bold', pad=20)

    # 3. 强制修改色条（Colorbar）及两端的 Low / High 文字
    if len(fig.axes) > 1:
        cax = fig.axes[-1]  # 色条坐标轴
        # 强制覆盖 "Feature value"
        cax.set_ylabel("Feature value", fontsize=16)
        # 强制覆盖 "Low" 和 "High" 的刻度字号
        cax.tick_params(labelsize=16)

        # 遍历所有文本元素强制修改 Low/High 等文本
        for t in cax.texts:
            t.set_fontsize(16)
        for t in fig.texts:
            if t.get_text() in ["Low", "High", "Feature value"]:
                t.set_fontsize(16)
    # ===========================================================================

    plt.grid(True, axis='x', linestyle=':', alpha=0.6, color='gray')
    plt.tight_layout()

    plt.savefig(os.path.join(CONFIG["output_path"], "shap_beeswarm_plot.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(CONFIG["output_path"], "shap_beeswarm_plot.pdf"), bbox_inches='tight')
    plt.close()

    # 恢复全局字体设置
    plt.rcParams.update({'font.size': original_font_size})

    modify_shap_beeswarm_colors()


def plot_positive_negative_correlation(important_features, important_shap_data, shap_data):
    """绘制正负相关图与特征散点图"""
    from matplotlib.colors import LinearSegmentedColormap
    print("绘制正负相关图...")

    custom_colors = ["#F3D08B", "#E29B67", "#D15636", "#A83020", "#7D1812"]
    target_cmap = LinearSegmentedColormap.from_list("target_maroon_v2", custom_colors, N=256)

    important_shap_values = important_shap_data['shap_values']
    important_indices = important_features['feature_index'].values
    X_original = shap_data['X_original'][:, important_indices]

    mean_shap_when_present = np.zeros(len(important_indices))

    for i in range(len(important_indices)):
        active_indices = np.where(X_original[:, i] == 1)[0]
        if len(active_indices) > 0:
            mean_shap_when_present[i] = important_shap_values[active_indices, i].mean()
        else:
            mean_shap_when_present[i] = 0.0

    top_n = min(CONFIG["visualization"]["top_n_features"], len(important_indices))
    top_indices = important_features.head(top_n)['feature_index'].values
    top_impacts = mean_shap_when_present[:top_n]

    # ---------- 条形图字体全面加大 ----------
    plt.figure(figsize=(14, 12))
    colors = ['#E29B67' if x < 0 else '#A83020' for x in top_impacts]
    y_pos = np.arange(top_n)

    plt.barh(y_pos, top_impacts, color=colors, alpha=0.8)
    plt.yticks(y_pos, [f"FP_{idx}" for idx in top_indices], fontsize=18)  # 14->18
    plt.xlabel('Mean SHAP Value (When Feature is Present)', fontsize=20)  # 16->20
    plt.title('Feature Impact Direction\n(Warm Orange: Decrease Output, Dark Red: Increase Output)', fontsize=24,
              fontweight='bold')  # 20->24

    for i, v in enumerate(top_impacts):
        offset = 0.05 if v >= 0 else -0.05
        plt.text(v + offset, i, f'{v:.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=17, color='black',
                 fontweight='bold')  # 13->17

    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_path"], "positive_negative_correlation.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(CONFIG["output_path"], "positive_negative_correlation.pdf"), bbox_inches='tight')
    plt.show()

    # ---------- 散点图字体全面加大 ----------
    plt.figure(figsize=(12, 10))
    top_features = important_features.head(top_n)

    sc = plt.scatter(top_features['importance'], top_impacts, c=top_impacts, cmap=target_cmap, alpha=0.9, s=120,
                     edgecolor='#444444', linewidth=0.7)

    for i, row in top_features.iterrows():
        plt.annotate(f"FP_{row['feature_index']}", (row['importance'], top_impacts[i]), xytext=(5, 5),
                     textcoords='offset points', fontsize=16)  # 12->16

    plt.xlabel('Feature Importance (Mean |SHAP|)', fontsize=20)  # 16->20
    plt.ylabel('Impact Direction\n(Mean SHAP when feature is present)', fontsize=20)  # 16->20
    plt.title('Feature Importance vs Impact Direction', fontsize=24, fontweight='bold')  # 20->24

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.4)
    x_median = np.median(top_features['importance'])
    plt.axvline(x=x_median, color='black', linestyle='--', alpha=0.4)

    # 色条和刻度标签放大
    cbar = plt.colorbar(sc, label='Impact (Mean SHAP when present)')
    cbar.set_label('Impact (Mean SHAP when present)', fontsize=18)  # 14->18
    cbar.ax.tick_params(labelsize=16)  # 12->16

    plt.xticks(fontsize=18)  # 14->18
    plt.yticks(fontsize=18)  # 14->18
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_path"], "importance_vs_impact.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(CONFIG["output_path"], "importance_vs_impact.pdf"), bbox_inches='tight')
    plt.show()


def visualize_morgan_fingerprints(important_features, smiles_list, top_n=10):
    """摩根指纹可视化"""
    print("进行摩根指纹可视化（带原子高亮）...")
    top_features = important_features.head(top_n)
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, (ax, (idx, row)) in enumerate(zip(axes, top_features.iterrows())):
        if i >= len(axes):
            break

        feature_idx = row['feature_index']
        feature_importance = row['importance']
        feature_percentage = row['percentage']
        print(f"\n处理特征 FP_{feature_idx} (重要性: {feature_importance:.6f}, 占比: {feature_percentage:.4f}%)")

        sample_smiles = np.random.choice(smiles_list, min(100, len(smiles_list)), replace=False)
        found_substructure = False

        for smi in sample_smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    bit_info = {}
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=CONFIG["fingerprint"]["radius"],
                                                               nBits=CONFIG["fingerprint"]["n_bits"], bitInfo=bit_info)

                    if fp[feature_idx] and feature_idx in bit_info:
                        envs = bit_info[feature_idx]
                        if envs:
                            atom_idx, radius = envs[0]
                            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                            amap = {}
                            submol = Chem.PathToSubmol(mol, env, atomMap=amap)

                            if submol is not None and submol.GetNumAtoms() > 0:
                                highlight_atoms = {}
                                center_atom_in_submol = None
                                for submol_atom_idx in range(submol.GetNumAtoms()):
                                    if amap.get(submol_atom_idx) == atom_idx:
                                        center_atom_in_submol = submol_atom_idx
                                        break

                                for submol_atom_idx in range(submol.GetNumAtoms()):
                                    atom = submol.GetAtomWithIdx(submol_atom_idx)
                                    if submol_atom_idx == center_atom_in_submol:
                                        highlight_atoms[submol_atom_idx] = (0.0, 0.0, 1.0)  # 蓝
                                    elif atom.GetIsAromatic():
                                        highlight_atoms[submol_atom_idx] = (1.0, 1.0, 0.0)  # 黄
                                    else:
                                        highlight_atoms[submol_atom_idx] = (0.5, 0.5, 0.5)  # 灰

                                # 修改点 1：使用 rdMolDraw2D 替换 Draw.MolToImage
                                drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
                                drawer.DrawMolecule(
                                    submol,
                                    highlightAtoms=list(highlight_atoms.keys()),
                                    highlightAtomColors=highlight_atoms,
                                    highlightBonds=[]
                                )
                                drawer.FinishDrawing()
                                # 将底层的字节流转换回 PIL Image 格式供 matplotlib 展示
                                img = Image.open(io.BytesIO(drawer.GetDrawingText()))

                                ax.imshow(img)
                                ax.axis('off')
                                ax.set_title(f"FP_{feature_idx}\nR={radius}, {feature_percentage:.2f}%", fontsize=10,
                                             fontweight='bold')
                                found_substructure = True
                                break
            except Exception as e:
                continue

        if not found_substructure:
            ax.text(0.5, 0.5, f"FP_{feature_idx}\nNo substructure found", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.axis('off')

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.7, label='Center Atom'),
        plt.Rectangle((0, 0), 1, 1, fc='yellow', alpha=0.7, label='Aromatic Atom'),
        plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.7, label='Aliphatic Atom')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle(
        f'Top {top_n} Important Features Morgan Fingerprint Substructures\nFingerprint Radius={CONFIG["fingerprint"]["radius"]}',
        fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(CONFIG["morgan_path"], "morgan_substructures_with_highlight.png"), dpi=300,
                bbox_inches='tight')
    plt.show()

    print("\n生成详细子结构可视化（带原子高亮）...")
    for i, (idx, row) in enumerate(top_features.iterrows()):
        if i >= top_n:
            break

        feature_idx = row['feature_index']
        feature_importance = row['importance']
        feature_percentage = row['percentage']
        sample_smiles = np.random.choice(smiles_list, min(200, len(smiles_list)), replace=False)
        substructures = []

        for smi in sample_smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    bit_info = {}
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=CONFIG["fingerprint"]["radius"],
                                                               nBits=CONFIG["fingerprint"]["n_bits"], bitInfo=bit_info)
                    if fp[feature_idx] and feature_idx in bit_info:
                        envs = bit_info[feature_idx]
                        for atom_idx, radius in envs:
                            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                            amap = {}
                            submol = Chem.PathToSubmol(mol, env, atomMap=amap)

                            if submol is not None and submol.GetNumAtoms() > 0:
                                smi_sub = Chem.MolToSmiles(submol)
                                if smi_sub not in [Chem.MolToSmiles(s) for s, _ in substructures]:
                                    center_atom_in_submol = None
                                    for submol_atom_idx in range(submol.GetNumAtoms()):
                                        if amap.get(submol_atom_idx) == atom_idx:
                                            center_atom_in_submol = submol_atom_idx
                                            break
                                    substructures.append((submol, center_atom_in_submol))
                            if len(substructures) >= 5:
                                break
                        if len(substructures) >= 5:
                            break
            except Exception as e:
                continue

        if substructures:
            n_cols = min(5, len(substructures))
            n_rows = (len(substructures) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for j, (ax, (submol, center_atom)) in enumerate(zip(axes, substructures)):
                if j < len(substructures):
                    highlight_atoms = {}
                    for submol_atom_idx in range(submol.GetNumAtoms()):
                        atom = submol.GetAtomWithIdx(submol_atom_idx)
                        if submol_atom_idx == center_atom:
                            highlight_atoms[submol_atom_idx] = (0.0, 0.0, 1.0)  # 蓝
                        elif atom.GetIsAromatic():
                            highlight_atoms[submol_atom_idx] = (1.0, 1.0, 0.0)  # 黄
                        else:
                            highlight_atoms[submol_atom_idx] = (0.5, 0.5, 0.5)  # 灰

                    # 修改点 2：使用 rdMolDraw2D 替换子结构变体图的 Draw.MolToImage
                    drawer = rdMolDraw2D.MolDraw2DCairo(200, 200)
                    drawer.DrawMolecule(
                        submol,
                        highlightAtoms=list(highlight_atoms.keys()),
                        highlightAtomColors=highlight_atoms,
                        highlightBonds=[]
                    )
                    drawer.FinishDrawing()
                    img = Image.open(io.BytesIO(drawer.GetDrawingText()))

                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f"Substructure {j + 1}", fontsize=9)
                else:
                    ax.axis('off')

            fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.01))
            plt.suptitle(
                f'Feature FP_{feature_idx} Substructure Variants\nImportance: {feature_importance:.6f}, Percentage: {feature_percentage:.4f}%, Fingerprint Radius: {CONFIG["fingerprint"]["radius"]}',
                fontsize=12, fontweight='bold', y=0.95)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(os.path.join(CONFIG["morgan_path"], f"morgan_fp_{feature_idx}_substructures_highlighted.png"),
                        dpi=300, bbox_inches='tight')
            plt.show()
            print(f"  特征 FP_{feature_idx}: 找到 {len(substructures)} 个子结构变体")
        else:
            print(f"  特征 FP_{feature_idx}: 未找到子结构")

    print("\n创建指纹位激活频率热图...")
    sample_size = min(100, len(smiles_list))
    sample_smiles = np.random.choice(smiles_list, sample_size, replace=False)
    activation_matrix = np.zeros((sample_size, len(top_features)))
    feature_indices = top_features['feature_index'].values

    for i, smi in enumerate(sample_smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=CONFIG["fingerprint"]["radius"],
                                                           nBits=CONFIG["fingerprint"]["n_bits"])
                for j, idx in enumerate(feature_indices):
                    activation_matrix[i, j] = fp[idx]
        except:
            continue

    plt.figure(figsize=(15, 12))
    sns.heatmap(activation_matrix.T, cmap='YlOrRd', cbar_kws={'label': 'Activation Status (0/1)'}, xticklabels=False,
                yticklabels=[f"FP_{idx}" for idx in feature_indices])
    plt.title(f'Top {len(top_features)} Important Features Activation Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Molecule Samples')
    plt.ylabel('Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["morgan_path"], "fingerprint_activation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("开始执行第三步：绘图分析")
    important_features, shap_data, important_shap_data, smiles_list = load_analysis_data()
    plot_percentage_pie_chart(important_features)
    plot_shap_summary(important_features, shap_data, important_shap_data)
    plot_positive_negative_correlation(important_features, important_shap_data, shap_data)
    visualize_morgan_fingerprints(important_features, smiles_list, top_n=20)
    print(f"\n=== 第三步绘图完成 ===")


if __name__ == "__main__":
    main()
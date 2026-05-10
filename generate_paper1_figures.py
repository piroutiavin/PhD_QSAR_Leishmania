"""
Generate Paper 1 figures:
  1. Chemical space PCA  (train vs test, active vs inactive)
  2. Scaffold diversity  (unique Murcko scaffolds per split)
  3. pIC50 distribution  (train vs test overlay)
  4. Dataset composition summary bar chart
Outputs saved to figures/paper1_*.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(r'e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania')
DATA = ROOT / 'data' / 'processed'
FIG  = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

PALETTE = {'Active': '#2196F3', 'Inactive': '#FF5722'}
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data ...")
train_set = pd.read_csv(DATA / 'train_set.csv')
test_set  = pd.read_csv(DATA / 'test_set.csv')
curated   = pd.read_csv(DATA / 'curated_dataset.csv')

# Labels aligned with selected feature matrices
y_train_df = pd.read_csv(DATA / 'y_train.csv', index_col=0)
y_test_df  = pd.read_csv(DATA / 'y_test.csv',  index_col=0)

X_tr = pd.read_csv(DATA / 'X_train_selected.csv', index_col=0)
X_te = pd.read_csv(DATA / 'X_test_selected.csv',  index_col=0)

# Align
common_tr = X_tr.index.intersection(y_train_df.index)
common_te = X_te.index.intersection(y_test_df.index)
X_tr = X_tr.loc[common_tr];  y_tr_lbl = y_train_df.loc[common_tr, 'activity_class']
X_te = X_te.loc[common_te];  y_te_lbl = y_test_df.loc[common_te,  'activity_class']

print(f"  Train: {X_tr.shape}  |  Test: {X_te.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Chemical Space PCA
# ─────────────────────────────────────────────────────────────────────────────
print("Computing PCA ...")
X_all   = np.vstack([X_tr.values, X_te.values])
y_all   = np.concatenate([y_tr_lbl.values, y_te_lbl.values])
split   = np.array(['Train'] * len(X_tr) + ['Test'] * len(X_te))

pca = PCA(n_components=2, random_state=SEED)
Z   = pca.fit_transform(X_all)
var = pca.explained_variance_ratio_ * 100

df_pca = pd.DataFrame({
    'PC1': Z[:, 0], 'PC2': Z[:, 1],
    'Activity': y_all, 'Split': split
})

# Subsample for plot readability (max 3000 points)
rng = np.random.default_rng(SEED)
if len(df_pca) > 3000:
    idx = rng.choice(len(df_pca), 3000, replace=False)
    df_plot = df_pca.iloc[idx]
else:
    df_plot = df_pca

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, hue_col, palette, title_suffix in [
    (axes[0], 'Activity', PALETTE,
     'by Activity Class'),
    (axes[1], 'Split',
     {'Train': '#455A64', 'Test': '#FFC107'},
     'by Train / Test Split'),
]:
    for cat, grp in df_plot.groupby(hue_col):
        marker = 'o' if (hue_col == 'Split' and cat == 'Train') or hue_col == 'Activity' else 's'
        ax.scatter(grp['PC1'], grp['PC2'],
                   c=palette[cat], label=cat,
                   alpha=0.4, s=12, marker=marker, linewidths=0)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var[1]:.1f}% variance)', fontsize=11)
    ax.set_title(f'Chemical Space PCA — {title_suffix}', fontsize=12)
    ax.legend(fontsize=10, markerscale=2)
    sns.despine(ax=ax)

plt.suptitle(
    f'Chemical Space (943 selected descriptors, PCA)\n'
    f'n = {len(X_tr):,} train + {len(X_te):,} test compounds',
    fontsize=11, y=1.01
)
plt.tight_layout()
out = FIG / 'paper1_chemical_space_PCA.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Scaffold Diversity
# ─────────────────────────────────────────────────────────────────────────────
print("Computing Murcko scaffolds ...")

def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return ''
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc)
    except Exception:
        return ''

smiles_col = 'std_smiles' if 'std_smiles' in train_set.columns else 'canonical_smiles'

train_scaffolds = train_set[smiles_col].apply(get_scaffold)
test_scaffolds  = test_set[smiles_col].apply(get_scaffold)

n_tr_unique = train_scaffolds[train_scaffolds != ''].nunique()
n_te_unique = test_scaffolds[test_scaffolds != ''].nunique()
overlap     = set(train_scaffolds) & set(test_scaffolds) - {''}
n_overlap   = len(overlap)

print(f"  Train unique scaffolds : {n_tr_unique}")
print(f"  Test  unique scaffolds : {n_te_unique}")
print(f"  Scaffold overlap       : {n_overlap}")

# Scaffold frequency distribution
tr_freq = train_scaffolds[train_scaffolds != ''].value_counts()
te_freq = test_scaffolds[test_scaffolds != ''].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: singletons vs multi-compound scaffolds
ax = axes[0]
tr_bins = pd.cut(tr_freq, bins=[0,1,2,5,10,9999],
                 labels=['1', '2', '3-5', '6-10', '>10'])
te_bins = pd.cut(te_freq, bins=[0,1,2,5,10,9999],
                 labels=['1', '2', '3-5', '6-10', '>10'])

x_cats = ['1', '2', '3-5', '6-10', '>10']
tr_counts = [tr_bins.value_counts().get(c, 0) for c in x_cats]
te_counts = [te_bins.value_counts().get(c, 0) for c in x_cats]
x = np.arange(len(x_cats))
w = 0.38
ax.bar(x - w/2, tr_counts, w, color='#455A64', alpha=0.85, label='Train')
ax.bar(x + w/2, te_counts, w, color='#FFC107', alpha=0.85, label='Test')
ax.set_xticks(x); ax.set_xticklabels([f'{c} cpd' for c in x_cats])
ax.set_ylabel('Number of scaffolds'); ax.set_title('Scaffold Size Distribution')
ax.legend(); sns.despine(ax=ax)

# Panel B: diversity summary table as bar
ax = axes[1]
cats = ['Total\ncompounds', 'Unique\nscaffolds', 'Scaffold\noverlap (n)']
tr_vals = [len(train_set), n_tr_unique, n_overlap]
te_vals = [len(test_set),  n_te_unique, n_overlap]
x = np.arange(len(cats))
ax.bar(x - w/2, tr_vals, w, color='#455A64', alpha=0.85, label='Train')
ax.bar(x + w/2, te_vals, w, color='#FFC107', alpha=0.85, label='Test')
for xi, (tv, ev) in enumerate(zip(tr_vals, te_vals)):
    ax.text(xi - w/2, tv + max(tr_vals)*0.01, str(tv),
            ha='center', va='bottom', fontsize=9)
    ax.text(xi + w/2, ev + max(tr_vals)*0.01, str(ev),
            ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel('Count'); ax.set_title('Scaffold Diversity Summary')
ax.legend(); sns.despine(ax=ax)

plt.suptitle('Murcko Scaffold Analysis — Train / Test Split', fontsize=12)
plt.tight_layout()
out = FIG / 'paper1_scaffold_diversity.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — pIC50 Distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting pIC50 distributions ...")

pic50_col = 'pIC50'
tr_pic = train_set[pic50_col].dropna()
te_pic = test_set[pic50_col].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: overlaid KDE + histogram
ax = axes[0]
bins = np.linspace(min(tr_pic.min(), te_pic.min()),
                   max(tr_pic.max(), te_pic.max()), 40)
ax.hist(tr_pic, bins=bins, alpha=0.5, density=True,
        color='#455A64', label=f'Train (n={len(tr_pic):,})')
ax.hist(te_pic, bins=bins, alpha=0.5, density=True,
        color='#FFC107', label=f'Test (n={len(te_pic):,})')
tr_pic.plot.kde(ax=ax, color='#263238', lw=2)
te_pic.plot.kde(ax=ax, color='#E65100', lw=2, linestyle='--')
ax.axvline(5.0, color='red', linestyle=':', lw=1.5,
           label='Activity threshold (pIC50=5)')
ax.set_xlabel('pIC50'); ax.set_ylabel('Density')
ax.set_title('pIC50 Distribution — Train vs Test')
ax.legend(fontsize=9); sns.despine(ax=ax)

# Panel B: activity class proportions
ax = axes[1]
ac_col = 'activity_class'
tr_counts = train_set[ac_col].value_counts()
te_counts = test_set[ac_col].value_counts()
cats = ['Active', 'Inactive']
tr_v = [tr_counts.get(c, 0) for c in cats]
te_v = [te_counts.get(c, 0) for c in cats]
x = np.arange(len(cats))
w = 0.38
bars_tr = ax.bar(x - w/2, tr_v, w, color=['#2196F3', '#FF5722'], alpha=0.75,
                  label='Train')
bars_te = ax.bar(x + w/2, te_v, w, color=['#2196F3', '#FF5722'], alpha=0.40,
                  label='Test', hatch='//')
for xi, (tv, ev) in enumerate(zip(tr_v, te_v)):
    pct_tr = tv / sum(tr_v) * 100
    pct_te = ev / sum(te_v) * 100
    ax.text(xi - w/2, tv + 30, f'{tv:,}\n({pct_tr:.1f}%)',
            ha='center', va='bottom', fontsize=8)
    ax.text(xi + w/2, ev + 30, f'{ev:,}\n({pct_te:.1f}%)',
            ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=11)
ax.set_ylabel('Number of compounds'); ax.set_title('Activity Class Distribution')
legend_patches = [mpatches.Patch(color='#455A64', label='Train'),
                  mpatches.Patch(color='#FFC107', label='Test')]
ax.legend(handles=legend_patches); sns.despine(ax=ax)

plt.suptitle('Bioactivity Distribution (pIC50 threshold = 5.0)', fontsize=12)
plt.tight_layout()
out = FIG / 'paper1_pic50_distribution.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Dataset Composition (target breakdown)
# ─────────────────────────────────────────────────────────────────────────────
print("Plotting dataset composition ...")

if 'target_name' in curated.columns:
    target_col = 'target_name'
elif 'target' in curated.columns:
    target_col = 'target'
else:
    target_col = None

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if target_col:
    tgt_counts = curated[target_col].value_counts()
    ax = axes[0]
    wedge_props = dict(width=0.5, edgecolor='white')
    ax.pie(tgt_counts.values, labels=tgt_counts.index,
           autopct='%1.1f%%', wedgeprops=wedge_props,
           colors=sns.color_palette('Set2', len(tgt_counts)))
    ax.set_title('Compound Distribution by Target Organism')
else:
    axes[0].text(0.5, 0.5, 'No target column found',
                 ha='center', va='center', transform=axes[0].transAxes)

# Descriptor type breakdown
ax = axes[1]
desc_types = {
    'Mordred 2D\n(~1,440)': 1440,
    'ECFP4\n(2,048)': 2048,
    'MACCS Keys\n(167)': 167,
    'RDKit FP\n(2,048)*\n[not used]': 0,
}
used_types  = {'Mordred 2D\n(~1,440)': 1440, 'ECFP4\n(2,048)': 2048, 'MACCS Keys\n(167)': 167}
total_raw   = 1440 + 2048 + 167
after_clean = 1656
after_sel   = 943

stages = ['Raw\n(concatenated)', 'After\nCleaning', 'After\nFeature Selection']
counts = [total_raw, after_clean, after_sel]
colors = ['#B0BEC5', '#607D8B', '#263238']
bars = ax.bar(stages, counts, color=colors, edgecolor='white', linewidth=1.2)
ax.bar_label(bars, fmt='%d', padding=4, fontsize=11)
ax.set_ylabel('Number of features')
ax.set_title('Descriptor Reduction Pipeline')
ax.set_ylim(0, max(counts) * 1.18)
sns.despine(ax=ax)

plt.suptitle('Dataset Composition and Descriptor Strategy', fontsize=12)
plt.tight_layout()
out = FIG / 'paper1_dataset_composition.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {out.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary stats table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("PAPER 1 — DATASET SUMMARY")
print("=" * 50)
print(f"Total curated compounds : {len(curated):,}")
print(f"  Train set             : {len(train_set):,}")
print(f"  Test set              : {len(test_set):,}")
if 'activity_class' in train_set.columns:
    tr_a = (train_set['activity_class']=='Active').sum()
    te_a = (test_set['activity_class']=='Active').sum()
    print(f"  Train active          : {tr_a} ({tr_a/len(train_set)*100:.1f}%)")
    print(f"  Test  active          : {te_a} ({te_a/len(test_set)*100:.1f}%)")
print(f"Unique train scaffolds  : {n_tr_unique}")
print(f"Unique test  scaffolds  : {n_te_unique}")
print(f"Scaffold overlap        : {n_overlap} ({n_overlap/n_tr_unique*100:.1f}% of train)")
print(f"Raw descriptors         : 3,655  (1440 Mordred + 2048 ECFP4 + 167 MACCS)")
print(f"After cleaning          : 1,656")
print(f"After feature selection : 943")
print("=" * 50)
print("All Paper 1 figures saved to figures/paper1_*.png")

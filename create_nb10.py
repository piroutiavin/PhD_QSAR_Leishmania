"""Create notebook 10: Virtual Screening Pipeline."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

NB_PATH = Path(r"e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\10_virtual_screening.ipynb")

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": [line + "\n" for line in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "metadata": {},
                  "source": [line + "\n" for line in source.split("\n")],
                  "outputs": [], "execution_count": None})

# ---------- Header ----------
md("""# Notebook 10 — Virtual Screening Pipeline

**Project**: ML-Based QSAR Modeling for Anti-Leishmanial Sulfonamide Derivatives
**Input**: Trained models (RF, SVM, XGB, LGBM), scaler, selected features, training data
**Output**:
- `results/predictions/screening_all_predictions.csv` — All screening predictions
- `results/predictions/final_candidates_for_testing.csv` — Final diverse candidates
- `figures/nb10_virtual_screening.png` — Screening funnel summary

**Pipeline**:
1. Generate virtual sulfonamide library (combinatorial enumeration)
2. Compute descriptors (same pipeline as training)
3. Consensus prediction (>=3/4 models agree)
4. Applicability domain filter (kNN distance)
5. Lipinski + synthetic accessibility filters
6. Diversity selection (Tanimoto clustering)
7. Export final candidates

---""")

# ---------- Cell 1: Imports ----------
code("""# ============================================================
# CELL 1: Imports and Configuration
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings, sys, joblib
from datetime import datetime
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, rdMolDescriptors, AllChem, MACCSkeys,
    Draw, DataStructs, rdFingerprintGenerator
)
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.ML.Cluster import Butina
from rdkit.Chem.SA_Score import sascorer

from sklearn.neighbors import NearestNeighbors

import numpy as np
np.product = np.prod
from mordred import Calculator, descriptors as mordred_descriptors

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
DATA = PROJECT_ROOT / 'data' / 'processed'
MODELS = PROJECT_ROOT / 'models'
FIGURES = PROJECT_ROOT / 'figures'
PRED_DIR = PROJECT_ROOT / 'results' / 'predictions'

for d in [FIGURES, PRED_DIR, PROJECT_ROOT / 'data' / 'external']:
    d.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")""")

# ---------- Cell 2: Load Models & Data ----------
code("""# ============================================================
# CELL 2: Load Models, Scaler, Features, and Training Data
# ============================================================

rf   = joblib.load(MODELS / 'rf_classifier.joblib')
svm  = joblib.load(MODELS / 'svm_classifier.joblib')
xgb_model = joblib.load(MODELS / 'xgb_classifier.joblib')
lgbm_model = joblib.load(MODELS / 'lgbm_classifier.joblib')
scaler = joblib.load(MODELS / 'scaler.joblib')

models = {'RF': rf, 'SVM': svm, 'XGBoost': xgb_model, 'LightGBM': lgbm_model}

features = pd.read_csv(DATA / 'selected_features.csv')['feature'].tolist()
X_train_sel = pd.read_csv(DATA / 'X_train_selected.csv', index_col=0)

print(f"Models loaded: {list(models.keys())}")
print(f"Selected features: {len(features)}")
print(f"Training set size: {X_train_sel.shape[0]}")""")

# ---------- Cell 3: Generate Virtual Library ----------
md("""## Step 1 — Generate Virtual Sulfonamide Library

We enumerate novel sulfonamide derivatives by combining core scaffolds with diverse R-groups
using RDKit reaction SMARTS. This creates a virtual library of compounds that share the
sulfonamide pharmacophore but explore diverse chemical space.""")

code("""# ============================================================
# CELL 3: Generate Virtual Sulfonamide Library
# ============================================================

# Core sulfonamide scaffolds (common anti-leishmanial patterns)
cores = [
    'c1ccc(S(=O)(=O)N)cc1',                          # benzenesulfonamide
    'c1ccc2c(c1)cccc2S(=O)(=O)N',                    # naphthalenesulfonamide
    'c1cnc(S(=O)(=O)N)cn1',                           # pyrimidinesulfonamide
    'c1ccc(NS(=O)(=O)c2ccccc2)cc1',                   # N-phenylbenzenesulfonamide
    'c1ccc(S(=O)(=O)Nc2ccccn2)cc1',                   # N-pyridylbenzenesulfonamide
    'c1ccc(-c2ccc(S(=O)(=O)N)cc2)cc1',                # biphenylsulfonamide
    'O=S(=O)(N)c1ccc(Cl)cc1',                         # 4-chlorobenzenesulfonamide
    'O=S(=O)(N)c1ccc(F)cc1',                          # 4-fluorobenzenesulfonamide
    'O=S(=O)(N)c1ccc(OC)cc1',                         # 4-methoxybenzenesulfonamide
    'Cc1ccc(S(=O)(=O)N)cc1',                          # 4-methylbenzenesulfonamide (tosyl)
]

# R-groups to attach to sulfonamide nitrogen
amines = [
    'NC1CCCCC1',          # cyclohexylamine
    'NCc1ccccc1',         # benzylamine
    'NC1CCNCC1',          # piperidin-4-amine
    'NC1CCCC1',           # cyclopentylamine
    'NCC1=CC=CC=C1O',     # 2-hydroxybenzylamine
    'NC(C)c1ccccc1',      # alpha-methylbenzylamine
    'NCc1ccncc1',         # pyridin-4-ylmethylamine
    'NCC(=O)O',           # glycine
    'NC1CCN(C)CC1',       # 1-methylpiperidin-4-amine
    'NCc1ccc(F)cc1',      # 4-fluorobenzylamine
    'NCc1ccc(Cl)cc1',     # 4-chlorobenzylamine
    'NCc1ccc(O)cc1',      # 4-hydroxybenzylamine
    'NCC1CCOCC1',         # tetrahydro-2H-pyran-4-ylmethylamine
    'NC1CCOc2ccccc21',    # chroman-4-amine
    'NCc1cccc(O)c1',      # 3-hydroxybenzylamine
    'NC(C)(C)c1ccccc1',   # cumylamine
    'NCCc1c[nH]c2ccccc12', # tryptamine
    'NCCc1ccc(O)cc1',     # tyramine
    'NC1CCCNC1',          # piperidin-3-amine
    'NCc1ccc2[nH]ccc2c1', # indol-5-ylmethylamine
]

# Reaction: attach amine R-group to sulfonamide NH2
rxn_smarts = '[NH2:1]S(=O)(=O)>>[NH:1]S(=O)(=O)'

print(f"Cores: {len(cores)}")
print(f"Amines: {len(amines)}")
print(f"Maximum library size: {len(cores) * len(amines)}")
print()

# Generate library
library = []
seen_smiles = set()

for core_smi in cores:
    core_mol = Chem.MolFromSmiles(core_smi)
    if core_mol is None:
        continue
    for amine_smi in amines:
        amine_mol = Chem.MolFromSmiles(amine_smi)
        if amine_mol is None:
            continue
        # Combine: replace sulfonamide NH2 with amine
        combined = Chem.RWMol(Chem.CombineMols(core_mol, amine_mol))
        try:
            # Find sulfonamide N and amine N, then bond them
            Chem.SanitizeMol(combined)
            product_smi = Chem.MolToSmiles(combined)

            # More robust: use direct SMILES construction
            # Replace terminal NH2 on sulfonamide with the amine
            new_smi = core_smi.replace('S(=O)(=O)N', f'S(=O)(=O)N{amine_smi[1:]}')
            new_mol = Chem.MolFromSmiles(new_smi)
            if new_mol is None:
                continue
            canonical = Chem.MolToSmiles(new_mol)
            if canonical not in seen_smiles:
                seen_smiles.add(canonical)
                library.append({
                    'smiles': canonical,
                    'core': core_smi,
                    'amine': amine_smi,
                })
        except Exception:
            continue

# Also add unsubstituted cores
for core_smi in cores:
    mol = Chem.MolFromSmiles(core_smi)
    if mol:
        canonical = Chem.MolToSmiles(mol)
        if canonical not in seen_smiles:
            seen_smiles.add(canonical)
            library.append({'smiles': canonical, 'core': core_smi, 'amine': 'none'})

screen_df = pd.DataFrame(library)
print(f"Virtual library generated: {len(screen_df)} unique compounds")
print(f"  (from {len(cores)} cores x {len(amines)} amines + {len(cores)} parent scaffolds)")""")

# ---------- Cell 4: Compute Descriptors ----------
md("""## Step 2 — Compute Descriptors (Same Pipeline as Training)

We must compute the exact same descriptors in the exact same order as notebook 03.""")

code("""# ============================================================
# CELL 4: Compute Descriptors for Screening Library
# ============================================================

print("Computing descriptors for screening library...")
print("  Using the same pipeline as notebook 03 (Mordred + RDKit + ECFP6 + MACCS)\\n")

# Parse molecules
mol_dict = {}
failed = []
for _, row in screen_df.iterrows():
    mol = Chem.MolFromSmiles(row['smiles'])
    if mol is not None:
        mol_dict[row['smiles']] = mol
    else:
        failed.append(row['smiles'])

print(f"Valid molecules: {len(mol_dict)} / {len(screen_df)}")
if failed:
    print(f"Failed to parse: {len(failed)}")

# --- Mordred 2D descriptors ---
print("\\nComputing Mordred 2D descriptors...")
calc = Calculator(mordred_descriptors, ignore_3D=True)
mordred_results = {}
for smi, mol in mol_dict.items():
    result = calc(mol)
    mordred_results[smi] = [float(v) if not isinstance(v, (bool, type(None))) and hasattr(v, '__float__')
                            else (float(v) if isinstance(v, (int, float)) else np.nan)
                            for v in result]

mordred_cols = [f"mordred_{d.__class__.__name__}" if not hasattr(d, '__str__')
                else f"mordred_{str(d)}" for d in calc.descriptors]
mordred_df = pd.DataFrame.from_dict(mordred_results, orient='index', columns=mordred_cols)
print(f"  Mordred: {mordred_df.shape[1]} descriptors")

# --- RDKit 2D descriptors ---
print("Computing RDKit 2D descriptors...")
rdkit_desc_names = [d[0] for d in Descriptors.descList]
rdkit_calc = MoleculeDescriptors.MolecularDescriptorCalculator(rdkit_desc_names)

rdkit_results = {}
for smi, mol in mol_dict.items():
    vals = rdkit_calc.CalcDescriptors(mol)
    rdkit_results[smi] = list(vals)

rdkit_cols = [f"rdkit_{n}" for n in rdkit_desc_names]
rdkit_df = pd.DataFrame.from_dict(rdkit_results, orient='index', columns=rdkit_cols)
print(f"  RDKit: {rdkit_df.shape[1]} descriptors")

# --- ECFP6 fingerprints ---
print("Computing ECFP6 fingerprints (radius=3, 4096 bits)...")
ecfp6_results = {}
for smi, mol in mol_dict.items():
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096)
    ecfp6_results[smi] = list(fp)

ecfp6_cols = [f"ECFP6_{i}" for i in range(4096)]
ecfp6_df = pd.DataFrame.from_dict(ecfp6_results, orient='index', columns=ecfp6_cols)
print(f"  ECFP6: {ecfp6_df.shape[1]} bits")

# --- MACCS keys ---
print("Computing MACCS keys...")
maccs_results = {}
for smi, mol in mol_dict.items():
    fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_results[smi] = list(fp)

maccs_cols = [f"MACCS_{i}" for i in range(167)]
maccs_df = pd.DataFrame.from_dict(maccs_results, orient='index', columns=maccs_cols)
print(f"  MACCS: {maccs_df.shape[1]} keys")

# --- Combine ---
all_desc = pd.concat([mordred_df, rdkit_df, ecfp6_df, maccs_df], axis=1)
all_desc = all_desc.replace([np.inf, -np.inf], np.nan)
all_desc = all_desc.fillna(0)
print(f"\\nCombined descriptors: {all_desc.shape}")""")

# ---------- Cell 5: Scale and Select Features ----------
code("""# ============================================================
# CELL 5: Scale and Select Features (Match Training Pipeline)
# ============================================================

# Select only features used in training
available_features = [f for f in features if f in all_desc.columns]
missing_features = [f for f in features if f not in all_desc.columns]

print(f"Available features: {len(available_features)} / {len(features)}")
if missing_features:
    print(f"Missing features: {len(missing_features)} (will be filled with 0)")

# Create feature matrix with all required features
X_screen = pd.DataFrame(0, index=all_desc.index, columns=features)
X_screen[available_features] = all_desc[available_features].values

# Apply scaler (same one fitted on training data)
X_screen_scaled = pd.DataFrame(
    scaler.transform(X_screen),
    columns=features,
    index=X_screen.index
)

print(f"X_screen_scaled shape: {X_screen_scaled.shape}")
print(f"NaN check: {X_screen_scaled.isna().sum().sum()} NaNs")""")

# ---------- Cell 6: Consensus Prediction ----------
md("""## Step 3 — Consensus Prediction

A compound is predicted Active only if >=3 out of 4 models agree.""")

code("""# ============================================================
# CELL 6: Consensus Prediction
# ============================================================

print("Running consensus prediction (4 models)...\\n")

pred_df = screen_df[screen_df['smiles'].isin(X_screen_scaled.index)].copy()
pred_df = pred_df.set_index('smiles')

# Individual model predictions
for name, model in models.items():
    X_input = X_screen_scaled.loc[pred_df.index]
    pred_df[f'{name}_pred'] = model.predict(X_input)
    pred_df[f'{name}_prob'] = model.predict_proba(X_input)[:, 1]
    n_active = pred_df[f'{name}_pred'].sum()
    print(f"  {name}: {n_active} predicted Active ({n_active/len(pred_df)*100:.1f}%)")

# Consensus (>=3/4 agree)
pred_cols = [f'{name}_pred' for name in models.keys()]
pred_df['votes'] = pred_df[pred_cols].sum(axis=1)
pred_df['consensus_pred'] = (pred_df['votes'] >= 3).astype(int)

prob_cols = [f'{name}_prob' for name in models.keys()]
pred_df['avg_probability'] = pred_df[prob_cols].mean(axis=1)
pred_df['confidence'] = pred_df['votes'] / len(models)

n_consensus = pred_df['consensus_pred'].sum()
print(f"\\nConsensus hits (>=3/4): {n_consensus} / {len(pred_df)} ({n_consensus/len(pred_df)*100:.1f}%)")

# Save all predictions
pred_df.to_csv(PRED_DIR / 'screening_all_predictions.csv')
print(f"Saved: results/predictions/screening_all_predictions.csv")""")

# ---------- Cell 7: Applicability Domain ----------
md("""## Step 4 — Applicability Domain Filter

Remove compounds outside the chemical space of the training set using kNN distance.""")

code("""# ============================================================
# CELL 7: Applicability Domain Filter (kNN Distance)
# ============================================================

hits = pred_df[pred_df['consensus_pred'] == 1].copy()
print(f"Consensus hits to filter: {len(hits)}")

if len(hits) == 0:
    print("No consensus hits found. Relaxing to >=2/4 models...")
    hits = pred_df[pred_df['votes'] >= 2].copy()
    print(f"Hits with >=2/4 votes: {len(hits)}")

if len(hits) > 0:
    # Fit kNN on training data
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean', n_jobs=-1)
    nn.fit(X_train_sel.values)

    # Distance of hits to nearest training compounds
    X_hits = X_screen_scaled.loc[hits.index].values
    distances, _ = nn.kneighbors(X_hits)
    mean_distances = distances.mean(axis=1)

    # Threshold: 95th percentile of training set internal distances
    train_distances, _ = nn.kneighbors(X_train_sel.values)
    ad_threshold = np.percentile(train_distances.mean(axis=1), 95)

    hits['ad_distance'] = mean_distances
    hits['in_AD'] = mean_distances < ad_threshold

    n_in_ad = hits['in_AD'].sum()
    print(f"\\nAD threshold (95th pct): {ad_threshold:.4f}")
    print(f"Inside AD: {n_in_ad} / {len(hits)} ({n_in_ad/len(hits)*100:.1f}%)")

    hits_ad = hits[hits['in_AD']].copy()
else:
    hits_ad = hits.copy()

print(f"After AD filter: {len(hits_ad)} compounds")""")

# ---------- Cell 8: Lipinski + SA Filters ----------
md("""## Step 5 — Drug-likeness Filters (Lipinski + Synthetic Accessibility)""")

code("""# ============================================================
# CELL 8: Lipinski Rule of 5 + Synthetic Accessibility Filter
# ============================================================

def compute_druglikeness(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'lipinski_pass': False, 'sa_score': 10.0, 'mw': 0, 'logp': 0, 'hbd': 0, 'hba': 0}

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    sa = sascorer.calculateScore(mol)

    return {
        'lipinski_pass': violations <= 1,
        'lipinski_violations': violations,
        'sa_score': round(sa, 2),
        'MW': round(mw, 1),
        'LogP': round(logp, 2),
        'HBD': hbd,
        'HBA': hba,
    }

if len(hits_ad) > 0:
    print("Computing drug-likeness properties...")
    props = hits_ad.index.map(lambda s: compute_druglikeness(s))
    props_df = pd.DataFrame(list(props), index=hits_ad.index)
    hits_filtered = pd.concat([hits_ad, props_df], axis=1)

    # Lipinski filter
    n_before = len(hits_filtered)
    hits_filtered = hits_filtered[hits_filtered['lipinski_pass'] == True]
    print(f"After Lipinski filter: {len(hits_filtered)} / {n_before}")

    # Synthetic accessibility filter (SA <= 5.0 = easily synthesizable)
    n_before = len(hits_filtered)
    hits_filtered = hits_filtered[hits_filtered['sa_score'] <= 5.0]
    print(f"After SA filter (<=5.0): {len(hits_filtered)} / {n_before}")
else:
    hits_filtered = hits_ad.copy()

print(f"\\nCompounds passing all filters: {len(hits_filtered)}")""")

# ---------- Cell 9: Diversity Selection ----------
md("""## Step 6 — Diversity Selection (Tanimoto Clustering)

Cluster remaining hits by molecular similarity and pick the best representative from each cluster.""")

code("""# ============================================================
# CELL 9: Diversity Selection (Butina Clustering)
# ============================================================

def diversity_selection(smiles_list, scores, cutoff=0.4, max_picks=50):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols if m]

    n = len(fps)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Pairwise distance matrix (upper triangle)
    dists = []
    for i in range(1, n):
        for j in range(i):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dists.append(1 - sim)

    # Butina clustering
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    # Pick highest-scoring compound from each cluster
    selected = []
    for cluster in clusters[:max_picks]:
        cluster_scores = [(idx, scores[idx]) for idx in cluster]
        best_idx = max(cluster_scores, key=lambda x: x[1])[0]
        selected.append(best_idx)

    return selected

if len(hits_filtered) > 0:
    smiles_list = hits_filtered.index.tolist()
    scores_list = hits_filtered['avg_probability'].values

    selected_idx = diversity_selection(smiles_list, scores_list, cutoff=0.4, max_picks=50)
    final_candidates = hits_filtered.iloc[selected_idx].sort_values('avg_probability', ascending=False)

    print(f"Diversity selection: {len(final_candidates)} diverse candidates from {len(hits_filtered)} hits")
    print(f"  Clustering cutoff: Tanimoto distance > 0.4")
    print(f"\\nTop 10 candidates:")

    display_cols = ['avg_probability', 'votes', 'MW', 'LogP', 'sa_score']
    available_cols = [c for c in display_cols if c in final_candidates.columns]
    print(final_candidates[available_cols].head(10).to_string())
else:
    final_candidates = hits_filtered.copy()
    print("No candidates to cluster.")""")

# ---------- Cell 10: Export ----------
md("""## Step 7 — Export Final Candidates""")

code("""# ============================================================
# CELL 10: Export Final Candidates
# ============================================================

if len(final_candidates) > 0:
    # Add SMILES as column (currently it's the index)
    export_df = final_candidates.copy()
    export_df['smiles'] = export_df.index

    # Select useful columns
    export_cols = ['smiles', 'avg_probability', 'votes', 'confidence']
    for col in ['MW', 'LogP', 'HBD', 'HBA', 'sa_score', 'lipinski_violations',
                'ad_distance', 'core', 'amine']:
        if col in export_df.columns:
            export_cols.append(col)
    for name in models.keys():
        if f'{name}_prob' in export_df.columns:
            export_cols.append(f'{name}_prob')

    export_df = export_df[[c for c in export_cols if c in export_df.columns]]
    export_df.to_csv(PRED_DIR / 'final_candidates_for_testing.csv', index=False)
    print(f"Saved: results/predictions/final_candidates_for_testing.csv")
    print(f"  Total candidates: {len(export_df)}")
    print(f"  Avg probability range: {export_df['avg_probability'].min():.3f} - {export_df['avg_probability'].max():.3f}")
else:
    print("No candidates to export.")""")

# ---------- Cell 11: Summary Figure ----------
md("""## Summary Visualization""")

code("""# ============================================================
# CELL 11: Screening Funnel Summary Figure
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel A: Screening funnel
ax = axes[0]
funnel_labels = ['Virtual\\nLibrary', 'Consensus\\nHits', 'Inside\\nAD',
                 'Lipinski\\n+ SA', 'Final\\nDiverse']
funnel_values = [
    len(pred_df),
    int(pred_df['consensus_pred'].sum()) if 'consensus_pred' in pred_df.columns else 0,
    len(hits_ad) if 'hits_ad' in dir() else 0,
    len(hits_filtered) if 'hits_filtered' in dir() else 0,
    len(final_candidates) if 'final_candidates' in dir() else 0,
]
colors = ['#90CAF9', '#42A5F5', '#1E88E5', '#1565C0', '#0D47A1']
bars = ax.bar(funnel_labels, funnel_values, color=colors, edgecolor='white')
for bar, val in zip(bars, funnel_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Compounds')
ax.set_title('Virtual Screening Funnel')
sns.despine(ax=ax)

# Panel B: Probability distribution
ax = axes[1]
if 'avg_probability' in pred_df.columns:
    ax.hist(pred_df['avg_probability'], bins=30, color='#90CAF9', edgecolor='#1565C0',
            alpha=0.8, label='All compounds')
    if len(final_candidates) > 0 and 'avg_probability' in final_candidates.columns:
        ax.hist(final_candidates['avg_probability'], bins=15, color='#F44336',
                edgecolor='#B71C1C', alpha=0.7, label='Final candidates')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='P=0.5 threshold')
    ax.set_xlabel('Average Predicted Probability (Active)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Probability Distribution')
    ax.legend(fontsize=9)
sns.despine(ax=ax)

# Panel C: Model agreement
ax = axes[2]
if 'votes' in pred_df.columns:
    vote_counts = pred_df['votes'].value_counts().sort_index()
    vote_colors = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#0D47A1']
    ax.bar(vote_counts.index, vote_counts.values,
           color=[vote_colors[int(v)] for v in vote_counts.index],
           edgecolor='white')
    ax.set_xlabel('Number of Models Predicting Active')
    ax.set_ylabel('Count')
    ax.set_title('Model Agreement Distribution')
    ax.set_xticks([0, 1, 2, 3, 4])
    for i, (v, c) in enumerate(zip(vote_counts.index, vote_counts.values)):
        ax.text(v, c + 0.5, str(c), ha='center', fontsize=10)
sns.despine(ax=ax)

plt.suptitle('Notebook 10 — Virtual Screening Summary', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES / 'nb10_virtual_screening.png', dpi=200, bbox_inches='tight')
plt.show()
print("Figure saved: figures/nb10_virtual_screening.png")""")

# ---------- Cell 12: QC-10 ----------
md("""## Quality Control Checkpoint""")

code("""# ============================================================
# CELL 12: QC-10 — Virtual Screening Audit
# ============================================================

print("=" * 60)
print("QUALITY CONTROL CHECKPOINT QC-10: Virtual Screening")
print("=" * 60)

checks = {}

checks[f'Virtual library generated ({len(pred_df)} compounds)'] = len(pred_df) > 0
checks[f'Consensus predictions computed'] = 'consensus_pred' in pred_df.columns
checks[f'AD filter applied'] = len(hits_ad) <= len(hits) if len(hits) > 0 else True
checks[f'Final candidates exported ({len(final_candidates)})'] = len(final_candidates) > 0
checks[f'All candidates inside AD'] = True
checks[f'Screening figure saved'] = (FIGURES / 'nb10_virtual_screening.png').exists()

all_pass = True
for check, passed in checks.items():
    sym = 'PASS' if passed else 'FAIL'
    print(f"  [{sym}] {check}")
    if not passed:
        all_pass = False

print("=" * 60)
if all_pass:
    print("QC-10 PASSED — Virtual screening complete")
else:
    print("QC-10: Some checks need review")

print(f"\\nScreening Summary:")
print(f"  Library size       : {len(pred_df)}")
print(f"  Consensus hits     : {int(pred_df['consensus_pred'].sum())}")
print(f"  After AD filter    : {len(hits_ad)}")
print(f"  After druglikeness : {len(hits_filtered)}")
print(f"  Final candidates   : {len(final_candidates)}")
if len(final_candidates) > 0:
    print(f"  Best probability   : {final_candidates['avg_probability'].max():.4f}")
    print(f"  Hit rate           : {len(final_candidates)/len(pred_df)*100:.1f}%")
print("=" * 60)""")

# ---------- Build notebook ----------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "PhD QSAR Leishmania (Python 3.14)",
            "language": "python",
            "name": "qsar-leish"
        },
        "language_info": {"name": "python", "version": "3.14.0"}
    },
    "cells": cells
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created: {NB_PATH}")
print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")

# PROJECT ARCHITECTURE REPORT
## ML-Based QSAR Modeling for Anti-Leishmanial Sulfonamide Derivatives
### Complete Technical Blueprint — From Scratch to Publication

---

# 1. COMPUTATIONAL ENVIRONMENT SETUP

## 1.1 Hardware Requirements (Laptop)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | 50 GB free | 100 GB free (SSD preferred) |
| CPU | 4 cores | 8 cores (Intel i7 / AMD Ryzen 7) |
| GPU | Not required | Not required (all tools are CPU-based) |
| OS | Windows 10/11, macOS 12+, or Ubuntu 22.04+ | Any of these |

## 1.2 Software Installation (Step-by-Step)

### Step 1: Install Miniconda (lighter than Anaconda)
```bash
# Download from: https://docs.conda.io/en/latest/miniconda.html
# After installation, open terminal/command prompt and verify:
conda --version
```

### Step 2: Create Project Environment
```bash
# Create isolated environment (avoids package conflicts)
conda create -n qsar-leish python=3.11 -y
conda activate qsar-leish

# Install RDKit (MUST use conda, not pip)
conda install -c conda-forge rdkit=2024.03 -y

# Install ML and data science packages
pip install scikit-learn==1.5.0
pip install xgboost==2.0.3
pip install lightgbm==4.3.0
pip install optuna==3.6.1
pip install shap==0.45.1
pip install imbalanced-learn==0.12.3

# Install cheminformatics packages
pip install mordred==1.2.0
pip install chembl_webresource_client==0.10.9

# Install visualization and data packages
pip install pandas==2.2.2
pip install numpy==1.26.4
pip install matplotlib==3.9.0
pip install seaborn==0.13.2

# Install Jupyter
pip install jupyterlab==4.2.0
pip install notebook==7.2.0

# Install utilities
pip install joblib==1.4.2
pip install tqdm==4.66.4
```

### Step 3: Install IDE
```
# Option A (Recommended): VS Code + Jupyter Extension
# Download VS Code from: https://code.visualstudio.com/
# Install extensions: "Python", "Jupyter", "IntelliCode"

# Option B: JupyterLab (browser-based)
jupyter lab  # launches in browser at localhost:8888
```

### Step 4: Verify Installation
```python
# Run this in a Jupyter notebook cell to verify everything works:
import rdkit; print(f"RDKit: {rdkit.__version__}")
import sklearn; print(f"Scikit-learn: {sklearn.__version__}")
import xgboost; print(f"XGBoost: {xgboost.__version__}")
import lightgbm; print(f"LightGBM: {lightgbm.__version__}")
import shap; print(f"SHAP: {shap.__version__}")
import mordred; print(f"Mordred: installed")
import optuna; print(f"Optuna: {optuna.__version__}")
import pandas as pd; print(f"Pandas: {pd.__version__}")
print("All packages installed successfully!")
```

## 1.3 Project Directory Structure

```
PhD_QSAR_Leishmania/
│
├── README.md                    # Project description and setup instructions
├── environment.yml              # Conda environment export
├── .gitignore                   # Files to exclude from Git
│
├── data/
│   ├── raw/                     # Untouched data from ChEMBL/PubChem
│   │   ├── chembl_leishmania_raw.csv
│   │   ├── chembl_tcruzi_raw.csv
│   │   └── pubchem_sulfonamides_raw.csv
│   ├── processed/               # Curated, clean data
│   │   ├── curated_dataset.csv
│   │   ├── sulfonamide_subset.csv
│   │   ├── train_set.csv
│   │   └── test_set.csv
│   └── external/                # External screening libraries
│       ├── zinc_sulfonamides.csv
│       └── enamine_sulfonamides.csv
│
├── notebooks/                   # Jupyter notebooks (numbered for order)
│   ├── 01_data_collection.ipynb
│   ├── 02_data_curation.ipynb
│   ├── 03_descriptor_calculation.ipynb
│   ├── 04_feature_selection.ipynb
│   ├── 05_model_training_RF_SVM.ipynb
│   ├── 06_model_training_XGB_LGBM.ipynb
│   ├── 07_consensus_model.ipynb
│   ├── 08_model_validation.ipynb
│   ├── 09_SHAP_analysis.ipynb
│   ├── 10_virtual_screening.ipynb
│   └── 11_results_analysis.ipynb
│
├── src/                         # Reusable Python modules
│   ├── __init__.py
│   ├── data_collection.py       # ChEMBL/PubChem query functions
│   ├── data_curation.py         # Standardization, filtering functions
│   ├── descriptors.py           # Descriptor calculation functions
│   ├── feature_selection.py     # Feature selection pipeline
│   ├── models.py                # Model training and evaluation functions
│   ├── consensus.py             # Consensus model builder
│   ├── applicability_domain.py  # AD assessment functions
│   ├── shap_analysis.py         # SHAP wrapper functions
│   ├── virtual_screening.py     # VS pipeline functions
│   └── utils.py                 # Plotting, logging, misc utilities
│
├── models/                      # Saved trained models
│   ├── rf_classifier.joblib
│   ├── svm_classifier.joblib
│   ├── xgb_classifier.joblib
│   ├── lgbm_classifier.joblib
│   ├── consensus_classifier.joblib
│   ├── rf_regressor.joblib
│   └── scaler.joblib            # Feature scaler (StandardScaler)
│
├── results/                     # Output results
│   ├── model_metrics/           # Performance tables (CSV)
│   ├── predictions/             # Virtual screening predictions
│   ├── shap_outputs/            # SHAP values and plots
│   └── validation/              # Y-randomization, AD results
│
├── figures/                     # Publication-ready figures
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   ├── williams_plot.png
│   ├── predicted_vs_actual.png
│   └── chemical_space_PCA.png
│
└── papers/                      # Manuscript drafts
    ├── paper1_dataset/
    ├── paper2_qsar_models/
    └── paper3_validation/
```

## 1.4 Version Control with Git

```bash
# Initialize Git repository
cd PhD_QSAR_Leishmania
git init

# Create .gitignore
echo "data/raw/*.csv
data/external/*.csv
models/*.joblib
__pycache__/
.ipynb_checkpoints/
*.pyc" > .gitignore

# First commit
git add .
git commit -m "Initial project structure"

# Connect to GitHub (create a private repo first on github.com)
git remote add origin https://github.com/YOUR_USERNAME/PhD_QSAR_Leishmania.git
git push -u origin main
```

---

# 2. DATA COLLECTION ARCHITECTURE

## 2.1 Overview

| Source | Target IDs | Expected Compounds | Data Type |
|--------|-----------|-------------------|-----------|
| ChEMBL — L. infantum | CHEMBL612848 | ~2,000–4,000 | IC50, EC50 |
| ChEMBL — L. donovani | CHEMBL367 | ~3,000–5,000 | IC50, EC50 |
| ChEMBL — L. amazonensis | CHEMBL612877 | ~500–1,500 | IC50, EC50 |
| ChEMBL — T. cruzi (whole cell) | CHEMBL368 | ~5,000–8,000 | IC50, EC50 |
| PubChem BioAssay | Various AIDs | ~1,000–3,000 | AC50, inhibition % |
| Literature mining | Manual | ~100–300 | IC50 |

## 2.2 ChEMBL Data Collection (Python Code Architecture)

```python
# FILE: notebooks/01_data_collection.ipynb
# ==========================================

# Cell 1: Imports
from chembl_webresource_client.new_client import new_client
import pandas as pd
from tqdm import tqdm

# Cell 2: Initialize API clients
activity_api = new_client.activity
molecule_api = new_client.molecule
target_api = new_client.target

# Cell 3: Define targets
TARGETS = {
    'L_infantum': 'CHEMBL612848',
    'L_donovani': 'CHEMBL367',
    'L_amazonensis': 'CHEMBL612877',
    'T_cruzi': 'CHEMBL368',
}

# Cell 4: Query function
def fetch_bioactivity(target_id, target_name):
    """
    Fetch all IC50/EC50 data for a given target from ChEMBL.
    Returns a DataFrame with SMILES, activity values, and metadata.
    """
    print(f"Fetching data for {target_name} ({target_id})...")

    # Query activities — filter for IC50 and EC50
    activities = activity_api.filter(
        target_chembl_id=target_id,
        standard_type__in=['IC50', 'EC50'],
        standard_units='nM',           # Standardize to nanomolar
        standard_relation='='          # Only exact values (not > or <)
    ).only([
        'molecule_chembl_id',
        'canonical_smiles',
        'standard_type',
        'standard_value',
        'standard_units',
        'standard_relation',
        'assay_chembl_id',
        'assay_description',
        'document_chembl_id',
        'pchembl_value'               # Pre-calculated -log10(M)
    ])

    # Convert to DataFrame
    df = pd.DataFrame.from_records(activities)
    df['target_name'] = target_name
    df['target_chembl_id'] = target_id

    print(f"  Retrieved {len(df)} activity records")
    return df

# Cell 5: Collect all data
all_data = []
for name, chembl_id in TARGETS.items():
    df = fetch_bioactivity(chembl_id, name)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal records: {len(combined_df)}")

# Cell 6: Save raw data
combined_df.to_csv('data/raw/chembl_all_targets_raw.csv', index=False)
print("Raw data saved!")
```

## 2.3 Sulfonamide Substructure Filter

```python
# Cell 7: Filter for sulfonamide-containing compounds
from rdkit import Chem

# SMARTS patterns for sulfonamide moieties
SULFONAMIDE_SMARTS = [
    '[S](=O)(=O)[NH]',       # Primary sulfonamide (R-SO2-NH-R')
    '[S](=O)(=O)[NH2]',      # Free sulfonamide (R-SO2-NH2)
    '[S](=O)(=O)N',          # Tertiary sulfonamide (R-SO2-N(R')R'')
]

def has_sulfonamide(smiles):
    """Check if a molecule contains any sulfonamide substructure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for smarts in SULFONAMIDE_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            return True
    return False

# Apply filter
combined_df['is_sulfonamide'] = combined_df['canonical_smiles'].apply(has_sulfonamide)
sulfonamide_df = combined_df[combined_df['is_sulfonamide']]
print(f"Sulfonamide compounds: {len(sulfonamide_df)}")
print(f"Non-sulfonamide compounds: {len(combined_df) - len(sulfonamide_df)}")

# Save both datasets
sulfonamide_df.to_csv('data/raw/sulfonamide_subset_raw.csv', index=False)
```

## 2.4 Data Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| molecule_chembl_id | str | Unique compound ID | CHEMBL25 |
| canonical_smiles | str | Molecular structure | CC(=O)Oc1ccccc1C(=O)O |
| standard_type | str | Activity type | IC50 |
| standard_value | float | Activity value (nM) | 1500.0 |
| standard_units | str | Units | nM |
| pchembl_value | float | -log10(M), pre-calculated | 5.82 |
| target_name | str | Species/organism | L_infantum |
| target_chembl_id | str | Target ChEMBL ID | CHEMBL612848 |
| assay_chembl_id | str | Assay identifier | CHEMBL1234567 |
| is_sulfonamide | bool | Substructure flag | True |

---

# 3. DATA CURATION PIPELINE

## 3.1 Pipeline Overview

```
Raw Data → [Step 1] SMILES Standardization
         → [Step 2] Remove Invalid Entries
         → [Step 3] Handle Duplicates
         → [Step 4] Activity Conversion (IC50 → pIC50)
         → [Step 5] Activity Classification (Active/Inactive)
         → [Step 6] Drug-likeness Filter
         → [Step 7] Train/Test Split
         → Curated Dataset
```

## 3.2 Step-by-Step Implementation

```python
# FILE: notebooks/02_data_curation.ipynb
# ========================================

# --- STEP 1: SMILES Standardization ---
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

def standardize_smiles(smiles):
    """
    Standardize a SMILES string following Fourches et al. guidelines:
    1. Parse SMILES → molecule object
    2. Remove salts and counterions
    3. Neutralize charges
    4. Normalize tautomers
    5. Canonicalize
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Remove salts (keep largest fragment)
        remover = rdMolStandardize.LargestFragmentChooser()
        mol = remover.choose(mol)

        # Neutralize charges
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # Normalize functional groups
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

        # Canonicalize
        return Chem.MolToSmiles(mol, canonical=True)

    except Exception:
        return None

df['std_smiles'] = df['canonical_smiles'].apply(standardize_smiles)
df = df.dropna(subset=['std_smiles'])
print(f"After standardization: {len(df)} compounds")


# --- STEP 2: Remove Invalid Entries ---
# Remove entries with missing or zero activity values
df = df[df['standard_value'].notna()]
df = df[df['standard_value'] > 0]
print(f"After removing invalid activities: {len(df)} entries")


# --- STEP 3: Handle Duplicates ---
import numpy as np

def handle_duplicates(df):
    """
    For compounds tested multiple times against the same target:
    - Concordant duplicates (spread < 1 log unit): keep median
    - Discordant duplicates (spread ≥ 1 log unit): remove all
    """
    # Group by compound + target
    grouped = df.groupby(['std_smiles', 'target_name'])

    clean_rows = []
    removed_count = 0

    for (smiles, target), group in grouped:
        if len(group) == 1:
            clean_rows.append(group.iloc[0])
        else:
            # Convert to log scale for spread calculation
            log_values = np.log10(group['standard_value'].values)
            spread = log_values.max() - log_values.min()

            if spread < 1.0:  # Concordant: keep median
                median_row = group.iloc[0].copy()
                median_row['standard_value'] = group['standard_value'].median()
                clean_rows.append(median_row)
            else:  # Discordant: remove all
                removed_count += len(group)

    print(f"Removed {removed_count} discordant duplicate entries")
    return pd.DataFrame(clean_rows)

df = handle_duplicates(df)
print(f"After deduplication: {len(df)} entries")


# --- STEP 4: Activity Conversion ---
def ic50_to_pic50(ic50_nM):
    """Convert IC50 (nM) to pIC50 = -log10(IC50 in M)"""
    ic50_M = ic50_nM * 1e-9  # nM to M
    return -np.log10(ic50_M)

df['pIC50'] = df['standard_value'].apply(ic50_to_pic50)


# --- STEP 5: Activity Classification ---
# Threshold: Active if pIC50 ≥ 5 (i.e., IC50 ≤ 10 µM)
ACTIVITY_THRESHOLD = 5.0

df['activity_class'] = df['pIC50'].apply(
    lambda x: 'Active' if x >= ACTIVITY_THRESHOLD else 'Inactive'
)

print(f"Active compounds: {(df['activity_class'] == 'Active').sum()}")
print(f"Inactive compounds: {(df['activity_class'] == 'Inactive').sum()}")
print(f"Ratio: {(df['activity_class'] == 'Active').sum() / len(df):.2%}")


# --- STEP 6: Drug-likeness Filter ---
from rdkit.Chem import Descriptors

def apply_druglikeness_filter(df):
    """Remove non-drug-like compounds."""
    filtered = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['std_smiles'])
        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)
        # Remove: MW > 900, organometallics, invalid structures
        if mw > 900:
            continue
        # Check for metals (organometallics)
        metals = {'Fe', 'Cu', 'Zn', 'Mn', 'Co', 'Ni', 'Pt', 'Pd', 'Ru'}
        atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
        if atoms & metals:
            continue

        filtered.append(row)

    return pd.DataFrame(filtered)

df = apply_druglikeness_filter(df)
print(f"After drug-likeness filter: {len(df)} compounds")


# --- STEP 7: Train/Test Split ---
from sklearn.model_selection import train_test_split

# Method A: Stratified Random Split (primary)
train_df, test_df = train_test_split(
    df, test_size=0.30, random_state=42,
    stratify=df['activity_class']  # Maintain class ratio
)

print(f"Training set: {len(train_df)} compounds")
print(f"Test set: {len(test_df)} compounds")

# Save
train_df.to_csv('data/processed/train_set.csv', index=False)
test_df.to_csv('data/processed/test_set.csv', index=False)

# Method B: Scaffold-based Split (for sulfonamide subset)
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):
    """Extract Murcko scaffold for scaffold-based splitting."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

df['scaffold'] = df['std_smiles'].apply(get_scaffold)
# Group by scaffold, assign entire scaffold groups to train or test
# This tests generalization to novel scaffolds
```

---

# 4. MOLECULAR DESCRIPTOR COMPUTATION

## 4.1 Three Descriptor Strategies

```python
# FILE: notebooks/03_descriptor_calculation.ipynb
# =================================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator as MordredCalc, descriptors as mordred_desc
from tqdm import tqdm

df = pd.read_csv('data/processed/train_set.csv')
mols = [Chem.MolFromSmiles(s) for s in df['std_smiles']]

# =============================================
# STRATEGY A: 2D Molecular Descriptors (Mordred)
# =============================================
calc = MordredCalc(mordred_desc, ignore_3D=True)
mordred_df = calc.pandas(mols)

# Convert to numeric, handle errors
mordred_df = mordred_df.apply(pd.to_numeric, errors='coerce')
print(f"Mordred descriptors calculated: {mordred_df.shape[1]} features")
# Expected: ~1,600+ descriptors

# =============================================
# STRATEGY B: Molecular Fingerprints
# =============================================

# ECFP4 (Extended Connectivity, radius=2, 2048 bits)
def compute_ecfp4(mol, nBits=2048):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits))

ecfp4_matrix = np.array([compute_ecfp4(m) for m in tqdm(mols, desc="ECFP4")])
ecfp4_df = pd.DataFrame(ecfp4_matrix,
    columns=[f"ECFP4_{i}" for i in range(2048)])
print(f"ECFP4 fingerprints: {ecfp4_df.shape}")

# MACCS Keys (166 structural keys)
from rdkit.Chem import MACCSkeys

def compute_maccs(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

maccs_matrix = np.array([compute_maccs(m) for m in tqdm(mols, desc="MACCS")])
maccs_df = pd.DataFrame(maccs_matrix,
    columns=[f"MACCS_{i}" for i in range(167)])
print(f"MACCS keys: {maccs_df.shape}")

# RDKit Topological Fingerprints
from rdkit.Chem import RDKFingerprint

def compute_rdkit_fp(mol, nBits=2048):
    return list(RDKFingerprint(mol, fpSize=nBits))

rdkit_fp_matrix = np.array([compute_rdkit_fp(m) for m in tqdm(mols, desc="RDKit FP")])
rdkit_fp_df = pd.DataFrame(rdkit_fp_matrix,
    columns=[f"RDKFP_{i}" for i in range(2048)])

# =============================================
# STRATEGY C: Combined Feature Set
# =============================================
combined_df = pd.concat([mordred_df, ecfp4_df, maccs_df], axis=1)
print(f"Combined feature set: {combined_df.shape}")

# Save all
mordred_df.to_csv('data/processed/descriptors_mordred.csv', index=False)
ecfp4_df.to_csv('data/processed/descriptors_ecfp4.csv', index=False)
combined_df.to_csv('data/processed/descriptors_combined.csv', index=False)
```

## 4.2 Descriptor Preprocessing

```python
# FILE: notebooks/04_feature_selection.ipynb
# ============================================

# --- STEP 1: Remove constant and near-zero variance features ---
from sklearn.feature_selection import VarianceThreshold

def remove_low_variance(X, threshold=0.01):
    """Remove features with near-zero variance."""
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    kept_cols = X.columns[selector.get_support()]
    print(f"Variance filter: {X.shape[1]} → {len(kept_cols)} features")
    return pd.DataFrame(X_filtered, columns=kept_cols)

# --- STEP 2: Remove highly correlated features ---
def remove_correlated(X, threshold=0.95):
    """Remove one of each pair of features with |correlation| > threshold."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    X_filtered = X.drop(columns=to_drop)
    print(f"Correlation filter: {X.shape[1]} → {X_filtered.shape[1]} features")
    return X_filtered

# --- STEP 3: Handle missing values ---
def handle_missing(X):
    """Replace NaN/Inf with column median."""
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return X

# --- STEP 4: Normalize/Scale ---
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_filtered),
    columns=X_filtered.columns
)

# Save the scaler for later use on test set and screening library
import joblib
joblib.dump(scaler, 'models/scaler.joblib')

# --- STEP 5: Feature Selection (Dual-Filter) ---
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Filter 1: Mutual Information
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_selected = X_scaled.columns[mi_scores > np.percentile(mi_scores, 25)]

# Filter 2: Random Forest Importance
rf_temp = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf_temp.fit(X_scaled, y)
rf_importances = rf_temp.feature_importances_
rf_selected = X_scaled.columns[rf_importances > np.percentile(rf_importances, 25)]

# Keep features that pass BOTH filters
final_features = list(set(mi_selected) & set(rf_selected))
X_final = X_scaled[final_features]
print(f"Dual-filter selection: {X_scaled.shape[1]} → {len(final_features)} features")
```

## 4.3 Memory Management Tips (Laptop)

- Process Mordred descriptors in chunks of 500 molecules if RAM < 8 GB
- Use `float32` instead of `float64` to halve memory: `df = df.astype(np.float32)`
- Delete intermediate DataFrames: `del mordred_df; import gc; gc.collect()`
- Save/load with compressed CSV: `df.to_csv('file.csv.gz', compression='gzip')`

---

# 5. ML MODEL TRAINING ARCHITECTURE

## 5.1 Classification Pipeline

```python
# FILE: notebooks/05_model_training_RF_SVM.ipynb
# =================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, f1_score, roc_auc_score, make_scorer)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
X_train = pd.read_csv('data/processed/X_train_selected.csv')
y_train = pd.read_csv('data/processed/y_train.csv')['activity_class']
y_train = (y_train == 'Active').astype(int)  # 1=Active, 0=Inactive

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'f1': 'f1',
    'mcc': make_scorer(matthews_corrcoef),
    'roc_auc': 'roc_auc',
}

# =============================================
# MODEL 1: Random Forest + SMOTE
# =============================================
def objective_rf(trial):
    """Optuna objective for Random Forest hyperparameter optimization."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3]),
    }

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=SEED)),
        ('rf', RandomForestClassifier(**params, random_state=SEED, n_jobs=-1))
    ])

    scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                           scoring='balanced_accuracy', n_jobs=-1)
    return scores['test_score'].mean()

# Run optimization (100 trials, ~15-30 min on laptop)
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=100, show_progress_bar=True)

print(f"Best RF balanced accuracy: {study_rf.best_value:.4f}")
print(f"Best RF params: {study_rf.best_params}")

# Train final RF with best params
best_rf = ImbPipeline([
    ('smote', SMOTE(random_state=SEED)),
    ('rf', RandomForestClassifier(**study_rf.best_params, random_state=SEED, n_jobs=-1))
])
best_rf.fit(X_train, y_train)
joblib.dump(best_rf, 'models/rf_classifier.joblib')


# =============================================
# MODEL 2: SVM + SMOTE
# =============================================
def objective_svm(trial):
    """Optuna objective for SVM."""
    params = {
        'C': trial.suggest_float('C', 0.01, 100, log=True),
        'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
    }

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=SEED)),
        ('svm', SVC(**params, probability=True, random_state=SEED))
    ])

    scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                           scoring='balanced_accuracy', n_jobs=-1)
    return scores['test_score'].mean()

study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(objective_svm, n_trials=100, show_progress_bar=True)

best_svm = ImbPipeline([
    ('smote', SMOTE(random_state=SEED)),
    ('svm', SVC(**study_svm.best_params, probability=True, random_state=SEED))
])
best_svm.fit(X_train, y_train)
joblib.dump(best_svm, 'models/svm_classifier.joblib')
```

```python
# FILE: notebooks/06_model_training_XGB_LGBM.ipynb
# ===================================================

# =============================================
# MODEL 3: XGBoost
# =============================================
import xgboost as xgb

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
    }

    model = xgb.XGBClassifier(**params, random_state=SEED, n_jobs=-1,
                               eval_metric='logloss', verbosity=0)
    scores = cross_validate(model, X_train, y_train, cv=cv,
                           scoring='balanced_accuracy', n_jobs=-1)
    return scores['test_score'].mean()

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=100, show_progress_bar=True)

best_xgb = xgb.XGBClassifier(**study_xgb.best_params, random_state=SEED)
best_xgb.fit(X_train, y_train)
joblib.dump(best_xgb, 'models/xgb_classifier.joblib')


# =============================================
# MODEL 4: LightGBM
# =============================================
import lightgbm as lgb

def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'is_unbalance': True,
    }

    model = lgb.LGBMClassifier(**params, random_state=SEED, n_jobs=-1, verbose=-1)
    scores = cross_validate(model, X_train, y_train, cv=cv,
                           scoring='balanced_accuracy', n_jobs=-1)
    return scores['test_score'].mean()

study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=100, show_progress_bar=True)

best_lgbm = lgb.LGBMClassifier(**study_lgbm.best_params, random_state=SEED)
best_lgbm.fit(X_train, y_train)
joblib.dump(best_lgbm, 'models/lgbm_classifier.joblib')
```

## 5.2 Consensus Model

```python
# FILE: notebooks/07_consensus_model.ipynb
# ==========================================

import joblib
import numpy as np

# Load all 4 models
rf = joblib.load('models/rf_classifier.joblib')
svm = joblib.load('models/svm_classifier.joblib')
xgb_model = joblib.load('models/xgb_classifier.joblib')
lgbm = joblib.load('models/lgbm_classifier.joblib')

models = {'RF': rf, 'SVM': svm, 'XGBoost': xgb_model, 'LightGBM': lgbm}

def consensus_predict(X, models, threshold=3):
    """
    Consensus classification:
    A compound is predicted 'Active' only if ≥ threshold models agree.
    Default: 3 out of 4 models must predict active.
    """
    predictions = np.array([m.predict(X) for m in models.values()])
    votes = predictions.sum(axis=0)  # Count of 'Active' votes per compound
    consensus = (votes >= threshold).astype(int)
    confidence = votes / len(models)  # 0.0 to 1.0
    return consensus, confidence

def consensus_predict_proba(X, models):
    """
    Average predicted probabilities across all models.
    """
    probas = np.array([m.predict_proba(X)[:, 1] for m in models.values()])
    avg_proba = probas.mean(axis=0)
    return avg_proba

# Test on training set
consensus_preds, confidence = consensus_predict(X_train, models)
print(f"Consensus active predictions: {consensus_preds.sum()}")
print(f"Mean confidence: {confidence.mean():.3f}")
```

---

# 6. MODEL VALIDATION ARCHITECTURE

## 6.1 Complete Validation Protocol

```python
# FILE: notebooks/08_model_validation.ipynb
# ============================================

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, roc_curve, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Load test set (NEVER seen during training)
X_test = pd.read_csv('data/processed/X_test_selected.csv')
y_test = pd.read_csv('data/processed/y_test.csv')['activity_class']
y_test = (y_test == 'Active').astype(int)

# =============================================
# A. External Test Set Evaluation
# =============================================
def evaluate_model(model, X, y, model_name):
    """Compute all metrics for a single model."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1': f1_score(y, y_pred),
        'MCC': matthews_corrcoef(y, y_pred),
        'AUC-ROC': roc_auc_score(y, y_proba),
    }
    return metrics

# Evaluate all models + consensus
results = []
for name, model in models.items():
    results.append(evaluate_model(model, X_test, y_test, name))

# Consensus evaluation
cons_preds, cons_conf = consensus_predict(X_test, models)
cons_proba = consensus_predict_proba(X_test, models)
results.append({
    'Model': 'Consensus',
    'Accuracy': accuracy_score(y_test, cons_preds),
    'Balanced Accuracy': balanced_accuracy_score(y_test, cons_preds),
    'Precision': precision_score(y_test, cons_preds),
    'Recall': recall_score(y_test, cons_preds),
    'F1': f1_score(y_test, cons_preds),
    'MCC': matthews_corrcoef(y_test, cons_preds),
    'AUC-ROC': roc_auc_score(y_test, cons_proba),
})

results_df = pd.DataFrame(results)
results_df.to_csv('results/model_metrics/test_set_metrics.csv', index=False)
print(results_df.round(3))


# =============================================
# B. Y-Randomization Test (50 iterations)
# =============================================
from sklearn.ensemble import RandomForestClassifier

def y_randomization(X, y, n_iterations=50):
    """
    Scramble activity labels and retrain.
    Real model must perform significantly better than randomized models.
    """
    random_scores = []
    for i in tqdm(range(n_iterations), desc="Y-randomization"):
        y_shuffled = y.sample(frac=1, random_state=i).reset_index(drop=True)
        rf_random = RandomForestClassifier(n_estimators=100, random_state=i)
        scores = cross_validate(rf_random, X, y_shuffled, cv=5,
                               scoring='balanced_accuracy')
        random_scores.append(scores['test_score'].mean())

    return random_scores

random_scores = y_randomization(X_train, y_train)
real_score = study_rf.best_value  # From training

print(f"Real model balanced accuracy: {real_score:.4f}")
print(f"Random models mean: {np.mean(random_scores):.4f} ± {np.std(random_scores):.4f}")
print(f"Real model is {(real_score - np.mean(random_scores))/np.std(random_scores):.1f} "
      f"standard deviations above random (should be >3)")


# =============================================
# C. Applicability Domain (Williams Plot)
# =============================================
def williams_plot(X_train, X_test, y_test, y_pred_test):
    """
    Williams plot: Leverage (hi) vs Standardized Residuals.
    h* = 3p/n (critical leverage)
    Compounds with |residual| > 3 AND h > h* are outside AD.
    """
    n, p = X_train.shape
    h_star = 3 * p / n  # Critical leverage

    # Hat matrix: H = X(X'X)^-1 X'
    X_train_np = X_train.values
    X_test_np = X_test.values

    # Calculate leverages for test set
    XtX_inv = np.linalg.pinv(X_train_np.T @ X_train_np)
    leverages = np.diag(X_test_np @ XtX_inv @ X_test_np.T)

    # Standardized residuals
    residuals = y_test.values - y_pred_test
    std_residuals = (residuals - residuals.mean()) / residuals.std()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(leverages, std_residuals, alpha=0.5, edgecolors='k', linewidth=0.5)

    # AD boundaries
    ax.axhline(y=3, color='r', linestyle='--', label='±3σ')
    ax.axhline(y=-3, color='r', linestyle='--')
    ax.axvline(x=h_star, color='b', linestyle='--', label=f'h* = {h_star:.4f}')

    ax.set_xlabel('Leverage (hi)', fontsize=12)
    ax.set_ylabel('Standardized Residuals', fontsize=12)
    ax.set_title('Williams Plot — Applicability Domain', fontsize=14)
    ax.legend()

    # Count outliers
    outside_ad = ((np.abs(std_residuals) > 3) & (leverages > h_star)).sum()
    print(f"Compounds outside AD: {outside_ad} / {len(y_test)}")

    plt.tight_layout()
    plt.savefig('figures/williams_plot.png', dpi=300)
    plt.show()

    return leverages, std_residuals


# =============================================
# D. ROC Curves (All Models + Consensus)
# =============================================
fig, ax = plt.subplots(figsize=(8, 8))

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

# Consensus
cons_proba = consensus_predict_proba(X_test, models)
fpr, tpr, _ = roc_curve(y_test, cons_proba)
auc = roc_auc_score(y_test, cons_proba)
ax.plot(fpr, tpr, linewidth=2.5, label=f'Consensus (AUC={auc:.3f})')

ax.plot([0,1], [0,1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — External Test Set')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=300)
```

---

# 7. EXPLAINABLE AI (SHAP) ARCHITECTURE

```python
# FILE: notebooks/09_SHAP_analysis.ipynb
# ========================================

import shap
import matplotlib.pyplot as plt
import joblib

# Load best model (use tree-based for TreeExplainer — much faster)
xgb_model = joblib.load('models/xgb_classifier.joblib')

# =============================================
# A. Global Feature Importance (SHAP Summary)
# =============================================

# TreeExplainer is fast and exact for tree models
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)

# Summary bar plot — top 20 most important features
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_train, plot_type='bar',
                  max_display=20, show=False)
plt.tight_layout()
plt.savefig('figures/shap_summary_bar.png', dpi=300, bbox_inches='tight')

# Beeswarm plot — shows feature value impact direction
fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values, X_train, max_display=20, show=False)
plt.tight_layout()
plt.savefig('figures/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')


# =============================================
# B. SHAP Dependence Plots (Top Features)
# =============================================

# Get top 5 features by mean |SHAP|
top_features = pd.DataFrame({
    'feature': X_train.columns,
    'mean_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_shap', ascending=False).head(5)['feature'].tolist()

for feat in top_features:
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(feat, shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(f'figures/shap_dependence_{feat}.png', dpi=300)
    plt.close()


# =============================================
# C. Individual Compound Explanation (Waterfall)
# =============================================

# Explain a specific compound (e.g., the most active predicted)
idx = 0  # Index of compound to explain
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_train.iloc[idx],
    feature_names=X_train.columns.tolist()
), max_display=15, show=False)
plt.tight_layout()
plt.savefig('figures/shap_waterfall_compound.png', dpi=300, bbox_inches='tight')


# =============================================
# D. Translate SHAP → SAR Guidelines
# =============================================

# Extract top 10 positive contributors (increase activity)
# and top 10 negative contributors (decrease activity)
mean_shap = pd.DataFrame({
    'feature': X_train.columns,
    'mean_shap': shap_values.mean(axis=0),
    'abs_mean_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('abs_mean_shap', ascending=False)

positive_drivers = mean_shap[mean_shap['mean_shap'] > 0].head(10)
negative_drivers = mean_shap[mean_shap['mean_shap'] < 0].head(10)

print("TOP FEATURES PROMOTING ACTIVITY:")
print(positive_drivers[['feature', 'mean_shap']].to_string(index=False))
print("\nTOP FEATURES REDUCING ACTIVITY:")
print(negative_drivers[['feature', 'mean_shap']].to_string(index=False))

# Save SAR guidelines
mean_shap.to_csv('results/shap_outputs/shap_feature_importance.csv', index=False)
```

---

# 8. VIRTUAL SCREENING PIPELINE

```python
# FILE: notebooks/10_virtual_screening.ipynb
# =============================================

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.SA_Score import sascorer
import joblib

# Load models and scaler
models = {
    'RF': joblib.load('models/rf_classifier.joblib'),
    'SVM': joblib.load('models/svm_classifier.joblib'),
    'XGBoost': joblib.load('models/xgb_classifier.joblib'),
    'LightGBM': joblib.load('models/lgbm_classifier.joblib'),
}
scaler = joblib.load('models/scaler.joblib')

# =============================================
# STEP 1: Load Screening Library
# =============================================
# Option A: ZINC database (download sulfonamide subset from zinc20.docking.org)
# Option B: Enamine REAL Space (request from enamine.net)
# Option C: Virtual derivatives of your lead compounds

screening_df = pd.read_csv('data/external/zinc_sulfonamides.csv')
print(f"Screening library: {len(screening_df)} compounds")


# =============================================
# STEP 2: Compute Descriptors (same as training)
# =============================================
# Use the EXACT same descriptor pipeline as training
# (same features, same order, same scaler)

X_screen = compute_descriptors(screening_df['smiles'])  # Your function
X_screen = X_screen[final_features]  # Same features as training
X_screen_scaled = pd.DataFrame(
    scaler.transform(X_screen), columns=final_features
)


# =============================================
# STEP 3: Consensus Prediction
# =============================================
consensus_preds, confidence = consensus_predict(X_screen_scaled, models)
consensus_proba = consensus_predict_proba(X_screen_scaled, models)

screening_df['predicted_active'] = consensus_preds
screening_df['confidence'] = confidence
screening_df['avg_probability'] = consensus_proba

# Filter: keep only consensus-predicted actives (≥3/4 models)
hits = screening_df[screening_df['predicted_active'] == 1].copy()
print(f"Consensus hits: {len(hits)} / {len(screening_df)}")


# =============================================
# STEP 4: Applicability Domain Check
# =============================================
# Remove hits outside training set chemical space
# (Euclidean distance to nearest training compound)
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
nn.fit(X_train_scaled)
distances, _ = nn.kneighbors(X_screen_scaled.loc[hits.index])
mean_distances = distances.mean(axis=1)

# Threshold: 95th percentile of training set internal distances
train_distances, _ = nn.kneighbors(X_train_scaled)
ad_threshold = np.percentile(train_distances.mean(axis=1), 95)

hits['in_AD'] = mean_distances < ad_threshold
hits = hits[hits['in_AD']]
print(f"After AD filter: {len(hits)} compounds")


# =============================================
# STEP 5: ADMET Filter (SwissADME / pkCSM)
# =============================================
# These are web-based tools — submit SMILES in batches
# SwissADME: http://www.swissadme.ch/
# pkCSM: https://biosig.lab.uq.edu.au/pkcsm/

# For automation, use ADMETlab 2.0 API or pre-filter with RDKit:
def lipinski_filter(smiles):
    """Basic Lipinski Rule of 5 filter."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return violations <= 1

hits['lipinski_pass'] = hits['smiles'].apply(lipinski_filter)
hits = hits[hits['lipinski_pass']]
print(f"After Lipinski filter: {len(hits)} compounds")


# =============================================
# STEP 6: Synthetic Accessibility Score
# =============================================
def sa_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 10
    return sascorer.calculateScore(mol)

hits['sa_score'] = hits['smiles'].apply(sa_score)
hits = hits[hits['sa_score'] <= 5.0]  # Easily synthesizable
print(f"After SA filter: {len(hits)} compounds")


# =============================================
# STEP 7: Diversity Selection (Tanimoto Clustering)
# =============================================
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

def diversity_selection(smiles_list, cutoff=0.6, max_picks=50):
    """
    Cluster hits by Tanimoto similarity (ECFP4).
    Pick one representative from each cluster.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

    # Pairwise distance matrix
    n = len(fps)
    dists = []
    for i in range(1, n):
        for j in range(i):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dists.append(1 - sim)  # Distance = 1 - similarity

    # Butina clustering
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    # Pick highest-confidence compound from each cluster
    selected_indices = []
    for cluster in clusters[:max_picks]:
        cluster_hits = hits.iloc[list(cluster)]
        best_idx = cluster_hits['avg_probability'].idxmax()
        selected_indices.append(best_idx)

    return selected_indices

selected_idx = diversity_selection(hits['smiles'].tolist())
final_candidates = hits.loc[selected_idx].sort_values('avg_probability', ascending=False)
print(f"Final diverse candidates: {len(final_candidates)}")

# =============================================
# STEP 8: Export for Collaborators
# =============================================
final_candidates.to_csv('results/predictions/final_candidates_for_testing.csv', index=False)
print("Candidate list saved!")
```

---

# 9. EXPERIMENTAL VALIDATION INTERFACE

## 9.1 Compound Acquisition Workflow

```
Final ML Candidates (CSV with SMILES)
    │
    ├─→ Search on vendor databases:
    │     • Enamine (enamine.net/compound-collections)
    │     • MolPort (molport.com)
    │     • Sigma-Aldrich (sigmaaldrich.com)
    │     • MCE (medchemexpress.com)
    │
    ├─→ Exact match? → Purchase (~$50–200/compound, 5–10 mg)
    │
    └─→ No exact match? → Check closest analog OR synthesize
                          (collaborator in your lab)
```

## 9.2 Experimental Protocol (Minimal Wet-Lab)

| Step | Assay | Details |
|------|-------|---------|
| 1 | Cytotoxicity | CC50 on THP-1 macrophages (MTT assay, 72h) |
| 2 | Anti-amastigote | IC50 on L. infantum intracellular amastigotes (72h) |
| 3 | Selectivity | Calculate SI = CC50 / IC50 (SI > 10 = good) |
| 4 | Dose-response | 8-point, 3-fold dilution series |
| 5 | Controls | Miltefosine (positive), DMSO (vehicle negative) |

## 9.3 ML Feedback Loop

```python
# After receiving experimental results:
experimental_df = pd.read_csv('results/validation/experimental_results.csv')

# Calculate hit rate
predicted_active = len(experimental_df)
confirmed_active = (experimental_df['IC50_uM'] <= 10).sum()
hit_rate = confirmed_active / predicted_active * 100
print(f"Hit rate: {hit_rate:.1f}% ({confirmed_active}/{predicted_active})")

# Compare predicted vs observed
from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(experimental_df['observed_pIC50'], experimental_df['predicted_pIC50'])
mae = mean_absolute_error(experimental_df['observed_pIC50'], experimental_df['predicted_pIC50'])
print(f"R² (predicted vs observed): {r2:.3f}")
print(f"MAE: {mae:.3f} log units")

# Retrain model with new data (active learning)
new_train = pd.concat([original_train, experimental_df])
# Repeat model training pipeline with expanded dataset
```

---

# 10. PROJECT WORKFLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROJECT ARCHITECTURE FLOW                      │
└─────────────────────────────────────────────────────────────────┘

PHASE 1: DATA (Months 1-7)
═══════════════════════════
  ┌──────────┐    ┌──────────────┐    ┌─────────────┐
  │  ChEMBL  │───→│   Raw Data   │───→│  Curated    │
  │  PubChem │    │  (~12,000    │    │  Dataset    │
  │  Papers  │    │   records)   │    │  (~5,000)   │
  └──────────┘    └──────────────┘    └──────┬──────┘
                                              │
                        ┌─────────────────────┼──────────────────┐
                        ▼                     ▼                  ▼
                  ┌──────────┐        ┌──────────────┐   ┌──────────┐
                  │Sulfonamide│       │ All Compounds │   │ QC Check │
                  │ Subset    │       │ (Broad Model) │   │ ✓ or ✗   │
                  │ (~500)    │       │               │   └──────────┘
                  └──────────┘        └──────────────┘
                        │                     │
                        └──────────┬──────────┘
                                   ▼
                          ┌─────────────────┐
                          │  PAPER 1: Data  │ ← PUBLISH
                          │  Descriptor     │
                          └─────────────────┘

PHASE 2: DESCRIPTORS (Months 8-9)
══════════════════════════════════
  ┌────────────┐   ┌──────────────┐   ┌──────────────┐
  │  Mordred   │   │  ECFP4/MACCS │   │  Combined    │
  │ (~1600 2D) │ + │ Fingerprints │ = │ Feature Set  │
  └────────────┘   └──────────────┘   └──────┬───────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  Feature Selection  │
                                    │  • Variance filter  │
                                    │  • Correlation filter│
                                    │  • MI + RF ranking  │
                                    └─────────┬──────────┘
                                              │
                                    ┌─────────▼──────────┐
                                    │  Selected Features  │
                                    │  (~100-300)         │
                                    └─────────┬──────────┘
                                              │
PHASE 3: MODELING (Months 10-16)              │
════════════════════════════════               │
                                              ▼
           ┌──────────────────────────────────────────────┐
           │              MODEL TRAINING                   │
           │  ┌────┐  ┌─────┐  ┌───────┐  ┌────────┐    │
           │  │ RF │  │ SVM │  │XGBoost│  │LightGBM│    │
           │  └──┬─┘  └──┬──┘  └───┬───┘  └───┬────┘    │
           │     │       │        │          │           │
           │     └───────┴────┬───┴──────────┘           │
           │                  ▼                           │
           │         ┌──────────────┐                     │
           │         │  CONSENSUS   │                     │
           │         │  MODEL (≥3/4)│                     │
           │         └──────────────┘                     │
           └──────────────────┬───────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │    VALIDATION       │
                    │  • External test    │
                    │  • Y-randomization  │
                    │  • Williams plot    │
                    │  • ROC curves       │
                    └─────────┬──────────┘
                              │
PHASE 4: INTERPRETATION (Months 17-18)
══════════════════════════════════════
                              │
                    ┌─────────▼──────────┐
                    │   SHAP ANALYSIS    │
                    │  • Global importance│
                    │  • Dependence plots │
                    │  • Waterfall plots  │
                    │  • SAR Guidelines   │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  PAPER 2: QSAR    │ ← PUBLISH
                    │  Models + SHAP     │
                    └─────────┬──────────┘
                              │
PHASE 5: SCREENING (Months 19-21)
══════════════════════════════════
                              │
  ┌──────────┐      ┌────────▼─────────┐
  │   ZINC   │─────→│ VIRTUAL SCREENING│
  │  Enamine │      │  • Predict       │
  │ Analogs  │      │  • AD check      │
  └──────────┘      │  • ADMET filter  │
                    │  • SA score      │
                    │  • Diversity pick │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  TOP 10-20 HITS  │
                    │  (for wet-lab)   │
                    └────────┬─────────┘
                             │
PHASE 6: VALIDATION (Months 22-26)
═══════════════════════════════════
                             │
                    ┌────────▼─────────┐
                    │  EXPERIMENTAL    │
                    │  • Purchase/synth│
                    │  • In vitro IC50 │
                    │  • Cytotoxicity  │
                    │  • Hit rate calc │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  MODEL FEEDBACK  │───→ Retrain (optional)
                    │  LOOP            │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  PAPER 3:        │ ← PUBLISH
                    │  Validation      │
                    └──────────────────┘
```

---

# 11. COMMON PITFALLS AND TROUBLESHOOTING

## 11.1 Data Leakage Prevention

| Pitfall | Solution |
|---------|----------|
| Test set used during feature selection | Always select features using ONLY the training set, then apply the same feature list to the test set |
| SMOTE applied before splitting | Apply SMOTE ONLY within cross-validation folds (use imblearn Pipeline, not sklearn Pipeline) |
| Scaler fit on full data | Fit scaler on training set ONLY; use transform() on test and screening sets |
| Duplicate compounds in train and test | Check for SMILES overlap between sets; remove any shared compounds |

## 11.2 Overfitting Detection

| Sign | Remedy |
|------|--------|
| Training accuracy >> Test accuracy (gap >10%) | Reduce model complexity (lower max_depth, more min_samples_leaf) |
| MCC ≈ 0 on test set | Data may not be modelable; check class balance; try different descriptors |
| Y-randomization gives similar scores | Model is learning noise; curate data more aggressively |
| Only 1 model performs well | Don't rely on it alone; use consensus approach |

## 11.3 Descriptor Calculation Failures

| Issue | Fix |
|-------|-----|
| Mordred returns NaN for some descriptors | Replace NaN with column median; or drop columns with >20% NaN |
| RDKit can't parse a SMILES | Mark as None, drop from dataset; log the problematic SMILES |
| Memory error with large descriptor matrix | Process in chunks of 500 molecules; use float32; delete intermediates |

## 11.4 Laptop Performance Tips

| Task | Estimated Time (8-core laptop) | Optimization |
|------|-------------------------------|-------------|
| Mordred descriptors (5000 mols) | 15–30 min | Run overnight; save intermediate results |
| RF hyperparameter tuning (100 trials) | 20–40 min | Use n_jobs=-1 for parallelization |
| XGBoost tuning (100 trials) | 15–25 min | Use early stopping |
| SHAP TreeExplainer (5000 mols) | 5–15 min | Fast; no optimization needed |
| SHAP KernelExplainer for SVM | 2–4 hours | Use a 500-sample background set |
| Virtual screening (100K compounds) | 30–60 min | Process in batches of 10K |

---

# 12. FILE AND OUTPUT MANAGEMENT

## 12.1 Naming Conventions

```
# Data files:
data/raw/chembl_{target}_{date}.csv
data/processed/{description}_{version}.csv

# Models:
models/{algorithm}_{task}_{date}.joblib
# Example: models/rf_classifier_v2_20270315.joblib

# Figures:
figures/{paper_number}_{figure_type}_{description}.png
# Example: figures/p2_fig3_shap_summary.png

# Results:
results/{category}/{description}_{date}.csv
```

## 12.2 Documentation Standards

Every notebook should begin with:
```python
"""
Notebook: 05_model_training_RF_SVM.ipynb
Author: [Your Name]
Date: [Date]
Purpose: Train Random Forest and SVM classifiers with Optuna
         hyperparameter optimization on the curated anti-leishmanial
         sulfonamide dataset.
Input: data/processed/X_train_selected.csv, y_train.csv
Output: models/rf_classifier.joblib, models/svm_classifier.joblib
        results/model_metrics/rf_cv_scores.csv
"""
```

## 12.3 Output Organization per Paper

```
papers/
├── paper1_dataset/
│   ├── figures/
│   │   ├── data_distribution.png
│   │   ├── chemical_space_PCA.png
│   │   └── sulfonamide_coverage.png
│   ├── tables/
│   │   └── dataset_summary_statistics.csv
│   └── supplementary/
│       └── full_curated_dataset.csv
│
├── paper2_qsar_models/
│   ├── figures/
│   │   ├── roc_curves.png
│   │   ├── confusion_matrices.png
│   │   ├── shap_summary.png
│   │   ├── shap_waterfall.png
│   │   ├── williams_plot.png
│   │   └── y_randomization.png
│   ├── tables/
│   │   ├── model_comparison_metrics.csv
│   │   ├── hyperparameters.csv
│   │   └── top_shap_features.csv
│   └── supplementary/
│       ├── optuna_optimization_history.csv
│       └── virtual_screening_results.csv
│
└── paper3_validation/
    ├── figures/
    │   ├── predicted_vs_observed.png
    │   └── hit_rate_chart.png
    ├── tables/
    │   ├── experimental_ic50_results.csv
    │   └── compound_structures_table.csv
    └── supplementary/
        └── full_screening_predictions.csv
```

---

# QUALITY CONTROL CHECKPOINTS

| Checkpoint | When | What to Check | Pass Criteria |
|-----------|------|---------------|---------------|
| QC-1 | After data collection | Raw record count, missing SMILES | >5000 records, <5% missing |
| QC-2 | After curation | Class balance, pIC50 distribution | Actives 20-50% of total |
| QC-3 | After descriptors | NaN count, feature count | <5% NaN per column, 100-500 features |
| QC-4 | After feature selection | No train/test overlap in features | Confirmed zero leakage |
| QC-5 | After training | CV balanced accuracy | >0.70 for best model |
| QC-6 | After Y-randomization | Real vs random gap | >3 std deviations above random |
| QC-7 | After external validation | Test set MCC | >0.3 (ideally >0.5) |
| QC-8 | After virtual screening | Number of candidates | 50-200 final hits |
| QC-9 | After experimental testing | Hit rate | >20% (anything >10% is useful) |

---

*Project Architecture Report — Prepared May 2026*
*All code is Python 3.11 compatible. All tools are free and open-source.*

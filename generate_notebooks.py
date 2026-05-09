#!/usr/bin/env python3
"""Generate Jupyter notebooks for the PhD QSAR Leishmania project."""

import nbformat as nbf
from pathlib import Path


def create_notebook_01_data_collection():
    """Create Notebook 01: Data Collection from ChEMBL and PubChem."""
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (qsar-leish)",
        "language": "python",
        "name": "python3",
    }

    cells = []

    # ─── Title Cell ──────────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""# Notebook 01 — Data Collection from ChEMBL and PubChem

**Project**: ML-Based QSAR Modeling for Anti-Leishmanial Sulfonamide Derivatives  
**Author**: [Your Name]  
**Date**: May 2026  
**Purpose**: Query ChEMBL and PubChem databases to retrieve bioactivity data for
anti-kinetoplastid compounds, with focus on sulfonamide-containing molecules.

**Input**: None (queries databases directly)  
**Output**:
- `data/raw/chembl_all_targets_raw.csv` — Combined raw dataset
- `data/raw/chembl_L_infantum_raw.csv` — L. infantum data
- `data/raw/chembl_L_donovani_raw.csv` — L. donovani data
- `data/raw/chembl_L_amazonensis_raw.csv` — L. amazonensis data
- `data/raw/chembl_T_cruzi_raw.csv` — T. cruzi data
- `data/raw/sulfonamide_subset_raw.csv` — Sulfonamide-containing compounds

---
"""))

    # ─── Cell 1: Imports ─────────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("## 1. Setup and Imports"))
    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 1: Imports and Configuration
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import time
import sys
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Add project root to path for importing src modules
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# Create data directories
(PROJECT_ROOT / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'data' / 'external').mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Python: {sys.version}")
"""))

    # ─── Cell 2: Verify Dependencies ─────────────────────────────
    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 2: Verify All Dependencies
# ============================================================

dependencies = {}
try:
    import rdkit
    dependencies['RDKit'] = rdkit.__version__
except ImportError:
    dependencies['RDKit'] = 'NOT INSTALLED — run: conda install -c conda-forge rdkit'

try:
    from chembl_webresource_client.new_client import new_client
    dependencies['ChEMBL Client'] = 'OK'
except ImportError:
    dependencies['ChEMBL Client'] = 'NOT INSTALLED — run: pip install chembl_webresource_client'

try:
    from tqdm import tqdm
    dependencies['tqdm'] = 'OK'
except ImportError:
    dependencies['tqdm'] = 'NOT INSTALLED — run: pip install tqdm'

for pkg, status in dependencies.items():
    symbol = '✓' if 'NOT' not in str(status) else '✗'
    print(f"  {symbol} {pkg}: {status}")

# Fail fast if critical dependencies missing
assert 'NOT' not in str(dependencies.get('RDKit', '')), "RDKit is required!"
assert 'NOT' not in str(dependencies.get('ChEMBL Client', '')), "ChEMBL client is required!"
print("\\nAll critical dependencies verified!")
"""))

    # ─── Cell 3: Define Targets ──────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## 2. Target Definitions

We collect bioactivity data for **four targets** across kinetoplastid parasites:

| Target | ChEMBL ID | Organism | Role |
|--------|-----------|----------|------|
| L. infantum | CHEMBL612848 | *Leishmania infantum* | **Primary** — thesis focus |
| L. donovani | CHEMBL367 | *Leishmania donovani* | Cross-species expansion |
| L. amazonensis | CHEMBL612877 | *Leishmania amazonensis* | Cross-species expansion |
| T. cruzi | CHEMBL368 | *Trypanosoma cruzi* | Cross-kinetoplastid (multitask) |
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 3: Define Targets and Query Parameters
# ============================================================

TARGETS = {
    'L_infantum': {
        'chembl_id': 'CHEMBL612848',
        'organism': 'Leishmania infantum',
        'priority': 'primary',
        'description': 'Visceral leishmaniasis — primary thesis target'
    },
    'L_donovani': {
        'chembl_id': 'CHEMBL367',
        'organism': 'Leishmania donovani',
        'priority': 'cross-species',
        'description': 'Visceral leishmaniasis — closely related species'
    },
    'L_amazonensis': {
        'chembl_id': 'CHEMBL612877',
        'organism': 'Leishmania amazonensis',
        'priority': 'cross-species',
        'description': 'Cutaneous leishmaniasis — New World species'
    },
    'T_cruzi': {
        'chembl_id': 'CHEMBL368',
        'organism': 'Trypanosoma cruzi',
        'priority': 'cross-kinetoplastid',
        'description': 'Chagas disease — shared sulfonamide targets (TcME)'
    },
}

# Activity types of interest
ACTIVITY_TYPES = ['IC50', 'EC50']

# Fields to retrieve from ChEMBL API
FIELDS = [
    'molecule_chembl_id',
    'canonical_smiles',
    'standard_type',
    'standard_value',
    'standard_units',
    'standard_relation',
    'pchembl_value',
    'assay_chembl_id',
    'assay_description',
    'assay_type',
    'document_chembl_id',
    'target_chembl_id',
]

print("Targets configured:")
for name, info in TARGETS.items():
    print(f"  • {name} ({info['chembl_id']}) — {info['description']}")
"""))

    # ─── Cell 4: ChEMBL Query Function ───────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## 3. Data Collection from ChEMBL

### 3.1 Query Function

The ChEMBL Web Resource Client provides a Python interface to the ChEMBL REST API.
We filter for:
- **IC50 and EC50** measurements only
- **Standard units = nM** (nanomolar)
- **Exact values only** (relation = '=', excluding '>' or '<')
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 4: ChEMBL Data Retrieval Function
# ============================================================

from chembl_webresource_client.new_client import new_client

# Initialize API clients
activity_api = new_client.activity
molecule_api = new_client.molecule
target_api = new_client.target

def fetch_bioactivity(target_id, target_name, activity_types=None,
                      max_retries=3):
    \"\"\"
    Fetch bioactivity data from ChEMBL for a given target.
    
    Parameters
    ----------
    target_id : str
        ChEMBL target ID (e.g., 'CHEMBL612848')
    target_name : str
        Human-readable name (e.g., 'L_infantum')
    activity_types : list
        Activity measurement types to retrieve
    max_retries : int
        Number of retry attempts on API failure
        
    Returns
    -------
    pd.DataFrame with bioactivity records
    \"\"\"
    if activity_types is None:
        activity_types = ACTIVITY_TYPES
    
    print(f"\\n{'─'*50}")
    print(f"Fetching: {target_name} ({target_id})")
    print(f"{'─'*50}")
    
    for attempt in range(max_retries):
        try:
            # Query ChEMBL API
            activities = activity_api.filter(
                target_chembl_id=target_id,
                standard_type__in=activity_types,
                standard_units='nM',
                standard_relation='=',
            ).only(FIELDS)
            
            # Convert to list (triggers the actual API call)
            print(f"  Querying API (attempt {attempt+1})...")
            records = list(activities)
            
            # Convert to DataFrame
            df = pd.DataFrame.from_records(records)
            
            if df.empty:
                print(f"  ⚠ No records found for {target_name}")
                return pd.DataFrame()
            
            # Add metadata
            df['target_name'] = target_name
            df['target_organism'] = TARGETS[target_name]['organism']
            df['data_source'] = 'ChEMBL'
            
            # Quick stats
            n_compounds = df['molecule_chembl_id'].nunique()
            n_with_smiles = df['canonical_smiles'].notna().sum()
            n_with_pchembl = df['pchembl_value'].notna().sum()
            
            print(f"  ✓ Retrieved {len(df)} activity records")
            print(f"    Unique compounds: {n_compounds}")
            print(f"    With SMILES: {n_with_smiles}")
            print(f"    With pChEMBL value: {n_with_pchembl}")
            
            return df
            
        except Exception as e:
            print(f"  ✗ Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ✗ All attempts failed for {target_name}")
                return pd.DataFrame()

print("Query function defined ✓")
"""))

    # ─── Cell 5: Execute Queries ─────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""### 3.2 Execute Queries for All Targets

⏱ **Expected time**: 5–15 minutes depending on internet speed and server load.

The function queries each target sequentially with a 3-second pause between
queries to avoid overloading the ChEMBL API server.
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 5: Collect Data for All Targets
# ============================================================

all_data = []
collection_summary = []

print("=" * 60)
print("STARTING DATA COLLECTION")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

for name, info in TARGETS.items():
    df = fetch_bioactivity(info['chembl_id'], name)
    
    if not df.empty:
        # Save individual target file
        filepath = PROJECT_ROOT / 'data' / 'raw' / f"chembl_{name}_raw.csv"
        df.to_csv(filepath, index=False)
        print(f"  💾 Saved to {filepath.name}")
        
        all_data.append(df)
        collection_summary.append({
            'Target': name,
            'ChEMBL ID': info['chembl_id'],
            'Records': len(df),
            'Unique Compounds': df['molecule_chembl_id'].nunique(),
            'With pChEMBL': df['pchembl_value'].notna().sum(),
            'Activity Types': ', '.join(df['standard_type'].unique()),
        })
    
    # Be polite to the API — wait between queries
    time.sleep(3)

print("\\n" + "=" * 60)
print("DATA COLLECTION COMPLETE")
print("=" * 60)
"""))

    # ─── Cell 6: Combine Data ────────────────────────────────────
    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 6: Combine All Targets into Single Dataset
# ============================================================

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save combined file
    combined_path = PROJECT_ROOT / 'data' / 'raw' / 'chembl_all_targets_raw.csv'
    combined_df.to_csv(combined_path, index=False)
    
    print(f"Combined dataset: {len(combined_df)} total records")
    print(f"Unique compounds: {combined_df['molecule_chembl_id'].nunique()}")
    print(f"Saved to: {combined_path.name}")
    
    # Display summary table
    summary_df = pd.DataFrame(collection_summary)
    print("\\n" + summary_df.to_string(index=False))
else:
    print("⚠ No data was collected! Check your internet connection and retry.")
    combined_df = pd.DataFrame()
"""))

    # ─── Cell 7: Explore Raw Data ────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## 4. Exploratory Analysis of Raw Data

Quick look at the retrieved data before curation.
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 7: Basic Data Exploration
# ============================================================

if not combined_df.empty:
    print("Dataset Shape:", combined_df.shape)
    print("\\nColumn Types:")
    print(combined_df.dtypes)
    print("\\nMissing Values:")
    print(combined_df.isnull().sum())
    print("\\nFirst 5 Rows:")
    display(combined_df.head())
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 8: Activity Value Distribution
# ============================================================

if not combined_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convert standard_value to numeric
    combined_df['standard_value'] = pd.to_numeric(
        combined_df['standard_value'], errors='coerce'
    )
    valid_data = combined_df[combined_df['standard_value'] > 0].copy()
    valid_data['log_value'] = np.log10(valid_data['standard_value'])
    
    # Plot 1: Distribution of log(IC50) by target
    for target in valid_data['target_name'].unique():
        subset = valid_data[valid_data['target_name'] == target]
        axes[0].hist(subset['log_value'], bins=40, alpha=0.5, label=target)
    
    axes[0].axvline(x=np.log10(10000), color='red', linestyle='--',
                    linewidth=1.5, label='10 μM threshold')
    axes[0].set_xlabel('log₁₀(IC50 in nM)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Activity Distribution (Raw)')
    axes[0].legend(fontsize=8)
    
    # Plot 2: Records per target
    target_counts = combined_df['target_name'].value_counts()
    bars = axes[1].bar(target_counts.index, target_counts.values,
                       color=['#2563EB', '#DC2626', '#16A34A', '#F59E0B'])
    axes[1].set_title('Records per Target')
    axes[1].set_ylabel('Number of Records')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, target_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                     f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'figures' / 'raw_data_overview.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    print("Figure saved: figures/raw_data_overview.png")
"""))

    # ─── Cell 9: Sulfonamide Filter ──────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## 5. Sulfonamide Substructure Filtering

We identify sulfonamide-containing compounds using SMARTS patterns:

| Pattern | Description |
|---------|-------------|
| `[S](=O)(=O)[NH]` | Secondary sulfonamide (R-SO₂-NH-R') |
| `[S](=O)(=O)[NH2]` | Primary sulfonamide (R-SO₂-NH₂) |
| `[S](=O)(=O)N` | Tertiary sulfonamide (R-SO₂-NR'R'') |

This is a **critical filter** — our QSAR models will be specifically optimized
for this chemical class.
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 9: Sulfonamide Substructure Filter
# ============================================================

from rdkit import Chem
from rdkit.Chem import Draw, AllChem

# Define SMARTS patterns for sulfonamide moieties
SULFONAMIDE_SMARTS = [
    '[S](=O)(=O)[NH]',     # Secondary sulfonamide
    '[S](=O)(=O)[NH2]',    # Primary sulfonamide
    '[S](=O)(=O)N',        # Tertiary sulfonamide (catches all N-substituted)
]

# Compile patterns
compiled_patterns = []
for smarts in SULFONAMIDE_SMARTS:
    pat = Chem.MolFromSmarts(smarts)
    if pat is not None:
        compiled_patterns.append((smarts, pat))
        print(f"  ✓ Compiled SMARTS: {smarts}")
    else:
        print(f"  ✗ Failed to compile: {smarts}")

def has_sulfonamide(smiles):
    \"\"\"Check if a molecule contains any sulfonamide substructure.\"\"\"
    if pd.isna(smiles):
        return False
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    for _, pattern in compiled_patterns:
        if mol.HasSubstructMatch(pattern):
            return True
    return False

print(f"\\nApplying sulfonamide filter to {len(combined_df)} compounds...")
combined_df['is_sulfonamide'] = combined_df['canonical_smiles'].apply(has_sulfonamide)

n_sulfonamide = combined_df['is_sulfonamide'].sum()
n_total = len(combined_df)
print(f"\\nResults:")
print(f"  Sulfonamide compounds:     {n_sulfonamide} ({n_sulfonamide/n_total*100:.1f}%)")
print(f"  Non-sulfonamide compounds: {n_total - n_sulfonamide} ({(n_total-n_sulfonamide)/n_total*100:.1f}%)")
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 10: Sulfonamide Distribution by Target
# ============================================================

sulfonamide_by_target = combined_df.groupby('target_name')['is_sulfonamide'].agg(
    ['sum', 'count']
).rename(columns={'sum': 'Sulfonamides', 'count': 'Total'})
sulfonamide_by_target['Percentage'] = (
    sulfonamide_by_target['Sulfonamides'] / sulfonamide_by_target['Total'] * 100
).round(1)

print("Sulfonamide Compounds per Target:")
print(sulfonamide_by_target.to_string())

# Visualize
fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(sulfonamide_by_target))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x],
               sulfonamide_by_target['Total'] - sulfonamide_by_target['Sulfonamides'],
               width, label='Non-Sulfonamide', color='#94A3B8')
bars2 = ax.bar([i + width/2 for i in x],
               sulfonamide_by_target['Sulfonamides'],
               width, label='Sulfonamide', color='#2563EB')
ax.set_xticks(x)
ax.set_xticklabels(sulfonamide_by_target.index, rotation=45)
ax.set_ylabel('Number of Compounds')
ax.set_title('Sulfonamide vs Non-Sulfonamide Compounds by Target')
ax.legend()
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'figures' / 'sulfonamide_distribution.png',
            dpi=300, bbox_inches='tight')
plt.show()
"""))

    # ─── Cell 11: Visualize Example Sulfonamides ─────────────────
    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 11: Visualize Example Sulfonamide Structures
# ============================================================

from rdkit.Chem import Draw

# Get unique sulfonamide SMILES
sulfo_df = combined_df[combined_df['is_sulfonamide']].copy()
unique_sulfo_smiles = sulfo_df['canonical_smiles'].dropna().unique()

# Draw up to 12 example structures
n_examples = min(12, len(unique_sulfo_smiles))
example_mols = []
for smi in unique_sulfo_smiles[:n_examples]:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        example_mols.append(mol)

if example_mols:
    img = Draw.MolsToGridImage(
        example_mols[:12],
        molsPerRow=4,
        subImgSize=(350, 300),
        legends=[f"Mol {i+1}" for i in range(len(example_mols[:12]))]
    )
    display(img)
    print(f"Showing {len(example_mols[:12])} example sulfonamide structures")
else:
    print("No valid sulfonamide structures to display")
"""))

    # ─── Cell 12: Save Datasets ──────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## 6. Save Raw Datasets"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 12: Save All Raw Datasets
# ============================================================

raw_dir = PROJECT_ROOT / 'data' / 'raw'

# 1. Full combined dataset (already saved above)
print("Saved files:")
print(f"  ✓ chembl_all_targets_raw.csv ({len(combined_df)} records)")

# 2. Sulfonamide subset
sulfo_subset = combined_df[combined_df['is_sulfonamide']].copy()
sulfo_path = raw_dir / 'sulfonamide_subset_raw.csv'
sulfo_subset.to_csv(sulfo_path, index=False)
print(f"  ✓ sulfonamide_subset_raw.csv ({len(sulfo_subset)} records)")

# 3. Non-sulfonamide (for broader model training)
non_sulfo = combined_df[~combined_df['is_sulfonamide']].copy()
non_sulfo_path = raw_dir / 'non_sulfonamide_raw.csv'
non_sulfo.to_csv(non_sulfo_path, index=False)
print(f"  ✓ non_sulfonamide_raw.csv ({len(non_sulfo)} records)")

# List all saved files
print(f"\\nAll files in {raw_dir}:")
for f in sorted(raw_dir.glob('*.csv')):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name} ({size_mb:.2f} MB)")
"""))

    # ─── Cell 13: Data Quality Summary ───────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## 7. Data Quality Report

### QC-1 Checkpoint

| Criterion | Pass Condition | Status |
|-----------|---------------|--------|
| Total records | > 5,000 | ✓/✗ |
| Missing SMILES | < 5% | ✓/✗ |
| Each target has data | All 4 targets | ✓/✗ |
| Sulfonamide subset | > 100 compounds | ✓/✗ |
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 13: Quality Control Checkpoint QC-1
# ============================================================

print("=" * 60)
print("QUALITY CONTROL CHECKPOINT QC-1: Data Collection")
print("=" * 60)

checks = {}

# Check 1: Total records > 5000
checks['Total records > 5,000'] = len(combined_df) > 5000

# Check 2: Missing SMILES < 5%
missing_smiles_pct = combined_df['canonical_smiles'].isna().mean() * 100
checks[f'Missing SMILES < 5% (actual: {missing_smiles_pct:.1f}%)'] = missing_smiles_pct < 5

# Check 3: All targets have data
targets_with_data = combined_df['target_name'].nunique()
checks[f'All 4 targets have data (actual: {targets_with_data})'] = targets_with_data == 4

# Check 4: Sulfonamide subset > 100
n_sulfo = combined_df['is_sulfonamide'].sum()
checks[f'Sulfonamide subset > 100 (actual: {n_sulfo})'] = n_sulfo > 100

# Print results
all_passed = True
for check, passed in checks.items():
    symbol = '✓ PASS' if passed else '✗ FAIL'
    print(f"  {symbol}: {check}")
    if not passed:
        all_passed = False

print(f"\\n{'='*60}")
if all_passed:
    print("✓ ALL CHECKS PASSED — Ready for Notebook 02 (Data Curation)")
else:
    print("⚠ SOME CHECKS FAILED — Review data before proceeding")
print(f"{'='*60}")
"""))

    # ─── Cell 14: Next Steps ─────────────────────────────────────
    cells.append(nbf.v4.new_markdown_cell("""## 8. Next Steps

✅ **Data collection complete!** Proceed to:

→ **Notebook 02: Data Curation** (`02_data_curation.ipynb`)
  - SMILES standardization (RDKit)
  - Duplicate handling (concordant/discordant)
  - Activity conversion (IC50 → pIC50)
  - Activity classification (Active/Inactive at 10 μM)
  - Drug-likeness filtering
  - Train/test split (70/30 stratified)

### Data collected:
- **Full dataset**: All IC50/EC50 data for 4 kinetoplastid targets
- **Sulfonamide subset**: Compounds containing -SO₂-NH- substructure
- **Metadata**: Assay descriptions, document references, pChEMBL values

---
*End of Notebook 01*
"""))

    nb.cells = cells
    return nb


def create_notebook_02_data_curation():
    """Create Notebook 02: Data Curation Pipeline."""
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (qsar-leish)",
        "language": "python",
        "name": "python3",
    }

    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Notebook 02 — Data Curation Pipeline

**Project**: ML-Based QSAR Modeling for Anti-Leishmanial Sulfonamide Derivatives  
**Author**: [Your Name]  
**Date**: May 2026  
**Purpose**: Clean and standardize the raw ChEMBL dataset following  
Fourches et al. (2010, 2016) curation guidelines.

**Input**: `data/raw/chembl_all_targets_raw.csv`  
**Output**:
- `data/processed/curated_dataset.csv`
- `data/processed/sulfonamide_subset.csv`
- `data/processed/train_set.csv`
- `data/processed/test_set.csv`

### Curation Pipeline:
```
Raw Data → SMILES Standardization → Remove Invalid → Handle Duplicates
         → IC50 → pIC50 → Activity Classification → Drug-likeness Filter
         → Train/Test Split → Curated Dataset
```

---
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 1: Setup
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

print(f"Project root: {PROJECT_ROOT}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 2: Load Raw Data
# ============================================================
raw_path = PROJECT_ROOT / 'data' / 'raw' / 'chembl_all_targets_raw.csv'
df = pd.read_csv(raw_path)
print(f"Loaded raw data: {len(df)} records, {df['molecule_chembl_id'].nunique()} unique compounds")
print(f"Targets: {df['target_name'].unique().tolist()}")
print(f"\\nShape: {df.shape}")
df.head()
"""))

    cells.append(nbf.v4.new_markdown_cell("## Step 1: SMILES Standardization\\n\\nFollowing Fourches et al. guidelines:\\n- Remove salts and counterions (keep largest fragment)\\n- Neutralize charges\\n- Normalize tautomers\\n- Canonicalize SMILES"))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 3: SMILES Standardization
# ============================================================

def standardize_smiles(smiles):
    \"\"\"Standardize SMILES: remove salts, neutralize, normalize, canonicalize.\"\"\"
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        remover = rdMolStandardize.LargestFragmentChooser()
        mol = remover.choose(mol)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

print("Standardizing SMILES (this may take a few minutes)...")
df['std_smiles'] = df['canonical_smiles'].apply(standardize_smiles)

n_before = len(df)
df = df.dropna(subset=['std_smiles'])
print(f"  Valid after standardization: {len(df)}/{n_before} ({n_before - len(df)} removed)")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Step 2: Remove Invalid Activity Entries"))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 4: Remove Invalid Activities
# ============================================================
df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
n_before = len(df)
df = df[df['standard_value'].notna()]
df = df[df['standard_value'] > 0]
print(f"After removing invalid activities: {len(df)}/{n_before}")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Step 3: Handle Duplicates\\n\\n- **Concordant** (spread < 1 log unit): keep median\\n- **Discordant** (spread ≥ 1 log unit): remove all"))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 5: Handle Duplicates
# ============================================================

grouped = df.groupby(['std_smiles', 'target_name'])
clean_rows = []
removed_discordant = 0
merged_concordant = 0

for (smi, target), group in grouped:
    if len(group) == 1:
        clean_rows.append(group.iloc[0])
    else:
        log_values = np.log10(group['standard_value'].values.astype(float))
        spread = log_values.max() - log_values.min()
        if spread < 1.0:
            median_row = group.iloc[0].copy()
            median_row['standard_value'] = group['standard_value'].median()
            clean_rows.append(median_row)
            merged_concordant += len(group) - 1
        else:
            removed_discordant += len(group)

df = pd.DataFrame(clean_rows).reset_index(drop=True)
print(f"Merged {merged_concordant} concordant duplicates")
print(f"Removed {removed_discordant} discordant entries")
print(f"After deduplication: {len(df)} entries")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Step 4: Convert IC50 → pIC50"))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 6: Activity Conversion and Classification
# ============================================================

# Convert IC50 (nM) to pIC50 = -log10(IC50 in M)
df['pIC50'] = -np.log10(df['standard_value'].astype(float) * 1e-9)
print(f"pIC50 range: {df['pIC50'].min():.2f} – {df['pIC50'].max():.2f}")
print(f"pIC50 median: {df['pIC50'].median():.2f}")

# Classify: Active (pIC50 >= 5, i.e., IC50 <= 10 μM) vs Inactive
THRESHOLD = 5.0
df['activity_class'] = df['pIC50'].apply(lambda x: 'Active' if x >= THRESHOLD else 'Inactive')

n_active = (df['activity_class'] == 'Active').sum()
print(f"\\nActive: {n_active} ({n_active/len(df)*100:.1f}%)")
print(f"Inactive: {len(df) - n_active} ({(len(df)-n_active)/len(df)*100:.1f}%)")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Step 5: Drug-likeness Filter"))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 7: Drug-likeness Filter
# ============================================================
metals = {'Fe', 'Cu', 'Zn', 'Mn', 'Co', 'Ni', 'Pt', 'Pd', 'Ru',
          'Rh', 'Ir', 'Os', 'Au', 'Ag', 'Hg', 'Cd', 'Cr', 'Mo', 'W'}

keep_mask = []
for _, row in df.iterrows():
    mol = Chem.MolFromSmiles(str(row['std_smiles']))
    if mol is None:
        keep_mask.append(False)
        continue
    mw = Descriptors.MolWt(mol)
    if mw > 900:
        keep_mask.append(False)
        continue
    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if atom_symbols & metals:
        keep_mask.append(False)
        continue
    keep_mask.append(True)

df = df[keep_mask].reset_index(drop=True)
print(f"After drug-likeness filter: {len(df)} compounds")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Step 6: Add Scaffolds and Split Dataset"))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 8: Murcko Scaffolds + Train/Test Split
# ============================================================

# Compute Murcko scaffolds
def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ''
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

print("Computing Murcko scaffolds...")
df['scaffold'] = df['std_smiles'].apply(get_scaffold)
print(f"Unique scaffolds: {df['scaffold'].nunique()}")

# Stratified random split (70/30)
train_df, test_df = train_test_split(
    df, test_size=0.30, random_state=42,
    stratify=df['activity_class']
)
print(f"\\nTraining set: {len(train_df)} compounds")
print(f"Test set: {len(test_df)} compounds")
print(f"Train active ratio: {(train_df['activity_class'] == 'Active').mean():.1%}")
print(f"Test active ratio: {(test_df['activity_class'] == 'Active').mean():.1%}")
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 9: Save Curated Datasets
# ============================================================
processed_dir = PROJECT_ROOT / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(processed_dir / 'curated_dataset.csv', index=False)
train_df.to_csv(processed_dir / 'train_set.csv', index=False)
test_df.to_csv(processed_dir / 'test_set.csv', index=False)

if 'is_sulfonamide' in df.columns:
    sulfo = df[df['is_sulfonamide']]
    sulfo.to_csv(processed_dir / 'sulfonamide_subset.csv', index=False)
    print(f"Sulfonamide subset: {len(sulfo)} compounds")

print("\\nAll curated files saved:")
for f in sorted(processed_dir.glob('*.csv')):
    print(f"  ✓ {f.name}")
"""))

    cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 10: Visualization — Curated Data Summary
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. pIC50 distribution
axes[0,0].hist(df['pIC50'], bins=40, color='#2563EB', alpha=0.7, edgecolor='white')
axes[0,0].axvline(x=5.0, color='red', linestyle='--', linewidth=1.5, label='Active threshold')
axes[0,0].set_xlabel('pIC50')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('pIC50 Distribution (Curated)')
axes[0,0].legend()

# 2. Class balance by target
class_counts = df.groupby(['target_name', 'activity_class']).size().unstack(fill_value=0)
class_counts.plot(kind='bar', ax=axes[0,1], color=['#DC2626', '#16A34A'])
axes[0,1].set_title('Active vs Inactive by Target')
axes[0,1].set_ylabel('Count')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Activity type distribution
df['standard_type'].value_counts().plot(kind='bar', ax=axes[1,0], color='#8B5CF6')
axes[1,0].set_title('Activity Type Distribution')
axes[1,0].set_ylabel('Count')

# 4. Molecular weight distribution
mw_values = []
for smi in df['std_smiles']:
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        mw_values.append(Descriptors.MolWt(mol))
axes[1,1].hist(mw_values, bins=40, color='#F59E0B', alpha=0.7, edgecolor='white')
axes[1,1].set_xlabel('Molecular Weight (Da)')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('Molecular Weight Distribution')

plt.suptitle('Curated Dataset Summary', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / 'figures' / 'curated_data_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\n✓ Curation complete! Proceed to Notebook 03: Descriptor Calculation")
"""))

    nb.cells = cells
    return nb


# ─── Main: Generate All Notebooks ───────────────────────────────
if __name__ == "__main__":
    output_dir = Path("/home/claude/PhD_QSAR_Leishmania/notebooks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Notebook 01
    nb1 = create_notebook_01_data_collection()
    nb1_path = output_dir / "01_data_collection.ipynb"
    with open(nb1_path, "w") as f:
        nbf.write(nb1, f)
    print(f"✓ Created: {nb1_path}")

    # Notebook 02
    nb2 = create_notebook_02_data_curation()
    nb2_path = output_dir / "02_data_curation.ipynb"
    with open(nb2_path, "w") as f:
        nbf.write(nb2, f)
    print(f"✓ Created: {nb2_path}")

    print("\nNotebooks generated successfully!")

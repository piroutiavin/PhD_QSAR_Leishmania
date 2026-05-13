"""Fix: restore main header and update correct Step 5 markdown cell."""
import json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

NB = Path(r'e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\02_data_curation.ipynb')

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0 was overwritten with Step 5 content. Restore main header.
nb['cells'][0]['source'] = [
    "# Notebook 02 — Data Curation Pipeline\n",
    "\n",
    "**Project**: ML-Based QSAR Modeling for Anti-Leishmanial Sulfonamide Derivatives  \n",
    "**Author**: [Your Name]  \n",
    "**Date**: May 2026  \n",
    "**Purpose**: Clean and standardize the raw ChEMBL dataset following  \n",
    "Fourches et al. (2010, 2016) curation guidelines.\n",
    "\n",
    "**Input**: `data/raw/chembl_all_targets_raw.csv`  \n",
    "**Output**:\n",
    "- `data/processed/curated_dataset.csv`\n",
    "- `data/processed/sulfonamide_subset.csv`\n",
    "- `data/processed/train_set.csv`\n",
    "- `data/processed/test_set.csv`\n",
    "\n",
    "### Curation Pipeline:\n",
    "```\n",
    "Raw Data → SMILES Standardization → Remove Invalid → Handle Duplicates\n",
    "         → IC50 → pIC50 → Activity Classification → Drug-likeness Filter (Lipinski Ro5)\n",
    "         → Train/Test Split → Curated Dataset\n",
    "```\n",
    "\n",
    "---"
]
print("Restored main header at cell 0")

# Find the Step 5 markdown cell (should be right before Cell 7 code)
code7_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'CELL 7' in src and 'Lipinski' in src and cell['cell_type'] == 'code':
        code7_idx = i
        break

if code7_idx:
    md_idx = code7_idx - 1
    if nb['cells'][md_idx]['cell_type'] == 'markdown':
        nb['cells'][md_idx]['source'] = [
            "## Step 5: Drug-likeness Filter (Lipinski's Rule of Five)\n",
            "\n",
            "Lipinski's Rule of Five criteria:\n",
            "- **MW** ≤ 500 Da\n",
            "- **LogP** ≤ 5\n",
            "- **HBD** (hydrogen bond donors) ≤ 5\n",
            "- **HBA** (hydrogen bond acceptors) ≤ 10\n",
            "\n",
            "Compounds with **≥ 2 violations** are removed as non-drug-like.  \n",
            "Organometallic compounds (containing Fe, Cu, Zn, etc.) are also removed."
        ]
        print(f"Updated Step 5 markdown at cell {md_idx}")
    else:
        print(f"Cell {md_idx} is {nb['cells'][md_idx]['cell_type']}, not markdown")
else:
    print("Could not find Cell 7 code cell")

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done!")

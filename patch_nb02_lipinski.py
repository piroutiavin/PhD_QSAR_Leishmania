"""Patch notebook 02: Replace Cell 7 Drug-likeness Filter with Lipinski Ro5."""
import json
from pathlib import Path

NB = Path(r'e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\02_data_curation.ipynb')

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Cell 7 by searching for "Drug-likeness Filter" in code cells
target_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'CELL 7' in src and 'Drug-likeness' in src and cell['cell_type'] == 'code':
        target_idx = i
        break

if target_idx is None:
    print("ERROR: Could not find Cell 7 (Drug-likeness Filter)")
    raise SystemExit(1)

print(f"Found Cell 7 at index {target_idx}")
print(f"Old source preview: {(''.join(nb['cells'][target_idx]['source']))[:100]}...")

# Also find the markdown cell before Cell 7 (Step 5 header)
md_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'Drug-likeness Filter' in src and cell['cell_type'] == 'markdown':
        md_idx = i
        break

if md_idx is not None:
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
    print(f"Updated markdown header at index {md_idx}")

# Replace Cell 7 code
new_source = [
    "# ============================================================\n",
    "# CELL 7: Drug-likeness Filter (Lipinski's Rule of Five)\n",
    "# ============================================================\n",
    "\n",
    "metals = {'Fe', 'Cu', 'Zn', 'Mn', 'Co', 'Ni', 'Pt', 'Pd', 'Ru',\n",
    "          'Rh', 'Ir', 'Os', 'Au', 'Ag', 'Hg', 'Cd', 'Cr', 'Mo', 'W'}\n",
    "\n",
    "keep_mask = []\n",
    "lipinski_data = []\n",
    "removed_metals = 0\n",
    "removed_lipinski = 0\n",
    "\n",
    "for _, row in df.iterrows():\n",
    "    mol = Chem.MolFromSmiles(str(row['std_smiles']))\n",
    "    if mol is None:\n",
    "        keep_mask.append(False)\n",
    "        lipinski_data.append({'MW': None, 'LogP': None, 'HBD': None, 'HBA': None, 'violations': None})\n",
    "        continue\n",
    "\n",
    "    # Check for metals (organometallics)\n",
    "    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}\n",
    "    if atom_symbols & metals:\n",
    "        keep_mask.append(False)\n",
    "        removed_metals += 1\n",
    "        lipinski_data.append({'MW': None, 'LogP': None, 'HBD': None, 'HBA': None, 'violations': None})\n",
    "        continue\n",
    "\n",
    "    # Lipinski's Rule of Five\n",
    "    mw   = Descriptors.MolWt(mol)\n",
    "    logp = Descriptors.MolLogP(mol)\n",
    "    hbd  = Descriptors.NumHDonors(mol)\n",
    "    hba  = Descriptors.NumHAcceptors(mol)\n",
    "    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])\n",
    "\n",
    "    lipinski_data.append({'MW': mw, 'LogP': logp, 'HBD': hbd, 'HBA': hba, 'violations': violations})\n",
    "\n",
    "    if violations >= 2:\n",
    "        keep_mask.append(False)\n",
    "        removed_lipinski += 1\n",
    "    else:\n",
    "        keep_mask.append(True)\n",
    "\n",
    "n_before = len(df)\n",
    "df = df[keep_mask].reset_index(drop=True)\n",
    "\n",
    "print(f\"Drug-likeness Filter (Lipinski's Rule of Five)\")\n",
    "print(f\"  Before filter          : {n_before:,}\")\n",
    "print(f\"  Removed (metals)       : {removed_metals}\")\n",
    "print(f\"  Removed (Lipinski >=2) : {removed_lipinski:,}\")\n",
    "print(f\"  After filter           : {len(df):,}\")\n",
    "print()\n",
    "\n",
    "# Show Lipinski compliance summary\n",
    "lip_df = pd.DataFrame(lipinski_data).dropna()\n",
    "print(\"Lipinski Property Violations (before filtering):\")\n",
    "print(f\"  MW  > 500:  {(lip_df['MW'] > 500).sum():,} ({(lip_df['MW'] > 500).mean()*100:.1f}%)\")\n",
    "print(f\"  LogP > 5:   {(lip_df['LogP'] > 5).sum():,} ({(lip_df['LogP'] > 5).mean()*100:.1f}%)\")\n",
    "print(f\"  HBD > 5:    {(lip_df['HBD'] > 5).sum():,} ({(lip_df['HBD'] > 5).mean()*100:.1f}%)\")\n",
    "print(f\"  HBA > 10:   {(lip_df['HBA'] > 10).sum():,} ({(lip_df['HBA'] > 10).mean()*100:.1f}%)\")\n",
]

nb['cells'][target_idx]['source'] = new_source
nb['cells'][target_idx]['outputs'] = []
nb['cells'][target_idx]['execution_count'] = None

# Also clear outputs for all subsequent cells (they'll be regenerated on re-run)
for i in range(target_idx + 1, len(nb['cells'])):
    if nb['cells'][i]['cell_type'] == 'code':
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nNotebook patched successfully!")
print(f"Cell 7 at index {target_idx} updated with Lipinski Ro5 filter")

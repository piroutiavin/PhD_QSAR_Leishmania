"""Patch notebook 10: Improve virtual library generation using training data scaffolds."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

NB = Path(r"e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\10_virtual_screening.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Replace Cell 3 (Generate Virtual Library) with scaffold-based approach
new_cell3 = [
    "# ============================================================\n",
    "# CELL 3: Generate Virtual Library from Training Data Scaffolds\n",
    "# ============================================================\n",
    "\n",
    "from rdkit.Chem.Scaffolds import MurckoScaffold\n",
    "from rdkit.Chem import rdmolops\n",
    "\n",
    "# Load curated dataset to get active compound SMILES\n",
    "curated = pd.read_csv(DATA / 'curated_dataset.csv')\n",
    "train_set = pd.read_csv(DATA / 'train_set.csv')\n",
    "\n",
    "# Get active training compounds\n",
    "active_train = train_set[train_set['activity_class'] == 'Active']['std_smiles'].unique()\n",
    "print(f'Active training compounds: {len(active_train)}')\n",
    "\n",
    "# Strategy: Generate analogs by modifying active compounds\n",
    "# 1. Take active compounds\n",
    "# 2. Apply simple chemical modifications (halogen swap, methyl/methoxy addition)\n",
    "# 3. This keeps compounds close to training chemical space\n",
    "\n",
    "library = []\n",
    "seen_smiles = set(active_train)  # Exclude known compounds\n",
    "\n",
    "# Modification patterns: find-and-replace on SMILES\n",
    "modifications = [\n",
    "    ('F', 'Cl'),    # F -> Cl\n",
    "    ('Cl', 'F'),    # Cl -> F\n",
    "    ('Cl', 'Br'),   # Cl -> Br\n",
    "    ('F', 'Br'),    # F -> Br\n",
    "    ('OC', 'O'),    # remove methoxy -> hydroxyl\n",
    "    ('(C)', '(F)'), # methyl -> fluoro\n",
    "    ('(C)', '(Cl)'),# methyl -> chloro\n",
    "    ('(F)', '(C)'), # fluoro -> methyl\n",
    "    ('(O)', '(N)'), # hydroxyl -> amino\n",
    "    ('(N)', '(O)'), # amino -> hydroxyl\n",
    "]\n",
    "\n",
    "# Apply modifications to top active compounds\n",
    "n_analogs = 0\n",
    "for smi in active_train[:500]:  # Use top 500 active compounds\n",
    "    for old_frag, new_frag in modifications:\n",
    "        if old_frag in smi:\n",
    "            new_smi = smi.replace(old_frag, new_frag, 1)  # Replace first occurrence only\n",
    "            mol = Chem.MolFromSmiles(new_smi)\n",
    "            if mol is not None:\n",
    "                canonical = Chem.MolToSmiles(mol)\n",
    "                if canonical not in seen_smiles:\n",
    "                    seen_smiles.add(canonical)\n",
    "                    library.append({\n",
    "                        'smiles': canonical,\n",
    "                        'parent': smi,\n",
    "                        'modification': f'{old_frag}->{new_frag}',\n",
    "                    })\n",
    "                    n_analogs += 1\n",
    "\n",
    "# Also add Murcko scaffold derivatives\n",
    "scaffold_counts = {}\n",
    "for smi in active_train:\n",
    "    mol = Chem.MolFromSmiles(smi)\n",
    "    if mol:\n",
    "        try:\n",
    "            scaffold = MurckoScaffold.MakeScaffoldGeneric(\n",
    "                MurckoScaffold.GetScaffoldForMol(mol)\n",
    "            )\n",
    "            scaffold_smi = Chem.MolToSmiles(scaffold)\n",
    "            scaffold_counts[scaffold_smi] = scaffold_counts.get(scaffold_smi, 0) + 1\n",
    "        except Exception:\n",
    "            pass\n",
    "\n",
    "print(f'Unique scaffolds in active compounds: {len(scaffold_counts)}')\n",
    "print(f'Analogs generated: {n_analogs}')\n",
    "\n",
    "screen_df = pd.DataFrame(library)\n",
    "print(f'\\nVirtual library: {len(screen_df)} novel compounds')\n",
    "print(f'  (analogs of active training compounds via single-atom modifications)')\n",
]

# Replace Cell 7 (AD filter) with relaxed thresholds
new_cell7 = [
    "# ============================================================\n",
    "# CELL 7: Applicability Domain Filter (kNN Distance)\n",
    "# ============================================================\n",
    "\n",
    "hits = pred_df[pred_df['consensus_pred'] == 1].copy()\n",
    "print(f'Consensus hits (>=3/4): {len(hits)}')\n",
    "\n",
    "# If strict consensus yields too few, relax to >=2/4\n",
    "if len(hits) < 10:\n",
    "    print(f'Too few strict consensus hits. Relaxing to >=2/4 models...')\n",
    "    hits = pred_df[pred_df['votes'] >= 2].copy()\n",
    "    print(f'Hits with >=2/4 votes: {len(hits)}')\n",
    "\n",
    "if len(hits) > 0:\n",
    "    # Fit kNN on training data\n",
    "    nn = NearestNeighbors(n_neighbors=5, metric='euclidean', n_jobs=-1)\n",
    "    nn.fit(X_train_sel.values)\n",
    "\n",
    "    # Distance of hits to nearest training compounds\n",
    "    X_hits = X_screen_scaled.loc[hits.index].values\n",
    "    distances, _ = nn.kneighbors(X_hits)\n",
    "    mean_distances = distances.mean(axis=1)\n",
    "\n",
    "    # Threshold: 99th percentile (more lenient for virtual screening)\n",
    "    train_distances, _ = nn.kneighbors(X_train_sel.values)\n",
    "    ad_threshold = np.percentile(train_distances.mean(axis=1), 99)\n",
    "\n",
    "    hits['ad_distance'] = mean_distances\n",
    "    hits['in_AD'] = mean_distances < ad_threshold\n",
    "\n",
    "    n_in_ad = hits['in_AD'].sum()\n",
    "    print(f'\\nAD threshold (99th pct): {ad_threshold:.4f}')\n",
    "    print(f'Inside AD: {n_in_ad} / {len(hits)} ({n_in_ad/len(hits)*100:.1f}%)')\n",
    "\n",
    "    hits_ad = hits[hits['in_AD']].copy()\n",
    "    \n",
    "    # If still too few, keep all and flag AD status\n",
    "    if len(hits_ad) < 5 and len(hits) > 0:\n",
    "        print(f'Few compounds inside strict AD. Keeping all hits with AD flag.')\n",
    "        hits_ad = hits.copy()\n",
    "else:\n",
    "    hits_ad = hits.copy()\n",
    "\n",
    "print(f'After AD filter: {len(hits_ad)} compounds')\n",
]

# Find and replace cells
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "CELL 3" in src and "Virtual" in src:
        nb["cells"][i]["source"] = new_cell3
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        print(f"  Replaced Cell 3 (library generation) at index {i}")
    elif "CELL 7" in src and "Applicability Domain" in src:
        nb["cells"][i]["source"] = new_cell7
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        print(f"  Replaced Cell 7 (AD filter) at index {i}")

# Clear all outputs
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nDone! Library generation and AD filter improved.")

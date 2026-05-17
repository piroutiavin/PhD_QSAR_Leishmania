"""Patch notebook 03: Change ECFP4 (radius=2, 2048 bits) to ECFP6 (radius=3, 4096 bits)."""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

NB = Path(r'e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\03_descriptor_calculation.ipynb')

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    # Find ECFP4 cell (Cell 5)
    if 'ECFP4' in src and 'GetMorganFingerprintAsBitVect' in src:
        new_source = [
            '# ============================================================\n',
            '# CELL 5: ECFP6 Fingerprints (4096 bits)\n',
            '# ============================================================\n',
            '\n',
            'print("Computing ECFP6 fingerprints (radius=3, nBits=4096)...")\n',
            '\n',
            'ecfp6_rows = {}\n',
            'for smi, mol in mol_dict.items():\n',
            '    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096)\n',
            '    ecfp6_rows[smi] = list(fp)\n',
            '\n',
            'ecfp6_cols = [f"ECFP6_{i}" for i in range(4096)]\n',
            'ecfp6_df = pd.DataFrame.from_dict(ecfp6_rows, orient=\'index\', columns=ecfp6_cols)\n',
            '\n',
            'print(f"ECFP6 bits computed : {ecfp6_df.shape[1]}")\n',
            'print(f"Mean bit density    : {ecfp6_df.mean().mean()*100:.1f}%")\n',
        ]
        nb['cells'][i]['source'] = new_source
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        changes += 1
        print(f"  Updated Cell 5 (ECFP4 -> ECFP6) at index {i}")

    # Find the combine cell (Cell 7) — update variable name
    if 'CELL 7' in src and 'Combine All' in src and 'ecfp4_df' in src:
        new_src = src.replace('ecfp4_df', 'ecfp6_df')
        new_src = new_src.replace('ECFP4', 'ECFP6')
        nb['cells'][i]['source'] = new_src if isinstance(cell['source'], str) else [new_src]
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        changes += 1
        print(f"  Updated Cell 7 (combine) at index {i}")

    # Find the cleaning cell — update ECFP4 references
    if 'CELL 8' in src and 'Clean Descriptor' in src and 'ECFP4' in src:
        new_src = src.replace('ECFP4', 'ECFP6')
        nb['cells'][i]['source'] = new_src if isinstance(cell['source'], str) else [new_src]
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        changes += 1
        print(f"  Updated Cell 8 (cleaning) at index {i}")

    # Find the visualization cell — update ECFP4 references
    if 'CELL 12' in src and 'ECFP4' in src:
        new_src = src.replace('ECFP4', 'ECFP6')
        nb['cells'][i]['source'] = new_src if isinstance(cell['source'], str) else [new_src]
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        changes += 1
        print(f"  Updated Cell 12 (viz) at index {i}")

# Clear all remaining code cell outputs
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nDone! {changes} cells updated.")

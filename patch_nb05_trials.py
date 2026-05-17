"""Patch notebook 05: Increase Optuna trials RF=100, SVM=50."""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

NB = Path(r'e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\05_model_training_RF_SVM.ipynb')

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src_list = cell['source'] if isinstance(cell['source'], list) else [cell['source']]
    new_src = []
    for line in src_list:
        if 'study_rf.optimize' in line and 'n_trials=30' in line:
            line = line.replace('n_trials=30', 'n_trials=100')
            line = line.replace('timeout=1500', 'timeout=5400')
            changes += 1
            print(f"  RF: 30 -> 100 trials at cell {i}")
        elif 'study_svm.optimize' in line and 'n_trials=20' in line:
            line = line.replace('n_trials=20', 'n_trials=50')
            line = line.replace('timeout=900', 'timeout=3600')
            changes += 1
            print(f"  SVM: 20 -> 50 trials at cell {i}")
        new_src.append(line)
    nb['cells'][i]['source'] = new_src

# Clear all outputs
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nDone! {changes} trial counts updated in nb05.")

"""Patch notebook 04: Add Boruta feature selection alongside MI∩RF."""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

NB = Path(r'e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\04_feature_selection.ipynb')

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update imports cell to include Boruta
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'CELL 1' in src and 'Imports' in src:
        if 'boruta' not in src.lower():
            nb['cells'][i]['source'].insert(-2, "from boruta import BorutaPy\n")
            print(f"  Added Boruta import to Cell 1 at index {i}")
        break

# Find Cell 6 (Dual-Filter Intersection) and replace with Boruta + MI∩RF union
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'CELL 6' in src and 'Dual-Filter' in src:
        new_source = [
            '# ============================================================\n',
            '# CELL 6: Boruta Feature Selection + MI∩RF Union\n',
            '# ============================================================\n',
            '\n',
            '# --- Boruta wrapper method ---\n',
            'print("Running Boruta feature selection (this may take 10-20 min)...")\n',
            'print("  Boruta creates shadow features and compares real vs random importance.")\n',
            '\n',
            'rf_boruta = RandomForestClassifier(\n',
            '    n_estimators=200, max_depth=7, random_state=SEED, n_jobs=-1\n',
            ')\n',
            '\n',
            'boruta = BorutaPy(\n',
            '    estimator=rf_boruta,\n',
            '    n_estimators="auto",\n',
            '    max_iter=100,\n',
            '    random_state=SEED,\n',
            '    verbose=0\n',
            ')\n',
            '\n',
            'boruta.fit(X_train_scaled.values, y_train.values)\n',
            '\n',
            'boruta_confirmed = X_train_scaled.columns[boruta.support_].tolist()\n',
            'boruta_tentative = X_train_scaled.columns[boruta.support_weak_].tolist()\n',
            '\n',
            'print(f"Boruta confirmed  : {len(boruta_confirmed)} features")\n',
            'print(f"Boruta tentative  : {len(boruta_tentative)} features")\n',
            'print(f"Boruta rejected   : {(~boruta.support_ & ~boruta.support_weak_).sum()} features")\n',
            '\n',
            '# --- Combine: Boruta confirmed+tentative ∪ MI∩RF ---\n',
            'mi_rf_features = sorted(set(mi_selected) & set(rf_selected))\n',
            'boruta_all = set(boruta_confirmed + boruta_tentative)\n',
            '\n',
            'final_features = sorted(boruta_all | set(mi_rf_features))\n',
            '\n',
            'print(f"\\nMI∩RF features    : {len(mi_rf_features)}")\n',
            'print(f"Boruta features   : {len(boruta_all)}")\n',
            'print(f"Union (final)     : {len(final_features)}")\n',
            'print(f"Overlap           : {len(boruta_all & set(mi_rf_features))}")\n',
            'print(f"Boruta-only       : {len(boruta_all - set(mi_rf_features))}")\n',
            'print(f"MI∩RF-only        : {len(set(mi_rf_features) - boruta_all)}")\n',
            '\n',
            'print(f"\\nRetention rate: {len(final_features)/X_train_scaled.shape[1]*100:.1f}% of original features")\n',
            '\n',
            'X_train_selected = X_train_scaled[final_features].copy()\n',
            'print(f"\\nX_train_selected shape: {X_train_selected.shape}")\n',
        ]
        nb['cells'][i]['source'] = new_source
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        print(f"  Replaced Cell 6 with Boruta+MI∩RF union at index {i}")
        break

# Update markdown header
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if cell['cell_type'] == 'markdown' and 'Notebook 04' in src and 'Feature Selection' in src:
        nb['cells'][i]['source'] = [
            "# Notebook 04 — Feature Selection\n",
            "\n",
            "**Project**: ML-Based QSAR Modeling for Anti-Leishmanial Sulfonamide Derivatives  \n",
            "**Input**: `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`  \n",
            "**Output**:\n",
            "- `data/processed/X_train_selected.csv` — Training descriptors after feature selection\n",
            "- `data/processed/X_test_selected.csv` — Test descriptors (same features, no leakage)\n",
            "- `data/processed/selected_features.csv` — Names of selected features\n",
            "- `models/scaler.joblib` — Fitted StandardScaler\n",
            "\n",
            "**Pipeline** (training set only — applied to test set after):\n",
            "1. StandardScaler normalization\n",
            "2. Mutual Information ranking → keep top 75% (MI > 25th percentile)\n",
            "3. Random Forest importance ranking → keep top 75%\n",
            "4. Boruta wrapper feature selection (shadow feature comparison)\n",
            "5. Final features = Boruta ∪ (MI∩RF intersection)\n",
            "\n",
            "**QC-4**: Zero data leakage — scaler and feature selector fitted on train only.\n",
            "\n",
            "---"
        ]
        print(f"  Updated markdown header at index {i}")
        break

# Clear all cell outputs
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nDone! Notebook 04 patched with Boruta feature selection.")

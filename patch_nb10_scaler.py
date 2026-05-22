"""Patch notebook 10: Fix scaler to use full feature set before selection."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

NB = Path(r"e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\10_virtual_screening.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Replace Cell 5 (Scale and Select Features)
new_cell5 = [
    "# ============================================================\n",
    "# CELL 5: Scale and Select Features (Match Training Pipeline)\n",
    "# ============================================================\n",
    "\n",
    "# The scaler was fitted on ALL descriptors (2111 features) in nb04.\n",
    "# We must scale using ALL features first, THEN select the 1163.\n",
    "\n",
    "# Get the full feature list the scaler was fitted on\n",
    "scaler_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None\n",
    "\n",
    "if scaler_features is not None:\n",
    "    print(f'Scaler was fitted on {len(scaler_features)} features')\n",
    "    # Create full descriptor matrix matching scaler features\n",
    "    X_full = pd.DataFrame(0.0, index=all_desc.index, columns=scaler_features)\n",
    "    common_cols = [c for c in scaler_features if c in all_desc.columns]\n",
    "    X_full[common_cols] = all_desc[common_cols].values\n",
    "    print(f'  Matched {len(common_cols)} / {len(scaler_features)} features from screening descriptors')\n",
    "    \n",
    "    # Scale using the full scaler\n",
    "    X_full_scaled = pd.DataFrame(\n",
    "        scaler.transform(X_full), columns=scaler_features, index=X_full.index\n",
    "    )\n",
    "    \n",
    "    # Now select only the 1163 features used in modeling\n",
    "    available_selected = [f for f in features if f in X_full_scaled.columns]\n",
    "    X_screen_scaled = X_full_scaled[available_selected].copy()\n",
    "else:\n",
    "    # Fallback: scaler has no feature names (numpy-fitted)\n",
    "    print('Scaler has no feature names, using numpy arrays')\n",
    "    descriptor_names = pd.read_csv(DATA / 'descriptor_names.csv')['feature'].tolist()\n",
    "    X_full = pd.DataFrame(0.0, index=all_desc.index, columns=descriptor_names)\n",
    "    common_cols = [c for c in descriptor_names if c in all_desc.columns]\n",
    "    X_full[common_cols] = all_desc[common_cols].values\n",
    "    print(f'  Matched {len(common_cols)} / {len(descriptor_names)} features')\n",
    "    \n",
    "    X_full_scaled_np = scaler.transform(X_full.values)\n",
    "    X_full_scaled = pd.DataFrame(X_full_scaled_np, columns=descriptor_names, index=X_full.index)\n",
    "    X_screen_scaled = X_full_scaled[features].copy()\n",
    "\n",
    "print(f'\\nX_screen_scaled shape: {X_screen_scaled.shape}')\n",
    "print(f'NaN check: {X_screen_scaled.isna().sum().sum()} NaNs')\n",
]

# Find and replace Cell 5
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "CELL 5" in src and "Scale and Select" in src:
        nb["cells"][i]["source"] = new_cell5
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        print(f"  Replaced Cell 5 at index {i}")
        break

# Clear all outputs
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done! Scaler pipeline fixed.")

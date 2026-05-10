"""Regenerate X_train_selected.csv from X_train.csv + scaler + selected_features."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(r'e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania')
DATA = ROOT / 'data' / 'processed'

print("Loading X_train.csv (102 MB) …")
X_train = pd.read_csv(DATA / 'X_train.csv', index_col=0)
print(f"  X_train shape: {X_train.shape}")

print("Loading scaler …")
scaler = joblib.load(ROOT / 'models' / 'scaler.joblib')

print("Scaling X_train …")
X_scaled = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

print("Loading selected features …")
features = pd.read_csv(DATA / 'selected_features.csv')['feature'].tolist()
print(f"  {len(features)} features selected")

# Verify all features exist in X_scaled
missing = [f for f in features if f not in X_scaled.columns]
if missing:
    print(f"  WARNING: {len(missing)} features missing from X_train!")
    print(f"  First few: {missing[:5]}")
else:
    print("  All selected features present in X_train — OK")

X_train_selected = X_scaled[features].copy()
print(f"  X_train_selected shape: {X_train_selected.shape}")

print("Saving X_train_selected.csv …")
X_train_selected.to_csv(DATA / 'X_train_selected.csv')
print("Done.")

# Verify
check = pd.read_csv(DATA / 'X_train_selected.csv', index_col=0)
print(f"Verification read-back shape: {check.shape}")
assert check.shape == (9563, 943), f"Unexpected shape: {check.shape}"
print("Shape confirmed: (9563, 943) ✓")

"""Check Lipinski Rule of Five compliance for all curated compounds."""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_csv("data/processed/curated_dataset.csv")
print(f"Total curated compounds: {len(df):,}")
print()

results = []
for _, row in df.iterrows():
    mol = Chem.MolFromSmiles(str(row["std_smiles"]))
    if mol is None:
        results.append({"MW": None, "LogP": None, "HBD": None, "HBA": None, "violations": None})
        continue
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    results.append({"MW": mw, "LogP": logp, "HBD": hbd, "HBA": hba, "violations": violations})

props = pd.DataFrame(results)

print("=== Lipinski Properties Summary ===")
print(f"  MW  > 500:  {(props['MW'] > 500).sum():,}  ({(props['MW'] > 500).mean()*100:.1f}%)")
print(f"  LogP > 5:   {(props['LogP'] > 5).sum():,}  ({(props['LogP'] > 5).mean()*100:.1f}%)")
print(f"  HBD > 5:    {(props['HBD'] > 5).sum():,}  ({(props['HBD'] > 5).mean()*100:.1f}%)")
print(f"  HBA > 10:   {(props['HBA'] > 10).sum():,}  ({(props['HBA'] > 10).mean()*100:.1f}%)")
print()
print("=== Number of Violations ===")
for v in range(5):
    n = (props["violations"] == v).sum()
    pct = n / len(props) * 100
    print(f"  {v} violations: {n:,} compounds ({pct:.1f}%)")

strict = props["violations"] < 2
print(f"\n=== Impact of Filtering ===")
print(f"  Keep (0-1 violations): {strict.sum():,} ({strict.mean()*100:.1f}%)")
print(f"  Remove (2+ violations): {(~strict).sum():,} ({(~strict).mean()*100:.1f}%)")

df["violations"] = props["violations"]
kept = df[df["violations"] < 2]
removed = df[df["violations"] >= 2]
print(f"\n=== Activity in KEPT compounds ===")
print(f"  Active:   {(kept['activity_class']=='Active').sum():,} ({(kept['activity_class']=='Active').mean()*100:.1f}%)")
print(f"  Inactive: {(kept['activity_class']=='Inactive').sum():,} ({(kept['activity_class']=='Inactive').mean()*100:.1f}%)")
print(f"\n=== Activity in REMOVED compounds ===")
if len(removed) > 0:
    print(f"  Active:   {(removed['activity_class']=='Active').sum():,} ({(removed['activity_class']=='Active').mean()*100:.1f}%)")
    print(f"  Inactive: {(removed['activity_class']=='Inactive').sum():,} ({(removed['activity_class']=='Inactive').mean()*100:.1f}%)")
else:
    print("  (no compounds removed)")
print(f"\n=== Mean pIC50 ===")
print(f"  Kept compounds:    {kept['pIC50'].mean():.2f}")
if len(removed) > 0:
    print(f"  Removed compounds: {removed['pIC50'].mean():.2f}")

print(f"\n=== Per-target impact ===")
for target in df["target_name"].unique():
    sub = df[df["target_name"] == target]
    n_total = len(sub)
    n_remove = (sub["violations"] >= 2).sum()
    print(f"  {target}: {n_total:,} total, {n_remove:,} would be removed ({n_remove/n_total*100:.1f}%)")

"""
Phase 2.2a — Mordred 2D descriptor computation for the sulfonamide panel
========================================================================
Computes continuous Mordred 2D descriptors (interpretable, SHAP-friendly) for
every unique molecule in the panel, cleans them, and saves a single descriptor
matrix keyed by standardized SMILES. Per-organism feature *reduction* is done
separately in feature_selection.py.

Run:  .venv/Scripts/python.exe -m src.descriptors
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from mordred import Calculator, descriptors

RDLogger.DisableLog("rdApp.*")

PANEL_DIR = Path("data/processed/sulfonamide_panel")
OUT = PANEL_DIR / "mordred_descriptors.csv"

MAX_NA_FRAC = 0.10   # drop descriptor columns missing in >10% of molecules


def unique_panel_smiles() -> list[str]:
    smis = set()
    for f in sorted(PANEL_DIR.glob("sulfonamide_*.csv")):
        smis.update(pd.read_csv(f)["std_smiles"].tolist())
    return sorted(smis)


def compute(smiles_list: list[str]) -> pd.DataFrame:
    calc = Calculator(descriptors, ignore_3D=True)   # ~1600 2D descriptors
    mols, valid = [], []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m); valid.append(smi)
    print(f"Computing {len(calc.descriptors)} Mordred 2D descriptors "
          f"for {len(mols)} molecules ...")
    df = calc.pandas(mols, nproc=1)
    # Mordred returns error objects for failures -> coerce to numeric NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    df.insert(0, "std_smiles", valid)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    key = df["std_smiles"]
    X = df.drop(columns=["std_smiles"])
    n0 = X.shape[1]

    # drop columns too often missing, then near-constant, then impute remainder
    X = X.loc[:, X.isna().mean() <= MAX_NA_FRAC]
    n_na = X.shape[1]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    n_const = X.shape[1]
    X = X.fillna(X.median(numeric_only=True))

    print(f"  descriptors: {n0} -> {n_na} (NA filter) -> {n_const} (constant filter)")
    out = pd.concat([key.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    return out


if __name__ == "__main__":
    smis = unique_panel_smiles()
    print(f"Unique panel molecules: {len(smis)}")
    raw = compute(smis)
    cleaned = clean(raw)
    cleaned.to_csv(OUT, index=False)
    print(f"Saved {cleaned.shape[0]} x {cleaned.shape[1]-1} descriptor matrix -> {OUT}")

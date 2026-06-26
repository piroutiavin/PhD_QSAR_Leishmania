"""
Phase 1 — Build the Sulfonamide Per-Organism QSAR Panel
=======================================================
Rebuilds the dataset from raw ChEMBL pulls into 4 leakage-aware,
per-organism sulfonamide datasets, applying the curation fixes agreed in
ENHANCEMENT_TODO.md:

  1.1  Sulfonamide SMARTS filter            S(=O)(=O)N
  1.4  Tautomer canonicalization, prefer pchembl_value, tag standard_type
  1.3  Per-organism dedup (concordant->median, discordant->drop), classify
  1.2  Partition by organism into 4 independent datasets + Murcko scaffold
  1.5  Report per-organism counts

Run:  python -m src.build_sulfonamide_panel
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("phase1")

RAW = Path("data/raw/chembl_all_targets_raw.csv")
OUT_DIR = Path("data/processed/sulfonamide_panel")

SULFONAMIDE_SMARTS = "S(=O)(=O)N"          # primary/secondary/tertiary sulfonamide
ACTIVITY_THRESHOLD = 5.0                    # pIC50 >= 5.0  -> Active (IC50 <= 10 uM)
DISCORDANT_LOG = 1.0                        # >1 log-unit spread among replicates -> drop
MAX_MW = 900.0

ORGANISMS = {
    "L_infantum": "Leishmania infantum",
    "L_donovani": "Leishmania donovani",
    "T_cruzi": "Trypanosoma cruzi",
    "L_amazonensis": "Leishmania amazonensis",
}

# ── lazy-built standardizers (reused across calls) ──────────────────
_LFC = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()
_NORMALIZER = rdMolStandardize.Normalizer()
_TAUTOMER = rdMolStandardize.TautomerEnumerator()
_SULFO_PATT = Chem.MolFromSmarts(SULFONAMIDE_SMARTS)


def standardize_smiles(smiles: str) -> str | None:
    """Salt-strip -> neutralize -> normalize -> canonical tautomer -> canonical SMILES."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        mol = _LFC.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        mol = _NORMALIZER.normalize(mol)
        mol = _TAUTOMER.Canonicalize(mol)          # <-- the 1.4 fix (was missing before)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def is_sulfonamide(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    return mol is not None and mol.HasSubstructMatch(_SULFO_PATT)


def passes_druglikeness(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    if Descriptors.MolWt(mol) > MAX_MW:
        return False
    metals = {"Fe", "Cu", "Zn", "Mn", "Co", "Ni", "Pt", "Pd", "Ru",
              "Rh", "Ir", "Os", "Au", "Ag", "Hg", "Cd", "Cr", "Mo", "W"}
    return not ({a.GetSymbol() for a in mol.GetAtoms()} & metals)


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def to_pic50(row: pd.Series) -> float | None:
    """Prefer ChEMBL pchembl_value; else derive from standard_value (nM)."""
    pchembl = row.get("pchembl_value")
    if pd.notna(pchembl):
        return float(pchembl)
    val = row.get("standard_value")
    if pd.isna(val) or float(val) <= 0:
        return None
    return -np.log10(float(val) * 1e-9)          # raw is all nM (verified)


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading raw: %s", RAW)
    df = pd.read_csv(RAW)
    logger.info("  raw rows: %d", len(df))

    # 1.1 — cheap pre-filter on raw canonical_smiles, then expensive standardize on subset
    df = df[df["canonical_smiles"].notna()].copy()
    logger.info("Pre-filtering sulfonamides on raw SMILES ...")
    df = df[df["canonical_smiles"].apply(is_sulfonamide)].copy()
    logger.info("  sulfonamide rows (pre-standardization): %d", len(df))

    # 1.4 — standardize (incl. tautomer canonicalization)
    logger.info("Standardizing %d SMILES (with tautomer canonicalization) ...", len(df))
    df["std_smiles"] = df["canonical_smiles"].apply(standardize_smiles)
    df = df.dropna(subset=["std_smiles"])
    # re-confirm sulfonamide survives standardization
    df = df[df["std_smiles"].apply(is_sulfonamide)].copy()
    logger.info("  after standardization + re-check: %d rows", len(df))

    # validity / activity value sanity + drug-likeness
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["std_smiles"].apply(passes_druglikeness)].copy()

    # 1.4 — pIC50 preferring pchembl_value; keep standard_type tag (IC50/EC50)
    df["pIC50"] = df.apply(to_pic50, axis=1)
    df = df.dropna(subset=["pIC50"])
    logger.info("  rows with valid pIC50: %d", len(df))

    summary_rows = []
    for key, organism in ORGANISMS.items():
        sub = df[df["target_organism"] == organism].copy()

        # 1.3 — per-organism dedup: concordant -> median pIC50, discordant -> drop
        clean, n_disc, n_merged = [], 0, 0
        for smi, grp in sub.groupby("std_smiles"):
            if len(grp) == 1:
                row = grp.iloc[0].copy()
            else:
                spread = grp["pIC50"].max() - grp["pIC50"].min()
                if spread >= DISCORDANT_LOG:
                    n_disc += 1
                    continue
                row = grp.iloc[0].copy()
                row["pIC50"] = grp["pIC50"].median()
                # record endpoint provenance for the merged record
                row["standard_type"] = "/".join(sorted(grp["standard_type"].unique()))
                n_merged += 1
            clean.append(row)

        org_df = pd.DataFrame(clean).reset_index(drop=True)

        # classify + scaffold
        org_df["activity_class"] = np.where(
            org_df["pIC50"] >= ACTIVITY_THRESHOLD, "Active", "Inactive")
        org_df["scaffold"] = org_df["std_smiles"].apply(murcko_scaffold)
        org_df["organism_key"] = key

        keep_cols = ["molecule_chembl_id", "std_smiles", "canonical_smiles",
                     "pIC50", "activity_class", "standard_type",
                     "target_organism", "organism_key", "scaffold",
                     "assay_chembl_id", "assay_description"]
        org_df = org_df[[c for c in keep_cols if c in org_df.columns]]

        out_path = OUT_DIR / f"sulfonamide_{key}.csv"
        org_df.to_csv(out_path, index=False)

        n = len(org_df)
        n_act = int((org_df["activity_class"] == "Active").sum())
        n_ec50 = int(org_df["standard_type"].str.contains("EC50").sum())
        summary_rows.append({
            "organism": organism,
            "key": key,
            "molecules": n,
            "active": n_act,
            "inactive": n - n_act,
            "active_pct": round(100 * n_act / n, 1) if n else 0,
            "scaffolds": org_df["scaffold"].nunique(),
            "ec50_records": n_ec50,
            "discordant_dropped": n_disc,
            "concordant_merged": n_merged,
        })
        logger.info("  %-22s n=%d  active=%d  scaffolds=%d  EC50=%d  (dropped %d discordant)",
                    organism, n, n_act, org_df["scaffold"].nunique(), n_ec50, n_disc)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "panel_summary.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))
    logger.info("Saved 4 datasets + panel_summary.csv to %s", OUT_DIR)


if __name__ == "__main__":
    build()

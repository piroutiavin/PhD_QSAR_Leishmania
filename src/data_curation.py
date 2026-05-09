"""
Data Curation Module
====================
Standardization, duplicate handling, activity conversion, and
drug-likeness filtering following Fourches et al. (2010, 2016) guidelines.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def standardize_smiles(smiles: str) -> str | None:
    """
    Standardize a SMILES string:
    1. Parse SMILES → molecule object
    2. Remove salts/counterions (keep largest fragment)
    3. Neutralize charges
    4. Normalize tautomers
    5. Return canonical SMILES
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        # Remove salts — keep largest fragment
        remover = rdMolStandardize.LargestFragmentChooser()
        mol = remover.choose(mol)

        # Neutralize charges
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # Normalize functional groups
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)

        return Chem.MolToSmiles(mol, canonical=True)

    except Exception:
        return None


def standardize_dataset(df: pd.DataFrame,
                        smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    """
    Apply SMILES standardization to entire dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    smiles_col : str
        Column containing SMILES strings

    Returns
    -------
    pd.DataFrame
        DataFrame with 'std_smiles' column added, invalid entries removed
    """
    logger.info(f"Standardizing {len(df)} SMILES...")
    df = df.copy()
    df["std_smiles"] = df[smiles_col].apply(standardize_smiles)

    n_before = len(df)
    df = df.dropna(subset=["std_smiles"])
    n_after = len(df)
    logger.info(f"  Valid after standardization: {n_after}/{n_before} "
                f"({n_before - n_after} removed)")
    return df


def remove_invalid_activities(df: pd.DataFrame,
                               value_col: str = "standard_value") -> pd.DataFrame:
    """Remove entries with missing, zero, or negative activity values."""
    logger.info("Removing invalid activity entries...")
    n_before = len(df)
    df = df.copy()
    df = df[df[value_col].notna()]
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[df[value_col] > 0]
    logger.info(f"  Valid activities: {len(df)}/{n_before}")
    return df


def handle_duplicates(df: pd.DataFrame,
                      smiles_col: str = "std_smiles",
                      target_col: str = "target_name",
                      value_col: str = "standard_value",
                      log_threshold: float = 1.0) -> pd.DataFrame:
    """
    Handle duplicate measurements for the same compound–target pair.

    Concordant duplicates (spread < log_threshold): keep median.
    Discordant duplicates (spread >= log_threshold): remove all entries.

    Parameters
    ----------
    df : pd.DataFrame
    smiles_col, target_col, value_col : str
    log_threshold : float
        Maximum allowed spread in log10 units (default: 1.0)

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Handling duplicate measurements...")
    grouped = df.groupby([smiles_col, target_col])

    clean_rows = []
    removed_discordant = 0
    merged_concordant = 0

    for (smi, target), group in grouped:
        if len(group) == 1:
            clean_rows.append(group.iloc[0])
        else:
            log_values = np.log10(group[value_col].values.astype(float))
            spread = log_values.max() - log_values.min()

            if spread < log_threshold:
                # Concordant: keep median
                median_row = group.iloc[0].copy()
                median_row[value_col] = group[value_col].median()
                clean_rows.append(median_row)
                merged_concordant += len(group) - 1
            else:
                # Discordant: remove all
                removed_discordant += len(group)

    result = pd.DataFrame(clean_rows).reset_index(drop=True)
    logger.info(f"  Merged {merged_concordant} concordant duplicates")
    logger.info(f"  Removed {removed_discordant} discordant entries")
    logger.info(f"  Final count: {len(result)}")
    return result


def convert_ic50_to_pic50(df: pd.DataFrame,
                           value_col: str = "standard_value",
                           units_col: str = "standard_units") -> pd.DataFrame:
    """
    Convert IC50 values to pIC50 = -log10(IC50 in M).
    Handles nM and μM units.
    """
    logger.info("Converting IC50 to pIC50...")
    df = df.copy()

    def to_pic50(row):
        val = float(row[value_col])
        units = str(row.get(units_col, "nM"))
        if units == "nM":
            return -np.log10(val * 1e-9)
        elif units in ("uM", "μM"):
            return -np.log10(val * 1e-6)
        else:
            return -np.log10(val * 1e-9)  # Default to nM

    df["pIC50"] = df.apply(to_pic50, axis=1)
    logger.info(f"  pIC50 range: {df['pIC50'].min():.2f} – {df['pIC50'].max():.2f}")
    logger.info(f"  pIC50 median: {df['pIC50'].median():.2f}")
    return df


def classify_activity(df: pd.DataFrame,
                      pic50_col: str = "pIC50",
                      threshold: float = 5.0) -> pd.DataFrame:
    """
    Classify compounds as Active (pIC50 >= threshold) or Inactive.
    Default threshold: 5.0 (IC50 <= 10 μM).
    """
    logger.info(f"Classifying activity (threshold pIC50 = {threshold})...")
    df = df.copy()
    df["activity_class"] = df[pic50_col].apply(
        lambda x: "Active" if x >= threshold else "Inactive"
    )

    n_active = (df["activity_class"] == "Active").sum()
    n_inactive = (df["activity_class"] == "Inactive").sum()
    ratio = n_active / len(df) * 100

    logger.info(f"  Active: {n_active} ({ratio:.1f}%)")
    logger.info(f"  Inactive: {n_inactive} ({100-ratio:.1f}%)")
    return df


def apply_druglikeness_filter(df: pd.DataFrame,
                               smiles_col: str = "std_smiles",
                               max_mw: float = 900.0) -> pd.DataFrame:
    """
    Remove non-drug-like compounds:
    - MW > max_mw
    - Organometallics (containing transition metals)
    - Invalid structures
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    logger.info("Applying drug-likeness filter...")

    metals = {"Fe", "Cu", "Zn", "Mn", "Co", "Ni", "Pt", "Pd", "Ru",
              "Rh", "Ir", "Os", "Au", "Ag", "Hg", "Cd", "Cr", "Mo", "W"}

    keep_mask = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row[smiles_col]))
        if mol is None:
            keep_mask.append(False)
            continue

        mw = Descriptors.MolWt(mol)
        if mw > max_mw:
            keep_mask.append(False)
            continue

        atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        if atom_symbols & metals:
            keep_mask.append(False)
            continue

        keep_mask.append(True)

    result = df[keep_mask].reset_index(drop=True)
    logger.info(f"  After filter: {len(result)}/{len(df)} compounds retained")
    return result


def split_dataset(df: pd.DataFrame,
                  test_size: float = 0.30,
                  random_state: int = 42,
                  stratify_col: str = "activity_class") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into training and test sets with stratification.
    """
    from sklearn.model_selection import train_test_split

    logger.info(f"Splitting dataset (test_size={test_size})...")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df[stratify_col]
    )

    logger.info(f"  Training set: {len(train_df)} compounds")
    logger.info(f"  Test set: {len(test_df)} compounds")
    logger.info(f"  Train active ratio: "
                f"{(train_df[stratify_col] == 'Active').mean():.1%}")
    logger.info(f"  Test active ratio: "
                f"{(test_df[stratify_col] == 'Active').mean():.1%}")
    return train_df, test_df


def get_murcko_scaffold(smiles: str) -> str:
    """Extract Murcko scaffold from a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def run_full_curation_pipeline(raw_df: pd.DataFrame,
                                save_dir: str = "data/processed") -> dict:
    """
    Run the complete curation pipeline end-to-end.

    Returns dict with curated DataFrames and summary statistics.
    """
    from pathlib import Path
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING FULL CURATION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Standardize SMILES
    df = standardize_dataset(raw_df)

    # Step 2: Remove invalid activities
    df = remove_invalid_activities(df)

    # Step 3: Handle duplicates
    df = handle_duplicates(df)

    # Step 4: Convert to pIC50
    df = convert_ic50_to_pic50(df)

    # Step 5: Classify activity
    df = classify_activity(df)

    # Step 6: Drug-likeness filter
    df = apply_druglikeness_filter(df)

    # Step 7: Add scaffolds
    logger.info("Computing Murcko scaffolds...")
    df["scaffold"] = df["std_smiles"].apply(get_murcko_scaffold)

    # Step 8: Split
    train_df, test_df = split_dataset(df)

    # Save outputs
    df.to_csv(save_path / "curated_dataset.csv", index=False)
    train_df.to_csv(save_path / "train_set.csv", index=False)
    test_df.to_csv(save_path / "test_set.csv", index=False)

    # Sulfonamide subset
    if "is_sulfonamide" in df.columns:
        sulfo_df = df[df["is_sulfonamide"]].copy()
        sulfo_df.to_csv(save_path / "sulfonamide_subset.csv", index=False)
        logger.info(f"Sulfonamide subset saved: {len(sulfo_df)} compounds")

    logger.info("=" * 60)
    logger.info("CURATION COMPLETE")
    logger.info(f"Final curated dataset: {len(df)} compounds")
    logger.info("=" * 60)

    return {
        "curated": df,
        "train": train_df,
        "test": test_df,
        "sulfonamide_subset": sulfo_df if "is_sulfonamide" in df.columns else None,
    }

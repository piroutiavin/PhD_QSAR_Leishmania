"""
Data Collection Module
======================
Functions for querying ChEMBL and PubChem databases to retrieve
bioactivity data for anti-kinetoplastid compounds.

Targets:
    CHEMBL612848 — L. infantum
    CHEMBL367    — L. donovani
    CHEMBL612877 — L. amazonensis
    CHEMBL368    — T. cruzi (whole cell)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ─── Target Definitions ─────────────────────────────────────────
TARGETS = {
    "L_infantum": {
        "chembl_id": "CHEMBL612848",
        "organism": "Leishmania infantum",
        "priority": "primary",
    },
    "L_donovani": {
        "chembl_id": "CHEMBL367",
        "organism": "Leishmania donovani",
        "priority": "cross-species",
    },
    "L_amazonensis": {
        "chembl_id": "CHEMBL612877",
        "organism": "Leishmania amazonensis",
        "priority": "cross-species",
    },
    "T_cruzi": {
        "chembl_id": "CHEMBL368",
        "organism": "Trypanosoma cruzi",
        "priority": "cross-kinetoplastid",
    },
}

# Fields to retrieve from ChEMBL
ACTIVITY_FIELDS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_value",
    "standard_units",
    "standard_relation",
    "pchembl_value",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "document_chembl_id",
    "target_chembl_id",
]


def fetch_chembl_bioactivity(target_id: str, target_name: str,
                              activity_types: list = None,
                              max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch bioactivity data from ChEMBL for a given target.

    Parameters
    ----------
    target_id : str
        ChEMBL target ID (e.g., 'CHEMBL612848')
    target_name : str
        Human-readable name (e.g., 'L_infantum')
    activity_types : list, optional
        Activity types to retrieve. Default: ['IC50', 'EC50']
    max_retries : int
        Number of retry attempts for API failures

    Returns
    -------
    pd.DataFrame
        DataFrame with bioactivity records
    """
    from chembl_webresource_client.new_client import new_client

    if activity_types is None:
        activity_types = ["IC50", "EC50"]

    activity_api = new_client.activity

    logger.info(f"Fetching data for {target_name} ({target_id})...")

    for attempt in range(max_retries):
        try:
            # Query activities
            activities = activity_api.filter(
                target_chembl_id=target_id,
                standard_type__in=activity_types,
                standard_units="nM",
                standard_relation="=",
            ).only(ACTIVITY_FIELDS)

            # Convert to DataFrame
            records = list(activities)
            df = pd.DataFrame.from_records(records)

            if df.empty:
                logger.warning(f"No records found for {target_name}")
                return pd.DataFrame()

            # Add metadata columns
            df["target_name"] = target_name
            df["target_organism"] = TARGETS[target_name]["organism"]
            df["data_source"] = "ChEMBL"

            logger.info(f"  Retrieved {len(df)} activity records for {target_name}")
            return df

        except Exception as e:
            logger.warning(f"  Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"  All attempts failed for {target_name}")
                return pd.DataFrame()


def fetch_all_targets(targets: dict = None,
                      save_dir: str = "data/raw") -> pd.DataFrame:
    """
    Fetch bioactivity data for all defined targets and combine.

    Parameters
    ----------
    targets : dict, optional
        Target definitions. Default: uses TARGETS constant
    save_dir : str
        Directory to save raw CSV files

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all targets
    """
    if targets is None:
        targets = TARGETS

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_data = []
    summary = []

    for name, info in targets.items():
        df = fetch_chembl_bioactivity(info["chembl_id"], name)

        if not df.empty:
            # Save individual target file
            filepath = save_path / f"chembl_{name}_raw.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"  Saved to {filepath}")

            all_data.append(df)
            summary.append({
                "target": name,
                "chembl_id": info["chembl_id"],
                "records": len(df),
                "unique_compounds": df["molecule_chembl_id"].nunique(),
                "has_pchembl": df["pchembl_value"].notna().sum(),
            })

        # Be polite to the API
        time.sleep(2)

    if not all_data:
        logger.error("No data retrieved from any target!")
        return pd.DataFrame()

    # Combine all targets
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save combined file
    combined_path = save_path / "chembl_all_targets_raw.csv"
    combined_df.to_csv(combined_path, index=False)
    logger.info(f"\nCombined dataset saved: {combined_path}")

    # Print summary
    summary_df = pd.DataFrame(summary)
    logger.info(f"\n{'='*60}")
    logger.info("DATA COLLECTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"\n{summary_df.to_string(index=False)}")
    logger.info(f"\nTotal records: {len(combined_df)}")
    logger.info(f"Total unique compounds: {combined_df['molecule_chembl_id'].nunique()}")

    return combined_df


def filter_sulfonamides(df: pd.DataFrame,
                        smiles_col: str = "canonical_smiles") -> pd.DataFrame:
    """
    Filter compounds containing sulfonamide substructures using SMARTS.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with SMILES column
    smiles_col : str
        Name of the SMILES column

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with sulfonamide flag added
    """
    from rdkit import Chem

    # SMARTS patterns for sulfonamide moieties
    SULFONAMIDE_SMARTS = [
        "[S](=O)(=O)[NH]",     # Secondary sulfonamide (R-SO2-NH-R')
        "[S](=O)(=O)[NH2]",    # Primary sulfonamide (R-SO2-NH2)
        "[S](=O)(=O)N",        # Tertiary sulfonamide (R-SO2-NR'R'')
        "[NH]S(=O)(=O)",       # Reverse pattern
    ]

    compiled_patterns = []
    for smarts in SULFONAMIDE_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            compiled_patterns.append(pat)

    def has_sulfonamide(smiles):
        if pd.isna(smiles):
            return False
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return False
        for pattern in compiled_patterns:
            if mol.HasSubstructMatch(pattern):
                return True
        return False

    logger.info("Filtering for sulfonamide substructures...")
    df = df.copy()
    df["is_sulfonamide"] = df[smiles_col].apply(has_sulfonamide)

    n_sulfonamide = df["is_sulfonamide"].sum()
    n_total = len(df)
    logger.info(f"  Sulfonamide compounds: {n_sulfonamide} / {n_total} "
                f"({n_sulfonamide/n_total*100:.1f}%)")

    return df


def fetch_chembl_additional_assay_info(assay_ids: list) -> pd.DataFrame:
    """
    Fetch detailed assay metadata from ChEMBL for assay-level filtering.
    Useful for distinguishing intracellular amastigote assays from
    promastigote or axenic assays.

    Parameters
    ----------
    assay_ids : list
        List of ChEMBL assay IDs

    Returns
    -------
    pd.DataFrame
        Assay metadata
    """
    from chembl_webresource_client.new_client import new_client

    assay_api = new_client.assay

    logger.info(f"Fetching metadata for {len(assay_ids)} assays...")

    records = []
    for batch_start in tqdm(range(0, len(assay_ids), 50)):
        batch = assay_ids[batch_start:batch_start + 50]
        for aid in batch:
            try:
                result = assay_api.get(aid)
                if result:
                    records.append({
                        "assay_chembl_id": result.get("assay_chembl_id"),
                        "assay_type": result.get("assay_type"),
                        "assay_organism": result.get("assay_organism"),
                        "description": result.get("description"),
                        "assay_cell_type": result.get("assay_cell_type"),
                    })
            except Exception:
                continue
        time.sleep(1)  # Rate limiting

    return pd.DataFrame(records)


def generate_collection_report(df: pd.DataFrame) -> str:
    """
    Generate a markdown-formatted data collection report.

    Parameters
    ----------
    df : pd.DataFrame
        Combined raw dataset

    Returns
    -------
    str
        Markdown report string
    """
    report = []
    report.append("# Data Collection Report\n")
    report.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    report.append(f"**Total records**: {len(df)}\n")
    report.append(f"**Unique compounds**: {df['molecule_chembl_id'].nunique()}\n")

    report.append("\n## Records per Target\n")
    target_counts = df.groupby("target_name").agg(
        records=("molecule_chembl_id", "count"),
        unique_compounds=("molecule_chembl_id", "nunique"),
        median_pchembl=("pchembl_value", "median"),
    ).reset_index()
    report.append(target_counts.to_markdown(index=False))

    report.append("\n\n## Activity Type Distribution\n")
    type_counts = df["standard_type"].value_counts()
    for atype, count in type_counts.items():
        report.append(f"- **{atype}**: {count} records\n")

    if "is_sulfonamide" in df.columns:
        n_sulfo = df["is_sulfonamide"].sum()
        report.append(f"\n## Sulfonamide Subset\n")
        report.append(f"- Sulfonamide compounds: **{n_sulfo}**\n")
        report.append(f"- Non-sulfonamide compounds: **{len(df) - n_sulfo}**\n")

    return "\n".join(report)

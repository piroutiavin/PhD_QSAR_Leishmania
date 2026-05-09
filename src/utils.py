"""
Utilities Module
================
Plotting helpers, logging setup, and miscellaneous utility functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ─── Plot Style Configuration ───────────────────────────────────
PLOT_STYLE = {
    "figure.figsize": (10, 7),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Publication-quality color palette
COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "success": "#16A34A",
    "warning": "#F59E0B",
    "info": "#8B5CF6",
    "models": ["#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#8B5CF6"],
}


def setup_plot_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(PLOT_STYLE)
    sns.set_palette(COLORS["models"])


def plot_activity_distribution(df: pd.DataFrame,
                                pic50_col: str = "pIC50",
                                target_col: str = "target_name",
                                save_path: str = None):
    """
    Plot pIC50 distribution by target species.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    for target in df[target_col].unique():
        subset = df[df[target_col] == target]
        axes[0].hist(subset[pic50_col], bins=30, alpha=0.6, label=target)
    axes[0].axvline(x=5.0, color="red", linestyle="--", linewidth=1.5, label="Active threshold (pIC50=5)")
    axes[0].set_xlabel("pIC50")
    axes[0].set_ylabel("Count")
    axes[0].set_title("pIC50 Distribution by Target")
    axes[0].legend(fontsize=9)

    # Class balance
    if "activity_class" in df.columns:
        class_counts = df.groupby([target_col, "activity_class"]).size().unstack(fill_value=0)
        class_counts.plot(kind="bar", ax=axes[1], color=[COLORS["success"], COLORS["secondary"]])
        axes[1].set_title("Active vs Inactive per Target")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Figure saved: {save_path}")
    plt.show()


def plot_chemical_space_pca(df: pd.DataFrame,
                             smiles_col: str = "std_smiles",
                             color_col: str = "activity_class",
                             save_path: str = None):
    """
    PCA visualization of chemical space using Morgan fingerprints.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.decomposition import PCA

    setup_plot_style()

    # Compute fingerprints
    fps = []
    valid_idx = []
    for i, smi in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps.append(list(fp))
            valid_idx.append(i)

    X_fp = np.array(fps)
    labels = df.iloc[valid_idx][color_col].values

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_fp)

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in np.unique(labels):
        mask = labels == label
        color = COLORS["success"] if label == "Active" else COLORS["secondary"]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   alpha=0.5, s=20, label=label, color=color)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Chemical Space — PCA of Morgan Fingerprints")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary statistics table for the dataset."""
    summary = {
        "Total records": len(df),
        "Unique compounds": df["std_smiles"].nunique() if "std_smiles" in df.columns else "N/A",
        "Unique scaffolds": df["scaffold"].nunique() if "scaffold" in df.columns else "N/A",
    }

    if "pIC50" in df.columns:
        summary.update({
            "pIC50 mean": f"{df['pIC50'].mean():.2f}",
            "pIC50 median": f"{df['pIC50'].median():.2f}",
            "pIC50 std": f"{df['pIC50'].std():.2f}",
            "pIC50 range": f"{df['pIC50'].min():.2f} – {df['pIC50'].max():.2f}",
        })

    if "activity_class" in df.columns:
        n_active = (df["activity_class"] == "Active").sum()
        summary.update({
            "Active compounds": n_active,
            "Inactive compounds": len(df) - n_active,
            "Active ratio": f"{n_active/len(df)*100:.1f}%",
        })

    if "is_sulfonamide" in df.columns:
        summary["Sulfonamide compounds"] = df["is_sulfonamide"].sum()

    return pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

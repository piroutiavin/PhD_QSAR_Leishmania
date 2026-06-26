"""
Phase 2.1 — Leakage-free scaffold splitting
===========================================
Provides train/test and cross-validation splitters in which whole Murcko
scaffolds never span the train and test partitions. Because each per-organism
dataset already has exactly one row per molecule, grouping by scaffold also
guarantees no molecule leaks across folds.

Two tools:
  * scaffold_holdout_split  — single leakage-free train/test split (diagnostics)
  * RepeatedScaffoldCV      — repeated, activity-stratified, scaffold-grouped CV
                              (the splitter used for model evaluation in 2.3)

Run as a script to verify zero leakage across the 4 panel datasets:
  .venv/Scripts/python.exe -m src.data_splitting
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PANEL_DIR = Path("data/processed/sulfonamide_panel")


def scaffold_groups(df: pd.DataFrame, scaffold_col: str = "scaffold") -> np.ndarray:
    """Map each row's Murcko scaffold string to a stable integer group id."""
    codes, _ = pd.factorize(df[scaffold_col], sort=True)
    return codes


def scaffold_holdout_split(
    df: pd.DataFrame,
    y: np.ndarray | pd.Series,
    test_frac: float = 0.2,
    seed: int = 42,
    scaffold_col: str = "scaffold",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single leakage-free, activity-stratified train/test split.

    Implemented as the first fold of a StratifiedGroupKFold with
    n_splits = round(1/test_frac): whole scaffolds stay on one side AND the
    class balance is matched between train and test. Returns positional indices.

    NB: per the plan, model evaluation uses RepeatedScaffoldCV; this single
    split is for quick diagnostics / illustrative figures only.
    """
    df = df.reset_index(drop=True)
    y = np.asarray(y)
    groups = scaffold_groups(df, scaffold_col)
    n_splits = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(np.zeros((len(df), 1)), y, groups))
    return np.sort(train_idx), np.sort(test_idx)


class RepeatedScaffoldCV:
    """
    Repeated, activity-stratified, scaffold-grouped K-fold cross-validation.

    Wraps sklearn's StratifiedGroupKFold (which balances the class label while
    keeping each group wholly within one fold) and repeats it with reshuffled
    seeds to reduce the high variance expected on these small datasets.
    """

    def __init__(self, n_splits: int = 5, n_repeats: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups):
        for r in range(self.n_repeats):
            sgkf = StratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.seed + r
            )
            yield from sgkf.split(X, y, groups)


def _verify(df: pd.DataFrame, name: str) -> None:
    y = (df["activity_class"] == "Active").astype(int).to_numpy()
    groups = scaffold_groups(df)
    X = np.zeros((len(df), 1))

    # holdout check
    tr, te = scaffold_holdout_split(df, y)
    tr_scaf = set(df.iloc[tr]["scaffold"]); te_scaf = set(df.iloc[te]["scaffold"])
    assert not (tr_scaf & te_scaf), "scaffold leak in holdout!"
    print(f"  holdout: train={len(tr)} (act {y[tr].mean():.0%})  "
          f"test={len(te)} (act {y[te].mean():.0%})  shared_scaffolds={len(tr_scaf & te_scaf)}")

    # repeated CV check
    cv = RepeatedScaffoldCV(n_splits=5, n_repeats=3, seed=42)
    fold_sizes, leaks = [], 0
    for tri, tei in cv.split(X, y, groups):
        if set(groups[tri]) & set(groups[tei]):
            leaks += 1
        fold_sizes.append(len(tei))
    print(f"  CV (5x3): folds={cv.get_n_splits()}  "
          f"test-fold size {min(fold_sizes)}–{max(fold_sizes)}  scaffold_leaks={leaks}")


if __name__ == "__main__":
    print("Verifying leakage-free splitting on the sulfonamide panel:\n")
    for f in sorted(PANEL_DIR.glob("sulfonamide_*.csv")):
        df = pd.read_csv(f)
        print(f"{f.stem}  (n={len(df)}, scaffolds={df['scaffold'].nunique()})")
        _verify(df, f.stem)
        print()

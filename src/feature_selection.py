"""
Phase 2.2b — Leakage-aware feature reduction per organism
=========================================================
Two stages:
  1. Label-free reduction (variance + correlation filter |r|>0.95) -> candidate
     pool. Uses no labels, so it cannot leak into cross-validation.
  2. Stability selection: across the training folds of RepeatedScaffoldCV, rank
     candidate descriptors by mutual information and RandomForest importance and
     keep those selected most consistently. Labels are touched only on train
     folds, then aggregated -> a robust ~10-15 descriptor set per organism.

Outputs results/feature_selection/<key>_selected_features.csv and a summary.

Run:  .venv/Scripts/python.exe -m src.feature_selection
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

from src.data_splitting import RepeatedScaffoldCV, scaffold_groups

PANEL_DIR = Path("data/processed/sulfonamide_panel")
DESC = PANEL_DIR / "mordred_descriptors.csv"
OUT_DIR = Path("results/feature_selection")

CORR_THRESHOLD = 0.95
TOP_K_PER_FOLD = 15          # descriptors flagged per fold by each ranker
FINAL_K = 12                 # final feature-set size target per organism


def label_free_reduce(X: pd.DataFrame) -> list[str]:
    """Drop near-constant columns and one of each highly-correlated pair."""
    X = X.loc[:, X.var() > 1e-8]
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
    return [c for c in X.columns if c not in drop]


def stability_select(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Frequency with which each descriptor lands in the per-fold top-K."""
    cols = list(X.columns)
    mi_hits = {c: 0 for c in cols}
    rf_hits = {c: 0 for c in cols}
    cv = RepeatedScaffoldCV(n_splits=5, n_repeats=5, seed=42)
    n_folds = 0
    for tri, _ in cv.split(X.values, y, groups):
        n_folds += 1
        Xtr, ytr = X.iloc[tri], y[tri]
        Xs = StandardScaler().fit_transform(Xtr)

        mi = mutual_info_classif(Xs, ytr, random_state=0)
        for c in np.array(cols)[np.argsort(mi)[::-1][:TOP_K_PER_FOLD]]:
            mi_hits[c] += 1

        rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
        rf.fit(Xtr, ytr)
        for c in np.array(cols)[np.argsort(rf.feature_importances_)[::-1][:TOP_K_PER_FOLD]]:
            rf_hits[c] += 1

    res = pd.DataFrame({
        "descriptor": cols,
        "mi_freq": [mi_hits[c] / n_folds for c in cols],
        "rf_freq": [rf_hits[c] / n_folds for c in cols],
    })
    res["mean_freq"] = res[["mi_freq", "rf_freq"]].mean(axis=1)
    return res.sort_values("mean_freq", ascending=False).reset_index(drop=True)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc = pd.read_csv(DESC)
    summary = []

    for f in sorted(PANEL_DIR.glob("sulfonamide_*.csv")):
        key = f.stem.replace("sulfonamide_", "")
        d = pd.read_csv(f).merge(desc, on="std_smiles", how="left")
        feat_cols = [c for c in desc.columns if c != "std_smiles"]
        X = d[feat_cols].astype(float)
        y = (d["activity_class"] == "Active").astype(int).to_numpy()
        groups = scaffold_groups(d)

        pool = label_free_reduce(X)
        ranked = stability_select(X[pool], y, groups)
        ranked.to_csv(OUT_DIR / f"{key}_feature_ranking.csv", index=False)

        final = ranked.head(FINAL_K)["descriptor"].tolist()
        pd.Series(final, name="feature").to_csv(
            OUT_DIR / f"{key}_selected_features.csv", index=False)

        print(f"\n{key}: {len(feat_cols)} desc -> pool {len(pool)} -> top {FINAL_K}")
        print(ranked.head(FINAL_K)[["descriptor", "mean_freq"]].to_string(index=False))
        summary.append({"organism_key": key, "n_descriptors": len(feat_cols),
                        "candidate_pool": len(pool), "final_k": len(final),
                        "top_feature": final[0]})

    pd.DataFrame(summary).to_csv(OUT_DIR / "selection_summary.csv", index=False)
    print(f"\nSaved selections to {OUT_DIR}")


if __name__ == "__main__":
    run()

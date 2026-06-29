"""
Phase 2.3 — Honest repeated cross-validation evaluation
=======================================================
Evaluates candidate models per organism using RepeatedScaffoldCV (scaffold-
grouped, activity-stratified). Feature scaling and supervised feature selection
are wrapped in a Pipeline so they are refit inside every training fold — no
information from the held-out fold leaks into selection. This replaces the old
single leaky held-out test.

Models are deliberately light (appropriate for n=72-398) and a DummyClassifier
provides the no-skill baseline. Hyperparameter search is added in Phase 2.4.

Run:  .venv/Scripts/python.exe -m src.evaluate
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data_splitting import RepeatedScaffoldCV, scaffold_groups

PANEL_DIR = Path("data/processed/sulfonamide_panel")
DESC = PANEL_DIR / "mordred_descriptors.csv"
OUT_DIR = Path("results/model_eval")
K_FEATURES = 12


def _mi(X, y):
    return mutual_info_classif(X, y, random_state=0)


def make_models() -> dict:
    """Light models suited to small data; class_weight balances imbalanced sets."""
    return {
        "Dummy": DummyClassifier(strategy="stratified", random_state=0),
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "SVM-RBF": SVC(kernel="rbf", probability=False, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=0, n_jobs=-1),
        "GradBoost": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0),
    }


def pipeline_for(model) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=_mi, k=K_FEATURES)),
        ("clf", model),
    ])


def _scores(clf, X):
    """Continuous score for ROC-AUC from proba or decision_function."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def evaluate_dataset(df: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    d = df.merge(desc, on="std_smiles", how="left")
    feat_cols = [c for c in desc.columns if c != "std_smiles"]
    X = d[feat_cols].astype(float).to_numpy()
    y = (d["activity_class"] == "Active").astype(int).to_numpy()
    groups = scaffold_groups(d)
    cv = RepeatedScaffoldCV(n_splits=5, n_repeats=5, seed=42)

    rows = []
    for name, model in make_models().items():
        per_fold = {"BalAcc": [], "F1": [], "MCC": [], "ROC_AUC": []}
        for tri, tei in cv.split(X, y, groups):
            pipe = pipeline_for(model)
            # SVC needs probability for AUC; use decision_function instead
            pipe.fit(X[tri], y[tri])
            yp = pipe.predict(X[tei])
            ys = _scores(pipe.named_steps["clf"],
                         pipe[:-1].transform(X[tei]))
            per_fold["BalAcc"].append(balanced_accuracy_score(y[tei], yp))
            per_fold["F1"].append(f1_score(y[tei], yp, zero_division=0))
            per_fold["MCC"].append(matthews_corrcoef(y[tei], yp))
            per_fold["ROC_AUC"].append(roc_auc_score(y[tei], ys))
        row = {"model": name}
        for m, vals in per_fold.items():
            row[f"{m}_mean"] = np.mean(vals)
            row[f"{m}_std"] = np.std(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc = pd.read_csv(DESC)
    all_summ = []
    for f in sorted(PANEL_DIR.glob("sulfonamide_*.csv")):
        key = f.stem.replace("sulfonamide_", "")
        res = evaluate_dataset(pd.read_csv(f), desc)
        res.insert(0, "organism", key)
        res.to_csv(OUT_DIR / f"{key}_cv_results.csv", index=False)
        all_summ.append(res)
        print(f"\n=== {key} ===")
        print(res[["model", "MCC_mean", "MCC_std", "ROC_AUC_mean",
                   "BalAcc_mean", "F1_mean"]].round(3).to_string(index=False))
    pd.concat(all_summ, ignore_index=True).to_csv(
        OUT_DIR / "cv_results_all.csv", index=False)
    print(f"\nSaved CV results to {OUT_DIR}")


if __name__ == "__main__":
    run()

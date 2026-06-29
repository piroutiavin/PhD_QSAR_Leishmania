"""
Phase 2.4 — Hyperparameter tuning + final per-organism models
=============================================================
For each organism, on its fixed top-12 interpretable descriptor set (from
Phase 2.2), tune the three winning model families (LogReg, SVM-RBF,
RandomForest) by NESTED cross-validation:
  outer = RepeatedScaffoldCV (honest estimate of the *tuning procedure*)
  inner = StratifiedGroupKFold GridSearchCV (scaffold-grouped, MCC-scored)

The best family per organism (by nested-CV MCC) is then refit on the full
dataset with an inner-CV grid search and saved as the deployable model.

Run:  .venv/Scripts/python.exe -m src.train_final
"""

from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data_splitting import RepeatedScaffoldCV, scaffold_groups

PANEL_DIR = Path("data/processed/sulfonamide_panel")
DESC = PANEL_DIR / "mordred_descriptors.csv"
FS_DIR = Path("results/feature_selection")
OUT_DIR = Path("results/model_eval")
MODEL_DIR = Path("models/sulfonamide_panel")
MCC = make_scorer(matthews_corrcoef)

FAMILIES = {
    "LogReg": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]),
        {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "SVM-RBF": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced"))]),
        {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]},
    ),
    "RandomForest": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", RandomForestClassifier(n_estimators=400,
                       class_weight="balanced", random_state=0, n_jobs=-1))]),
        {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3, 5]},
    ),
}


def load_xy(key: str, desc: pd.DataFrame):
    df = pd.read_csv(PANEL_DIR / f"sulfonamide_{key}.csv")
    feats = pd.read_csv(FS_DIR / f"{key}_selected_features.csv")["feature"].tolist()
    d = df.merge(desc[["std_smiles"] + feats], on="std_smiles", how="left")
    X = d[feats].astype(float).to_numpy()
    y = (d["activity_class"] == "Active").astype(int).to_numpy()
    groups = scaffold_groups(d)
    return X, y, groups, feats


def nested_mcc(pipe, grid, X, y, groups) -> tuple[float, float]:
    outer = RepeatedScaffoldCV(n_splits=5, n_repeats=3, seed=42)
    scores = []
    for tri, tei in outer.split(X, y, groups):
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        gs = GridSearchCV(pipe, grid, scoring=MCC, cv=inner, n_jobs=-1)
        gs.fit(X[tri], y[tri], groups=groups[tri])
        scores.append(matthews_corrcoef(y[tei], gs.predict(X[tei])))
    return float(np.mean(scores)), float(np.std(scores))


def run() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc = pd.read_csv(DESC)
    summary = []

    for f in sorted(PANEL_DIR.glob("sulfonamide_*.csv")):
        key = f.stem.replace("sulfonamide_", "")
        X, y, groups, feats = load_xy(key, desc)
        print(f"\n=== {key} (n={len(y)}, {len(feats)} features) ===")

        rows = []
        for name, (pipe, grid) in FAMILIES.items():
            m, s = nested_mcc(pipe, grid, X, y, groups)
            rows.append({"family": name, "nested_MCC_mean": m, "nested_MCC_std": s})
            print(f"  {name:13s} nested MCC = {m:.3f} ± {s:.3f}")
        rank = pd.DataFrame(rows).sort_values("nested_MCC_mean", ascending=False)
        rank.insert(0, "organism", key)
        rank.to_csv(OUT_DIR / f"{key}_nested_tuning.csv", index=False)

        best = rank.iloc[0]["family"]
        pipe, grid = FAMILIES[best]
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        gs = GridSearchCV(pipe, grid, scoring=MCC, cv=inner, n_jobs=-1)
        gs.fit(X, y, groups=groups)            # refit best family on full data

        joblib.dump({"pipeline": gs.best_estimator_, "features": feats,
                     "family": best, "best_params": gs.best_params_},
                    MODEL_DIR / f"{key}_final.joblib")
        print(f"  -> FINAL: {best}  params={gs.best_params_}")
        summary.append({"organism": key, "n": len(y), "best_family": best,
                        "nested_MCC_mean": round(rank.iloc[0]["nested_MCC_mean"], 3),
                        "nested_MCC_std": round(rank.iloc[0]["nested_MCC_std"], 3),
                        "best_params": str(gs.best_params_)})

    pd.DataFrame(summary).to_csv(OUT_DIR / "final_models_summary.csv", index=False)
    print(f"\nSaved final models to {MODEL_DIR} and summary to {OUT_DIR}")


if __name__ == "__main__":
    run()

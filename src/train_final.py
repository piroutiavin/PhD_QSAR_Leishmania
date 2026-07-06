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
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_splitting import RepeatedScaffoldCV, scaffold_groups

PANEL_DIR = Path("data/processed/sulfonamide_panel")
DESC = PANEL_DIR / "mordred_descriptors.csv"
FS_DIR = Path("results/feature_selection")
OUT_DIR = Path("results/model_eval")
MODEL_DIR = Path("models/sulfonamide_panel")
MCC = make_scorer(matthews_corrcoef)

# Parsimony rule (agreed): where the full nested-CV panel ties within noise,
# choose the simplest interpretable model for a clean SHAP/SAR story. The full
# comparison (incl. XGBoost/LightGBM/Consensus/Stacking) is still reported in
# <key>_nested_tuning.csv; only the *final* deployed model is fixed here.
FINAL_CHOICE = {
    "L_infantum": "LogReg",       # ties Stacking (0.509 vs 0.508); LogReg interpretable
    "T_cruzi": "SVM-RBF",         # ties Consensus/RF within noise; adequate + simpler
    "L_donovani": "RandomForest", # clear panel best
    "L_amazonensis": "LogReg",    # clear panel best
}


def _scaled(clf) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def make_families(pos_weight: float = 1.0) -> dict:
    """Tunable model families + fixed ensembles (empty grid = single candidate)."""
    base = [
        ("LogReg", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ("SVM-RBF", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ("RandomForest", RandomForestClassifier(n_estimators=400,
             class_weight="balanced", random_state=0, n_jobs=-1)),
        ("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
             scale_pos_weight=pos_weight, random_state=0, n_jobs=-1)),
        ("LightGBM", LGBMClassifier(n_estimators=300, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
             random_state=0, n_jobs=-1, verbose=-1)),
    ]
    estimators = [(n, _scaled(c)) for n, c in base]

    fam = {
        "LogReg": (_scaled(base[0][1]), {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "SVM-RBF": (_scaled(base[1][1]),
                    {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]}),
        "RandomForest": (_scaled(base[2][1]),
                    {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3, 5]}),
        "XGBoost": (_scaled(base[3][1]),
                    {"clf__max_depth": [2, 3, 4]}),
        "LightGBM": (_scaled(base[4][1]),
                    {"clf__max_depth": [2, 3, 4], "clf__num_leaves": [7, 15]}),
        # ensembles: no tuning grid (single candidate)
        "Consensus": (VotingClassifier(estimators=estimators, voting="soft"), {}),
        "Stacking": (StackingClassifier(estimators=estimators,
                        final_estimator=LogisticRegression(max_iter=5000),
                        cv=StratifiedKFold(5, shuffle=True, random_state=0)), {}),
    }
    return fam


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
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        families = make_families(pos_weight)
        print(f"\n=== {key} (n={len(y)}, {len(feats)} features) ===")

        rows = []
        for name, (pipe, grid) in families.items():
            m, s = nested_mcc(pipe, grid, X, y, groups)
            rows.append({"family": name, "nested_MCC_mean": m, "nested_MCC_std": s})
            print(f"  {name:13s} nested MCC = {m:.3f} ± {s:.3f}")
        rank = pd.DataFrame(rows).sort_values("nested_MCC_mean", ascending=False)
        rank.insert(0, "organism", key)
        rank.to_csv(OUT_DIR / f"{key}_nested_tuning.csv", index=False)

        argmax_family = rank.iloc[0]["family"]
        chosen = FINAL_CHOICE[key]                    # parsimony-based final model
        chosen_row = rank[rank["family"] == chosen].iloc[0]
        pipe, grid = families[chosen]
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        gs = GridSearchCV(pipe, grid, scoring=MCC, cv=inner, n_jobs=-1)
        gs.fit(X, y, groups=groups)                   # refit chosen family on full data

        joblib.dump({"pipeline": gs.best_estimator_, "features": feats,
                     "family": chosen, "best_params": gs.best_params_},
                    MODEL_DIR / f"{key}_final.joblib")
        print(f"  -> FINAL (parsimony): {chosen}  params={gs.best_params_}  "
              f"[panel argmax was {argmax_family}]")
        summary.append({"organism": key, "n": len(y),
                        "final_family": chosen,
                        "final_nested_MCC_mean": round(chosen_row["nested_MCC_mean"], 3),
                        "final_nested_MCC_std": round(chosen_row["nested_MCC_std"], 3),
                        "panel_argmax_family": argmax_family,
                        "panel_argmax_MCC": round(rank.iloc[0]["nested_MCC_mean"], 3),
                        "best_params": str(gs.best_params_)})

    pd.DataFrame(summary).to_csv(OUT_DIR / "final_models_summary.csv", index=False)
    print(f"\nSaved final models to {MODEL_DIR} and summary to {OUT_DIR}")


if __name__ == "__main__":
    run()

"""
Phase 3.1 — Applicability domain + Y-randomization
==================================================
Per organism, on the final model's 12 descriptors:

  * Applicability domain (Williams plot): leverage h_i from the scaled feature
    matrix vs out-of-fold standardized residuals. Warning leverage h* =
    3(p+1)/n; compounds with h>h* or |std residual|>3 are outside the AD.

  * Y-randomization: refit the final model family under scaffold-grouped CV on
    many label shuffles. If the real MCC sits far above the shuffled
    distribution, the model captures genuine signal (not chance).

Outputs figures/sulfonamide_panel/<key>_williams.png, _yrandom.png and
results/validation/*.csv.

Run:  .venv/Scripts/python.exe -m src.validation
"""

from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.data_splitting import RepeatedScaffoldCV, scaffold_groups

PANEL_DIR = Path("data/processed/sulfonamide_panel")
DESC = PANEL_DIR / "mordred_descriptors.csv"
MODEL_DIR = Path("models/sulfonamide_panel")
FIG_DIR = Path("figures/sulfonamide_panel")
OUT_DIR = Path("results/validation")
N_SHUFFLE = 50


def load(key: str, desc: pd.DataFrame):
    bundle = joblib.load(MODEL_DIR / f"{key}_final.joblib")
    feats = bundle["features"]
    df = pd.read_csv(PANEL_DIR / f"sulfonamide_{key}.csv")
    d = df.merge(desc[["std_smiles"] + feats], on="std_smiles", how="left")
    X = d[feats].astype(float).to_numpy()
    y = (d["activity_class"] == "Active").astype(int).to_numpy()
    groups = scaffold_groups(d)
    return bundle, X, y, groups, feats


def williams(key, bundle, X, y, groups) -> dict:
    p = X.shape[1]
    n = X.shape[0]
    Xs = StandardScaler().fit_transform(X)
    # leverage (hat) diagonal from the scaled descriptor matrix
    H = Xs @ np.linalg.pinv(Xs.T @ Xs) @ Xs.T
    h = np.diag(H)
    h_star = 3 * (p + 1) / n

    cv = RepeatedScaffoldCV(n_splits=5, n_repeats=1, seed=42)
    proba = cross_val_predict(clone(bundle["pipeline"]), X, y, cv=cv,
                              groups=groups, method="predict_proba")[:, 1]
    resid = y - proba
    std_res = resid / resid.std()

    inside = (h <= h_star) & (np.abs(std_res) <= 3)
    plt.figure(figsize=(6, 4.2))
    plt.scatter(h[y == 1], std_res[y == 1], s=18, c="#c0392b", label="Active", alpha=.7)
    plt.scatter(h[y == 0], std_res[y == 0], s=18, c="#2c7fb8", label="Inactive", alpha=.7)
    plt.axvline(h_star, ls="--", c="grey"); plt.axhline(3, ls="--", c="grey")
    plt.axhline(-3, ls="--", c="grey")
    plt.xlabel(f"Leverage (h*={h_star:.2f})"); plt.ylabel("Std. residual")
    plt.title(f"{key} — Williams plot ({bundle['family']})")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / f"{key}_williams.png", dpi=200); plt.close()
    return {"organism": key, "h_star": round(h_star, 3),
            "pct_inside_AD": round(100 * inside.mean(), 1),
            "n_outside": int((~inside).sum())}


def y_randomize(key, bundle, X, y, groups) -> dict:
    # single 5-fold partition so cross_val_predict has one prediction per sample
    cv = RepeatedScaffoldCV(n_splits=5, n_repeats=1, seed=7)

    def cv_mcc(yy):
        pred = cross_val_predict(clone(bundle["pipeline"]), X, yy, cv=cv,
                                 groups=groups, method="predict")
        return matthews_corrcoef(yy, pred)

    real = cv_mcc(y)
    rng = np.random.default_rng(0)
    shuffled = np.array([cv_mcc(rng.permutation(y)) for _ in range(N_SHUFFLE)])
    p_val = (np.sum(shuffled >= real) + 1) / (N_SHUFFLE + 1)

    plt.figure(figsize=(6, 4.2))
    plt.hist(shuffled, bins=15, color="#95a5a6", label="Y-randomized")
    plt.axvline(real, c="#c0392b", lw=2, label=f"Real MCC = {real:.2f}")
    plt.xlabel("MCC"); plt.ylabel("Count")
    plt.title(f"{key} — Y-randomization ({N_SHUFFLE} shuffles)")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / f"{key}_yrandom.png", dpi=200); plt.close()
    return {"organism": key, "real_MCC": round(real, 3),
            "yrand_MCC_mean": round(shuffled.mean(), 3),
            "yrand_MCC_max": round(shuffled.max(), 3), "p_value": round(p_val, 3)}


def run() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc = pd.read_csv(DESC)
    ad, yr = [], []
    for f in sorted(PANEL_DIR.glob("sulfonamide_*.csv")):
        key = f.stem.replace("sulfonamide_", "")
        bundle, X, y, groups, feats = load(key, desc)
        ad.append(williams(key, bundle, X, y, groups))
        yr.append(y_randomize(key, bundle, X, y, groups))
        print(f"{key}: AD_inside={ad[-1]['pct_inside_AD']}%  "
              f"real_MCC={yr[-1]['real_MCC']}  yrand_mean={yr[-1]['yrand_MCC_mean']}  "
              f"p={yr[-1]['p_value']}")
    pd.DataFrame(ad).to_csv(OUT_DIR / "applicability_domain.csv", index=False)
    pd.DataFrame(yr).to_csv(OUT_DIR / "y_randomization.csv", index=False)
    print(f"\nSaved validation results to {OUT_DIR} and figures to {FIG_DIR}")


if __name__ == "__main__":
    run()

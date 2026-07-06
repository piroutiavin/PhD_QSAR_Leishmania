"""
Phase 2.5 — SHAP interpretation of the final per-organism models
================================================================
Explains each organism's final model (LogReg / RandomForest / SVM-RBF) on its
12 interpretable Mordred descriptors using a model-agnostic SHAP explainer
(KernelExplainer on the pipeline's P(active)). Produces, per organism:
  - <key>_shap_importance.csv   mean |SHAP| global ranking
  - <key>_shap_bar.png          global importance bar
  - <key>_shap_beeswarm.png     per-compound SHAP distribution
and a combined cross-organism importance table for the SAR discussion (3.4).

Run:  .venv/Scripts/python.exe -m src.shap_analysis
"""

from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

PANEL_DIR = Path("data/processed/sulfonamide_panel")
DESC = PANEL_DIR / "mordred_descriptors.csv"
MODEL_DIR = Path("models/sulfonamide_panel")
FIG_DIR = Path("figures/sulfonamide_panel")
OUT_DIR = Path("results/shap_panel")
SEED = 0


def load_model_xy(key: str, desc: pd.DataFrame):
    bundle = joblib.load(MODEL_DIR / f"{key}_final.joblib")
    feats = bundle["features"]
    df = pd.read_csv(PANEL_DIR / f"sulfonamide_{key}.csv")
    d = df.merge(desc[["std_smiles"] + feats], on="std_smiles", how="left")
    X = d[feats].astype(float).reset_index(drop=True)
    return bundle, X, feats


def explain(key: str, desc: pd.DataFrame) -> pd.DataFrame:
    bundle, X, feats = load_model_xy(key, desc)
    pipe = bundle["pipeline"]

    def f(data):
        return pipe.predict_proba(data)[:, 1]

    # compact, reproducible background; explain all molecules
    bg = shap.kmeans(X.values, min(15, len(X)))
    explainer = shap.KernelExplainer(f, bg, seed=SEED)
    sv = explainer.shap_values(X.values, nsamples=200, silent=True)
    sv = np.asarray(sv)

    imp = (pd.DataFrame({"descriptor": feats,
                         "mean_abs_shap": np.abs(sv).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    imp.insert(0, "organism", key)
    imp.to_csv(OUT_DIR / f"{key}_shap_importance.csv", index=False)

    # bar plot
    plt.figure(figsize=(6, 4))
    shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=12)
    plt.title(f"{key} — SHAP global importance ({bundle['family']})")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{key}_shap_bar.png", dpi=200); plt.close()

    # beeswarm
    plt.figure(figsize=(6, 4))
    shap.summary_plot(sv, X, show=False, max_display=12)
    plt.title(f"{key} — SHAP summary ({bundle['family']})")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{key}_shap_beeswarm.png", dpi=200); plt.close()

    print(f"{key} ({bundle['family']}): top-5 -> "
          + ", ".join(imp['descriptor'].head(5)))
    return imp


def run() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc = pd.read_csv(DESC)
    all_imp = []
    for f in sorted(PANEL_DIR.glob("sulfonamide_*.csv")):
        key = f.stem.replace("sulfonamide_", "")
        all_imp.append(explain(key, desc))

    # combined cross-organism importance (for SAR comparison in Phase 3.4)
    combined = (pd.concat(all_imp, ignore_index=True)
                .pivot_table(index="descriptor", columns="organism",
                             values="mean_abs_shap", fill_value=0.0))
    combined.to_csv(OUT_DIR / "shap_importance_all_organisms.csv")
    print(f"\nSaved SHAP outputs to {OUT_DIR} and figures to {FIG_DIR}")


if __name__ == "__main__":
    run()

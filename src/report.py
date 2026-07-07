"""
Phase 3.2 + 3.4 — Consolidated honest results & cross-species SAR
================================================================
Pulls the HONEST Phase 2.3 metrics (feature selection refit in-fold) for each
organism's final model, joins the applicability-domain and Y-randomization
outcomes, and writes a single master results table + summary figure.

Also builds the cross-species SAR comparison from the combined SHAP importance
table: which descriptors are shared across organisms (general anti-kinetoplastid
signal) vs organism-specific.

Run:  .venv/Scripts/python.exe -m src.report
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANEL_DIR = Path("data/processed/sulfonamide_panel")
EVAL_DIR = Path("results/model_eval")
VAL_DIR = Path("results/validation")
SHAP_DIR = Path("results/shap_panel")
FIG_DIR = Path("figures/sulfonamide_panel")
OUT_DIR = Path("results/report")

# final model family per organism (parsimony choice from Phase 2.4)
FINAL = {"L_infantum": "LogReg", "L_donovani": "RandomForest",
         "T_cruzi": "SVM-RBF", "L_amazonensis": "LogReg"}
LABELS = {"L_infantum": "L. infantum", "L_donovani": "L. donovani",
          "T_cruzi": "T. cruzi", "L_amazonensis": "L. amazonensis"}


def master_table() -> pd.DataFrame:
    ad = pd.read_csv(VAL_DIR / "applicability_domain.csv").set_index("organism")
    yr = pd.read_csv(VAL_DIR / "y_randomization.csv").set_index("organism")
    rows = []
    for key, fam in FINAL.items():
        panel = pd.read_csv(PANEL_DIR / f"sulfonamide_{key}.csv")
        cv = pd.read_csv(EVAL_DIR / f"{key}_cv_results.csv")
        r = cv[cv["model"] == fam].iloc[0]
        n = len(panel); n_act = (panel["activity_class"] == "Active").sum()
        rows.append({
            "organism": LABELS[key], "n": n,
            "active_pct": round(100 * n_act / n, 1),
            "final_model": fam,
            "MCC": f"{r['MCC_mean']:.2f} ± {r['MCC_std']:.2f}",
            "ROC_AUC": f"{r['ROC_AUC_mean']:.2f} ± {r['ROC_AUC_std']:.2f}",
            "BalAcc": f"{r['BalAcc_mean']:.2f}",
            "AD_inside_%": ad.loc[key, "pct_inside_AD"],
            "Yrand_p": yr.loc[key, "p_value"],
            "_mcc": r["MCC_mean"], "_mcc_std": r["MCC_std"],
        })
    return pd.DataFrame(rows)


def summary_figure(mt: pd.DataFrame) -> None:
    order = mt.sort_values("_mcc", ascending=False)
    plt.figure(figsize=(6.5, 4))
    plt.bar(order["organism"], order["_mcc"], yerr=order["_mcc_std"],
            color="#2c7fb8", capsize=4, alpha=.85)
    plt.axhline(0, color="grey", lw=.8)
    plt.ylabel("Honest CV MCC (mean ± std)")
    plt.title("Sulfonamide per-organism models — leakage-free performance")
    for i, (_, r) in enumerate(order.iterrows()):
        plt.text(i, r["_mcc"] + r["_mcc_std"] + .02,
                 f"{r['final_model']}\nn={r['n']}", ha="center", fontsize=8)
    plt.ylim(-.05, .8); plt.tight_layout()
    plt.savefig(FIG_DIR / "panel_performance_summary.png", dpi=200); plt.close()


def cross_species_sar() -> pd.DataFrame:
    t = pd.read_csv(SHAP_DIR / "shap_importance_all_organisms.csv", index_col=0)
    # rank within each organism; a descriptor "matters" if in that organism's top-8
    ranks = t.rank(ascending=False)
    in_top8 = (ranks <= 8) & (t > 0)
    summary = pd.DataFrame({
        "descriptor": t.index,
        "n_organisms_top8": in_top8.sum(axis=1).values,
        "organisms": [", ".join(sorted(c for c in t.columns if in_top8.loc[d, c]))
                      for d in t.index],
        "mean_importance": t.mean(axis=1).values,
    }).sort_values(["n_organisms_top8", "mean_importance"], ascending=False)
    return summary.reset_index(drop=True)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mt = master_table()
    mt.drop(columns=["_mcc", "_mcc_std"]).to_csv(OUT_DIR / "master_results.csv", index=False)
    summary_figure(mt)

    sar = cross_species_sar()
    sar.to_csv(OUT_DIR / "cross_species_sar.csv", index=False)
    shared = sar[sar["n_organisms_top8"] >= 2]

    print("=== MASTER RESULTS (honest Phase 2.3 metrics) ===")
    print(mt.drop(columns=["_mcc", "_mcc_std"]).to_string(index=False))
    print(f"\n=== CROSS-SPECIES SAR ===")
    print(f"Descriptors in top-8 of >=2 organisms (shared signal): {len(shared)}")
    print(shared.head(12).to_string(index=False))
    print(f"\nSaved report to {OUT_DIR} and summary figure to {FIG_DIR}")


if __name__ == "__main__":
    run()

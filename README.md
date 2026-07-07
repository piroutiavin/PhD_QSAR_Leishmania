# Leakage-Free QSAR Panel for Anti-Kinetoplastid Sulfonamides

**PhD Thesis Project** — Machine-learning QSAR models predicting the activity of **sulfonamide derivatives** against kinetoplastid parasites, with ***Leishmania infantum*** as the primary target.

> **Scope note (read first).** An earlier version of this project trained a single
> classifier on the full multi-organism ChEMBL dataset. That approach mixed four
> parasites, was only ~6% sulfonamides, and used a random train/test split that
> leaked ~29% of test molecules into training (inflating performance). This
> repository is the **corrected rebuild**: sulfonamide-only, one model per
> organism, and a leakage-free scaffold-grouped protocol. See
> [ENHANCEMENT_TODO.md](ENHANCEMENT_TODO.md) for the full rationale.

## What this project delivers

A **panel of four per-organism binary classifiers** (Active vs Inactive at
pIC50 ≥ 5.0, i.e. IC50 ≤ 10 µM) for sulfonamide compounds, each built and
validated under an identical, honest protocol:

| Organism | n | Active % | Final model | **Honest CV MCC** | ROC-AUC | AD coverage | Y-rand p |
|---|---|---|---|---|---|---|---|
| ***L. infantum*** (primary) | 142 | 50.0 | Logistic Regression | **0.42 ± 0.21** | 0.74 ± 0.14 | 98.6% | 0.02 |
| *L. amazonensis* | 72 | 47.2 | Logistic Regression | 0.39 ± 0.23 | 0.69 ± 0.11 | 95.8% | 0.02 |
| *T. cruzi* | 398 | 70.1 | SVM-RBF | 0.33 ± 0.16 | 0.73 ± 0.10 | 97.5% | 0.02 |
| *L. donovani* | 238 | 45.4 | Random Forest | 0.28 ± 0.24 | 0.70 ± 0.14 | 97.1% | 0.02 |

Metrics are the mean ± std over **repeated, scaffold-grouped, activity-stratified
cross-validation** with feature selection refit inside every fold (no selection
leakage). These are modest but *honest* small-data results; the models beat a
Y-randomized baseline (p = 0.02) with 95–99% applicability-domain coverage.

## Methodology (corrected pipeline)

1. **Dataset rebuild** — filter raw ChEMBL to sulfonamides (SMARTS `S(=O)(=O)N`);
   standardize with tautomer canonicalization; prefer `pchembl_value`; per-organism
   dedup (concordant → median, discordant → drop); partition **by organism**.
2. **Leakage-free splitting** — Murcko scaffold-grouped CV; no molecule or scaffold
   spans train/test.
3. **Descriptors & feature reduction** — ~1,600 Mordred 2D descriptors → label-free
   correlation filter → **stability selection to 12 interpretable descriptors** per
   organism.
4. **Model evaluation** — full algorithm panel (LogReg, SVM-RBF, RandomForest,
   GradientBoosting, **XGBoost, LightGBM, soft-voting Consensus, Stacking**) under
   the leakage-free CV. None outperformed the best single interpretable model within
   uncertainty.
5. **Final models** — nested-CV hyperparameter tuning; **parsimony rule** selects the
   simplest interpretable model where the panel ties within noise.
6. **Interpretation (SHAP)** — per-organism SAR; the drivers are largely
   **organism-specific** (only one descriptor is shared across ≥2 organisms).
7. **Validation** — Williams-plot applicability domain + Y-randomization.

### IC50 vs EC50
These are whole-cell phenotypic assays where IC50/EC50 denote the same 50%-growth
effect, so they are pooled for the classifier; `standard_type` is retained per
record so the choice is auditable and an IC50-only sensitivity check is possible.

## Pipeline (`src/`) — run order

```bash
# uses the project virtualenv (sklearn, xgboost, lightgbm, shap, rdkit, mordred)
python -m src.build_sulfonamide_panel   # Phase 1  -> data/processed/sulfonamide_panel/
python -m src.data_splitting            # Phase 2.1 (verify zero leakage)
python -m src.descriptors               # Phase 2.2a Mordred descriptor matrix
python -m src.feature_selection         # Phase 2.2b top-12 per organism
python -m src.evaluate                  # Phase 2.3 honest full-panel CV
python -m src.train_final               # Phase 2.4 tuning + final models
python -m src.shap_analysis             # Phase 2.5 SHAP / SAR
python -m src.validation                # Phase 3.1 applicability domain + Y-randomization
python -m src.report                    # Phase 3.2/3.4 master table + cross-species SAR
```

> The numbered notebooks `notebooks/01–11` are the **legacy** (pre-rebuild)
> pipeline, retained for provenance. The `src/` modules above supersede them.

## Key outputs

- `data/processed/sulfonamide_panel/` — the four per-organism datasets + summary
- `results/model_eval/` — full-panel CV results, nested tuning, final model summary
- `results/feature_selection/` — per-organism selected descriptors
- `results/shap_panel/`, `results/validation/`, `results/report/` — SAR, AD/Y-rand, master table
- `figures/sulfonamide_panel/` — SHAP, Williams, and performance-summary figures
- `models/sulfonamide_panel/` — deployable per-organism pipelines (gitignored, regenerable)

## Targets

| Target | ChEMBL ID | Role |
|--------|-----------|------|
| *L. infantum* | CHEMBL612848 | **Primary** |
| *L. donovani* | CHEMBL367 | Cross-species |
| *L. amazonensis* | CHEMBL612877 | Cross-species |
| *T. cruzi* | CHEMBL368 | Cross-kinetoplastid |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (use source .venv/bin/activate on Unix)
pip install -r requirements.txt
```

## Software stack (free & open-source)

Python 3 · RDKit · Mordred · scikit-learn · XGBoost · LightGBM · SHAP · pandas/NumPy · matplotlib

## License

Part of a PhD thesis. Code provided for academic use.

## Author

Avin — *affiliation to be added*

---
*Rebuild completed: 2026-07.*

# ML-Based QSAR Modeling for Anti-Leishmanial Sulfonamide Derivatives

**PhD Thesis Project** — Machine Learning-Based QSAR Modeling for Prediction and Optimization of Anti-Leishmanial Activity of Sulfonamide Derivatives Against *Leishmania infantum*

## Project Overview

This project develops machine learning (ML)-based Quantitative Structure-Activity Relationship (QSAR) models to predict the anti-leishmanial activity of sulfonamide compounds against *Leishmania infantum* intracellular amastigotes. The study integrates cheminformatics, machine learning, and medicinal chemistry for neglected tropical disease drug discovery.

## Project Structure

```
PhD_QSAR_Leishmania/
├── data/
│   ├── raw/                  # Untouched data from ChEMBL/PubChem
│   ├── processed/            # Curated, clean datasets
│   └── external/             # Screening libraries (ZINC, Enamine)
├── notebooks/                # Jupyter notebooks (numbered sequentially)
│   ├── 01_data_collection.ipynb
│   ├── 02_data_curation.ipynb
│   ├── 03_descriptor_calculation.ipynb
│   ├── 04_feature_selection.ipynb
│   ├── 05_model_training_RF_SVM.ipynb
│   ├── 06_model_training_XGB_LGBM.ipynb
│   ├── 07_consensus_model.ipynb
│   ├── 08_model_validation.ipynb
│   ├── 09_SHAP_analysis.ipynb
│   ├── 10_virtual_screening.ipynb
│   └── 11_results_analysis.ipynb
├── src/                      # Reusable Python modules
├── models/                   # Saved trained models (.joblib)
├── results/                  # Output results and metrics
├── figures/                  # Publication-ready figures
└── papers/                   # Manuscript drafts organized by paper
```

## Pipeline Overview

1. **Data Collection** — Mine ChEMBL, PubChem, and literature for anti-kinetoplastid bioactivity data
2. **Data Curation** — Standardize structures, handle duplicates, filter sulfonamides
3. **Descriptor Calculation** — Mordred 2D descriptors, ECFP4/MACCS fingerprints
4. **Feature Selection** — Dual-filter (Mutual Information + RF importance)
5. **Model Training** — RF, SVM, XGBoost, LightGBM with Optuna tuning
6. **Consensus Model** — Voting ensemble (≥3/4 agreement)
7. **SHAP Analysis** — Explainable AI for SAR interpretation
8. **Virtual Screening** — Screen ZINC/Enamine with AD and ADMET filters
9. **Experimental Validation** — In vitro testing of top 10–20 candidates

## Setup Instructions

### 1. Install Miniconda
Download from: https://docs.conda.io/en/latest/miniconda.html

### 2. Create Environment
```bash
conda env create -f environment.yml
conda activate qsar-leish
```

### 3. Verify Installation
```bash
python -c "import rdkit; import sklearn; import xgboost; print('All packages OK')"
```

### 4. Launch Jupyter
```bash
jupyter lab
# Then open notebooks/01_data_collection.ipynb
```

## Key Targets

| Target | ChEMBL ID | Species |
|--------|-----------|---------|
| *L. infantum* | CHEMBL612848 | Primary target |
| *L. donovani* | CHEMBL367 | Cross-species |
| *L. amazonensis* | CHEMBL612877 | Cross-species |
| *T. cruzi* | CHEMBL368 | Cross-kinetoplastid |

## Activity Thresholds

- **Active**: pIC50 ≥ 5.0 (IC50 ≤ 10 μM)
- **Inactive**: pIC50 < 5.0 (IC50 > 10 μM)

## Publication Plan

1. **Paper 1** — Curated sulfonamide anti-kinetoplastid bioactivity dataset
2. **Paper 2** — ML-QSAR consensus models + SHAP-based SAR
3. **Paper 3** — Experimental validation of ML-predicted sulfonamides

## Software Stack (All Free & Open-Source)

- **Python 3.11** + Jupyter
- **RDKit** — Molecular handling, fingerprints, SMARTS
- **Mordred** — 1,800+ molecular descriptors
- **Scikit-learn** — RF, SVM, cross-validation
- **XGBoost / LightGBM** — Gradient boosting
- **Optuna** — Bayesian hyperparameter optimization
- **SHAP** — Explainable AI
- **imbalanced-learn** — SMOTE for class imbalance

## License

This project is part of a PhD thesis. Code is provided for academic use.

## Author

[Your Name] — [Your University]

---
*Prepared: May 2026*

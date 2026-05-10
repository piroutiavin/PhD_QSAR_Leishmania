"""Build notebook 05: RF + SVM training with Optuna, following Project Architecture Section 05."""
import json, uuid
from pathlib import Path

NB_PATH = Path(r'e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\05_model_training_RF_SVM.ipynb')

def code_cell(source, cell_id=None):
    return {"cell_type": "code", "execution_count": None,
            "id": cell_id or uuid.uuid4().hex[:8], "metadata": {},
            "outputs": [], "source": source}

def md_cell(source, cell_id=None):
    return {"cell_type": "markdown", "id": cell_id or uuid.uuid4().hex[:8],
            "metadata": {}, "source": source}

# ── Update markdown header ──────────────────────────────────────────────────
HEADER = """\
# Notebook 05 — Model Training: Random Forest & SVM with Optuna

**Project**: ML-Based QSAR Modeling for Anti-Leishmanial Sulfonamide Derivatives
**Input**: `data/processed/X_train_selected.csv`, `y_train.csv`
**Output**:
- `models/rf_classifier.joblib` — Optimised RF + SMOTE pipeline
- `models/svm_classifier.joblib` — Optimised SVM + SMOTE pipeline
- `results/model_metrics/cv_scores_nb05.csv` — 5-fold CV metrics for both models

**Pipeline**:
1. Load feature-selected training data
2. RF — Optuna optimisation (50 trials, 5-fold StratifiedKFold, SMOTE inside folds)
3. SVM — Optuna optimisation (30 trials, subsample 3 000 for speed, final fit on full data)
4. Save both pipelines as `.joblib`

**QC-5**: CV balanced accuracy > 0.70 for best model.

---"""

# ── Cell 1: Imports ──────────────────────────────────────────────────────────
CELL1 = """\
# ============================================================
# CELL 1: Imports and Configuration
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, matthews_corrcoef, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

(PROJECT_ROOT / 'models').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'results' / 'model_metrics').mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / 'figures').mkdir(parents=True, exist_ok=True)

print(f"Project root : {PROJECT_ROOT}")
print(f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
"""

# ── Cell 2: Load data ────────────────────────────────────────────────────────
CELL2 = """\
# ============================================================
# CELL 2: Load Feature-Selected Training Data
# ============================================================

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

X_train = pd.read_csv(DATA_DIR / 'X_train_selected.csv', index_col=0)

y_df = pd.read_csv(DATA_DIR / 'y_train.csv', index_col=0)
y_train = (y_df['activity_class'] == 'Active').astype(int)

n_active   = int(y_train.sum())
n_inactive = len(y_train) - n_active

print(f"X_train shape : {X_train.shape}")
print(f"y_train       : {len(y_train)} samples")
print(f"  Active (1)  : {n_active}  ({n_active/len(y_train)*100:.1f}%)")
print(f"  Inactive (0): {n_inactive}  ({n_inactive/len(y_train)*100:.1f}%)")
print(f"\\nClass imbalance ratio: 1 : {n_inactive/n_active:.1f}")
"""

# ── Cell 3: CV strategy + scoring ───────────────────────────────────────────
CELL3 = """\
# ============================================================
# CELL 3: Cross-Validation Strategy and Scoring
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

scoring = {
    'accuracy'          : 'accuracy',
    'balanced_accuracy' : 'balanced_accuracy',
    'f1'                : 'f1',
    'mcc'               : make_scorer(matthews_corrcoef),
    'roc_auc'           : 'roc_auc',
}

print("Cross-validation setup:")
print(f"  Strategy : StratifiedKFold (5 splits, shuffle=True)")
print(f"  Metrics  : accuracy, balanced_accuracy, F1, MCC, ROC-AUC")
print(f"  SMOTE    : applied inside each training fold (ImbPipeline)")
"""

# ── Cell 4: RF Optuna optimisation ──────────────────────────────────────────
CELL4 = """\
# ============================================================
# CELL 4: Random Forest — Optuna Hyperparameter Optimisation
# ============================================================

def objective_rf(trial):
    params = {
        'n_estimators'      : trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth'         : trial.suggest_int('max_depth', 5, 50),
        'min_samples_split' : trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf'  : trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features'      : trial.suggest_categorical('max_features',
                                                         ['sqrt', 'log2', 0.3]),
    }
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=SEED)),
        ('rf', RandomForestClassifier(**params, random_state=SEED, n_jobs=-1)),
    ])
    # n_jobs=1 in cross_validate avoids nested multiprocessing with RF n_jobs=-1
    result = cross_validate(pipeline, X_train, y_train, cv=cv,
                            scoring='balanced_accuracy', n_jobs=1)
    return result['test_score'].mean()

print("Starting RF optimisation (50 trials) …")
study_rf = optuna.create_study(direction='maximize',
                               sampler=optuna.samplers.TPESampler(seed=SEED))
study_rf.optimize(objective_rf, n_trials=50, show_progress_bar=False)

print(f"RF optimisation complete.")
print(f"  Best balanced accuracy (CV) : {study_rf.best_value:.4f}")
print(f"  Best params                 : {study_rf.best_params}")
"""

# ── Cell 5: Train final RF ───────────────────────────────────────────────────
CELL5 = """\
# ============================================================
# CELL 5: Train Final RF Pipeline + 5-Fold CV Evaluation
# ============================================================

best_rf_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=SEED)),
    ('rf', RandomForestClassifier(**study_rf.best_params, random_state=SEED, n_jobs=-1)),
])

# Full 5-fold CV with all metrics (for reporting)
print("Running 5-fold CV with full scoring on best RF …")
rf_cv_results = cross_validate(best_rf_pipeline, X_train, y_train,
                               cv=cv, scoring=scoring, n_jobs=1,
                               return_train_score=False)

rf_cv_mean = {k.replace('test_', ''): v.mean() for k, v in rf_cv_results.items()
              if k.startswith('test_')}
rf_cv_std  = {k.replace('test_', ''): v.std()  for k, v in rf_cv_results.items()
              if k.startswith('test_')}

print("RF 5-Fold CV Results:")
for metric, mean_val in rf_cv_mean.items():
    print(f"  {metric:<22}: {mean_val:.4f} ± {rf_cv_std[metric]:.4f}")

# Fit final model on full training set
print("\\nFitting final RF on full training set …")
best_rf_pipeline.fit(X_train, y_train)
joblib.dump(best_rf_pipeline, PROJECT_ROOT / 'models' / 'rf_classifier.joblib')
print("Saved → models/rf_classifier.joblib")
"""

# ── Cell 6: SVM Optuna optimisation ─────────────────────────────────────────
CELL6 = """\
# ============================================================
# CELL 6: SVM — Optuna Hyperparameter Optimisation
#         (subsample 3000 for speed; final fit on full data)
# ============================================================

from sklearn.model_selection import StratifiedShuffleSplit

# Subsample 3000 for SVM hyperparameter search (SVM is O(n²))
sss = StratifiedShuffleSplit(n_splits=1, test_size=None,
                              train_size=3000, random_state=SEED)
sub_idx, _ = next(sss.split(X_train, y_train))
X_sub = X_train.iloc[sub_idx]
y_sub = y_train.iloc[sub_idx]
print(f"SVM search subsample: {X_sub.shape}  "
      f"(active={y_sub.sum()}, inactive={len(y_sub)-y_sub.sum()})")

cv_sub = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def objective_svm(trial):
    params = {
        'C'      : trial.suggest_float('C', 0.01, 100, log=True),
        'gamma'  : trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        'kernel' : trial.suggest_categorical('kernel', ['rbf', 'linear']),
    }
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=SEED)),
        ('svm', SVC(**params, probability=True, random_state=SEED)),
    ])
    result = cross_validate(pipeline, X_sub, y_sub, cv=cv_sub,
                            scoring='balanced_accuracy', n_jobs=-1)
    return result['test_score'].mean()

print("Starting SVM optimisation (30 trials, timeout=900s) …")
study_svm = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
study_svm.optimize(objective_svm, n_trials=30, timeout=900,
                   show_progress_bar=False)

print(f"SVM optimisation complete.")
print(f"  Best balanced accuracy (CV, subsample): {study_svm.best_value:.4f}")
print(f"  Best params                            : {study_svm.best_params}")
"""

# ── Cell 7: Train final SVM ──────────────────────────────────────────────────
CELL7 = """\
# ============================================================
# CELL 7: Train Final SVM Pipeline + 5-Fold CV Evaluation
# ============================================================

best_svm_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=SEED)),
    ('svm', SVC(**study_svm.best_params, probability=True, random_state=SEED)),
])

# Full 5-fold CV on full training set with all metrics
print("Running 5-fold CV with full scoring on best SVM …")
svm_cv_results = cross_validate(best_svm_pipeline, X_train, y_train,
                                cv=cv, scoring=scoring, n_jobs=-1,
                                return_train_score=False)

svm_cv_mean = {k.replace('test_', ''): v.mean() for k, v in svm_cv_results.items()
               if k.startswith('test_')}
svm_cv_std  = {k.replace('test_', ''): v.std()  for k, v in svm_cv_results.items()
               if k.startswith('test_')}

print("SVM 5-Fold CV Results:")
for metric, mean_val in svm_cv_mean.items():
    print(f"  {metric:<22}: {mean_val:.4f} ± {svm_cv_std[metric]:.4f}")

# Fit final model on full training set
print("\\nFitting final SVM on full training set …")
best_svm_pipeline.fit(X_train, y_train)
joblib.dump(best_svm_pipeline, PROJECT_ROOT / 'models' / 'svm_classifier.joblib')
print("Saved → models/svm_classifier.joblib")
"""

# ── Cell 8: Save metrics CSV ─────────────────────────────────────────────────
CELL8 = """\
# ============================================================
# CELL 8: Save CV Metrics and Optuna Study Results
# ============================================================

metrics_dir = PROJECT_ROOT / 'results' / 'model_metrics'

# CV scores summary
cv_summary = pd.DataFrame([
    {'model': 'RF',  'n_trials': len(study_rf.trials),
     'best_optuna_BA': study_rf.best_value,
     **{f'cv_{k}': v for k, v in rf_cv_mean.items()},
     **{f'cv_{k}_std': v for k, v in rf_cv_std.items()}},
    {'model': 'SVM', 'n_trials': len(study_svm.trials),
     'best_optuna_BA': study_svm.best_value,
     **{f'cv_{k}': v for k, v in svm_cv_mean.items()},
     **{f'cv_{k}_std': v for k, v in svm_cv_std.items()}},
])
cv_summary.to_csv(metrics_dir / 'cv_scores_nb05.csv', index=False)
print("Saved → results/model_metrics/cv_scores_nb05.csv")

# Optuna histories
rf_history = study_rf.trials_dataframe()[['number', 'value', 'state']]
rf_history.columns = ['trial', 'balanced_accuracy', 'state']
rf_history['model'] = 'RF'

svm_history = study_svm.trials_dataframe()[['number', 'value', 'state']]
svm_history.columns = ['trial', 'balanced_accuracy', 'state']
svm_history['model'] = 'SVM'

optuna_history = pd.concat([rf_history, svm_history], ignore_index=True)
optuna_history.to_csv(metrics_dir / 'optuna_history_nb05.csv', index=False)
print("Saved → results/model_metrics/optuna_history_nb05.csv")

# Best hyperparameters
hp_df = pd.DataFrame([
    {'model': 'RF',  **study_rf.best_params},
    {'model': 'SVM', **study_svm.best_params},
])
hp_df.to_csv(metrics_dir / 'best_hyperparams_nb05.csv', index=False)
print("Saved → results/model_metrics/best_hyperparams_nb05.csv")
print()
print(cv_summary[['model', 'best_optuna_BA', 'cv_balanced_accuracy',
                   'cv_f1', 'cv_mcc', 'cv_roc_auc']].round(4).to_string(index=False))
"""

# ── Cell 9: Visualisation ────────────────────────────────────────────────────
CELL9 = """\
# ============================================================
# CELL 9: Visualisation — Optimisation History & CV Scores
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Panel 1: RF Optuna optimisation history ---
ax = axes[0]
rf_vals = [t.value for t in study_rf.trials if t.value is not None]
ax.plot(range(1, len(rf_vals)+1), rf_vals, 'o-', color='#4C72B0',
        alpha=0.6, markersize=4, linewidth=1, label='Trial')
ax.axhline(study_rf.best_value, color='red', linestyle='--',
           linewidth=1.5, label=f'Best={study_rf.best_value:.4f}')
ax.set_xlabel('Trial')
ax.set_ylabel('Balanced Accuracy (CV)')
ax.set_title('RF Optuna Optimisation')
ax.legend(fontsize=9)
sns.despine(ax=ax)

# --- Panel 2: SVM Optuna optimisation history ---
ax = axes[1]
svm_vals = [t.value for t in study_svm.trials if t.value is not None]
ax.plot(range(1, len(svm_vals)+1), svm_vals, 's-', color='#C44E52',
        alpha=0.6, markersize=4, linewidth=1, label='Trial')
ax.axhline(study_svm.best_value, color='navy', linestyle='--',
           linewidth=1.5, label=f'Best={study_svm.best_value:.4f}')
ax.set_xlabel('Trial')
ax.set_ylabel('Balanced Accuracy (CV, subsample)')
ax.set_title('SVM Optuna Optimisation')
ax.legend(fontsize=9)
sns.despine(ax=ax)

# --- Panel 3: CV metric comparison (bar chart) ---
ax = axes[2]
metrics_plot = ['balanced_accuracy', 'f1', 'mcc', 'roc_auc']
x = np.arange(len(metrics_plot))
width = 0.35
rf_vals_bar  = [rf_cv_mean[m]  for m in metrics_plot]
svm_vals_bar = [svm_cv_mean[m] for m in metrics_plot]
rf_err  = [rf_cv_std[m]  for m in metrics_plot]
svm_err = [svm_cv_std[m] for m in metrics_plot]

ax.bar(x - width/2, rf_vals_bar,  width, yerr=rf_err,  capsize=4,
       color='#4C72B0', alpha=0.85, label='RF')
ax.bar(x + width/2, svm_vals_bar, width, yerr=svm_err, capsize=4,
       color='#C44E52', alpha=0.85, label='SVM')
ax.axhline(0.70, color='green', linestyle='--', linewidth=1.2,
           label='QC-5 threshold (0.70 BA)')
ax.set_xticks(x)
ax.set_xticklabels(['Bal. Acc.', 'F1', 'MCC', 'ROC-AUC'])
ax.set_ylabel('Score (5-fold CV)')
ax.set_title('RF vs SVM — CV Performance')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
sns.despine(ax=ax)

plt.tight_layout()
fig_path = PROJECT_ROOT / 'figures' / 'nb05_training_summary.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figure saved → figures/nb05_training_summary.png")
"""

# ── Cell 10: QC-5 ───────────────────────────────────────────────────────────
CELL10 = """\
# ============================================================
# CELL 10: QC-5 — Training Quality Checkpoint
# ============================================================

print("=" * 55)
print("QC-5: Model Training Audit")
print("=" * 55)

best_ba_rf  = rf_cv_mean['balanced_accuracy']
best_ba_svm = svm_cv_mean['balanced_accuracy']
best_ba     = max(best_ba_rf, best_ba_svm)

print(f"RF  CV Balanced Accuracy : {best_ba_rf:.4f} ± {rf_cv_std['balanced_accuracy']:.4f}")
print(f"SVM CV Balanced Accuracy : {best_ba_svm:.4f} ± {svm_cv_std['balanced_accuracy']:.4f}")
print(f"Best model (BA)          : {'RF' if best_ba_rf >= best_ba_svm else 'SVM'} ({best_ba:.4f})")

qc5_pass = best_ba >= 0.70
print(f"\\nQC-5 threshold ≥ 0.70   : {'PASS ✓' if qc5_pass else 'WARN — below threshold'}")
assert qc5_pass, f"QC-5 FAILED: best balanced accuracy {best_ba:.4f} < 0.70"

# Check model files exist
rf_path  = PROJECT_ROOT / 'models' / 'rf_classifier.joblib'
svm_path = PROJECT_ROOT / 'models' / 'svm_classifier.joblib'
assert rf_path.exists(),  "rf_classifier.joblib not found!"
assert svm_path.exists(), "svm_classifier.joblib not found!"
print(f"[OK] models/rf_classifier.joblib  ({rf_path.stat().st_size//1024} KB)")
print(f"[OK] models/svm_classifier.joblib ({svm_path.stat().st_size//1024} KB)")

# Best hyperparameters summary
print(f"\\nBest RF params  : {study_rf.best_params}")
print(f"Best SVM params : {study_svm.best_params}")

print("=" * 55)
print("QC-5 PASSED — RF and SVM models ready")
print("  Next: notebook 06 — XGBoost + LightGBM training")
print("=" * 55)
"""

# ── Assemble notebook ────────────────────────────────────────────────────────
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'] = [
    md_cell(HEADER,  cell_id='afb2bb71'),   # keep original id
    code_cell(CELL1, cell_id='3dd055b6'),   # replace placeholder
    code_cell(CELL2),
    code_cell(CELL3),
    code_cell(CELL4),
    code_cell(CELL5),
    code_cell(CELL6),
    code_cell(CELL7),
    code_cell(CELL8),
    code_cell(CELL9),
    code_cell(CELL10),
]

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {len(nb['cells'])} cells")
for i, c in enumerate(nb['cells']):
    print(f"  Cell {i}: {c['id']} [{c['cell_type']}]")

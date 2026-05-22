"""Create notebook 09: SHAP Analysis (Explainable AI)."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

NB_PATH = Path(r"e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\09_SHAP_analysis.ipynb")

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": [line + "\n" for line in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "metadata": {},
                  "source": [line + "\n" for line in source.split("\n")],
                  "outputs": [], "execution_count": None})

# ---------- Header ----------
md("""# Notebook 09 — SHAP Analysis (Explainable AI)

**Project**: ML-Based QSAR Modeling for Anti-Leishmanial Sulfonamide Derivatives
**Input**: Trained XGBoost model, selected features, training/test data
**Output**:
- `figures/shap_summary_bar.png` — Global feature importance (top 20)
- `figures/shap_summary_beeswarm.png` — Feature value impact direction
- `figures/shap_dependence_*.png` — Dependence plots for top 5 features
- `figures/shap_waterfall_compound.png` — Individual compound explanation
- `results/shap_outputs/shap_feature_importance.csv` — Full SHAP importance table

**Sections**:
1. Global SHAP feature importance (bar + beeswarm)
2. SHAP dependence plots (top 5 features)
3. Individual compound waterfall explanations
4. SAR guidelines — features promoting vs reducing activity

---""")

# ---------- Cell 1: Imports ----------
code("""# ============================================================
# CELL 1: Imports and Configuration
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings, sys, joblib
from datetime import datetime
warnings.filterwarnings('ignore')

import shap

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
DATA = PROJECT_ROOT / 'data' / 'processed'
MODELS = PROJECT_ROOT / 'models'
FIGURES = PROJECT_ROOT / 'figures'
SHAP_DIR = PROJECT_ROOT / 'results' / 'shap_outputs'

for d in [FIGURES, SHAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"SHAP version: {shap.__version__}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")""")

# ---------- Cell 2: Load Data & Model ----------
code("""# ============================================================
# CELL 2: Load Data and Best Tree Model
# ============================================================

X_train = pd.read_csv(DATA / 'X_train_selected.csv', index_col=0)
X_test  = pd.read_csv(DATA / 'X_test_selected.csv',  index_col=0)
y_train_df = pd.read_csv(DATA / 'y_train.csv', index_col=0)
y_test_df  = pd.read_csv(DATA / 'y_test.csv',  index_col=0)

common_tr = X_train.index.intersection(y_train_df.index)
common_te = X_test.index.intersection(y_test_df.index)
X_train = X_train.loc[common_tr]
X_test  = X_test.loc[common_te]
y_train = (y_train_df.loc[common_tr, 'activity_class'] == 'Active').astype(int)
y_test  = (y_test_df.loc[common_te,  'activity_class'] == 'Active').astype(int)

# Use XGBoost — TreeExplainer is exact and fast for tree models
xgb_model = joblib.load(MODELS / 'xgb_classifier.joblib')

print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"Model: XGBoost ({type(xgb_model).__name__})")
print(f"Features: {X_train.shape[1]}")""")

# ---------- Cell 3: Compute SHAP values ----------
md("""## Part A — Compute SHAP Values

Using TreeExplainer (exact, fast for tree-based models).""")

code("""# ============================================================
# CELL 3: Compute SHAP Values
# ============================================================

print("Computing SHAP values with TreeExplainer...")
print("  This is exact (not approximate) for tree-based models.\\n")

explainer = shap.TreeExplainer(xgb_model)

# SHAP for training set (for global importance)
shap_values_train = explainer.shap_values(X_train)
print(f"SHAP values (train) shape: {shap_values_train.shape}")

# SHAP for test set (for individual explanations)
shap_values_test = explainer.shap_values(X_test)
print(f"SHAP values (test)  shape: {shap_values_test.shape}")

print(f"Expected value (base rate): {explainer.expected_value:.4f}")""")

# ---------- Cell 4: Summary Bar Plot ----------
md("""## Part B — Global Feature Importance (SHAP Summary)""")

code("""# ============================================================
# CELL 4: SHAP Summary Bar Plot — Top 20 Features
# ============================================================

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_train, X_train, plot_type='bar',
                  max_display=20, show=False)
plt.title('SHAP Feature Importance — Top 20 (XGBoost)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(FIGURES / 'shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved: figures/shap_summary_bar.png")""")

# ---------- Cell 5: Beeswarm Plot ----------
code("""# ============================================================
# CELL 5: SHAP Beeswarm Plot — Feature Value Impact
# ============================================================

fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values_train, X_train, max_display=20, show=False)
plt.title('SHAP Beeswarm — Feature Value vs Impact (XGBoost)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(FIGURES / 'shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved: figures/shap_summary_beeswarm.png")

print("\\nHow to read the beeswarm plot:")
print("  - Each dot = one compound")
print("  - X-axis = SHAP value (positive = pushes toward Active)")
print("  - Color = feature value (red = high, blue = low)")
print("  - Width = density of compounds at that SHAP value")""")

# ---------- Cell 6: Dependence Plots ----------
md("""## Part C — SHAP Dependence Plots (Top 5 Features)

Shows how individual feature values affect the prediction, with automatic interaction detection.""")

code("""# ============================================================
# CELL 6: SHAP Dependence Plots — Top 5 Features
# ============================================================

# Identify top 5 features by mean |SHAP|
mean_abs_shap = np.abs(shap_values_train).mean(axis=0)
top5_idx = np.argsort(mean_abs_shap)[-5:][::-1]
top5_features = X_train.columns[top5_idx].tolist()

print("Top 5 features by mean |SHAP|:")
for i, feat in enumerate(top5_features):
    print(f"  {i+1}. {feat} (mean |SHAP| = {mean_abs_shap[top5_idx[i]]:.4f})")

fig, axes = plt.subplots(1, 5, figsize=(25, 5))

for idx, (feat, ax) in enumerate(zip(top5_features, axes)):
    feat_idx = X_train.columns.get_loc(feat)
    ax.scatter(X_train[feat].values, shap_values_train[:, feat_idx],
               c=X_train[feat].values, cmap='coolwarm', alpha=0.3, s=5, edgecolors='none')
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel('SHAP value' if idx == 0 else '')
    ax.set_title(f'#{idx+1}: {feat}', fontsize=10)
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
    sns.despine(ax=ax)

plt.suptitle('SHAP Dependence Plots — Top 5 Features', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES / 'shap_dependence_top5.png', dpi=300, bbox_inches='tight')
plt.show()
print("\\nFigure saved: figures/shap_dependence_top5.png")""")

# ---------- Cell 7: Waterfall ----------
md("""## Part D — Individual Compound Explanation (Waterfall)

Explains why a specific compound was predicted as Active or Inactive.""")

code("""# ============================================================
# CELL 7: Waterfall Plots — Individual Compound Explanations
# ============================================================

# Find the most confidently Active and most confidently Inactive predictions
y_prob_test = xgb_model.predict_proba(X_test)[:, 1]

most_active_idx = np.argmax(y_prob_test)
most_inactive_idx = np.argmin(y_prob_test)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax_idx, (comp_idx, label) in enumerate([
    (most_active_idx, 'Most Confident Active'),
    (most_inactive_idx, 'Most Confident Inactive')
]):
    explanation = shap.Explanation(
        values=shap_values_test[comp_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[comp_idx].values,
        feature_names=X_test.columns.tolist()
    )

    plt.sca(axes[ax_idx])
    shap.plots.waterfall(explanation, max_display=15, show=False)
    actual = 'Active' if y_test.iloc[comp_idx] == 1 else 'Inactive'
    prob = y_prob_test[comp_idx]
    axes[ax_idx].set_title(f'{label}\\n(Actual: {actual}, P(Active)={prob:.3f})', fontsize=11)

plt.tight_layout()
plt.savefig(FIGURES / 'shap_waterfall_compound.png', dpi=300, bbox_inches='tight')
plt.show()
print("Figure saved: figures/shap_waterfall_compound.png")

print(f"\\nMost confident Active:   index={most_active_idx}, P(Active)={y_prob_test[most_active_idx]:.4f}")
print(f"Most confident Inactive: index={most_inactive_idx}, P(Active)={y_prob_test[most_inactive_idx]:.4f}")""")

# ---------- Cell 8: SAR Guidelines ----------
md("""## Part E — SAR Guidelines: Features Promoting vs Reducing Activity

Translate SHAP importance into actionable structure-activity relationship insights.""")

code("""# ============================================================
# CELL 8: SAR Guidelines — Positive vs Negative Drivers
# ============================================================

mean_shap_df = pd.DataFrame({
    'feature': X_train.columns,
    'mean_shap': shap_values_train.mean(axis=0),
    'abs_mean_shap': np.abs(shap_values_train).mean(axis=0),
    'std_shap': shap_values_train.std(axis=0),
}).sort_values('abs_mean_shap', ascending=False)

# Classify feature types
def classify_feature(name):
    if name.startswith('ECFP6_'):
        return 'ECFP6 Fingerprint'
    elif name.startswith('MACCS_'):
        return 'MACCS Key'
    elif name.startswith('mordred_'):
        return 'Mordred 2D'
    elif name.startswith('rdkit_'):
        return 'RDKit 2D'
    else:
        return 'Other'

mean_shap_df['type'] = mean_shap_df['feature'].apply(classify_feature)

# Top 10 promoting activity (positive SHAP = pushes toward Active)
positive = mean_shap_df[mean_shap_df['mean_shap'] > 0].head(10)
negative = mean_shap_df[mean_shap_df['mean_shap'] < 0].head(10)

print("=" * 60)
print("TOP 10 FEATURES PROMOTING ACTIVITY (positive SHAP)")
print("  Higher values of these features push predictions toward Active")
print("=" * 60)
print(positive[['feature', 'type', 'mean_shap', 'abs_mean_shap']].to_string(index=False))

print()
print("=" * 60)
print("TOP 10 FEATURES REDUCING ACTIVITY (negative SHAP)")
print("  Higher values of these features push predictions toward Inactive")
print("=" * 60)
print(negative[['feature', 'type', 'mean_shap', 'abs_mean_shap']].to_string(index=False))

# Feature type breakdown
print()
print("=" * 60)
print("Feature Type Breakdown (Top 50 most important)")
print("=" * 60)
top50_types = mean_shap_df.head(50)['type'].value_counts()
for t, n in top50_types.items():
    print(f"  {t:<20}: {n} features")

# Save full table
mean_shap_df.to_csv(SHAP_DIR / 'shap_feature_importance.csv', index=False)
print(f"\\nSaved: results/shap_outputs/shap_feature_importance.csv ({len(mean_shap_df)} features)")""")

# ---------- Cell 9: Summary Figure ----------
md("""## Summary Visualization""")

code("""# ============================================================
# CELL 9: Summary — Feature Type Importance Distribution
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Top 15 features bar chart
top15 = mean_shap_df.head(15).copy()
colors_map = {
    'ECFP6 Fingerprint': '#E91E63',
    'MACCS Key': '#FF9800',
    'Mordred 2D': '#2196F3',
    'RDKit 2D': '#4CAF50',
    'Other': '#9E9E9E'
}
bar_colors = [colors_map.get(t, '#9E9E9E') for t in top15['type']]

ax = axes[0]
y_pos = np.arange(len(top15))
ax.barh(y_pos, top15['abs_mean_shap'].values, color=bar_colors, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(top15['feature'].values, fontsize=8)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('Top 15 Features by SHAP Importance')
ax.invert_yaxis()
sns.despine(ax=ax)

# Panel B: Feature type contribution (pie chart)
ax = axes[1]
type_importance = mean_shap_df.groupby('type')['abs_mean_shap'].sum()
type_importance = type_importance.sort_values(ascending=False)
pie_colors = [colors_map.get(t, '#9E9E9E') for t in type_importance.index]
ax.pie(type_importance.values, labels=type_importance.index, colors=pie_colors,
       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
ax.set_title('Total SHAP Importance by Feature Type')

plt.suptitle('Notebook 09 — SHAP Explainability Summary', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES / 'nb09_shap_summary.png', dpi=200, bbox_inches='tight')
plt.show()
print("Figure saved: figures/nb09_shap_summary.png")""")

# ---------- Cell 10: QC-9 ----------
md("""## Quality Control Checkpoint""")

code("""# ============================================================
# CELL 10: QC-9 — SHAP Analysis Audit
# ============================================================

print("=" * 60)
print("QUALITY CONTROL CHECKPOINT QC-9: SHAP Analysis")
print("=" * 60)

checks = {}

# Check 1: SHAP values computed for all samples
checks[f'SHAP computed for all train samples ({shap_values_train.shape[0]})'] = (
    shap_values_train.shape[0] == X_train.shape[0]
)

# Check 2: SHAP values have correct feature dimension
checks[f'SHAP feature dimension matches ({shap_values_train.shape[1]} features)'] = (
    shap_values_train.shape[1] == X_train.shape[1]
)

# Check 3: No NaN in SHAP values
checks['No NaN in SHAP values'] = not np.isnan(shap_values_train).any()

# Check 4: Feature importance file saved
shap_file = SHAP_DIR / 'shap_feature_importance.csv'
checks[f'SHAP importance file saved ({shap_file.stat().st_size//1024} KB)'] = shap_file.exists()

# Check 5: Figures saved
for fig_name in ['shap_summary_bar.png', 'shap_summary_beeswarm.png',
                 'shap_dependence_top5.png', 'shap_waterfall_compound.png']:
    checks[f'Figure: {fig_name}'] = (FIGURES / fig_name).exists()

all_pass = True
for check, passed in checks.items():
    sym = 'PASS' if passed else 'FAIL'
    print(f"  [{sym}] {check}")
    if not passed:
        all_pass = False

print("=" * 60)
if all_pass:
    print("QC-9 PASSED — SHAP analysis complete")
else:
    print("QC-9: Some checks need review")

print(f"\\nTop 3 most important features:")
for i, row in mean_shap_df.head(3).iterrows():
    print(f"  {row['feature']} ({row['type']}, |SHAP|={row['abs_mean_shap']:.4f})")
print("=" * 60)""")

# ---------- Build notebook ----------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "PhD QSAR Leishmania (Python 3.14)",
            "language": "python",
            "name": "qsar-leish"
        },
        "language_info": {"name": "python", "version": "3.14.0"}
    },
    "cells": cells
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created: {NB_PATH}")
print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")

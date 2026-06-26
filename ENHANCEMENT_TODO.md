# Enhancement To-Do — Sulfonamide Per-Organism QSAR Panel

**Status:** Approved 2026-06-26. Not yet implemented.
**Goal:** Rebuild the project so scope, data, and reported performance match the stated aim
(anti-leishmanial sulfonamide derivatives, *L. infantum* primary).

## Why this rebuild is needed (findings)
- **Scope mismatch:** current models train on the full 14,431-row dataset, which is ~50% *T. cruzi*,
  only ~12% *L. infantum*, and only ~5.6% actual sulfonamides. `filter_sulfonamides()` exists but is never called.
- **Data leakage:** the random stratified split puts **28.9% of test molecules also in the training set**
  (same compound across organisms) → inflated metrics (test MCC≈0.51, AUC≈0.83).
- **Label conflicts:** 818 molecules carry both Active and Inactive labels (cross-organism pooling).

---

## Phase 1 — Dataset rebuild
- [ ] 1.1 Apply `filter_sulfonamides()` to curated data → sulfonamide-only set (~643 molecules)
- [ ] 1.2 Split into 4 per-organism datasets: *L. infantum* (137, primary), *L. donovani* (230), *T. cruzi* (362), *L. amazonensis* (72)
- [ ] 1.3 Resolve duplicate/conflicting labels **within each organism** (one label per molecule)
- [ ] 1.4 Fix curation gaps: tautomer canonicalization; prefer existing `pchembl_value`; IC50/EC50 per decision below
- [ ] 1.5 Report real per-organism counts (molecules, actives/inactives, scaffolds) before modeling

## Phase 2 — Leakage-free, small-data modeling (per organism)
- [ ] 2.1 Scaffold / grouped-by-molecule splitter — no molecule or scaffold spans train & test
- [ ] 2.2 Aggressive feature reduction: 1,164 → ~5–15 descriptors (correlation filter → MI/RF, capped by sample size)
- [ ] 2.3 Replace single held-out test with nested / repeated stratified CV; report mean ± std
- [ ] 2.4 Lighter models + modest hyperparameter search (avoid overfitting Optuna on small n)
- [ ] 2.5 Retain SHAP for SAR interpretation on final per-organism model

## Phase 3 — Validation & honest reporting
- [ ] 3.1 Re-run validation suite (Williams plot / applicability domain, Y-randomization) per organism
- [ ] 3.2 Report metrics honestly (expect a drop from the current leaky AUC≈0.83 — that is correct)
- [ ] 3.3 Rewrite README/title to match true scope; fill author/affiliation placeholders
- [ ] 3.4 Add cross-species SAR comparison across the 4 organism models

## Phase 4 — Reproducibility cleanup
- [ ] 4.1 Consolidate/remove ad-hoc `patch_*.py` / `create_nb*.py` scaffolding scripts
- [ ] 4.2 Document the new end-to-end run order in README

---

## Decision — IC50 vs EC50: HYBRID (pool for classifier + sensitivity check)
These are **whole-cell phenotypic** anti-kinetoplastid assays where IC50 and EC50 denote the same
operational quantity (50% parasite growth/viability reduction), so the label is mostly terminology.
The dominant noise source is combining across different **assays/labs**, not the IC50-vs-EC50 label
(Landrum & Riniker, *JCIM* 2024, doi:10.1021/acs.jcim.4c00049; Kalliokoski et al., *PLOS One* 2013).

Implementation:
- Pool IC50+EC50 for the binary Active/Inactive classifier (task is classification at pIC50 ≥ 5, not regression).
- Keep/tag the `standard_type` column so pooling is reversible and auditable.
- Verify IC50-vs-EC50 agreement on shared compounds within each organism; confirm EC50 records are
  whole-cell parasite assays (not host cytotoxicity / a different endpoint).
- Report an IC50-only sensitivity analysis (with vs without EC50) for reviewers.
- Prefer single-assay / maximal-curation matching where feasible to limit source-mixing noise.

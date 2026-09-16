# Insulin-Resistance and Lipid-Related Indices and In-Hospital Mortality in Critically Ill Patients with CMM

Analysis code for the study comparing eight insulin-resistance (IR) and lipid-related indices
(TyG, SPISE, TG/HDL-C, METS-IR, AIP, TyG-BMI, TyG-RC, CHG) for their associations with
in-hospital mortality in critically ill adults with cardiometabolic multimorbidity (CMM),
using **eICU-CRD** (development cohort) and **MIMIC-IV** (external validation cohort).

## Repository structure

```
sql/
  01-population-eicu.sql       # eICU-CRD cohort construction & variable extraction
  01-population-mimic.sql      # MIMIC-IV cohort construction & variable extraction
R/
  main_analysis_eicu.R         # eICU main pipeline: imputation (mice), covariate selection
                               #   (LASSO + VIF), Cox models pooled with Rubin's rules,
                               #   adjusted survival curves (G-computation), RCS,
                               #   sensitivity analyses (complete-case, logistic,
                               #   time-dependent Cox, clinically prespecified covariate sets),
                               #   subgroup analyses, PH tests
  11_ml_section_eicu.R         # leakage-free ML data pipeline: 7:3 stratified split first,
                               #   within-partition imputation, feature selection
                               #   (LASSO lambda.min / Boruta / RFE) on the training set only
  11b_ml_selection_figures.R   # feature-selection figures (LASSO paths, Boruta, RFE, Venn)
  main_analysis_mimic.R        # MIMIC-IV cohort analysis (same framework)
  external_validation_mimic.R  # external validation of the locked prediction model
python/
  ml_modeling_evaluation.py    # model training & evaluation: 7 algorithms, 5-fold CV grid
                               #   search, training-set Youden thresholds (locked), PR curves
                               #   / AUPRC, calibration intercept/slope/Brier, SHAP
```

## Reproduction notes

- **Software**: R 4.4.1 (survival, rms, mice, glmnet, Boruta, caret, pROC, shapviz);
  Python 3.12.10 (scikit-learn, statsmodels, xgboost, lightgbm, imbalanced-learn, shap).
- **Random seeds**: R pipeline `set.seed(20260531)`; Python `random_state = 123`.
- **Run order**: SQL extraction (eICU / MIMIC) → `main_analysis_eicu.R` →
  `11_ml_section_eicu.R` → `python/ml_modeling_evaluation.py` →
  `main_analysis_mimic.R` + `external_validation_mimic.R`.
- **Data availability**: eICU-CRD and MIMIC-IV are restricted-access critical care databases
  available via PhysioNet (https://physionet.org/) after completion of the required training
  and data-use agreement. No patient-level data are included in this repository.
- The final prediction model is an unpenalized logistic regression (no tuned hyperparameters);
  comparator algorithms were tuned by 5-fold cross-validated grid search within the training set.

## License

MIT License (code only; no data are distributed).

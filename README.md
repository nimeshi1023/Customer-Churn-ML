# Customer Churn Prediction: README

End-to-end churn prediction project: data audit, EDA, feature engineering, modelling, evaluation, a top-20% retention simulation and a business report.

## 1. Project goal

A subscription provider can contact only a limited share of customers per retention cycle. The project therefore focuses on **ranking customers by churn risk** and explaining the ranking, not on accuracy alone.

## 2. Deliverables

| File | Description |
|---|---|
| `01_data_science_notebook.ipynb` | Full analysis: audit, EDA, features, models, tuning, evaluation, simulation, export |
| `02_cleaned_dataset.csv` | Processed dataset including the 7 engineered features (written by the notebook) |
| `03_model_artifact.joblib` | Saved tuned Random Forest pipeline (preprocessing + model) |
| `04_business_report.pdf` | 8-page business report |
| `05_README.md` | This file |

## 3. Folder layout

The notebook uses relative paths, so keep this structure:

```
project/
  notebooks/01_data_science_notebook.ipynb
  data/customer_churn_advanced_dataset.csv     (input, 1,200 rows x 20 columns)
  data/02_cleaned_dataset.csv                  (output)
  models/03_model_artifact.joblib              (output)
  reports/04_business_report.pdf
  README.md
```

Run the notebook from inside `notebooks/` so that `../data` and `../models` resolve. Create the `models/` folder first if it does not exist.

## 4. Setup

Python 3.13.5 was used (from the notebook metadata).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib jupyter
```

Package versions were not recorded in the notebook. After a successful run, record them with:

```bash
pip freeze > requirements.txt
```

## 5. Execution steps

1. Place `customer_churn_advanced_dataset.csv` in `data/`.
2. Start Jupyter from `notebooks/` and open `01_data_science_notebook.ipynb`.
3. Choose **Kernel > Restart & Run All**.
4. Check that `data/02_cleaned_dataset.csv` and `models/03_model_artifact.joblib` were created.

The notebook runs in this order: load data, audit, EDA, feature engineering, stratified split, preprocessing pipelines, baseline models, hyperparameter search, evaluation, intervention simulation, export.

## 6. Reproducibility

- Random seed: `42` for the train/test split, the models and the search.
- Split: stratified on `churn`, 80% train (960) / 20% test (240).
- Preprocessing is inside a scikit-learn `Pipeline` / `ColumnTransformer`, fitted on training data only.

## 7. Data summary

- 1,200 customers, 20 columns, no duplicate rows or customer IDs.
- Missing values: `satisfaction_score` 4.0%, `monthly_usage_gb` 3.0%, `monthly_charges` 2.5%, `total_charges` 2.0%. No rows were removed; median imputation is used inside the pipeline.
- Target `churn`: 86.3% churned, 13.7% retained (highly imbalanced toward churn).
- `customer_id` is an identifier and is excluded from features.

## 8. Method summary

**Preprocessing:** numeric = median imputer + StandardScaler; categorical = most-frequent imputer + OneHotEncoder (unknown categories ignored).

**Engineered features:** `charge_per_tenure`, `payment_risk`, `complaint_intensity`, `has_late_payment`, `has_complaint`, `low_satisfaction`, `high_usage`.

**Models:** Logistic Regression and Random Forest (300 trees) as baselines; Random Forest tuned with `RandomizedSearchCV` (5 candidates, 3-fold CV, ROC-AUC scoring). Best parameters: `n_estimators=100`, `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=2`, `class_weight="balanced"`.

**Imbalance:** `class_weight="balanced"` in the tuned forest; no resampling.

## 9. Key results

| Model (test set, threshold 0.50) | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8917 | 0.9022 | 0.9807 | 0.9398 | 0.9171 | 0.9833 |
| Random Forest (baseline) | 0.8625 | 0.8655 | 0.9952 | 0.9258 | 0.8878 | 0.9764 |
| Random Forest (tuned) | 0.900* | 0.965* | 0.918* | 0.941* | not printed | 0.9808 |

\* Calculated from the tuned model's confusion matrix (TN 26, FP 7, FN 17, TP 190).

Top-20% targeting: 48 customers contacted, all churners; capture rate 0.2319 (48 of 207 churners) versus about 41.4 expected at random (lift about 1.16x).

## 10. Assumptions

- **Cost-benefit (hypothetical, not company data):** contact cost 5 per customer, value of a retained customer 100, 10% of contacted churners retained. Under these, the break-even churn probability is 0.50.
- **Contact capacity:** the company can contact only the top 20% of customers by predicted churn probability.
- **Missing values:** treated as missing at random and imputed with the training-set median (numeric) or most frequent value (categorical).
- **Outliers:** extreme values (for example monthly charges up to 235.01) are treated as legitimate and kept.
- Model probabilities are used as a ranking score. The calibration curve shows the tuned forest under-states churn probability in the upper range, so calibrate before using probabilities as true risk.
- Feature importance describes prediction, not causation.

## 11. Known limitations and open work

- Only two model families were built; the brief asks for at least three (add HistGradientBoosting or XGBoost/LightGBM).
- The tuning run was light (5 candidates, 3 folds); the planned 20-iteration, 5-fold search was overwritten.
- Logistic Regression was not tuned or class-weighted.
- The threshold analysis list (0.10 to 0.90) was defined but no precision/recall table was produced.
- SHAP and permutation importance were set up but no results were printed; no customer-level explanations exist yet.
- No fold-to-fold variation or outlier-sensitivity robustness check.
- The `high_usage` cut-off is computed on the full dataset (mild leakage); move it into the training fold.
- Missing satisfaction/usage values become 0 in the `low_satisfaction` and `high_usage` flags.
- The test set was used to compare baseline models, not held back for one final evaluation.
- Package versions are not recorded.
- Churn at 86% is unusually high; confirm label direction and population with the data owner.

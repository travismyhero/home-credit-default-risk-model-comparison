# Home Credit Default Risk: Five-Model Comparison

This repository contains a reproducible credit-risk classification project based on Kaggle's **Home Credit Default Risk** application data. It compares:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

The workflow covers data audit, leakage-safe preprocessing, business feature engineering, feature selection, hyperparameter tuning, validation-only threshold selection, test evaluation, calibration, risk deciles, interpretability, and reproducible reporting.

## Final Report

[Download the formal final report (PDF)](outputs/home_credit_model_comparison_report_formal.pdf)

The supplementary Markdown report is available at [outputs/model_comparison_report.md](outputs/model_comparison_report.md).
The formal PDF, its English figures, and the retained LaTeX source can be rebuilt with:

```bash
python generate_home_credit_report_formal.py
```

This command requires XeLaTeX in addition to the Python dependencies.

## Notebooks

- [Source notebook](home_credit_model_comparison.ipynb): clean, sequentially runnable notebook.
- [Executed notebook](home_credit_model_comparison.executed.ipynb): full-data run with outputs.

## Main Results

| Model | Test AUC | KS | PR-AUC | F1 |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.7716 | 0.4072 | 0.2587 | 0.3184 |
| LightGBM | 0.7702 | 0.4064 | 0.2555 | 0.3168 |
| Logistic Regression | 0.7553 | 0.3790 | 0.2314 | 0.2987 |
| Random Forest | 0.7538 | 0.3802 | 0.2290 | 0.2984 |
| Decision Tree | 0.7231 | 0.3247 | 0.2010 | 0.2745 |

The full experiment uses 307,511 observations, retains 108 raw/engineered features after training-only selection, and produces 232 encoded model features. The `outputs/` directory contains model artifacts, result tables, calibration and decile analyses, and 70 figures.

## Data

Download `application_train.csv` from the [Kaggle Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place it at:

```text
data/application_train.csv
```

The raw Kaggle data is intentionally excluded from Git.

## Environment

```bash
python -m pip install -r requirements.txt
```

For a quick end-to-end validation:

```bash
HOME_CREDIT_DEBUG=1 jupyter nbconvert \
  --to notebook --execute home_credit_model_comparison.ipynb \
  --output home_credit_model_comparison.debug.executed.ipynb \
  --ExecutePreprocessor.timeout=3600
```

Run the source notebook normally for the complete experiment.

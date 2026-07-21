# Home Credit Default Risk: Five-Model Comparison

This repository contains a reproducible credit-risk classification project based on Kaggle's **Home Credit Default Risk** application data. It compares:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

The workflow covers data audit, training-fitted preprocessing, business feature engineering, limited parameter screening, final refitting, probability calibration, threshold selection, statistical uncertainty, segment diagnostics, random-split PSI, risk deciles, and illustrative cost sensitivity.

## Final Report

[Download the formal final report (PDF)](outputs/home_credit_model_comparison_report_formal.pdf)

The supplementary Markdown report is available at [outputs/model_comparison_report.md](outputs/model_comparison_report.md).
The formal PDF and its English figures can be rebuilt from the current CSV and JSON evidence with:

```bash
python generate_home_credit_report_formal.py
```

The report uses ReportLab and does not require a separate LaTeX installation.

## Notebooks

- [Source notebook](home_credit_model_comparison.ipynb): clean, sequentially runnable notebook.
- [Executed notebook](home_credit_model_comparison.executed.ipynb): full-data run with outputs.

## Evidence Produced

The current run is recorded in machine-readable files rather than duplicated as fixed numbers in this README:

- [Final model metrics](outputs/model_metrics.csv)
- [Bootstrap metric intervals](outputs/bootstrap_metric_intervals.csv)
- [Paired AUC bootstrap differences](outputs/paired_bootstrap_auc_differences.csv)
- [DeLong AUC test](outputs/delong_auc_test.csv)
- [Calibration diagnostics](outputs/calibration_diagnostics.csv)
- [Customer-segment performance](outputs/segment_performance.csv)
- [Random-split PSI](outputs/random_split_psi.csv)
- [Illustrative cost sensitivity](outputs/cost_sensitivity.csv)

Training, model-selection, calibration, threshold, and final-test samples are mutually exclusive. Boosted models use early stopping only during candidate screening; the selected iteration count is then fitted on the complete training partition so that reported training time has the same definition across model families. LightGBM row sampling is activated explicitly with `subsample_freq=1`.

## Evidence Limits

This is a development benchmark on a public application table, not a production model validation. The data lacks a reliable booking timestamp, rejected-applicant outcomes, long-run portfolio default rates, and lender-specific LGD, EAD, revenue, funding-cost, and capital inputs. The repository therefore does not claim out-of-time stability, reject inference, through-the-cycle PD calibration, expected-loss optimization, or regulatory capital impact. Historical bureau and repayment-table aggregation is not implemented.

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
  --ExecutePreprocessor.timeout=7200
```

Run the source notebook normally for the complete experiment.

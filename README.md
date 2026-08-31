# Home Credit Default Risk: Five-Model Comparison

## Project Question

I built this project to answer a practical modeling question: if the same Home Credit application data is given to five common classifiers, how much does model complexity improve risk ranking, and is the improvement large enough to justify the added implementation cost?

The comparison includes Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM. I use the application table only. That keeps the experiment focused on model choice rather than turning it into a competition-style feature aggregation exercise.

## My Model Choice

The full run uses 307,511 labeled applications. XGBoost has the highest observed test AUC, but its advantage over LightGBM is only 0.00035. The paired DeLong test gives `p=0.6445`, and the bootstrap intervals overlap.

| Model | Test AUC | KS | Role in the comparison |
| --- | ---: | ---: | --- |
| XGBoost | 0.7674 | 0.4087 | Challenger |
| LightGBM | 0.7671 | 0.4038 | Preferred next-round model |
| Random Forest | 0.7526 | 0.3801 | Nonlinear bagging benchmark |
| Logistic Regression | 0.7514 | 0.3848 | Transparent baseline |
| Decision Tree | 0.7231 | 0.3281 | Readable diagnostic model |

For the next development round, I would take **LightGBM** forward and keep XGBoost as the challenger. The measured discrimination is effectively tied in this run, while the current candidate screening takes about 18 seconds for LightGBM and 22 minutes for XGBoost. Logistic Regression remains the benchmark I would use to check direction, stability, and reason-code plausibility.

The table is a rounded snapshot of the current full run. Exact metrics and run-dependent timing values are in [model_metrics.csv](outputs/model_metrics.csv), [best_parameters.csv](outputs/best_parameters.csv), and [delong_auc_test.csv](outputs/delong_auc_test.csv).

## Decisions Made Before the Final Test

| Decision | Reason |
| --- | --- |
| Use AUC, KS, and PR-AUC as primary metrics | Accuracy is not informative enough with an event rate near 8%. |
| Keep a separate final test partition | I wanted one sample with no role in tuning, calibration, or threshold selection. |
| Limit the search budget | The goal is a fair model-family comparison on local hardware, not a Kaggle leaderboard search. |
| Calibrate only the two leading candidates | Class weighting helps ranking but changes the probability prior; calibrating every weak candidate adds little value. |
| Compare the leaders with paired inference | A difference in the third decimal place should not be treated as a meaningful win without an uncertainty check. |

More detail, including alternatives I rejected, is recorded in [modeling_decisions.md](docs/modeling_decisions.md).

## What Changed During Development

The first working version exposed several issues that affected the interpretation more than the code itself:

- one validation sample was doing too many jobs, so calibration and threshold selection were separated;
- LightGBM varied `subsample` without activating row bagging, so `subsample_freq=1` was added;
- boosting time originally reflected candidate fitting rather than a comparable final refit;
- class-weighted probabilities were initially discussed too much like PD estimates;
- report values were moved out of the report source and into run artifacts.

These changes and their consequences are summarized in [development_notes.md](docs/development_notes.md).

## Files

- [Source notebook](home_credit_model_comparison.ipynb): sequential modeling workflow.
- [Executed notebook](home_credit_model_comparison.executed.ipynb): full-data run with outputs.
- [Formal report](outputs/home_credit_model_comparison_report_formal.pdf): concise decision report.
- [Editable model control form](docs/home_credit_model_control_form_cn.docx): Chinese learning guide and machine-readable settings for the next run.
- [Validation utilities](credit_risk_validation.py): bootstrap, DeLong, calibration, PSI, segment, and cost functions.
- `outputs/`: tables, predictions, models, and figures from the current run.

## Reproduce the Run

Download `application_train.csv` from the [Kaggle Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place it at `data/application_train.csv`.

```bash
python -m pip install -r requirements.txt

# quick pipeline check
HOME_CREDIT_DEBUG=1 jupyter nbconvert \
  --to notebook --execute home_credit_model_comparison.ipynb \
  --output home_credit_model_comparison.debug.executed.ipynb \
  --ExecutePreprocessor.timeout=7200

# rebuild the PDF after a completed run
python generate_home_credit_report_formal.py
```

## Limits of This Dataset

The public table has no reliable booking timeline, rejected-applicant outcomes, or lender economics. I therefore use the random holdout to compare development candidates, not to estimate future-vintage stability, reject inference, expected loss, or capital impact. Those are different projects and require different data.

## Unified Benchmark V2

The repository is now being extended from the executed five-model baseline to a controlled linear-versus-nonlinear benchmark with:

- a common-input Logistic Regression reference;
- the existing WOE/scorecard-style Logistic Regression;
- Random Forest, CatBoost, LightGBM, XGBoost, and MLP;
- versioned data, feature, selection, and final-Test contracts;
- shared Python utilities and leakage-focused automated tests;
- a separate controlled experiment and model-specific best-practice comparison.

The existing notebook and reports remain the historical executed baseline. The V2 protocol is documented in:

- [Detailed modeling workflow](machinelearning_final/README.md)
- [Repository integration workflow](docs/repository_workflow.md)
- [Experiment contract](configs/experiment.yaml)
- [Feature policy](configs/feature_policy.yaml)
- [Model search spaces](configs/model_search_spaces.yaml)
- [Contribution guide](CONTRIBUTING.md)
- [V2 executable notebooks](notebooks/README.md)
- [Final-report findings notes](docs/report_notes.md)

Validate the shared layer without downloading the Kaggle data:

```bash
python -m pip install -e . pytest
make check-config
make test
```

V2 development is additive: existing notebooks have not been moved, and historical outputs retain their original interpretation. All required V2 configurations were locked before the one-time final-Test scoring pass; the resulting artifacts are under `artifacts/final_test/20260831T073511_991324Z/`. The fixed random Test partition had been inspected in earlier historical work, so the documented V2 results are a controlled comparison rather than a claim based on a never-before-seen holdout.

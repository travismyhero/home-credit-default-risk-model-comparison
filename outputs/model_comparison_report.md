# Home Credit Default Risk: Model Development Evidence

## Executive Summary

Five models were developed on mutually exclusive training, model-selection, calibration, threshold, and final-test partitions. On the final random holdout, XGBoost has the highest observed AUC (0.7674; bootstrap 95% CI 0.7589-0.7765), followed by LightGBM (0.7671; 0.7584-0.7765). The pre-specified DeLong comparison uses the two candidates selected by model-selection AUC: LightGBM versus XGBoost, with an AUC difference of -0.0003 (p=0.6445); this difference is not statistically distinguishable at the 5% level under the paired DeLong test. The evidence supports a short list of leading candidates, not a claim of universal algorithm superiority.

## Data and Experimental Design

- Public Home Credit application records: 307,511; observed event rate: 8.07%.
- Training/model-selection/calibration/threshold/test shares: 60%/10%/10%/10%/10%.
- Feature filtering and preprocessing are fitted on training data only.
- Parameter screening is deliberately limited to the candidate counts reported below.
- Boosting candidates use early stopping on model-selection data; selected iteration counts are then refitted on the complete training partition.
- Final training time is measured from this clean complete-training fit for every model.

## Final Test Results

| Model | AUC | KS | PR-AUC | F1 | Brier | LogLoss | TrainTime | TuneTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.7674 | 0.4087 | 0.2609 | 0.3196 | 0.1925 | 0.5663 | 4.6321 | 1341.3128 |
| LightGBM | 0.7671 | 0.4038 | 0.2593 | 0.3195 | 0.1945 | 0.5696 | 2.3404 | 17.9929 |
| Random Forest | 0.7526 | 0.3801 | 0.2364 | 0.2983 | 0.1748 | 0.5321 | 41.3208 | 541.0899 |
| Logistic Regression | 0.7514 | 0.3848 | 0.2286 | 0.2907 | 0.2048 | 0.5976 | 6.0337 | 39.8692 |
| Decision Tree | 0.7231 | 0.3281 | 0.2081 | 0.2737 | 0.2115 | 0.6083 | 2.3499 | 12.9963 |

## Statistical Uncertainty

| Model | Estimate | Lower | Upper |
| --- | --- | --- | --- |
| XGBoost | 0.7674 | 0.7589 | 0.7765 |
| LightGBM | 0.7671 | 0.7584 | 0.7765 |
| Random Forest | 0.7526 | 0.7434 | 0.7615 |
| Logistic Regression | 0.7514 | 0.7428 | 0.7604 |
| Decision Tree | 0.7231 | 0.7126 | 0.7330 |

Paired bootstrap differences and the DeLong test are saved separately. Bootstrap intervals quantify uncertainty conditional on this fitted sample and model. They do not measure performance variability across new booking periods or fully repeated model-development splits.

## Probability Calibration in Credit-Risk Context

Balanced class weights and `scale_pos_weight` alter the effective training prior. Raw boosted-model probabilities therefore overstate portfolio PD levels and are treated as ranking scores. The notebook compares analytical prior correction, Platt scaling, and Isotonic regression on an independent calibration sample, selects the mapping without test data, and reports calibration-in-the-large, calibration slope, observed/expected ratio, Brier score, and Log Loss.

The observed bad rate in this public sample is a point-in-time sample rate, not a through-the-cycle long-run default rate. No long-run portfolio anchor is available, so the project does not claim TTC calibration, grade-level long-run calibration, or regulatory PD estimation.

## Segment and Drift Diagnostics

Segment AUC and bad rates are reported for gender, contract type, age band, and income band subject to minimum event counts. PSI compares the random training and test partitions only. The maximum observed random-split PSI is 0.0018; this is a pipeline consistency diagnostic and is not evidence of temporal stability.

## Decision Costs

The threshold cost table uses explicit illustrative ratios between approving a bad applicant and rejecting a good applicant. It is a sensitivity analysis, not an expected-loss, profitability, capital, or pricing model. Real strategy selection requires lender-specific LGD, EAD, margin, funding cost, operating cost, capacity, and capital assumptions.

## Scope Boundaries

- No reliable application timestamp is supplied, so out-of-time validation is not performed.
- Labels are unavailable for Kaggle's official test file; the final test is a reserved random holdout.
- No accepted/rejected application outcome mechanism is supplied, so reject inference is not performed.
- Bureau, previous-application, and repayment-history aggregation is not implemented in this repository and is not represented as a completed module.
- Built-in coefficients and feature importance are descriptive. SHAP analysis is not claimed or silently substituted.
- A single development split cannot establish ranking stability across random seeds or future customer vintages.

## Model-Selection Position

XGBoost is the observed leader on this final holdout, while LightGBM remains close in observed AUC. A production champion should be chosen only after out-of-time testing, repeated development samples, segment review, calibration anchoring, reason-code review, and economic strategy analysis. Logistic Regression remains the transparent benchmark.

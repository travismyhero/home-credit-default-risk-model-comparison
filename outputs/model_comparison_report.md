# Comparing Five Machine Learning Models for Credit Risk Using Home Credit Data

## 1. Executive Summary

This study uses the Home Credit Default Risk application table to compare Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM under one evaluation framework. On the held-out test set, **XGBoost** provides the highest AUC (AUC=0.7716, KS=0.4072, PR-AUC=0.2587), while **Random Forest** has the lowest raw-probability Brier Score (0.1736).

## 2. Data and Prediction Task

- Source: Kaggle Home Credit Default Risk, using `application_train.csv`.
- Observations: 307,511; original columns: 122.
- `TARGET=1` indicates repayment difficulty; the bad-applicant rate is 8.07%.
- Data is split 60%/20%/20% into training, validation, and test sets.
- Training-only selection retains 108 raw/engineered features and produces 232 encoded features.
- Seven interpretable business ratios are added, with leakage-safe imputation and encoding fitted on training data only.

## 3. Modeling Methods

Logistic Regression provides a linear and interpretable baseline. Decision Tree learns explicit nonlinear split rules. Random Forest uses bagging to reduce variance. XGBoost applies regularized gradient boosting, while LightGBM combines histogram learning with leaf-wise growth. All models share the same split and training-fitted preprocessing.

## 4. Experimental Design

- Logistic Regression uses GridSearchCV; Decision Tree and Random Forest use RandomizedSearchCV.
- XGBoost and LightGBM use validation AUC and early stopping.
- Cross-validation uses 5 folds with a 60,000-row stratified tuning subset.
- Thresholds and calibration methods are selected on validation data only.
- The test set is used only for final model evaluation.

## 5. Results

| Model | AUC | AR | KS | PR-AUC | Precision | Recall | F1 | Brier | LogLoss | TrainTime | PredictTime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.7716 | 0.5431 | 0.4072 | 0.2587 | 0.2584 | 0.4147 | 0.3184 | 0.1848 | 0.5478 | 1138.0417 | 0.3262 |
| LightGBM | 0.7702 | 0.5404 | 0.4064 | 0.2555 | 0.2570 | 0.4127 | 0.3168 | 0.1849 | 0.5467 | 4.5298 | 0.1779 |
| Logistic Regression | 0.7553 | 0.5107 | 0.3790 | 0.2314 | 0.2296 | 0.4272 | 0.2987 | 0.2023 | 0.5914 | 23.5474 | 0.0123 |
| Random Forest | 0.7538 | 0.5075 | 0.3802 | 0.2290 | 0.2395 | 0.3960 | 0.2984 | 0.1736 | 0.5295 | 40.3037 | 0.1984 |
| Decision Tree | 0.7231 | 0.4462 | 0.3247 | 0.2010 | 0.2037 | 0.4205 | 0.2745 | 0.2100 | 0.6058 | 2.3345 | 0.0176 |

The Decision Tree train/validation AUC gap is 0.0206; the Random Forest gap is 0.1039. Complete threshold, feature-importance, calibration, and risk-decile results are available in the accompanying CSV and PNG files.

## 6. Model Strengths and Limitations

### Logistic Regression
Interpretable, fast, easy to deploy, and appropriate as a regulated credit-scoring baseline. Its linear structure is less effective for complex nonlinear relationships and interactions.

### Decision Tree
Provides transparent nonlinear rules without scaling, but has high variance and can overfit quickly as depth increases.

### Random Forest
More stable than a single tree and effective for nonlinear interactions. It is larger, less interpretable, and its probabilities may require calibration.

### XGBoost
Provides strong tabular discrimination and mature regularization. Its principal costs are tuning time, parameter complexity, and more demanding deployment and explanation.

### LightGBM
Offers an excellent speed-memory-performance balance on large sparse data. Leaf-wise growth requires careful depth, leaf-size, and regularization controls.

## 7. Model Selection Recommendations

1. **Maximum predictive performance:** choose **XGBoost** with test AUC=0.7716 and KS=0.4072.
2. **Interpretability and regulatory acceptance:** choose **Logistic Regression** and retain coefficient direction and business-variable explanations.
3. **Speed-performance balance:** choose **LightGBM** among models within 0.01 AUC of the best result; its recorded training time is 4.53 seconds.

The fastest fitted model is **Decision Tree**. A production decision should also consider calibration, stability, reject strategy, interpretability, operational cost, and deployment constraints.

## 8. Limitations

- This is a historical Kaggle dataset, and `TARGET` is not a public regulatory default definition.
- There is no clear natural time index for strict out-of-time validation.
- Kaggle does not publish labels for the official test set, so this study reserves an internal test split.
- The features reflect Home Credit's business context and require revalidation across lenders, regions, and periods.
- Uncalibrated scores should not be interpreted directly as real probability of default.
- Real LGD, EAD, funding cost, revenue, and profitability are not included.
- `bureau`, `previous_application`, and `installments_payments` are not aggregated in this first version.

## 9. Conclusion

**XGBoost** provides the strongest discrimination, **Random Forest** has the lowest raw-probability error, and Logistic Regression remains the easiest to explain. LightGBM is the strongest speed-performance compromise. The value of the complex-model gain should be judged against calibration, stability, interpretability, and computational cost.

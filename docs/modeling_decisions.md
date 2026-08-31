# Modeling Decisions

This note records the choices that materially changed the experiment. It is intentionally separate from the code so that a reviewer can challenge the reasoning without reading every cell.

## 1. Application Table Only

I considered aggregating `bureau.csv`, `previous_application.csv`, and repayment tables. I left them out because the first question was whether model family mattered when the information set was held constant. Adding historical tables would make feature construction the dominant source of performance and blur that comparison.

## 2. Five Data Roles

The split is 60% training and 10% each for model selection, calibration, threshold choice, and final testing. This costs training data, but it prevents one convenient validation set from influencing every decision. With more development time I would compare this design with nested cross-validation; for this project, the explicit roles are easier to audit and explain.

## 3. Limited Search Budget

I capped the full search at eight candidates for each tree ensemble and used a 60,000-row stratified training subset for screening. This is a hardware and project-scope decision, not a claim that the hyperparameters are globally optimal. XGBoost screening already takes roughly 22 minutes on the current machine, while LightGBM finishes the same candidate count in under 20 seconds.

## 4. Metric Priority

The event rate is close to 8%, so accuracy is secondary. I rank models with AUC, KS, and PR-AUC, then inspect threshold behavior separately. F1 is reported at a threshold chosen on its own partition; it is not used to select the model family.

## 5. Class Weighting and Calibration

Balanced class weights improve minority-class learning, but they also alter the effective event prior. I therefore treat raw model probabilities as scores. Only the two leaders are taken through prior correction, Platt scaling, and Isotonic regression. Calibration method selection uses the calibration partition and never the final test.

## 6. Final Recommendation

XGBoost is the point-estimate leader, but its AUC advantage over LightGBM is 0.00035 and the paired DeLong p-value is 0.6445. I do not regard that as a useful performance win. LightGBM is my preferred next-round model because it reaches essentially the same discrimination with far lower search and fit time. XGBoost remains the challenger, and Logistic Regression remains the interpretation benchmark.

## 7. Supporting Checks

Bootstrap intervals, segment tables, random-split PSI, and cost ratios are supporting checks. They answer narrow questions about uncertainty, subgroup dispersion, pipeline consistency, and threshold sensitivity. They are not separate claims that the model is ready for deployment.

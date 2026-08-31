# Development Notes

The project did not start with the current validation structure. These are the changes that mattered most.

## Validation Reuse

An earlier version used one validation set for early stopping, model comparison, calibration, and threshold selection. The code ran, but the interpretation was too optimistic because repeated decisions were made on the same observations. I split those roles before rerunning the full experiment.

## LightGBM Sampling

The search varied `subsample`, but LightGBM does not activate row bagging unless `subsample_freq` is positive. I added `subsample_freq=1` and included a check against the saved final parameters.

## Boosting Iterations and Timing

The earlier XGBoost search stopped close to its iteration cap, and boosting time was not comparable with the scikit-learn models. The revised search allows more rounds, records boundary contact, and performs a fresh fit on the complete training partition. `TuneTime` and `TrainTime` now describe different operations.

## Probability Interpretation

The first report emphasized improved Brier scores after calibration without making the role of class weighting explicit enough. The revised version shows prior correction alongside Platt and Isotonic mappings and reports calibration-in-the-large, slope, and observed-to-expected ratio.

## Report Values

Model results were once written directly into the report source. The report now reads the completed run tables. Narrative choices, such as preferring LightGBM for the next round, remain explicit author decisions rather than being inferred from whichever AUC is marginally largest.

## Warnings

Warnings are not globally suppressed. During the update I removed an obsolete Logistic Regression argument, addressed convergence settings, consolidated feature-column insertion, and used the LightGBM Booster prediction interface consistently. The executed notebook currently contains no warning output.

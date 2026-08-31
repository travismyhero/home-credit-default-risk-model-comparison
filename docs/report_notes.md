# Report Notes

This file records conclusions that should be incorporated into the final model-comparison report.

## Linear baselines and the WOE trade-off

The controlled common-input Logistic Regression is the primary linear baseline for measuring the
incremental value of nonlinear models. It retains all 126 approved common features and applies only
train-fitted operational preprocessing: numeric median imputation with missing indicators, scaling,
categorical missing-value handling, One-Hot encoding, and regularized Logistic Regression.

The WOE Logistic Regression is a separate interpretable scorecard benchmark. Its full workflow uses
train-fitted monotonic binning, explicit missing bins, rare-category grouping, WOE transformation,
IV screening, correlation screening, VIF, p-values, and coefficient-direction constraints. This
workflow reduces 126 input variables to 18 final scorecard variables.

Current development results:

| Model | Train AR | Selection AR | Absolute Train-Selection AR gap |
|---|---:|---:|---:|
| Common-input Logistic | 0.49953 | 0.49629 | 0.00324 |
| WOE Logistic | 0.48433 | 0.48384 | 0.00049 |

The WOE model's lower Selection AR is not evidence of a failed workflow. The scorecard pipeline is a
deliberately lossy compression of the input information:

- continuous values become a limited number of risk bands, and monotonic merging removes local
  variation;
- a multi-level categorical variable becomes one WOE column instead of multiple independently
  estimated One-Hot coefficients;
- IV, correlation, VIF, p-value, and sign rules remove weak or conditionally useful predictors even
  though those predictors may still add a small amount of ranking power collectively;
- these statistical and governance rules optimize interpretability, coefficient stability, and
  operational simplicity rather than AR directly.

The comparison therefore demonstrates a genuine trade-off. Relative to the common-input Logistic,
the WOE scorecard gives up about 0.01245 Selection AR while reducing the Train-Selection AR gap from
about 0.00324 to 0.00049 and producing a compact, directionally interpretable 18-variable model.
Because the training sample contains roughly 215,000 observations, the regularized common-input
Logistic can estimate a larger encoded feature space relatively reliably; consequently, WOE's
variance-reduction benefit does not necessarily offset its information loss in this dataset.

The final report should present the models as serving different purposes:

- **Common-input Logistic:** primary controlled linear baseline for nonlinear uplift.
- **WOE Logistic:** interpretable scorecard benchmark illustrating the performance-stability-
  interpretability trade-off.

All figures above are development results. The final Test sample remains sealed until every required
model family has been selected and locked.

## Random Forest: nonlinear capacity without development uplift

Random Forest is the first controlled nonlinear challenger. It uses the same 126 approved raw
features as the common-input Logistic. Tree-specific preprocessing is fitted on Train only: numeric
median imputation with missing indicators and categorical missing-value handling plus One-Hot
encoding. The resulting forest input contains 321 encoded features.

The search was deliberately separated into three roles:

1. six versioned structure profiles were screened on a stratified 60,000-row Train subset using
   out-of-bag AUC/AR, without consulting Selection;
2. the two strongest OOB profiles were fitted on the complete Train sample and evaluated on
   Selection;
3. the selected structure was checked at 300 and 500 trees. The 300-tree model was retained because
   500 trees added only about 0.00018 Selection AR while slightly increasing the Train-Selection gap
   and computation time.

The locked structure is `feature_rich_regularized`: 300 trees, depth 12, minimum 100 observations
per leaf, minimum 200 observations per split, 50% feature sampling, 70% row sampling, bootstrap, and
balanced-subsample class weights.

Development results:

| Model | Train AR | Selection AR | Absolute Train-Selection AR gap |
|---|---:|---:|---:|
| Common-input Logistic | 0.49953 | 0.49629 | 0.00324 |
| Random Forest | 0.61485 | 0.49084 | 0.12401 |

Random Forest therefore does not improve the controlled linear baseline on Selection. Its paired AR
difference versus common-input Logistic is -0.00545, with a 95% paired-bootstrap interval of
[-0.01200, 0.00086]. The interval crosses zero: the observed shortfall is not conclusive evidence of
a population-level disadvantage, but there is also no evidence of nonlinear uplift. The much larger
Train-Selection gap is direct evidence that the forest uses additional capacity mainly to fit Train
rather than to improve unseen-sample ranking.

The top Selection decile captures 32.14% of bad applicants with lift 3.21. The corresponding
common-input Logistic values are 32.73% and 3.27, so the forest does not improve top-decile targeting
either. The forest importance ranking is dominated by `EXT_SOURCE_3`, `EXT_SOURCE_2`,
`EXT_SOURCE_1`, `GOODS_CREDIT_RATIO`, `EMPLOYED_YEARS`, and `AGE_YEARS`. This shows nonlinear tree
capacity concentrating heavily on the same strong external-score and affordability signals rather
than uncovering a clearly superior new ranking structure.

Raw Random Forest Logloss is much worse than Logistic because balanced-subsample class weights shift
the probability level; those raw probabilities are useful for ranking but are not calibrated PDs.
Probability-quality comparisons should therefore be deferred to the later calibration stage and
must not be interpreted as a fair calibrated-probability comparison at this point.

The report conclusion should be that model nonlinearity alone is not sufficient. A bagged tree model
can increase Train discrimination and computation cost substantially without creating measurable
Selection uplift. Boosting models remain necessary challengers because their sequential residual
fitting and regularization differ materially from independent bagged trees.

## CatBoost: material nonlinear uplift with controlled native-category boosting

CatBoost is the second nonlinear challenger and uses exactly the same 126 approved raw features as
the common-input Logistic and Random Forest. It keeps the 111 numeric variables in their native
numeric form (including missing values) and supplies the 15 physical categorical variables to
CatBoost as Train-fitted string categories with an explicit missing label. It deliberately does not
One-Hot encode categories. Every candidate is trained on Train only; Selection is used only for
early stopping, the predeclared selection rule, paired comparison, and a 3,000-row stratified SHAP
summary after the model is locked. Test remains sealed.

Six fixed candidate profiles cover shallow and deeper trees, learning rates, L2 regularization,
row/column sampling, and randomization. The base cap is 3,000 iterations with 150-round patience.
Any boundary-reaching candidate in the near-best set is checked at a 5,000-iteration cap. The
extension priority is itself governed: first extend the model that the common Selection rule would
select at the base cap, then extend the other highest-AR boundary candidates. This matters because
the final choice favours stability among near-equal AR models rather than simply choosing the
largest model.

The highest Selection AR observed was 0.53577 for the depth-6 `legacy_baseline` extension, but its
Train-Selection AR gap was 0.12946. The locked `shallow_faster` profile is depth 4, learning rate
0.03, L2 5, 80% row sampling, 90% feature sampling, and random strength 0.5. It early-stopped at
3,770 trees (the fit attempted 3,920, so it did not hit the 5,000-tree cap), with Selection AR
0.53476. Its AR is only 0.00101 below the maximum—well inside the predeclared 0.005 tolerance—while
its gap is 0.06336, materially below the other near-best boosted candidates. It is therefore the
correct locked model under the stated rule, rather than an after-the-fact preference for a smaller
model.

Development results:

| Model | Train AR | Selection AR | Absolute Train-Selection AR gap |
|---|---:|---:|---:|
| Common-input Logistic | 0.49953 | 0.49629 | 0.00324 |
| Random Forest | 0.61485 | 0.49084 | 0.12401 |
| CatBoost | 0.59812 | 0.53476 | 0.06336 |

CatBoost improves Selection AR by 0.03848 versus the controlled common-input Logistic. Its 95%
paired-bootstrap interval is [0.03168, 0.04460], wholly above zero. It also improves Selection KS
by 0.03269, AP by 0.01938, and Logloss by 0.00526; the Logloss difference is negative because lower
is better, and its paired interval [-0.00622, -0.00428] also excludes zero. Unlike the class-weighted
Random Forest, CatBoost's raw probability comparison is not confounded by balanced class weights,
but it should still not be called a calibrated PD model before a separate calibration stage.

At a 10% review rate, CatBoost captures 34.61% of bad applicants on Selection with lift 3.46. This
is better than the Random Forest's 32.14% capture and lift 3.21, and makes the discrimination uplift
operationally visible rather than only an AUC/AR result. CatBoost has a larger generalization gap
than Logistic, so the result is not a claim that extra complexity is free; it is evidence that this
particular regularized boosting configuration converts some of that added capacity into meaningful
Selection ranking improvement, unlike the bagged forest.

Both built-in split importance and Selection-sample mean absolute SHAP rank `EXT_SOURCE_3`,
`EXT_SOURCE_2`, and `EXT_SOURCE_1` as the dominant signals. The next tier includes
`CREDIT_ANNUITY_RATIO`, `EMPLOYED_YEARS`, `AMT_ANNUITY`, `GOODS_CREDIT_RATIO`, `AGE_YEARS`, and
`NAME_EDUCATION_TYPE`. The nonlinear uplift therefore appears to come from richer interactions and
threshold effects around the same external-score, affordability, employment, and demographic
signals—not from a wholly different data source. The final report should use this interpretation
carefully: global SHAP magnitude shows influence, not causal effect or a permitted reason code.

During lock review, the first CatBoost run revealed that a stability-selected boundary candidate had
not itself received an extension because extensions had initially been ordered only by AR. The search
rule was corrected before final lock, the complete run was repeated from the same data fingerprint,
split, feature set, candidates, and random seed, and all six base-candidate metrics reproduced.
The corrected locked candidate then early-stopped before the extended cap. This audit trail should be
retained as evidence that the final CatBoost result did not select an incompletely trained candidate.

## LightGBM: efficient leaf-wise boosting with significant but slightly smaller uplift

LightGBM is the third nonlinear challenger and again uses the same 126 approved raw features. The
111 numeric features retain native missing values. The 15 physical categorical features are converted
to pandas categorical columns using a category vocabulary fixed from Train; missing values receive an
explicit missing category and Selection-only levels receive `__UNSEEN__`. This preserves the native
categorical-input approach while preventing Selection from defining the training vocabulary. No
One-Hot encoding, class weighting, feature selection, or Test access is used.

The seven predeclared profiles span leaf-wise capacity (`num_leaves`, `max_depth`, and
`min_child_samples`), learning rate, L1/L2 regularization, row and column sampling, minimum split
gain, and categorical smoothing. Row sampling is active because every model uses `subsample_freq=1`.
All seven candidates early-stopped before the 3,000-tree cap, so no boundary extension was needed.

The highest observed Selection AR was 0.53336 for `shallow_stable` (31 leaves, depth 6), but it had
a Train-Selection AR gap of 0.15075. The locked `conservative_smooth` profile uses 15 leaves, depth
5, a 300-observation minimum leaf, learning rate 0.02, L2 10, 70% row sampling, 80% column sampling,
and category smoothing 20. It stopped at 1,237 trees with Selection AR 0.53187. This is 0.00149 below
the maximum—inside the predeclared 0.005 AR tolerance—while reducing the gap to 0.08067. The choice
therefore follows the same near-best-then-stability rule used for every model family.

Development results:

| Model | Train AR | Selection AR | Absolute Train-Selection AR gap |
|---|---:|---:|---:|
| Common-input Logistic | 0.49953 | 0.49629 | 0.00324 |
| Random Forest | 0.61485 | 0.49084 | 0.12401 |
| LightGBM | 0.61253 | 0.53187 | 0.08067 |
| CatBoost | 0.59812 | 0.53476 | 0.06336 |

LightGBM improves Selection AR by 0.03558 over the controlled Logistic. The 95% paired-bootstrap
interval is [0.02893, 0.04310], wholly above zero. It also improves KS by 0.02376, AP by 0.02203,
and Logloss by 0.00510; the Logloss difference is negative because lower is better, with interval
[-0.00607, -0.00419]. Its top Selection decile captures 34.59% of bad applicants with lift 3.46,
which is practically the same targeting level as CatBoost (34.61% and 3.46) and higher than Random
Forest (32.14% and 3.21).

The CatBoost point estimate remains 0.00290 AR higher than the locked LightGBM, while LightGBM trains
much faster in this controlled search (approximately 10–22 seconds per base candidate rather than
minutes). This is an observed development comparison, not yet a direct paired-inference verdict
between the two boosting models. Both should remain locked challengers until the remaining model
families are chosen and the final Test protocol is opened once.

LightGBM gain importance is led by `EXT_SOURCE_3`, `EXT_SOURCE_2`, `ORGANIZATION_TYPE`,
`EXT_SOURCE_1`, and `CREDIT_ANNUITY_RATIO`. The 3,000-row Selection Tree SHAP summary instead ranks
the three external-source variables first, followed by `CREDIT_ANNUITY_RATIO`,
`GOODS_CREDIT_RATIO`, `EMPLOYED_YEARS`, `ORGANIZATION_TYPE`, `AMT_ANNUITY`, and `OWN_CAR_AGE`.
The distinction is useful: a high-cardinality categorical variable such as `ORGANIZATION_TYPE` can
accumulate many split gains, while mean absolute SHAP better summarizes its typical contribution to
predictions. Neither measure establishes causality or substitutes for approved customer-level reason
codes.

## XGBoost: stable depthwise boosting with the strongest top-decile point estimate

XGBoost is the fourth nonlinear challenger. It uses the same 126 raw inputs and fixed split, but its
model-appropriate category path is explicit Train-fitted dense One-Hot encoding rather than native
categories. The resulting matrix has 254 columns: 111 numeric variables retaining `NaN`, plus 143
category dummies. The encoder is fitted on Train only; an unseen Selection category is represented
by an all-zero dummy group. This is intentional and preserves the distinction between numeric zero
and missing values for the `hist` tree method. Trees use CPU `hist` construction and depthwise growth,
with depth, minimum child Hessian, learning rate, L1/L2, row/column sampling, and gamma controlled
by seven predeclared profiles.

All seven candidates early-stopped before 3,000 rounds, so no boundary extension was necessary. The
highest Selection AR was 0.53446 for `shallow_stable` (depth 3, minimum child weight 10), but the
locked `conservative_depth3` model has Selection AR 0.53296 and a smaller Train-Selection AR gap:
0.06079 versus 0.06984. The AR difference is 0.00151, within the predeclared 0.005 tolerance. The
locked model therefore uses depth 3, minimum child weight 20, learning rate 0.02, L2 10, 70% row
sampling, 80% column sampling, and 2,796 trees; it is the stability-selected choice rather than a
post-hoc attempt to maximize the development point estimate.

Development results:

| Model | Train AR | Selection AR | Absolute Train-Selection AR gap |
|---|---:|---:|---:|
| Common-input Logistic | 0.49953 | 0.49629 | 0.00324 |
| Random Forest | 0.61485 | 0.49084 | 0.12401 |
| LightGBM | 0.61253 | 0.53187 | 0.08067 |
| XGBoost | 0.59374 | 0.53296 | 0.06079 |
| CatBoost | 0.59812 | 0.53476 | 0.06336 |

XGBoost improves Selection AR by 0.03667 over the controlled Logistic; its 95% paired-bootstrap
interval is [0.03013, 0.04353], entirely above zero. It also improves KS by 0.02767, AP by 0.02212,
and Logloss by 0.00528, with a Logloss interval of [-0.00623, -0.00437]. At a 10% review rate it
captures 34.77% of bad applicants with lift 3.48—the strongest top-decile point estimate among the
currently locked boosting models. The AR point estimate is 0.00181 below CatBoost and 0.00109 above
LightGBM; direct paired inference between the three boosters should be added only after all remaining
models are locked, rather than using these small point differences to declare a winner now.

The XGBoost candidate fits take roughly 34–75 seconds, slower than LightGBM's 10–22 seconds but much
shorter than CatBoost's multi-minute candidate fits on the same hardware. Its smaller selected gap
than LightGBM and CatBoost is encouraging but remains a single-Selection result, not a substitute for
the one-time sealed Test evaluation.

Raw dummy-level gain aggregated to original variables gives high importance to `ORGANIZATION_TYPE`,
`OCCUPATION_TYPE`, and `NAME_EDUCATION_TYPE`; a multi-level category can accumulate many dummy gains.
The 3,000-row Selection Tree SHAP summary, after summing all dummies for each original feature before
taking absolute values, ranks `EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1`,
`CREDIT_ANNUITY_RATIO`, `AMT_ANNUITY`, `GOODS_CREDIT_RATIO`, `AMT_GOODS_PRICE`, and `AGE_YEARS` first.
That reconciliation again supports the interpretation that external scores and affordability signals
drive the typical prediction, while categorical variables help refine segments. Importance is not
causality, policy approval, or an individual adverse-action explanation.

## MLP: a stable neural-network control that does not improve the tabular baseline

The MLP is the final nonlinear challenger. It uses Train-fitted numeric median imputation with
missing indicators and sparse standardization, plus Train-fitted categorical missing handling and
One-Hot encoding. The input has 321 encoded columns. Unlike the historical demonstration notebook,
the production comparison does not use sklearn's default internal validation split or accuracy-based
early stopping. Each epoch updates only Train via `partial_fit`; Selection AUC determines early
stopping, and the best epoch's weights are restored. This gives the neural model the same permitted
Selection role as the boosting models.

Six network profiles covered one and two hidden layers (64, 128, and 64–32 / 128–64 units), L2
regularization, and learning rates. Every candidate stopped before the 200-epoch cap, so no extension
was required. The highest Selection AR was 0.49764 for `compact_strong_l2`, but the locked
`wider_strong_l2` network (128 and 64 units, alpha 0.01, learning rate 0.0008, batch size 512) has
Selection AR 0.49584 and the smallest gap, 0.00402. The AR shortfall of 0.00179 is within the
predeclared 0.005 tolerance, so it is the appropriate stability-selected MLP at 19 epochs.

Development results:

| Model | Train AR | Selection AR | Absolute Train-Selection AR gap |
|---|---:|---:|---:|
| Common-input Logistic | 0.49953 | 0.49629 | 0.00324 |
| WOE Logistic | 0.48433 | 0.48384 | 0.00049 |
| Random Forest | 0.61485 | 0.49084 | 0.12401 |
| MLP | 0.49986 | 0.49584 | 0.00402 |
| LightGBM | 0.61253 | 0.53187 | 0.08067 |
| XGBoost | 0.59374 | 0.53296 | 0.06079 |
| CatBoost | 0.59812 | 0.53476 | 0.06336 |

The MLP's paired Selection AR difference versus the common-input Logistic is -0.00044, with 95%
interval [-0.00278, 0.00168]. It crosses zero, so there is no evidence of either meaningful uplift or
meaningful deterioration in population ranking. Its Logloss is worse by 0.00138 and that interval
[0.00104, 0.00171] is entirely above zero, providing evidence that its raw probability quality is
weaker in this configuration. The MLP captures 32.44% of bad applicants in the top 10% with lift
3.24, well below the 34.59–34.77% captured by the locked boosting models.

Selection permutation importance at the original-variable level identifies `EXT_SOURCE_3`,
`EXT_SOURCE_2`, and `EXT_SOURCE_1` as by far the strongest inputs, followed at much smaller scale by
education, phone-change timing, employment years, and affordability features. This agrees with the
tree-model explanations about the dominant information sources. Permutation importance measures the
drop in AUC when one raw feature is shuffled in the 3,000-row Selection explanation sample; correlated
features can share or mask importance, so it is a sensitivity diagnostic rather than a causal claim.

The neural result is substantively useful: a fully standardized, regularized, two-layer network does
not automatically discover better tabular risk ranking than regularized Logistic Regression, whereas
the three boosting families do. It should be reported as evidence against assuming that a more complex
architecture alone creates value.

## Final Test scoring: boosting delivers the nonlinear uplift

After all seven configurations were locked in `artifacts/locked_models.json`, the final scoring
workflow was run once on the fixed 46,127-row Test partition. No model was re-trained, re-selected,
or modified after Test access. The scoring output—including row-level predictions, metrics, paired
bootstrap comparisons, deciles, and a fingerprint manifest—is retained in
`artifacts/final_test/20260831T073511_991324Z/`.

| Model | Test AR | Test AUC | Test KS | Test AP | Test Logloss |
|---|---:|---:|---:|---:|---:|
| Common-input Logistic | 0.49755 | 0.74878 | 0.37434 | 0.22802 | 0.24900 |
| WOE Logistic | 0.48583 | 0.74291 | 0.36652 | 0.21949 | 0.25066 |
| Random Forest | 0.49653 | 0.74826 | 0.37531 | 0.22979 | 0.52919 |
| CatBoost | 0.54116 | 0.77058 | 0.40966 | 0.25611 | 0.24264 |
| LightGBM | 0.53479 | 0.76740 | 0.40884 | 0.25069 | 0.24369 |
| XGBoost | 0.54020 | 0.77010 | 0.40983 | 0.25490 | 0.24280 |
| MLP | 0.49633 | 0.74816 | 0.37003 | 0.22538 | 0.25051 |

Relative to the controlled common-input Logistic, CatBoost adds 0.04361 Test AR (95% paired-
bootstrap interval [0.03675, 0.05023]); XGBoost adds 0.04265 [0.03596, 0.04910]; and LightGBM adds
0.03724 [0.03067, 0.04417]. Each interval is fully above zero. CatBoost and XGBoost are practically
very close on this fixed Test sample; no direct paired comparison between those two models was used
to name a winner. CatBoost has the largest AR point estimate and captures 34.40% of bad applicants
in the highest-risk decile (lift 3.44); XGBoost has the highest top-decile capture, 34.51% (lift
3.45). These are both material improvements on Logistic's 32.71% capture (lift 3.27).

The WOE scorecard remains the interpretable benchmark, not the performance leader: its Test AR is
0.01173 below the common-input Logistic, with a 95% interval [-0.01729, -0.00711]. Random Forest
and MLP have no significant Test AR improvement over Logistic, matching their Selection findings.
Random Forest's raw Logloss is again not a calibration comparison because its balanced-subsample
class weights intentionally change the probability prior.

One important qualification belongs in every final presentation: this same random Test partition was
inspected in earlier historical work. It was not used to choose any V2 configuration, and this V2
scoring pass did not make post-Test changes, but it is not a never-before-seen holdout. The results
support the controlled linear-versus-nonlinear comparison; a stronger claim about future performance
requires a newly sealed holdout or an out-of-time validation sample.

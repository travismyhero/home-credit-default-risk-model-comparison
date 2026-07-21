"""Statistical and credit-risk validation utilities for the Home Credit project."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic used in credit scoring."""
    fpr, tpr, _ = roc_curve(y_true, probability)
    return float(np.max(tpr - fpr))


def bootstrap_metric_intervals(
    y_true,
    probabilities: dict[str, np.ndarray],
    n_bootstrap: int = 500,
    random_state: int = 42,
    confidence_level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate stratified bootstrap intervals and paired AUC differences."""
    y = np.asarray(y_true, dtype=int)
    predictions = {name: np.asarray(values, dtype=float) for name, values in probabilities.items()}
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Both outcome classes are required for bootstrap inference")

    rng = np.random.default_rng(random_state)
    metric_names = ("AUC", "KS", "PR-AUC")
    draws = {
        model: {metric: np.empty(n_bootstrap, dtype=float) for metric in metric_names}
        for model in predictions
    }

    for draw in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ])
        sampled_y = y[sampled]
        for model, values in predictions.items():
            sampled_probability = values[sampled]
            draws[model]["AUC"][draw] = roc_auc_score(sampled_y, sampled_probability)
            draws[model]["KS"][draw] = ks_statistic(sampled_y, sampled_probability)
            draws[model]["PR-AUC"][draw] = average_precision_score(sampled_y, sampled_probability)

    alpha = (1 - confidence_level) / 2
    interval_rows = []
    for model, values in predictions.items():
        point_estimates = {
            "AUC": roc_auc_score(y, values),
            "KS": ks_statistic(y, values),
            "PR-AUC": average_precision_score(y, values),
        }
        for metric in metric_names:
            samples = draws[model][metric]
            interval_rows.append({
                "Model": model,
                "Metric": metric,
                "Estimate": float(point_estimates[metric]),
                "Lower": float(np.quantile(samples, alpha)),
                "Upper": float(np.quantile(samples, 1 - alpha)),
                "BootstrapSE": float(np.std(samples, ddof=1)),
                "BootstrapReplicates": int(n_bootstrap),
                "ConfidenceLevel": float(confidence_level),
            })

    difference_rows = []
    for model_a, model_b in combinations(predictions, 2):
        difference = draws[model_a]["AUC"] - draws[model_b]["AUC"]
        point_difference = roc_auc_score(y, predictions[model_a]) - roc_auc_score(y, predictions[model_b])
        difference_rows.append({
            "ModelA": model_a,
            "ModelB": model_b,
            "AUCDifference": float(point_difference),
            "Lower": float(np.quantile(difference, alpha)),
            "Upper": float(np.quantile(difference, 1 - alpha)),
            "BootstrapSE": float(np.std(difference, ddof=1)),
            "ProbabilityModelABetter": float(np.mean(difference > 0)),
            "BootstrapReplicates": int(n_bootstrap),
        })

    return pd.DataFrame(interval_rows), pd.DataFrame(difference_rows)


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    count = len(values)
    midranks = np.empty(count, dtype=float)
    start = 0
    while start < count:
        end = start
        while end < count and sorted_values[end] == sorted_values[start]:
            end += 1
        midranks[start:end] = 0.5 * (start + end - 1)
        start = end
    result = np.empty(count, dtype=float)
    result[order] = midranks + 1
    return result


def _fast_delong(predictions_sorted: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    model_count, total_count = predictions_sorted.shape
    negative_count = total_count - positive_count
    positive_predictions = predictions_sorted[:, :positive_count]
    negative_predictions = predictions_sorted[:, positive_count:]
    positive_ranks = np.empty((model_count, positive_count), dtype=float)
    negative_ranks = np.empty((model_count, negative_count), dtype=float)
    total_ranks = np.empty((model_count, total_count), dtype=float)

    for model_index in range(model_count):
        positive_ranks[model_index] = _compute_midrank(positive_predictions[model_index])
        negative_ranks[model_index] = _compute_midrank(negative_predictions[model_index])
        total_ranks[model_index] = _compute_midrank(predictions_sorted[model_index])

    aucs = total_ranks[:, :positive_count].sum(axis=1) / (positive_count * negative_count)
    aucs -= (positive_count + 1) / (2 * negative_count)
    positive_components = (total_ranks[:, :positive_count] - positive_ranks) / negative_count
    negative_components = 1 - (total_ranks[:, positive_count:] - negative_ranks) / positive_count
    covariance = np.cov(positive_components) / positive_count + np.cov(negative_components) / negative_count
    return aucs, np.atleast_2d(covariance)


def delong_auc_test(y_true, probability_a, probability_b, model_a: str, model_b: str) -> pd.DataFrame:
    """Run a paired two-sided DeLong test for two correlated ROC-AUC values."""
    y = np.asarray(y_true, dtype=int)
    first = np.asarray(probability_a, dtype=float)
    second = np.asarray(probability_b, dtype=float)
    order = np.argsort(-y)
    positive_count = int(y.sum())
    aucs, covariance = _fast_delong(np.vstack([first, second])[:, order], positive_count)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    difference = float(aucs[0] - aucs[1])
    if standard_error == 0:
        z_score = np.inf if difference else 0.0
        p_value = 0.0 if difference else 1.0
    else:
        z_score = difference / standard_error
        p_value = 2 * norm.sf(abs(z_score))
    return pd.DataFrame([{
        "ModelA": model_a,
        "ModelB": model_b,
        "AUCA": float(aucs[0]),
        "AUCB": float(aucs[1]),
        "AUCDifference": difference,
        "StandardError": standard_error,
        "Lower95": difference - 1.96 * standard_error,
        "Upper95": difference + 1.96 * standard_error,
        "Z": float(z_score),
        "PValueTwoSided": float(p_value),
    }])


def population_stability_index(reference, comparison, bins: int = 10) -> float:
    """Calculate PSI using quantile breakpoints learned from the reference sample."""
    reference_values = pd.Series(reference).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    comparison_values = pd.Series(comparison).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(reference_values) == 0 or len(comparison_values) == 0:
        return np.nan
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    comparison_counts, _ = np.histogram(comparison_values, bins=edges)
    epsilon = 1e-6
    reference_share = np.clip(reference_counts / reference_counts.sum(), epsilon, None)
    comparison_share = np.clip(comparison_counts / comparison_counts.sum(), epsilon, None)
    return float(np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share)))


def build_psi_table(
    reference_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    columns: list[str],
    dataset_label: str = "random_test_vs_train",
) -> pd.DataFrame:
    """Calculate feature PSI for numeric columns shared by two samples."""
    rows = []
    for column in columns:
        if column not in reference_frame or column not in comparison_frame:
            continue
        psi = population_stability_index(reference_frame[column], comparison_frame[column])
        rows.append({
            "Feature": column,
            "PSI": psi,
            "Comparison": dataset_label,
            "Interpretation": (
                "not_interpretable" if not np.isfinite(psi)
                else "large_shift" if psi >= 0.25
                else "moderate_shift" if psi >= 0.10
                else "small_shift"
            ),
        })
    return pd.DataFrame(rows).sort_values("PSI", ascending=False, na_position="last").reset_index(drop=True)


def segment_performance_table(
    y_true,
    probabilities: dict[str, np.ndarray],
    segments: pd.DataFrame,
    minimum_events: int = 25,
    minimum_non_events: int = 25,
) -> pd.DataFrame:
    """Calculate model discrimination and event rate for pre-defined customer segments."""
    y = np.asarray(y_true, dtype=int)
    if len(segments) != len(y):
        raise ValueError("Segment rows must align with outcomes")
    rows = []
    for segment_name in segments.columns:
        segment_values = segments[segment_name].astype("string").fillna("Missing")
        for segment_value in sorted(segment_values.unique()):
            mask = np.asarray(segment_values == segment_value)
            segment_y = y[mask]
            events = int(segment_y.sum())
            non_events = int(len(segment_y) - events)
            for model, probability in probabilities.items():
                auc = np.nan
                if events >= minimum_events and non_events >= minimum_non_events:
                    auc = roc_auc_score(segment_y, np.asarray(probability)[mask])
                rows.append({
                    "SegmentVariable": segment_name,
                    "SegmentValue": str(segment_value),
                    "Model": model,
                    "Rows": int(mask.sum()),
                    "BadCount": events,
                    "BadRate": float(segment_y.mean()) if len(segment_y) else np.nan,
                    "AUC": float(auc) if np.isfinite(auc) else np.nan,
                    "AUCReported": bool(np.isfinite(auc)),
                })
    return pd.DataFrame(rows)


def prior_probability_correction(
    weighted_probability,
    population_bad_rate: float,
    effective_training_bad_rate: float = 0.5,
) -> np.ndarray:
    """Correct posterior odds after class weighting changes the effective class prior."""
    probability = np.clip(np.asarray(weighted_probability, dtype=float), 1e-7, 1 - 1e-7)
    population_odds = population_bad_rate / (1 - population_bad_rate)
    effective_odds = effective_training_bad_rate / (1 - effective_training_bad_rate)
    corrected_odds = probability / (1 - probability) * population_odds / effective_odds
    return corrected_odds / (1 + corrected_odds)


def calibration_diagnostics(y_true, probability, model: str, method: str) -> dict[str, float | str]:
    """Return calibration-in-the-large, slope, O/E ratio, Brier score, and log loss."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    score = logit(p)

    def negative_log_likelihood(intercept: float) -> float:
        adjusted = expit(score + intercept)
        return float(-np.sum(y * np.log(adjusted) + (1 - y) * np.log(1 - adjusted)))

    intercept_result = minimize_scalar(negative_log_likelihood, bounds=(-10, 10), method="bounded")
    calibration_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
    calibration_model.fit(score.reshape(-1, 1), y)
    observed_rate = float(y.mean())
    predicted_rate = float(p.mean())
    return {
        "Model": model,
        "Calibration": method,
        "Rows": int(len(y)),
        "ObservedBadRate": observed_rate,
        "MeanPredictedPD": predicted_rate,
        "CalibrationInTheLarge": float(intercept_result.x),
        "CalibrationSlope": float(calibration_model.coef_[0, 0]),
        "ObservedExpectedRatio": observed_rate / predicted_rate if predicted_rate else np.nan,
        "Brier": float(brier_score_loss(y, p)),
        "LogLoss": float(log_loss(y, p)),
    }


def cost_sensitivity_table(
    y_true,
    probabilities: dict[str, np.ndarray],
    bad_approval_costs: tuple[float, ...] = (1.0, 5.0, 10.0),
    good_rejection_cost: float = 1.0,
) -> pd.DataFrame:
    """Find the lowest-cost threshold under explicitly illustrative cost ratios."""
    y = np.asarray(y_true, dtype=int)
    rows = []
    for model, values in probabilities.items():
        probability = np.asarray(values, dtype=float)
        thresholds = np.unique(np.quantile(probability, np.linspace(0.01, 0.99, 99)))
        for bad_cost in bad_approval_costs:
            candidates = []
            for threshold in thresholds:
                reject = probability >= threshold
                false_rejections = int(np.sum((y == 0) & reject))
                bad_approvals = int(np.sum((y == 1) & ~reject))
                total_cost = false_rejections * good_rejection_cost + bad_approvals * bad_cost
                candidates.append((total_cost, threshold, reject, false_rejections, bad_approvals))
            total_cost, threshold, reject, false_rejections, bad_approvals = min(candidates, key=lambda x: x[0])
            captured_bad = int(np.sum((y == 1) & reject))
            rows.append({
                "Model": model,
                "BadApprovalCost": float(bad_cost),
                "GoodRejectionCost": float(good_rejection_cost),
                "CostRatioBadToGood": float(bad_cost / good_rejection_cost),
                "SelectedThreshold": float(threshold),
                "RejectRate": float(reject.mean()),
                "BadCaptureRate": float(captured_bad / y.sum()),
                "FalseRejections": false_rejections,
                "BadApprovals": bad_approvals,
                "IllustrativeCostPerApplicant": float(total_cost / len(y)),
            })
    return pd.DataFrame(rows)

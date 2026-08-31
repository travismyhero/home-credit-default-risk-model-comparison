"""Common discrimination, probability-quality, ranking, and paired inference metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve


def _validated_binary_inputs(y_true, probability) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    prediction = np.asarray(probability, dtype=float)
    if y.ndim != 1 or prediction.ndim != 1 or len(y) != len(prediction):
        raise ValueError("Outcome and prediction must be aligned one-dimensional arrays")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation data must contain both binary outcome classes")
    if not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 1)):
        raise ValueError("Predictions must be finite values in [0, 1]")
    return y, prediction


def classification_metrics(y_true, probability) -> dict[str, float | int]:
    """Return the model-agnostic metrics used by every experiment branch."""
    y, prediction = _validated_binary_inputs(y_true, probability)
    auc = float(roc_auc_score(y, prediction))
    fpr, tpr, _ = roc_curve(y, prediction)
    return {
        "n": int(len(y)),
        "bad_rate": float(y.mean()),
        "AUC": auc,
        "AR": float(2 * auc - 1),
        "KS": float(np.max(tpr - fpr)),
        "AP": float(average_precision_score(y, prediction)),
        "Logloss": float(log_loss(y, prediction, labels=[0, 1])),
    }


def evaluate_predictions(
    predictions: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Evaluate named samples such as Train, Selection, and locked Test."""
    if set(predictions) != set(labels):
        raise ValueError("Prediction and label dictionaries must have identical sample names")
    return pd.DataFrame(
        [{"sample": sample, **classification_metrics(labels[sample], prediction)} for sample, prediction in predictions.items()]
    ).set_index("sample")


def risk_decile_table(y_true, probability, groups: int = 10) -> pd.DataFrame:
    """Build equal-count risk groups with group 1 representing highest predicted risk."""
    y, prediction = _validated_binary_inputs(y_true, probability)
    if groups < 2 or groups > len(y):
        raise ValueError("groups must be between 2 and the number of rows")
    ranked = pd.DataFrame({"target": y, "probability": prediction})
    ranked = ranked.sort_values("probability", ascending=False, kind="stable").reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["risk_group"] = np.minimum(np.arange(len(ranked)) * groups // len(ranked) + 1, groups)
    result = ranked.groupby("risk_group", observed=True).agg(
        rank_min=("rank", "min"),
        rank_max=("rank", "max"),
        n=("target", "size"),
        bad_n=("target", "sum"),
        bad_rate=("target", "mean"),
        probability_min=("probability", "min"),
        probability_max=("probability", "max"),
    ).reset_index()
    result["lift"] = result["bad_rate"] / y.mean()
    result["cumulative_bad_capture"] = result["bad_n"].cumsum() / y.sum()
    return result


def paired_bootstrap_vs_reference(
    y_true,
    probabilities: dict[str, np.ndarray],
    reference_model: str,
    n_bootstrap: int = 500,
    random_state: int = 42,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Estimate paired metric differences from a pre-declared linear reference."""
    if reference_model not in probabilities:
        raise ValueError(f"Reference model is missing: {reference_model}")
    y = np.asarray(y_true, dtype=int)
    validated = {name: _validated_binary_inputs(y, values)[1] for name, values in probabilities.items()}
    good = np.flatnonzero(y == 0)
    bad = np.flatnonzero(y == 1)
    rng = np.random.default_rng(random_state)
    alpha = (1 - confidence_level) / 2
    rows = []

    def metrics(yy, pp):
        result = classification_metrics(yy, pp)
        return {name: float(result[name]) for name in ("AUC", "AR", "KS", "AP", "Logloss")}

    point = {name: metrics(y, prediction) for name, prediction in validated.items()}
    for model, prediction in validated.items():
        if model == reference_model:
            continue
        draws = {metric: np.empty(n_bootstrap, dtype=float) for metric in point[model]}
        for draw in range(n_bootstrap):
            sampled = np.concatenate(
                [rng.choice(good, len(good), replace=True), rng.choice(bad, len(bad), replace=True)]
            )
            model_metrics = metrics(y[sampled], prediction[sampled])
            reference_metrics = metrics(y[sampled], validated[reference_model][sampled])
            for metric in draws:
                draws[metric][draw] = model_metrics[metric] - reference_metrics[metric]
        for metric, values in draws.items():
            rows.append(
                {
                    "model": model,
                    "reference_model": reference_model,
                    "metric": metric,
                    "difference": point[model][metric] - point[reference_model][metric],
                    "lower": float(np.quantile(values, alpha)),
                    "upper": float(np.quantile(values, 1 - alpha)),
                    "bootstrap_se": float(np.std(values, ddof=1)),
                    "confidence_level": float(confidence_level),
                    "bootstrap_replicates": int(n_bootstrap),
                    "higher_is_better": metric != "Logloss",
                }
            )
    return pd.DataFrame(rows)

"""WOE/IV scorecard-style Logistic Regression workflow using development data only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..artifacts import create_run_directory, write_frame, write_json
from ..development import DevelopmentData
from ..metrics import evaluate_predictions
from ..profiling import MISSING_LABEL, RARE_LABEL, category_text, feature_quality_report, is_categorical
from ..test_policy import register_model_lock


@dataclass
class WOELogisticResult:
    run_directory: Path
    metrics: pd.DataFrame
    iv_table: pd.DataFrame
    final_parameter_table: pd.DataFrame
    selection_history: pd.DataFrame
    final_features: list[str]
    locked_config: dict[str, Any]
    train_prediction: np.ndarray
    selection_prediction: np.ndarray


def _woe_table(labels: pd.Series, y: pd.Series, tiny_missing_n: int = 100) -> tuple[pd.DataFrame, float]:
    temporary = pd.DataFrame({"bin": labels.astype(object), "target": np.asarray(y)})
    result = (
        temporary.groupby("bin", dropna=False, observed=False, sort=False)["target"]
        .agg(["count", "sum"])
        .rename(columns={"sum": "bad"})
    )
    result["good"] = result["count"] - result["bad"]
    total_bad, total_good = result.bad.sum(), result.good.sum()
    groups = len(result)
    result["bad_pct"] = (result.bad + 0.5) / (total_bad + 0.5 * groups)
    result["good_pct"] = (result.good + 0.5) / (total_good + 0.5 * groups)
    result["woe"] = np.log(result.bad_pct / result.good_pct)
    if MISSING_LABEL in result.index and result.loc[MISSING_LABEL, "count"] < tiny_missing_n:
        result.loc[MISSING_LABEL, "woe"] = 0.0
        result.loc[MISSING_LABEL, "bad_pct"] = result.loc[MISSING_LABEL, "good_pct"]
    result["iv_bin"] = (result.bad_pct - result.good_pct) * result.woe
    result["bad_rate"] = result.bad / result["count"]
    return result.reset_index(), float(result.iv_bin.sum())


def _numeric_labels(series: pd.Series, edges: np.ndarray) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    labels = pd.Series(
        pd.cut(numeric, bins=edges, include_lowest=True, duplicates="drop").astype(object),
        index=series.index,
    )
    labels.loc[numeric.isna()] = MISSING_LABEL
    return labels


def fit_numeric_woe(
    series: pd.Series,
    y: pd.Series,
    quantile_bins: int = 10,
    minimum_nonmissing_bins: int = 2,
    tiny_missing_n: int = 100,
) -> dict[str, Any] | None:
    numeric = pd.to_numeric(series, errors="coerce")
    nonmissing = numeric.notna()
    unique = int(numeric[nonmissing].nunique())
    if nonmissing.sum() == 0 or unique < 2:
        return None
    try:
        _, edges = pd.qcut(
            numeric[nonmissing], q=min(quantile_bins, unique), retbins=True, duplicates="drop"
        )
    except ValueError:
        edges = np.unique(
            np.nanquantile(numeric[nonmissing], np.linspace(0, 1, min(quantile_bins, unique) + 1))
        )
    edges = np.unique(np.asarray(edges, dtype=float))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf

    def ordered_stats(current_edges):
        bins = pd.cut(numeric[nonmissing], current_edges, include_lowest=True, duplicates="drop")
        temporary = pd.DataFrame(
            {"bin": bins, "target": np.asarray(y)[nonmissing.to_numpy()]}
        )
        return temporary.groupby("bin", observed=True)["target"].agg(["count", "mean"]).reset_index()

    initial = ordered_stats(edges)
    if len(initial) < 2:
        return None
    rho = spearmanr(np.arange(len(initial)), initial["mean"]).statistic
    direction = "increasing" if np.isnan(rho) or rho >= 0 else "decreasing"

    while True:
        stats = ordered_stats(edges)
        rates = stats["mean"].to_numpy()
        violation = None
        for index in range(1, len(rates)):
            if direction == "increasing" and rates[index] < rates[index - 1]:
                violation = index
                break
            if direction == "decreasing" and rates[index] > rates[index - 1]:
                violation = index
                break
        if violation is None:
            break
        if len(edges) <= 3:
            direction = "increasing" if rates[-1] >= rates[0] else "decreasing"
            break
        edges = np.delete(edges, violation)

    if len(ordered_stats(edges)) < minimum_nonmissing_bins:
        return None
    table, iv = _woe_table(_numeric_labels(numeric, edges), y, tiny_missing_n=tiny_missing_n)
    return {
        "type": "numeric",
        "edges": edges,
        "direction": direction,
        "table": table,
        "iv": iv,
        "woe_map": dict(zip(table["bin"], table["woe"])),
    }


def fit_categorical_woe(
    series: pd.Series,
    y: pd.Series,
    rare_share: float = 0.005,
    rare_min_count: int = 200,
    tiny_missing_n: int = 100,
) -> dict[str, Any]:
    values = category_text(series)
    counts = values.value_counts()
    threshold = max(rare_min_count, int(len(values) * rare_share))
    rare = set(counts[(counts < threshold) & (counts.index != MISSING_LABEL)].index)
    grouped = values.where(~values.isin(rare), RARE_LABEL)
    table, iv = _woe_table(grouped, y, tiny_missing_n=tiny_missing_n)
    if RARE_LABEL in table["bin"].values:
        rare_mask = table.bin.eq(RARE_LABEL)
        if int(table.loc[rare_mask, "count"].iloc[0]) < rare_min_count:
            table.loc[rare_mask, ["woe", "iv_bin"]] = 0.0
            iv = float(table.iv_bin.sum())
    return {
        "type": "categorical",
        "rare_levels": rare,
        "table": table,
        "iv": iv,
        "woe_map": dict(zip(table["bin"], table["woe"])),
    }


def fit_feature_woe(series: pd.Series, y: pd.Series, quantile_bins: int = 10):
    unique = int(series.nunique(dropna=True))
    if unique < 2:
        return None
    if is_categorical(series) or unique <= 10:
        return fit_categorical_woe(series, y)
    return fit_numeric_woe(series, y, quantile_bins=quantile_bins)


def transform_feature_woe(series: pd.Series, spec: dict[str, Any]) -> pd.Series:
    if spec["type"] == "numeric":
        labels = _numeric_labels(series, spec["edges"])
    else:
        labels = category_text(series)
        labels = labels.where(~labels.isin(spec["rare_levels"]), RARE_LABEL)
    return labels.map(spec["woe_map"]).fillna(0.0).astype(float)


def transform_woe_frame(frame: pd.DataFrame, specs: dict[str, dict[str, Any]], features: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {feature: transform_feature_woe(frame[feature], specs[feature]) for feature in features},
        index=frame.index,
    )


def correlation_prune(
    X_woe: pd.DataFrame,
    feature_info: pd.DataFrame,
    threshold: float = 0.70,
    ar_gap: float = 0.03,
) -> tuple[list[str], pd.DataFrame]:
    info = feature_info.set_index("feature")
    correlation = X_woe.corr().abs()
    features = X_woe.columns.tolist()
    dropped, decisions = set(), []

    def choose(first, second):
        first_ar, second_ar = info.loc[first, "single_ar"], info.loc[second, "single_ar"]
        if abs(first_ar - second_ar) >= ar_gap:
            keep = first if first_ar >= second_ar else second
            reason = "higher_single_AR"
        else:
            first_missing, second_missing = info.loc[first, "missing_rate"], info.loc[second, "missing_rate"]
            if not np.isclose(first_missing, second_missing):
                keep = first if first_missing <= second_missing else second
                reason = "AR_close_lower_missing"
            else:
                keep = first if info.loc[first, "iv"] >= info.loc[second, "iv"] else second
                reason = "higher_IV"
        remove = second if keep == first else first
        return keep, remove, reason

    for index, first in enumerate(features):
        if first in dropped:
            continue
        for second in features[index + 1 :]:
            if second in dropped:
                continue
            value = correlation.loc[first, second]
            if np.isfinite(value) and value > threshold:
                keep, remove, reason = choose(first, second)
                dropped.add(remove)
                decisions.append(
                    {
                        "feature_a": first,
                        "feature_b": second,
                        "correlation": value,
                        "keep": keep,
                        "drop": remove,
                        "reason": reason,
                    }
                )
                if remove == first:
                    break
    return [feature for feature in features if feature not in dropped], pd.DataFrame(decisions)


def calculate_vif(frame: pd.DataFrame) -> pd.DataFrame:
    values = sm.add_constant(frame, has_constant="add").to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {"feature": feature, "VIF": variance_inflation_factor(values, index)}
            for index, feature in enumerate(frame.columns, start=1)
        ]
    ).sort_values("VIF", ascending=False)


def fit_statsmodels_logit(X: pd.DataFrame, y: pd.Series, features: list[str]):
    design = sm.add_constant(X[features], has_constant="add").astype(float)
    return sm.Logit(y.astype(float), design).fit(disp=0, maxiter=200)


def refine_logit_features(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    p_value_max: float = 0.05,
    vif_max: float = 10.0,
    enforce_positive: bool = True,
):
    selected, history = list(features), []
    while len(selected) > 1:
        vif = calculate_vif(X[selected])
        if len(vif) and (not np.isfinite(vif.iloc[0].VIF) or vif.iloc[0].VIF > vif_max):
            remove = str(vif.iloc[0].feature)
            history.append({"feature": remove, "reason": "VIF", "value": float(vif.iloc[0].VIF)})
            selected.remove(remove)
            continue
        result = fit_statsmodels_logit(X, y, selected)
        p_values = result.pvalues.drop("const")
        if p_values.max() > p_value_max:
            remove = str(p_values.idxmax())
            history.append({"feature": remove, "reason": "P_value", "value": float(p_values.max())})
            selected.remove(remove)
            continue
        if enforce_positive:
            negative = result.params.drop("const").loc[lambda values: values < 0]
            if len(negative):
                remove = str(negative.idxmin())
                history.append(
                    {"feature": remove, "reason": "negative_coefficient", "value": float(negative.min())}
                )
                selected.remove(remove)
                continue
        return selected, result, pd.DataFrame(history)
    result = fit_statsmodels_logit(X, y, selected)
    return selected, result, pd.DataFrame(history)


def run_woe_logistic(
    development: DevelopmentData,
    output_directory: str | Path | None = None,
    register_lock: bool = True,
) -> WOELogisticResult:
    """Fit every WOE and feature-selection rule on Train; use Selection only for evaluation."""
    settings = development.configs["model_search_spaces"]["woe_logistic"]
    experiment = development.configs["experiment"]
    run_directory = Path(output_directory) if output_directory else create_run_directory(
        development.project_root, "woe_logistic"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    quality = feature_quality_report(development.X_train, development.y_train)
    candidates = quality.loc[~quality.hard_drop, "feature"].tolist()
    specs, iv_rows = {}, []
    for feature in candidates:
        spec = fit_feature_woe(development.X_train[feature], development.y_train)
        if spec is None:
            continue
        specs[feature] = spec
        iv_rows.append(
            {
                "feature": feature,
                "iv": spec["iv"],
                "type": spec["type"],
                "bins": len(spec["table"]),
                "direction": spec.get("direction"),
                "missing_rate": float(development.X_train[feature].isna().mean()),
            }
        )
    iv_table = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
    iv_features = iv_table.loc[iv_table.iv >= float(settings["iv_min"]), "feature"].tolist()
    if not iv_features:
        raise RuntimeError("No feature passed the configured IV threshold")

    X_train_woe = transform_woe_frame(development.X_train, specs, iv_features)
    X_selection_woe = transform_woe_frame(development.X_selection, specs, iv_features)
    single_ar = {}
    for feature in iv_features:
        auc = roc_auc_score(development.y_train, X_train_woe[feature])
        single_ar[feature] = float(2 * auc - 1)
    iv_table["single_ar"] = iv_table.feature.map(single_ar)
    correlation_features, correlation_decisions = correlation_prune(
        X_train_woe,
        iv_table.dropna(subset=["single_ar"]),
        threshold=float(settings["correlation_threshold"]),
        ar_gap=float(settings["single_ar_gap"]),
    )
    final_features, statsmodels_result, selection_history = refine_logit_features(
        X_train_woe,
        development.y_train,
        correlation_features,
        p_value_max=float(settings["p_value_max"]),
        vif_max=float(settings["vif_max"]),
        enforce_positive=bool(settings["enforce_positive_coefficients"]),
    )
    final_vif = calculate_vif(X_train_woe[final_features])
    final_parameter_table = pd.DataFrame(
        {
            "feature": final_features,
            "coefficient": statsmodels_result.params[final_features].to_numpy(),
            "p_value": statsmodels_result.pvalues[final_features].to_numpy(),
        }
    ).merge(
        iv_table[["feature", "iv", "single_ar", "missing_rate"]], on="feature", how="left"
    ).merge(final_vif, on="feature", how="left")

    final_model = LogisticRegression(C=np.inf, l1_ratio=0.0, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        final_model.fit(X_train_woe[final_features], development.y_train)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(f"WOE Logistic failed to converge: {convergence[-1].message}")
    train_prediction = final_model.predict_proba(X_train_woe[final_features])[:, 1]
    selection_prediction = final_model.predict_proba(X_selection_woe[final_features])[:, 1]
    metrics = evaluate_predictions(
        {"Train": train_prediction, "Selection": selection_prediction},
        {"Train": development.y_train, "Selection": development.y_selection},
    ).reset_index()
    selection_ar = float(metrics.loc[metrics["sample"] == "Selection", "AR"].iloc[0])
    locked_config = {
        "model_name": "woe_logistic",
        "experiment_track": "best_practice",
        "data_sha256": development.data_sha256,
        "split_sha256": development.split_sha256,
        "feature_policy_version": development.configs["feature_policy"]["policy"]["version"],
        "training_sample": "Train only",
        "selection_role": "evaluation only after Train-fitted scorecard selection",
        "test_accessed": False,
        "parameters": settings,
        "logistic_model": {"C": "infinity", "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 1000},
        "input_feature_count": len(candidates),
        "woe_feature_count": len(specs),
        "iv_selected_count": len(iv_features),
        "final_feature_count": len(final_features),
        "final_features": final_features,
        "selection_AR": selection_ar,
        "train_selection_AR_gap": float(
            metrics.loc[metrics["sample"] == "Train", "AR"].iloc[0] - selection_ar
        ),
        "fit_seconds": perf_counter() - started,
        "artifact_directory": str(run_directory),
    }

    bin_tables = []
    for feature, spec in specs.items():
        table = spec["table"].copy()
        table.insert(0, "feature", feature)
        table["bin"] = table["bin"].astype(str)
        bin_tables.append(table)
    joblib.dump(
        {"specs": specs, "features": final_features, "model": final_model},
        run_directory / "woe_model_bundle.joblib",
    )
    write_frame(run_directory / "feature_quality.csv", quality)
    write_frame(run_directory / "iv_table.csv", iv_table)
    write_frame(run_directory / "woe_bin_tables.csv", pd.concat(bin_tables, ignore_index=True))
    write_frame(run_directory / "correlation_decisions.csv", correlation_decisions)
    write_frame(run_directory / "selection_history.csv", selection_history)
    write_frame(run_directory / "final_parameter_table.csv", final_parameter_table)
    write_frame(run_directory / "development_metrics.csv", metrics)
    write_frame(
        run_directory / "development_predictions.csv",
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "sample": "Train",
                        "SK_ID_CURR": development.id_train,
                        "TARGET": development.y_train,
                        "prediction": train_prediction,
                    }
                ),
                pd.DataFrame(
                    {
                        "sample": "Selection",
                        "SK_ID_CURR": development.id_selection,
                        "TARGET": development.y_selection,
                        "prediction": selection_prediction,
                    }
                ),
            ],
            ignore_index=True,
        ),
    )
    write_json(run_directory / "locked_config.json", locked_config)
    if register_lock:
        policy = experiment["test_policy"]
        register_model_lock(
            development.project_root / policy["registry_path"],
            "woe_logistic",
            locked_config,
            policy["required_models"],
        )
    return WOELogisticResult(
        run_directory=run_directory,
        metrics=metrics,
        iv_table=iv_table,
        final_parameter_table=final_parameter_table,
        selection_history=selection_history,
        final_features=final_features,
        locked_config=locked_config,
        train_prediction=train_prediction,
        selection_prediction=selection_prediction,
    )

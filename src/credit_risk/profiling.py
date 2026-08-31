"""Train-fitted, Selection-validated model-agnostic profiling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .artifacts import create_run_directory, write_frame, write_json
from .development import DevelopmentData


MISSING_LABEL = "__MISSING__"
RARE_LABEL = "__RARE__"
UNSEEN_LABEL = "__UNSEEN__"


@dataclass
class DevelopmentProfileResult:
    run_directory: Path
    feature_quality: pd.DataFrame
    missing: pd.DataFrame
    category_summary: pd.DataFrame
    category_levels: pd.DataFrame
    psi_summary: pd.DataFrame
    psi_detail: pd.DataFrame
    numeric_anomalies: pd.DataFrame
    decisions: pd.DataFrame
    candidate_features: list[str]
    review_features: list[str]


def is_categorical(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    )


def category_text(series: pd.Series) -> pd.Series:
    return ("V:" + series.astype("string")).fillna(MISSING_LABEL)


def feature_quality_report(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    high_missing: float = 0.95,
    quasi_constant: float = 0.995,
    id_like_ratio: float = 0.95,
    low_cardinality_max: int = 10,
) -> pd.DataFrame:
    """Build common hard-drop and review flags using Train only."""
    rows = []
    for feature in X_train.columns:
        series = X_train[feature]
        missing = series.isna()
        unique = int(series.nunique(dropna=True))
        categorical = is_categorical(series)
        unique_ratio = unique / max(int((~missing).sum()), 1)
        top_frequency = float(series.value_counts(dropna=False, normalize=True).iloc[0])
        if missing.all():
            hard_reason = "all_missing"
        elif series.nunique(dropna=False) <= 1:
            hard_reason = "constant"
        elif categorical and unique_ratio > id_like_ratio:
            hard_reason = "id_like_categorical"
        else:
            hard_reason = ""
        review = []
        if missing.mean() > high_missing:
            review.append("high_missing")
        if top_frequency > quasi_constant and not hard_reason:
            review.append("quasi_constant")
        low_card = (not categorical) and unique <= low_cardinality_max
        if low_card:
            review.append("low_card_numeric_review")
        rows.append(
            {
                "feature": feature,
                "dtype": str(series.dtype),
                "physical_categorical": categorical,
                "low_card_numeric_candidate": low_card,
                "n_unique_nonmissing": unique,
                "unique_ratio_nonmissing": unique_ratio,
                "missing_n": int(missing.sum()),
                "missing_rate": float(missing.mean()),
                "top_frequency": top_frequency,
                "bad_rate_missing": float(y_train[missing].mean()) if missing.any() else np.nan,
                "bad_rate_nonmissing": float(y_train[~missing].mean()) if (~missing).any() else np.nan,
                "hard_drop": bool(hard_reason),
                "hard_drop_reason": hard_reason,
                "quality_warning": "|".join(review),
            }
        )
    return pd.DataFrame(rows)


def missing_report(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_selection: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    result = quality[
        ["feature", "dtype", "missing_n", "missing_rate", "bad_rate_missing", "bad_rate_nonmissing"]
    ].copy()
    result["bad_rate_gap_missing_minus_nonmissing"] = result.bad_rate_missing - result.bad_rate_nonmissing
    result["abs_bad_rate_gap"] = result.bad_rate_gap_missing_minus_nonmissing.abs()
    result["selection_missing_rate"] = X_selection.isna().mean().reindex(result.feature).to_numpy()
    result["selection_minus_train_missing_rate"] = result.selection_missing_rate - result.missing_rate
    return result.sort_values(["missing_rate", "abs_bad_rate_gap"], ascending=False).reset_index(drop=True)


def category_reports(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_selection: pd.DataFrame,
    quality: pd.DataFrame,
    rare_min_count: int = 50,
    rare_min_share: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scan = quality.loc[
        quality.physical_categorical | quality.low_card_numeric_candidate,
        ["feature", "physical_categorical"],
    ]
    summaries, details = [], []
    for row in scan.itertuples(index=False):
        feature = row.feature
        train = category_text(X_train[feature])
        selection = category_text(X_selection[feature])
        known = set(train.unique())
        threshold = max(rare_min_count, int(np.ceil(len(train) * rare_min_share)))
        counts = train.value_counts()
        rare = set(counts[counts < threshold].index) - {MISSING_LABEL}
        detail = (
            pd.DataFrame({"level": train, "target": y_train})
            .groupby("level", observed=False)["target"]
            .agg(["count", "sum", "mean"])
            .reset_index()
            .rename(columns={"sum": "bad_n", "mean": "bad_rate"})
        )
        detail.insert(0, "feature", feature)
        detail["share"] = detail["count"] / len(train)
        detail["is_missing"] = detail.level.eq(MISSING_LABEL)
        detail["is_rare"] = detail.level.isin(rare)
        details.append(detail)
        summaries.append(
            {
                "feature": feature,
                "category_kind": "physical_categorical" if row.physical_categorical else "low_card_numeric_candidate",
                "levels_nonmissing_train": int(X_train[feature].nunique(dropna=True)),
                "missing_rate_train": float(X_train[feature].isna().mean()),
                "top_level_share_train": float(counts.iloc[0] / len(train)),
                "rare_level_n": len(rare),
                "rare_sample_share_train": float(train.isin(rare).mean()),
                "selection_unseen_level_n": len(set(selection.unique()) - known),
                "selection_unseen_sample_rate": float((~selection.isin(known)).mean()),
            }
        )
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def fit_psi_spec(series: pd.Series, bins: int = 10, rare_min_count: int = 50, rare_min_share: float = 0.001):
    if pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() >= 2:
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        edges = np.unique(np.nanquantile(values.dropna(), np.linspace(0, 1, bins + 1))).astype(float)
        if len(edges) >= 3:
            edges[0], edges[-1] = -np.inf, np.inf
            return {"type": "numeric", "edges": edges}
    labels = category_text(series)
    counts = labels.value_counts()
    threshold = max(rare_min_count, int(np.ceil(len(labels) * rare_min_share)))
    rare = set(counts[counts < threshold].index) - {MISSING_LABEL}
    return {"type": "categorical", "known": set(labels.unique()) - rare, "rare": rare}


def apply_psi_spec(series: pd.Series, spec: dict[str, Any], reference: bool) -> pd.Series:
    if spec["type"] == "numeric":
        values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return pd.cut(values, spec["edges"], include_lowest=True, duplicates="drop").astype("string").fillna(MISSING_LABEL)
    labels = category_text(series).where(~category_text(series).isin(spec["rare"]), RARE_LABEL)
    if not reference:
        allowed = spec["known"] | {RARE_LABEL, MISSING_LABEL}
        labels = labels.where(labels.isin(allowed), UNSEEN_LABEL)
    return labels


def psi_reports(
    X_train: pd.DataFrame,
    X_selection: pd.DataFrame,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    summaries, details, specs = [], [], {}
    for feature in X_train.columns:
        spec = fit_psi_spec(X_train[feature], bins=bins)
        specs[feature] = spec
        reference = apply_psi_spec(X_train[feature], spec, reference=True)
        comparison = apply_psi_spec(X_selection[feature], spec, reference=False)
        ref_count = reference.value_counts()
        cmp_count = comparison.value_counts()
        levels = ref_count.index.union(cmp_count.index)
        detail = pd.DataFrame({"bin": levels})
        detail["reference_n"] = detail.bin.map(ref_count).fillna(0).astype(int)
        detail["selection_n"] = detail.bin.map(cmp_count).fillna(0).astype(int)
        detail["reference_pct"] = (detail.reference_n / len(reference)).clip(lower=epsilon)
        detail["selection_pct"] = (detail.selection_n / len(comparison)).clip(lower=epsilon)
        detail["psi_component"] = (detail.reference_pct - detail.selection_pct) * np.log(
            detail.reference_pct / detail.selection_pct
        )
        detail.insert(0, "feature", feature)
        details.append(detail)
        psi = float(detail.psi_component.sum())
        summaries.append(
            {
                "feature": feature,
                "psi_type": spec["type"],
                "psi_selection": psi,
                "psi_flag": "investigate" if psi >= 0.25 else ("watch" if psi >= 0.10 else "stable"),
            }
        )
    return (
        pd.DataFrame(summaries).sort_values("psi_selection", ascending=False).reset_index(drop=True),
        pd.concat(details, ignore_index=True),
        specs,
    )


def numeric_anomaly_report(
    X_train: pd.DataFrame,
    X_selection: pd.DataFrame,
    known_special_values: dict[str, list[float]] | None = None,
    quantiles: tuple[float, float] = (0.005, 0.995),
) -> pd.DataFrame:
    special_values = known_special_values or {}
    rows = []
    for feature in X_train.select_dtypes(include="number"):
        raw = pd.to_numeric(X_train[feature], errors="coerce")
        infinity = np.isinf(raw)
        series = raw.mask(infinity)
        valid = series.dropna()
        if valid.empty:
            continue
        low_tail, q25, median, q75, high_tail = valid.quantile([quantiles[0], 0.25, 0.5, 0.75, quantiles[1]])
        iqr = q75 - q25
        iqr_low, iqr_high = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        selection = pd.to_numeric(X_selection[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        special = {str(value): int(series.eq(value).sum()) for value in special_values.get(feature, [])}
        rows.append(
            {
                "feature": feature,
                "valid_n": int(valid.size),
                "inf_n": int(infinity.sum()),
                "min": float(valid.min()),
                "q005": float(low_tail),
                "median": float(median),
                "q995": float(high_tail),
                "max": float(valid.max()),
                "iqr_low": float(iqr_low),
                "iqr_high": float(iqr_high),
                "iqr_outlier_rate": float(((series < iqr_low) | (series > iqr_high)).mean()),
                "selection_outside_train_minmax_rate": float(
                    ((selection < valid.min()) | (selection > valid.max())).mean()
                ),
                "selection_outside_train_q005_q995_rate": float(
                    ((selection < low_tail) | (selection > high_tail)).mean()
                ),
                "known_special_counts": json.dumps(special, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values("iqr_outlier_rate", ascending=False).reset_index(drop=True)


def decision_report(
    quality: pd.DataFrame,
    psi: pd.DataFrame,
    categories: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> pd.DataFrame:
    result = quality.merge(psi, on="feature", how="left")
    if not categories.empty:
        result = result.merge(
            categories[["feature", "selection_unseen_sample_rate"]], on="feature", how="left"
        )
    if not anomalies.empty:
        result = result.merge(
            anomalies[["feature", "inf_n", "iqr_outlier_rate", "known_special_counts"]],
            on="feature",
            how="left",
        )

    def reasons(row):
        values = row.quality_warning.split("|") if row.quality_warning else []
        if row.psi_flag in {"watch", "investigate"}:
            values.append(f"psi_{row.psi_flag}")
        if pd.notna(row.get("inf_n")) and row.get("inf_n", 0) > 0:
            values.append("infinite_values")
        if pd.notna(row.get("selection_unseen_sample_rate")) and row.get("selection_unseen_sample_rate", 0) > 0:
            values.append("selection_unseen_categories")
        special = row.get("known_special_counts")
        if isinstance(special, str) and special != "{}":
            values.append("known_special_value")
        return "|".join(dict.fromkeys(values))

    result["review_reason"] = result.apply(reasons, axis=1)
    result["recommended_action"] = np.where(
        result.hard_drop, "drop", np.where(result.review_reason.ne(""), "review", "keep_candidate")
    )
    return result.sort_values(["hard_drop", "psi_selection", "missing_rate"], ascending=False).reset_index(drop=True)


def run_development_profile(
    development: DevelopmentData,
    output_directory: str | Path | None = None,
) -> DevelopmentProfileResult:
    """Run and persist the model-agnostic Train/Selection profile without materializing Test."""
    feature_policy = development.configs["feature_policy"]
    quality_policy = feature_policy["quality_screen"]["review_only"]
    categorical_policy = feature_policy["categorical"]
    run_directory = Path(output_directory) if output_directory else create_run_directory(
        development.project_root, "data_profile"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    quality = feature_quality_report(
        development.X_train,
        development.y_train,
        high_missing=float(quality_policy["high_missing_threshold"]),
        quasi_constant=float(quality_policy["quasi_constant_threshold"]),
        low_cardinality_max=int(quality_policy["low_cardinality_numeric_max"]),
    )
    missing = missing_report(development.X_train, development.y_train, development.X_selection, quality)
    category_summary, category_levels = category_reports(
        development.X_train,
        development.y_train,
        development.X_selection,
        quality,
        rare_min_count=int(categorical_policy["rare_min_count"]),
        rare_min_share=float(categorical_policy["rare_min_share"]),
    )
    psi_summary, psi_detail, psi_specs = psi_reports(development.X_train, development.X_selection)
    anomalies = numeric_anomaly_report(
        development.X_train,
        development.X_selection,
        known_special_values=feature_policy.get("known_special_values", {}),
    )
    decisions = decision_report(quality, psi_summary, category_summary, anomalies)
    candidate_features = decisions.loc[~decisions.hard_drop, "feature"].tolist()
    review_features = decisions.loc[
        (~decisions.hard_drop) & decisions.review_reason.ne(""), "feature"
    ].tolist()
    reports = {
        "feature_quality.csv": quality,
        "missing_report.csv": missing,
        "category_summary.csv": category_summary,
        "category_levels_train.csv": category_levels,
        "psi_selection_summary.csv": psi_summary,
        "psi_selection_detail.csv": psi_detail,
        "numeric_anomaly_report.csv": anomalies,
        "feature_decision_report.csv": decisions,
    }
    for filename, report in reports.items():
        write_frame(run_directory / filename, report)
    joblib.dump(psi_specs, run_directory / "psi_train_specs.joblib")
    write_json(
        run_directory / "profile_manifest.json",
        {
            "data_sha256": development.data_sha256,
            "split_sha256": development.split_sha256,
            "feature_policy_version": feature_policy["policy"]["version"],
            "profile_samples": ["Train", "Selection"],
            "test_accessed": False,
            "input_feature_count": len(development.feature_columns),
            "candidate_feature_count": len(candidate_features),
            "review_feature_count": len(review_features),
            "hard_drop_feature_count": int(decisions.hard_drop.sum()),
            "candidate_features": candidate_features,
            "review_features": review_features,
            "hard_drop_features": decisions.loc[decisions.hard_drop, "feature"].tolist(),
        },
    )
    return DevelopmentProfileResult(
        run_directory=run_directory,
        feature_quality=quality,
        missing=missing,
        category_summary=category_summary,
        category_levels=category_levels,
        psi_summary=psi_summary,
        psi_detail=psi_detail,
        numeric_anomalies=anomalies,
        decisions=decisions,
        candidate_features=candidate_features,
        review_features=review_features,
    )

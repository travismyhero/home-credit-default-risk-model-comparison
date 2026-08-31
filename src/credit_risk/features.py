"""Fixed, row-wise feature engineering shared by every controlled experiment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_DAY_TO_YEAR = {
    "DAYS_BIRTH": "AGE_YEARS",
    "DAYS_EMPLOYED": "EMPLOYED_YEARS",
    "DAYS_REGISTRATION": "REGISTRATION_YEARS",
    "DAYS_ID_PUBLISH": "ID_PUBLISH_YEARS",
    "DAYS_LAST_PHONE_CHANGE": "PHONE_CHANGE_YEARS",
}

DEFAULT_RATIOS = {
    "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "CREDIT_ANNUITY_RATIO": ("AMT_CREDIT", "AMT_ANNUITY"),
    "GOODS_CREDIT_RATIO": ("AMT_GOODS_PRICE", "AMT_CREDIT"),
    "INCOME_PER_PERSON": ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
    "EMPLOYED_AGE_RATIO": ("EMPLOYED_YEARS", "AGE_YEARS"),
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while preserving zero-denominator and non-finite results as missing."""
    result = numerator.div(denominator.replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan)


def validate_raw_schema(
    raw: pd.DataFrame,
    required_columns: Iterable[str],
    numeric_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Select an ordered raw schema and reject invalid non-empty numeric text."""
    if not isinstance(raw, pd.DataFrame) or not raw.columns.is_unique:
        raise ValueError("Input must be a DataFrame with unique column names")
    required = list(required_columns)
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required model-input columns: {missing}")
    result = raw.loc[:, required].copy()
    for column in numeric_columns:
        if column not in result:
            continue
        converted = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"{column} contains {int(invalid.sum())} non-empty values that are not numeric")
        result[column] = converted
    return result


def prepare_fixed_features(
    raw: pd.DataFrame,
    exclude_features: Iterable[str] = (),
    day_to_year: Mapping[str, str] = DEFAULT_DAY_TO_YEAR,
    ratios: Mapping[str, tuple[str, str] | list[str]] = DEFAULT_RATIOS,
    drop_original_days: bool = True,
    cast_numeric_float32: bool = True,
) -> pd.DataFrame:
    """Apply only fixed, label-free transformations that depend on one applicant row."""
    if not isinstance(raw, pd.DataFrame) or not raw.columns.is_unique:
        raise ValueError("Raw features must be a DataFrame with unique columns")
    result = raw.drop(columns=list(exclude_features), errors="ignore").copy()
    numeric = result.select_dtypes(include="number").columns
    result.loc[:, numeric] = result.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    derived: dict[str, pd.Series] = {}

    if "DAYS_EMPLOYED" in result:
        special = result["DAYS_EMPLOYED"].eq(365243)
        derived["EMPLOYED_SPECIAL_FLAG"] = special.astype("int8")
        result["DAYS_EMPLOYED"] = result["DAYS_EMPLOYED"].mask(special)

    converted_day_columns = []
    for source, destination in day_to_year.items():
        if source not in result:
            continue
        derived[destination] = -pd.to_numeric(result[source], errors="raise") / 365.25
        converted_day_columns.append(source)

    for name, pair in ratios.items():
        numerator, denominator = pair
        available = set(result.columns) | set(derived)
        if numerator in available and denominator in available:
            numerator_values = derived[numerator] if numerator in derived else result[numerator]
            denominator_values = derived[denominator] if denominator in derived else result[denominator]
            derived[name] = safe_divide(
                pd.to_numeric(numerator_values, errors="raise"),
                pd.to_numeric(denominator_values, errors="raise"),
            )

    if drop_original_days and converted_day_columns:
        result = result.drop(columns=converted_day_columns)
    if derived:
        result = pd.concat([result, pd.DataFrame(derived, index=result.index)], axis=1)

    numeric = result.select_dtypes(include="number").columns
    result.loc[:, numeric] = result.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    if cast_numeric_float32:
        result[numeric] = result[numeric].astype("float32")
    return result.copy()

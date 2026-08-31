"""Shared candidate-selection rules for model-family searches."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def choose_best_candidate(
    results: pd.DataFrame,
    primary_metric: str = "selection_ar",
    tolerance: float = 0.005,
    gap_metric: str = "ar_gap",
    capacity_columns: Sequence[str] = ("capacity", "best_iteration"),
) -> pd.Series:
    """Shortlist near-best Selection results, then prefer stable and simpler candidates."""
    if results.empty:
        raise ValueError("Candidate results are empty")
    required = {primary_metric, gap_metric}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Candidate results are missing columns: {missing}")
    if tolerance < 0:
        raise ValueError("Metric tolerance must be non-negative")
    if not np.isfinite(results[primary_metric]).all() or not np.isfinite(results[gap_metric]).all():
        raise ValueError("Selection metric and gap must be finite")

    best_value = float(results[primary_metric].max())
    shortlist = results.loc[results[primary_metric] >= best_value - tolerance].copy()
    complexity = [column for column in capacity_columns if column in shortlist]
    sort_columns = [gap_metric, *complexity, primary_metric]
    ascending = [True] * (1 + len(complexity)) + [False]
    return shortlist.sort_values(sort_columns, ascending=ascending, kind="stable").iloc[0]

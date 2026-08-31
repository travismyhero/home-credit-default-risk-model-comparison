"""Development-sample comparisons against already locked reference models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .development import DevelopmentData


def load_locked_selection_prediction(
    development: DevelopmentData,
    model_name: str,
) -> np.ndarray:
    """Load and align one locked model's Selection prediction without exposing Test."""
    registry_path = (
        development.project_root
        / development.configs["experiment"]["test_policy"]["registry_path"]
    )
    if not registry_path.is_file():
        raise FileNotFoundError(f"Model lock registry does not exist: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry.get("models", {}).get(model_name, {})
    if entry.get("locked") is not True:
        raise ValueError(f"Reference model is not locked: {model_name}")
    config = entry.get("config", {})
    if config.get("data_sha256") != development.data_sha256:
        raise ValueError(f"Reference model uses a different data fingerprint: {model_name}")
    if config.get("split_sha256") != development.split_sha256:
        raise ValueError(f"Reference model uses a different split fingerprint: {model_name}")

    prediction_path = Path(config["artifact_directory"]) / "development_predictions.csv"
    predictions = pd.read_csv(prediction_path)
    selection = predictions.loc[
        predictions["sample"].eq("Selection"),
        ["SK_ID_CURR", "TARGET", "prediction"],
    ]
    expected = pd.DataFrame(
        {
            "row_order": np.arange(len(development.id_selection)),
            "SK_ID_CURR": np.asarray(development.id_selection),
            "expected_target": np.asarray(development.y_selection),
        }
    )
    aligned = expected.merge(
        selection,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    ).sort_values("row_order")
    if aligned["prediction"].isna().any() or len(aligned) != len(expected):
        raise ValueError(f"Reference Selection predictions are incomplete: {model_name}")
    if not np.array_equal(aligned["TARGET"].to_numpy(), aligned["expected_target"].to_numpy()):
        raise ValueError(f"Reference Selection targets do not match: {model_name}")
    return aligned["prediction"].to_numpy(dtype=float)

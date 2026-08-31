"""Load Train and Selection while deliberately withholding final-Test frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_project_configs
from .data import DataContract, load_application_data, modeling_columns
from .features import prepare_fixed_features
from .split import SplitIndices, create_stratified_split


@dataclass
class DevelopmentData:
    """Development-only data bundle; no Test features or labels are exposed."""

    project_root: Path
    configs: dict[str, dict[str, Any]]
    overview: pd.Series
    data_sha256: str
    split: SplitIndices
    split_sha256: str
    feature_columns: list[str]
    X_train: pd.DataFrame
    y_train: pd.Series
    id_train: pd.Series
    X_selection: pd.DataFrame
    y_selection: pd.Series
    id_selection: pd.Series

    @property
    def split_overview(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample": ["Train", "Selection", "Test (sealed)"],
                "n": [len(self.split.train), len(self.split.selection), len(self.split.test)],
                "bad_rate": [self.y_train.mean(), self.y_selection.mean(), np.nan],
            }
        )


def load_development_data(project_root: str | Path) -> DevelopmentData:
    """Read full data for mechanical checks, then materialize only Train and Selection."""
    root = Path(project_root).expanduser().absolute()
    configs = load_project_configs(root)
    experiment = configs["experiment"]
    feature_policy = configs["feature_policy"]
    data_config = experiment["data"]
    split_config = experiment["split"]
    contract = DataContract(
        target=data_config["target"],
        identifier=data_config["id"],
        labels=tuple(data_config["expected_labels"]),
        require_unique_id=bool(data_config["require_unique_id"]),
    )
    frame, overview, data_sha256 = load_application_data(
        root / data_config["path"],
        contract=contract,
        expected_sha256=data_config.get("expected_sha256"),
    )
    target = frame[contract.target].astype("int8")
    split = create_stratified_split(
        target,
        train_share=float(split_config["train"]),
        selection_share=float(split_config["selection"]),
        test_share=float(split_config["test"]),
        random_state=int(split_config["random_state"]),
    )
    if split.fingerprint != split_config["expected_sha256"]:
        raise ValueError(
            f"Split fingerprint mismatch: expected {split_config['expected_sha256']}, observed {split.fingerprint}"
        )

    source_features = modeling_columns(frame, contract, exclude=feature_policy.get("exclude_features", []))
    fixed = feature_policy["fixed_feature_engineering"]
    engineered = prepare_fixed_features(
        frame[source_features],
        day_to_year=fixed["day_to_year"],
        ratios=fixed["ratios"],
        drop_original_days=bool(fixed["drop_original_day_features"]),
    )
    identifiers = frame[contract.identifier]
    return DevelopmentData(
        project_root=root,
        configs=configs,
        overview=overview,
        data_sha256=data_sha256,
        split=split,
        split_sha256=split.fingerprint,
        feature_columns=engineered.columns.tolist(),
        X_train=engineered.iloc[split.train].reset_index(drop=True),
        y_train=target.iloc[split.train].reset_index(drop=True),
        id_train=identifiers.iloc[split.train].reset_index(drop=True),
        X_selection=engineered.iloc[split.selection].reset_index(drop=True),
        y_selection=target.iloc[split.selection].reset_index(drop=True),
        id_selection=identifiers.iloc[split.selection].reset_index(drop=True),
    )

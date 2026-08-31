"""Export locked WOE Test predictions from an environment compatible with its joblib artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from credit_risk.config import load_project_configs
from credit_risk.data import DataContract, load_application_data, modeling_columns
from credit_risk.features import prepare_fixed_features
from credit_risk.models.woe_logistic import transform_woe_frame
from credit_risk.split import create_stratified_split
from credit_risk.test_policy import assert_test_access_allowed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    root = args.project_root.expanduser().absolute()
    configs = load_project_configs(root)
    experiment = configs["experiment"]
    policy = experiment["test_policy"]
    registry = assert_test_access_allowed(
        root / policy["registry_path"],
        policy["required_models"],
        expected_data_sha256=experiment["data"]["expected_sha256"],
        expected_split_sha256=experiment["split"]["expected_sha256"],
    )
    data_config = experiment["data"]
    contract = DataContract(
        target=data_config["target"],
        identifier=data_config["id"],
        labels=tuple(data_config["expected_labels"]),
        require_unique_id=bool(data_config["require_unique_id"]),
    )
    frame, _, _ = load_application_data(
        root / data_config["path"], contract=contract, expected_sha256=data_config["expected_sha256"]
    )
    split_config = experiment["split"]
    split = create_stratified_split(
        frame[contract.target],
        train_share=float(split_config["train"]),
        selection_share=float(split_config["selection"]),
        test_share=float(split_config["test"]),
        random_state=int(split_config["random_state"]),
    )
    source_features = modeling_columns(
        frame, contract, exclude=configs["feature_policy"].get("exclude_features", [])
    )
    fixed = configs["feature_policy"]["fixed_feature_engineering"]
    test_features = prepare_fixed_features(
        frame[source_features].iloc[split.test].reset_index(drop=True),
        day_to_year=fixed["day_to_year"],
        ratios=fixed["ratios"],
        drop_original_days=bool(fixed["drop_original_day_features"]),
    )
    woe_dir = Path(registry["models"]["woe_logistic"]["config"]["artifact_directory"])
    bundle = joblib.load(woe_dir / "woe_model_bundle.joblib")
    transformed = transform_woe_frame(test_features, bundle["specs"], bundle["features"])
    output = pd.DataFrame(
        {
            "SK_ID_CURR": frame[contract.identifier].iloc[split.test].to_numpy(),
            "TARGET": frame[contract.target].iloc[split.test].to_numpy(),
            "prediction": bundle["model"].predict_proba(transformed)[:, 1],
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    print(json.dumps({"rows": len(output), "output": str(args.output_path)}))


if __name__ == "__main__":
    main()

"""One-time final-Test scoring for already locked model families."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .artifacts import create_run_directory, write_frame, write_json
from .config import load_project_configs
from .data import DataContract, load_application_data, modeling_columns
from .features import prepare_fixed_features
from .metrics import evaluate_predictions, paired_bootstrap_vs_reference, risk_decile_table
from .models.woe_logistic import transform_woe_frame
from .profiling import MISSING_LABEL, UNSEEN_LABEL, category_text, is_categorical
from .split import create_stratified_split
from .test_policy import assert_test_access_allowed


@dataclass
class FinalTestData:
    project_root: Path
    configs: dict
    X_test: pd.DataFrame
    y_test: pd.Series
    id_test: pd.Series
    data_sha256: str
    split_sha256: str


@dataclass
class FinalEvaluationResult:
    run_directory: Path
    metrics: pd.DataFrame
    uplift_vs_logistic: pd.DataFrame
    deciles: pd.DataFrame
    predictions: pd.DataFrame


def load_final_test_data(project_root: str | Path) -> tuple[FinalTestData, dict]:
    """Open Test only after checking every locked model and its data fingerprints."""
    root = Path(project_root).expanduser().absolute()
    configs = load_project_configs(root)
    experiment = configs["experiment"]
    policy = experiment["test_policy"]
    registry = assert_test_access_allowed(
        root / policy["registry_path"],
        policy["required_models"],
        expected_data_sha256=experiment["data"].get("expected_sha256"),
        expected_split_sha256=experiment["split"].get("expected_sha256"),
    )
    data_config = experiment["data"]
    contract = DataContract(
        target=data_config["target"],
        identifier=data_config["id"],
        labels=tuple(data_config["expected_labels"]),
        require_unique_id=bool(data_config["require_unique_id"]),
    )
    frame, _, data_sha256 = load_application_data(
        root / data_config["path"], contract=contract, expected_sha256=data_config.get("expected_sha256")
    )
    target = frame[contract.target].astype("int8")
    split_config = experiment["split"]
    split = create_stratified_split(
        target,
        train_share=float(split_config["train"]),
        selection_share=float(split_config["selection"]),
        test_share=float(split_config["test"]),
        random_state=int(split_config["random_state"]),
    )
    if split.fingerprint != split_config["expected_sha256"]:
        raise RuntimeError("Final Test split fingerprint does not match the locked experiment")
    source_features = modeling_columns(
        frame, contract, exclude=configs["feature_policy"].get("exclude_features", [])
    )
    fixed = configs["feature_policy"]["fixed_feature_engineering"]
    engineered = prepare_fixed_features(
        frame[source_features],
        day_to_year=fixed["day_to_year"],
        ratios=fixed["ratios"],
        drop_original_days=bool(fixed["drop_original_day_features"]),
    )
    return (
        FinalTestData(
            project_root=root,
            configs=configs,
            X_test=engineered.iloc[split.test].reset_index(drop=True),
            y_test=target.iloc[split.test].reset_index(drop=True),
            id_test=frame[contract.identifier].iloc[split.test].reset_index(drop=True),
            data_sha256=data_sha256,
            split_sha256=split.fingerprint,
        ),
        registry,
    )


def _artifact_directory(registry: dict, model_name: str) -> Path:
    entry = registry["models"].get(model_name, {})
    if entry.get("locked") is not True:
        raise RuntimeError(f"Model is not locked: {model_name}")
    return Path(entry["config"]["artifact_directory"])


def _catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result:
        if is_categorical(result[column]):
            result[column] = result[column].where(result[column].notna(), MISSING_LABEL).astype(str)
    return result


def _lightgbm_frame(frame: pd.DataFrame, category_levels: dict[str, list[str]]) -> pd.DataFrame:
    result = frame.copy()
    for column, levels in category_levels.items():
        values = category_text(result[column])
        result[column] = pd.Categorical(
            values.where(values.isin(set(levels)), UNSEEN_LABEL), categories=levels
        )
    return result


def _xgboost_matrix(frame: pd.DataFrame, bundle: dict) -> np.ndarray:
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]
    numeric_values = frame[numeric].to_numpy(dtype=np.float32, copy=True)
    encoder = bundle["onehot_encoder"]
    if encoder is None:
        return np.ascontiguousarray(numeric_values)
    category_frame = pd.DataFrame({column: category_text(frame[column]) for column in categorical})
    encoded = encoder.transform(category_frame).astype(np.float32, copy=False)
    return np.ascontiguousarray(np.column_stack([numeric_values, encoded]), dtype=np.float32)


def _locked_predictions(
    data: FinalTestData, registry: dict, woe_prediction_path: str | Path | None = None
) -> dict[str, np.ndarray]:
    frame = data.X_test
    predictions: dict[str, np.ndarray] = {}
    common_dir = _artifact_directory(registry, "common_logistic")
    common_preprocessor = joblib.load(common_dir / "preprocessor.joblib")
    common_model = joblib.load(common_dir / "model.joblib")
    predictions["common_logistic"] = common_model.predict_proba(common_preprocessor.transform(frame))[:, 1]

    if woe_prediction_path is not None:
        external = pd.read_csv(woe_prediction_path)
        aligned = pd.DataFrame({"SK_ID_CURR": data.id_test, "TARGET": data.y_test}).merge(
            external[["SK_ID_CURR", "TARGET", "prediction"]], on="SK_ID_CURR", how="left", validate="one_to_one"
        )
        if aligned["prediction"].isna().any() or not np.array_equal(aligned["TARGET_x"], aligned["TARGET_y"]):
            raise RuntimeError("External WOE Test predictions do not align with the locked Test sample")
        predictions["woe_logistic"] = aligned["prediction"].to_numpy(dtype=float)
    else:
        woe_dir = _artifact_directory(registry, "woe_logistic")
        woe_bundle = joblib.load(woe_dir / "woe_model_bundle.joblib")
        woe_frame = transform_woe_frame(frame, woe_bundle["specs"], woe_bundle["features"])
        predictions["woe_logistic"] = woe_bundle["model"].predict_proba(woe_frame)[:, 1]

    rf_dir = _artifact_directory(registry, "random_forest")
    rf_preprocessor = joblib.load(rf_dir / "preprocessor.joblib")
    rf_model = joblib.load(rf_dir / "model.joblib")
    predictions["random_forest"] = rf_model.predict_proba(rf_preprocessor.transform(frame))[:, 1]

    cat_dir = _artifact_directory(registry, "catboost")
    cat_model = joblib.load(cat_dir / "model.joblib")
    predictions["catboost"] = cat_model.predict_proba(_catboost_frame(frame))[:, 1]

    lgb_dir = _artifact_directory(registry, "lightgbm")
    lgb_config = registry["models"]["lightgbm"]["config"]
    lgb_model = joblib.load(lgb_dir / "model.joblib")
    predictions["lightgbm"] = lgb_model.predict_proba(
        _lightgbm_frame(frame, lgb_config["category_levels_from_train"])
    )[:, 1]

    xgb_dir = _artifact_directory(registry, "xgboost")
    xgb_model = joblib.load(xgb_dir / "model.joblib")
    xgb_bundle = joblib.load(xgb_dir / "preprocessor.joblib")
    xgb_iterations = int(registry["models"]["xgboost"]["config"]["parameters"]["n_estimators"])
    predictions["xgboost"] = xgb_model.predict_proba(
        _xgboost_matrix(frame, xgb_bundle), iteration_range=(0, xgb_iterations)
    )[:, 1]

    mlp_dir = _artifact_directory(registry, "mlp")
    mlp_preprocessor = joblib.load(mlp_dir / "preprocessor.joblib")
    mlp_model = joblib.load(mlp_dir / "model.joblib")
    predictions["mlp"] = mlp_model.predict_proba(mlp_preprocessor.transform(frame))[:, 1]

    for name, prediction in predictions.items():
        values = np.asarray(prediction, dtype=float)
        if len(values) != len(data.y_test) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise RuntimeError(f"Invalid final-Test predictions from {name}")
        predictions[name] = values
    return predictions


def run_final_test_evaluation(
    project_root: str | Path,
    output_directory: str | Path | None = None,
    woe_prediction_path: str | Path | None = None,
) -> FinalEvaluationResult:
    """Score every locked model once on Test after the suite lock has closed."""
    data, registry = load_final_test_data(project_root)
    run_directory = Path(output_directory) if output_directory else create_run_directory(
        data.project_root, "final_test"
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    predictions = _locked_predictions(data, registry, woe_prediction_path=woe_prediction_path)
    metrics = evaluate_predictions(
        {name: prediction for name, prediction in predictions.items()},
        {name: data.y_test for name in predictions},
    ).reset_index(names="model")
    experiment = data.configs["experiment"]
    uplift = paired_bootstrap_vs_reference(
        data.y_test,
        predictions,
        reference_model="common_logistic",
        n_bootstrap=int(experiment["evaluation"]["bootstrap_replicates"]),
        random_state=int(experiment["split"]["random_state"]),
        confidence_level=float(experiment["evaluation"]["bootstrap_confidence_level"]),
    )
    deciles = pd.concat(
        [
            risk_decile_table(data.y_test, prediction, groups=10).assign(model=name)
            for name, prediction in predictions.items()
        ],
        ignore_index=True,
    )
    prediction_frame = pd.DataFrame({"SK_ID_CURR": data.id_test, "TARGET": data.y_test, **predictions})
    manifest = {
        "stage": "final_test_evaluation",
        "test_accessed": True,
        "data_sha256": data.data_sha256,
        "split_sha256": data.split_sha256,
        "model_names": list(predictions),
        "test_n": len(data.y_test),
        "prior_test_exposure_disclosure": experiment["test_policy"].get("disclosure", ""),
        "registry_path": str(data.project_root / experiment["test_policy"]["registry_path"]),
    }
    write_frame(run_directory / "final_test_metrics.csv", metrics)
    write_frame(run_directory / "final_test_uplift_vs_common_logistic.csv", uplift)
    write_frame(run_directory / "final_test_deciles.csv", deciles)
    write_frame(run_directory / "final_test_predictions.csv", prediction_frame)
    write_json(run_directory / "final_test_manifest.json", manifest)
    return FinalEvaluationResult(run_directory, metrics, uplift, deciles, prediction_frame)

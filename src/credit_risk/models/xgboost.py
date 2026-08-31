"""Controlled common-input XGBoost workflow with Train-fitted dense One-Hot categories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ..artifacts import create_run_directory, write_frame, write_json
from ..development import DevelopmentData
from ..metrics import evaluate_predictions, paired_bootstrap_vs_reference, risk_decile_table
from ..model_selection import choose_best_candidate
from ..profiling import category_text, is_categorical
from ..test_policy import register_model_lock


@dataclass
class XGBoostResult:
    run_directory: Path
    search_results: pd.DataFrame
    metrics: pd.DataFrame
    uplift_vs_logistic: pd.DataFrame
    encoded_feature_importance: pd.DataFrame
    raw_feature_importance: pd.DataFrame
    shap_importance: pd.DataFrame
    selection_deciles: pd.DataFrame
    locked_config: dict
    train_prediction: np.ndarray
    selection_prediction: np.ndarray


def prepare_xgboost_matrices(
    X_train: pd.DataFrame,
    X_selection: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, OneHotEncoder | None, list[str], list[str], list[str], list[str]]:
    """Fit dense One-Hot categories on Train and preserve numeric NaN values for XGBoost."""
    if list(X_train.columns) != list(X_selection.columns):
        raise ValueError("Train and Selection XGBoost columns must match in the same order")
    categorical = [column for column in X_train if is_categorical(X_train[column])]
    numeric = [column for column in X_train if column not in categorical]
    for frame_name, frame in (("Train", X_train), ("Selection", X_selection)):
        numeric_values = frame[numeric].select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if np.isinf(numeric_values).any():
            raise ValueError(f"{frame_name} contains infinite numeric values")
    train_numeric = X_train[numeric].to_numpy(dtype=np.float32, copy=True)
    selection_numeric = X_selection[numeric].to_numpy(dtype=np.float32, copy=True)
    if not categorical:
        feature_names = list(numeric)
        return (
            np.ascontiguousarray(train_numeric),
            np.ascontiguousarray(selection_numeric),
            None,
            numeric,
            categorical,
            feature_names,
            feature_names,
        )
    train_categories = pd.DataFrame(
        {column: category_text(X_train[column]) for column in categorical}, index=X_train.index
    )
    selection_categories = pd.DataFrame(
        {column: category_text(X_selection[column]) for column in categorical}, index=X_selection.index
    )
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    train_encoded = encoder.fit_transform(train_categories)
    selection_encoded = encoder.transform(selection_categories)
    encoded_feature_names = numeric + encoder.get_feature_names_out(categorical).tolist()
    raw_feature_names = numeric + [
        feature for feature, levels in zip(categorical, encoder.categories_) for _ in range(len(levels))
    ]
    if len(encoded_feature_names) != len(raw_feature_names):
        raise RuntimeError("XGBoost encoded features could not be mapped to raw inputs")
    return (
        np.ascontiguousarray(np.column_stack([train_numeric, train_encoded]), dtype=np.float32),
        np.ascontiguousarray(np.column_stack([selection_numeric, selection_encoded]), dtype=np.float32),
        encoder,
        numeric,
        categorical,
        encoded_feature_names,
        raw_feature_names,
    )


def _candidate_profiles(settings: dict) -> list[dict]:
    required = {
        "name",
        "max_depth",
        "min_child_weight",
        "learning_rate",
        "reg_lambda",
        "reg_alpha",
        "subsample",
        "colsample_bytree",
        "gamma",
    }
    profiles = []
    for profile in settings.get("candidate_profiles", []):
        missing = sorted(required - set(profile))
        if missing:
            raise ValueError(f"XGBoost candidate is missing fields: {missing}")
        profiles.append(dict(profile))
    if not profiles:
        raise ValueError("XGBoost requires at least one candidate profile")
    names = [str(profile["name"]) for profile in profiles]
    if len(names) != len(set(names)):
        raise ValueError("XGBoost candidate profile names must be unique")
    return profiles


def _model_parameters(
    profile: dict, iterations: int, random_state: int, early_stopping_rounds: int | None
) -> dict:
    if iterations <= 0:
        raise ValueError("XGBoost iterations must be positive")
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": int(iterations),
        "max_depth": int(profile["max_depth"]),
        "min_child_weight": float(profile["min_child_weight"]),
        "learning_rate": float(profile["learning_rate"]),
        "reg_lambda": float(profile["reg_lambda"]),
        "reg_alpha": float(profile["reg_alpha"]),
        "subsample": float(profile["subsample"]),
        "colsample_bytree": float(profile["colsample_bytree"]),
        "gamma": float(profile["gamma"]),
        "tree_method": "hist",
        "device": "cpu",
        "grow_policy": "depthwise",
        "max_bin": 256,
        "missing": np.nan,
        "random_state": int(random_state),
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": early_stopping_rounds,
    }


def _capacity_index(params: dict, best_tree_count: int) -> float:
    return float(
        (2 ** int(params["max_depth"]))
        * int(best_tree_count)
        * float(params["colsample_bytree"])
        * float(params["subsample"])
        / float(params["min_child_weight"])
    )


def _fit_candidate(
    candidate_number: int,
    stage: str,
    profile: dict,
    max_iterations: int,
    early_stopping_rounds: int,
    random_state: int,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_selection: np.ndarray,
    y_selection: pd.Series,
) -> tuple[dict, XGBClassifier]:
    params = _model_parameters(profile, max_iterations, random_state, early_stopping_rounds)
    model = XGBClassifier(**params)
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train, eval_set=[(X_selection, y_selection)], verbose=False)
    if caught:
        messages = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught)
        raise RuntimeError(f"XGBoost fit emitted warnings: {messages}")
    fit_seconds = perf_counter() - started
    if model.best_iteration is None:
        best_tree_count = int(model.get_booster().num_boosted_rounds())
    else:
        best_tree_count = int(model.best_iteration + 1)
    attempted_iterations = int(model.get_booster().num_boosted_rounds())
    if best_tree_count <= 0 or attempted_iterations <= 0:
        raise RuntimeError(f"XGBoost candidate {profile['name']} produced no trees")
    train_prediction = np.asarray(
        model.predict_proba(X_train, iteration_range=(0, best_tree_count))[:, 1], dtype=float
    )
    selection_prediction = np.asarray(
        model.predict_proba(X_selection, iteration_range=(0, best_tree_count))[:, 1], dtype=float
    )
    metrics = evaluate_predictions(
        {"Train": train_prediction, "Selection": selection_prediction},
        {"Train": y_train, "Selection": y_selection},
    )
    train_metrics, selection_metrics = metrics.loc["Train"], metrics.loc["Selection"]
    row = {
        "candidate": int(candidate_number),
        "stage": stage,
        "profile": str(profile["name"]),
        "max_iterations": int(max_iterations),
        "attempted_iterations": attempted_iterations,
        "best_iteration": best_tree_count,
        "reached_iteration_boundary": bool(attempted_iterations >= max_iterations),
        **{
            key: params[key]
            for key in (
                "max_depth",
                "min_child_weight",
                "learning_rate",
                "reg_lambda",
                "reg_alpha",
                "subsample",
                "colsample_bytree",
                "gamma",
            )
        },
        "train_auc": float(train_metrics["AUC"]),
        "selection_auc": float(selection_metrics["AUC"]),
        "train_ar": float(train_metrics["AR"]),
        "selection_ar": float(selection_metrics["AR"]),
        "ar_gap": float(abs(train_metrics["AR"] - selection_metrics["AR"])),
        "train_ks": float(train_metrics["KS"]),
        "selection_ks": float(selection_metrics["KS"]),
        "selection_ap": float(selection_metrics["AP"]),
        "selection_logloss": float(selection_metrics["Logloss"]),
        "capacity": _capacity_index(params, best_tree_count),
        "fit_seconds": float(fit_seconds),
        "parameters_json": json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
        "eligible_for_lock": True,
    }
    print(
        f"XGBoost candidate {candidate_number} [{stage}/{profile['name']}] "
        f"Selection AR={row['selection_ar']:.6f}, gap={row['ar_gap']:.6f}, "
        f"trees={best_tree_count}/{attempted_iterations}, fit={fit_seconds:.1f}s"
    )
    return row, model


def _stratified_sample_positions(y: pd.Series, sample_size: int, random_state: int) -> np.ndarray:
    positions = np.arange(len(y))
    if sample_size <= 0:
        raise ValueError("XGBoost SHAP sample size must be positive")
    if sample_size >= len(positions):
        return positions
    selected, _ = train_test_split(
        positions,
        train_size=sample_size,
        stratify=np.asarray(y),
        random_state=random_state,
    )
    return np.sort(selected)


def run_xgboost(
    development: DevelopmentData,
    reference_selection_prediction: np.ndarray | None = None,
    output_directory: str | Path | None = None,
    register_lock: bool = True,
) -> XGBoostResult:
    """Tune dense-One-Hot XGBoost using Train fits and Selection choice only."""
    settings = development.configs["model_search_spaces"]["xgboost"]
    experiment = development.configs["experiment"]
    random_state = int(settings.get("random_state", experiment["split"]["random_state"]))
    tolerance = float(experiment["selection"]["ar_tolerance"])
    max_iterations = int(settings["max_iterations"])
    extension_max_iterations = int(settings.get("extension_max_iterations", max_iterations))
    early_stopping_rounds = int(settings["early_stopping_rounds"])
    max_boundary_extensions = int(settings.get("max_boundary_extensions", 0))
    if extension_max_iterations < max_iterations:
        raise ValueError("XGBoost extension_max_iterations cannot be below max_iterations")
    run_directory = Path(output_directory) if output_directory else create_run_directory(
        development.project_root, "xgboost"
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    X_train, X_selection, encoder, numeric, categorical, encoded_names, raw_names = prepare_xgboost_matrices(
        development.X_train, development.X_selection
    )
    profiles = _candidate_profiles(settings)
    profile_lookup = {str(profile["name"]): profile for profile in profiles}
    rows: list[dict] = []
    for profile in profiles:
        row, model = _fit_candidate(
            len(rows) + 1,
            "base_search",
            profile,
            max_iterations,
            early_stopping_rounds,
            random_state,
            X_train,
            development.y_train,
            X_selection,
            development.y_selection,
        )
        rows.append(row)
        del model

    base_results = pd.DataFrame(rows)
    best_base_ar = float(base_results["selection_ar"].max())
    boundary_shortlist = base_results.loc[
        base_results["reached_iteration_boundary"]
        & base_results["selection_ar"].ge(best_base_ar - tolerance)
    ].sort_values(["selection_ar", "ar_gap"], ascending=[False, True])
    preliminary_best = choose_best_candidate(base_results, tolerance=tolerance)
    extension_order: list[int] = []
    if bool(preliminary_best["reached_iteration_boundary"]):
        extension_order.append(int(preliminary_best["candidate"]))
    extension_order.extend(
        int(candidate)
        for candidate in boundary_shortlist["candidate"]
        if int(candidate) not in extension_order
    )
    extension_candidates = base_results.loc[
        base_results["candidate"].isin(extension_order[:max_boundary_extensions])
    ].copy()
    extension_candidates["extension_order"] = extension_candidates["candidate"].map(
        {candidate: order for order, candidate in enumerate(extension_order, start=1)}
    )
    extension_candidates = extension_candidates.sort_values("extension_order")
    for original in extension_candidates.itertuples(index=False):
        rows[int(original.candidate) - 1]["eligible_for_lock"] = False
        profile = profile_lookup[str(original.profile)]
        row, model = _fit_candidate(
            len(rows) + 1,
            "boundary_extension",
            profile,
            extension_max_iterations,
            early_stopping_rounds,
            random_state,
            X_train,
            development.y_train,
            X_selection,
            development.y_selection,
        )
        rows.append(row)
        del model

    eligible = pd.DataFrame(rows).loc[lambda frame: frame["eligible_for_lock"]].copy()
    best = choose_best_candidate(eligible, tolerance=tolerance)
    selected_profile = str(best["profile"])
    selected_tree_count = int(best["best_iteration"])
    selected_parameters = _model_parameters(
        profile_lookup[selected_profile], selected_tree_count, random_state, None
    )

    final_model = XGBClassifier(**selected_parameters)
    final_started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final_model.fit(X_train, development.y_train, verbose=False)
    if caught:
        messages = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught)
        raise RuntimeError(f"Locked XGBoost fit emitted warnings: {messages}")
    final_fit_seconds = perf_counter() - final_started
    train_prediction = np.asarray(
        final_model.predict_proba(X_train, iteration_range=(0, selected_tree_count))[:, 1], dtype=float
    )
    selection_prediction = np.asarray(
        final_model.predict_proba(X_selection, iteration_range=(0, selected_tree_count))[:, 1], dtype=float
    )
    metrics = evaluate_predictions(
        {"Train": train_prediction, "Selection": selection_prediction},
        {"Train": development.y_train, "Selection": development.y_selection},
    ).reset_index()
    selection_ar = float(metrics.loc[metrics["sample"].eq("Selection"), "AR"].iloc[0])
    if abs(selection_ar - float(best["selection_ar"])) > 1e-8:
        raise RuntimeError("Locked XGBoost did not reproduce its selected Selection AR")

    encoded_values = np.asarray(final_model.feature_importances_, dtype=float)
    if len(encoded_values) != len(encoded_names):
        raise RuntimeError("XGBoost encoded feature importance length does not match input width")
    encoded_feature_importance = pd.DataFrame(
        {"encoded_feature": encoded_names, "raw_feature": raw_names, "importance": encoded_values}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    raw_feature_importance = (
        encoded_feature_importance.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    shap_positions = _stratified_sample_positions(
        development.y_selection,
        int(settings.get("shap_sample_size", 3000)),
        random_state,
    )
    shap_matrix = X_selection[shap_positions]
    booster = final_model.get_booster()
    shap_contributions = np.asarray(
        booster.predict(
            xgb.DMatrix(shap_matrix, missing=np.nan),
            pred_contribs=True,
            iteration_range=(0, selected_tree_count),
        ),
        dtype=float,
    )
    raw_margin = np.asarray(
        booster.predict(
            xgb.DMatrix(shap_matrix, missing=np.nan),
            output_margin=True,
            iteration_range=(0, selected_tree_count),
        ),
        dtype=float,
    )
    if shap_contributions.shape != (len(shap_positions), len(encoded_names) + 1):
        raise RuntimeError("XGBoost SHAP contribution output has an unexpected shape")
    if not np.allclose(shap_contributions.sum(axis=1), raw_margin, atol=2e-5, rtol=1e-5):
        raise RuntimeError("XGBoost SHAP contributions do not reconstruct raw margins")
    raw_order = numeric + categorical
    raw_position = {feature: position for position, feature in enumerate(raw_order)}
    raw_shap = np.zeros((len(shap_positions), len(raw_order)), dtype=float)
    for encoded_position, raw_feature in enumerate(raw_names):
        raw_shap[:, raw_position[raw_feature]] += shap_contributions[:, encoded_position]
    shap_importance = pd.DataFrame(
        {"feature": raw_order, "mean_abs_shap": np.abs(raw_shap).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    selection_deciles = risk_decile_table(development.y_selection, selection_prediction, groups=10)

    uplift_vs_logistic = pd.DataFrame()
    if reference_selection_prediction is not None:
        reference = np.asarray(reference_selection_prediction, dtype=float)
        if len(reference) != len(selection_prediction):
            raise ValueError("Reference Selection prediction length does not match XGBoost")
        uplift_vs_logistic = paired_bootstrap_vs_reference(
            development.y_selection,
            {"common_logistic": reference, "xgboost": selection_prediction},
            reference_model="common_logistic",
            n_bootstrap=int(experiment["evaluation"]["bootstrap_replicates"]),
            random_state=random_state,
            confidence_level=float(experiment["evaluation"]["bootstrap_confidence_level"]),
        )

    train_ar = float(metrics.loc[metrics["sample"].eq("Train"), "AR"].iloc[0])
    locked_config = {
        "model_name": "xgboost",
        "experiment_track": "controlled",
        "data_sha256": development.data_sha256,
        "split_sha256": development.split_sha256,
        "feature_policy_version": development.configs["feature_policy"]["policy"]["version"],
        "training_sample": "Train only",
        "selection_role": "early stopping, candidate selection, and paired comparison only",
        "test_accessed": False,
        "parameters": selected_parameters,
        "selected_profile": selected_profile,
        "candidate_count": len(rows),
        "base_candidate_count": len(profiles),
        "boundary_extensions": int(len(extension_candidates)),
        "selected_candidate_reached_extension_boundary": bool(best["reached_iteration_boundary"]),
        "final_refit_on_train_only_at_fixed_iteration": True,
        "raw_feature_count": len(development.feature_columns),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "encoded_feature_count": len(encoded_names),
        "selection_AR": selection_ar,
        "train_selection_AR_gap": train_ar - selection_ar,
        "selection_top_decile_bad_capture": float(
            selection_deciles.iloc[0]["cumulative_bad_capture"]
        ),
        "selection_top_decile_lift": float(selection_deciles.iloc[0]["lift"]),
        "shap_sample_size": len(shap_positions),
        "final_fit_seconds": float(final_fit_seconds),
        "artifact_directory": str(run_directory),
    }
    if not uplift_vs_logistic.empty:
        ar_row = uplift_vs_logistic.loc[uplift_vs_logistic["metric"].eq("AR")].iloc[0]
        locked_config["selection_AR_uplift_vs_common_logistic"] = float(ar_row["difference"])
        locked_config["selection_AR_uplift_ci"] = [
            float(ar_row["lower"]),
            float(ar_row["upper"]),
        ]

    search_results = pd.DataFrame(rows).sort_values(
        ["selection_ar", "ar_gap"], ascending=[False, True]
    ).reset_index(drop=True)
    booster.save_model(str(run_directory / "model.json"))
    joblib.dump(final_model, run_directory / "model.joblib")
    joblib.dump(
        {
            "onehot_encoder": encoder,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "encoded_feature_names": encoded_names,
            "raw_feature_names": raw_names,
        },
        run_directory / "preprocessor.joblib",
    )
    write_frame(run_directory / "search_results.csv", search_results)
    write_frame(run_directory / "development_metrics.csv", metrics)
    write_frame(run_directory / "encoded_feature_importance.csv", encoded_feature_importance)
    write_frame(run_directory / "raw_feature_importance.csv", raw_feature_importance)
    write_frame(run_directory / "selection_shap_importance.csv", shap_importance)
    write_frame(run_directory / "selection_deciles.csv", selection_deciles)
    write_frame(run_directory / "selection_uplift_vs_common_logistic.csv", uplift_vs_logistic)
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
            "xgboost",
            locked_config,
            policy["required_models"],
        )
    return XGBoostResult(
        run_directory=run_directory,
        search_results=search_results,
        metrics=metrics,
        uplift_vs_logistic=uplift_vs_logistic,
        encoded_feature_importance=encoded_feature_importance,
        raw_feature_importance=raw_feature_importance,
        shap_importance=shap_importance,
        selection_deciles=selection_deciles,
        locked_config=locked_config,
        train_prediction=train_prediction,
        selection_prediction=selection_prediction,
    )

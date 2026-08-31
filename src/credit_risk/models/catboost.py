"""Controlled common-input CatBoost workflow with native categorical handling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
import warnings

from catboost import CatBoostClassifier, Pool
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..artifacts import create_run_directory, write_frame, write_json
from ..development import DevelopmentData
from ..metrics import evaluate_predictions, paired_bootstrap_vs_reference, risk_decile_table
from ..model_selection import choose_best_candidate
from ..profiling import MISSING_LABEL, is_categorical
from ..test_policy import register_model_lock


@dataclass
class CatBoostResult:
    run_directory: Path
    search_results: pd.DataFrame
    metrics: pd.DataFrame
    uplift_vs_logistic: pd.DataFrame
    feature_importance: pd.DataFrame
    shap_importance: pd.DataFrame
    selection_deciles: pd.DataFrame
    locked_config: dict
    train_prediction: np.ndarray
    selection_prediction: np.ndarray


def prepare_catboost_frames(
    X_train: pd.DataFrame,
    X_selection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Apply only CatBoost's required categorical string/missing representation."""
    if list(X_train.columns) != list(X_selection.columns):
        raise ValueError("Train and Selection CatBoost columns must match in the same order")
    train = X_train.copy()
    selection = X_selection.copy()
    categorical = [column for column in train if is_categorical(train[column])]
    numeric = [column for column in train if column not in categorical]
    for column in categorical:
        train[column] = train[column].where(train[column].notna(), MISSING_LABEL).astype(str)
        selection[column] = selection[column].where(
            selection[column].notna(), MISSING_LABEL
        ).astype(str)
    for frame_name, frame in (("Train", train), ("Selection", selection)):
        numeric_values = frame[numeric].select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if np.isinf(numeric_values).any():
            raise ValueError(f"{frame_name} contains infinite numeric values")
    return train, selection, numeric, categorical


def _candidate_profiles(settings: dict) -> list[dict]:
    required = {
        "name",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "subsample",
        "rsm",
        "random_strength",
    }
    profiles = []
    for profile in settings.get("candidate_profiles", []):
        missing = sorted(required - set(profile))
        if missing:
            raise ValueError(f"CatBoost candidate is missing fields: {missing}")
        profiles.append(dict(profile))
    if not profiles:
        raise ValueError("CatBoost requires at least one candidate profile")
    names = [str(profile["name"]) for profile in profiles]
    if len(names) != len(set(names)):
        raise ValueError("CatBoost candidate profile names must be unique")
    return profiles


def _model_parameters(profile: dict, iterations: int, random_state: int) -> dict:
    if iterations <= 0:
        raise ValueError("CatBoost iterations must be positive")
    return {
        "iterations": int(iterations),
        "depth": int(profile["depth"]),
        "learning_rate": float(profile["learning_rate"]),
        "l2_leaf_reg": float(profile["l2_leaf_reg"]),
        "subsample": float(profile["subsample"]),
        "rsm": float(profile["rsm"]),
        "random_strength": float(profile["random_strength"]),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "bootstrap_type": "Bernoulli",
        "random_seed": int(random_state),
        "thread_count": -1,
        "allow_writing_files": False,
        "verbose": False,
    }


def _capacity_index(params: dict, best_tree_count: int) -> float:
    """Approximate symmetric-tree capacity for deterministic tie-breaking."""
    return float(
        (2 ** int(params["depth"]))
        * int(best_tree_count)
        * float(params["rsm"])
        * float(params["subsample"])
        / float(params["l2_leaf_reg"])
    )


def _fit_candidate(
    candidate_number: int,
    stage: str,
    profile: dict,
    max_iterations: int,
    early_stopping_rounds: int,
    random_state: int,
    train_pool: Pool,
    selection_pool: Pool,
    y_train: pd.Series,
    y_selection: pd.Series,
) -> tuple[dict, CatBoostClassifier]:
    params = _model_parameters(profile, max_iterations, random_state)
    model = CatBoostClassifier(**params)
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(
            train_pool,
            eval_set=selection_pool,
            use_best_model=True,
            early_stopping_rounds=int(early_stopping_rounds),
            verbose=False,
        )
    if caught:
        messages = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught)
        raise RuntimeError(f"CatBoost fit emitted warnings: {messages}")
    fit_seconds = perf_counter() - started
    train_prediction = np.asarray(model.predict_proba(train_pool)[:, 1], dtype=float)
    selection_prediction = np.asarray(model.predict_proba(selection_pool)[:, 1], dtype=float)
    metrics = evaluate_predictions(
        {"Train": train_prediction, "Selection": selection_prediction},
        {"Train": y_train, "Selection": y_selection},
    )
    train_metrics, selection_metrics = metrics.loc["Train"], metrics.loc["Selection"]
    eval_history = model.get_evals_result().get("validation", {}).get("AUC", [])
    attempted_iterations = len(eval_history)
    best_tree_count = int(model.tree_count_)
    if best_tree_count <= 0:
        raise RuntimeError(f"CatBoost candidate {profile['name']} produced no trees")
    row = {
        "candidate": int(candidate_number),
        "stage": stage,
        "profile": str(profile["name"]),
        "max_iterations": int(max_iterations),
        "attempted_iterations": int(attempted_iterations),
        "best_iteration": best_tree_count,
        "reached_iteration_boundary": bool(attempted_iterations >= max_iterations),
        **{key: params[key] for key in (
            "depth",
            "learning_rate",
            "l2_leaf_reg",
            "subsample",
            "rsm",
            "random_strength",
        )},
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
        "parameters_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "eligible_for_lock": True,
    }
    print(
        f"CatBoost candidate {candidate_number} [{stage}/{profile['name']}] "
        f"Selection AR={row['selection_ar']:.6f}, gap={row['ar_gap']:.6f}, "
        f"trees={best_tree_count}/{attempted_iterations}, fit={fit_seconds:.1f}s"
    )
    return row, model


def _stratified_sample_positions(y: pd.Series, sample_size: int, random_state: int) -> np.ndarray:
    positions = np.arange(len(y))
    if sample_size <= 0:
        raise ValueError("CatBoost SHAP sample size must be positive")
    if sample_size >= len(positions):
        return positions
    selected, _ = train_test_split(
        positions,
        train_size=sample_size,
        stratify=np.asarray(y),
        random_state=random_state,
    )
    return np.sort(selected)


def run_catboost(
    development: DevelopmentData,
    reference_selection_prediction: np.ndarray | None = None,
    output_directory: str | Path | None = None,
    register_lock: bool = True,
) -> CatBoostResult:
    """Tune native-category CatBoost using Train fits and Selection choice only."""
    settings = development.configs["model_search_spaces"]["catboost"]
    experiment = development.configs["experiment"]
    random_state = int(settings.get("random_state", experiment["split"]["random_state"]))
    tolerance = float(experiment["selection"]["ar_tolerance"])
    max_iterations = int(settings["max_iterations"])
    extension_max_iterations = int(settings.get("extension_max_iterations", max_iterations))
    early_stopping_rounds = int(settings["early_stopping_rounds"])
    max_boundary_extensions = int(settings.get("max_boundary_extensions", 0))
    if extension_max_iterations < max_iterations:
        raise ValueError("CatBoost extension_max_iterations cannot be below max_iterations")
    run_directory = Path(output_directory) if output_directory else create_run_directory(
        development.project_root, "catboost"
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    X_train, X_selection, numeric, categorical = prepare_catboost_frames(
        development.X_train, development.X_selection
    )
    train_pool = Pool(X_train, development.y_train, cat_features=categorical)
    selection_pool = Pool(X_selection, development.y_selection, cat_features=categorical)
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
            train_pool,
            selection_pool,
            development.y_train,
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
            train_pool,
            selection_pool,
            development.y_train,
            development.y_selection,
        )
        rows.append(row)
        del model

    eligible = pd.DataFrame(rows).loc[lambda frame: frame["eligible_for_lock"]].copy()
    best = choose_best_candidate(eligible, tolerance=tolerance)
    selected_profile = str(best["profile"])
    selected_tree_count = int(best["best_iteration"])
    selected_parameters = _model_parameters(
        profile_lookup[selected_profile], selected_tree_count, random_state
    )

    final_model = CatBoostClassifier(**selected_parameters)
    final_started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final_model.fit(train_pool, verbose=False)
    if caught:
        messages = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught)
        raise RuntimeError(f"Locked CatBoost fit emitted warnings: {messages}")
    final_fit_seconds = perf_counter() - final_started
    train_prediction = np.asarray(final_model.predict_proba(train_pool)[:, 1], dtype=float)
    selection_prediction = np.asarray(final_model.predict_proba(selection_pool)[:, 1], dtype=float)
    metrics = evaluate_predictions(
        {"Train": train_prediction, "Selection": selection_prediction},
        {"Train": development.y_train, "Selection": development.y_selection},
    ).reset_index()
    selection_ar = float(metrics.loc[metrics["sample"].eq("Selection"), "AR"].iloc[0])
    if abs(selection_ar - float(best["selection_ar"])) > 1e-8:
        raise RuntimeError("Locked CatBoost did not reproduce its selected Selection AR")

    feature_values = np.asarray(final_model.get_feature_importance(), dtype=float)
    feature_importance = pd.DataFrame(
        {"feature": development.feature_columns, "importance": feature_values}
    )
    total_importance = float(feature_importance["importance"].sum())
    if total_importance <= 0:
        raise RuntimeError("Locked CatBoost produced zero feature importance")
    feature_importance["importance"] /= total_importance
    feature_importance = feature_importance.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    shap_positions = _stratified_sample_positions(
        development.y_selection,
        int(settings.get("shap_sample_size", 3000)),
        random_state,
    )
    shap_pool = Pool(
        X_selection.iloc[shap_positions],
        development.y_selection.iloc[shap_positions],
        cat_features=categorical,
    )
    shap_values = np.asarray(
        final_model.get_feature_importance(shap_pool, type="ShapValues"), dtype=float
    )[:, :-1]
    shap_importance = pd.DataFrame(
        {
            "feature": development.feature_columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    selection_deciles = risk_decile_table(development.y_selection, selection_prediction, groups=10)

    uplift_vs_logistic = pd.DataFrame()
    if reference_selection_prediction is not None:
        reference = np.asarray(reference_selection_prediction, dtype=float)
        if len(reference) != len(selection_prediction):
            raise ValueError("Reference Selection prediction length does not match CatBoost")
        uplift_vs_logistic = paired_bootstrap_vs_reference(
            development.y_selection,
            {"common_logistic": reference, "catboost": selection_prediction},
            reference_model="common_logistic",
            n_bootstrap=int(experiment["evaluation"]["bootstrap_replicates"]),
            random_state=random_state,
            confidence_level=float(experiment["evaluation"]["bootstrap_confidence_level"]),
        )

    train_ar = float(metrics.loc[metrics["sample"].eq("Train"), "AR"].iloc[0])
    locked_config = {
        "model_name": "catboost",
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
        "selected_candidate_reached_extension_boundary": bool(
            best["reached_iteration_boundary"]
        ),
        "final_refit_on_train_only_at_fixed_iteration": True,
        "raw_feature_count": len(development.feature_columns),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
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
    final_model.save_model(run_directory / "model.cbm")
    joblib.dump(final_model, run_directory / "model.joblib")
    write_frame(run_directory / "search_results.csv", search_results)
    write_frame(run_directory / "development_metrics.csv", metrics)
    write_frame(run_directory / "feature_importance.csv", feature_importance)
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
            "catboost",
            locked_config,
            policy["required_models"],
        )
    return CatBoostResult(
        run_directory=run_directory,
        search_results=search_results,
        metrics=metrics,
        uplift_vs_logistic=uplift_vs_logistic,
        feature_importance=feature_importance,
        shap_importance=shap_importance,
        selection_deciles=selection_deciles,
        locked_config=locked_config,
        train_prediction=train_prediction,
        selection_prediction=selection_prediction,
    )

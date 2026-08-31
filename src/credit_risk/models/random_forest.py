"""Controlled common-input Random Forest workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..artifacts import create_run_directory, write_frame, write_json
from ..development import DevelopmentData
from ..metrics import (
    evaluate_predictions,
    paired_bootstrap_vs_reference,
    risk_decile_table,
)
from ..model_selection import choose_best_candidate
from ..profiling import is_categorical
from ..test_policy import register_model_lock


@dataclass
class RandomForestResult:
    run_directory: Path
    search_results: pd.DataFrame
    metrics: pd.DataFrame
    uplift_vs_logistic: pd.DataFrame
    encoded_feature_importance: pd.DataFrame
    raw_feature_importance: pd.DataFrame
    selection_deciles: pd.DataFrame
    locked_config: dict
    train_prediction: np.ndarray
    selection_prediction: np.ndarray


def build_random_forest_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Create tree-appropriate, Train-fitted missing-value and category handling."""
    categorical = [column for column in X_train if is_categorical(X_train[column])]
    numeric = [column for column in X_train if column not in categorical]
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    transformer = ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    return transformer, numeric, categorical


def _structure_candidates(settings: dict) -> list[dict]:
    """Return the versioned, auditable structure profiles at a common screening tree count."""
    screening_trees = int(settings["screening_n_estimators"])
    if screening_trees <= 0:
        raise ValueError("Random Forest screening tree count must be positive")
    candidates = []
    for profile in settings["structure_profiles"]:
        params = {key: value for key, value in profile.items() if key != "name"}
        params["n_estimators"] = screening_trees
        params["profile"] = profile["name"]
        candidates.append(params)
    if not candidates:
        raise ValueError("Random Forest requires at least one structure profile")
    return candidates


def _capacity_index(params: dict, encoded_feature_count: int) -> float:
    depth = 64.0 if params["max_depth"] is None else float(params["max_depth"])
    if params["max_features"] == "sqrt":
        feature_share = np.sqrt(encoded_feature_count) / encoded_feature_count
    else:
        feature_share = float(params["max_features"])
    sample_share = 1.0 if params["max_samples"] is None else float(params["max_samples"])
    return float(
        params["n_estimators"]
        * depth
        * feature_share
        * sample_share
        / float(params["min_samples_leaf"])
    )


def _fit_without_warnings(model, features, target):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features, target)
    if caught:
        messages = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught)
        raise RuntimeError(f"Random Forest fit emitted warnings: {messages}")
    return model


def _parameters_from_result(row: pd.Series) -> dict:
    names = [
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "max_features",
        "max_samples",
        "bootstrap",
    ]
    params = {
        name: None
        if pd.isna(row[name])
        else row[name].item()
        if isinstance(row[name], np.generic)
        else row[name]
        for name in names
    }
    params["n_estimators"] = int(params["n_estimators"])
    if params["max_depth"] is not None:
        params["max_depth"] = int(params["max_depth"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])
    params["min_samples_split"] = int(params["min_samples_split"])
    return params


def _raw_feature_mapping(
    preprocessor: ColumnTransformer,
    numeric: list[str],
    categorical: list[str],
) -> list[str]:
    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    numeric_names = numeric_imputer.get_feature_names_out(numeric).tolist()
    numeric_raw = [
        name.removeprefix("missingindicator_") if name.startswith("missingindicator_") else name
        for name in numeric_names
    ]
    onehot = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    categorical_raw = [
        feature
        for feature, levels in zip(categorical, onehot.categories_)
        for _ in range(len(levels))
    ]
    return numeric_raw + categorical_raw


def run_random_forest(
    development: DevelopmentData,
    reference_selection_prediction: np.ndarray | None = None,
    output_directory: str | Path | None = None,
    register_lock: bool = True,
) -> RandomForestResult:
    """Tune Random Forest on common inputs using Train fits and Selection choice only."""
    settings = development.configs["model_search_spaces"]["random_forest"]
    experiment = development.configs["experiment"]
    random_state = int(settings.get("random_state", experiment["split"]["random_state"]))
    class_weight = settings.get("class_weight", "balanced_subsample")
    tolerance = float(experiment["selection"]["ar_tolerance"])
    run_directory = Path(output_directory) if output_directory else create_run_directory(
        development.project_root, "random_forest"
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    preprocessor, numeric, categorical = build_random_forest_preprocessor(development.X_train)
    transform_started = perf_counter()
    X_train = preprocessor.fit_transform(development.X_train).astype(np.float32)
    X_selection = preprocessor.transform(development.X_selection).astype(np.float32)
    transform_seconds = perf_counter() - transform_started
    feature_names = preprocessor.get_feature_names_out().tolist()
    raw_feature_names = _raw_feature_mapping(preprocessor, numeric, categorical)
    if len(feature_names) != len(raw_feature_names):
        raise RuntimeError("Encoded Random Forest features could not be mapped to raw inputs")

    structure_candidates = _structure_candidates(settings)
    rows: list[dict] = []
    full_models: dict[int, RandomForestClassifier] = {}

    screening_rows = min(int(settings["screening_sample_size"]), len(development.y_train))
    all_positions = np.arange(len(development.y_train))
    if screening_rows < len(all_positions):
        screen_positions, _ = train_test_split(
            all_positions,
            train_size=screening_rows,
            stratify=np.asarray(development.y_train),
            random_state=random_state,
        )
        screen_positions = np.sort(screen_positions)
    else:
        screen_positions = all_positions
    X_screen = X_train[screen_positions]
    y_screen = development.y_train.iloc[screen_positions].reset_index(drop=True)

    for candidate_number, candidate in enumerate(structure_candidates, start=1):
        profile = str(candidate["profile"])
        params = {key: value for key, value in candidate.items() if key != "profile"}
        model = RandomForestClassifier(
            **params,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
            oob_score=True,
        )
        started = perf_counter()
        _fit_without_warnings(model, X_screen, y_screen)
        oob_prediction = np.asarray(model.oob_decision_function_[:, 1], dtype=float)
        if not np.isfinite(oob_prediction).all():
            raise RuntimeError(f"Random Forest OOB predictions are incomplete for {profile}")
        oob_auc = float(roc_auc_score(y_screen, oob_prediction))
        elapsed = perf_counter() - started
        row = {
            "candidate": candidate_number,
            "stage": "train_oob_screen",
            "profile": profile,
            **params,
            "class_weight": class_weight,
            "screening_rows": len(y_screen),
            "oob_auc": oob_auc,
            "oob_ar": 2 * oob_auc - 1,
            "capacity": _capacity_index(params, len(feature_names)),
            "fit_seconds": elapsed,
            "parameters_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)
        print(
            f"RF candidate {candidate_number} [Train-OOB/{profile}] "
            f"OOB AR={row['oob_ar']:.6f}, fit={elapsed:.1f}s"
        )
        del model

    screening_results = pd.DataFrame(rows).sort_values(
        ["oob_ar", "capacity"], ascending=[False, True]
    )
    finalist_count = min(int(settings["screening_finalists"]), len(screening_results))
    finalist_profiles = screening_results.head(finalist_count)["profile"].tolist()

    def fit_full_candidate(
        candidate_number: int, stage: str, profile: str, params: dict
    ) -> tuple[dict, RandomForestClassifier]:
        model = RandomForestClassifier(
            **params,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
        started = perf_counter()
        _fit_without_warnings(model, X_train, development.y_train)
        train_prediction = model.predict_proba(X_train)[:, 1]
        selection_prediction = model.predict_proba(X_selection)[:, 1]
        metrics = evaluate_predictions(
            {"Train": train_prediction, "Selection": selection_prediction},
            {"Train": development.y_train, "Selection": development.y_selection},
        )
        train_metrics, selection_metrics = metrics.loc["Train"], metrics.loc["Selection"]
        row = {
            "candidate": candidate_number,
            "stage": stage,
            "profile": profile,
            **params,
            "class_weight": class_weight,
            "train_auc": train_metrics["AUC"],
            "selection_auc": selection_metrics["AUC"],
            "train_ar": train_metrics["AR"],
            "selection_ar": selection_metrics["AR"],
            "ar_gap": abs(train_metrics["AR"] - selection_metrics["AR"]),
            "train_ks": train_metrics["KS"],
            "selection_ks": selection_metrics["KS"],
            "selection_ap": selection_metrics["AP"],
            "selection_logloss": selection_metrics["Logloss"],
            "capacity": _capacity_index(params, len(feature_names)),
            "fit_seconds": perf_counter() - started,
            "parameters_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        }
        print(
            f"RF candidate {candidate_number} [{stage}/{profile}] "
            f"Selection AR={row['selection_ar']:.6f}, gap={row['ar_gap']:.6f}, "
            f"fit={row['fit_seconds']:.1f}s"
        )
        return row, model

    tree_counts = sorted({int(value) for value in settings["tree_count_refinement"]})
    if not tree_counts or tree_counts[0] <= 0:
        raise ValueError("Random Forest tree-count refinement must contain positive values")
    full_tree_count = tree_counts[0]
    full_rows = []
    profile_lookup = {candidate["profile"]: candidate for candidate in structure_candidates}
    for profile in finalist_profiles:
        candidate = profile_lookup[profile]
        params = {key: value for key, value in candidate.items() if key != "profile"}
        params["n_estimators"] = full_tree_count
        candidate_number = len(rows) + 1
        row, model = fit_full_candidate(candidate_number, "full_train_finalist", profile, params)
        rows.append(row)
        full_rows.append(row)
        full_models[candidate_number] = model

    structure_results = pd.DataFrame(full_rows)
    best_structure = choose_best_candidate(structure_results, tolerance=tolerance)
    selected_profile = str(best_structure["profile"])
    selected_structure_params = _parameters_from_result(best_structure)
    tree_rows = [dict(best_structure)]
    screening_trees = int(selected_structure_params["n_estimators"])
    for tree_count in tree_counts:
        if tree_count == screening_trees:
            continue
        params = {**selected_structure_params, "n_estimators": tree_count}
        candidate_number = len(rows) + 1
        row, model = fit_full_candidate(
            candidate_number, "tree_count_refinement", selected_profile, params
        )
        rows.append(row)
        tree_rows.append(row)
        full_models[candidate_number] = model

    tree_results = pd.DataFrame(tree_rows)
    best = choose_best_candidate(tree_results, tolerance=tolerance)
    search_results = pd.DataFrame(rows)
    search_results["eligible_for_lock"] = (
        search_results["profile"].eq(selected_profile)
        & ~search_results["stage"].eq("train_oob_screen")
    )
    search_results = search_results.sort_values(
        ["selection_ar", "ar_gap"], ascending=[False, True]
    ).reset_index(drop=True)
    best_params = _parameters_from_result(best)
    selected_candidate = int(best["candidate"])
    final_model = full_models[selected_candidate]
    final_fit_seconds = float(best["fit_seconds"])
    train_prediction = final_model.predict_proba(X_train)[:, 1]
    selection_prediction = final_model.predict_proba(X_selection)[:, 1]
    metrics = evaluate_predictions(
        {"Train": train_prediction, "Selection": selection_prediction},
        {"Train": development.y_train, "Selection": development.y_selection},
    ).reset_index()
    selection_ar = float(metrics.loc[metrics["sample"].eq("Selection"), "AR"].iloc[0])
    if abs(selection_ar - float(best["selection_ar"])) > 1e-12:
        raise RuntimeError("Locked Random Forest did not reproduce its selected Selection AR")

    encoded_feature_importance = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "raw_feature": raw_feature_names,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    raw_feature_importance = (
        encoded_feature_importance.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    selection_deciles = risk_decile_table(
        development.y_selection, selection_prediction, groups=10
    )

    uplift_vs_logistic = pd.DataFrame()
    if reference_selection_prediction is not None:
        reference = np.asarray(reference_selection_prediction, dtype=float)
        if len(reference) != len(selection_prediction):
            raise ValueError("Reference Selection prediction length does not match Random Forest")
        uplift_vs_logistic = paired_bootstrap_vs_reference(
            development.y_selection,
            {"common_logistic": reference, "random_forest": selection_prediction},
            reference_model="common_logistic",
            n_bootstrap=int(experiment["evaluation"]["bootstrap_replicates"]),
            random_state=random_state,
            confidence_level=float(experiment["evaluation"]["bootstrap_confidence_level"]),
        )

    selection_train_ar = float(metrics.loc[metrics["sample"].eq("Train"), "AR"].iloc[0])
    locked_config = {
        "model_name": "random_forest",
        "experiment_track": "controlled",
        "data_sha256": development.data_sha256,
        "split_sha256": development.split_sha256,
        "feature_policy_version": development.configs["feature_policy"]["policy"]["version"],
        "training_sample": "Train only",
        "selection_role": "candidate selection and paired comparison only",
        "test_accessed": False,
        "parameters": {**best_params, "class_weight": class_weight, "random_state": random_state},
        "selected_structure_profile": selected_profile,
        "candidate_count": len(rows),
        "screening_sample_size": screening_rows,
        "screening_finalists": finalist_profiles,
        "final_full_train_fit_reused_from_selection_candidate": True,
        "raw_feature_count": len(development.feature_columns),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "encoded_feature_count": len(feature_names),
        "selection_AR": selection_ar,
        "train_selection_AR_gap": selection_train_ar - selection_ar,
        "selection_top_decile_bad_capture": float(selection_deciles.iloc[0]["cumulative_bad_capture"]),
        "selection_top_decile_lift": float(selection_deciles.iloc[0]["lift"]),
        "transform_seconds": transform_seconds,
        "final_fit_seconds": final_fit_seconds,
        "artifact_directory": str(run_directory),
    }
    if not uplift_vs_logistic.empty:
        ar_row = uplift_vs_logistic.loc[uplift_vs_logistic["metric"].eq("AR")].iloc[0]
        locked_config["selection_AR_uplift_vs_common_logistic"] = float(ar_row["difference"])
        locked_config["selection_AR_uplift_ci"] = [float(ar_row["lower"]), float(ar_row["upper"])]

    joblib.dump(preprocessor, run_directory / "preprocessor.joblib")
    joblib.dump(final_model, run_directory / "model.joblib")
    write_frame(run_directory / "search_results.csv", search_results)
    write_frame(run_directory / "development_metrics.csv", metrics)
    write_frame(run_directory / "encoded_feature_importance.csv", encoded_feature_importance)
    write_frame(run_directory / "raw_feature_importance.csv", raw_feature_importance)
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
            "random_forest",
            locked_config,
            policy["required_models"],
        )
    return RandomForestResult(
        run_directory=run_directory,
        search_results=search_results,
        metrics=metrics,
        uplift_vs_logistic=uplift_vs_logistic,
        encoded_feature_importance=encoded_feature_importance,
        raw_feature_importance=raw_feature_importance,
        selection_deciles=selection_deciles,
        locked_config=locked_config,
        train_prediction=train_prediction,
        selection_prediction=selection_prediction,
    )

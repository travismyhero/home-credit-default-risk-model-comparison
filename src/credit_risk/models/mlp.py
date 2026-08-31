"""Controlled common-input MLP workflow with Selection-AUC early stopping."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..artifacts import create_run_directory, write_frame, write_json
from ..development import DevelopmentData
from ..metrics import evaluate_predictions, paired_bootstrap_vs_reference, risk_decile_table
from ..model_selection import choose_best_candidate
from ..profiling import is_categorical
from ..test_policy import register_model_lock


@dataclass
class MLPResult:
    run_directory: Path
    search_results: pd.DataFrame
    metrics: pd.DataFrame
    uplift_vs_logistic: pd.DataFrame
    permutation_importance: pd.DataFrame
    selection_deciles: pd.DataFrame
    locked_config: dict
    train_prediction: np.ndarray
    selection_prediction: np.ndarray


def build_mlp_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build Train-fitted imputation, scaling, indicators, and One-Hot inputs for the MLP."""
    categorical = [column for column in X_train if is_categorical(X_train[column])]
    numeric = [column for column in X_train if column not in categorical]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
            ),
        ]
    )
    transformer = ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    return transformer, numeric, categorical


def _candidate_profiles(settings: dict) -> list[dict]:
    required = {"name", "hidden_layer_sizes", "alpha", "learning_rate_init", "batch_size"}
    profiles = []
    for profile in settings.get("candidate_profiles", []):
        missing = sorted(required - set(profile))
        if missing:
            raise ValueError(f"MLP candidate is missing fields: {missing}")
        profiles.append(dict(profile))
    if not profiles:
        raise ValueError("MLP requires at least one candidate profile")
    names = [str(profile["name"]) for profile in profiles]
    if len(names) != len(set(names)):
        raise ValueError("MLP candidate profile names must be unique")
    return profiles


def _model_parameters(profile: dict, random_state: int, activation: str) -> dict:
    return {
        "hidden_layer_sizes": tuple(int(value) for value in profile["hidden_layer_sizes"]),
        "activation": activation,
        "solver": "adam",
        "alpha": float(profile["alpha"]),
        "batch_size": int(profile["batch_size"]),
        "learning_rate_init": float(profile["learning_rate_init"]),
        "max_iter": 1,
        "shuffle": True,
        "random_state": int(random_state),
        "early_stopping": False,
        "tol": 0.0,
    }


def _capacity_index(params: dict, best_epoch: int) -> float:
    hidden_units = int(np.prod(params["hidden_layer_sizes"]))
    return float(hidden_units * int(best_epoch) / float(params["alpha"]))


def _restore_weights(model: MLPClassifier, weights: tuple[list[np.ndarray], list[np.ndarray]]) -> None:
    model.coefs_ = [value.copy() for value in weights[0]]
    model.intercepts_ = [value.copy() for value in weights[1]]


def _fit_candidate(
    candidate_number: int,
    stage: str,
    profile: dict,
    max_epochs: int,
    early_stopping_rounds: int,
    random_state: int,
    activation: str,
    X_train,
    y_train: pd.Series,
    X_selection,
    y_selection: pd.Series,
) -> tuple[dict, MLPClassifier]:
    if max_epochs <= 0 or early_stopping_rounds <= 0:
        raise ValueError("MLP epoch cap and early stopping rounds must be positive")
    params = _model_parameters(profile, random_state, activation)
    model = MLPClassifier(**params)
    best_auc = -np.inf
    best_epoch = 0
    best_weights: tuple[list[np.ndarray], list[np.ndarray]] | None = None
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(1, max_epochs + 1):
            if epoch == 1:
                model.partial_fit(X_train, y_train, classes=np.array([0, 1], dtype=int))
            else:
                model.partial_fit(X_train, y_train)
            selection_prediction = np.asarray(model.predict_proba(X_selection)[:, 1], dtype=float)
            selection_auc = float(roc_auc_score(y_selection, selection_prediction))
            if selection_auc > best_auc + 1e-12:
                best_auc = selection_auc
                best_epoch = epoch
                best_weights = (
                    [value.copy() for value in model.coefs_],
                    [value.copy() for value in model.intercepts_],
                )
            if epoch - best_epoch >= early_stopping_rounds:
                break
    if caught:
        messages = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught)
        raise RuntimeError(f"MLP fit emitted warnings: {messages}")
    if best_weights is None or best_epoch <= 0:
        raise RuntimeError(f"MLP candidate {profile['name']} did not produce a valid epoch")
    _restore_weights(model, best_weights)
    attempted_epochs = epoch
    fit_seconds = perf_counter() - started
    train_prediction = np.asarray(model.predict_proba(X_train)[:, 1], dtype=float)
    selection_prediction = np.asarray(model.predict_proba(X_selection)[:, 1], dtype=float)
    metrics = evaluate_predictions(
        {"Train": train_prediction, "Selection": selection_prediction},
        {"Train": y_train, "Selection": y_selection},
    )
    train_metrics, selection_metrics = metrics.loc["Train"], metrics.loc["Selection"]
    row = {
        "candidate": int(candidate_number),
        "stage": stage,
        "profile": str(profile["name"]),
        "max_epochs": int(max_epochs),
        "attempted_epochs": int(attempted_epochs),
        "best_iteration": int(best_epoch),
        "reached_iteration_boundary": bool(attempted_epochs >= max_epochs),
        "hidden_layer_sizes": json.dumps(list(params["hidden_layer_sizes"])),
        "alpha": params["alpha"],
        "learning_rate_init": params["learning_rate_init"],
        "batch_size": params["batch_size"],
        "train_auc": float(train_metrics["AUC"]),
        "selection_auc": float(selection_metrics["AUC"]),
        "train_ar": float(train_metrics["AR"]),
        "selection_ar": float(selection_metrics["AR"]),
        "ar_gap": float(abs(train_metrics["AR"] - selection_metrics["AR"])),
        "train_ks": float(train_metrics["KS"]),
        "selection_ks": float(selection_metrics["KS"]),
        "selection_ap": float(selection_metrics["AP"]),
        "selection_logloss": float(selection_metrics["Logloss"]),
        "capacity": _capacity_index(params, best_epoch),
        "fit_seconds": float(fit_seconds),
        "parameters_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "eligible_for_lock": True,
    }
    print(
        f"MLP candidate {candidate_number} [{stage}/{profile['name']}] "
        f"Selection AR={row['selection_ar']:.6f}, gap={row['ar_gap']:.6f}, "
        f"epoch={best_epoch}/{attempted_epochs}, fit={fit_seconds:.1f}s"
    )
    return row, model


def _fit_fixed_epochs(
    profile: dict,
    epochs: int,
    random_state: int,
    activation: str,
    X_train,
    y_train: pd.Series,
) -> MLPClassifier:
    params = _model_parameters(profile, random_state, activation)
    model = MLPClassifier(**params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(1, epochs + 1):
            if epoch == 1:
                model.partial_fit(X_train, y_train, classes=np.array([0, 1], dtype=int))
            else:
                model.partial_fit(X_train, y_train)
    if caught:
        messages = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught)
        raise RuntimeError(f"Locked MLP fit emitted warnings: {messages}")
    return model


def _stratified_sample_positions(y: pd.Series, sample_size: int, random_state: int) -> np.ndarray:
    positions = np.arange(len(y))
    if sample_size <= 0:
        raise ValueError("MLP permutation sample size must be positive")
    if sample_size >= len(positions):
        return positions
    selected, _ = train_test_split(
        positions,
        train_size=sample_size,
        stratify=np.asarray(y),
        random_state=random_state,
    )
    return np.sort(selected)


def _permutation_importance(
    model: MLPClassifier,
    preprocessor: ColumnTransformer,
    X_selection: pd.DataFrame,
    y_selection: pd.Series,
    sample_size: int,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    if repeats <= 0:
        raise ValueError("MLP permutation repeats must be positive")
    positions = _stratified_sample_positions(y_selection, sample_size, random_state)
    features = X_selection.iloc[positions].reset_index(drop=True)
    target = y_selection.iloc[positions].reset_index(drop=True)
    baseline_matrix = preprocessor.transform(features)
    baseline_auc = float(roc_auc_score(target, model.predict_proba(baseline_matrix)[:, 1]))
    rng = np.random.default_rng(random_state)
    rows = []
    for feature in features.columns:
        drops = []
        for _ in range(repeats):
            permuted = features.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            probability = model.predict_proba(preprocessor.transform(permuted))[:, 1]
            drops.append(baseline_auc - float(roc_auc_score(target, probability)))
        rows.append(
            {
                "feature": feature,
                "baseline_auc": baseline_auc,
                "mean_auc_decrease": float(np.mean(drops)),
                "std_auc_decrease": float(np.std(drops, ddof=1)) if repeats > 1 else 0.0,
                "permutation_repeats": int(repeats),
                "sample_size": len(features),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_auc_decrease", ascending=False).reset_index(drop=True)


def run_mlp(
    development: DevelopmentData,
    reference_selection_prediction: np.ndarray | None = None,
    output_directory: str | Path | None = None,
    register_lock: bool = True,
) -> MLPResult:
    """Tune sparse-One-Hot MLP using Train updates and Selection-AUC early stopping only."""
    settings = development.configs["model_search_spaces"]["mlp"]
    experiment = development.configs["experiment"]
    random_state = int(settings.get("random_state", experiment["split"]["random_state"]))
    tolerance = float(experiment["selection"]["ar_tolerance"])
    max_epochs = int(settings["max_epochs"])
    extension_max_epochs = int(settings.get("extension_max_epochs", max_epochs))
    early_stopping_rounds = int(settings["early_stopping_rounds"])
    max_boundary_extensions = int(settings.get("max_boundary_extensions", 0))
    activation = str(settings["activation"])
    if extension_max_epochs < max_epochs:
        raise ValueError("MLP extension_max_epochs cannot be below max_epochs")
    run_directory = Path(output_directory) if output_directory else create_run_directory(
        development.project_root, "mlp"
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    preprocessor, numeric, categorical = build_mlp_preprocessor(development.X_train)
    transform_started = perf_counter()
    X_train = preprocessor.fit_transform(development.X_train).astype(np.float32)
    X_selection = preprocessor.transform(development.X_selection).astype(np.float32)
    if not sparse.issparse(X_train) or not sparse.issparse(X_selection):
        raise RuntimeError("MLP preprocessing must preserve a sparse One-Hot representation")
    transform_seconds = perf_counter() - transform_started
    profiles = _candidate_profiles(settings)
    profile_lookup = {str(profile["name"]): profile for profile in profiles}
    rows: list[dict] = []
    for profile in profiles:
        row, model = _fit_candidate(
            len(rows) + 1,
            "base_search",
            profile,
            max_epochs,
            early_stopping_rounds,
            random_state,
            activation,
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
            extension_max_epochs,
            early_stopping_rounds,
            random_state,
            activation,
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
    selected_epochs = int(best["best_iteration"])
    final_started = perf_counter()
    final_model = _fit_fixed_epochs(
        profile_lookup[selected_profile],
        selected_epochs,
        random_state,
        activation,
        X_train,
        development.y_train,
    )
    final_fit_seconds = perf_counter() - final_started
    train_prediction = np.asarray(final_model.predict_proba(X_train)[:, 1], dtype=float)
    selection_prediction = np.asarray(final_model.predict_proba(X_selection)[:, 1], dtype=float)
    metrics = evaluate_predictions(
        {"Train": train_prediction, "Selection": selection_prediction},
        {"Train": development.y_train, "Selection": development.y_selection},
    ).reset_index()
    selection_ar = float(metrics.loc[metrics["sample"].eq("Selection"), "AR"].iloc[0])
    if abs(selection_ar - float(best["selection_ar"])) > 1e-8:
        raise RuntimeError("Locked MLP did not reproduce its selected Selection AR")

    permutation_importance = _permutation_importance(
        final_model,
        preprocessor,
        development.X_selection,
        development.y_selection,
        int(settings.get("permutation_sample_size", 3000)),
        int(settings.get("permutation_repeats", 3)),
        random_state,
    )
    selection_deciles = risk_decile_table(development.y_selection, selection_prediction, groups=10)

    uplift_vs_logistic = pd.DataFrame()
    if reference_selection_prediction is not None:
        reference = np.asarray(reference_selection_prediction, dtype=float)
        if len(reference) != len(selection_prediction):
            raise ValueError("Reference Selection prediction length does not match MLP")
        uplift_vs_logistic = paired_bootstrap_vs_reference(
            development.y_selection,
            {"common_logistic": reference, "mlp": selection_prediction},
            reference_model="common_logistic",
            n_bootstrap=int(experiment["evaluation"]["bootstrap_replicates"]),
            random_state=random_state,
            confidence_level=float(experiment["evaluation"]["bootstrap_confidence_level"]),
        )

    train_ar = float(metrics.loc[metrics["sample"].eq("Train"), "AR"].iloc[0])
    selected_parameters = _model_parameters(profile_lookup[selected_profile], random_state, activation)
    locked_config = {
        "model_name": "mlp",
        "experiment_track": "controlled",
        "data_sha256": development.data_sha256,
        "split_sha256": development.split_sha256,
        "feature_policy_version": development.configs["feature_policy"]["policy"]["version"],
        "training_sample": "Train only",
        "selection_role": "AUC early stopping, candidate selection, and paired comparison only",
        "test_accessed": False,
        "parameters": selected_parameters,
        "epochs": selected_epochs,
        "selected_profile": selected_profile,
        "candidate_count": len(rows),
        "base_candidate_count": len(profiles),
        "boundary_extensions": int(len(extension_candidates)),
        "selected_candidate_reached_extension_boundary": bool(best["reached_iteration_boundary"]),
        "final_refit_on_train_only_at_fixed_epoch": True,
        "raw_feature_count": len(development.feature_columns),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "encoded_feature_count": int(X_train.shape[1]),
        "selection_AR": selection_ar,
        "train_selection_AR_gap": train_ar - selection_ar,
        "selection_top_decile_bad_capture": float(
            selection_deciles.iloc[0]["cumulative_bad_capture"]
        ),
        "selection_top_decile_lift": float(selection_deciles.iloc[0]["lift"]),
        "permutation_sample_size": int(settings.get("permutation_sample_size", 3000)),
        "permutation_repeats": int(settings.get("permutation_repeats", 3)),
        "transform_seconds": float(transform_seconds),
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
    joblib.dump(preprocessor, run_directory / "preprocessor.joblib")
    joblib.dump(final_model, run_directory / "model.joblib")
    write_frame(run_directory / "search_results.csv", search_results)
    write_frame(run_directory / "development_metrics.csv", metrics)
    write_frame(run_directory / "selection_permutation_importance.csv", permutation_importance)
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
            "mlp",
            locked_config,
            policy["required_models"],
        )
    return MLPResult(
        run_directory=run_directory,
        search_results=search_results,
        metrics=metrics,
        uplift_vs_logistic=uplift_vs_logistic,
        permutation_importance=permutation_importance,
        selection_deciles=selection_deciles,
        locked_config=locked_config,
        train_prediction=train_prediction,
        selection_prediction=selection_prediction,
    )

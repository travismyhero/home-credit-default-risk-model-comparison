"""Controlled common-input Logistic Regression workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..artifacts import create_run_directory, write_frame, write_json
from ..development import DevelopmentData
from ..metrics import classification_metrics, evaluate_predictions
from ..model_selection import choose_best_candidate
from ..profiling import is_categorical
from ..test_policy import register_model_lock


@dataclass
class CommonLogisticResult:
    run_directory: Path
    search_results: pd.DataFrame
    metrics: pd.DataFrame
    locked_config: dict
    feature_names: list[str]
    train_prediction: np.ndarray
    selection_prediction: np.ndarray


def build_common_logistic_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
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
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
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


def run_common_logistic(
    development: DevelopmentData,
    output_directory: str | Path | None = None,
    register_lock: bool = True,
) -> CommonLogisticResult:
    """Fit preprocessing and all candidates on Train; use Selection only for choice."""
    search = development.configs["model_search_spaces"]["common_logistic"]
    experiment = development.configs["experiment"]
    candidates = [float(value) for value in search["candidates"]["C"]]
    tolerance = float(experiment["selection"]["ar_tolerance"])
    run_directory = Path(output_directory) if output_directory else create_run_directory(
        development.project_root, "common_logistic"
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    preprocessor, numeric, categorical = build_common_logistic_preprocessor(development.X_train)
    transform_start = perf_counter()
    X_train = preprocessor.fit_transform(development.X_train)
    X_selection = preprocessor.transform(development.X_selection)
    transform_seconds = perf_counter() - transform_start
    feature_names = preprocessor.get_feature_names_out().tolist()

    def fit_without_convergence_warning(model, features, target):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features, target)
        convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
        if convergence:
            raise RuntimeError(f"Common Logistic failed to converge: {convergence[-1].message}")
        return model

    rows = []
    for C in candidates:
        model = LogisticRegression(
            C=C,
            l1_ratio=0.0,
            solver="liblinear",
            max_iter=1000,
            tol=1e-6,
            class_weight=None,
            random_state=int(experiment["split"]["random_state"]),
        )
        started = perf_counter()
        fit_without_convergence_warning(model, X_train, development.y_train)
        train_prediction = model.predict_proba(X_train)[:, 1]
        selection_prediction = model.predict_proba(X_selection)[:, 1]
        train_metrics = classification_metrics(development.y_train, train_prediction)
        selection_metrics = classification_metrics(development.y_selection, selection_prediction)
        row = {
            "C": C,
            "penalty": "l2",
            "train_auc": train_metrics["AUC"],
            "selection_auc": selection_metrics["AUC"],
            "train_ar": train_metrics["AR"],
            "selection_ar": selection_metrics["AR"],
            "ar_gap": abs(train_metrics["AR"] - selection_metrics["AR"]),
            "train_ks": train_metrics["KS"],
            "selection_ks": selection_metrics["KS"],
            "selection_ap": selection_metrics["AP"],
            "selection_logloss": selection_metrics["Logloss"],
            "capacity": C,
            "best_iteration": int(np.max(model.n_iter_)),
            "fit_seconds": perf_counter() - started,
        }
        if row["best_iteration"] >= model.max_iter:
            raise RuntimeError(f"Common Logistic did not converge for C={C}")
        rows.append(row)

    search_results = pd.DataFrame(rows).sort_values(
        ["selection_ar", "ar_gap"], ascending=[False, True]
    ).reset_index(drop=True)
    best = choose_best_candidate(search_results, tolerance=tolerance)
    best_C = float(best["C"])

    final_model = LogisticRegression(
        C=best_C,
        l1_ratio=0.0,
        solver="liblinear",
        max_iter=1000,
        tol=1e-6,
        class_weight=None,
        random_state=int(experiment["split"]["random_state"]),
    )
    fit_without_convergence_warning(final_model, X_train, development.y_train)
    train_prediction = final_model.predict_proba(X_train)[:, 1]
    selection_prediction = final_model.predict_proba(X_selection)[:, 1]
    metrics = evaluate_predictions(
        {"Train": train_prediction, "Selection": selection_prediction},
        {"Train": development.y_train, "Selection": development.y_selection},
    ).reset_index()
    refit_selection_ar = float(metrics.loc[metrics["sample"] == "Selection", "AR"].iloc[0])
    if abs(refit_selection_ar - float(best["selection_ar"])) > 1e-10:
        raise RuntimeError("Locked common Logistic did not reproduce its selected Selection AR")

    locked_config = {
        "model_name": "common_logistic",
        "experiment_track": "controlled",
        "data_sha256": development.data_sha256,
        "split_sha256": development.split_sha256,
        "feature_policy_version": development.configs["feature_policy"]["policy"]["version"],
        "training_sample": "Train only",
        "selection_role": "hyperparameter selection only",
        "test_accessed": False,
        "parameters": {
            "C": best_C,
            "penalty": "l2",
            "solver": "liblinear",
            "max_iter": 1000,
            "tol": 1e-6,
            "onehot_drop": "first",
        },
        "raw_feature_count": len(development.feature_columns),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "encoded_feature_count": len(feature_names),
        "selection_AR": refit_selection_ar,
        "train_selection_AR_gap": float(
            metrics.loc[metrics["sample"] == "Train", "AR"].iloc[0] - refit_selection_ar
        ),
        "transform_seconds": transform_seconds,
        "artifact_directory": str(run_directory),
    }

    joblib.dump(preprocessor, run_directory / "preprocessor.joblib")
    joblib.dump(final_model, run_directory / "model.joblib")
    write_frame(run_directory / "search_results.csv", search_results)
    write_frame(run_directory / "development_metrics.csv", metrics)
    write_frame(run_directory / "feature_names.csv", pd.DataFrame({"feature": feature_names}))
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
            "common_logistic",
            locked_config,
            policy["required_models"],
        )
    return CommonLogisticResult(
        run_directory=run_directory,
        search_results=search_results,
        metrics=metrics,
        locked_config=locked_config,
        feature_names=feature_names,
        train_prediction=train_prediction,
        selection_prediction=selection_prediction,
    )

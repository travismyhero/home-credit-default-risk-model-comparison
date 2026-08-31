from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.development import DevelopmentData
from credit_risk.comparison import load_locked_selection_prediction
from credit_risk.models.common_logistic import run_common_logistic
from credit_risk.models.catboost import prepare_catboost_frames, run_catboost
from credit_risk.models.lightgbm import prepare_lightgbm_frames, run_lightgbm
from credit_risk.models.mlp import run_mlp
from credit_risk.models.random_forest import run_random_forest
from credit_risk.models.xgboost import prepare_xgboost_matrices, run_xgboost
from credit_risk.models.woe_logistic import run_woe_logistic
from credit_risk.profiling import run_development_profile
from credit_risk.split import SplitIndices


def synthetic_development_data(tmp_path: Path, seed: int = 3) -> DevelopmentData:
    rng = np.random.default_rng(seed)
    train_n, selection_n, test_n = 600, 200, 200

    def make_frame(rows):
        risk = rng.normal(size=rows)
        supporting = 0.4 * risk + rng.normal(scale=0.9, size=rows)
        category = np.where(risk > 0.7, "high", np.where(risk < -0.7, "low", "middle"))
        missing_signal = rng.normal(size=rows)
        missing_signal[rng.random(rows) < 0.15] = np.nan
        frame = pd.DataFrame(
            {
                "risk": risk,
                "supporting": supporting,
                "missing_signal": missing_signal,
                "category": category,
            }
        )
        probability = 1 / (1 + np.exp(-(-1.2 + 1.0 * risk + 0.35 * supporting)))
        target = pd.Series(rng.binomial(1, probability), dtype="int8")
        return frame, target

    X_train, y_train = make_frame(train_n)
    X_selection, y_selection = make_frame(selection_n)
    configs = {
        "experiment": {
            "split": {"random_state": 42},
            "selection": {"ar_tolerance": 0.005},
            "evaluation": {"bootstrap_replicates": 20, "bootstrap_confidence_level": 0.95},
            "test_policy": {
                "registry_path": "artifacts/locked_models.json",
                "required_models": ["common_logistic", "woe_logistic"],
            },
        },
        "feature_policy": {
            "policy": {"version": 1},
            "known_special_values": {},
            "quality_screen": {
                "review_only": {
                    "high_missing_threshold": 0.95,
                    "quasi_constant_threshold": 0.995,
                    "low_cardinality_numeric_max": 10,
                }
            },
            "categorical": {"rare_min_count": 5, "rare_min_share": 0.001},
        },
        "model_search_spaces": {
            "common_logistic": {"candidates": {"C": [0.1, 1.0]}},
            "woe_logistic": {
                "iv_min": 0.0,
                "correlation_threshold": 0.95,
                "single_ar_gap": 0.03,
                "vif_max": 100.0,
                "p_value_max": 1.0,
                "enforce_positive_coefficients": False,
            },
            "random_forest": {
                "class_weight": "balanced_subsample",
                "random_state": 42,
                "screening_sample_size": 300,
                "screening_n_estimators": 30,
                "screening_finalists": 2,
                "tree_count_refinement": [10],
                "structure_profiles": [
                    {
                        "name": "small",
                        "max_depth": 3,
                        "min_samples_leaf": 2,
                        "min_samples_split": 4,
                        "max_features": "sqrt",
                        "max_samples": None,
                        "bootstrap": True,
                    },
                    {
                        "name": "larger",
                        "max_depth": 5,
                        "min_samples_leaf": 2,
                        "min_samples_split": 4,
                        "max_features": "sqrt",
                        "max_samples": None,
                        "bootstrap": True,
                    },
                ],
            },
            "catboost": {
                "max_iterations": 30,
                "extension_max_iterations": 50,
                "early_stopping_rounds": 5,
                "max_boundary_extensions": 1,
                "random_state": 42,
                "shap_sample_size": 100,
                "candidate_profiles": [
                    {
                        "name": "shallow",
                        "depth": 2,
                        "learning_rate": 0.10,
                        "l2_leaf_reg": 5.0,
                        "subsample": 0.8,
                        "rsm": 0.8,
                        "random_strength": 0.5,
                    },
                    {
                        "name": "balanced",
                        "depth": 3,
                        "learning_rate": 0.10,
                        "l2_leaf_reg": 5.0,
                        "subsample": 0.8,
                        "rsm": 1.0,
                        "random_strength": 1.0,
                    },
                ],
            },
            "lightgbm": {
                "max_iterations": 40,
                "extension_max_iterations": 60,
                "early_stopping_rounds": 5,
                "max_boundary_extensions": 1,
                "random_state": 42,
                "shap_sample_size": 100,
                "candidate_profiles": [
                    {
                        "name": "shallow",
                        "num_leaves": 7,
                        "max_depth": 3,
                        "min_child_samples": 10,
                        "learning_rate": 0.10,
                        "reg_lambda": 5.0,
                        "reg_alpha": 0.0,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "min_split_gain": 0.0,
                        "cat_smooth": 10.0,
                    },
                    {
                        "name": "balanced",
                        "num_leaves": 15,
                        "max_depth": 4,
                        "min_child_samples": 10,
                        "learning_rate": 0.10,
                        "reg_lambda": 5.0,
                        "reg_alpha": 0.5,
                        "subsample": 0.8,
                        "colsample_bytree": 1.0,
                        "min_split_gain": 0.0,
                        "cat_smooth": 10.0,
                    },
                ],
            },
            "xgboost": {
                "max_iterations": 40,
                "extension_max_iterations": 60,
                "early_stopping_rounds": 5,
                "max_boundary_extensions": 1,
                "random_state": 42,
                "shap_sample_size": 100,
                "candidate_profiles": [
                    {
                        "name": "shallow",
                        "max_depth": 2,
                        "min_child_weight": 2.0,
                        "learning_rate": 0.10,
                        "reg_lambda": 5.0,
                        "reg_alpha": 0.0,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "gamma": 0.0,
                    },
                    {
                        "name": "balanced",
                        "max_depth": 3,
                        "min_child_weight": 2.0,
                        "learning_rate": 0.10,
                        "reg_lambda": 5.0,
                        "reg_alpha": 0.5,
                        "subsample": 0.8,
                        "colsample_bytree": 1.0,
                        "gamma": 0.0,
                    },
                ],
            },
            "mlp": {
                "max_epochs": 15,
                "extension_max_epochs": 20,
                "early_stopping_rounds": 4,
                "max_boundary_extensions": 1,
                "random_state": 42,
                "permutation_sample_size": 100,
                "permutation_repeats": 1,
                "activation": "relu",
                "candidate_profiles": [
                    {
                        "name": "small",
                        "hidden_layer_sizes": [8],
                        "alpha": 0.001,
                        "learning_rate_init": 0.01,
                        "batch_size": 128,
                    },
                    {
                        "name": "larger",
                        "hidden_layer_sizes": [16, 8],
                        "alpha": 0.001,
                        "learning_rate_init": 0.01,
                        "batch_size": 128,
                    },
                ],
            },
        },
    }
    split = SplitIndices(
        train=np.arange(train_n),
        selection=np.arange(train_n, train_n + selection_n),
        test=np.arange(train_n + selection_n, train_n + selection_n + test_n),
        random_state=42,
    )
    return DevelopmentData(
        project_root=tmp_path,
        configs=configs,
        overview=pd.Series({"rows": train_n + selection_n + test_n}),
        data_sha256="data",
        split=split,
        split_sha256="split",
        feature_columns=X_train.columns.tolist(),
        X_train=X_train,
        y_train=y_train,
        id_train=pd.Series(np.arange(train_n)),
        X_selection=X_selection,
        y_selection=y_selection,
        id_selection=pd.Series(np.arange(train_n, train_n + selection_n)),
    )


def test_development_bundle_has_no_test_frames():
    assert "X_test" not in DevelopmentData.__dataclass_fields__
    assert "y_test" not in DevelopmentData.__dataclass_fields__


def test_development_profile_and_common_logistic_smoke(tmp_path):
    development = synthetic_development_data(tmp_path)
    profile = run_development_profile(development, tmp_path / "profile")
    result = run_common_logistic(development, tmp_path / "common", register_lock=False)

    assert len(profile.candidate_features) == len(development.feature_columns)
    assert set(result.metrics["sample"]) == {"Train", "Selection"}
    assert result.locked_config["test_accessed"] is False
    assert (result.run_directory / "model.joblib").is_file()


def test_woe_logistic_smoke_uses_development_samples_only(tmp_path):
    development = synthetic_development_data(tmp_path)
    result = run_woe_logistic(development, tmp_path / "woe", register_lock=False)

    assert set(result.metrics["sample"]) == {"Train", "Selection"}
    assert result.final_features
    assert result.locked_config["test_accessed"] is False
    assert (result.run_directory / "woe_model_bundle.joblib").is_file()


def test_random_forest_smoke_compares_locked_logistic_without_test(tmp_path):
    development = synthetic_development_data(tmp_path)
    common = run_common_logistic(development, tmp_path / "common", register_lock=True)
    reference = load_locked_selection_prediction(development, "common_logistic")
    result = run_random_forest(
        development,
        reference_selection_prediction=reference,
        output_directory=tmp_path / "random_forest",
        register_lock=False,
    )

    assert np.allclose(reference, common.selection_prediction)
    assert set(result.metrics["sample"]) == {"Train", "Selection"}
    assert set(result.uplift_vs_logistic["metric"]) == {"AUC", "AR", "KS", "AP", "Logloss"}
    assert result.locked_config["test_accessed"] is False
    assert result.locked_config["candidate_count"] == 4
    assert np.isclose(result.raw_feature_importance["importance"].sum(), 1.0)
    assert (result.run_directory / "model.joblib").is_file()


def test_catboost_smoke_uses_native_categories_and_keeps_test_sealed(tmp_path):
    development = synthetic_development_data(tmp_path)
    common = run_common_logistic(development, tmp_path / "common", register_lock=True)
    reference = load_locked_selection_prediction(development, "common_logistic")
    development.X_train.loc[0, "category"] = None
    development.X_selection.loc[0, "category"] = "selection_only_level"
    prepared_train, prepared_selection, numeric, categorical = prepare_catboost_frames(
        development.X_train, development.X_selection
    )

    assert categorical == ["category"]
    assert set(numeric) == {"risk", "supporting", "missing_signal"}
    assert prepared_train.loc[0, "category"] == "__MISSING__"
    assert prepared_selection.loc[0, "category"] == "selection_only_level"

    result = run_catboost(
        development,
        reference_selection_prediction=reference,
        output_directory=tmp_path / "catboost",
        register_lock=False,
    )

    assert np.allclose(reference, common.selection_prediction)
    assert set(result.metrics["sample"]) == {"Train", "Selection"}
    assert set(result.uplift_vs_logistic["metric"]) == {"AUC", "AR", "KS", "AP", "Logloss"}
    assert result.locked_config["test_accessed"] is False
    assert result.locked_config["base_candidate_count"] == 2
    assert np.isclose(result.feature_importance["importance"].sum(), 1.0)
    assert len(result.shap_importance) == len(development.feature_columns)
    assert (result.run_directory / "model.cbm").is_file()
    assert (result.run_directory / "selection_shap_importance.csv").is_file()


def test_lightgbm_smoke_uses_train_categories_and_keeps_test_sealed(tmp_path):
    development = synthetic_development_data(tmp_path)
    common = run_common_logistic(development, tmp_path / "common", register_lock=True)
    reference = load_locked_selection_prediction(development, "common_logistic")
    development.X_train.loc[0, "category"] = None
    development.X_selection.loc[0, "category"] = "selection_only_level"
    prepared_train, prepared_selection, numeric, categorical, levels = prepare_lightgbm_frames(
        development.X_train, development.X_selection
    )

    assert categorical == ["category"]
    assert set(numeric) == {"risk", "supporting", "missing_signal"}
    assert prepared_train["category"].dtype.name == "category"
    assert prepared_train.loc[0, "category"] == "__MISSING__"
    assert prepared_selection.loc[0, "category"] == "__UNSEEN__"
    assert "__UNSEEN__" in levels["category"]

    result = run_lightgbm(
        development,
        reference_selection_prediction=reference,
        output_directory=tmp_path / "lightgbm",
        register_lock=False,
    )

    assert np.allclose(reference, common.selection_prediction)
    assert set(result.metrics["sample"]) == {"Train", "Selection"}
    assert set(result.uplift_vs_logistic["metric"]) == {"AUC", "AR", "KS", "AP", "Logloss"}
    assert result.locked_config["test_accessed"] is False
    assert result.locked_config["base_candidate_count"] == 2
    assert np.isclose(result.feature_importance["gain_pct"].sum(), 1.0)
    assert len(result.shap_importance) == len(development.feature_columns)
    assert (result.run_directory / "model.txt").is_file()
    assert (result.run_directory / "selection_shap_importance.csv").is_file()


def test_xgboost_smoke_uses_train_onehot_and_keeps_test_sealed(tmp_path):
    development = synthetic_development_data(tmp_path)
    common = run_common_logistic(development, tmp_path / "common", register_lock=True)
    reference = load_locked_selection_prediction(development, "common_logistic")
    development.X_train.loc[0, "category"] = None
    development.X_selection.loc[0, "category"] = "selection_only_level"
    X_train, X_selection, encoder, numeric, categorical, encoded_names, raw_names = (
        prepare_xgboost_matrices(development.X_train, development.X_selection)
    )

    assert encoder is not None
    assert categorical == ["category"]
    assert set(numeric) == {"risk", "supporting", "missing_signal"}
    assert X_train.dtype == np.float32
    assert X_selection.dtype == np.float32
    assert len(encoded_names) == len(raw_names) == X_train.shape[1]
    assert "selection_only_level" not in set(encoder.categories_[0])

    result = run_xgboost(
        development,
        reference_selection_prediction=reference,
        output_directory=tmp_path / "xgboost",
        register_lock=False,
    )

    assert np.allclose(reference, common.selection_prediction)
    assert set(result.metrics["sample"]) == {"Train", "Selection"}
    assert set(result.uplift_vs_logistic["metric"]) == {"AUC", "AR", "KS", "AP", "Logloss"}
    assert result.locked_config["test_accessed"] is False
    assert result.locked_config["base_candidate_count"] == 2
    assert np.isclose(result.raw_feature_importance["importance"].sum(), 1.0)
    assert len(result.shap_importance) == len(development.feature_columns)
    assert (result.run_directory / "model.json").is_file()
    assert (result.run_directory / "preprocessor.joblib").is_file()


def test_mlp_smoke_uses_selection_auc_early_stopping_and_keeps_test_sealed(tmp_path):
    development = synthetic_development_data(tmp_path)
    common = run_common_logistic(development, tmp_path / "common", register_lock=True)
    reference = load_locked_selection_prediction(development, "common_logistic")
    result = run_mlp(
        development,
        reference_selection_prediction=reference,
        output_directory=tmp_path / "mlp",
        register_lock=False,
    )

    assert np.allclose(reference, common.selection_prediction)
    assert set(result.metrics["sample"]) == {"Train", "Selection"}
    assert set(result.uplift_vs_logistic["metric"]) == {"AUC", "AR", "KS", "AP", "Logloss"}
    assert result.locked_config["test_accessed"] is False
    assert result.locked_config["base_candidate_count"] == 2
    assert len(result.permutation_importance) == len(development.feature_columns)
    assert result.permutation_importance["permutation_repeats"].eq(1).all()
    assert (result.run_directory / "model.joblib").is_file()
    assert (result.run_directory / "selection_permutation_importance.csv").is_file()

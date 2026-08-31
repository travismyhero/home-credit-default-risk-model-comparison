from pathlib import Path

from credit_risk.config import load_project_configs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_versioned_project_configs_are_valid():
    configs = load_project_configs(REPOSITORY_ROOT)

    experiment = configs["experiment"]
    required = experiment["test_policy"]["required_models"]
    assert required == [
        "common_logistic",
        "woe_logistic",
        "random_forest",
        "catboost",
        "lightgbm",
        "xgboost",
        "mlp",
    ]
    assert experiment["experiment_tracks"]["controlled"]["linear_reference"] == "common_logistic"


def test_every_required_model_has_a_search_policy():
    configs = load_project_configs(REPOSITORY_ROOT)
    required = set(configs["experiment"]["test_policy"]["required_models"])
    configured = set(configs["model_search_spaces"])
    assert required <= configured


def test_feature_policy_keeps_test_diagnostics_after_lock():
    configs = load_project_configs(REPOSITORY_ROOT)
    policy = configs["feature_policy"]["policy"]
    assert policy["fit_statistics_on"] == "Train"
    assert policy["validate_on"] == "Selection"
    assert policy["final_diagnostics_on"] == "Test_after_lock"

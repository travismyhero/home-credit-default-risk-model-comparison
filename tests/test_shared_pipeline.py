import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import DataContract, validate_application_frame
from credit_risk.features import prepare_fixed_features, safe_divide, validate_raw_schema
from credit_risk.metrics import (
    classification_metrics,
    paired_bootstrap_vs_reference,
    risk_decile_table,
)
from credit_risk.model_selection import choose_best_candidate
from credit_risk.split import SplitIndices, create_stratified_split
from credit_risk.test_policy import TestAccessError, assert_test_access_allowed, register_model_lock


def synthetic_target(rows=1000):
    return np.asarray(([0] * 8 + [1] * 2) * (rows // 10), dtype=int)


def test_split_is_deterministic_disjoint_complete_and_stratified(tmp_path):
    y = synthetic_target()
    first = create_stratified_split(y, random_state=42)
    second = create_stratified_split(y, random_state=42)

    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.selection, second.selection)
    assert np.array_equal(first.test, second.test)
    assert [len(first.train), len(first.selection), len(first.test)] == [700, 150, 150]
    assert np.isclose(y[first.train].mean(), y.mean())
    assert np.isclose(y[first.selection].mean(), y.mean())
    assert np.isclose(y[first.test].mean(), y.mean())
    assert first.fingerprint == second.fingerprint

    path = tmp_path / "split_indices.npz"
    first.save(path)
    loaded = SplitIndices.load(path, row_count=len(y))
    assert loaded.fingerprint == first.fingerprint


def test_split_validation_rejects_overlap():
    broken = SplitIndices(
        train=np.array([0, 1]),
        selection=np.array([1]),
        test=np.array([2]),
        random_state=42,
    )
    with pytest.raises(ValueError, match="overlap"):
        broken.validate(4)


def test_application_contract_rejects_duplicate_applicants():
    frame = pd.DataFrame({"SK_ID_CURR": [1, 1], "TARGET": [0, 1], "feature": [2.0, 3.0]})
    with pytest.raises(ValueError, match="must be unique"):
        validate_application_frame(frame, DataContract())


def test_fixed_features_preserve_special_information_and_safe_ratios():
    raw = pd.DataFrame(
        {
            "DAYS_BIRTH": [-3652.5, -7305.0],
            "DAYS_EMPLOYED": [365243, -3652.5],
            "DAYS_REGISTRATION": [-365.25, -730.5],
            "DAYS_ID_PUBLISH": [-1000.0, -2000.0],
            "DAYS_LAST_PHONE_CHANGE": [-100.0, -200.0],
            "AMT_CREDIT": [100.0, 200.0],
            "AMT_INCOME_TOTAL": [0.0, 100.0],
            "AMT_ANNUITY": [10.0, 20.0],
            "AMT_GOODS_PRICE": [80.0, 180.0],
            "CNT_FAM_MEMBERS": [1.0, 2.0],
            "CODE_GENDER": ["F", "M"],
        }
    )
    features = prepare_fixed_features(raw, exclude_features=["CODE_GENDER"])

    assert "CODE_GENDER" not in features
    assert "DAYS_EMPLOYED" not in features
    assert features["EMPLOYED_SPECIAL_FLAG"].tolist() == [1.0, 0.0]
    assert np.isnan(features.loc[0, "EMPLOYED_YEARS"])
    assert np.isclose(features.loc[1, "EMPLOYED_YEARS"], 10.0)
    assert np.isnan(features.loc[0, "CREDIT_INCOME_RATIO"])
    assert np.isclose(features.loc[1, "CREDIT_INCOME_RATIO"], 2.0)
    assert not np.isinf(features.select_dtypes(include="number").to_numpy()).any()


def test_safe_divide_does_not_turn_unknown_ratio_into_zero():
    result = safe_divide(pd.Series([4.0, 4.0]), pd.Series([2.0, 0.0]))
    assert result.iloc[0] == 2.0
    assert np.isnan(result.iloc[1])


def test_schema_validation_rejects_invalid_numeric_text():
    frame = pd.DataFrame({"amount": ["10", "not-a-number"], "category": ["A", "B"]})
    with pytest.raises(ValueError, match="not numeric"):
        validate_raw_schema(frame, ["amount", "category"], numeric_columns=["amount"])


def test_common_metrics_and_deciles_are_consistent():
    y = synthetic_target(100)
    prediction = np.linspace(0.01, 0.99, len(y))
    metrics = classification_metrics(y, prediction)
    assert np.isclose(metrics["AR"], 2 * metrics["AUC"] - 1)
    deciles = risk_decile_table(y, prediction)
    assert deciles["n"].sum() == len(y)
    assert deciles.iloc[0]["probability_min"] >= deciles.iloc[-1]["probability_max"]
    assert np.isclose(deciles.iloc[-1]["cumulative_bad_capture"], 1.0)


def test_paired_bootstrap_uses_declared_linear_reference():
    rng = np.random.default_rng(7)
    y = synthetic_target(200)
    linear = np.clip(0.15 + 0.45 * y + rng.normal(0, 0.20, len(y)), 0.001, 0.999)
    nonlinear = np.clip(0.12 + 0.55 * y + rng.normal(0, 0.16, len(y)), 0.001, 0.999)
    result = paired_bootstrap_vs_reference(
        y,
        {"common_logistic": linear, "xgboost": nonlinear},
        reference_model="common_logistic",
        n_bootstrap=20,
        random_state=9,
    )
    auc = result.loc[result.metric == "AUC"].iloc[0]
    assert auc["model"] == "xgboost"
    assert auc["reference_model"] == "common_logistic"
    assert auc["difference"] > 0


def test_candidate_selection_prefers_gap_and_lower_capacity_within_tolerance():
    candidates = pd.DataFrame(
        {
            "name": ["highest", "stable_simple", "stable_complex"],
            "selection_ar": [0.500, 0.497, 0.497],
            "ar_gap": [0.030, 0.010, 0.010],
            "capacity": [16, 8, 32],
            "best_iteration": [400, 300, 250],
        }
    )
    best = choose_best_candidate(candidates, tolerance=0.005)
    assert best["name"] == "stable_simple"


def test_final_test_remains_sealed_until_every_model_is_locked(tmp_path):
    registry = tmp_path / "locked_models.json"
    required = ["common_logistic", "xgboost"]
    fingerprint = {"data_sha256": "data", "split_sha256": "split"}

    register_model_lock(registry, "common_logistic", fingerprint, required)
    with pytest.raises(TestAccessError, match="xgboost"):
        assert_test_access_allowed(registry, required)

    register_model_lock(registry, "xgboost", fingerprint, required)
    result = assert_test_access_allowed(
        registry,
        required,
        expected_data_sha256="data",
        expected_split_sha256="split",
    )
    assert result["suite_locked"] is True
    assert json.loads(registry.read_text(encoding="utf-8"))["suite_locked"] is True


def test_test_gate_rejects_mismatched_fingerprint(tmp_path):
    registry = tmp_path / "locked_models.json"
    required = ["common_logistic"]
    register_model_lock(
        registry,
        "common_logistic",
        {"data_sha256": "old-data", "split_sha256": "split"},
        required,
    )
    with pytest.raises(TestAccessError, match="different data"):
        assert_test_access_allowed(registry, required, expected_data_sha256="new-data")

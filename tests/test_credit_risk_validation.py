import numpy as np
from sklearn.metrics import roc_auc_score

from credit_risk_validation import (
    bootstrap_metric_intervals,
    calibration_diagnostics,
    delong_auc_test,
    population_stability_index,
    prior_probability_correction,
)


def sample_predictions(seed=17):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(800, dtype=int), np.ones(200, dtype=int)]
    model_a = np.clip(0.15 + 0.45 * y + rng.normal(0, 0.22, len(y)), 0.001, 0.999)
    model_b = np.clip(0.18 + 0.38 * y + rng.normal(0, 0.24, len(y)), 0.001, 0.999)
    return y, model_a, model_b


def test_delong_auc_matches_sklearn():
    y, model_a, model_b = sample_predictions()
    result = delong_auc_test(y, model_a, model_b, "A", "B").iloc[0]

    assert np.isclose(result["AUCA"], roc_auc_score(y, model_a), atol=1e-12)
    assert np.isclose(result["AUCB"], roc_auc_score(y, model_b), atol=1e-12)
    assert 0 <= result["PValueTwoSided"] <= 1


def test_bootstrap_is_reproducible():
    y, model_a, model_b = sample_predictions()
    first = bootstrap_metric_intervals(
        y, {"A": model_a, "B": model_b}, n_bootstrap=40, random_state=9
    )
    second = bootstrap_metric_intervals(
        y, {"A": model_a, "B": model_b}, n_bootstrap=40, random_state=9
    )

    assert first[0].equals(second[0])
    assert first[1].equals(second[1])


def test_psi_is_zero_for_the_same_sample():
    _, model_a, _ = sample_predictions()
    assert population_stability_index(model_a, model_a) < 1e-12


def test_prior_correction_restores_requested_rate_at_neutral_odds():
    corrected = prior_probability_correction(
        np.full(100, 0.5), population_bad_rate=0.08, effective_training_bad_rate=0.5
    )
    assert np.allclose(corrected, 0.08)


def test_calibration_diagnostics_are_finite():
    y, model_a, _ = sample_predictions()
    diagnostics = calibration_diagnostics(y, model_a, "A", "Raw")

    for key in ("CalibrationInTheLarge", "CalibrationSlope", "Brier", "LogLoss"):
        assert np.isfinite(diagnostics[key])

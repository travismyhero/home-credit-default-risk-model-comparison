"""Configuration loading and contract checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a repository configuration is incomplete or inconsistent."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping with safe parsing."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ConfigError(f"Top-level YAML content must be a mapping: {config_path}")
    return content


def _require(mapping: dict[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required configuration key: {location}.{key}")
    return mapping[key]


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate the minimum experiment contract used by shared utilities."""
    for section in ("project", "data", "split", "selection", "evaluation", "test_policy"):
        _require(config, section, "experiment")

    data = config["data"]
    for key in ("path", "target", "id", "expected_labels"):
        _require(data, key, "experiment.data")
    if sorted(data["expected_labels"]) != [0, 1]:
        raise ConfigError("experiment.data.expected_labels must be [0, 1]")

    split = config["split"]
    shares = [float(_require(split, name, "experiment.split")) for name in ("train", "selection", "test")]
    if any(share <= 0 for share in shares) or abs(sum(shares) - 1.0) > 1e-12:
        raise ConfigError("Train/Selection/Test shares must be positive and sum to 1")
    _require(split, "random_state", "experiment.split")
    split_fingerprint = _require(split, "expected_sha256", "experiment.split")
    if not isinstance(split_fingerprint, str) or len(split_fingerprint) != 64:
        raise ConfigError("experiment.split.expected_sha256 must be a 64-character SHA256 digest")

    test_policy = config["test_policy"]
    required_models = _require(test_policy, "required_models", "experiment.test_policy")
    if not isinstance(required_models, list) or len(required_models) != len(set(required_models)):
        raise ConfigError("required_models must be a non-empty list without duplicates")


def validate_feature_policy(config: dict[str, Any]) -> None:
    """Validate the fixed feature-engineering inputs needed by the common layer."""
    fixed = _require(config, "fixed_feature_engineering", "feature_policy")
    for key in ("day_to_year", "ratios", "drop_original_day_features"):
        _require(fixed, key, "feature_policy.fixed_feature_engineering")
    for name, pair in fixed["ratios"].items():
        if not isinstance(pair, list) or len(pair) != 2:
            raise ConfigError(f"Ratio {name} must contain exactly [numerator, denominator]")


def load_project_configs(root: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate the three versioned project configuration files."""
    config_dir = Path(root) / "configs"
    result = {
        "experiment": load_yaml(config_dir / "experiment.yaml"),
        "feature_policy": load_yaml(config_dir / "feature_policy.yaml"),
        "model_search_spaces": load_yaml(config_dir / "model_search_spaces.yaml"),
    }
    validate_experiment_config(result["experiment"])
    validate_feature_policy(result["feature_policy"])
    return result

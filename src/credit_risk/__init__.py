"""Shared, leakage-aware utilities for the Home Credit model benchmark."""

from .config import ConfigError, load_project_configs, load_yaml
from .data import DataContract, file_sha256, load_application_data, validate_application_frame
from .development import DevelopmentData, load_development_data
from .features import prepare_fixed_features, safe_divide, validate_raw_schema
from .metrics import classification_metrics, evaluate_predictions, paired_bootstrap_vs_reference, risk_decile_table
from .model_selection import choose_best_candidate
from .paths import find_project_root
from .split import SplitIndices, create_stratified_split
from .test_policy import TestAccessError, assert_test_access_allowed, register_model_lock

__all__ = [
    "ConfigError",
    "DataContract",
    "DevelopmentData",
    "SplitIndices",
    "TestAccessError",
    "assert_test_access_allowed",
    "choose_best_candidate",
    "classification_metrics",
    "create_stratified_split",
    "evaluate_predictions",
    "file_sha256",
    "find_project_root",
    "load_application_data",
    "load_development_data",
    "load_project_configs",
    "load_yaml",
    "paired_bootstrap_vs_reference",
    "prepare_fixed_features",
    "register_model_lock",
    "risk_decile_table",
    "safe_divide",
    "validate_application_frame",
    "validate_raw_schema",
]

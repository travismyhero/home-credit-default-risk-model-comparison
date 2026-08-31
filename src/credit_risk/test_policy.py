"""Explicit final-Test access gate based on a versioned model lock registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TestAccessError(RuntimeError):
    """Raised when final-Test access is attempted before the suite is locked."""

    __test__ = False


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"suite_locked": False, "models": {}}
    with path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    if not isinstance(registry, dict) or not isinstance(registry.get("models", {}), dict):
        raise ValueError(f"Invalid model lock registry: {path}")
    return registry


def register_model_lock(
    registry_path: str | Path,
    model_name: str,
    locked_config: dict[str, Any],
    required_models: list[str],
) -> dict[str, Any]:
    """Register one model lock and close the suite only when every required model is locked."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = _load_registry(path)
    models = registry.setdefault("models", {})
    models[model_name] = {
        "locked": True,
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "config": locked_config,
    }
    registry["required_models"] = list(required_models)
    registry["suite_locked"] = all(models.get(name, {}).get("locked") is True for name in required_models)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2)
    return registry


def assert_test_access_allowed(
    registry_path: str | Path,
    required_models: list[str],
    expected_data_sha256: str | None = None,
    expected_split_sha256: str | None = None,
) -> dict[str, Any]:
    """Return the registry only after every required model and fingerprint is locked."""
    path = Path(registry_path)
    registry = _load_registry(path)
    models = registry.get("models", {})
    missing = [name for name in required_models if models.get(name, {}).get("locked") is not True]
    if missing or registry.get("suite_locked") is not True:
        raise TestAccessError(f"Final Test remains sealed; unlocked models: {missing}")

    for model_name in required_models:
        config = models[model_name].get("config", {})
        if expected_data_sha256 is not None and config.get("data_sha256") != expected_data_sha256:
            raise TestAccessError(f"{model_name} was locked with a different data fingerprint")
        if expected_split_sha256 is not None and config.get("split_sha256") != expected_split_sha256:
            raise TestAccessError(f"{model_name} was locked with a different split fingerprint")
    return registry

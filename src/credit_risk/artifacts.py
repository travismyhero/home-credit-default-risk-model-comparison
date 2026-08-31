"""Small, explicit artifact-writing helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def create_run_directory(project_root: str | Path, model_name: str) -> Path:
    """Create a non-overwriting UTC run directory under ignored artifacts/."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = Path(project_root) / "artifacts" / model_name / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def write_json(path: str | Path, content: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(content), handle, ensure_ascii=False, indent=2)


def write_frame(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")

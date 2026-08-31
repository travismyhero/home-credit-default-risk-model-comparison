"""Repository path discovery."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing the V2 experiment contract."""
    origin = Path(start or Path.cwd()).expanduser().absolute()
    candidates = [origin, *origin.parents]
    for candidate in candidates:
        if (candidate / "configs" / "experiment.yaml").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find project root from: {origin}")

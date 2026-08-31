"""Data loading, fingerprinting, and application-table validation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataContract:
    """Minimum schema contract for one-row-per-applicant binary modeling data."""

    target: str = "TARGET"
    identifier: str = "SK_ID_CURR"
    labels: tuple[int, int] = (0, 1)
    require_unique_id: bool = True


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a file without loading it into memory."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_application_frame(frame: pd.DataFrame, contract: DataContract = DataContract()) -> pd.Series:
    """Validate the full labeled application table without silently deleting rows."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Application data must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique")
    missing_columns = [name for name in (contract.identifier, contract.target) if name not in frame]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    target = frame[contract.target]
    if target.isna().any():
        raise ValueError(f"{contract.target} must not contain missing values")
    observed_labels = set(pd.unique(target))
    if observed_labels != set(contract.labels):
        raise ValueError(
            f"{contract.target} must contain exactly {sorted(contract.labels)}; observed {sorted(observed_labels)}"
        )

    identifier = frame[contract.identifier]
    if identifier.isna().any():
        raise ValueError(f"{contract.identifier} must not contain missing values")
    if contract.require_unique_id and not identifier.is_unique:
        raise ValueError(f"{contract.identifier} must be unique for a one-row-per-applicant split")

    numeric = frame.select_dtypes(include="number")
    infinity_count = int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum()) if not numeric.empty else 0
    return pd.Series(
        {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "features": int(frame.shape[1] - 2),
            "bad_n": int(target.sum()),
            "bad_rate": float(target.mean()),
            "unique_id": int(identifier.nunique()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "infinite_numeric_values": infinity_count,
        },
        name="dataset_overview",
    )


def load_application_data(
    path: str | Path,
    contract: DataContract = DataContract(),
    expected_sha256: str | None = None,
    low_memory: bool = False,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load the source CSV, check its digest, and enforce the application contract."""
    csv_path = Path(path).expanduser().absolute()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Application CSV does not exist: {csv_path}")
    digest = file_sha256(csv_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Data SHA256 mismatch: expected {expected_sha256}, observed {digest}")
    frame = pd.read_csv(csv_path, low_memory=low_memory)
    overview = validate_application_frame(frame, contract)
    overview.loc["sha256"] = digest
    overview.loc["source"] = str(csv_path)
    return frame, overview, digest


def modeling_columns(
    frame: pd.DataFrame,
    contract: DataContract = DataContract(),
    exclude: Iterable[str] = (),
) -> list[str]:
    """Return ordered model inputs after explicit ID, target, and policy exclusions."""
    blocked = {contract.identifier, contract.target, *exclude}
    return [column for column in frame.columns if column not in blocked]

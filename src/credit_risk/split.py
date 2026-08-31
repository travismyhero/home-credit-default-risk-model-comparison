"""Deterministic Train/Selection/Test split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    """Immutable row positions for the three experiment samples."""

    train: np.ndarray
    selection: np.ndarray
    test: np.ndarray
    random_state: int

    def validate(self, row_count: int) -> None:
        parts = [np.asarray(part, dtype=np.int64) for part in (self.train, self.selection, self.test)]
        if any(part.ndim != 1 for part in parts):
            raise ValueError("Split indices must be one-dimensional")
        joined = np.concatenate(parts)
        if len(joined) != row_count:
            raise ValueError(f"Split contains {len(joined)} rows; expected {row_count}")
        if len(np.unique(joined)) != row_count:
            raise ValueError("Train/Selection/Test indices overlap or contain duplicates")
        if row_count and (joined.min() != 0 or joined.max() != row_count - 1):
            raise ValueError("Split indices do not cover every source row exactly once")

    @property
    def fingerprint(self) -> str:
        digest = sha256()
        digest.update(f"random_state={self.random_state}".encode())
        for label, values in (("train", self.train), ("selection", self.selection), ("test", self.test)):
            digest.update(label.encode())
            digest.update(np.asarray(values, dtype="<i8").tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            train_idx=np.asarray(self.train, dtype=np.int64),
            selection_idx=np.asarray(self.selection, dtype=np.int64),
            test_idx=np.asarray(self.test, dtype=np.int64),
            random_state=np.asarray([self.random_state], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path, row_count: int | None = None) -> "SplitIndices":
        with np.load(Path(path), allow_pickle=False) as content:
            random_state = int(content["random_state"][0]) if "random_state" in content else 42
            result = cls(
                train=np.asarray(content["train_idx"], dtype=np.int64),
                selection=np.asarray(content["selection_idx"], dtype=np.int64),
                test=np.asarray(content["test_idx"], dtype=np.int64),
                random_state=random_state,
            )
        if row_count is not None:
            result.validate(row_count)
        return result


def create_stratified_split(
    target,
    train_share: float = 0.70,
    selection_share: float = 0.15,
    test_share: float = 0.15,
    random_state: int = 42,
) -> SplitIndices:
    """Create the same two-stage stratified split used by the model notebooks."""
    shares = np.asarray([train_share, selection_share, test_share], dtype=float)
    if np.any(shares <= 0) or not np.isclose(shares.sum(), 1.0):
        raise ValueError("Split shares must be positive and sum to 1")
    y = np.asarray(target)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Target must be one-dimensional and contain both 0 and 1")

    rows = np.arange(len(y), dtype=np.int64)
    train, remainder = train_test_split(
        rows,
        test_size=float(selection_share + test_share),
        stratify=y,
        random_state=random_state,
    )
    relative_test_share = float(test_share / (selection_share + test_share))
    selection, test = train_test_split(
        remainder,
        test_size=relative_test_share,
        stratify=y[remainder],
        random_state=random_state,
    )
    result = SplitIndices(train=train, selection=selection, test=test, random_state=random_state)
    result.validate(len(y))
    return result

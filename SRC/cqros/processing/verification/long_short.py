"""CQROS long/short dataset verification.

Purpose:
    Inspect long/short account frames and report structural findings without
    cleaning or mutating input data.

Responsibilities:
    - Validate required long/short columns
    - Count duplicate timestamps, nulls, NaNs, invalid timestamps, and
      negative account or ratio values
    - Allow zero values
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``LongShortVerifier``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = ["LongShortVerifier"]

_COL_TIMESTAMP: Final[str] = "timestamp"
_COL_LONG_ACCOUNT: Final[str] = "long_account"
_COL_SHORT_ACCOUNT: Final[str] = "short_account"
_COL_LONG_SHORT_RATIO: Final[str] = "long_short_ratio"

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    _COL_TIMESTAMP,
    _COL_LONG_ACCOUNT,
    _COL_SHORT_ACCOUNT,
    _COL_LONG_SHORT_RATIO,
)

_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    _COL_LONG_ACCOUNT,
    _COL_SHORT_ACCOUNT,
    _COL_LONG_SHORT_RATIO,
)

_WARN_DUPLICATES: Final[str] = "Duplicate timestamps detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_NUMERIC: Final[str] = "Invalid long/short numeric values."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by timestamp."


class LongShortVerifier(BaseVerifier):
    """Deterministic long/short verifier that reports findings only.

    Inspects structural quality of a long/short frame. Does not clean rows,
    fill gaps, sort timestamps, mutate values, access storage, or compute
    features. Zero account and ratio values are valid.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input long/short DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            ProcessingValidationError: If any required column is missing.
        """
        self._validate_required_columns(frame, _REQUIRED_COLUMNS)

        duplicate_timestamp_rows = self._count_duplicate_timestamp_rows(
            frame,
            _COL_TIMESTAMP,
        )
        null_rows = self._count_null_rows(frame, _REQUIRED_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _NUMERIC_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_TIMESTAMP,
        )
        invalid_numeric_rows = self._count_invalid_numeric_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_TIMESTAMP)

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            is_sorted=is_sorted,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and is_sorted
        )
        return VerificationReport(
            rows_checked=frame.height,
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            warnings=warnings,
            passed=passed,
        )

    def _count_invalid_numeric_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with negative account or ratio values.

        Zero values are valid. Each qualifying row is counted once.

        Args:
            frame: Input long/short DataFrame. Must not be mutated.

        Returns:
            Count of invalid numeric rows.
        """
        if frame.height == 0:
            return 0
        invalid_mask = (
            (pl.col(_COL_LONG_ACCOUNT) < 0)
            | (pl.col(_COL_SHORT_ACCOUNT) < 0)
            | (pl.col(_COL_LONG_SHORT_RATIO) < 0)
        )
        return int(frame.select(invalid_mask.sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    invalid_numeric_rows: int,
    is_sorted: bool,
) -> tuple[str, ...]:
    """Return deterministic warnings for non-zero counters and sort failures."""
    warnings: list[str] = []
    if duplicate_timestamp_rows > 0:
        warnings.append(_WARN_DUPLICATES)
    if null_rows > 0:
        warnings.append(_WARN_NULLS)
    if nan_rows > 0:
        warnings.append(_WARN_NANS)
    if invalid_timestamp_rows > 0:
        warnings.append(_WARN_TIMESTAMPS)
    if invalid_numeric_rows > 0:
        warnings.append(_WARN_NUMERIC)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)

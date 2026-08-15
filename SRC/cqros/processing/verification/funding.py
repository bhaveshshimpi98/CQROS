"""CQROS funding dataset verification.

Purpose:
    Inspect funding frames and report structural findings without cleaning or
    mutating input data.

Responsibilities:
    - Validate required funding columns (including nullable ``mark_price``)
    - Count duplicate timestamps, nulls in non-nullable required columns,
      NaNs, invalid timestamps, and invalid non-null ``mark_price`` values
    - Allow null ``mark_price`` and negative ``funding_rate`` values
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``FundingVerifier``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = ["FundingVerifier"]

_COL_FUNDING_TIME: Final[str] = "funding_time"
_COL_FUNDING_RATE: Final[str] = "funding_rate"
_COL_MARK_PRICE: Final[str] = "mark_price"

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    _COL_FUNDING_TIME,
    _COL_FUNDING_RATE,
    _COL_MARK_PRICE,
)

# ``mark_price`` is required to exist but may be null (canonical ``Price | None``).
_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = (
    _COL_FUNDING_TIME,
    _COL_FUNDING_RATE,
)

_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    _COL_FUNDING_RATE,
    _COL_MARK_PRICE,
)

_WARN_DUPLICATES: Final[str] = "Duplicate timestamps detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_NUMERIC: Final[str] = "Invalid mark_price values."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by funding_time."


class FundingVerifier(BaseVerifier):
    """Deterministic funding verifier that reports findings only.

    Inspects structural quality of a funding frame. Does not clean rows,
    fill gaps, sort timestamps, mutate values, access storage, or compute
    features. Null ``mark_price`` and negative ``funding_rate`` values are valid.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input funding DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            ProcessingValidationError: If any required column is missing.
        """
        self._validate_required_columns(frame, _REQUIRED_COLUMNS)

        duplicate_timestamp_rows = self._count_duplicate_timestamp_rows(
            frame,
            _COL_FUNDING_TIME,
        )
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _NUMERIC_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_FUNDING_TIME,
        )
        invalid_numeric_rows = self._count_invalid_numeric_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_FUNDING_TIME)

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
        """Return rows with negative non-null ``mark_price``.

        Null ``mark_price`` values are valid and are not counted.
        Negative ``funding_rate`` values are valid and are not counted.
        Each qualifying row is counted once.

        Args:
            frame: Input funding DataFrame. Must not be mutated.

        Returns:
            Count of invalid numeric rows.
        """
        if frame.height == 0:
            return 0
        mark_price = pl.col(_COL_MARK_PRICE)
        invalid_mask = mark_price.is_not_null() & (mark_price < 0)
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

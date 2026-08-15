"""CQROS merged feature dataset verification.

Purpose:
    Inspect merged feature frames and report structural findings without
    cleaning or mutating input data.

Responsibilities:
    - Validate required merged-feature columns and expected dtypes
    - Validate canonical column order
    - Count duplicate ``(symbol, timeframe, open_time)`` keys, nulls, NaNs,
      infinite values, and invalid ``open_time`` values
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.features.schema``,
    ``cqros.features.verification.exceptions``,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``FeatureVerifier``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.features.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.features.verification.exceptions import (
    ERROR_SCHEMA_MISMATCH,
    FeatureValidationError,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = ["FeatureVerifier"]

_COL_OPEN_TIME: Final[str] = "open_time"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

_WARN_DUPLICATES: Final[str] = "Duplicate timestamps detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_NUMERIC: Final[str] = "Infinite feature values detected."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."


class FeatureVerifier(BaseVerifier):
    """Deterministic merged-feature verifier that reports findings only.

    Inspects structural quality of a merged feature frame against
    ``cqros.features.schema``. Does not clean rows, fill gaps, sort
    timestamps, mutate values, access storage, or compute features.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input merged feature DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            ProcessingValidationError: If any required column is missing.
            FeatureValidationError: If column dtypes do not match the
                merged feature schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, REQUIRED_COLUMNS)
        nan_rows = self._count_nan_rows(frame, FEATURE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_OPEN_TIME,
        )
        invalid_numeric_rows = self._count_infinite_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            is_sorted=is_sorted,
            is_canonical_order=is_canonical_order,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and is_sorted
            and is_canonical_order
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

    def _validate_column_dtypes(self, frame: pl.DataFrame) -> None:
        """Raise when any required column dtype differs from the schema.

        Args:
            frame: Input DataFrame. Must not be mutated.

        Raises:
            FeatureValidationError: If one or more column dtypes do not match
                ``COLUMN_DTYPES``.
        """
        mismatched: list[dict[str, object]] = []
        for column in REQUIRED_COLUMNS:
            expected = COLUMN_DTYPES[column]
            actual = frame.schema[column]
            if actual != expected:
                mismatched.append(
                    {
                        "column": column,
                        "expected": str(expected),
                        "actual": str(actual),
                    }
                )
        if mismatched:
            raise FeatureValidationError(
                "merged feature schema dtype mismatch",
                error_code=ERROR_SCHEMA_MISMATCH,
                details={
                    "mismatched_columns": tuple(item["column"] for item in mismatched),
                    "mismatches": tuple(mismatched),
                },
            )

    def _count_duplicate_key_rows(self, frame: pl.DataFrame) -> int:
        """Return rows beyond the first ``(symbol, timeframe, open_time)``.

        Uses keep-first semantics over the canonical merged-feature primary
        key.

        Args:
            frame: Input feature DataFrame. Must not be mutated.

        Returns:
            Count of duplicate primary-key rows.
        """
        if frame.height == 0:
            return 0
        unique_count = int(frame.select(pl.struct(*_DUPLICATE_KEY_COLUMNS).n_unique()).item())
        return frame.height - unique_count

    def _count_infinite_rows(self, frame: pl.DataFrame) -> int:
        """Return rows containing at least one infinite floating value.

        Non-floating columns listed in ``FEATURE_COLUMNS`` are ignored. NaN is
        not treated as infinite.

        Args:
            frame: Input feature DataFrame. Must not be mutated.

        Returns:
            Count of rows with one or more infinite feature values.
        """
        if frame.height == 0:
            return 0
        floating_columns = tuple(
            name
            for name in FEATURE_COLUMNS
            if name in frame.schema and frame.schema[name].is_float()
        )
        if not floating_columns:
            return 0
        infinite_mask = pl.any_horizontal(
            *(pl.col(name).is_infinite() for name in floating_columns)
        )
        return int(frame.select(infinite_mask.sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    invalid_numeric_rows: int,
    is_sorted: bool,
    is_canonical_order: bool,
) -> tuple[str, ...]:
    """Return deterministic warnings for non-zero counters and structure fails."""
    warnings: list[str] = []
    if not is_canonical_order:
        warnings.append(_WARN_COLUMN_ORDER)
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

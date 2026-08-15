"""CQROS merged portfolio dataset verification.

Purpose:
    Inspect canonical portfolio frames and report structural findings without
    cleaning or mutating input data.

Responsibilities:
    - Validate required merged-portfolio columns and expected dtypes
    - Hard-fail on missing ``optimizer`` or ``optimizer`` dtype mismatch
    - Validate canonical column order
    - Count duplicate ``(symbol, timeframe, open_time)`` keys, nulls, NaNs,
      invalid ``open_time`` values, invalid ``signal`` values, invalid empty
      string values (including empty ``optimizer``), and non-finite
      ``target_weight`` values
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.signals.enums``,
    ``cqros.portfolio.schema``, ``cqros.portfolio.verification.exceptions``,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``PortfolioVerifier``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.portfolio.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.portfolio.verification.exceptions import (
    ERROR_REQUIRED_COLUMNS,
    ERROR_SCHEMA_MISMATCH,
    PortfolioValidationError,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport
from cqros.signals.enums import values as signal_values

__all__ = ["PortfolioVerifier"]

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_SIGNAL: Final[str] = "signal"
_COL_TARGET_WEIGHT: Final[str] = "target_weight"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

# Non-primary-key columns inspected for NaN among floating dtypes.
_VALUE_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column not in PRIMARY_KEY_COLUMNS
)

_STRING_COLUMNS: Final[tuple[str, ...]] = tuple(
    column
    for column in REQUIRED_COLUMNS
    if COLUMN_DTYPES[column] == pl.Utf8 or COLUMN_DTYPES[column] == pl.String
)

_ALLOWED_SIGNALS: Final[tuple[str, ...]] = signal_values()

_WARN_DUPLICATES: Final[str] = "Duplicate timestamps detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_INVALID_SIGNAL: Final[str] = "Invalid signal values detected."
_WARN_INVALID_STRING: Final[str] = "Invalid string values detected."
_WARN_INVALID_WEIGHT: Final[str] = "Invalid target weight values detected."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."


class PortfolioVerifier(BaseVerifier):
    """Deterministic canonical-portfolio verifier that reports findings only.

    Inspects structural quality of a canonical portfolio frame against
    ``cqros.portfolio.schema`` and ``cqros.signals.enums.Signal``. Does not
    clean rows, fill gaps, sort timestamps, mutate values, access storage,
    optimize portfolios, or apply trading logic. Strategy-specific weight
    normalization is intentionally out of scope.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical portfolio DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            PortfolioValidationError: If any required column is missing or
                column dtypes do not match the merged portfolio schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, REQUIRED_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_OPEN_TIME,
        )
        invalid_signal_rows = self._count_invalid_signal_rows(frame)
        invalid_string_rows = self._count_invalid_string_rows(frame)
        invalid_weight_rows = self._count_invalid_weight_rows(frame)
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            invalid_signal_rows=invalid_signal_rows,
            invalid_string_rows=invalid_string_rows,
            invalid_weight_rows=invalid_weight_rows,
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

    def _validate_required_columns(
        self,
        frame: pl.DataFrame,
        required_columns: Sequence[str],
    ) -> None:
        """Raise when any required column is absent from ``frame``.

        Args:
            frame: Input DataFrame. Must not be mutated.
            required_columns: Column names that must be present.

        Raises:
            PortfolioValidationError: If one or more required columns are
                missing.
        """
        missing = tuple(name for name in required_columns if name not in frame.columns)
        if missing:
            raise PortfolioValidationError(
                f"missing required columns: {list(missing)}",
                error_code=ERROR_REQUIRED_COLUMNS,
                details={
                    "missing_columns": missing,
                    "required_columns": tuple(required_columns),
                    "available_columns": tuple(frame.columns),
                },
            )

    def _validate_column_dtypes(self, frame: pl.DataFrame) -> None:
        """Raise when any required column dtype differs from the schema.

        Args:
            frame: Input DataFrame. Must not be mutated.

        Raises:
            PortfolioValidationError: If one or more column dtypes do not match
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
            raise PortfolioValidationError(
                "merged portfolio schema dtype mismatch",
                error_code=ERROR_SCHEMA_MISMATCH,
                details={
                    "mismatched_columns": tuple(item["column"] for item in mismatched),
                    "mismatches": tuple(mismatched),
                },
            )

    def _count_duplicate_key_rows(self, frame: pl.DataFrame) -> int:
        """Return rows beyond the first ``(symbol, timeframe, open_time)``.

        Uses keep-first semantics over the canonical merged-portfolio primary
        key.

        Args:
            frame: Input portfolio DataFrame. Must not be mutated.

        Returns:
            Count of duplicate primary-key rows.
        """
        if frame.height == 0:
            return 0
        unique_count = int(frame.select(pl.struct(*_DUPLICATE_KEY_COLUMNS).n_unique()).item())
        return frame.height - unique_count

    def _count_invalid_timestamp_rows(
        self,
        frame: pl.DataFrame,
        timestamp_column: str,
    ) -> int:
        """Return rows with NULL ``open_time`` values.

        Portfolio ``open_time`` is ``Datetime("us", "UTC")``. BaseVerifier's
        integer-only timestamp check is overridden so valid UTC datetimes are
        not treated as globally invalid.

        Args:
            frame: Input portfolio DataFrame. Must not be mutated.
            timestamp_column: Timestamp column name.

        Returns:
            Count of invalid timestamp rows.
        """
        if frame.height == 0:
            return 0
        expected = COLUMN_DTYPES[timestamp_column]
        actual = frame.schema[timestamp_column]
        if actual != expected:
            return frame.height
        return int(frame.select(pl.col(timestamp_column).is_null().sum()).item())

    def _count_invalid_signal_rows(self, frame: pl.DataFrame) -> int:
        """Return rows whose ``signal`` value is outside the Signal enum.

        NULL signals are counted by ``_count_null_rows`` and excluded here.

        Args:
            frame: Input portfolio DataFrame. Must not be mutated.

        Returns:
            Count of rows with unexpected non-null signal values.
        """
        if frame.height == 0:
            return 0
        allowed = list(_ALLOWED_SIGNALS)
        invalid_mask = pl.col(_COL_SIGNAL).is_not_null() & ~pl.col(_COL_SIGNAL).is_in(allowed)
        return int(frame.select(invalid_mask.sum()).item())

    def _count_invalid_string_rows(self, frame: pl.DataFrame) -> int:
        """Return rows containing at least one empty string value.

        Args:
            frame: Input portfolio DataFrame. Must not be mutated.

        Returns:
            Count of rows with one or more empty string fields.
        """
        if frame.height == 0 or not _STRING_COLUMNS:
            return 0
        empty_mask = pl.any_horizontal(*(pl.col(name) == "" for name in _STRING_COLUMNS))
        return int(frame.select(empty_mask.sum()).item())

    def _count_invalid_weight_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with infinite ``target_weight`` values.

        NaN weights are reported through ``_count_nan_rows``. NULL weights are
        reported through ``_count_null_rows``.

        Args:
            frame: Input portfolio DataFrame. Must not be mutated.

        Returns:
            Count of rows with infinite target weights.
        """
        if frame.height == 0:
            return 0
        if (
            _COL_TARGET_WEIGHT not in frame.schema
            or not frame.schema[_COL_TARGET_WEIGHT].is_float()
        ):
            return 0
        return int(frame.select(pl.col(_COL_TARGET_WEIGHT).is_infinite().sum()).item())

    def _count_invalid_value_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with invalid signals, empty strings, or infinite weights.

        Args:
            frame: Input portfolio DataFrame. Must not be mutated.

        Returns:
            Count of rows with one or more invalid categorical/string/weight
            values.
        """
        if frame.height == 0:
            return 0
        allowed = list(_ALLOWED_SIGNALS)
        signal_invalid = pl.col(_COL_SIGNAL).is_not_null() & ~pl.col(_COL_SIGNAL).is_in(allowed)
        empty_string = pl.any_horizontal(*(pl.col(name) == "" for name in _STRING_COLUMNS))
        weight_invalid = pl.col(_COL_TARGET_WEIGHT).is_infinite()
        return int(frame.select((signal_invalid | empty_string | weight_invalid).sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    invalid_signal_rows: int,
    invalid_string_rows: int,
    invalid_weight_rows: int,
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
    if invalid_signal_rows > 0:
        warnings.append(_WARN_INVALID_SIGNAL)
    if invalid_string_rows > 0:
        warnings.append(_WARN_INVALID_STRING)
    if invalid_weight_rows > 0:
        warnings.append(_WARN_INVALID_WEIGHT)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)

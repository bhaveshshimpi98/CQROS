"""CQROS merged executed-trade dataset verification.

Purpose:
    Inspect canonical executed-trade frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required merged-trade columns and expected dtypes
    - Validate canonical column order
    - Count duplicate ``(symbol, timeframe, open_time)`` keys, nulls, NaNs,
      invalid ``open_time`` / ``execution_time`` values, and invalid
      ``status`` enum values
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.execution.exceptions``,
    ``cqros.execution.schema``, ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``ExecutionVerifier``, ``ERROR_REQUIRED_COLUMNS``, ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.execution.exceptions import ExecutionValidationError
from cqros.execution.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    ExecutionStatus,
    values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "ExecutionVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "EXEC-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "EXEC-VERIFICATION-002"

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_EXECUTION_TIME: Final[str] = "execution_time"
_COL_STATUS: Final[str] = "status"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

_VALUE_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column not in PRIMARY_KEY_COLUMNS
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = values(ExecutionStatus)

_WARN_DUPLICATES: Final[str] = "Duplicate timestamps detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid ExecutionStatus values detected."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."


class ExecutionVerifier(BaseVerifier):
    """Deterministic canonical executed-trade verifier that reports findings only.

    Inspects structural quality of a canonical trade frame against
    ``cqros.execution.schema`` / ``MERGED_TRADE_SCHEMA`` and the canonical
    execution status enumeration. Does not clean rows, fill gaps, sort
    timestamps, mutate values, access storage, simulate fills, or apply
    accounting logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical executed-trade DataFrame. Must not be
                mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            ExecutionValidationError: If any required column is missing or
                column dtypes do not match the merged trade schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, REQUIRED_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_OPEN_TIME,
        ) + self._count_invalid_timestamp_rows(
            frame,
            _COL_EXECUTION_TIME,
        )
        empty_status_rows = self._count_empty_string_rows(frame, _COL_STATUS)
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATUS,
            _ALLOWED_STATUSES,
        )
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_status_rows=empty_status_rows,
            invalid_status_rows=invalid_status_rows,
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
            ExecutionValidationError: If one or more required columns are
                missing.
        """
        missing = tuple(name for name in required_columns if name not in frame.columns)
        if missing:
            raise ExecutionValidationError(
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
            ExecutionValidationError: If one or more column dtypes do not
                match ``COLUMN_DTYPES`` / ``MERGED_TRADE_SCHEMA``.
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
            raise ExecutionValidationError(
                "merged trade schema dtype mismatch",
                error_code=ERROR_SCHEMA_MISMATCH,
                details={
                    "mismatched_columns": tuple(item["column"] for item in mismatched),
                    "mismatches": tuple(mismatched),
                },
            )

    def _count_duplicate_key_rows(self, frame: pl.DataFrame) -> int:
        """Return rows beyond the first primary-key occurrence.

        Uses keep-first semantics over ``(symbol, timeframe, open_time)``.

        Args:
            frame: Input trade DataFrame. Must not be mutated.

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
        """Return rows with NULL timestamp values in ``timestamp_column``.

        Trade timestamps are ``Datetime("us", "UTC")``. BaseVerifier's
        integer-only timestamp check is overridden so valid UTC datetimes
        are not treated as globally invalid.

        Args:
            frame: Input trade DataFrame. Must not be mutated.
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

    def _count_empty_string_rows(self, frame: pl.DataFrame, column: str) -> int:
        """Return rows containing an empty string in ``column``.

        Args:
            frame: Input trade DataFrame. Must not be mutated.
            column: String column name to inspect.

        Returns:
            Count of rows with an empty string value in ``column``.
        """
        if frame.height == 0:
            return 0
        return int(frame.select((pl.col(column) == "").sum()).item())

    def _count_invalid_enum_rows(
        self,
        frame: pl.DataFrame,
        column: str,
        allowed: Sequence[str],
    ) -> int:
        """Return rows whose ``column`` value is outside ``allowed``.

        NULL values are counted by ``_count_null_rows`` and excluded here.

        Args:
            frame: Input trade DataFrame. Must not be mutated.
            column: Categorical string column name.
            allowed: Canonical enumeration string values.

        Returns:
            Count of rows with unexpected non-null values.
        """
        if frame.height == 0:
            return 0
        allowed_list = list(allowed)
        invalid_mask = pl.col(column).is_not_null() & ~pl.col(column).is_in(allowed_list)
        return int(frame.select(invalid_mask.sum()).item())

    def _count_invalid_value_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with empty or invalid status values.

        Args:
            frame: Input trade DataFrame. Must not be mutated.

        Returns:
            Count of rows with invalid status categorical values.
        """
        if frame.height == 0:
            return 0
        status_invalid = (pl.col(_COL_STATUS) == "") | (
            pl.col(_COL_STATUS).is_not_null() & ~pl.col(_COL_STATUS).is_in(list(_ALLOWED_STATUSES))
        )
        return int(frame.select(status_invalid.sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    empty_status_rows: int,
    invalid_status_rows: int,
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
    if empty_status_rows > 0:
        warnings.append(_WARN_EMPTY_STATUS)
    if invalid_status_rows > 0:
        warnings.append(_WARN_INVALID_STATUS)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)

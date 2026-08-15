"""CQROS merged pyramiding recommendation dataset verification.

Purpose:
    Inspect canonical pyramiding frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required merged pyramiding columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, duplicate position IDs / trade IDs at
      the same timestamp, nulls, NaNs, invalid timestamps, invalid enum
      values, invalid booleans, non-negative size violations, and
      non-finite numerics
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.pyramiding.exceptions``,
    ``cqros.pyramiding.schema``, ``cqros.processing.verification.base``,
    and ``cqros.processing.verification.report``.

Public API:
    ``PyramidingVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport
from cqros.pyramiding.exceptions import PyramidingValidationError
from cqros.pyramiding.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    PyramidingReason,
    values,
)

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "PyramidingVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "PYR-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "PYR-VERIFICATION-002"

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_REASON: Final[str] = "reason"
_COL_ALLOW_PYRAMID: Final[str] = "allow_pyramid"
_COL_ENTRY: Final[str] = "entry_price"
_COL_CURRENT: Final[str] = "current_price"
_COL_HIGHEST: Final[str] = "highest_price"
_COL_POSITION_SIZE: Final[str] = "position_size"
_COL_ADD_NUMBER: Final[str] = "add_number"
_COL_MAX_ADDS: Final[str] = "max_adds"
_COL_ADDITIONAL: Final[str] = "additional_size"
_COL_RECOMMENDED: Final[str] = "recommended_size"
_COL_PROFIT: Final[str] = "profit_pct"
_COL_POSITION_ID: Final[str] = "position_id"
_COL_TRADE_ID: Final[str] = "trade_id"
_COL_SYMBOL: Final[str] = "symbol"
_COL_TIMEFRAME: Final[str] = "timeframe"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_ENTRY,
    _COL_CURRENT,
    _COL_HIGHEST,
    _COL_POSITION_SIZE,
    _COL_ADDITIONAL,
    _COL_RECOMMENDED,
    _COL_PROFIT,
)

_SIZE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_POSITION_SIZE,
    _COL_ADDITIONAL,
    _COL_RECOMMENDED,
)

_ALLOWED_REASONS: Final[tuple[str, ...]] = values(PyramidingReason)

_WARN_DUPLICATES: Final[str] = "Duplicate pyramiding primary keys detected."
_WARN_DUPLICATE_POSITION_IDS: Final[str] = (
    "Duplicate position_id values detected at the same open_time."
)
_WARN_DUPLICATE_TRADE_IDS: Final[str] = "Duplicate trade_id values detected at the same open_time."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_REASON: Final[str] = "Empty reason values detected."
_WARN_INVALID_REASON: Final[str] = "Invalid PyramidingReason values detected."
_WARN_INVALID_BOOLEAN: Final[str] = "Invalid allow_pyramid boolean values detected."
_WARN_NEGATIVE_SIZES: Final[str] = "Negative size values detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_LINEAGE: Final[str] = "Incomplete lineage metadata detected."

_LINEAGE_COLUMNS: Final[tuple[str, ...]] = ("manager",)


class PyramidingVerifier(BaseVerifier):
    """Deterministic canonical pyramiding verifier that reports findings only.

    Inspects structural quality of a canonical pyramiding frame against
    ``cqros.pyramiding.schema`` / ``MERGED_PYRAMIDING_SCHEMA`` and the
    canonical reason enumeration. Does not clean rows, fill gaps, sort
    timestamps, mutate values, access storage, or apply pyramiding logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical pyramiding DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            PyramidingValidationError: If any required column is missing or
                column dtypes do not match the merged pyramiding schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        duplicate_position_id_rows = self._count_duplicate_id_rows(
            frame,
            _COL_POSITION_ID,
        )
        duplicate_trade_id_rows = self._count_duplicate_id_rows(
            frame,
            _COL_TRADE_ID,
        )
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(frame, _COL_OPEN_TIME)
        empty_reason_rows = self._count_empty_string_rows(frame, _COL_REASON)
        invalid_reason_rows = self._count_invalid_enum_rows(
            frame,
            _COL_REASON,
            _ALLOWED_REASONS,
        )
        invalid_boolean_rows = self._count_invalid_boolean_rows(frame)
        negative_size_rows = self._count_negative_size_rows(frame)
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        incomplete_lineage_rows = self._count_incomplete_lineage_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            duplicate_position_id_rows=duplicate_position_id_rows,
            duplicate_trade_id_rows=duplicate_trade_id_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_reason_rows=empty_reason_rows,
            invalid_reason_rows=invalid_reason_rows,
            invalid_boolean_rows=invalid_boolean_rows,
            negative_size_rows=negative_size_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            incomplete_lineage_rows=incomplete_lineage_rows,
            is_sorted=is_sorted,
            is_canonical_order=is_canonical_order,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and duplicate_position_id_rows == 0
            and duplicate_trade_id_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and empty_reason_rows == 0
            and invalid_reason_rows == 0
            and invalid_boolean_rows == 0
            and negative_size_rows == 0
            and incomplete_lineage_rows == 0
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
        """Raise when any required column is absent from ``frame``."""
        missing = tuple(name for name in required_columns if name not in frame.columns)
        if missing:
            raise PyramidingValidationError(
                f"missing required columns: {list(missing)}",
                error_code=ERROR_REQUIRED_COLUMNS,
                details={
                    "missing_columns": missing,
                    "required_columns": tuple(required_columns),
                    "available_columns": tuple(frame.columns),
                },
            )

    def _validate_column_dtypes(self, frame: pl.DataFrame) -> None:
        """Raise when any required column dtype differs from the schema."""
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
            raise PyramidingValidationError(
                "merged pyramiding schema dtype mismatch",
                error_code=ERROR_SCHEMA_MISMATCH,
                details={
                    "mismatched_columns": tuple(item["column"] for item in mismatched),
                    "mismatches": tuple(mismatched),
                },
            )

    def _count_duplicate_key_rows(self, frame: pl.DataFrame) -> int:
        """Return rows beyond the first primary-key occurrence."""
        if frame.height == 0:
            return 0
        unique_count = int(frame.select(pl.struct(*_DUPLICATE_KEY_COLUMNS).n_unique()).item())
        return frame.height - unique_count

    def _count_duplicate_id_rows(self, frame: pl.DataFrame, column: str) -> int:
        """Return rows beyond first (symbol, timeframe, open_time, id) occurrence."""
        if frame.height == 0:
            return 0
        key_columns = (_COL_SYMBOL, _COL_TIMEFRAME, _COL_OPEN_TIME, column)
        unique_count = int(frame.select(pl.struct(*key_columns).n_unique()).item())
        return frame.height - unique_count

    def _count_invalid_timestamp_rows(
        self,
        frame: pl.DataFrame,
        timestamp_column: str,
    ) -> int:
        """Return rows with NULL timestamp values in ``timestamp_column``."""
        if frame.height == 0:
            return 0
        expected = COLUMN_DTYPES[timestamp_column]
        actual = frame.schema[timestamp_column]
        if actual != expected:
            return frame.height
        return int(frame.select(pl.col(timestamp_column).is_null().sum()).item())

    def _count_empty_string_rows(self, frame: pl.DataFrame, column: str) -> int:
        """Return rows containing an empty string in ``column``."""
        if frame.height == 0:
            return 0
        return int(frame.select((pl.col(column) == "").sum()).item())

    def _count_invalid_enum_rows(
        self,
        frame: pl.DataFrame,
        column: str,
        allowed: Sequence[str],
    ) -> int:
        """Return rows whose ``column`` value is outside ``allowed``."""
        if frame.height == 0:
            return 0
        allowed_list = list(allowed)
        invalid_mask = pl.col(column).is_not_null() & ~pl.col(column).is_in(allowed_list)
        return int(frame.select(invalid_mask.sum()).item())

    def _count_invalid_boolean_rows(self, frame: pl.DataFrame) -> int:
        """Return rows whose ``allow_pyramid`` value is null."""
        if frame.height == 0:
            return 0
        return int(frame.select(pl.col(_COL_ALLOW_PYRAMID).is_null().sum()).item())

    def _count_negative_size_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with negative size or add counters."""
        if frame.height == 0:
            return 0
        negative_sizes = pl.any_horizontal(*(pl.col(column) < 0.0 for column in _SIZE_COLUMNS))
        negative_counts = (pl.col(_COL_ADD_NUMBER) < 0) | (pl.col(_COL_MAX_ADDS) < 0)
        return int(frame.select((negative_sizes | negative_counts).sum()).item())

    def _count_incomplete_lineage_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with blank lineage metadata fields."""
        if frame.height == 0:
            return 0
        blank_mask = pl.any_horizontal(
            *((pl.col(column).is_null()) | (pl.col(column) == "") for column in _LINEAGE_COLUMNS)
        )
        return int(frame.select(blank_mask.sum()).item())

    def _count_invalid_value_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with non-finite required numerics."""
        if frame.height == 0:
            return 0
        non_finite = pl.any_horizontal(*(~pl.col(column).is_finite() for column in _VALUE_COLUMNS))
        return int(frame.select(non_finite.sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    duplicate_position_id_rows: int,
    duplicate_trade_id_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    empty_reason_rows: int,
    invalid_reason_rows: int,
    invalid_boolean_rows: int,
    negative_size_rows: int,
    invalid_numeric_rows: int,
    incomplete_lineage_rows: int,
    is_sorted: bool,
    is_canonical_order: bool,
) -> tuple[str, ...]:
    """Return deterministic warnings for non-zero counters and structure fails."""
    warnings: list[str] = []
    if not is_canonical_order:
        warnings.append(_WARN_COLUMN_ORDER)
    if duplicate_timestamp_rows > 0:
        warnings.append(_WARN_DUPLICATES)
    if duplicate_position_id_rows > 0:
        warnings.append(_WARN_DUPLICATE_POSITION_IDS)
    if duplicate_trade_id_rows > 0:
        warnings.append(_WARN_DUPLICATE_TRADE_IDS)
    if null_rows > 0:
        warnings.append(_WARN_NULLS)
    if nan_rows > 0:
        warnings.append(_WARN_NANS)
    if invalid_timestamp_rows > 0:
        warnings.append(_WARN_TIMESTAMPS)
    if empty_reason_rows > 0:
        warnings.append(_WARN_EMPTY_REASON)
    if invalid_reason_rows > 0:
        warnings.append(_WARN_INVALID_REASON)
    if invalid_boolean_rows > 0:
        warnings.append(_WARN_INVALID_BOOLEAN)
    if negative_size_rows > 0:
        warnings.append(_WARN_NEGATIVE_SIZES)
    if incomplete_lineage_rows > 0:
        warnings.append(_WARN_LINEAGE)
    if invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)

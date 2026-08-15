"""CQROS merged exit-engine recommendation dataset verification.

Purpose:
    Inspect canonical exit-engine frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required merged exit-engine columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, duplicate position IDs at the same
      timestamp, nulls, NaNs, invalid timestamps, invalid enum values,
      non-negative quantity violations, percent range violations, and
      non-finite numerics
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.exit_engine.exceptions``,
    ``cqros.exit_engine.schema``, ``cqros.processing.verification.base``,
    and ``cqros.processing.verification.report``.

Public API:
    ``ExitEngineVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.exit_engine.exceptions import ExitEngineValidationError
from cqros.exit_engine.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    ExitAction,
    ExitReason,
    values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "ExitEngineVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "EXIT-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "EXIT-VERIFICATION-002"

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_CREATED_AT: Final[str] = "created_at"
_COL_EXIT_ACTION: Final[str] = "exit_action"
_COL_EXIT_REASON: Final[str] = "exit_reason"
_COL_ENTRY: Final[str] = "entry_price"
_COL_CURRENT: Final[str] = "current_price"
_COL_QUANTITY: Final[str] = "quantity"
_COL_RR: Final[str] = "risk_reward_ratio"
_COL_REC_QTY: Final[str] = "recommended_quantity"
_COL_REC_PCT: Final[str] = "recommended_percent"
_COL_PRIORITY: Final[str] = "priority"
_COL_POSITION_ID: Final[str] = "position_id"
_COL_SYMBOL: Final[str] = "symbol"
_COL_TIMEFRAME: Final[str] = "timeframe"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = REQUIRED_COLUMNS

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_ENTRY,
    _COL_CURRENT,
    _COL_QUANTITY,
    _COL_RR,
    _COL_REC_QTY,
    _COL_REC_PCT,
)

_NON_NEGATIVE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_QUANTITY,
    _COL_REC_QTY,
)

_ALLOWED_ACTIONS: Final[tuple[str, ...]] = values(ExitAction)
_ALLOWED_REASONS: Final[tuple[str, ...]] = values(ExitReason)

_WARN_DUPLICATES: Final[str] = "Duplicate exit-engine primary keys detected."
_WARN_DUPLICATE_POSITION_IDS: Final[str] = (
    "Duplicate position_id values detected at the same open_time."
)
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_ACTION: Final[str] = "Empty exit_action values detected."
_WARN_EMPTY_REASON: Final[str] = "Empty exit_reason values detected."
_WARN_INVALID_ACTION: Final[str] = "Invalid ExitAction values detected."
_WARN_INVALID_REASON: Final[str] = "Invalid ExitReason values detected."
_WARN_NEGATIVE_QUANTITY: Final[str] = "Negative quantity values detected."
_WARN_PERCENT_RANGE: Final[str] = (
    "recommended_percent values outside the closed interval [0, 1] detected."
)
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_LINEAGE: Final[str] = "Incomplete lineage metadata detected."

_LINEAGE_COLUMNS: Final[tuple[str, ...]] = ("manager",)


class ExitEngineVerifier(BaseVerifier):
    """Deterministic canonical exit-engine verifier that reports findings only.

    Inspects structural quality of a canonical exit-engine frame against
    ``cqros.exit_engine.schema`` / ``MERGED_EXIT_ENGINE_SCHEMA`` and the
    canonical action/reason enumerations. Does not clean rows, fill gaps,
    sort timestamps, mutate values, access storage, or apply exit logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical exit-engine DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            ExitEngineValidationError: If any required column is missing or
                column dtypes do not match the merged exit-engine schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        duplicate_position_id_rows = self._count_duplicate_id_rows(
            frame,
            _COL_POSITION_ID,
        )
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_open_or_created_timestamps(frame)
        empty_action_rows = self._count_empty_string_rows(frame, _COL_EXIT_ACTION)
        empty_reason_rows = self._count_empty_string_rows(frame, _COL_EXIT_REASON)
        invalid_action_rows = self._count_invalid_enum_rows(
            frame,
            _COL_EXIT_ACTION,
            _ALLOWED_ACTIONS,
        )
        invalid_reason_rows = self._count_invalid_enum_rows(
            frame,
            _COL_EXIT_REASON,
            _ALLOWED_REASONS,
        )
        negative_quantity_rows = self._count_negative_quantity_rows(frame)
        percent_range_rows = self._count_percent_range_rows(frame)
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        incomplete_lineage_rows = self._count_incomplete_lineage_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            duplicate_position_id_rows=duplicate_position_id_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_action_rows=empty_action_rows,
            empty_reason_rows=empty_reason_rows,
            invalid_action_rows=invalid_action_rows,
            invalid_reason_rows=invalid_reason_rows,
            negative_quantity_rows=negative_quantity_rows,
            percent_range_rows=percent_range_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            incomplete_lineage_rows=incomplete_lineage_rows,
            is_sorted=is_sorted,
            is_canonical_order=is_canonical_order,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and duplicate_position_id_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and empty_action_rows == 0
            and empty_reason_rows == 0
            and invalid_action_rows == 0
            and invalid_reason_rows == 0
            and negative_quantity_rows == 0
            and percent_range_rows == 0
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
            raise ExitEngineValidationError(
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
            raise ExitEngineValidationError(
                "merged exit-engine schema dtype mismatch",
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

    def _count_invalid_open_or_created_timestamps(self, frame: pl.DataFrame) -> int:
        """Return rows with invalid ``open_time`` or ``created_at`` values."""
        if frame.height == 0:
            return 0
        for column in (_COL_OPEN_TIME, _COL_CREATED_AT):
            expected = COLUMN_DTYPES[column]
            actual = frame.schema[column]
            if actual != expected:
                return frame.height
        return int(
            frame.select(
                (pl.col(_COL_OPEN_TIME).is_null() | pl.col(_COL_CREATED_AT).is_null()).sum()
            ).item()
        )

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

    def _count_negative_quantity_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with negative quantity or recommended_quantity."""
        if frame.height == 0:
            return 0
        negative = pl.any_horizontal(*(pl.col(column) < 0.0 for column in _NON_NEGATIVE_COLUMNS))
        negative_priority = pl.col(_COL_PRIORITY) < 0
        return int(frame.select((negative | negative_priority).sum()).item())

    def _count_percent_range_rows(self, frame: pl.DataFrame) -> int:
        """Return rows whose recommended_percent is outside ``[0, 1]``."""
        if frame.height == 0:
            return 0
        out_of_range = (pl.col(_COL_REC_PCT) < 0.0) | (pl.col(_COL_REC_PCT) > 1.0)
        return int(frame.select(out_of_range.sum()).item())

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
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    empty_action_rows: int,
    empty_reason_rows: int,
    invalid_action_rows: int,
    invalid_reason_rows: int,
    negative_quantity_rows: int,
    percent_range_rows: int,
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
    if null_rows > 0:
        warnings.append(_WARN_NULLS)
    if nan_rows > 0:
        warnings.append(_WARN_NANS)
    if invalid_timestamp_rows > 0:
        warnings.append(_WARN_TIMESTAMPS)
    if empty_action_rows > 0:
        warnings.append(_WARN_EMPTY_ACTION)
    if empty_reason_rows > 0:
        warnings.append(_WARN_EMPTY_REASON)
    if invalid_action_rows > 0:
        warnings.append(_WARN_INVALID_ACTION)
    if invalid_reason_rows > 0:
        warnings.append(_WARN_INVALID_REASON)
    if negative_quantity_rows > 0:
        warnings.append(_WARN_NEGATIVE_QUANTITY)
    if percent_range_rows > 0:
        warnings.append(_WARN_PERCENT_RANGE)
    if incomplete_lineage_rows > 0:
        warnings.append(_WARN_LINEAGE)
    if invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)

"""CQROS merged trade management decision dataset verification.

Purpose:
    Inspect canonical trade-management frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required merged trade-management columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, nulls, NaNs, invalid timestamps, invalid
      enum values, invalid booleans, price inconsistencies, and non-finite
      numerics
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.trade_management.exceptions``,
    ``cqros.trade_management.schema``, ``cqros.processing.verification.base``,
    and ``cqros.processing.verification.report``.

Public API:
    ``TradeManagementVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport
from cqros.trade_management.exceptions import TradeManagementValidationError
from cqros.trade_management.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    ManagementAction,
    ShutdownReason,
    values,
)

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "TradeManagementVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "TME-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "TME-VERIFICATION-002"

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_ACTION: Final[str] = "management_action"
_COL_REASON: Final[str] = "action_reason"
_COL_ALLOW_PYRAMID: Final[str] = "allow_pyramid"
_COL_QUANTITY: Final[str] = "quantity"
_COL_ENTRY: Final[str] = "entry_price"
_COL_CURRENT: Final[str] = "current_price"
_COL_HIGHEST: Final[str] = "highest_price"
_COL_LOWEST: Final[str] = "lowest_price"
_COL_UNREALIZED: Final[str] = "unrealized_pnl"
_COL_STOP: Final[str] = "stop_price"
_COL_TAKE_PROFIT: Final[str] = "take_profit_price"
_COL_TRAIL: Final[str] = "trail_price"
_COL_BREAKEVEN: Final[str] = "breakeven_price"
_COL_EXIT_QTY: Final[str] = "exit_quantity"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

# Nullable advisory price columns excluded from strict null checks.
_NULLABLE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        _COL_STOP,
        _COL_TAKE_PROFIT,
        _COL_BREAKEVEN,
    }
)

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column not in _NULLABLE_COLUMNS
)

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_QUANTITY,
    _COL_ENTRY,
    _COL_CURRENT,
    _COL_HIGHEST,
    _COL_LOWEST,
    _COL_UNREALIZED,
    _COL_TRAIL,
    _COL_EXIT_QTY,
)

_ALLOWED_ACTIONS: Final[tuple[str, ...]] = values(ManagementAction)
_ALLOWED_REASONS: Final[tuple[str, ...]] = values(ShutdownReason)

_WARN_DUPLICATES: Final[str] = "Duplicate trade-management primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_ACTION: Final[str] = "Empty management_action values detected."
_WARN_INVALID_ACTION: Final[str] = "Invalid ManagementAction values detected."
_WARN_INVALID_REASON: Final[str] = "Invalid ShutdownReason values detected."
_WARN_INVALID_BOOLEAN: Final[str] = "Invalid allow_pyramid boolean values detected."
_WARN_PRICE_CONSISTENCY: Final[str] = "Inconsistent price relationships detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_LINEAGE: Final[str] = "Incomplete lineage metadata detected."

_LINEAGE_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "model_name",
    "model_version",
    "optimizer",
    "policy",
)


class TradeManagementVerifier(BaseVerifier):
    """Deterministic canonical trade-management verifier that reports findings only.

    Inspects structural quality of a canonical trade-management frame against
    ``cqros.trade_management.schema`` / ``MERGED_TRADE_MANAGEMENT_SCHEMA`` and
    the canonical action / reason enumerations. Does not clean rows, fill
    gaps, sort timestamps, mutate values, access storage, or apply management
    logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical trade-management DataFrame. Must not be
                mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            TradeManagementValidationError: If any required column is missing
                or column dtypes do not match the merged trade-management
                schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(frame, _COL_OPEN_TIME)
        empty_action_rows = self._count_empty_string_rows(frame, _COL_ACTION)
        invalid_action_rows = self._count_invalid_enum_rows(
            frame,
            _COL_ACTION,
            _ALLOWED_ACTIONS,
        )
        invalid_reason_rows = self._count_invalid_enum_rows(
            frame,
            _COL_REASON,
            _ALLOWED_REASONS,
        )
        invalid_boolean_rows = self._count_invalid_boolean_rows(frame)
        inconsistent_price_rows = self._count_inconsistent_price_rows(frame)
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        incomplete_lineage_rows = self._count_incomplete_lineage_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_action_rows=empty_action_rows,
            invalid_action_rows=invalid_action_rows,
            invalid_reason_rows=invalid_reason_rows,
            invalid_boolean_rows=invalid_boolean_rows,
            inconsistent_price_rows=inconsistent_price_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            incomplete_lineage_rows=incomplete_lineage_rows,
            is_sorted=is_sorted,
            is_canonical_order=is_canonical_order,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and empty_action_rows == 0
            and invalid_action_rows == 0
            and invalid_reason_rows == 0
            and invalid_boolean_rows == 0
            and inconsistent_price_rows == 0
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
            raise TradeManagementValidationError(
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
            raise TradeManagementValidationError(
                "merged trade-management schema dtype mismatch",
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

    def _count_inconsistent_price_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with inconsistent price relationships.

        Checks:
            - ``highest_price >= lowest_price``
            - ``highest_price >= current_price``
            - ``lowest_price <= current_price``
            - ``entry_price > 0``, ``current_price > 0``, ``quantity >= 0``
            - ``trail_price`` is finite and non-negative when present
            - ``stop_price`` present when ``management_action=UPDATE_STOP``
            - ``breakeven_price == entry_price`` when reason is ``BREAKEVEN``
        """
        if frame.height == 0:
            return 0
        high_low = pl.col(_COL_HIGHEST) < pl.col(_COL_LOWEST)
        high_current = pl.col(_COL_HIGHEST) < pl.col(_COL_CURRENT)
        low_current = pl.col(_COL_LOWEST) > pl.col(_COL_CURRENT)
        non_positive_entry = pl.col(_COL_ENTRY) <= 0.0
        non_positive_current = pl.col(_COL_CURRENT) <= 0.0
        negative_quantity = pl.col(_COL_QUANTITY) < 0.0
        negative_trail = pl.col(_COL_TRAIL) < 0.0
        missing_stop_on_update = (
            pl.col(_COL_ACTION) == ManagementAction.UPDATE_STOP.value
        ) & pl.col(_COL_STOP).is_null()
        bad_breakeven = (pl.col(_COL_REASON) == ShutdownReason.BREAKEVEN.value) & (
            pl.col(_COL_BREAKEVEN).is_null() | (pl.col(_COL_BREAKEVEN) != pl.col(_COL_ENTRY))
        )
        invalid_mask = (
            high_low
            | high_current
            | low_current
            | non_positive_entry
            | non_positive_current
            | negative_quantity
            | negative_trail
            | missing_stop_on_update
            | bad_breakeven
        )
        return int(frame.select(invalid_mask.sum()).item())

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
        optional_non_finite = pl.any_horizontal(
            *(
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                for column in (_COL_STOP, _COL_TAKE_PROFIT, _COL_BREAKEVEN)
            )
        )
        return int(frame.select((non_finite | optional_non_finite).sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    empty_action_rows: int,
    invalid_action_rows: int,
    invalid_reason_rows: int,
    invalid_boolean_rows: int,
    inconsistent_price_rows: int,
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
    if null_rows > 0:
        warnings.append(_WARN_NULLS)
    if nan_rows > 0:
        warnings.append(_WARN_NANS)
    if invalid_timestamp_rows > 0:
        warnings.append(_WARN_TIMESTAMPS)
    if empty_action_rows > 0:
        warnings.append(_WARN_EMPTY_ACTION)
    if invalid_action_rows > 0:
        warnings.append(_WARN_INVALID_ACTION)
    if invalid_reason_rows > 0:
        warnings.append(_WARN_INVALID_REASON)
    if invalid_boolean_rows > 0:
        warnings.append(_WARN_INVALID_BOOLEAN)
    if inconsistent_price_rows > 0:
        warnings.append(_WARN_PRICE_CONSISTENCY)
    if incomplete_lineage_rows > 0:
        warnings.append(_WARN_LINEAGE)
    if invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)

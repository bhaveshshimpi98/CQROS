"""CQROS merged backtesting performance dataset verification.

Purpose:
    Inspect canonical backtesting frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required merged-backtesting columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, nulls, NaNs, invalid timestamps,
      invalid status enum values, drawdown range violations, win-rate
      range violations, profit-factor violations, and non-finite numerics
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.backtesting.exceptions``,
    ``cqros.backtesting.schema``, ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``BacktestingVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.backtesting.exceptions import BacktestingValidationError
from cqros.backtesting.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    BacktestingStatus,
    values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "BacktestingVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "BT-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "BT-VERIFICATION-002"

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_STATUS: Final[str] = "status"
_COL_EQUITY: Final[str] = "equity"
_COL_CASH: Final[str] = "cash"
_COL_POSITION_VALUE: Final[str] = "position_value"
_COL_REALIZED_PNL: Final[str] = "realized_pnl"
_COL_UNREALIZED_PNL: Final[str] = "unrealized_pnl"
_COL_TOTAL_PNL: Final[str] = "total_pnl"
_COL_DRAWDOWN: Final[str] = "drawdown"
_COL_PEAK_EQUITY: Final[str] = "peak_equity"
_COL_DAILY_RETURN: Final[str] = "daily_return"
_COL_CUMULATIVE_RETURN: Final[str] = "cumulative_return"
_COL_TRADE_COUNT: Final[str] = "trade_count"
_COL_WINNING_TRADES: Final[str] = "winning_trades"
_COL_LOSING_TRADES: Final[str] = "losing_trades"
_COL_WIN_RATE: Final[str] = "win_rate"
_COL_PROFIT_FACTOR: Final[str] = "profit_factor"
_COL_SHARPE_STUB: Final[str] = "sharpe_stub"
_COL_SORTINO_STUB: Final[str] = "sortino_stub"
_COL_MAX_DRAWDOWN: Final[str] = "max_drawdown"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = tuple(
    column
    for column in REQUIRED_COLUMNS
    if column not in (_COL_PROFIT_FACTOR, _COL_SHARPE_STUB, _COL_SORTINO_STUB)
)

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_EQUITY,
    _COL_CASH,
    _COL_POSITION_VALUE,
    _COL_REALIZED_PNL,
    _COL_UNREALIZED_PNL,
    _COL_TOTAL_PNL,
    _COL_DRAWDOWN,
    _COL_PEAK_EQUITY,
    _COL_DAILY_RETURN,
    _COL_CUMULATIVE_RETURN,
    _COL_WIN_RATE,
    _COL_MAX_DRAWDOWN,
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = values(BacktestingStatus)

_WARN_DUPLICATES: Final[str] = "Duplicate backtesting primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid BacktestingStatus values detected."
_WARN_DRAWDOWN_RANGE: Final[str] = "Drawdown values outside [0, 1] detected."
_WARN_MAX_DRAWDOWN: Final[str] = "max_drawdown less than drawdown detected."
_WARN_WIN_RATE: Final[str] = "Win rate values outside [0, 1] detected."
_WARN_PROFIT_FACTOR: Final[str] = "Negative profit_factor values detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_LINEAGE: Final[str] = "Incomplete lineage metadata detected."

_LINEAGE_COLUMNS: Final[tuple[str, ...]] = ("manager",)


class BacktestingVerifier(BaseVerifier):
    """Deterministic canonical backtesting verifier that reports findings only.

    Inspects structural quality of a canonical backtesting frame against
    ``cqros.backtesting.schema`` / ``MERGED_BACKTESTING_SCHEMA`` and the
    canonical status enumeration. Does not clean rows, fill gaps, sort
    timestamps, mutate values, access storage, or apply performance logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical backtesting DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            BacktestingValidationError: If any required column is missing or
                column dtypes do not match the merged backtesting schema.
        """
        self._validate_required_columns(frame, REQUIRED_COLUMNS)
        self._validate_column_dtypes(frame)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _NULL_CHECK_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _VALUE_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(frame, _COL_OPEN_TIME)
        empty_status_rows = self._count_empty_string_rows(frame, _COL_STATUS)
        invalid_status_rows = self._count_invalid_enum_rows(
            frame,
            _COL_STATUS,
            _ALLOWED_STATUSES,
        )
        invalid_numeric_rows = self._count_invalid_value_rows(frame)
        incomplete_lineage_rows = self._count_incomplete_lineage_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)
        is_canonical_order = tuple(frame.columns) == CANONICAL_COLUMN_ORDER

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            empty_status_rows=empty_status_rows,
            invalid_status_rows=invalid_status_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            incomplete_lineage_rows=incomplete_lineage_rows,
            is_sorted=is_sorted,
            is_canonical_order=is_canonical_order,
            frame=frame,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and empty_status_rows == 0
            and invalid_status_rows == 0
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
            raise BacktestingValidationError(
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
            raise BacktestingValidationError(
                "merged backtesting schema dtype mismatch",
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

    def _count_incomplete_lineage_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with blank lineage metadata fields."""
        if frame.height == 0:
            return 0
        blank_mask = pl.any_horizontal(
            *((pl.col(column).is_null()) | (pl.col(column) == "") for column in _LINEAGE_COLUMNS)
        )
        return int(frame.select(blank_mask.sum()).item())

    def _count_invalid_value_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with domain or non-finite numeric violations."""
        if frame.height == 0:
            return 0
        non_finite = pl.any_horizontal(*(~pl.col(column).is_finite() for column in _VALUE_COLUMNS))
        drawdown_invalid = (pl.col(_COL_DRAWDOWN) < 0.0) | (pl.col(_COL_DRAWDOWN) > 1.0)
        max_dd_invalid = pl.col(_COL_MAX_DRAWDOWN) < pl.col(_COL_DRAWDOWN)
        win_rate_invalid = (pl.col(_COL_WIN_RATE) < 0.0) | (pl.col(_COL_WIN_RATE) > 1.0)
        profit_factor_invalid = pl.col(_COL_PROFIT_FACTOR).is_not_null() & (
            pl.col(_COL_PROFIT_FACTOR) < 0.0
        )
        invalid_mask = (
            non_finite
            | drawdown_invalid
            | max_dd_invalid
            | win_rate_invalid
            | profit_factor_invalid
        )
        return int(frame.select(invalid_mask.sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    empty_status_rows: int,
    invalid_status_rows: int,
    invalid_numeric_rows: int,
    incomplete_lineage_rows: int,
    is_sorted: bool,
    is_canonical_order: bool,
    frame: pl.DataFrame,
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
    if incomplete_lineage_rows > 0:
        warnings.append(_WARN_LINEAGE)
    if frame.height > 0:
        if (
            int(
                frame.select(
                    ((pl.col(_COL_DRAWDOWN) < 0.0) | (pl.col(_COL_DRAWDOWN) > 1.0)).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_DRAWDOWN_RANGE)
        if int(frame.select((pl.col(_COL_MAX_DRAWDOWN) < pl.col(_COL_DRAWDOWN)).sum()).item()) > 0:
            warnings.append(_WARN_MAX_DRAWDOWN)
        if (
            int(
                frame.select(
                    ((pl.col(_COL_WIN_RATE) < 0.0) | (pl.col(_COL_WIN_RATE) > 1.0)).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_WIN_RATE)
        if (
            int(
                frame.select(
                    (
                        pl.col(_COL_PROFIT_FACTOR).is_not_null()
                        & (pl.col(_COL_PROFIT_FACTOR) < 0.0)
                    ).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_PROFIT_FACTOR)
        if (
            int(
                frame.select(
                    pl.any_horizontal(
                        *(~pl.col(column).is_finite() for column in _VALUE_COLUMNS)
                    ).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_NON_FINITE)
    elif invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)

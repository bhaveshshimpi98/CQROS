"""CQROS merged performance metrics dataset verification.

Purpose:
    Inspect canonical performance frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required performance columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, nulls, NaNs, invalid timestamps,
      invalid status enum values, win-rate range violations, max-drawdown
      range violations, drawdown-duration violations, trade-count
      consistency violations, and non-finite numerics
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.performance.exceptions``,
    ``cqros.performance.schema``, ``cqros.processing.verification.base``,
    and ``cqros.processing.verification.report``.

Public API:
    ``PerformanceVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.performance.exceptions import PerformanceValidationError
from cqros.performance.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    PerformanceStatus,
    values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "PerformanceVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "PERF-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "PERF-VERIFICATION-002"

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_FIRST_TRADE_TIME: Final[str] = "first_trade_time"
_COL_LAST_TRADE_TIME: Final[str] = "last_trade_time"
_COL_STATUS: Final[str] = "status"
_COL_TOTAL_RETURN: Final[str] = "total_return"
_COL_CAGR: Final[str] = "cagr"
_COL_VOLATILITY: Final[str] = "volatility"
_COL_DOWNSIDE_VOLATILITY: Final[str] = "downside_volatility"
_COL_MAX_DRAWDOWN: Final[str] = "max_drawdown"
_COL_DRAWDOWN_DURATION: Final[str] = "drawdown_duration"
_COL_SHARPE_RATIO: Final[str] = "sharpe_ratio"
_COL_SORTINO_RATIO: Final[str] = "sortino_ratio"
_COL_CALMAR_RATIO: Final[str] = "calmar_ratio"
_COL_TOTAL_TRADES: Final[str] = "total_trades"
_COL_WINNING_TRADES: Final[str] = "winning_trades"
_COL_LOSING_TRADES: Final[str] = "losing_trades"
_COL_WIN_RATE: Final[str] = "win_rate"
_COL_AVERAGE_WIN: Final[str] = "average_win"
_COL_AVERAGE_LOSS: Final[str] = "average_loss"
_COL_PROFIT_FACTOR: Final[str] = "profit_factor"
_COL_EXPECTANCY: Final[str] = "expectancy"
_COL_STARTING_EQUITY: Final[str] = "starting_equity"
_COL_ENDING_EQUITY: Final[str] = "ending_equity"
_COL_NET_PROFIT: Final[str] = "net_profit"
_COL_GROSS_PROFIT: Final[str] = "gross_profit"
_COL_GROSS_LOSS: Final[str] = "gross_loss"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

# Optional / nullable metric columns excluded from mandatory null checks.
_NULLABLE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        _COL_SHARPE_RATIO,
        _COL_SORTINO_RATIO,
        _COL_CALMAR_RATIO,
        _COL_AVERAGE_WIN,
        _COL_AVERAGE_LOSS,
        _COL_PROFIT_FACTOR,
        _COL_FIRST_TRADE_TIME,
        _COL_LAST_TRADE_TIME,
    }
)

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column not in _NULLABLE_COLUMNS
)

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_TOTAL_RETURN,
    _COL_CAGR,
    _COL_VOLATILITY,
    _COL_DOWNSIDE_VOLATILITY,
    _COL_MAX_DRAWDOWN,
    _COL_WIN_RATE,
    _COL_EXPECTANCY,
    _COL_STARTING_EQUITY,
    _COL_ENDING_EQUITY,
    _COL_NET_PROFIT,
    _COL_GROSS_PROFIT,
    _COL_GROSS_LOSS,
)

_OPTIONAL_FINITE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_SHARPE_RATIO,
    _COL_SORTINO_RATIO,
    _COL_CALMAR_RATIO,
    _COL_AVERAGE_WIN,
    _COL_AVERAGE_LOSS,
    _COL_PROFIT_FACTOR,
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = values(PerformanceStatus)

_WARN_DUPLICATES: Final[str] = "Duplicate performance primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid PerformanceStatus values detected."
_WARN_WIN_RATE: Final[str] = "Win rate values outside [0, 1] detected."
_WARN_MAX_DRAWDOWN: Final[str] = "max_drawdown values outside [0, 1] detected."
_WARN_DRAWDOWN_DURATION: Final[str] = "Negative drawdown_duration values detected."
_WARN_TRADE_COUNTS: Final[str] = "total_trades less than winning_trades + losing_trades detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_LINEAGE: Final[str] = "Incomplete lineage metadata detected."

_LINEAGE_COLUMNS: Final[tuple[str, ...]] = ("manager",)


class PerformanceVerifier(BaseVerifier):
    """Deterministic canonical performance verifier that reports findings only.

    Inspects structural quality of a canonical performance frame against
    ``cqros.performance.schema`` / ``PERFORMANCE_SCHEMA`` and the canonical
    status enumeration. Does not clean rows, fill gaps, sort timestamps,
    mutate values, access storage, or apply performance logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical performance DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            PerformanceValidationError: If any required column is missing or
                column dtypes do not match the performance schema.
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
            raise PerformanceValidationError(
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
            raise PerformanceValidationError(
                "performance schema dtype mismatch",
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
        required_non_finite = pl.any_horizontal(
            *(~pl.col(column).is_finite() for column in _VALUE_COLUMNS)
        )
        optional_non_finite = pl.any_horizontal(
            *(
                pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                for column in _OPTIONAL_FINITE_COLUMNS
            )
        )
        win_rate_invalid = (pl.col(_COL_WIN_RATE) < 0.0) | (pl.col(_COL_WIN_RATE) > 1.0)
        max_drawdown_invalid = (pl.col(_COL_MAX_DRAWDOWN) < 0.0) | (pl.col(_COL_MAX_DRAWDOWN) > 1.0)
        drawdown_duration_invalid = pl.col(_COL_DRAWDOWN_DURATION) < 0
        trade_count_invalid = pl.col(_COL_TOTAL_TRADES) < (
            pl.col(_COL_WINNING_TRADES) + pl.col(_COL_LOSING_TRADES)
        )
        invalid_mask = (
            required_non_finite
            | optional_non_finite
            | win_rate_invalid
            | max_drawdown_invalid
            | drawdown_duration_invalid
            | trade_count_invalid
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
                    ((pl.col(_COL_WIN_RATE) < 0.0) | (pl.col(_COL_WIN_RATE) > 1.0)).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_WIN_RATE)
        if (
            int(
                frame.select(
                    ((pl.col(_COL_MAX_DRAWDOWN) < 0.0) | (pl.col(_COL_MAX_DRAWDOWN) > 1.0)).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_MAX_DRAWDOWN)
        if int(frame.select((pl.col(_COL_DRAWDOWN_DURATION) < 0).sum()).item()) > 0:
            warnings.append(_WARN_DRAWDOWN_DURATION)
        if (
            int(
                frame.select(
                    (
                        pl.col(_COL_TOTAL_TRADES)
                        < (pl.col(_COL_WINNING_TRADES) + pl.col(_COL_LOSING_TRADES))
                    ).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_TRADE_COUNTS)
        required_non_finite = int(
            frame.select(
                pl.any_horizontal(*(~pl.col(column).is_finite() for column in _VALUE_COLUMNS)).sum()
            ).item()
        )
        optional_non_finite = int(
            frame.select(
                pl.any_horizontal(
                    *(
                        pl.col(column).is_not_null() & ~pl.col(column).is_finite()
                        for column in _OPTIONAL_FINITE_COLUMNS
                    )
                ).sum()
            ).item()
        )
        if required_non_finite > 0 or optional_non_finite > 0:
            warnings.append(_WARN_NON_FINITE)
    elif invalid_numeric_rows > 0:
        warnings.append(_WARN_NON_FINITE)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)

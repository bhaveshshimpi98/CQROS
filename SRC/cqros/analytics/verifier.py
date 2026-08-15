"""CQROS merged analytics metrics dataset verification.

Purpose:
    Inspect canonical analytics frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required analytics columns and expected dtypes
    - Validate canonical column order
    - Count duplicate primary keys, nulls, NaNs, invalid timestamps,
      invalid status enum values, rolling win-rate/max-drawdown range
      violations, rolling-volatility and tracking-error non-negativity
      violations, correlation range violations, and non-finite numerics
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library, ``cqros.analytics.exceptions``,
    ``cqros.analytics.schema``, ``cqros.processing.verification.base``,
    and ``cqros.processing.verification.report``.

Public API:
    ``AnalyticsVerifier``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_SCHEMA_MISMATCH``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.analytics.exceptions import AnalyticsValidationError
from cqros.analytics.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    analytics_status_values,
)
from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = [
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "AnalyticsVerifier",
]

ERROR_REQUIRED_COLUMNS: Final[str] = "ANA-VERIFICATION-001"
ERROR_SCHEMA_MISMATCH: Final[str] = "ANA-VERIFICATION-002"

_COL_OPEN_TIME: Final[str] = "open_time"
_COL_STATUS: Final[str] = "status"
_COL_ROLLING_RETURN: Final[str] = "rolling_return"
_COL_ROLLING_VOLATILITY: Final[str] = "rolling_volatility"
_COL_ROLLING_SHARPE: Final[str] = "rolling_sharpe"
_COL_ROLLING_SORTINO: Final[str] = "rolling_sortino"
_COL_ROLLING_MAX_DRAWDOWN: Final[str] = "rolling_max_drawdown"
_COL_ROLLING_WIN_RATE: Final[str] = "rolling_win_rate"
_COL_ROLLING_PROFIT_FACTOR: Final[str] = "rolling_profit_factor"
_COL_ROLLING_EXPECTANCY: Final[str] = "rolling_expectancy"
_COL_ROLLING_CAGR: Final[str] = "rolling_cagr"
_COL_ROLLING_CALMAR: Final[str] = "rolling_calmar"
_COL_ROLLING_RECOVERY_FACTOR: Final[str] = "rolling_recovery_factor"
_COL_BENCHMARK_RETURN: Final[str] = "benchmark_return"
_COL_BENCHMARK_ALPHA: Final[str] = "benchmark_alpha"
_COL_BENCHMARK_BETA: Final[str] = "benchmark_beta"
_COL_BENCHMARK_CORRELATION: Final[str] = "benchmark_correlation"
_COL_BENCHMARK_TRACKING_ERROR: Final[str] = "benchmark_tracking_error"
_COL_BENCHMARK_INFORMATION_RATIO: Final[str] = "benchmark_information_ratio"

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = PRIMARY_KEY_COLUMNS

# Optional / nullable metric columns excluded from mandatory null checks.
_NULLABLE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        _COL_ROLLING_SHARPE,
        _COL_ROLLING_SORTINO,
        _COL_ROLLING_PROFIT_FACTOR,
        _COL_ROLLING_CALMAR,
        _COL_ROLLING_RECOVERY_FACTOR,
    }
)

_NULL_CHECK_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column in REQUIRED_COLUMNS if column not in _NULLABLE_COLUMNS
)

_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_ROLLING_RETURN,
    _COL_ROLLING_VOLATILITY,
    _COL_ROLLING_MAX_DRAWDOWN,
    _COL_ROLLING_WIN_RATE,
    _COL_ROLLING_EXPECTANCY,
    _COL_ROLLING_CAGR,
    _COL_BENCHMARK_RETURN,
    _COL_BENCHMARK_ALPHA,
    _COL_BENCHMARK_BETA,
    _COL_BENCHMARK_CORRELATION,
    _COL_BENCHMARK_TRACKING_ERROR,
    _COL_BENCHMARK_INFORMATION_RATIO,
)

_OPTIONAL_FINITE_COLUMNS: Final[tuple[str, ...]] = (
    _COL_ROLLING_SHARPE,
    _COL_ROLLING_SORTINO,
    _COL_ROLLING_PROFIT_FACTOR,
    _COL_ROLLING_CALMAR,
    _COL_ROLLING_RECOVERY_FACTOR,
)

_ALLOWED_STATUSES: Final[tuple[str, ...]] = analytics_status_values()

_WARN_DUPLICATES: Final[str] = "Duplicate analytics primary keys detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_EMPTY_STATUS: Final[str] = "Empty status values detected."
_WARN_INVALID_STATUS: Final[str] = "Invalid AnalyticsStatus values detected."
_WARN_WIN_RATE: Final[str] = "rolling_win_rate values outside [0, 1] detected."
_WARN_MAX_DRAWDOWN: Final[str] = "rolling_max_drawdown values outside [0, 1] detected."
_WARN_VOLATILITY: Final[str] = "Negative rolling_volatility values detected."
_WARN_CORRELATION: Final[str] = "benchmark_correlation values outside [-1, 1] detected."
_WARN_TRACKING_ERROR: Final[str] = "Negative benchmark_tracking_error values detected."
_WARN_NON_FINITE: Final[str] = "Non-finite numeric values detected."
_WARN_COLUMN_ORDER: Final[str] = "Frame column order does not match canonical order."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."
_WARN_LINEAGE: Final[str] = "Incomplete lineage metadata detected."

_LINEAGE_COLUMNS: Final[tuple[str, ...]] = ("manager",)


class AnalyticsVerifier(BaseVerifier):
    """Deterministic canonical analytics verifier that reports findings only.

    Inspects structural quality of a canonical analytics frame against
    ``cqros.analytics.schema`` / ``ANALYTICS_SCHEMA`` and the canonical
    status enumeration. Does not clean rows, fill gaps, sort timestamps,
    mutate values, access storage, or apply analytics logic.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input canonical analytics DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            AnalyticsValidationError: If any required column is missing or
                column dtypes do not match the analytics schema.
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
            raise AnalyticsValidationError(
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
            raise AnalyticsValidationError(
                "analytics schema dtype mismatch",
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
        win_rate_invalid = (pl.col(_COL_ROLLING_WIN_RATE) < 0.0) | (
            pl.col(_COL_ROLLING_WIN_RATE) > 1.0
        )
        max_drawdown_invalid = (pl.col(_COL_ROLLING_MAX_DRAWDOWN) < 0.0) | (
            pl.col(_COL_ROLLING_MAX_DRAWDOWN) > 1.0
        )
        volatility_invalid = pl.col(_COL_ROLLING_VOLATILITY) < 0.0
        correlation_invalid = (pl.col(_COL_BENCHMARK_CORRELATION) < -1.0) | (
            pl.col(_COL_BENCHMARK_CORRELATION) > 1.0
        )
        tracking_error_invalid = pl.col(_COL_BENCHMARK_TRACKING_ERROR) < 0.0
        invalid_mask = (
            required_non_finite
            | optional_non_finite
            | win_rate_invalid
            | max_drawdown_invalid
            | volatility_invalid
            | correlation_invalid
            | tracking_error_invalid
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
                    (
                        (pl.col(_COL_ROLLING_WIN_RATE) < 0.0)
                        | (pl.col(_COL_ROLLING_WIN_RATE) > 1.0)
                    ).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_WIN_RATE)
        if (
            int(
                frame.select(
                    (
                        (pl.col(_COL_ROLLING_MAX_DRAWDOWN) < 0.0)
                        | (pl.col(_COL_ROLLING_MAX_DRAWDOWN) > 1.0)
                    ).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_MAX_DRAWDOWN)
        if int(frame.select((pl.col(_COL_ROLLING_VOLATILITY) < 0.0).sum()).item()) > 0:
            warnings.append(_WARN_VOLATILITY)
        if (
            int(
                frame.select(
                    (
                        (pl.col(_COL_BENCHMARK_CORRELATION) < -1.0)
                        | (pl.col(_COL_BENCHMARK_CORRELATION) > 1.0)
                    ).sum()
                ).item()
            )
            > 0
        ):
            warnings.append(_WARN_CORRELATION)
        if int(frame.select((pl.col(_COL_BENCHMARK_TRACKING_ERROR) < 0.0).sum()).item()) > 0:
            warnings.append(_WARN_TRACKING_ERROR)
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

"""CQROS market-data validation for downloaded OHLCV datasets.

Purpose:
    Validate Polars OHLCV frames before persistence so only structurally
    sound, temporally consistent market data enters storage.

Responsibilities:
    - Represent immutable ``ValidationIssue`` and ``ValidationReport`` values
    - Validate schema, nulls, OHLC consistency, volume, and timestamps
    - Detect duplicate, missing, misaligned, and non-monotonic open times
    - Never mutate caller-supplied frames and never perform file I/O

Dependencies:
    ``polars``, ``cqros.core``, and ``cqros.data.timeframes``.

Public API:
    ``ValidationSeverity``, ``ValidationIssue``, ``ValidationReport``, and
    ``MarketDataValidator``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, cast

import polars as pl

from cqros.core.constants import MILLISECONDS_PER_SECOND
from cqros.core.exceptions import ValidationError
from cqros.core.types import Timeframe
from cqros.data.timeframes import Timeframe as CanonicalTimeframe
from cqros.data.timeframes import to_seconds

__all__ = [
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationReport",
    "MarketDataValidator",
]

_OHLCV_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "symbol": pl.String,
        "timeframe": pl.String,
        "open_time": pl.Int64,
        "close_time": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "quote_volume": pl.Float64,
        "trade_count": pl.Int64,
    }
)

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(_OHLCV_SCHEMA.keys())
_PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")
_VOLUME_COLUMNS: Final[tuple[str, ...]] = ("volume", "quote_volume")
_NULLABLE_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    *_PRICE_COLUMNS,
    *_VOLUME_COLUMNS,
    "open_time",
    "close_time",
    "trade_count",
)
_SAMPLE_LIMIT: Final[int] = 5
_WEEKLY_TIMEFRAME: Final[str] = CanonicalTimeframe.W1.value

_logger = logging.getLogger(__name__)


class ValidationSeverity(StrEnum):
    """Severity assigned to a market-data validation finding."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single market-data validation finding.

    Attributes:
        severity: Whether the issue is an error or a warning.
        check: Stable identifier of the failed validation check.
        message: Human-readable description of the problem.
        column: Column associated with the finding, when applicable.
        row_index: First affected row index in the input frame, when known.
        count: Number of affected rows or values, when aggregated.
        value: Representative invalid value, when available.
    """

    severity: ValidationSeverity
    check: str
    message: str
    column: str | None = None
    row_index: int | None = None
    count: int | None = None
    value: object | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Immutable outcome of market-data validation.

    Attributes:
        timeframe: Timeframe used for temporal validation rules.
        row_count: Number of rows in the validated frame.
        issues: Ordered findings discovered during validation.
    """

    timeframe: Timeframe
    row_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when no error-severity issues were recorded."""
        return not self.has_errors()

    def has_errors(self) -> bool:
        """Return ``True`` when at least one error-severity issue exists."""
        return any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Return ``True`` when at least one warning-severity issue exists."""
        return any(issue.severity is ValidationSeverity.WARNING for issue in self.issues)

    def errors(self) -> tuple[ValidationIssue, ...]:
        """Return all error-severity issues in discovery order."""
        return tuple(issue for issue in self.issues if issue.severity is ValidationSeverity.ERROR)

    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Return all warning-severity issues in discovery order."""
        return tuple(issue for issue in self.issues if issue.severity is ValidationSeverity.WARNING)


class MarketDataValidator:
    """Validate downloaded OHLCV market-data frames before persistence.

    Validation never mutates the input frame. All findings are collected into
    a single ``ValidationReport``. Unsupported timeframes raise immediately
    because they are caller contract violations, not data-quality findings.

    Args:
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_logger",)

    _logger: logging.Logger

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize the validator with an optional logger dependency.

        Args:
            logger: Optional logger instance.
        """
        self._logger = logger if logger is not None else _logger

    def validate(self, df: pl.DataFrame, timeframe: Timeframe) -> ValidationReport:
        """Validate an OHLCV DataFrame for the given timeframe.

        Args:
            df: Candidate OHLCV frame. Must not be mutated by this method.
            timeframe: Canonical bar interval identifier (for example ``1m``).

        Returns:
            Immutable validation report. ``report.is_valid`` is ``True`` when
            no error-severity issues were found.

        Raises:
            ValidationError: If ``timeframe`` is not a supported CQROS interval.
        """
        interval_ms = _resolve_interval_ms(timeframe)
        issues: list[ValidationIssue] = []

        schema_ok = self._check_schema(df, issues)
        if schema_ok:
            self._check_nulls(df, issues)
            self._check_ohlc_consistency(df, issues)
            self._check_non_negative_volume(df, issues)
            self._check_duplicate_timestamps(df, issues)
            self._check_timestamp_monotonicity(df, issues)
            self._check_timeframe_alignment(df, timeframe, interval_ms, issues)
            self._check_missing_timestamps(df, interval_ms, issues)

        if df.height == 0:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check="empty_dataset",
                    message="dataset contains no rows",
                    count=0,
                )
            )

        report = ValidationReport(
            timeframe=timeframe,
            row_count=df.height,
            issues=tuple(issues),
        )
        self._logger.info(
            "Completed market-data validation",
            extra={
                "timeframe": timeframe,
                "row_count": df.height,
                "is_valid": report.is_valid,
                "error_count": len(report.errors()),
                "warning_count": len(report.warnings()),
            },
        )
        return report

    def _check_schema(self, df: pl.DataFrame, issues: list[ValidationIssue]) -> bool:
        """Validate required columns and dtypes.

        Args:
            df: Candidate frame.
            issues: Mutable issue accumulator.

        Returns:
            ``True`` when the frame has every required column with the
            expected dtype; otherwise ``False``.
        """
        schema_ok = True
        columns = set(df.columns)
        missing = [name for name in _REQUIRED_COLUMNS if name not in columns]
        if missing:
            schema_ok = False
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check="schema",
                    message=f"missing required columns: {missing}",
                    value=missing,
                    count=len(missing),
                )
            )

        unexpected = sorted(columns.difference(_REQUIRED_COLUMNS))
        if unexpected:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    check="schema",
                    message=f"unexpected columns present: {unexpected}",
                    value=unexpected,
                    count=len(unexpected),
                )
            )

        for name in _REQUIRED_COLUMNS:
            if name not in columns:
                continue
            actual = df.schema[name]
            expected = _OHLCV_SCHEMA[name]
            if actual != expected:
                schema_ok = False
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        check="schema",
                        message=(
                            f"column {name!r} has dtype {actual!s}; " f"expected {expected!s}"
                        ),
                        column=name,
                        value=str(actual),
                    )
                )
        return schema_ok

    def _check_nulls(self, df: pl.DataFrame, issues: list[ValidationIssue]) -> None:
        """Detect null and NaN values in required columns."""
        for column in _REQUIRED_COLUMNS:
            null_count = int(df.get_column(column).null_count())
            dtype = df.schema[column]
            is_float = _is_float_dtype(dtype)
            nan_count = 0
            if column in _NULLABLE_NUMERIC_COLUMNS and is_float:
                nan_frame = df.select(  # pyright: ignore[reportUnknownMemberType]
                    pl.col(column).is_nan().sum()
                )
                nan_count = int(nan_frame.item())
            total = null_count + nan_count
            if total <= 0:
                continue
            parts: list[str] = []
            if null_count > 0:
                parts.append(f"{null_count} null")
            if nan_count > 0:
                parts.append(f"{nan_count} NaN")
            invalid = pl.col(column).is_null()
            if is_float:
                invalid = invalid | pl.col(column).is_nan()
            row_index = _first_invalid_row_index(df, invalid)
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    check="nulls",
                    message=f"column {column!r} contains {' and '.join(parts)} value(s)",
                    column=column,
                    row_index=row_index,
                    count=total,
                )
            )

    def _check_ohlc_consistency(
        self,
        df: pl.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        """Validate High/Low/Open/Close cross-field consistency."""
        rules: tuple[tuple[str, pl.Expr], ...] = (
            (
                "high must be greater than or equal to low",
                pl.col("high") < pl.col("low"),
            ),
            (
                "high must be greater than or equal to open",
                pl.col("high") < pl.col("open"),
            ),
            (
                "high must be greater than or equal to close",
                pl.col("high") < pl.col("close"),
            ),
            (
                "low must be less than or equal to open",
                pl.col("low") > pl.col("open"),
            ),
            (
                "low must be less than or equal to close",
                pl.col("low") > pl.col("close"),
            ),
        )
        for message, predicate in rules:
            _append_predicate_issue(
                df,
                issues,
                check="ohlc_consistency",
                message=message,
                predicate=predicate,
                column="high" if "high" in message else "low",
            )

    def _check_non_negative_volume(
        self,
        df: pl.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        """Validate that volume fields are non-negative."""
        for column in _VOLUME_COLUMNS:
            _append_predicate_issue(
                df,
                issues,
                check="non_negative_volume",
                message=f"{column} must be greater than or equal to 0",
                predicate=pl.col(column) < 0,
                column=column,
            )
        _append_predicate_issue(
            df,
            issues,
            check="non_negative_volume",
            message="trade_count must be greater than or equal to 0",
            predicate=pl.col("trade_count") < 0,
            column="trade_count",
        )

    def _check_duplicate_timestamps(
        self,
        df: pl.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        """Detect duplicate ``open_time`` values."""
        if df.height == 0:
            return
        duplicate_mask = df.get_column("open_time").is_duplicated()
        count = int(duplicate_mask.sum())  # pyright: ignore[reportUnknownMemberType]
        if count <= 0:
            return
        row_index = _first_true_index(duplicate_mask)
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                check="duplicate_timestamps",
                message=f"found {count} duplicate open_time value(s)",
                column="open_time",
                row_index=row_index,
                count=count,
            )
        )

    def _check_timestamp_monotonicity(
        self,
        df: pl.DataFrame,
        issues: list[ValidationIssue],
    ) -> None:
        """Require strictly increasing ``open_time`` values in row order."""
        if df.height <= 1:
            return
        diffs = df.get_column("open_time").diff()
        # First diff is null; remaining values must be strictly positive.
        violation_mask = diffs.le(0).fill_null(False)
        count = int(violation_mask.sum())  # pyright: ignore[reportUnknownMemberType]
        if count <= 0:
            return
        row_index = _first_true_index(violation_mask)
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                check="timestamp_monotonicity",
                message=(
                    "open_time must be strictly increasing; "
                    f"found {count} non-increasing step(s)"
                ),
                column="open_time",
                row_index=row_index,
                count=count,
            )
        )

    def _check_timeframe_alignment(
        self,
        df: pl.DataFrame,
        timeframe: Timeframe,
        interval_ms: int,
        issues: list[ValidationIssue],
    ) -> None:
        """Require ``open_time`` values to align to the timeframe grid."""
        if df.height == 0:
            return
        open_times = df.get_column("open_time")
        if timeframe == _WEEKLY_TIMEFRAME:
            misaligned = [
                index
                for index, value in enumerate(open_times.to_list())
                if value is not None and not _is_weekly_aligned(int(value))
            ]
        else:
            remainders = open_times % interval_ms
            misaligned = [
                index
                for index, remainder in enumerate(remainders.to_list())
                if remainder is not None and int(remainder) != 0
            ]

        count = len(misaligned)
        if count <= 0:
            return
        sample = misaligned[:_SAMPLE_LIMIT]
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                check="timeframe_alignment",
                message=(
                    f"found {count} open_time value(s) not aligned to " f"timeframe {timeframe!r}"
                ),
                column="open_time",
                row_index=sample[0],
                count=count,
                value=sample,
            )
        )

    def _check_missing_timestamps(
        self,
        df: pl.DataFrame,
        interval_ms: int,
        issues: list[ValidationIssue],
    ) -> None:
        """Detect gaps larger than one timeframe interval in ``open_time``."""
        if df.height <= 1:
            return
        unique_times = df.get_column("open_time").drop_nulls().unique().sort()
        if unique_times.len() <= 1:
            return

        diffs = unique_times.diff().drop_nulls()
        gap_steps = diffs.filter(diffs > interval_ms)
        if gap_steps.len() == 0:
            return

        missing_bars = int(((gap_steps // interval_ms) - 1).sum())
        if missing_bars <= 0:
            return
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                check="missing_timestamps",
                message=(f"found {missing_bars} missing bar(s) for interval " f"{interval_ms}ms"),
                column="open_time",
                count=missing_bars,
                value=int(gap_steps.len()),
            )
        )


def _resolve_interval_ms(timeframe: Timeframe) -> int:
    """Resolve a timeframe string to its duration in milliseconds.

    Args:
        timeframe: Candidate timeframe identifier.

    Returns:
        Interval length in Unix milliseconds.

    Raises:
        ValidationError: If ``timeframe`` is unsupported.
    """
    try:
        canonical = CanonicalTimeframe(timeframe)
    except ValueError as exc:
        raise ValidationError(
            f"unsupported timeframe: {timeframe!r}",
            error_code="INGESTION-VALIDATOR-001",
            details={
                "parameter": "timeframe",
                "value": timeframe,
                "allowed": sorted(item.value for item in CanonicalTimeframe),
            },
        ) from exc
    return to_seconds(canonical) * MILLISECONDS_PER_SECOND


def _is_weekly_aligned(open_time_ms: int) -> bool:
    """Return whether ``open_time_ms`` is Monday 00:00:00.000 UTC."""
    moment = datetime.fromtimestamp(open_time_ms / MILLISECONDS_PER_SECOND, tz=UTC)
    return (
        moment.weekday() == 0
        and moment.hour == 0
        and moment.minute == 0
        and moment.second == 0
        and moment.microsecond == 0
    )


def _append_predicate_issue(
    df: pl.DataFrame,
    issues: list[ValidationIssue],
    *,
    check: str,
    message: str,
    predicate: pl.Expr,
    column: str,
) -> None:
    """Append an aggregated issue when ``predicate`` matches any rows."""
    if df.height == 0:
        return
    count = int(df.select(predicate.sum()).item())  # pyright: ignore[reportUnknownMemberType]
    if count <= 0:
        return
    row_index = _first_invalid_row_index(df, predicate)
    issues.append(
        ValidationIssue(
            severity=ValidationSeverity.ERROR,
            check=check,
            message=f"{message}; found {count} violating row(s)",
            column=column,
            row_index=row_index,
            count=count,
        )
    )


def _is_float_dtype(dtype: pl.DataType) -> bool:
    """Return whether ``dtype`` is a floating-point Polars type."""
    return dtype == pl.Float64 or dtype == pl.Float32


def _first_invalid_row_index(df: pl.DataFrame, predicate: pl.Expr) -> int | None:
    """Return the first row index where ``predicate`` is true."""
    # with_row_index creates a derived frame; the caller frame is untouched.
    indexed = df.with_row_index(name="_row_index")  # pyright: ignore[reportUnknownMemberType]
    matched = indexed.filter(predicate)  # pyright: ignore[reportUnknownMemberType]
    selected = matched.select("_row_index")  # pyright: ignore[reportUnknownMemberType]
    if selected.height == 0:
        return None
    return int(selected.item(0, 0))


def _first_true_index(mask: pl.Series) -> int | None:
    """Return the first index where a boolean series is ``True``."""
    for index, flag in enumerate(cast(list[bool | None], mask.to_list())):
        if flag:
            return index
    return None

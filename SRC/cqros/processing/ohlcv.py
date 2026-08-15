"""CQROS OHLCV processing steps and composed pipeline.

Purpose:
    Transform raw OHLCV frames into research-ready OHLCV using the Data
    Processing Framework, without filling gaps or altering market values.

Responsibilities:
    - Sort and deduplicate by ``open_time``
    - Validate schema, timestamps, OHLC consistency, and volume
    - Detect missing candles and report gap metadata
    - Compose the ordered ``OHLCVProcessingPipeline``
    - Remain free of storage, repository, feature, and interpolation logic

Dependencies:
    ``polars``, ``cqros.core.exceptions``, ``cqros.processing.base``,
    ``cqros.processing.exceptions``, and ``cqros.processing.pipeline``.

Public API:
    The processors, reports, and ``OHLCVProcessingPipeline`` listed in
    ``__all__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.processing.base import BaseProcessingStep
from cqros.processing.exceptions import ProcessingValidationError
from cqros.processing.pipeline import ProcessingPipeline

__all__ = [
    "GapDetectionReport",
    "SortByTimestampProcessor",
    "RemoveDuplicateTimestampProcessor",
    "ValidateSchemaProcessor",
    "ValidateTimestampProcessor",
    "ValidateOHLCProcessor",
    "ValidateVolumeProcessor",
    "DetectGapProcessor",
    "OHLCVProcessingPipeline",
]

_COL_SYMBOL: Final[str] = "symbol"
_COL_TIMEFRAME: Final[str] = "timeframe"
_COL_OPEN_TIME: Final[str] = "open_time"
_COL_CLOSE_TIME: Final[str] = "close_time"
_COL_OPEN: Final[str] = "open"
_COL_HIGH: Final[str] = "high"
_COL_LOW: Final[str] = "low"
_COL_CLOSE: Final[str] = "close"
_COL_VOLUME: Final[str] = "volume"
_COL_QUOTE_VOLUME: Final[str] = "quote_volume"
_COL_TRADE_COUNT: Final[str] = "trade_count"

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    _COL_SYMBOL,
    _COL_TIMEFRAME,
    _COL_OPEN_TIME,
    _COL_CLOSE_TIME,
    _COL_OPEN,
    _COL_HIGH,
    _COL_LOW,
    _COL_CLOSE,
    _COL_VOLUME,
    _COL_QUOTE_VOLUME,
    _COL_TRADE_COUNT,
)

_VERSION: Final[str] = "1.0.0"

_ERROR_SCHEMA: Final[str] = "PROCESSING-OHLCV-001"
_ERROR_OPEN_TIME_DTYPE: Final[str] = "PROCESSING-OHLCV-002"
_ERROR_CLOSE_TIME_DTYPE: Final[str] = "PROCESSING-OHLCV-003"
_ERROR_OPEN_TIME_MONOTONIC: Final[str] = "PROCESSING-OHLCV-004"
_ERROR_CLOSE_AFTER_OPEN: Final[str] = "PROCESSING-OHLCV-005"
_ERROR_FUTURE_TIMESTAMP: Final[str] = "PROCESSING-OHLCV-006"
_ERROR_OHLC: Final[str] = "PROCESSING-OHLCV-007"
_ERROR_VOLUME: Final[str] = "PROCESSING-OHLCV-008"
_ERROR_INTERVAL: Final[str] = "PROCESSING-OHLCV-009"
_ERROR_MISSING_COLUMN: Final[str] = "PROCESSING-OHLCV-010"


@dataclass(frozen=True, slots=True)
class GapDetectionReport:
    """Immutable gap-detection metadata for an OHLCV frame.

    Attributes:
        missing_intervals: Sorted missing ``open_time`` values that would
            complete the timeframe grid between the first and last bar.
        gap_count: Number of contiguous gap regions.
        largest_gap: Largest contiguous missing-bar count.
    """

    missing_intervals: tuple[int, ...]
    gap_count: int
    largest_gap: int


@dataclass(frozen=True, slots=True)
class SortByTimestampProcessor(BaseProcessingStep):
    """Sort OHLCV rows ascending by ``open_time`` with a stable sort.

    Attributes:
        name: Stable processing-step identifier.
        version: Processor formula version.
        description: Human-readable processor summary.
    """

    name: str = "sort_by_timestamp"
    version: str = _VERSION
    description: str = "Sort OHLCV rows ascending by open_time using a stable sort."

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return a stably sorted copy of ``frame`` by ``open_time``.

        Args:
            frame: Input OHLCV DataFrame. Must not be mutated.

        Returns:
            A new DataFrame sorted ascending by ``open_time``.

        Raises:
            ProcessingValidationError: If ``open_time`` is missing.
        """
        _require_column(frame, _COL_OPEN_TIME)
        return frame.sort(_COL_OPEN_TIME, maintain_order=True)


@dataclass(frozen=True, slots=True)
class RemoveDuplicateTimestampProcessor(BaseProcessingStep):
    """Remove duplicate ``open_time`` rows, keeping the first occurrence.

    After ``process``, ``removed_count`` reports how many rows were dropped.

    Attributes:
        name: Stable processing-step identifier.
        version: Processor formula version.
        description: Human-readable processor summary.
    """

    name: str = "remove_duplicate_timestamps"
    version: str = _VERSION
    description: str = "Remove duplicate open_time rows, keeping the first occurrence."
    _removed_count: int = field(default=0, init=False, repr=False, compare=False, hash=False)

    @property
    def removed_count(self) -> int:
        """Return rows removed by the most recent ``process`` call."""
        return self._removed_count

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` without duplicate ``open_time`` rows.

        Args:
            frame: Input OHLCV DataFrame. Must not be mutated.

        Returns:
            A new DataFrame with the first row retained for each ``open_time``.

        Raises:
            ProcessingValidationError: If ``open_time`` is missing.
        """
        _require_column(frame, _COL_OPEN_TIME)
        deduplicated = frame.unique(
            subset=[_COL_OPEN_TIME],
            keep="first",
            maintain_order=True,
        )
        removed = frame.height - deduplicated.height
        object.__setattr__(self, "_removed_count", removed)
        return deduplicated


@dataclass(frozen=True, slots=True)
class ValidateSchemaProcessor(BaseProcessingStep):
    """Validate that all required OHLCV columns are present.

    Attributes:
        name: Stable processing-step identifier.
        version: Processor formula version.
        description: Human-readable processor summary.
    """

    name: str = "validate_ohlcv_schema"
    version: str = _VERSION
    description: str = "Validate that required OHLCV columns are present."

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Validate required columns and return an unchanged clone.

        Args:
            frame: Input OHLCV DataFrame. Must not be mutated.

        Returns:
            A clone of ``frame`` when the schema is valid.

        Raises:
            ProcessingValidationError: If any required column is missing.
        """
        missing = [name for name in _REQUIRED_COLUMNS if name not in frame.columns]
        if missing:
            raise ProcessingValidationError(
                f"missing required OHLCV columns: {missing}",
                error_code=_ERROR_SCHEMA,
                details={
                    "missing_columns": tuple(missing),
                    "required_columns": _REQUIRED_COLUMNS,
                    "available_columns": tuple(frame.columns),
                },
            )
        return frame.clone()


@dataclass(frozen=True, slots=True)
class ValidateTimestampProcessor(BaseProcessingStep):
    """Validate OHLCV timestamp integrity.

    Checks integer dtypes, strictly increasing ``open_time``,
    ``close_time > open_time``, and optionally rejects future timestamps.

    Attributes:
        name: Stable processing-step identifier.
        version: Processor formula version.
        description: Human-readable processor summary.
        reject_future: When ``True``, reject timestamps after the reference
            time.
        reference_time_ms: Optional UTC Unix-millisecond ceiling used when
            ``reject_future`` is enabled. When omitted, the current UTC time
            is used.
    """

    name: str = "validate_ohlcv_timestamps"
    version: str = _VERSION
    description: str = (
        "Validate integer timestamps, strictly increasing open_time, "
        "and close_time greater than open_time."
    )
    reject_future: bool = False
    reference_time_ms: int | None = None

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Validate timestamp contracts and return an unchanged clone.

        Args:
            frame: Input OHLCV DataFrame. Must not be mutated.

        Returns:
            A clone of ``frame`` when timestamp validation passes.

        Raises:
            ProcessingValidationError: If timestamp contracts are violated.
        """
        _require_column(frame, _COL_OPEN_TIME)
        _require_column(frame, _COL_CLOSE_TIME)
        _require_integer_dtype(frame, _COL_OPEN_TIME, error_code=_ERROR_OPEN_TIME_DTYPE)
        _require_integer_dtype(frame, _COL_CLOSE_TIME, error_code=_ERROR_CLOSE_TIME_DTYPE)

        if frame.height > 1:
            diffs = frame.get_column(_COL_OPEN_TIME).diff()
            violation_mask = diffs.le(0).fill_null(False)
            count = int(violation_mask.sum())  # pyright: ignore[reportUnknownMemberType]
            if count > 0:
                raise ProcessingValidationError(
                    "open_time must be strictly increasing; "
                    f"found {count} non-increasing step(s)",
                    error_code=_ERROR_OPEN_TIME_MONOTONIC,
                    details={
                        "column": _COL_OPEN_TIME,
                        "violation_count": count,
                        "first_row_index": _first_true_index(violation_mask),
                    },
                )

        if frame.height > 0:
            invalid_close = pl.col(_COL_CLOSE_TIME) <= pl.col(_COL_OPEN_TIME)
            close_count = int(
                frame.select(invalid_close.sum()).item()  # pyright: ignore[reportUnknownMemberType]
            )
            if close_count > 0:
                raise ProcessingValidationError(
                    "close_time must be greater than open_time; "
                    f"found {close_count} violating row(s)",
                    error_code=_ERROR_CLOSE_AFTER_OPEN,
                    details={
                        "violation_count": close_count,
                        "first_row_index": _first_invalid_row_index(frame, invalid_close),
                    },
                )

        if self.reject_future and frame.height > 0:
            reference = (
                self.reference_time_ms if self.reference_time_ms is not None else _utc_now_ms()
            )
            future_mask = (pl.col(_COL_OPEN_TIME) > reference) | (
                pl.col(_COL_CLOSE_TIME) > reference
            )
            future_count = int(
                frame.select(future_mask.sum()).item()  # pyright: ignore[reportUnknownMemberType]
            )
            if future_count > 0:
                raise ProcessingValidationError(
                    "timestamps must not be in the future; "
                    f"found {future_count} violating row(s)",
                    error_code=_ERROR_FUTURE_TIMESTAMP,
                    details={
                        "reference_time_ms": reference,
                        "violation_count": future_count,
                        "first_row_index": _first_invalid_row_index(frame, future_mask),
                    },
                )

        return frame.clone()


@dataclass(frozen=True, slots=True)
class ValidateOHLCProcessor(BaseProcessingStep):
    """Validate OHLC cross-field consistency for every row.

    Attributes:
        name: Stable processing-step identifier.
        version: Processor formula version.
        description: Human-readable processor summary.
    """

    name: str = "validate_ohlc"
    version: str = _VERSION
    description: str = "Validate high/low consistency against open and close."

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Validate OHLC rules and return an unchanged clone.

        Args:
            frame: Input OHLCV DataFrame. Must not be mutated.

        Returns:
            A clone of ``frame`` when OHLC validation passes.

        Raises:
            ProcessingValidationError: If any OHLC consistency rule fails.
        """
        for column in (_COL_OPEN, _COL_HIGH, _COL_LOW, _COL_CLOSE):
            _require_column(frame, column)

        rules: tuple[tuple[str, pl.Expr], ...] = (
            ("high must be greater than or equal to open", pl.col(_COL_HIGH) < pl.col(_COL_OPEN)),
            (
                "high must be greater than or equal to close",
                pl.col(_COL_HIGH) < pl.col(_COL_CLOSE),
            ),
            ("high must be greater than or equal to low", pl.col(_COL_HIGH) < pl.col(_COL_LOW)),
            ("low must be less than or equal to open", pl.col(_COL_LOW) > pl.col(_COL_OPEN)),
            ("low must be less than or equal to close", pl.col(_COL_LOW) > pl.col(_COL_CLOSE)),
        )
        for message, predicate in rules:
            if frame.height == 0:
                break
            count = int(
                frame.select(predicate.sum()).item()  # pyright: ignore[reportUnknownMemberType]
            )
            if count > 0:
                raise ProcessingValidationError(
                    f"{message}; found {count} violating row(s)",
                    error_code=_ERROR_OHLC,
                    details={
                        "rule": message,
                        "violation_count": count,
                        "first_row_index": _first_invalid_row_index(frame, predicate),
                    },
                )
        return frame.clone()


@dataclass(frozen=True, slots=True)
class ValidateVolumeProcessor(BaseProcessingStep):
    """Validate non-negative volume, quote volume, and trade count.

    Attributes:
        name: Stable processing-step identifier.
        version: Processor formula version.
        description: Human-readable processor summary.
    """

    name: str = "validate_volume"
    version: str = _VERSION
    description: str = "Validate volume, quote_volume, and trade_count are non-negative."

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Validate volume contracts and return an unchanged clone.

        Args:
            frame: Input OHLCV DataFrame. Must not be mutated.

        Returns:
            A clone of ``frame`` when volume validation passes.

        Raises:
            ProcessingValidationError: If any volume field is negative.
        """
        checks: tuple[tuple[str, pl.Expr], ...] = (
            (_COL_VOLUME, pl.col(_COL_VOLUME) < 0),
            (_COL_QUOTE_VOLUME, pl.col(_COL_QUOTE_VOLUME) < 0),
            (_COL_TRADE_COUNT, pl.col(_COL_TRADE_COUNT) < 0),
        )
        for column, predicate in checks:
            _require_column(frame, column)
            if frame.height == 0:
                continue
            count = int(
                frame.select(predicate.sum()).item()  # pyright: ignore[reportUnknownMemberType]
            )
            if count > 0:
                raise ProcessingValidationError(
                    f"{column} must be greater than or equal to 0; "
                    f"found {count} violating row(s)",
                    error_code=_ERROR_VOLUME,
                    details={
                        "column": column,
                        "violation_count": count,
                        "first_row_index": _first_invalid_row_index(frame, predicate),
                    },
                )
        return frame.clone()


@dataclass(frozen=True, slots=True)
class DetectGapProcessor(BaseProcessingStep):
    """Detect missing OHLCV candles without filling them.

    After ``process``, ``last_report`` exposes gap metadata. The input frame
    is returned unchanged (as a clone).

    Attributes:
        name: Stable processing-step identifier.
        version: Processor formula version.
        description: Human-readable processor summary.
        interval_ms: Expected bar interval in Unix milliseconds.
    """

    name: str = "detect_ohlcv_gaps"
    version: str = _VERSION
    description: str = "Detect missing OHLCV candles without filling gaps."
    interval_ms: int = field(kw_only=True)
    _last_report: GapDetectionReport | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        """Validate constructor invariants including ``interval_ms``.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                a non-positive ``interval_ms``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseProcessingStep.__post_init__(self)
        interval = cast(object, self.interval_ms)
        if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
            raise ValidationError(
                "interval_ms must be a positive integer",
                error_code=_ERROR_INTERVAL,
                details={"parameter": "interval_ms", "value": self.interval_ms},
            )

    @property
    def last_report(self) -> GapDetectionReport | None:
        """Return gap metadata from the most recent ``process`` call."""
        return self._last_report

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Detect gaps, store metadata, and return an unchanged clone.

        Args:
            frame: Input OHLCV DataFrame. Must not be mutated.

        Returns:
            A clone of ``frame``. Rows are never filled or removed.

        Raises:
            ProcessingValidationError: If ``open_time`` is missing.
        """
        _require_column(frame, _COL_OPEN_TIME)
        report = _detect_gaps(frame, interval_ms=self.interval_ms)
        object.__setattr__(self, "_last_report", report)
        return frame.clone()


class OHLCVProcessingPipeline:
    """Compose OHLCV processors into the research-ready preparation order.

    Execution order:

    1. ``ValidateSchemaProcessor``
    2. ``SortByTimestampProcessor``
    3. ``RemoveDuplicateTimestampProcessor``
    4. ``ValidateTimestampProcessor``
    5. ``ValidateOHLCProcessor``
    6. ``ValidateVolumeProcessor``
    7. ``DetectGapProcessor``

    Args:
        interval_ms: Expected bar interval in Unix milliseconds for gap
            detection.
        reject_future: Forwarded to ``ValidateTimestampProcessor``.
        reference_time_ms: Optional future-timestamp ceiling in UTC Unix
            milliseconds.
    """

    __slots__ = (
        "_dedupe",
        "_gaps",
        "_pipeline",
    )

    def __init__(
        self,
        *,
        interval_ms: int,
        reject_future: bool = False,
        reference_time_ms: int | None = None,
    ) -> None:
        """Initialize the composed OHLCV processing pipeline.

        Args:
            interval_ms: Expected bar interval in Unix milliseconds.
            reject_future: Reject timestamps after the reference time.
            reference_time_ms: Optional UTC Unix-millisecond ceiling.
        """
        self._dedupe = RemoveDuplicateTimestampProcessor()
        self._gaps = DetectGapProcessor(interval_ms=interval_ms)
        self._pipeline = ProcessingPipeline(
            (
                ValidateSchemaProcessor(),
                SortByTimestampProcessor(),
                self._dedupe,
                ValidateTimestampProcessor(
                    reject_future=reject_future,
                    reference_time_ms=reference_time_ms,
                ),
                ValidateOHLCProcessor(),
                ValidateVolumeProcessor(),
                self._gaps,
            )
        )

    @property
    def steps(self) -> tuple[BaseProcessingStep, ...]:
        """Return the ordered processing steps."""
        return cast(tuple[BaseProcessingStep, ...], self._pipeline.steps)

    @property
    def removed_duplicate_count(self) -> int:
        """Return rows removed by the dedupe step in the last ``run``."""
        return self._dedupe.removed_count

    @property
    def gap_report(self) -> GapDetectionReport | None:
        """Return gap metadata from the last ``run``, if available."""
        return self._gaps.last_report

    def run(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Execute the OHLCV processing sequence against ``frame``.

        Args:
            frame: Raw OHLCV DataFrame. Must not be mutated.

        Returns:
            A new research-ready OHLCV DataFrame.
        """
        return self._pipeline.run(frame)


def _require_column(frame: pl.DataFrame, column: str) -> None:
    """Raise ``ProcessingValidationError`` when ``column`` is absent."""
    if column not in frame.columns:
        raise ProcessingValidationError(
            f"required column missing: {column}",
            error_code=_ERROR_MISSING_COLUMN,
            details={
                "required_column": column,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_integer_dtype(frame: pl.DataFrame, column: str, *, error_code: str) -> None:
    """Raise when ``column`` is not an integer Polars dtype."""
    dtype = frame.schema[column]
    if not dtype.is_integer():
        raise ProcessingValidationError(
            f"column {column!r} must have an integer dtype; found {dtype!s}",
            error_code=error_code,
            details={"column": column, "dtype": str(dtype)},
        )


def _detect_gaps(frame: pl.DataFrame, *, interval_ms: int) -> GapDetectionReport:
    """Compute gap metadata for sorted unique ``open_time`` values."""
    if frame.height <= 1:
        return GapDetectionReport(missing_intervals=(), gap_count=0, largest_gap=0)

    open_times = frame.get_column(_COL_OPEN_TIME).drop_nulls().unique().sort()
    if open_times.len() <= 1:
        return GapDetectionReport(missing_intervals=(), gap_count=0, largest_gap=0)

    values = cast(list[int], open_times.to_list())
    missing: list[int] = []
    gap_count = 0
    largest_gap = 0

    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]
        delta = current - previous
        if delta <= interval_ms:
            continue
        missing_count = (delta // interval_ms) - 1
        if missing_count <= 0:
            continue
        gap_count += 1
        largest_gap = max(largest_gap, missing_count)
        for offset in range(1, missing_count + 1):
            missing.append(previous + (offset * interval_ms))

    return GapDetectionReport(
        missing_intervals=tuple(missing),
        gap_count=gap_count,
        largest_gap=largest_gap,
    )


def _utc_now_ms() -> int:
    """Return the current UTC Unix time in milliseconds."""
    return int(datetime.now(tz=UTC).timestamp() * 1000)


def _first_invalid_row_index(frame: pl.DataFrame, predicate: pl.Expr) -> int | None:
    """Return the first row index where ``predicate`` is true."""
    indexed = frame.with_row_index(name="_row_index")  # pyright: ignore[reportUnknownMemberType]
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

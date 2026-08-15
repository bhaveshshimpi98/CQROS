"""CQROS OHLCV cleaning for research-quality candle frames.

Purpose:
    Transform validated OHLCV frames into research-quality OHLCV by removing
    structurally invalid rows, without filling gaps or altering market values.

Responsibilities:
    - Remove exact duplicate rows
    - Remove rows with null mandatory values or NaN in mandatory numeric columns
    - Remove rows with non-positive or inconsistent OHLC prices
    - Remove rows with negative volume, quote volume, or trade count
    - Remove rows with non-positive candle duration
    - Sort remaining rows by ``open_time`` with a stable order
    - Emit an immutable ``CleaningReport`` describing removals

Dependencies:
    ``polars``, ``cqros.processing.exceptions``.

Public API:
    ``CleaningReport``, ``OHLCVCleaner``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.processing.exceptions import ProcessingValidationError

__all__ = [
    "CleaningReport",
    "OHLCVCleaner",
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

_MANDATORY_COLUMNS: Final[tuple[str, ...]] = (
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

_MANDATORY_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    _COL_OPEN,
    _COL_HIGH,
    _COL_LOW,
    _COL_CLOSE,
    _COL_VOLUME,
    _COL_QUOTE_VOLUME,
    _COL_TRADE_COUNT,
)

_ERROR_SCHEMA: Final[str] = "PROCESSING-CLEANING-OHLCV-001"

_WARN_DUPLICATES: Final[str] = "Removed {count} exact duplicate row(s)."
_WARN_NULLS: Final[str] = "Removed {count} row(s) with null or NaN mandatory values."
_WARN_PRICES: Final[str] = "Removed {count} row(s) with invalid OHLC prices."
_WARN_VOLUME: Final[str] = "Removed {count} row(s) with invalid volume fields."
_WARN_TRADE_COUNT: Final[str] = "Removed {count} row(s) with invalid trade_count."
_WARN_TIMESTAMPS: Final[str] = "Removed {count} row(s) with invalid timestamps."


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """Immutable summary of an OHLCV cleaning pass.

    Attributes:
        rows_before: Row count of the input frame before cleaning.
        rows_after: Row count of the cleaned frame.
        duplicates_removed: Exact duplicate rows removed.
        null_rows_removed: Rows removed for null mandatory values or NaN in
            mandatory numeric columns.
        invalid_price_rows_removed: Rows removed for non-positive or
            inconsistent OHLC prices.
        invalid_volume_rows_removed: Rows removed for negative ``volume`` or
            ``quote_volume``.
        invalid_trade_count_rows_removed: Rows removed for negative
            ``trade_count``.
        invalid_timestamp_rows_removed: Rows removed where
            ``close_time <= open_time``.
        warnings: Deterministic human-readable warnings describing removals.
    """

    rows_before: int
    rows_after: int
    duplicates_removed: int
    null_rows_removed: int
    invalid_price_rows_removed: int
    invalid_volume_rows_removed: int
    invalid_trade_count_rows_removed: int
    invalid_timestamp_rows_removed: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OHLCVCleaner:
    """Deterministic cleaner that produces research-quality OHLCV frames.

    Applies structural cleaning rules only. Does not fill missing candles,
    interpolate or smooth prices, clip outliers, modify timestamps, access
    repositories, write storage, or compute features or factors.
    """

    def clean(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
        """Return a cleaned OHLCV frame and an immutable cleaning report.

        Args:
            frame: Validated OHLCV DataFrame. Must not be mutated.

        Returns:
            A tuple of ``(cleaned_frame, CleaningReport)``. The cleaned frame
            is sorted ascending by ``open_time`` with stable relative order.

        Raises:
            ProcessingValidationError: If any mandatory column is missing.
        """
        _require_mandatory_columns(frame)
        rows_before = frame.height

        after_duplicates = frame.unique(maintain_order=True)
        duplicates_removed = rows_before - after_duplicates.height

        after_nulls = after_duplicates.filter(~_null_or_nan_row_predicate())
        null_rows_removed = after_duplicates.height - after_nulls.height

        price_predicate = (
            (pl.col(_COL_OPEN) > 0)
            & (pl.col(_COL_HIGH) > 0)
            & (pl.col(_COL_LOW) > 0)
            & (pl.col(_COL_CLOSE) > 0)
            & (
                pl.col(_COL_HIGH)
                >= pl.max_horizontal(pl.col(_COL_OPEN), pl.col(_COL_CLOSE), pl.col(_COL_LOW))
            )
            & (
                pl.col(_COL_LOW)
                <= pl.min_horizontal(pl.col(_COL_OPEN), pl.col(_COL_CLOSE), pl.col(_COL_HIGH))
            )
        )
        after_prices = after_nulls.filter(price_predicate)
        invalid_price_rows_removed = after_nulls.height - after_prices.height

        volume_predicate = (pl.col(_COL_VOLUME) >= 0) & (pl.col(_COL_QUOTE_VOLUME) >= 0)
        after_volume = after_prices.filter(volume_predicate)
        invalid_volume_rows_removed = after_prices.height - after_volume.height

        after_trade_count = after_volume.filter(pl.col(_COL_TRADE_COUNT) >= 0)
        invalid_trade_count_rows_removed = after_volume.height - after_trade_count.height

        after_timestamps = after_trade_count.filter(
            pl.col(_COL_CLOSE_TIME) > pl.col(_COL_OPEN_TIME)
        )
        invalid_timestamp_rows_removed = after_trade_count.height - after_timestamps.height

        cleaned = after_timestamps.sort(_COL_OPEN_TIME, maintain_order=True)
        warnings = _build_warnings(
            duplicates_removed=duplicates_removed,
            null_rows_removed=null_rows_removed,
            invalid_price_rows_removed=invalid_price_rows_removed,
            invalid_volume_rows_removed=invalid_volume_rows_removed,
            invalid_trade_count_rows_removed=invalid_trade_count_rows_removed,
            invalid_timestamp_rows_removed=invalid_timestamp_rows_removed,
        )
        report = CleaningReport(
            rows_before=rows_before,
            rows_after=cleaned.height,
            duplicates_removed=duplicates_removed,
            null_rows_removed=null_rows_removed,
            invalid_price_rows_removed=invalid_price_rows_removed,
            invalid_volume_rows_removed=invalid_volume_rows_removed,
            invalid_trade_count_rows_removed=invalid_trade_count_rows_removed,
            invalid_timestamp_rows_removed=invalid_timestamp_rows_removed,
            warnings=warnings,
        )
        return cleaned, report


def _null_or_nan_row_predicate() -> pl.Expr:
    """Return a row mask that is true when any mandatory value is null or NaN.

    Null checks cover every mandatory column. NaN checks cover mandatory
    numeric columns only (``open``, ``high``, ``low``, ``close``, ``volume``,
    ``quote_volume``, ``trade_count``).
    """
    predicates: list[pl.Expr] = [pl.col(column).is_null() for column in _MANDATORY_COLUMNS]
    predicates.extend(pl.col(column).is_nan() for column in _MANDATORY_NUMERIC_COLUMNS)
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = combined | predicate
    return combined


def _require_mandatory_columns(frame: pl.DataFrame) -> None:
    """Raise when any mandatory OHLCV column is absent."""
    missing = [name for name in _MANDATORY_COLUMNS if name not in frame.columns]
    if missing:
        raise ProcessingValidationError(
            f"missing required OHLCV columns: {missing}",
            error_code=_ERROR_SCHEMA,
            details={
                "missing_columns": tuple(missing),
                "required_columns": _MANDATORY_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _build_warnings(
    *,
    duplicates_removed: int,
    null_rows_removed: int,
    invalid_price_rows_removed: int,
    invalid_volume_rows_removed: int,
    invalid_trade_count_rows_removed: int,
    invalid_timestamp_rows_removed: int,
) -> tuple[str, ...]:
    """Return deterministic warning messages for non-zero removal counts."""
    warnings: list[str] = []
    if duplicates_removed > 0:
        warnings.append(_WARN_DUPLICATES.format(count=duplicates_removed))
    if null_rows_removed > 0:
        warnings.append(_WARN_NULLS.format(count=null_rows_removed))
    if invalid_price_rows_removed > 0:
        warnings.append(_WARN_PRICES.format(count=invalid_price_rows_removed))
    if invalid_volume_rows_removed > 0:
        warnings.append(_WARN_VOLUME.format(count=invalid_volume_rows_removed))
    if invalid_trade_count_rows_removed > 0:
        warnings.append(_WARN_TRADE_COUNT.format(count=invalid_trade_count_rows_removed))
    if invalid_timestamp_rows_removed > 0:
        warnings.append(_WARN_TIMESTAMPS.format(count=invalid_timestamp_rows_removed))
    return tuple(warnings)

"""CQROS taker-volume cleaning for research-quality buy/sell frames.

Purpose:
    Transform validated taker buy/sell volume frames into research-quality
    taker volume by removing structurally invalid rows and recomputing
    ``buy_sell_ratio``, without filling gaps or altering volumes.

Responsibilities:
    - Remove duplicate ``timestamp`` rows (keep first occurrence)
    - Remove rows with null or NaN ``buy_volume`` or ``sell_volume``
    - Remove rows with invalid ``timestamp`` values
    - Remove rows with negative ``buy_volume`` or ``sell_volume``
    - Recompute ``buy_sell_ratio`` as ``buy_volume / sell_volume``
    - Sort remaining rows by ``timestamp`` with a stable order
    - Emit an immutable ``CleaningReport`` describing removals

Dependencies:
    ``polars``, ``cqros.processing.exceptions``,
    ``cqros.processing.cleaning.ohlcv.CleaningReport``.

Public API:
    ``TakerVolumeCleaner``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.processing.cleaning.ohlcv import CleaningReport
from cqros.processing.exceptions import ProcessingValidationError

__all__ = [
    "TakerVolumeCleaner",
]

_COL_SYMBOL: Final[str] = "symbol"
_COL_TIMESTAMP: Final[str] = "timestamp"
_COL_BUY_VOLUME: Final[str] = "buy_volume"
_COL_SELL_VOLUME: Final[str] = "sell_volume"
_COL_BUY_SELL_RATIO: Final[str] = "buy_sell_ratio"

_MANDATORY_COLUMNS: Final[tuple[str, ...]] = (
    _COL_SYMBOL,
    _COL_TIMESTAMP,
    _COL_BUY_VOLUME,
    _COL_SELL_VOLUME,
)

_ERROR_SCHEMA: Final[str] = "PROCESSING-CLEANING-TAKER-VOLUME-001"

_WARN_DUPLICATES: Final[str] = "Removed {count} duplicate timestamp row(s)."
_WARN_NULLS: Final[str] = "Removed {count} row(s) with null or NaN buy_volume or sell_volume."
_WARN_TIMESTAMPS: Final[str] = "Removed {count} row(s) with invalid timestamps."
_WARN_VOLUMES: Final[str] = "Removed {count} row(s) with negative buy_volume or sell_volume."


@dataclass(frozen=True, slots=True)
class TakerVolumeCleaner:
    """Deterministic cleaner that produces research-quality taker-volume frames.

    Applies structural cleaning rules only. Does not interpolate or smooth
    volumes, fill missing observations, clip outliers, modify timestamps,
    access repositories, write storage, or compute features or factors.

    Zero volumes are valid. When ``sell_volume`` is zero, ``buy_sell_ratio``
    is set to null and the row is retained.
    """

    def clean(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
        """Return a cleaned taker-volume frame and an immutable cleaning report.

        Args:
            frame: Validated taker-volume DataFrame. Must not be mutated.

        Returns:
            A tuple of ``(cleaned_frame, CleaningReport)``. The cleaned frame
            is sorted ascending by ``timestamp`` with stable relative order.
            ``buy_sell_ratio`` is always recomputed from cleaned volumes.

            ``CleaningReport.invalid_volume_rows_removed`` counts negative
            ``buy_volume`` and ``sell_volume`` removals. Price and trade-count
            counters remain zero for taker-volume frames.

        Raises:
            ProcessingValidationError: If any mandatory column is missing.
        """
        _require_mandatory_columns(frame)
        rows_before = frame.height

        after_duplicates = frame.unique(
            subset=[_COL_TIMESTAMP],
            maintain_order=True,
        )
        duplicates_removed = rows_before - after_duplicates.height

        after_buy_nulls = after_duplicates.filter(~_null_or_nan_predicate(_COL_BUY_VOLUME))
        buy_null_rows_removed = after_duplicates.height - after_buy_nulls.height

        after_sell_nulls = after_buy_nulls.filter(~_null_or_nan_predicate(_COL_SELL_VOLUME))
        sell_null_rows_removed = after_buy_nulls.height - after_sell_nulls.height
        null_rows_removed = buy_null_rows_removed + sell_null_rows_removed

        after_timestamps = after_sell_nulls.filter(_valid_timestamp_predicate())
        invalid_timestamp_rows_removed = after_sell_nulls.height - after_timestamps.height

        after_buy_volume = after_timestamps.filter(pl.col(_COL_BUY_VOLUME) >= 0)
        negative_buy_rows_removed = after_timestamps.height - after_buy_volume.height

        after_sell_volume = after_buy_volume.filter(pl.col(_COL_SELL_VOLUME) >= 0)
        negative_sell_rows_removed = after_buy_volume.height - after_sell_volume.height
        invalid_volume_rows_removed = negative_buy_rows_removed + negative_sell_rows_removed

        with_ratio = after_sell_volume.with_columns(  # pyright: ignore[reportUnknownMemberType]
            _recomputed_buy_sell_ratio().alias(_COL_BUY_SELL_RATIO)
        )
        cleaned = with_ratio.sort(_COL_TIMESTAMP, maintain_order=True)
        warnings = _build_warnings(
            duplicates_removed=duplicates_removed,
            null_rows_removed=null_rows_removed,
            invalid_timestamp_rows_removed=invalid_timestamp_rows_removed,
            invalid_volume_rows_removed=invalid_volume_rows_removed,
        )
        report = CleaningReport(
            rows_before=rows_before,
            rows_after=cleaned.height,
            duplicates_removed=duplicates_removed,
            null_rows_removed=null_rows_removed,
            invalid_price_rows_removed=0,
            invalid_volume_rows_removed=invalid_volume_rows_removed,
            invalid_trade_count_rows_removed=0,
            invalid_timestamp_rows_removed=invalid_timestamp_rows_removed,
            warnings=warnings,
        )
        return cleaned, report


def _null_or_nan_predicate(column: str) -> pl.Expr:
    """Return a row mask that is true when ``column`` is null or NaN."""
    return pl.col(column).is_null() | pl.col(column).is_nan()


def _valid_timestamp_predicate() -> pl.Expr:
    """Return a row mask that is true for valid ``timestamp`` values.

    Valid timestamps are non-null integers strictly greater than zero.
    Fractional timestamps are rejected when the column dtype is floating.
    """
    timestamp = pl.col(_COL_TIMESTAMP)
    positive = timestamp.is_not_null() & (timestamp > 0)
    integer_valued = timestamp == timestamp.cast(pl.Int64)
    return positive & integer_valued


def _recomputed_buy_sell_ratio() -> pl.Expr:
    """Return ``buy_volume / sell_volume``, or null when ``sell_volume`` is 0."""
    buy = pl.col(_COL_BUY_VOLUME)
    sell = pl.col(_COL_SELL_VOLUME)
    return pl.when(sell == 0).then(None).otherwise(buy / sell)


def _require_mandatory_columns(frame: pl.DataFrame) -> None:
    """Raise when any mandatory taker-volume column is absent."""
    missing = [name for name in _MANDATORY_COLUMNS if name not in frame.columns]
    if missing:
        raise ProcessingValidationError(
            f"missing required taker volume columns: {missing}",
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
    invalid_timestamp_rows_removed: int,
    invalid_volume_rows_removed: int,
) -> tuple[str, ...]:
    """Return deterministic warning messages for non-zero removal counts."""
    warnings: list[str] = []
    if duplicates_removed > 0:
        warnings.append(_WARN_DUPLICATES.format(count=duplicates_removed))
    if null_rows_removed > 0:
        warnings.append(_WARN_NULLS.format(count=null_rows_removed))
    if invalid_timestamp_rows_removed > 0:
        warnings.append(_WARN_TIMESTAMPS.format(count=invalid_timestamp_rows_removed))
    if invalid_volume_rows_removed > 0:
        warnings.append(_WARN_VOLUMES.format(count=invalid_volume_rows_removed))
    return tuple(warnings)

"""CQROS funding-rate cleaning for research-quality funding frames.

Purpose:
    Transform validated funding-rate frames into research-quality funding by
    removing structurally invalid rows, without filling gaps or altering rates.

Responsibilities:
    - Remove duplicate ``funding_time`` rows (keep first occurrence)
    - Remove rows with null or NaN ``funding_rate``
    - Remove rows with invalid ``funding_time`` values
    - Remove rows with invalid ``mark_price`` when the column is present
    - Sort remaining rows by ``funding_time`` with a stable order
    - Emit an immutable ``CleaningReport`` describing removals

Dependencies:
    ``polars``, ``cqros.processing.exceptions``,
    ``cqros.processing.cleaning.ohlcv.CleaningReport``.

Public API:
    ``FundingCleaner``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.processing.cleaning.ohlcv import CleaningReport
from cqros.processing.exceptions import ProcessingValidationError

__all__ = [
    "FundingCleaner",
]

_COL_SYMBOL: Final[str] = "symbol"
_COL_FUNDING_TIME: Final[str] = "funding_time"
_COL_FUNDING_RATE: Final[str] = "funding_rate"
_COL_MARK_PRICE: Final[str] = "mark_price"

_MANDATORY_COLUMNS: Final[tuple[str, ...]] = (
    _COL_SYMBOL,
    _COL_FUNDING_TIME,
    _COL_FUNDING_RATE,
)

_ERROR_SCHEMA: Final[str] = "PROCESSING-CLEANING-FUNDING-001"

_WARN_DUPLICATES: Final[str] = "Removed {count} duplicate funding timestamp row(s)."
_WARN_NULLS: Final[str] = "Removed {count} row(s) with null or NaN funding_rate."
_WARN_TIMESTAMPS: Final[str] = "Removed {count} row(s) with invalid timestamps."
_WARN_MARK_PRICE: Final[str] = "Removed {count} row(s) with invalid mark_price."


@dataclass(frozen=True, slots=True)
class FundingCleaner:
    """Deterministic cleaner that produces research-quality funding frames.

    Applies structural cleaning rules only. Does not interpolate or smooth
    rates, fill missing settlements, clip outliers, modify timestamps, access
    repositories, write storage, or compute features or factors.

    Negative ``funding_rate`` values are valid and are retained.
    """

    def clean(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
        """Return a cleaned funding frame and an immutable cleaning report.

        Args:
            frame: Validated funding DataFrame. Must not be mutated.

        Returns:
            A tuple of ``(cleaned_frame, CleaningReport)``. The cleaned frame
            is sorted ascending by ``funding_time`` with stable relative order.

            ``CleaningReport.invalid_price_rows_removed`` counts invalid
            ``mark_price`` removals. Volume and trade-count counters remain
            zero for funding frames.

        Raises:
            ProcessingValidationError: If any mandatory column is missing.
        """
        _require_mandatory_columns(frame)
        rows_before = frame.height

        after_duplicates = frame.unique(
            subset=[_COL_FUNDING_TIME],
            maintain_order=True,
        )
        duplicates_removed = rows_before - after_duplicates.height

        after_nulls = after_duplicates.filter(~_null_or_nan_funding_rate_predicate())
        null_rows_removed = after_duplicates.height - after_nulls.height

        after_timestamps = after_nulls.filter(_valid_timestamp_predicate())
        invalid_timestamp_rows_removed = after_nulls.height - after_timestamps.height

        if _COL_MARK_PRICE in after_timestamps.columns:
            after_mark_price = after_timestamps.filter(_valid_mark_price_predicate())
            invalid_price_rows_removed = after_timestamps.height - after_mark_price.height
        else:
            after_mark_price = after_timestamps
            invalid_price_rows_removed = 0

        cleaned = after_mark_price.sort(_COL_FUNDING_TIME, maintain_order=True)
        warnings = _build_warnings(
            duplicates_removed=duplicates_removed,
            null_rows_removed=null_rows_removed,
            invalid_timestamp_rows_removed=invalid_timestamp_rows_removed,
            invalid_price_rows_removed=invalid_price_rows_removed,
        )
        report = CleaningReport(
            rows_before=rows_before,
            rows_after=cleaned.height,
            duplicates_removed=duplicates_removed,
            null_rows_removed=null_rows_removed,
            invalid_price_rows_removed=invalid_price_rows_removed,
            invalid_volume_rows_removed=0,
            invalid_trade_count_rows_removed=0,
            invalid_timestamp_rows_removed=invalid_timestamp_rows_removed,
            warnings=warnings,
        )
        return cleaned, report


def _null_or_nan_funding_rate_predicate() -> pl.Expr:
    """Return a row mask that is true when ``funding_rate`` is null or NaN."""
    return pl.col(_COL_FUNDING_RATE).is_null() | pl.col(_COL_FUNDING_RATE).is_nan()


def _valid_timestamp_predicate() -> pl.Expr:
    """Return a row mask that is true for valid ``funding_time`` values.

    Valid timestamps are non-null, finite integers strictly greater than zero.
    Fractional timestamps are rejected when the column dtype is floating.
    """
    funding_time = pl.col(_COL_FUNDING_TIME)
    positive = funding_time.is_not_null() & (funding_time > 0)
    integer_valued = funding_time == funding_time.cast(pl.Int64)
    return positive & integer_valued


def _valid_mark_price_predicate() -> pl.Expr:
    """Return a row mask that is true for retained ``mark_price`` values.

    Null ``mark_price`` is retained. NaN and negative values are removed.
    """
    mark_price = pl.col(_COL_MARK_PRICE)
    return mark_price.is_null() | ((~mark_price.is_nan()) & (mark_price >= 0))


def _require_mandatory_columns(frame: pl.DataFrame) -> None:
    """Raise when any mandatory funding column is absent."""
    missing = [name for name in _MANDATORY_COLUMNS if name not in frame.columns]
    if missing:
        raise ProcessingValidationError(
            f"missing required funding columns: {missing}",
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
    invalid_price_rows_removed: int,
) -> tuple[str, ...]:
    """Return deterministic warning messages for non-zero removal counts."""
    warnings: list[str] = []
    if duplicates_removed > 0:
        warnings.append(_WARN_DUPLICATES.format(count=duplicates_removed))
    if null_rows_removed > 0:
        warnings.append(_WARN_NULLS.format(count=null_rows_removed))
    if invalid_timestamp_rows_removed > 0:
        warnings.append(_WARN_TIMESTAMPS.format(count=invalid_timestamp_rows_removed))
    if invalid_price_rows_removed > 0:
        warnings.append(_WARN_MARK_PRICE.format(count=invalid_price_rows_removed))
    return tuple(warnings)

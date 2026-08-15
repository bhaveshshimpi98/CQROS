"""CQROS open-interest cleaning for research-quality OI frames.

Purpose:
    Transform validated open-interest frames into research-quality open
    interest by removing structurally invalid rows, without filling gaps or
    altering values.

Responsibilities:
    - Remove duplicate ``timestamp`` rows (keep first occurrence)
    - Remove rows with null or NaN ``open_interest``
    - Remove rows with invalid ``timestamp`` values
    - Remove rows with negative ``open_interest``
    - Sort remaining rows by ``timestamp`` with a stable order
    - Emit an immutable ``CleaningReport`` describing removals

Dependencies:
    ``polars``, ``cqros.processing.exceptions``,
    ``cqros.processing.cleaning.ohlcv.CleaningReport``.

Public API:
    ``OpenInterestCleaner``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.processing.cleaning.ohlcv import CleaningReport
from cqros.processing.exceptions import ProcessingValidationError

__all__ = [
    "OpenInterestCleaner",
]

_COL_SYMBOL: Final[str] = "symbol"
_COL_TIMESTAMP: Final[str] = "timestamp"
_COL_OPEN_INTEREST: Final[str] = "open_interest"

_MANDATORY_COLUMNS: Final[tuple[str, ...]] = (
    _COL_SYMBOL,
    _COL_TIMESTAMP,
    _COL_OPEN_INTEREST,
)

_ERROR_SCHEMA: Final[str] = "PROCESSING-CLEANING-OPEN-INTEREST-001"

_WARN_DUPLICATES: Final[str] = "Removed {count} duplicate timestamp row(s)."
_WARN_NULLS: Final[str] = "Removed {count} row(s) with null or NaN open_interest."
_WARN_TIMESTAMPS: Final[str] = "Removed {count} row(s) with invalid timestamps."
_WARN_OPEN_INTEREST: Final[str] = "Removed {count} row(s) with negative open_interest."


@dataclass(frozen=True, slots=True)
class OpenInterestCleaner:
    """Deterministic cleaner that produces research-quality open-interest frames.

    Applies structural cleaning rules only. Does not interpolate or smooth
    values, fill missing observations, clip outliers, modify timestamps,
    access repositories, write storage, or compute features or factors.

    Zero ``open_interest`` is valid and is retained.
    """

    def clean(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
        """Return a cleaned open-interest frame and an immutable cleaning report.

        Args:
            frame: Validated open-interest DataFrame. Must not be mutated.

        Returns:
            A tuple of ``(cleaned_frame, CleaningReport)``. The cleaned frame
            is sorted ascending by ``timestamp`` with stable relative order.

            ``CleaningReport.invalid_volume_rows_removed`` counts negative
            ``open_interest`` removals. Price and trade-count counters remain
            zero for open-interest frames.

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

        after_nulls = after_duplicates.filter(~_null_or_nan_open_interest_predicate())
        null_rows_removed = after_duplicates.height - after_nulls.height

        after_timestamps = after_nulls.filter(_valid_timestamp_predicate())
        invalid_timestamp_rows_removed = after_nulls.height - after_timestamps.height

        after_open_interest = after_timestamps.filter(pl.col(_COL_OPEN_INTEREST) >= 0)
        invalid_volume_rows_removed = after_timestamps.height - after_open_interest.height

        cleaned = after_open_interest.sort(_COL_TIMESTAMP, maintain_order=True)
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


def _null_or_nan_open_interest_predicate() -> pl.Expr:
    """Return a row mask that is true when ``open_interest`` is null or NaN."""
    return pl.col(_COL_OPEN_INTEREST).is_null() | pl.col(_COL_OPEN_INTEREST).is_nan()


def _valid_timestamp_predicate() -> pl.Expr:
    """Return a row mask that is true for valid ``timestamp`` values.

    Valid timestamps are non-null integers strictly greater than zero.
    Fractional timestamps are rejected when the column dtype is floating.
    """
    timestamp = pl.col(_COL_TIMESTAMP)
    positive = timestamp.is_not_null() & (timestamp > 0)
    integer_valued = timestamp == timestamp.cast(pl.Int64)
    return positive & integer_valued


def _require_mandatory_columns(frame: pl.DataFrame) -> None:
    """Raise when any mandatory open-interest column is absent."""
    missing = [name for name in _MANDATORY_COLUMNS if name not in frame.columns]
    if missing:
        raise ProcessingValidationError(
            f"missing required open interest columns: {missing}",
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
        warnings.append(_WARN_OPEN_INTEREST.format(count=invalid_volume_rows_removed))
    return tuple(warnings)

"""CQROS long/short ratio cleaning for research-quality positioning frames.

Purpose:
    Transform validated long/short ratio frames into research-quality
    positioning data by removing structurally invalid rows, without filling
    gaps or altering ratio values.

Responsibilities:
    - Remove duplicate ``timestamp`` rows (keep first occurrence)
    - Remove rows with null or NaN ``long_account``, ``short_account``, or
      ``long_short_ratio``
    - Remove rows with invalid ``timestamp`` values
    - Remove rows with negative ratio fields
    - Sort remaining rows by ``timestamp`` with a stable order
    - Emit an immutable ``CleaningReport`` describing removals

Notes:
    Operates on the common schema shared by global account, top-trader
    account, and top-trader position long/short datasets.

Dependencies:
    ``polars``, ``cqros.processing.exceptions``,
    ``cqros.processing.cleaning.ohlcv.CleaningReport``.

Public API:
    ``LongShortCleaner``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.processing.cleaning.ohlcv import CleaningReport
from cqros.processing.exceptions import ProcessingValidationError

__all__ = [
    "LongShortCleaner",
]

_COL_SYMBOL: Final[str] = "symbol"
_COL_TIMESTAMP: Final[str] = "timestamp"
_COL_LONG_ACCOUNT: Final[str] = "long_account"
_COL_SHORT_ACCOUNT: Final[str] = "short_account"
_COL_LONG_SHORT_RATIO: Final[str] = "long_short_ratio"

_MANDATORY_COLUMNS: Final[tuple[str, ...]] = (
    _COL_SYMBOL,
    _COL_TIMESTAMP,
    _COL_LONG_ACCOUNT,
    _COL_SHORT_ACCOUNT,
    _COL_LONG_SHORT_RATIO,
)

_RATIO_COLUMNS: Final[tuple[str, ...]] = (
    _COL_LONG_ACCOUNT,
    _COL_SHORT_ACCOUNT,
    _COL_LONG_SHORT_RATIO,
)

_ERROR_SCHEMA: Final[str] = "PROCESSING-CLEANING-LONG-SHORT-001"

_WARN_DUPLICATES: Final[str] = "Removed {count} duplicate timestamp row(s)."
_WARN_NULLS: Final[str] = "Removed {count} row(s) with null or NaN long/short ratio fields."
_WARN_TIMESTAMPS: Final[str] = "Removed {count} row(s) with invalid timestamps."
_WARN_RATIOS: Final[str] = "Removed {count} row(s) with invalid long/short ratios."


@dataclass(frozen=True, slots=True)
class LongShortCleaner:
    """Deterministic cleaner for research-quality long/short ratio frames.

    Applies structural cleaning rules only. Does not recompute, normalize,
    interpolate, or smooth ratios; does not modify timestamps, access
    repositories, write storage, or compute features or factors.

    Supports the common schema used by global account, top-trader account,
    and top-trader position long/short datasets. Zero ratio fields are valid
    and are retained.
    """

    def clean(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
        """Return a cleaned long/short frame and an immutable cleaning report.

        Args:
            frame: Validated long/short ratio DataFrame. Must not be mutated.

        Returns:
            A tuple of ``(cleaned_frame, CleaningReport)``. The cleaned frame
            is sorted ascending by ``timestamp`` with stable relative order.

            ``CleaningReport.invalid_volume_rows_removed`` counts negative
            ratio-field removals. Price and trade-count counters remain zero
            for long/short frames.

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

        after_nulls = after_duplicates.filter(~_null_or_nan_ratio_predicate())
        null_rows_removed = after_duplicates.height - after_nulls.height

        after_timestamps = after_nulls.filter(_valid_timestamp_predicate())
        invalid_timestamp_rows_removed = after_nulls.height - after_timestamps.height

        after_ratios = after_timestamps.filter(_valid_ratio_predicate())
        invalid_volume_rows_removed = after_timestamps.height - after_ratios.height

        cleaned = after_ratios.sort(_COL_TIMESTAMP, maintain_order=True)
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


def _null_or_nan_ratio_predicate() -> pl.Expr:
    """Return a row mask that is true when any ratio field is null or NaN."""
    predicates: list[pl.Expr] = [
        pl.col(column).is_null() | pl.col(column).is_nan() for column in _RATIO_COLUMNS
    ]
    combined = predicates[0]
    for predicate in predicates[1:]:
        combined = combined | predicate
    return combined


def _valid_timestamp_predicate() -> pl.Expr:
    """Return a row mask that is true for valid ``timestamp`` values.

    Valid timestamps are non-null integers strictly greater than zero.
    Fractional timestamps are rejected when the column dtype is floating.
    """
    timestamp = pl.col(_COL_TIMESTAMP)
    positive = timestamp.is_not_null() & (timestamp > 0)
    integer_valued = timestamp == timestamp.cast(pl.Int64)
    return positive & integer_valued


def _valid_ratio_predicate() -> pl.Expr:
    """Return a row mask that is true when all ratio fields are non-negative."""
    return (
        (pl.col(_COL_LONG_ACCOUNT) >= 0)
        & (pl.col(_COL_SHORT_ACCOUNT) >= 0)
        & (pl.col(_COL_LONG_SHORT_RATIO) >= 0)
    )


def _require_mandatory_columns(frame: pl.DataFrame) -> None:
    """Raise when any mandatory long/short column is absent."""
    missing = [name for name in _MANDATORY_COLUMNS if name not in frame.columns]
    if missing:
        raise ProcessingValidationError(
            f"missing required long/short columns: {missing}",
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
        warnings.append(_WARN_RATIOS.format(count=invalid_volume_rows_removed))
    return tuple(warnings)

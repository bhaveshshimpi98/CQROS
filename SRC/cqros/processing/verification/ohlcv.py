"""CQROS OHLCV dataset verification.

Purpose:
    Inspect OHLCV frames and report structural findings without cleaning or
    mutating input data.

Responsibilities:
    - Validate required OHLCV columns
    - Count duplicate ``(symbol, timeframe, open_time)`` keys, nulls, NaNs,
      invalid ``open_time`` values, and invalid numeric relationships
    - Validate ``close_time >= 0`` and ``close_time > open_time``
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``OHLCVVerifier``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = ["OHLCVVerifier"]

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

_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    _COL_OPEN,
    _COL_HIGH,
    _COL_LOW,
    _COL_CLOSE,
    _COL_VOLUME,
    _COL_QUOTE_VOLUME,
    _COL_TRADE_COUNT,
)

_DUPLICATE_KEY_COLUMNS: Final[tuple[str, ...]] = (
    _COL_SYMBOL,
    _COL_TIMEFRAME,
    _COL_OPEN_TIME,
)

_WARN_DUPLICATES: Final[str] = "Duplicate timestamps detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_NUMERIC: Final[str] = "Invalid OHLCV numeric relationships."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by open_time."


class OHLCVVerifier(BaseVerifier):
    """Deterministic OHLCV verifier that reports findings only.

    Inspects structural quality of an OHLCV frame. Does not clean rows,
    fill gaps, sort timestamps, mutate values, access storage, or compute
    features.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input OHLCV DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            ProcessingValidationError: If any required column is missing.
        """
        self._validate_required_columns(frame, _REQUIRED_COLUMNS)

        duplicate_timestamp_rows = self._count_duplicate_key_rows(frame)
        null_rows = self._count_null_rows(frame, _REQUIRED_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _NUMERIC_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_OPEN_TIME,
        )
        invalid_numeric_rows = self._count_invalid_numeric_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_OPEN_TIME)

        warnings = _build_warnings(
            duplicate_timestamp_rows=duplicate_timestamp_rows,
            null_rows=null_rows,
            nan_rows=nan_rows,
            invalid_timestamp_rows=invalid_timestamp_rows,
            invalid_numeric_rows=invalid_numeric_rows,
            is_sorted=is_sorted,
        )
        passed = (
            duplicate_timestamp_rows == 0
            and null_rows == 0
            and nan_rows == 0
            and invalid_timestamp_rows == 0
            and invalid_numeric_rows == 0
            and is_sorted
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

    def _count_duplicate_key_rows(self, frame: pl.DataFrame) -> int:
        """Return rows beyond the first ``(symbol, timeframe, open_time)``.

        Uses keep-first semantics over the canonical OHLCV primary key.

        Args:
            frame: Input OHLCV DataFrame. Must not be mutated.

        Returns:
            Count of duplicate primary-key rows.
        """
        if frame.height == 0:
            return 0
        unique_count = int(frame.select(pl.struct(*_DUPLICATE_KEY_COLUMNS).n_unique()).item())
        return frame.height - unique_count

    def _count_invalid_numeric_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with invalid OHLCV prices, volumes, or relationships.

        Each qualifying row is counted once regardless of how many numeric
        rules it violates. Includes ``close_time < 0`` and
        ``close_time <= open_time``.

        Args:
            frame: Input OHLCV DataFrame. Must not be mutated.

        Returns:
            Count of invalid numeric rows.
        """
        if frame.height == 0:
            return 0
        invalid_mask = (
            (pl.col(_COL_OPEN) <= 0)
            | (pl.col(_COL_HIGH) <= 0)
            | (pl.col(_COL_LOW) <= 0)
            | (pl.col(_COL_CLOSE) <= 0)
            | (pl.col(_COL_VOLUME) < 0)
            | (pl.col(_COL_QUOTE_VOLUME) < 0)
            | (pl.col(_COL_TRADE_COUNT) < 0)
            | (pl.col(_COL_HIGH) < pl.col(_COL_LOW))
            | (pl.col(_COL_HIGH) < pl.col(_COL_OPEN))
            | (pl.col(_COL_HIGH) < pl.col(_COL_CLOSE))
            | (pl.col(_COL_LOW) > pl.col(_COL_OPEN))
            | (pl.col(_COL_LOW) > pl.col(_COL_CLOSE))
            | (pl.col(_COL_CLOSE_TIME) < 0)
            | (pl.col(_COL_CLOSE_TIME) <= pl.col(_COL_OPEN_TIME))
        )
        return int(frame.select(invalid_mask.sum()).item())


def _build_warnings(
    *,
    duplicate_timestamp_rows: int,
    null_rows: int,
    nan_rows: int,
    invalid_timestamp_rows: int,
    invalid_numeric_rows: int,
    is_sorted: bool,
) -> tuple[str, ...]:
    """Return deterministic warnings for non-zero counters and sort failures."""
    warnings: list[str] = []
    if duplicate_timestamp_rows > 0:
        warnings.append(_WARN_DUPLICATES)
    if null_rows > 0:
        warnings.append(_WARN_NULLS)
    if nan_rows > 0:
        warnings.append(_WARN_NANS)
    if invalid_timestamp_rows > 0:
        warnings.append(_WARN_TIMESTAMPS)
    if invalid_numeric_rows > 0:
        warnings.append(_WARN_NUMERIC)
    if not is_sorted:
        warnings.append(_WARN_UNSORTED)
    return tuple(warnings)

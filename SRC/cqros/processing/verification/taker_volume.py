"""CQROS taker-volume dataset verification.

Purpose:
    Inspect taker buy/sell volume frames and report structural findings
    without cleaning or mutating input data.

Responsibilities:
    - Validate required taker-volume columns
    - Count duplicate timestamps, nulls, NaNs, invalid timestamps, and
      invalid volume or ``buy_sell_ratio`` values
    - Allow zero volumes and NULL ratios when ``sell_volume`` is zero
    - Emit deterministic warnings and a pass/fail outcome
    - Never sort, clean, repair, or mutate the input frame

Dependencies:
    ``polars``, the Python standard library,
    ``cqros.processing.verification.base``, and
    ``cqros.processing.verification.report``.

Public API:
    ``TakerVolumeVerifier``
"""

from __future__ import annotations

import math
from typing import Final

import polars as pl

from cqros.processing.verification.base import BaseVerifier
from cqros.processing.verification.report import VerificationReport

__all__ = ["TakerVolumeVerifier"]

_COL_TIMESTAMP: Final[str] = "timestamp"
_COL_BUY_VOLUME: Final[str] = "buy_volume"
_COL_SELL_VOLUME: Final[str] = "sell_volume"
_COL_BUY_SELL_RATIO: Final[str] = "buy_sell_ratio"

_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    _COL_TIMESTAMP,
    _COL_BUY_VOLUME,
    _COL_SELL_VOLUME,
    _COL_BUY_SELL_RATIO,
)

_NULL_COLUMNS: Final[tuple[str, ...]] = (
    _COL_TIMESTAMP,
    _COL_BUY_VOLUME,
    _COL_SELL_VOLUME,
)

_NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    _COL_BUY_VOLUME,
    _COL_SELL_VOLUME,
    _COL_BUY_SELL_RATIO,
)

_RATIO_REL_TOL: Final[float] = 1e-9
_RATIO_ABS_TOL: Final[float] = 1e-12

_WARN_DUPLICATES: Final[str] = "Duplicate timestamps detected."
_WARN_NULLS: Final[str] = "Rows containing NULL values."
_WARN_NANS: Final[str] = "Rows containing NaN values."
_WARN_TIMESTAMPS: Final[str] = "Invalid timestamps detected."
_WARN_NUMERIC: Final[str] = "Invalid taker volume numeric values."
_WARN_UNSORTED: Final[str] = "Frame is not sorted by timestamp."


class TakerVolumeVerifier(BaseVerifier):
    """Deterministic taker-volume verifier that reports findings only.

    Inspects structural quality of a taker-volume frame. Does not clean
    rows, fill gaps, sort timestamps, mutate values, access storage, or
    compute features. Zero volumes are valid. When ``sell_volume`` is zero,
    ``buy_sell_ratio`` must be NULL.
    """

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        """Verify ``frame`` and return an immutable verification report.

        Args:
            frame: Input taker-volume DataFrame. Must not be mutated.

        Returns:
            A ``VerificationReport`` describing counters, warnings, and
            overall pass/fail status.

        Raises:
            ProcessingValidationError: If any required column is missing.
        """
        self._validate_required_columns(frame, _REQUIRED_COLUMNS)

        duplicate_timestamp_rows = self._count_duplicate_timestamp_rows(
            frame,
            _COL_TIMESTAMP,
        )
        # ``buy_sell_ratio`` may legitimately be NULL when sell_volume == 0.
        null_rows = self._count_null_rows(frame, _NULL_COLUMNS)
        nan_rows = self._count_nan_rows(frame, _NUMERIC_COLUMNS)
        invalid_timestamp_rows = self._count_invalid_timestamp_rows(
            frame,
            _COL_TIMESTAMP,
        )
        invalid_numeric_rows = self._count_invalid_numeric_rows(frame)
        is_sorted = self._is_sorted(frame, _COL_TIMESTAMP)

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

    def _count_invalid_numeric_rows(self, frame: pl.DataFrame) -> int:
        """Return rows with negative volumes or inconsistent ``buy_sell_ratio``.

        Each qualifying row is counted once. Ratio consistency uses
        ``math.isclose`` for non-zero ``sell_volume`` rows.

        Args:
            frame: Input taker-volume DataFrame. Must not be mutated.

        Returns:
            Count of invalid numeric rows.
        """
        if frame.height == 0:
            return 0
        buys = frame.get_column(_COL_BUY_VOLUME).to_list()
        sells = frame.get_column(_COL_SELL_VOLUME).to_list()
        ratios = frame.get_column(_COL_BUY_SELL_RATIO).to_list()
        return sum(
            1
            for buy, sell, ratio in zip(buys, sells, ratios, strict=True)
            if _is_invalid_numeric_row(buy, sell, ratio)
        )


def _as_finite_float(value: object) -> float | None:
    """Return a finite float, or ``None`` for null/NaN/non-numeric values."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    return None


def _is_invalid_numeric_row(buy: object, sell: object, ratio: object) -> bool:
    """Return True when volumes are negative or the ratio is inconsistent."""
    buy_value = _as_finite_float(buy)
    sell_value = _as_finite_float(sell)
    if buy_value is not None and buy_value < 0:
        return True
    if sell_value is not None and sell_value < 0:
        return True
    if buy_value is None or sell_value is None:
        return False

    if sell_value == 0.0:
        return ratio is not None

    ratio_value = _as_finite_float(ratio)
    if ratio_value is None:
        return True
    expected = buy_value / sell_value
    return not math.isclose(
        expected,
        ratio_value,
        rel_tol=_RATIO_REL_TOL,
        abs_tol=_RATIO_ABS_TOL,
    )


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

"""CQROS ATR slope research factor.

Purpose:
    Compute the rolling OLS slope of average true range as a pure price
    alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``ATRSlopeFactor`` metadata
    - Append an ``atr_slope`` column using Polars expressions only
    - Fail fast on invalid lookback and missing OHLC columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``ATRSlopeFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["ATRSlopeFactor"]

_HIGH_COLUMN: Final[str] = "high"
_LOW_COLUMN: Final[str] = "low"
_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "atr_slope"
_FACTOR_NAME: Final[str] = "atr_slope"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Rolling OLS slope of average true range against a relative time index."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-ATR-SLOPE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-ATR-SLOPE-002"


@dataclass(frozen=True, slots=True)
class ATRSlopeFactor(BaseFactor):
    """ATR slope alpha factor from rolling OLS on ATR.

    True range is ``max(high - low, |high - prev_close|, |low - prev_close|)``.
    ATR is the rolling mean of true range over ``lookback``. Fits
    ``ATR ~ a + b * x`` over each trailing ``lookback`` ATR window where
    ``x`` is the relative index ``0 .. lookback-1``, and appends the slope
    ``b`` as ``atr_slope``. Incomplete ATR or slope windows are null.
    Missing values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``atr_slope``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: ATR and OLS window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_HIGH_COLUMN, _LOW_COLUMN, _CLOSE_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 20

    def __post_init__(self) -> None:
        """Validate base metadata and require lookback >= 2.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback < 2``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback < 2:
            raise ValidationError(
                "lookback must be an integer greater than or equal to 2",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append ATR slope without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``high``, ``low``, and
                ``close`` columns.

        Returns:
            A new DataFrame with all original columns plus ``atr_slope``.
            Incomplete ATR or slope windows of ``atr_slope`` are null.

        Raises:
            FactorError: If a required column is not present in ``frame``.
        """
        for column in self.required_features:
            if column not in frame.columns:
                raise FactorError(
                    f"required column missing: {column}",
                    error_code=_ERROR_MISSING_COLUMN,
                    details={
                        "factor": self.name,
                        "required_column": column,
                        "available_columns": tuple(frame.columns),
                    },
                )

        window = self.lookback
        prev_close = pl.col(_CLOSE_COLUMN).shift(1)
        true_range = pl.max_horizontal(
            pl.col(_HIGH_COLUMN) - pl.col(_LOW_COLUMN),
            (pl.col(_HIGH_COLUMN) - prev_close).abs(),
            (pl.col(_LOW_COLUMN) - prev_close).abs(),
        )
        atr = true_range.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        )
        time_index = pl.int_range(0, pl.len())
        sum_y = atr.rolling_sum(window_size=window)  # pyright: ignore[reportUnknownMemberType]
        sum_ty = (time_index * atr).rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        )
        t_start = time_index - window + 1
        sum_xy = sum_ty - t_start * sum_y
        sum_x = (window - 1) * window / 2.0
        sum_x2 = (window - 1) * window * (2 * window - 1) / 6.0
        denom = window * sum_x2 - sum_x * sum_x
        slope_expr = (window * sum_xy - sum_x * sum_y) / denom
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            slope_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

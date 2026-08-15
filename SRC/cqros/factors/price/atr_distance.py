"""CQROS ATR distance research factor.

Purpose:
    Compute the distance of close from average true range as a pure price
    alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``ATRDistanceFactor`` metadata
    - Append an ``atr_distance`` column using Polars expressions only
    - Fail fast on invalid lookback and missing OHLC columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``ATRDistanceFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["ATRDistanceFactor"]

_HIGH_COLUMN: Final[str] = "high"
_LOW_COLUMN: Final[str] = "low"
_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "atr_distance"
_FACTOR_NAME: Final[str] = "atr_distance"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = "Distance of close from average true range as close minus ATR."
_ERROR_LOOKBACK: Final[str] = "FACTOR-ATR-DISTANCE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-ATR-DISTANCE-002"


@dataclass(frozen=True, slots=True)
class ATRDistanceFactor(BaseFactor):
    """ATR distance alpha factor as close minus rolling ATR.

    True range is ``max(high - low, |high - prev_close|, |low - prev_close|)``.
    ATR is the rolling mean of true range over ``lookback``. The factor
    appends ``close - ATR`` as ``atr_distance``. The first ``lookback - 1``
    rows are null. Missing values are never filled. The input DataFrame is
    never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``atr_distance``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: ATR window size (must be >= 2).
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
        """Append ATR distance without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``high``, ``low``, and
                ``close`` columns.

        Returns:
            A new DataFrame with all original columns plus ``atr_distance``.
            The first ``lookback - 1`` rows of ``atr_distance`` are null.

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

        close = pl.col(_CLOSE_COLUMN)
        prev_close = close.shift(1)
        true_range = pl.max_horizontal(
            pl.col(_HIGH_COLUMN) - pl.col(_LOW_COLUMN),
            (pl.col(_HIGH_COLUMN) - prev_close).abs(),
            (pl.col(_LOW_COLUMN) - prev_close).abs(),
        )
        atr = true_range.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        distance_expr = close - atr
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            distance_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

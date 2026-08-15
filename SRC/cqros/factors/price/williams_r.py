"""CQROS Williams %R research factor.

Purpose:
    Compute Williams Percent Range from OHLC prices as a pure price alpha
    factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``WilliamsRFactor`` metadata
    - Append a ``williams_r`` column using Polars expressions only
    - Fail fast on invalid lookback and missing OHLC columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``WilliamsRFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["WilliamsRFactor"]

_HIGH_COLUMN: Final[str] = "high"
_LOW_COLUMN: Final[str] = "low"
_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "williams_r"
_FACTOR_NAME: Final[str] = "williams_r"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Williams Percent Range measuring close versus the rolling high-low range."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-WILLIAMS-R-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-WILLIAMS-R-002"


@dataclass(frozen=True, slots=True)
class WilliamsRFactor(BaseFactor):
    """Williams %R alpha factor from rolling high-low range.

    Computes
    ``-100 * (highest_high - close) / (highest_high - lowest_low)`` where
    ``highest_high = high.rolling_max(lookback)`` and
    ``lowest_low = low.rolling_min(lookback)``. Returns null when the range
    is zero. Incomplete windows are null. Missing values are never filled.
    The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``williams_r``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Williams %R window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_HIGH_COLUMN, _LOW_COLUMN, _CLOSE_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 14

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
        """Append Williams %R without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``high``, ``low``, and
                ``close`` columns.

        Returns:
            A new DataFrame with all original columns plus ``williams_r``.
            Incomplete windows of ``williams_r`` are null.

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

        highest_high = pl.col(_HIGH_COLUMN).rolling_max(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        lowest_low = pl.col(_LOW_COLUMN).rolling_min(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        price_range = highest_high - lowest_low
        williams_expr = (
            pl.when(price_range != 0)
            .then(-100.0 * (highest_high - pl.col(_CLOSE_COLUMN)) / price_range)
            .otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            williams_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

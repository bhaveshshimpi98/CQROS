"""CQROS Price Volume Trend research factor.

Purpose:
    Compute cumulative Price Volume Trend from close and volume as a pure
    volume alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``PriceVolumeTrendFactor`` metadata
    - Append a ``price_volume_trend`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``PriceVolumeTrendFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["PriceVolumeTrendFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_VOLUME_COLUMN: Final[str] = "volume"
_OUTPUT_COLUMN: Final[str] = "price_volume_trend"
_FACTOR_NAME: Final[str] = "price_volume_trend"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "volume"
_FACTOR_DESCRIPTION: Final[str] = "Price Volume Trend as cumulative close return scaled by volume."
_ERROR_LOOKBACK: Final[str] = "FACTOR-PRICE-VOLUME-TREND-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-PRICE-VOLUME-TREND-002"


@dataclass(frozen=True, slots=True)
class PriceVolumeTrendFactor(BaseFactor):
    """Price Volume Trend alpha factor from close returns and volume.

    Computes ``cumsum(((close / close.shift(1)) - 1) * volume)`` and appends
    the result as ``price_volume_trend``. The first row is null because no
    prior close exists. Missing values are never filled. The input DataFrame
    is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``price_volume_trend``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``volume``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Must be ``0``; PVT has no rolling window.
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN, _VOLUME_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 0

    def __post_init__(self) -> None:
        """Validate base metadata and require lookback equal to 0.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback != 0``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback != 0:
            raise ValidationError(
                "lookback must be 0 for price_volume_trend",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append Price Volume Trend without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``close`` and
                ``volume`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``price_volume_trend``. The first row is null.

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
        close_return = (close / close.shift(1)) - 1  # pyright: ignore[reportUnknownMemberType]
        pvt_delta = close_return * pl.col(_VOLUME_COLUMN)
        pvt_expr = pvt_delta.cum_sum()  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            pvt_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

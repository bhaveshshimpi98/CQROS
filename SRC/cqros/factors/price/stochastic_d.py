"""CQROS Stochastic %D research factor.

Purpose:
    Compute Stochastic %D as a moving average of Fast %K from OHLC prices as
    a pure price alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``StochasticDFactor`` metadata
    - Append a ``stochastic_d`` column using Polars expressions only
    - Fail fast on invalid parameters and missing OHLC columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``StochasticDFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["StochasticDFactor"]

_HIGH_COLUMN: Final[str] = "high"
_LOW_COLUMN: Final[str] = "low"
_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "stochastic_d"
_FACTOR_NAME: Final[str] = "stochastic_d"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = "Stochastic %D as a moving average of Fast Stochastic %K."
_ERROR_LOOKBACK: Final[str] = "FACTOR-STOCHASTIC-D-001"
_ERROR_SMOOTH: Final[str] = "FACTOR-STOCHASTIC-D-002"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-STOCHASTIC-D-003"


@dataclass(frozen=True, slots=True)
class StochasticDFactor(BaseFactor):
    """Stochastic %D alpha factor as a moving average of Fast %K.

    Computes Fast %K as
    ``100 * (close - lowest_low) / (highest_high - lowest_low)`` over
    ``lookback``, then appends the ``smooth``-period rolling mean of %K as
    ``stochastic_d``. Returns null when the %K range is zero or the smoothing
    window is incomplete. Incomplete windows are null. Missing values are
    never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``stochastic_d``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Stochastic %K window size (must be >= 2).
        smooth: Moving-average window applied to %K (must be >= 1).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_HIGH_COLUMN, _LOW_COLUMN, _CLOSE_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 14
    smooth: int = 3

    def __post_init__(self) -> None:
        """Validate base metadata, lookback >= 2, and smooth >= 1.

        Raises:
            ValidationError: If any metadata or parameter invariant is
                violated.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback < 2:
            raise ValidationError(
                "lookback must be an integer greater than or equal to 2",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )
        if (
            not isinstance(cast(object, self.smooth), int)
            or isinstance(self.smooth, bool)
            or self.smooth < 1
        ):
            raise ValidationError(
                "smooth must be an integer greater than or equal to 1",
                error_code=_ERROR_SMOOTH,
                details={"parameter": "smooth", "value": self.smooth},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append Stochastic %D without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``high``, ``low``, and
                ``close`` columns.

        Returns:
            A new DataFrame with all original columns plus ``stochastic_d``.
            Incomplete windows of ``stochastic_d`` are null.

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
        stochastic_k = (
            pl.when(price_range != 0)
            .then(100.0 * (pl.col(_CLOSE_COLUMN) - lowest_low) / price_range)
            .otherwise(None)
        )
        d_expr = stochastic_k.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.smooth
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            d_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

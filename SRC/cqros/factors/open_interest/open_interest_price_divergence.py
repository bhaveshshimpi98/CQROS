"""CQROS open interest / price divergence research factor.

Purpose:
    Compute normalized divergence between open interest momentum and price
    momentum as a pure open-interest alpha factor for the Factor Research
    Engine.

Responsibilities:
    - Expose immutable ``OpenInterestPriceDivergenceFactor`` metadata
    - Append an ``open_interest_price_divergence`` column using Polars
      expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``OpenInterestPriceDivergenceFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["OpenInterestPriceDivergenceFactor"]

_OI_COLUMN: Final[str] = "open_interest"
_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "open_interest_price_divergence"
_FACTOR_NAME: Final[str] = "open_interest_price_divergence"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "open_interest"
_FACTOR_DESCRIPTION: Final[str] = (
    "Normalized divergence between fractional open interest momentum and "
    "price momentum over the same lookback."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-OPEN-INTEREST-PRICE-DIVERGENCE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-OPEN-INTEREST-PRICE-DIVERGENCE-002"


@dataclass(frozen=True, slots=True)
class OpenInterestPriceDivergenceFactor(BaseFactor):
    """OI/price divergence alpha factor from fractional momentum difference.

    Computes fractional open interest momentum
    ``(open_interest / open_interest.shift(lookback)) - 1`` and price
    momentum ``(close / close.shift(lookback)) - 1``, then appends their
    difference as ``open_interest_price_divergence``. Returns null when a
    shifted denominator is zero. The first ``lookback`` rows are null.
    Missing values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``open_interest_price_divergence``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``open_interest``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Momentum horizon in rows (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_OI_COLUMN, _CLOSE_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 20

    def __post_init__(self) -> None:
        """Validate base metadata and require a strictly positive lookback.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback <= 0``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback < 1:
            raise ValidationError(
                "lookback must be an integer greater than 0",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append OI/price divergence without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``open_interest`` and
                ``close`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``open_interest_price_divergence``. The first ``lookback`` rows
            are null.

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

        oi = pl.col(_OI_COLUMN)
        close = pl.col(_CLOSE_COLUMN)
        oi_prior = oi.shift(self.lookback)  # pyright: ignore[reportUnknownMemberType]
        close_prior = close.shift(self.lookback)  # pyright: ignore[reportUnknownMemberType]
        oi_momentum = pl.when(oi_prior != 0).then((oi / oi_prior) - 1.0).otherwise(None)
        price_momentum = pl.when(close_prior != 0).then((close / close_prior) - 1.0).otherwise(None)
        divergence_expr = oi_momentum - price_momentum
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            divergence_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

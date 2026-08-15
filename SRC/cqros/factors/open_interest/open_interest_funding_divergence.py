"""CQROS open interest / funding divergence research factor.

Purpose:
    Compute normalized divergence between open interest and funding rate as
    a pure open-interest alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``OpenInterestFundingDivergenceFactor`` metadata
    - Append an ``open_interest_funding_divergence`` column using Polars
      expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``OpenInterestFundingDivergenceFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.rolling_zscore import rolling_zscore_expr

__all__ = ["OpenInterestFundingDivergenceFactor"]

_OI_COLUMN: Final[str] = "open_interest"
_RATE_COLUMN: Final[str] = "funding_rate"
_OUTPUT_COLUMN: Final[str] = "open_interest_funding_divergence"
_FACTOR_NAME: Final[str] = "open_interest_funding_divergence"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "open_interest"
_FACTOR_DESCRIPTION: Final[str] = (
    "Normalized divergence between open interest z-score and funding rate " "z-score."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-OPEN-INTEREST-FUNDING-DIVERGENCE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-OPEN-INTEREST-FUNDING-DIVERGENCE-002"


@dataclass(frozen=True, slots=True)
class OpenInterestFundingDivergenceFactor(BaseFactor):
    """OI/funding divergence alpha factor from rolling z-score difference.

    Computes population z-scores of ``open_interest`` and ``funding_rate``
    over ``lookback``, then appends ``oi_zscore - funding_zscore`` as
    ``open_interest_funding_divergence``. Returns ``0.0`` for a component
    when its rolling standard deviation is zero. Incomplete windows are null.
    Missing values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier
            (``open_interest_funding_divergence``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``open_interest``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_OI_COLUMN, _RATE_COLUMN)
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
        """Append OI/funding divergence without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``open_interest`` and
                ``funding_rate`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``open_interest_funding_divergence``. Incomplete windows are
            null.

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

        oi_z = rolling_zscore_expr(pl.col(_OI_COLUMN), window_size=self.lookback)
        rate_z = rolling_zscore_expr(pl.col(_RATE_COLUMN), window_size=self.lookback)
        divergence_expr = oi_z - rate_z
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            divergence_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

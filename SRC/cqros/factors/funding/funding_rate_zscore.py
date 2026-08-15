"""CQROS funding rate z-score research factor.

Purpose:
    Compute the rolling z-score of funding rate as a pure funding alpha
    factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``FundingRateZScoreFactor`` metadata
    - Append a ``funding_rate_zscore`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``funding_rate``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``FundingRateZScoreFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.rolling_zscore import rolling_zscore_expr

__all__ = ["FundingRateZScoreFactor"]

_RATE_COLUMN: Final[str] = "funding_rate"
_OUTPUT_COLUMN: Final[str] = "funding_rate_zscore"
_FACTOR_NAME: Final[str] = "funding_rate_zscore"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "funding"
_FACTOR_DESCRIPTION: Final[str] = (
    "Rolling z-score of funding rate versus its rolling mean and standard " "deviation."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-FUNDING-RATE-ZSCORE-001"
_ERROR_MISSING_RATE: Final[str] = "FACTOR-FUNDING-RATE-ZSCORE-002"


@dataclass(frozen=True, slots=True)
class FundingRateZScoreFactor(BaseFactor):
    """Funding rate z-score alpha factor from rolling mean and std.

    Computes ``(funding_rate - rolling_mean) / rolling_std`` using population
    standard deviation (``ddof=0``). Returns ``0.0`` when rolling standard
    deviation is zero. Incomplete windows are null. Missing values are never
    filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``funding_rate_zscore``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``funding``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_RATE_COLUMN,)
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
        """Append funding rate z-score without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``funding_rate``
                column.

        Returns:
            A new DataFrame with all original columns plus
            ``funding_rate_zscore``. Incomplete windows are null.

        Raises:
            FactorError: If ``funding_rate`` is not present in ``frame``.
        """
        if _RATE_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_RATE_COLUMN}",
                error_code=_ERROR_MISSING_RATE,
                details={
                    "factor": self.name,
                    "required_column": _RATE_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        zscore_expr = rolling_zscore_expr(pl.col(_RATE_COLUMN), window_size=self.lookback)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            zscore_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

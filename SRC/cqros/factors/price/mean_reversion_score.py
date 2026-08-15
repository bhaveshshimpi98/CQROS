"""CQROS mean reversion score research factor.

Purpose:
    Compute a mean-reversion score as the negative rolling z-score of close
    prices as a pure price alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``MeanReversionScoreFactor`` metadata
    - Append a ``mean_reversion_score`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``MeanReversionScoreFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.rolling_zscore import rolling_zscore_expr

__all__ = ["MeanReversionScoreFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "mean_reversion_score"
_FACTOR_NAME: Final[str] = "mean_reversion_score"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Mean reversion score as negative standardized distance from rolling mean."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-MEAN-REVERSION-SCORE-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-MEAN-REVERSION-SCORE-002"


@dataclass(frozen=True, slots=True)
class MeanReversionScoreFactor(BaseFactor):
    """Mean reversion score alpha factor from negative price z-score.

    Computes ``-(close - rolling_mean) / rolling_std`` using population
    standard deviation (``ddof=0``). Positive values favor mean reversion
    when price is below the rolling mean. Returns ``0.0`` when rolling
    standard deviation is zero. Incomplete windows are null. Missing values
    are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``mean_reversion_score``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN,)
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
        """Append mean reversion score without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``mean_reversion_score``. Incomplete windows are null.

        Raises:
            FactorError: If ``close`` is not present in ``frame``.
        """
        if _CLOSE_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_CLOSE_COLUMN}",
                error_code=_ERROR_MISSING_CLOSE,
                details={
                    "factor": self.name,
                    "required_column": _CLOSE_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        score_expr = -rolling_zscore_expr(pl.col(_CLOSE_COLUMN), window_size=self.lookback)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            score_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

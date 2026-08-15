"""CQROS open interest volatility research factor.

Purpose:
    Compute rolling standard deviation of open interest as a pure
    open-interest alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``OpenInterestVolatilityFactor`` metadata
    - Append an ``open_interest_volatility`` column using Polars expressions
      only
    - Fail fast on invalid lookback and missing ``open_interest``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``OpenInterestVolatilityFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["OpenInterestVolatilityFactor"]

_OI_COLUMN: Final[str] = "open_interest"
_OUTPUT_COLUMN: Final[str] = "open_interest_volatility"
_FACTOR_NAME: Final[str] = "open_interest_volatility"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "open_interest"
_FACTOR_DESCRIPTION: Final[str] = "Rolling population standard deviation of open interest."
_ERROR_LOOKBACK: Final[str] = "FACTOR-OPEN-INTEREST-VOLATILITY-001"
_ERROR_MISSING_OI: Final[str] = "FACTOR-OPEN-INTEREST-VOLATILITY-002"


@dataclass(frozen=True, slots=True)
class OpenInterestVolatilityFactor(BaseFactor):
    """Open interest volatility alpha factor from rolling OI std.

    Computes the population standard deviation (``ddof=0``) of
    ``open_interest`` over ``lookback`` and appends the result as
    ``open_interest_volatility``. Incomplete windows are null. Missing
    values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``open_interest_volatility``).
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
    required_features: tuple[str, ...] = (_OI_COLUMN,)
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
        """Append open interest volatility without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing an ``open_interest``
                column.

        Returns:
            A new DataFrame with all original columns plus
            ``open_interest_volatility``. Incomplete windows are null.

        Raises:
            FactorError: If ``open_interest`` is not present in ``frame``.
        """
        if _OI_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_OI_COLUMN}",
                error_code=_ERROR_MISSING_OI,
                details={
                    "factor": self.name,
                    "required_column": _OI_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        volatility_expr = pl.col(
            _OI_COLUMN
        ).rolling_std(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback,
            ddof=0,
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            volatility_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

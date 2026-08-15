"""CQROS Parkinson volatility research factor.

Purpose:
    Compute the Parkinson high-low volatility estimator as a pure price
    alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``ParkinsonVolatilityFactor`` metadata
    - Append a ``parkinson_volatility`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``high`` or ``low``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``ParkinsonVolatilityFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["ParkinsonVolatilityFactor"]

_HIGH_COLUMN: Final[str] = "high"
_LOW_COLUMN: Final[str] = "low"
_OUTPUT_COLUMN: Final[str] = "parkinson_volatility"
_FACTOR_NAME: Final[str] = "parkinson_volatility"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Parkinson high-low volatility estimator over a rolling lookback window."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-PARKINSON-VOLATILITY-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-PARKINSON-VOLATILITY-002"
_FOUR_LN_TWO: Final[float] = 4.0 * log(2.0)


@dataclass(frozen=True, slots=True)
class ParkinsonVolatilityFactor(BaseFactor):
    """Parkinson volatility alpha factor from high and low prices.

    Computes
    ``sqrt(mean(ln(high / low)^2, lookback) / (4 * ln(2)))`` and appends the
    result as ``parkinson_volatility``. Returns null when low is zero.
    Incomplete windows are null. Missing values are never filled. The input
    DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``parkinson_volatility``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Parkinson window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_HIGH_COLUMN, _LOW_COLUMN)
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
        """Append Parkinson volatility without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``high`` and ``low``.

        Returns:
            A new DataFrame with all original columns plus
            ``parkinson_volatility``. The first ``lookback - 1`` rows of
            ``parkinson_volatility`` are null.

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

        high = pl.col(_HIGH_COLUMN)
        low = pl.col(_LOW_COLUMN)
        log_hl_sq = (
            pl.when(low != 0).then((high / low).log().pow(2)).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        mean_log_hl_sq = log_hl_sq.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        volatility_expr = (mean_log_hl_sq / _FOUR_LN_TWO).sqrt()
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            volatility_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

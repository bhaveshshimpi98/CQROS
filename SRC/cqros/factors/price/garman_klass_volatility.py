"""CQROS Garman-Klass volatility research factor.

Purpose:
    Compute the Garman-Klass OHLC volatility estimator as a pure price
    alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``GarmanKlassVolatilityFactor`` metadata
    - Append a ``garman_klass_volatility`` column using Polars expressions only
    - Fail fast on invalid lookback and missing OHLC columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``GarmanKlassVolatilityFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["GarmanKlassVolatilityFactor"]

_OPEN_COLUMN: Final[str] = "open"
_HIGH_COLUMN: Final[str] = "high"
_LOW_COLUMN: Final[str] = "low"
_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "garman_klass_volatility"
_FACTOR_NAME: Final[str] = "garman_klass_volatility"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Garman-Klass OHLC volatility estimator over a rolling lookback window."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-GARMAN-KLASS-VOLATILITY-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-GARMAN-KLASS-VOLATILITY-002"
_GK_CO_COEFFICIENT: Final[float] = 2.0 * log(2.0) - 1.0


@dataclass(frozen=True, slots=True)
class GarmanKlassVolatilityFactor(BaseFactor):
    """Garman-Klass volatility alpha factor from open, high, low, and close.

    Computes the rolling mean of
    ``0.5 * ln(high / low)^2 - (2 * ln(2) - 1) * ln(close / open)^2`` and
    appends its square root as ``garman_klass_volatility``. Returns null
    when open or low is zero, or when the rolling mean is negative.
    Incomplete windows are null. Missing values are never filled. The input
    DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``garman_klass_volatility``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Garman-Klass window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (
        _OPEN_COLUMN,
        _HIGH_COLUMN,
        _LOW_COLUMN,
        _CLOSE_COLUMN,
    )
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
        """Append Garman-Klass volatility without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``open``, ``high``,
                ``low``, and ``close`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``garman_klass_volatility``. The first ``lookback - 1`` rows of
            ``garman_klass_volatility`` are null.

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

        open_ = pl.col(_OPEN_COLUMN)
        high = pl.col(_HIGH_COLUMN)
        low = pl.col(_LOW_COLUMN)
        close = pl.col(_CLOSE_COLUMN)
        log_hl_sq = (
            pl.when(low != 0).then((high / low).log().pow(2)).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        log_co_sq = (
            pl.when(open_ != 0).then((close / open_).log().pow(2)).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        gk_variance = 0.5 * log_hl_sq - _GK_CO_COEFFICIENT * log_co_sq
        mean_variance = gk_variance.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        volatility_expr = (
            pl.when(mean_variance >= 0).then(mean_variance.sqrt()).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            volatility_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

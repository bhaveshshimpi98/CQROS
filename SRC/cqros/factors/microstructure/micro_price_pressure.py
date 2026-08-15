"""CQROS micro-price pressure research factor.

Purpose:
    Compute smoothed close-versus-VWAP relative pressure as a pure
    microstructure alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``MicroPricePressureFactor`` metadata
    - Append a ``micro_price_pressure`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``MicroPricePressureFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["MicroPricePressureFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_VWAP_COLUMN: Final[str] = "vwap"
_OUTPUT_COLUMN: Final[str] = "micro_price_pressure"
_FACTOR_NAME: Final[str] = "micro_price_pressure"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "microstructure"
_FACTOR_DESCRIPTION: Final[str] = "Micro-price pressure as rolling mean of (close - vwap) / vwap."
_ERROR_LOOKBACK: Final[str] = "FACTOR-MICRO-PRICE-PRESSURE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-MICRO-PRICE-PRESSURE-002"


@dataclass(frozen=True, slots=True)
class MicroPricePressureFactor(BaseFactor):
    """Micro-price pressure alpha factor from close versus VWAP.

    Computes the rolling mean of ``(close - vwap) / vwap`` over ``lookback``
    and appends the result as ``micro_price_pressure``. Returns null when
    ``vwap`` is zero. Incomplete windows are null. Missing values are never
    filled. The input DataFrame is never mutated.

    This factor consumes the provided ``vwap`` column and does not recompute
    VWAP from OHLC.

    Attributes:
        name: Stable factor identifier (``micro_price_pressure``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``microstructure``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling mean window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN, _VWAP_COLUMN)
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
        """Append micro-price pressure without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``close`` and ``vwap``.

        Returns:
            A new DataFrame with all original columns plus
            ``micro_price_pressure``. Incomplete windows are null.

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
        vwap = pl.col(_VWAP_COLUMN)
        relative = (
            pl.when(vwap != 0).then((close - vwap) / vwap).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        pressure_expr = relative.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            pressure_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

"""CQROS trade intensity research factor.

Purpose:
    Compute relative trade intensity versus its rolling mean as a pure
    microstructure alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``TradeIntensityFactor`` metadata
    - Append a ``trade_intensity`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``trade_count``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``TradeIntensityFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["TradeIntensityFactor"]

_TRADE_COUNT_COLUMN: Final[str] = "trade_count"
_OUTPUT_COLUMN: Final[str] = "trade_intensity"
_FACTOR_NAME: Final[str] = "trade_intensity"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "microstructure"
_FACTOR_DESCRIPTION: Final[str] = (
    "Trade intensity as current trade_count divided by its rolling mean."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-TRADE-INTENSITY-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-TRADE-INTENSITY-002"


@dataclass(frozen=True, slots=True)
class TradeIntensityFactor(BaseFactor):
    """Trade intensity alpha factor from trade count.

    Computes ``trade_count / rolling_mean(trade_count, lookback)`` and appends
    the result as ``trade_intensity``. Returns null when the rolling mean is
    zero. Incomplete windows are null. Missing values are never filled. The
    input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``trade_intensity``).
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
    required_features: tuple[str, ...] = (_TRADE_COUNT_COLUMN,)
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
        """Append trade intensity without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``trade_count`` column.

        Returns:
            A new DataFrame with all original columns plus ``trade_intensity``.
            Incomplete windows are null.

        Raises:
            FactorError: If ``trade_count`` is not present in ``frame``.
        """
        if _TRADE_COUNT_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_TRADE_COUNT_COLUMN}",
                error_code=_ERROR_MISSING_COLUMN,
                details={
                    "factor": self.name,
                    "required_column": _TRADE_COUNT_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        trade_count = pl.col(_TRADE_COUNT_COLUMN)
        mean = trade_count.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        intensity_expr = (
            pl.when(mean != 0).then(trade_count / mean).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            intensity_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

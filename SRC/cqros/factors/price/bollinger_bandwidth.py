"""CQROS Bollinger bandwidth research factor.

Purpose:
    Compute Bollinger bandwidth from close prices as a pure price alpha
    factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``BollingerBandwidthFactor`` metadata
    - Append a ``bollinger_bandwidth`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``BollingerBandwidthFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["BollingerBandwidthFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "bollinger_bandwidth"
_FACTOR_NAME: Final[str] = "bollinger_bandwidth"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = "Bollinger bandwidth as band width divided by the middle SMA."
_ERROR_LOOKBACK: Final[str] = "FACTOR-BOLLINGER-BANDWIDTH-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-BOLLINGER-BANDWIDTH-002"
_STD_MULTIPLIER: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class BollingerBandwidthFactor(BaseFactor):
    """Bollinger bandwidth alpha factor from rolling Bollinger bands.

    Computes ``(upper - lower) / SMA`` where
    ``SMA = close.rolling_mean(lookback)``, ``upper = SMA + 2 * std``, and
    ``lower = SMA - 2 * std`` with population standard deviation. Returns
    null when SMA is zero. The first ``lookback - 1`` rows are null. Missing
    values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``bollinger_bandwidth``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Bollinger window size (must be >= 2).
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
        """Append Bollinger bandwidth without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``bollinger_bandwidth``. The first ``lookback - 1`` rows of
            ``bollinger_bandwidth`` are null.

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

        close = pl.col(_CLOSE_COLUMN)
        sma = close.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        std = close.rolling_std(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback,
            ddof=0,
        )
        width = (2.0 * _STD_MULTIPLIER) * std
        bandwidth_expr = (
            pl.when(sma != 0).then(width / sma).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            bandwidth_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

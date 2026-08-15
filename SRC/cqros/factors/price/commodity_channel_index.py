"""CQROS Commodity Channel Index research factor.

Purpose:
    Compute typical-price Commodity Channel Index from OHLC prices as a pure
    price alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``CommodityChannelIndexFactor`` metadata
    - Append a ``commodity_channel_index`` column using Polars expressions only
    - Fail fast on invalid lookback and missing OHLC columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``CommodityChannelIndexFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["CommodityChannelIndexFactor"]

_HIGH_COLUMN: Final[str] = "high"
_LOW_COLUMN: Final[str] = "low"
_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "commodity_channel_index"
_FACTOR_NAME: Final[str] = "commodity_channel_index"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Commodity Channel Index from typical price mean absolute deviation."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-COMMODITY-CHANNEL-INDEX-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-COMMODITY-CHANNEL-INDEX-002"
_CCI_CONSTANT: Final[float] = 0.015
_TYPICAL_PRICE_ALIAS: Final[str] = "__cqros_typical_price"
_MAD_ALIAS: Final[str] = "__cqros_cci_mad"


@dataclass(frozen=True, slots=True)
class CommodityChannelIndexFactor(BaseFactor):
    """Commodity Channel Index alpha factor from typical price.

    Computes typical price ``(high + low + close) / 3``, then
    ``(TP - SMA(TP)) / (0.015 * MAD(TP))`` where MAD is the mean absolute
    deviation of typical price from its rolling mean within the same window.
    Returns null when MAD is zero. Incomplete windows are null. Missing
    values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``commodity_channel_index``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: CCI window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_HIGH_COLUMN, _LOW_COLUMN, _CLOSE_COLUMN)
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
        """Append Commodity Channel Index without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``high``, ``low``, and
                ``close`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``commodity_channel_index``. Incomplete windows are null.

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

        lookback = self.lookback
        with_tp = frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            ((pl.col(_HIGH_COLUMN) + pl.col(_LOW_COLUMN) + pl.col(_CLOSE_COLUMN)) / 3.0)
            .cast(pl.Float64)
            .alias(_TYPICAL_PRICE_ALIAS)
        )
        mad = (
            with_tp.get_column(_TYPICAL_PRICE_ALIAS)
            .rolling_map(  # pyright: ignore[reportUnknownMemberType]
                function=_window_mean_absolute_deviation,
                window_size=lookback,
                min_samples=lookback,
            )
            .alias(_MAD_ALIAS)
        )
        sma = pl.col(_TYPICAL_PRICE_ALIAS).rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=lookback
        )
        cci_expr = (
            pl.when(pl.col(_MAD_ALIAS) != 0)
            .then((pl.col(_TYPICAL_PRICE_ALIAS) - sma) / (_CCI_CONSTANT * pl.col(_MAD_ALIAS)))
            .otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return (
            with_tp.with_columns(mad)  # pyright: ignore[reportUnknownMemberType]
            .with_columns(cci_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN))
            .drop([_TYPICAL_PRICE_ALIAS, _MAD_ALIAS])
        )


def _window_mean_absolute_deviation(window: pl.Series) -> float | None:
    """Return mean absolute deviation of ``window`` from its mean.

    Args:
        window: Rolling typical-price window.

    Returns:
        Mean absolute deviation, or ``None`` when the window contains nulls.
    """
    if window.null_count() > 0:
        return None
    mean_value = window.mean()
    if not isinstance(mean_value, (int, float)):
        return None
    mad_value = (
        (window - float(mean_value)).abs().mean()
    )  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(mad_value, (int, float)):
        return None
    return float(mad_value)

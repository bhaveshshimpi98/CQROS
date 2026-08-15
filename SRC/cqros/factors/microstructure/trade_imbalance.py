"""CQROS trade imbalance research factor.

Purpose:
    Compute rolling buy/sell trade-count imbalance as a pure microstructure
    alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``TradeImbalanceFactor`` metadata
    - Append a ``trade_imbalance`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``TradeImbalanceFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["TradeImbalanceFactor"]

_BUY_COUNT_COLUMN: Final[str] = "buy_trade_count"
_SELL_COUNT_COLUMN: Final[str] = "sell_trade_count"
_OUTPUT_COLUMN: Final[str] = "trade_imbalance"
_FACTOR_NAME: Final[str] = "trade_imbalance"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "microstructure"
_FACTOR_DESCRIPTION: Final[str] = (
    "Trade imbalance as rolling (buy_trade_count - sell_trade_count) over "
    "rolling total trade count."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-TRADE-IMBALANCE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-TRADE-IMBALANCE-002"


@dataclass(frozen=True, slots=True)
class TradeImbalanceFactor(BaseFactor):
    """Trade imbalance alpha factor from buy and sell trade counts.

    Computes
    ``(sum(buy_trade_count) - sum(sell_trade_count)) /
    (sum(buy_trade_count) + sum(sell_trade_count))``
    over ``lookback`` and appends the result as ``trade_imbalance``. Returns
    null when the rolling total trade count is zero. Incomplete windows are
    null. Missing values are never filled. The input DataFrame is never
    mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``trade_imbalance``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``microstructure``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_BUY_COUNT_COLUMN, _SELL_COUNT_COLUMN)
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
        """Append trade imbalance without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``buy_trade_count`` and
                ``sell_trade_count`` columns.

        Returns:
            A new DataFrame with all original columns plus ``trade_imbalance``.
            Incomplete windows are null.

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

        buy_sum = pl.col(_BUY_COUNT_COLUMN).rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        sell_sum = pl.col(
            _SELL_COUNT_COLUMN
        ).rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        total = buy_sum + sell_sum
        imbalance_expr = (
            pl.when(total == 0).then(None).otherwise((buy_sum - sell_sum) / total)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            imbalance_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

"""CQROS order-flow momentum research factor.

Purpose:
    Compute multi-period momentum of signed taker volume as a pure
    microstructure alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``OrderFlowMomentumFactor`` metadata
    - Append an ``order_flow_momentum`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``OrderFlowMomentumFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["OrderFlowMomentumFactor"]

_BUY_COLUMN: Final[str] = "taker_buy_volume"
_SELL_COLUMN: Final[str] = "taker_sell_volume"
_OUTPUT_COLUMN: Final[str] = "order_flow_momentum"
_FACTOR_NAME: Final[str] = "order_flow_momentum"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "microstructure"
_FACTOR_DESCRIPTION: Final[str] = (
    "Order-flow momentum as absolute change in signed taker volume over lookback."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-ORDER-FLOW-MOMENTUM-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-ORDER-FLOW-MOMENTUM-002"


@dataclass(frozen=True, slots=True)
class OrderFlowMomentumFactor(BaseFactor):
    """Order-flow momentum alpha factor from signed taker volume change.

    Computes signed volume ``taker_buy_volume - taker_sell_volume``, then
    ``signed - signed.shift(lookback)``, and appends the result as
    ``order_flow_momentum``. The first ``lookback`` rows are null. Missing
    values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``order_flow_momentum``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``microstructure``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Momentum horizon in rows (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_BUY_COLUMN, _SELL_COLUMN)
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
        """Append order-flow momentum without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``taker_buy_volume`` and
                ``taker_sell_volume`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``order_flow_momentum``. The first ``lookback`` rows are null.

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

        signed = pl.col(_BUY_COLUMN) - pl.col(_SELL_COLUMN)
        momentum_expr = signed - signed.shift(  # pyright: ignore[reportUnknownMemberType]
            self.lookback
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            momentum_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

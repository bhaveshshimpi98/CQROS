"""CQROS relative momentum versus ETH research factor.

Purpose:
    Compute rolling return-sum relative momentum of an asset versus Ethereum
    returns as a pure relative-value alpha factor for the Factor Research
    Engine.

Responsibilities:
    - Expose immutable ``RelativeMomentumETHFactor`` metadata
    - Append a ``relative_momentum_eth`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``RelativeMomentumETHFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["RelativeMomentumETHFactor"]

_ASSET_COLUMN: Final[str] = "asset_return"
_ETH_COLUMN: Final[str] = "eth_return"
_OUTPUT_COLUMN: Final[str] = "relative_momentum_eth"
_FACTOR_NAME: Final[str] = "relative_momentum_eth"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "relative"
_FACTOR_DESCRIPTION: Final[str] = (
    "Relative momentum as asset rolling return sum minus ETH rolling return "
    "sum over a lookback window."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-RELATIVE-MOMENTUM-ETH-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-RELATIVE-MOMENTUM-ETH-002"


@dataclass(frozen=True, slots=True)
class RelativeMomentumETHFactor(BaseFactor):
    """Relative momentum alpha factor versus Ethereum returns.

    Computes ``rolling_sum(asset_return) - rolling_sum(eth_return)`` over
    ``lookback`` and appends the result as ``relative_momentum_eth``.
    Incomplete windows are null. Missing values are never filled. The input
    DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``relative_momentum_eth``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``relative``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling momentum window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_ASSET_COLUMN, _ETH_COLUMN)
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
        """Append ETH relative momentum without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``asset_return`` and
                ``eth_return`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``relative_momentum_eth``. Incomplete windows are null.

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

        asset_mom = pl.col(_ASSET_COLUMN).rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        eth_mom = pl.col(_ETH_COLUMN).rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        momentum_expr = asset_mom - eth_mom
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            momentum_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

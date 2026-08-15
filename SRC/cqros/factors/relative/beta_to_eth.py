"""CQROS beta-to-ETH research factor.

Purpose:
    Compute rolling OLS beta of an asset versus Ethereum returns as a pure
    relative-value alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``BetaToETHFactor`` metadata
    - Append a ``beta_to_eth`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``BetaToETHFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["BetaToETHFactor"]

_ASSET_COLUMN: Final[str] = "asset_return"
_ETH_COLUMN: Final[str] = "eth_return"
_OUTPUT_COLUMN: Final[str] = "beta_to_eth"
_FACTOR_NAME: Final[str] = "beta_to_eth"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "relative"
_FACTOR_DESCRIPTION: Final[str] = (
    "Rolling beta of asset returns versus ETH returns as population "
    "covariance divided by ETH variance."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-BETA-TO-ETH-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-BETA-TO-ETH-002"


@dataclass(frozen=True, slots=True)
class BetaToETHFactor(BaseFactor):
    """Beta-to-ETH alpha factor from rolling population covariance.

    Computes ``cov(asset_return, eth_return) / var(eth_return)`` over
    ``lookback`` using population moments (``ddof=0``) and appends the
    result as ``beta_to_eth``. Returns null when ETH variance is zero.
    Incomplete windows are null. Missing values are never filled. The
    input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``beta_to_eth``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``relative``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling covariance/variance window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_ASSET_COLUMN, _ETH_COLUMN)
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
        """Append beta-to-ETH without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``asset_return`` and
                ``eth_return`` columns.

        Returns:
            A new DataFrame with all original columns plus ``beta_to_eth``.
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

        asset = pl.col(_ASSET_COLUMN)
        eth = pl.col(_ETH_COLUMN)
        window = self.lookback
        mean_asset = asset.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        )
        mean_eth = eth.rolling_mean(window_size=window)  # pyright: ignore[reportUnknownMemberType]
        mean_product = (asset * eth).rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        )
        covariance = mean_product - mean_asset * mean_eth
        variance = eth.rolling_var(  # pyright: ignore[reportUnknownMemberType]
            window_size=window,
            ddof=0,
        )
        beta_expr = (
            pl.when(variance != 0).then(covariance / variance).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            beta_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

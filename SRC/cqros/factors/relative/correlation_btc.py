"""CQROS correlation-to-BTC research factor.

Purpose:
    Compute rolling Pearson correlation of an asset versus Bitcoin returns as
    a pure relative-value alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``CorrelationBTCFactor`` metadata
    - Append a ``correlation_btc`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``CorrelationBTCFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["CorrelationBTCFactor"]

_ASSET_COLUMN: Final[str] = "asset_return"
_BTC_COLUMN: Final[str] = "btc_return"
_OUTPUT_COLUMN: Final[str] = "correlation_btc"
_FACTOR_NAME: Final[str] = "correlation_btc"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "relative"
_FACTOR_DESCRIPTION: Final[str] = (
    "Rolling Pearson correlation between asset returns and BTC returns."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-CORRELATION-BTC-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-CORRELATION-BTC-002"


@dataclass(frozen=True, slots=True)
class CorrelationBTCFactor(BaseFactor):
    """Correlation-to-BTC alpha factor from rolling Pearson correlation.

    Computes population Pearson correlation between ``asset_return`` and
    ``btc_return`` over ``lookback`` and appends the result as
    ``correlation_btc``. Returns null when either rolling standard
    deviation is zero. Incomplete windows are null. Missing values are
    never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``correlation_btc``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``relative``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling correlation window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_ASSET_COLUMN, _BTC_COLUMN)
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
        """Append BTC correlation without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``asset_return`` and
                ``btc_return`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``correlation_btc``. Incomplete windows are null.

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
        btc = pl.col(_BTC_COLUMN)
        window = self.lookback
        mean_asset = asset.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        )
        mean_btc = btc.rolling_mean(window_size=window)  # pyright: ignore[reportUnknownMemberType]
        mean_product = (asset * btc).rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=window
        )
        covariance = mean_product - mean_asset * mean_btc
        std_asset = asset.rolling_std(  # pyright: ignore[reportUnknownMemberType]
            window_size=window,
            ddof=0,
        )
        std_btc = btc.rolling_std(  # pyright: ignore[reportUnknownMemberType]
            window_size=window,
            ddof=0,
        )
        corr_expr = (
            pl.when((std_asset != 0) & (std_btc != 0))
            .then(covariance / (std_asset * std_btc))
            .otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            corr_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

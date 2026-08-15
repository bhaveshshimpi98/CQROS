"""CQROS liquidation z-score research factor.

Purpose:
    Compute the rolling z-score of total liquidation volume as a pure
    liquidation alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``LiquidationZScoreFactor`` metadata
    - Append a ``liquidation_zscore`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``total_liquidation_volume``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``LiquidationZScoreFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.rolling_zscore import rolling_zscore_expr

__all__ = ["LiquidationZScoreFactor"]

_TOTAL_LIQ_COLUMN: Final[str] = "total_liquidation_volume"
_OUTPUT_COLUMN: Final[str] = "liquidation_zscore"
_FACTOR_NAME: Final[str] = "liquidation_zscore"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "liquidation"
_FACTOR_DESCRIPTION: Final[str] = (
    "Rolling z-score of total liquidation volume versus its rolling mean and " "standard deviation."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-LIQUIDATION-ZSCORE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-LIQUIDATION-ZSCORE-002"


@dataclass(frozen=True, slots=True)
class LiquidationZScoreFactor(BaseFactor):
    """Liquidation z-score alpha factor from rolling mean and std.

    Computes ``(total_liquidation_volume - rolling_mean) / rolling_std``
    using population standard deviation (``ddof=0``). Returns ``0.0`` when
    rolling standard deviation is zero. Incomplete windows are null. Missing
    values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``liquidation_zscore``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``liquidation``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_TOTAL_LIQ_COLUMN,)
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
        """Append liquidation z-score without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a
                ``total_liquidation_volume`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``liquidation_zscore``. Incomplete windows are null.

        Raises:
            FactorError: If ``total_liquidation_volume`` is not present in
                ``frame``.
        """
        if _TOTAL_LIQ_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_TOTAL_LIQ_COLUMN}",
                error_code=_ERROR_MISSING_COLUMN,
                details={
                    "factor": self.name,
                    "required_column": _TOTAL_LIQ_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        zscore_expr = rolling_zscore_expr(
            pl.col(_TOTAL_LIQ_COLUMN),
            window_size=self.lookback,
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            zscore_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

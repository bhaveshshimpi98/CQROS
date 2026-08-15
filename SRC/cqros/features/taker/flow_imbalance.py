"""CQROS taker flow imbalance feature.

Purpose:
    Compute taker flow imbalance from raw taker-volume repository data.

Responsibilities:
    - Expose immutable ``FlowImbalanceFeature`` metadata
    - Append a ``flow_imbalance`` column using Polars expressions only
    - Fail fast when required volume columns are missing

Dependencies:
    ``polars``, ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``FlowImbalanceFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["FlowImbalanceFeature"]

_BUY_COLUMN: Final[str] = "buy_volume"
_SELL_COLUMN: Final[str] = "sell_volume"
_OUTPUT_COLUMN: Final[str] = "flow_imbalance"
_FEATURE_NAME: Final[str] = "flow_imbalance"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "taker"
_FEATURE_DESCRIPTION: Final[str] = (
    "Taker flow imbalance as (buy_volume - sell_volume) / (buy_volume + sell_volume)."
)
_ERROR_MISSING_COLUMN: Final[str] = "FEATURE-FLOW-IMBALANCE-001"


@dataclass(frozen=True, slots=True)
class FlowImbalanceFeature(BaseFeature):
    """Normalized taker buy/sell flow imbalance.

    Computes ``(buy_volume - sell_volume) / (buy_volume + sell_volume)``. When
    total volume is zero the output is null. Missing values are not filled.

    Attributes:
        name: Stable feature identifier (``flow_imbalance``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``taker``).
        description: Human-readable feature summary.
        required_columns: Input columns required by ``transform``.
        produced_columns: Output columns produced by ``transform``.
        lookback: Warm-up rows required (none for this point feature).
        dependencies: Upstream feature names (none for this feature).
    """

    name: str = _FEATURE_NAME
    version: str = _FEATURE_VERSION
    category: str = _FEATURE_CATEGORY
    description: str = _FEATURE_DESCRIPTION
    required_columns: tuple[str, ...] = (_BUY_COLUMN, _SELL_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 0
    dependencies: tuple[str, ...] = ()

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append flow imbalance without mutating ``frame``.

        Args:
            frame: Input DataFrame containing ``buy_volume`` and ``sell_volume``.

        Returns:
            A new DataFrame with all original columns plus ``flow_imbalance``.

        Raises:
            FeatureExecutionError: If a required column is missing.
        """
        for column in self.required_columns:
            if column not in frame.columns:
                raise FeatureExecutionError(
                    f"required column missing: {column}",
                    error_code=_ERROR_MISSING_COLUMN,
                    details={
                        "feature": self.name,
                        "required_column": column,
                        "available_columns": tuple(frame.columns),
                    },
                )

        buy = pl.col(_BUY_COLUMN)
        sell = pl.col(_SELL_COLUMN)
        total = buy + sell
        imbalance_expr = pl.when(total == 0).then(None).otherwise((buy - sell) / total)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            imbalance_expr.alias(_OUTPUT_COLUMN)
        )

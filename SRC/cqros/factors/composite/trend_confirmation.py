"""CQROS trend confirmation composite research factor.

Purpose:
    Combine price returns, taker flow imbalance, and open-interest momentum
    into a trend-confirmation alpha factor using Feature Engine outputs only.

Responsibilities:
    - Expose immutable ``TrendConfirmationFactor`` metadata
    - Append a ``trend_confirmation`` column using Polars expressions only
    - Fail fast when required feature columns are missing
    - Remain free of repository access, storage, and research metrics

Dependencies:
    ``polars``, ``cqros.factors.base.BaseFactor``,
    ``cqros.factors.composite._require``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``TrendConfirmationFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.factors.base import BaseFactor
from cqros.factors.composite._require import require_feature_columns

__all__ = ["TrendConfirmationFactor"]

_RETURNS_COLUMN: Final[str] = "returns"
_FLOW_COLUMN: Final[str] = "flow_imbalance"
_OI_COLUMN: Final[str] = "oi_momentum"
_OUTPUT_COLUMN: Final[str] = "trend_confirmation"
_FACTOR_NAME: Final[str] = "trend_confirmation"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "composite"
_FACTOR_DESCRIPTION: Final[str] = (
    "Trend confirmation as returns * flow_imbalance * oi_momentum from " "Feature Engine outputs."
)
_ERROR_MISSING_FEATURE: Final[str] = "FACTOR-TREND-CONFIRMATION-001"
_REQUIRED_FEATURES: Final[tuple[str, ...]] = (
    _RETURNS_COLUMN,
    _FLOW_COLUMN,
    _OI_COLUMN,
)


@dataclass(frozen=True, slots=True)
class TrendConfirmationFactor(BaseFactor):
    """Trend confirmation composite from price, flow, and open interest.

    Computes ``returns * flow_imbalance * oi_momentum`` and appends the
    result as ``trend_confirmation``. Nulls propagate; missing values are
    never filled. The input DataFrame is never mutated.

    Positive values indicate price direction aligned with taker flow and
    open-interest expansion. This factor consumes Feature Engine columns
    only and does not read raw repository data.

    Attributes:
        name: Stable factor identifier (``trend_confirmation``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``composite``).
        required_features: Feature columns required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Additional warm-up rows (none; features carry warm-up).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = _REQUIRED_FEATURES
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 0

    def __post_init__(self) -> None:
        """Validate base metadata invariants.

        Raises:
            ValidationError: If any metadata invariant is violated.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append trend confirmation without mutating ``frame``.

        Args:
            frame: Research DataFrame containing required feature columns.

        Returns:
            A new DataFrame with all original columns plus
            ``trend_confirmation``.

        Raises:
            FactorError: If a required feature column is missing.
        """
        require_feature_columns(
            frame,
            self.required_features,
            factor=self.name,
            error_code=_ERROR_MISSING_FEATURE,
        )
        signal = pl.col(_RETURNS_COLUMN) * pl.col(_FLOW_COLUMN) * pl.col(_OI_COLUMN)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            signal.alias(_OUTPUT_COLUMN)
        )

"""CQROS funding divergence composite research factor.

Purpose:
    Combine price returns and inverted funding z-score into a funding-
    divergence alpha factor using Feature Engine outputs only.

Responsibilities:
    - Expose immutable ``FundingDivergenceFactor`` metadata
    - Append a ``funding_divergence`` column using Polars expressions only
    - Fail fast when required feature columns are missing
    - Remain free of repository access, storage, and research metrics

Dependencies:
    ``polars``, ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.composite._require``.

Public API:
    ``FundingDivergenceFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.factors.base import BaseFactor
from cqros.factors.composite._require import require_feature_columns

__all__ = ["FundingDivergenceFactor"]

_RETURNS_COLUMN: Final[str] = "returns"
_FUNDING_COLUMN: Final[str] = "funding_zscore"
_OUTPUT_COLUMN: Final[str] = "funding_divergence"
_FACTOR_NAME: Final[str] = "funding_divergence"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "composite"
_FACTOR_DESCRIPTION: Final[str] = (
    "Funding divergence as returns * (-funding_zscore) from Feature Engine " "outputs."
)
_ERROR_MISSING_FEATURE: Final[str] = "FACTOR-FUNDING-DIVERGENCE-001"
_REQUIRED_FEATURES: Final[tuple[str, ...]] = (_RETURNS_COLUMN, _FUNDING_COLUMN)


@dataclass(frozen=True, slots=True)
class FundingDivergenceFactor(BaseFactor):
    """Funding divergence composite from price returns and funding z-score.

    Computes ``returns * (-funding_zscore)`` and appends the result as
    ``funding_divergence``. Nulls propagate; missing values are never filled.
    The input DataFrame is never mutated.

    Positive values indicate price rising while funding is below its recent
    mean (or price falling while funding is elevated). This factor consumes
    Feature Engine columns only and does not read raw repository data.

    Attributes:
        name: Stable factor identifier (``funding_divergence``).
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
        """Append funding divergence without mutating ``frame``.

        Args:
            frame: Research DataFrame containing required feature columns.

        Returns:
            A new DataFrame with all original columns plus
            ``funding_divergence``.

        Raises:
            FactorError: If a required feature column is missing.
        """
        require_feature_columns(
            frame,
            self.required_features,
            factor=self.name,
            error_code=_ERROR_MISSING_FEATURE,
        )
        signal = pl.col(_RETURNS_COLUMN) * (pl.lit(-1.0) * pl.col(_FUNDING_COLUMN))
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            signal.alias(_OUTPUT_COLUMN)
        )

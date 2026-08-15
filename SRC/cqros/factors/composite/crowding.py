"""CQROS crowding composite research factor.

Purpose:
    Combine long/short crowding score and funding z-score into a crowding
    alpha factor using Feature Engine outputs only.

Responsibilities:
    - Expose immutable ``CrowdingFactor`` metadata
    - Append a ``crowding`` column using Polars expressions only
    - Fail fast when required feature columns are missing
    - Remain free of repository access, storage, and research metrics

Dependencies:
    ``polars``, ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.composite._require``.

Public API:
    ``CrowdingFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.factors.base import BaseFactor
from cqros.factors.composite._require import require_feature_columns

__all__ = ["CrowdingFactor"]

_CROWDING_COLUMN: Final[str] = "crowding_score"
_FUNDING_COLUMN: Final[str] = "funding_zscore"
_OUTPUT_COLUMN: Final[str] = "crowding"
_FACTOR_NAME: Final[str] = "crowding"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "composite"
_FACTOR_DESCRIPTION: Final[str] = (
    "Crowding as crowding_score * funding_zscore from Feature Engine outputs."
)
_ERROR_MISSING_FEATURE: Final[str] = "FACTOR-CROWDING-001"
_REQUIRED_FEATURES: Final[tuple[str, ...]] = (_CROWDING_COLUMN, _FUNDING_COLUMN)


@dataclass(frozen=True, slots=True)
class CrowdingFactor(BaseFactor):
    """Crowding composite from long/short crowding and funding extremity.

    Computes ``crowding_score * funding_zscore`` and appends the result as
    ``crowding``. Nulls propagate; missing values are never filled. The
    input DataFrame is never mutated.

    Large absolute values indicate positioning crowded in the same direction
    as funding extremity. This factor consumes Feature Engine columns only
    and does not read raw repository data.

    Attributes:
        name: Stable factor identifier (``crowding``).
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
        """Append crowding without mutating ``frame``.

        Args:
            frame: Research DataFrame containing required feature columns.

        Returns:
            A new DataFrame with all original columns plus ``crowding``.

        Raises:
            FactorError: If a required feature column is missing.
        """
        require_feature_columns(
            frame,
            self.required_features,
            factor=self.name,
            error_code=_ERROR_MISSING_FEATURE,
        )
        signal = pl.col(_CROWDING_COLUMN) * pl.col(_FUNDING_COLUMN)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            signal.alias(_OUTPUT_COLUMN)
        )

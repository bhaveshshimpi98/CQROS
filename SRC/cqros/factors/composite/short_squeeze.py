"""CQROS short squeeze composite research factor.

Purpose:
    Combine inverted crowding score, price returns, and taker flow imbalance
    into a short-squeeze alpha factor using Feature Engine outputs only.

Responsibilities:
    - Expose immutable ``ShortSqueezeFactor`` metadata
    - Append a ``short_squeeze`` column using Polars expressions only
    - Fail fast when required feature columns are missing
    - Remain free of repository access, storage, and research metrics

Dependencies:
    ``polars``, ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.composite._require``.

Public API:
    ``ShortSqueezeFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.factors.base import BaseFactor
from cqros.factors.composite._require import require_feature_columns

__all__ = ["ShortSqueezeFactor"]

_CROWDING_COLUMN: Final[str] = "crowding_score"
_RETURNS_COLUMN: Final[str] = "returns"
_FLOW_COLUMN: Final[str] = "flow_imbalance"
_OUTPUT_COLUMN: Final[str] = "short_squeeze"
_FACTOR_NAME: Final[str] = "short_squeeze"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "composite"
_FACTOR_DESCRIPTION: Final[str] = (
    "Short squeeze as (-crowding_score) * returns * flow_imbalance from " "Feature Engine outputs."
)
_ERROR_MISSING_FEATURE: Final[str] = "FACTOR-SHORT-SQUEEZE-001"
_REQUIRED_FEATURES: Final[tuple[str, ...]] = (
    _CROWDING_COLUMN,
    _RETURNS_COLUMN,
    _FLOW_COLUMN,
)


@dataclass(frozen=True, slots=True)
class ShortSqueezeFactor(BaseFactor):
    """Short squeeze composite from crowding, returns, and taker flow.

    Computes ``(-crowding_score) * returns * flow_imbalance`` and appends
    the result as ``short_squeeze``. Nulls propagate; missing values are
    never filled. The input DataFrame is never mutated.

    Positive values indicate short crowding pressed by rising prices and
    buy-side flow. This factor consumes Feature Engine columns only and does
    not read raw repository data.

    Attributes:
        name: Stable factor identifier (``short_squeeze``).
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
        """Append short squeeze without mutating ``frame``.

        Args:
            frame: Research DataFrame containing required feature columns.

        Returns:
            A new DataFrame with all original columns plus ``short_squeeze``.

        Raises:
            FactorError: If a required feature column is missing.
        """
        require_feature_columns(
            frame,
            self.required_features,
            factor=self.name,
            error_code=_ERROR_MISSING_FEATURE,
        )
        signal = (
            (pl.lit(-1.0) * pl.col(_CROWDING_COLUMN))
            * pl.col(_RETURNS_COLUMN)
            * pl.col(_FLOW_COLUMN)
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            signal.alias(_OUTPUT_COLUMN)
        )

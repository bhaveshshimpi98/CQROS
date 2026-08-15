"""CQROS long squeeze composite research factor.

Purpose:
    Combine crowding score, inverted price returns, and taker sell pressure
    into a long-squeeze alpha factor using Feature Engine outputs only.

Responsibilities:
    - Expose immutable ``LongSqueezeFactor`` metadata
    - Append a ``long_squeeze`` column using Polars expressions only
    - Fail fast when required feature columns are missing
    - Remain free of repository access, storage, and research metrics

Dependencies:
    ``polars``, ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.composite._require``.

Public API:
    ``LongSqueezeFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.factors.base import BaseFactor
from cqros.factors.composite._require import require_feature_columns

__all__ = ["LongSqueezeFactor"]

_CROWDING_COLUMN: Final[str] = "crowding_score"
_RETURNS_COLUMN: Final[str] = "returns"
_SELL_PRESSURE_COLUMN: Final[str] = "sell_pressure"
_OUTPUT_COLUMN: Final[str] = "long_squeeze"
_FACTOR_NAME: Final[str] = "long_squeeze"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "composite"
_FACTOR_DESCRIPTION: Final[str] = (
    "Long squeeze as crowding_score * (-returns) * sell_pressure from " "Feature Engine outputs."
)
_ERROR_MISSING_FEATURE: Final[str] = "FACTOR-LONG-SQUEEZE-001"
_REQUIRED_FEATURES: Final[tuple[str, ...]] = (
    _CROWDING_COLUMN,
    _RETURNS_COLUMN,
    _SELL_PRESSURE_COLUMN,
)


@dataclass(frozen=True, slots=True)
class LongSqueezeFactor(BaseFactor):
    """Long squeeze composite from crowding, returns, and sell pressure.

    Computes ``crowding_score * (-returns) * sell_pressure`` and appends the
    result as ``long_squeeze``. Nulls propagate; missing values are never
    filled. The input DataFrame is never mutated.

    Positive values indicate long crowding pressed by falling prices and
    sell-side pressure. This factor consumes Feature Engine columns only and
    does not read raw repository data.

    Attributes:
        name: Stable factor identifier (``long_squeeze``).
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
        """Append long squeeze without mutating ``frame``.

        Args:
            frame: Research DataFrame containing required feature columns.

        Returns:
            A new DataFrame with all original columns plus ``long_squeeze``.

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
            pl.col(_CROWDING_COLUMN)
            * (pl.lit(-1.0) * pl.col(_RETURNS_COLUMN))
            * pl.col(_SELL_PRESSURE_COLUMN)
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            signal.alias(_OUTPUT_COLUMN)
        )

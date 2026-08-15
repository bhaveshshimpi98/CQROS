"""CQROS position build-up composite research factor.

Purpose:
    Combine open-interest momentum and long/short ratio momentum into a
    position build-up alpha factor using Feature Engine outputs only.

Responsibilities:
    - Expose immutable ``PositionBuildUpFactor`` metadata
    - Append a ``position_build_up`` column using Polars expressions only
    - Fail fast when required feature columns are missing
    - Remain free of repository access, storage, and research metrics

Dependencies:
    ``polars``, ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.composite._require``.

Public API:
    ``PositionBuildUpFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.factors.base import BaseFactor
from cqros.factors.composite._require import require_feature_columns

__all__ = ["PositionBuildUpFactor"]

_OI_COLUMN: Final[str] = "oi_momentum"
_RATIO_COLUMN: Final[str] = "ratio_momentum"
_OUTPUT_COLUMN: Final[str] = "position_build_up"
_FACTOR_NAME: Final[str] = "position_build_up"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "composite"
_FACTOR_DESCRIPTION: Final[str] = (
    "Position build-up as oi_momentum * ratio_momentum from Feature Engine " "outputs."
)
_ERROR_MISSING_FEATURE: Final[str] = "FACTOR-POSITION-BUILD-UP-001"
_REQUIRED_FEATURES: Final[tuple[str, ...]] = (_OI_COLUMN, _RATIO_COLUMN)


@dataclass(frozen=True, slots=True)
class PositionBuildUpFactor(BaseFactor):
    """Position build-up composite from open interest and long/short momentum.

    Computes ``oi_momentum * ratio_momentum`` and appends the result as
    ``position_build_up``. Nulls propagate; missing values are never filled.
    The input DataFrame is never mutated.

    Positive values indicate concurrent expansion of open interest and
    directional long/short positioning. This factor consumes Feature Engine
    columns only and does not read raw repository data.

    Attributes:
        name: Stable factor identifier (``position_build_up``).
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
        """Append position build-up without mutating ``frame``.

        Args:
            frame: Research DataFrame containing required feature columns.

        Returns:
            A new DataFrame with all original columns plus
            ``position_build_up``.

        Raises:
            FactorError: If a required feature column is missing.
        """
        require_feature_columns(
            frame,
            self.required_features,
            factor=self.name,
            error_code=_ERROR_MISSING_FEATURE,
        )
        signal = pl.col(_OI_COLUMN) * pl.col(_RATIO_COLUMN)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            signal.alias(_OUTPUT_COLUMN)
        )

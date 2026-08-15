"""CQROS leveraged long build-up composite research factor.

Purpose:
    Combine funding z-score, open-interest momentum, and taker buy pressure
    into a leveraged-long build-up alpha factor using Feature Engine outputs
    only.

Responsibilities:
    - Expose immutable ``LeveragedLongBuildUpFactor`` metadata
    - Append a ``leveraged_long_build_up`` column using Polars expressions only
    - Fail fast when required feature columns are missing
    - Remain free of repository access, storage, and research metrics

Dependencies:
    ``polars``, ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.composite._require``.

Public API:
    ``LeveragedLongBuildUpFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.factors.base import BaseFactor
from cqros.factors.composite._require import require_feature_columns

__all__ = ["LeveragedLongBuildUpFactor"]

_FUNDING_COLUMN: Final[str] = "funding_zscore"
_OI_COLUMN: Final[str] = "oi_momentum"
_BUY_PRESSURE_COLUMN: Final[str] = "buy_pressure"
_OUTPUT_COLUMN: Final[str] = "leveraged_long_build_up"
_FACTOR_NAME: Final[str] = "leveraged_long_build_up"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "composite"
_FACTOR_DESCRIPTION: Final[str] = (
    "Leveraged long build-up as funding_zscore * oi_momentum * buy_pressure "
    "from Feature Engine outputs."
)
_ERROR_MISSING_FEATURE: Final[str] = "FACTOR-LEVERAGED-LONG-BUILD-UP-001"
_REQUIRED_FEATURES: Final[tuple[str, ...]] = (
    _FUNDING_COLUMN,
    _OI_COLUMN,
    _BUY_PRESSURE_COLUMN,
)


@dataclass(frozen=True, slots=True)
class LeveragedLongBuildUpFactor(BaseFactor):
    """Leveraged long build-up from funding, open interest, and buy pressure.

    Computes ``funding_zscore * oi_momentum * buy_pressure`` and appends the
    result as ``leveraged_long_build_up``. Nulls propagate; missing values
    are never filled. The input DataFrame is never mutated.

    Positive values indicate elevated funding alongside open-interest growth
    and aggressive buying. This factor consumes Feature Engine columns only
    and does not read raw repository data.

    Attributes:
        name: Stable factor identifier (``leveraged_long_build_up``).
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
        """Append leveraged long build-up without mutating ``frame``.

        Args:
            frame: Research DataFrame containing required feature columns.

        Returns:
            A new DataFrame with all original columns plus
            ``leveraged_long_build_up``.

        Raises:
            FactorError: If a required feature column is missing.
        """
        require_feature_columns(
            frame,
            self.required_features,
            factor=self.name,
            error_code=_ERROR_MISSING_FEATURE,
        )
        signal = pl.col(_FUNDING_COLUMN) * pl.col(_OI_COLUMN) * pl.col(_BUY_PRESSURE_COLUMN)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            signal.alias(_OUTPUT_COLUMN)
        )

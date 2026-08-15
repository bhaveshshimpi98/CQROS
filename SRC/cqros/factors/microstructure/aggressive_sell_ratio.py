"""CQROS aggressive sell ratio research factor.

Purpose:
    Compute rolling aggressive sell volume share of total volume as a pure
    microstructure alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``AggressiveSellRatioFactor`` metadata
    - Append an ``aggressive_sell_ratio`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``AggressiveSellRatioFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["AggressiveSellRatioFactor"]

_SELL_COLUMN: Final[str] = "taker_sell_volume"
_VOLUME_COLUMN: Final[str] = "volume"
_OUTPUT_COLUMN: Final[str] = "aggressive_sell_ratio"
_FACTOR_NAME: Final[str] = "aggressive_sell_ratio"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "microstructure"
_FACTOR_DESCRIPTION: Final[str] = (
    "Aggressive sell ratio as rolling taker sell volume over rolling total volume."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-AGGRESSIVE-SELL-RATIO-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-AGGRESSIVE-SELL-RATIO-002"


@dataclass(frozen=True, slots=True)
class AggressiveSellRatioFactor(BaseFactor):
    """Aggressive sell ratio alpha factor from taker sell volume and volume.

    Computes ``sum(taker_sell_volume, lookback) / sum(volume, lookback)`` and
    appends the result as ``aggressive_sell_ratio``. Returns null when the
    rolling volume sum is zero. Incomplete windows are null. Missing values
    are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``aggressive_sell_ratio``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``microstructure``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_SELL_COLUMN, _VOLUME_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 20

    def __post_init__(self) -> None:
        """Validate base metadata and require a strictly positive lookback.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback <= 0``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback < 1:
            raise ValidationError(
                "lookback must be an integer greater than 0",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append aggressive sell ratio without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``taker_sell_volume`` and
                ``volume`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``aggressive_sell_ratio``. Incomplete windows are null.

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

        sell_sum = pl.col(_SELL_COLUMN).rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        volume_sum = pl.col(_VOLUME_COLUMN).rolling_sum(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        ratio_expr = (
            pl.when(volume_sum != 0).then(sell_sum / volume_sum).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            ratio_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

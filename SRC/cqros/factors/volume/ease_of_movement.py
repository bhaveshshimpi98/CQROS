"""CQROS Ease of Movement research factor.

Purpose:
    Compute Arm's Ease of Movement from high, low, and volume as a pure
    volume alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``EaseOfMovementFactor`` metadata
    - Append an ``ease_of_movement`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``EaseOfMovementFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["EaseOfMovementFactor"]

_HIGH_COLUMN: Final[str] = "high"
_LOW_COLUMN: Final[str] = "low"
_VOLUME_COLUMN: Final[str] = "volume"
_OUTPUT_COLUMN: Final[str] = "ease_of_movement"
_FACTOR_NAME: Final[str] = "ease_of_movement"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "volume"
_FACTOR_DESCRIPTION: Final[str] = (
    "Ease of Movement as smoothed midprice distance scaled by range over volume."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-EASE-OF-MOVEMENT-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-EASE-OF-MOVEMENT-002"


@dataclass(frozen=True, slots=True)
class EaseOfMovementFactor(BaseFactor):
    """Ease of Movement alpha factor from high, low, and volume.

    Computes one-period EMV as
    ``((mid - prev_mid) * (high - low)) / volume`` where
    ``mid = (high + low) / 2``, then appends the simple moving average over
    ``lookback`` as ``ease_of_movement``. Returns null when the quantity is
    undefined: ``volume == 0``, ``high == low``, or any other zero
    denominator. Never emits Inf or NaN from division by zero. Incomplete
    windows are null. Missing values are never filled. The input DataFrame
    is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``ease_of_movement``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``volume``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: EMV smoothing window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_HIGH_COLUMN, _LOW_COLUMN, _VOLUME_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 14

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
        """Append Ease of Movement without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``high``, ``low``, and
                ``volume`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``ease_of_movement``. Incomplete windows are null.

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

        high = pl.col(_HIGH_COLUMN)
        low = pl.col(_LOW_COLUMN)
        volume = pl.col(_VOLUME_COLUMN)
        mid = (high + low) / 2.0
        distance = mid - mid.shift(1)  # pyright: ignore[reportUnknownMemberType]
        range_hl = high - low
        # EMV = distance * range / volume. Undefined when volume == 0 or
        # high == low (zero box-ratio denominator). Prefer NULL over Inf/NaN.
        zero_denominator = (volume == 0) | (range_hl == 0)
        one_period = pl.when(zero_denominator).then(None).otherwise(distance * range_hl / volume)
        emv_expr = one_period.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            emv_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

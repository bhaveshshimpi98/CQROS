"""CQROS open interest / volume ratio research factor.

Purpose:
    Compute the ratio of rolling-mean open interest to rolling-mean volume
    as a pure open-interest alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``OpenInterestVolumeRatioFactor`` metadata
    - Append an ``open_interest_volume_ratio`` column using Polars
      expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``OpenInterestVolumeRatioFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["OpenInterestVolumeRatioFactor"]

_OI_COLUMN: Final[str] = "open_interest"
_VOLUME_COLUMN: Final[str] = "volume"
_OUTPUT_COLUMN: Final[str] = "open_interest_volume_ratio"
_FACTOR_NAME: Final[str] = "open_interest_volume_ratio"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "open_interest"
_FACTOR_DESCRIPTION: Final[str] = "Ratio of rolling-mean open interest to rolling-mean volume."
_ERROR_LOOKBACK: Final[str] = "FACTOR-OPEN-INTEREST-VOLUME-RATIO-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-OPEN-INTEREST-VOLUME-RATIO-002"


@dataclass(frozen=True, slots=True)
class OpenInterestVolumeRatioFactor(BaseFactor):
    """Open interest / volume ratio alpha factor from rolling means.

    Computes ``rolling_mean(open_interest) / rolling_mean(volume)`` over
    ``lookback`` and appends the result as ``open_interest_volume_ratio``.
    Returns null when the rolling volume mean is zero. Incomplete windows
    are null. Missing values are never filled. The input DataFrame is never
    mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``open_interest_volume_ratio``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``open_interest``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling mean window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_OI_COLUMN, _VOLUME_COLUMN)
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
        """Append open interest / volume ratio without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``open_interest`` and
                ``volume`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``open_interest_volume_ratio``. Incomplete windows are null.

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

        oi_mean = pl.col(_OI_COLUMN).rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        volume_mean = pl.col(
            _VOLUME_COLUMN
        ).rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        ratio_expr = (
            pl.when(volume_mean != 0).then(oi_mean / volume_mean).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            ratio_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

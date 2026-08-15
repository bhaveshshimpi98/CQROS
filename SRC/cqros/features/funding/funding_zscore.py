"""CQROS funding rate z-score feature.

Purpose:
    Compute the rolling z-score of funding rate from raw funding repository
    data.

Responsibilities:
    - Expose immutable ``FundingZScoreFeature`` metadata
    - Append a ``funding_zscore`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``funding_rate``

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``FundingZScoreFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["FundingZScoreFeature"]

_RATE_COLUMN: Final[str] = "funding_rate"
_OUTPUT_COLUMN: Final[str] = "funding_zscore"
_FEATURE_NAME: Final[str] = "funding_zscore"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "funding"
_FEATURE_DESCRIPTION: Final[str] = "Rolling z-score of funding rate over a lookback window."
_ERROR_LOOKBACK: Final[str] = "FEATURE-FUNDING-ZSCORE-001"
_ERROR_MISSING_RATE: Final[str] = "FEATURE-FUNDING-ZSCORE-002"


@dataclass(frozen=True, slots=True)
class FundingZScoreFeature(BaseFeature):
    """Rolling z-score of the funding rate.

    Computes ``(funding_rate - rolling_mean) / rolling_std`` over ``lookback``.
    When rolling standard deviation is zero the output is ``0.0`` (zero
    deviation from a constant window mean). The first ``lookback - 1`` rows
    are null.

    Attributes:
        name: Stable feature identifier (``funding_zscore``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``funding``).
        description: Human-readable feature summary.
        required_columns: Input columns required by ``transform``.
        produced_columns: Output columns produced by ``transform``.
        lookback: Rolling window size (must be > 0).
        dependencies: Upstream feature names (none for this feature).
    """

    name: str = _FEATURE_NAME
    version: str = _FEATURE_VERSION
    category: str = _FEATURE_CATEGORY
    description: str = _FEATURE_DESCRIPTION
    required_columns: tuple[str, ...] = (_RATE_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 20
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate base metadata and require a strictly positive lookback.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback <= 0``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFeature.__post_init__(self)
        if self.lookback < 1:
            raise ValidationError(
                "lookback must be an integer greater than 0",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append funding z-score without mutating ``frame``.

        Args:
            frame: Input DataFrame containing a ``funding_rate`` column.

        Returns:
            A new DataFrame with all original columns plus ``funding_zscore``.

        Raises:
            FeatureExecutionError: If ``funding_rate`` is not present.
        """
        if _RATE_COLUMN not in frame.columns:
            raise FeatureExecutionError(
                f"required column missing: {_RATE_COLUMN}",
                error_code=_ERROR_MISSING_RATE,
                details={
                    "feature": self.name,
                    "required_column": _RATE_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        rate = pl.col(_RATE_COLUMN)
        mean = rate.rolling_mean(
            window_size=self.lookback
        )  # pyright: ignore[reportUnknownMemberType]
        std = rate.rolling_std(
            window_size=self.lookback
        )  # pyright: ignore[reportUnknownMemberType]
        # Zero variance means every observation equals the window mean → z-score 0.
        zscore_expr = pl.when(std == 0).then(0.0).otherwise((rate - mean) / std)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            zscore_expr.alias(_OUTPUT_COLUMN)
        )

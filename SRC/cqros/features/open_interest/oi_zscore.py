"""CQROS open interest z-score feature.

Purpose:
    Compute the rolling z-score of open interest from raw open-interest
    repository data.

Responsibilities:
    - Expose immutable ``OIZScoreFeature`` metadata
    - Append an ``oi_zscore`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``open_interest``

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``OIZScoreFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["OIZScoreFeature"]

_OI_COLUMN: Final[str] = "open_interest"
_OUTPUT_COLUMN: Final[str] = "oi_zscore"
_FEATURE_NAME: Final[str] = "oi_zscore"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "open_interest"
_FEATURE_DESCRIPTION: Final[str] = "Rolling z-score of open interest over a lookback window."
_ERROR_LOOKBACK: Final[str] = "FEATURE-OI-ZSCORE-001"
_ERROR_MISSING_OI: Final[str] = "FEATURE-OI-ZSCORE-002"


@dataclass(frozen=True, slots=True)
class OIZScoreFeature(BaseFeature):
    """Rolling z-score of open interest.

    Computes ``(open_interest - rolling_mean) / rolling_std`` over ``lookback``.
    When rolling standard deviation is zero the output is null. The first
    ``lookback - 1`` rows are null.

    Attributes:
        name: Stable feature identifier (``oi_zscore``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``open_interest``).
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
    required_columns: tuple[str, ...] = (_OI_COLUMN,)
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
        """Append OI z-score without mutating ``frame``.

        Args:
            frame: Input DataFrame containing an ``open_interest`` column.

        Returns:
            A new DataFrame with all original columns plus ``oi_zscore``.

        Raises:
            FeatureExecutionError: If ``open_interest`` is not present.
        """
        if _OI_COLUMN not in frame.columns:
            raise FeatureExecutionError(
                f"required column missing: {_OI_COLUMN}",
                error_code=_ERROR_MISSING_OI,
                details={
                    "feature": self.name,
                    "required_column": _OI_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        oi = pl.col(_OI_COLUMN)
        mean = oi.rolling_mean(
            window_size=self.lookback
        )  # pyright: ignore[reportUnknownMemberType]
        std = oi.rolling_std(window_size=self.lookback)  # pyright: ignore[reportUnknownMemberType]
        zscore_expr = pl.when(std == 0).then(None).otherwise((oi - mean) / std)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            zscore_expr.alias(_OUTPUT_COLUMN)
        )

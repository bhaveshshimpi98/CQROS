"""CQROS long/short ratio z-score feature.

Purpose:
    Compute the rolling z-score of long/short ratio from raw long/short
    repository data.

Responsibilities:
    - Expose immutable ``RatioZScoreFeature`` metadata
    - Append a ``ratio_zscore`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``long_short_ratio``

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``RatioZScoreFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["RatioZScoreFeature"]

_RATIO_COLUMN: Final[str] = "long_short_ratio"
_OUTPUT_COLUMN: Final[str] = "ratio_zscore"
_FEATURE_NAME: Final[str] = "ratio_zscore"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "long_short"
_FEATURE_DESCRIPTION: Final[str] = "Rolling z-score of long/short ratio over a lookback window."
_ERROR_LOOKBACK: Final[str] = "FEATURE-RATIO-ZSCORE-001"
_ERROR_MISSING_RATIO: Final[str] = "FEATURE-RATIO-ZSCORE-002"


@dataclass(frozen=True, slots=True)
class RatioZScoreFeature(BaseFeature):
    """Rolling z-score of the long/short ratio.

    Computes ``(long_short_ratio - rolling_mean) / rolling_std`` over
    ``lookback``. When rolling standard deviation is zero the output is null.
    The first ``lookback - 1`` rows are null.

    Attributes:
        name: Stable feature identifier (``ratio_zscore``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``long_short``).
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
    required_columns: tuple[str, ...] = (_RATIO_COLUMN,)
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
        """Append ratio z-score without mutating ``frame``.

        Args:
            frame: Input DataFrame containing a ``long_short_ratio`` column.

        Returns:
            A new DataFrame with all original columns plus ``ratio_zscore``.

        Raises:
            FeatureExecutionError: If ``long_short_ratio`` is not present.
        """
        if _RATIO_COLUMN not in frame.columns:
            raise FeatureExecutionError(
                f"required column missing: {_RATIO_COLUMN}",
                error_code=_ERROR_MISSING_RATIO,
                details={
                    "feature": self.name,
                    "required_column": _RATIO_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        ratio = pl.col(_RATIO_COLUMN)
        mean = ratio.rolling_mean(
            window_size=self.lookback
        )  # pyright: ignore[reportUnknownMemberType]
        std = ratio.rolling_std(
            window_size=self.lookback
        )  # pyright: ignore[reportUnknownMemberType]
        zscore_expr = pl.when(std == 0).then(None).otherwise((ratio - mean) / std)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            zscore_expr.alias(_OUTPUT_COLUMN)
        )

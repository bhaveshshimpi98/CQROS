"""CQROS long/short crowding score feature.

Purpose:
    Compute a crowding score measuring how far the long/short ratio is from
    balanced positioning (ratio = 1.0), scaled by recent volatility.

Responsibilities:
    - Expose immutable ``CrowdingScoreFeature`` metadata
    - Append a ``crowding_score`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``long_short_ratio``

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``CrowdingScoreFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["CrowdingScoreFeature"]

_RATIO_COLUMN: Final[str] = "long_short_ratio"
_OUTPUT_COLUMN: Final[str] = "crowding_score"
_FEATURE_NAME: Final[str] = "crowding_score"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "long_short"
_FEATURE_DESCRIPTION: Final[str] = (
    "Crowding score as (long_short_ratio - 1.0) / rolling_std over a lookback " "window."
)
_ERROR_LOOKBACK: Final[str] = "FEATURE-CROWDING-SCORE-001"
_ERROR_MISSING_RATIO: Final[str] = "FEATURE-CROWDING-SCORE-002"
_BALANCED_RATIO: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class CrowdingScoreFeature(BaseFeature):
    """Crowding score relative to balanced long/short positioning.

    Computes ``(long_short_ratio - 1.0) / rolling_std(long_short_ratio)`` over
    ``lookback``. Unlike the ratio z-score, this measures distance from a
    balanced book (ratio = 1.0) rather than from the rolling mean. When
    rolling standard deviation is zero the output is null. The first
    ``lookback - 1`` rows are null.

    Attributes:
        name: Stable feature identifier (``crowding_score``).
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
        """Append crowding score without mutating ``frame``.

        Args:
            frame: Input DataFrame containing a ``long_short_ratio`` column.

        Returns:
            A new DataFrame with all original columns plus ``crowding_score``.

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
        std = ratio.rolling_std(
            window_size=self.lookback
        )  # pyright: ignore[reportUnknownMemberType]
        crowding_expr = pl.when(std == 0).then(None).otherwise((ratio - _BALANCED_RATIO) / std)
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            crowding_expr.alias(_OUTPUT_COLUMN)
        )

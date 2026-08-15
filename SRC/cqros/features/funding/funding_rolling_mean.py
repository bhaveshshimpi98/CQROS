"""CQROS funding rate rolling mean feature.

Purpose:
    Compute the rolling mean of funding rate from raw funding repository data.

Responsibilities:
    - Expose immutable ``FundingRollingMeanFeature`` metadata
    - Append a ``funding_rolling_mean`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``funding_rate``

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``FundingRollingMeanFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["FundingRollingMeanFeature"]

_RATE_COLUMN: Final[str] = "funding_rate"
_OUTPUT_COLUMN: Final[str] = "funding_rolling_mean"
_FEATURE_NAME: Final[str] = "funding_rolling_mean"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "funding"
_FEATURE_DESCRIPTION: Final[str] = "Rolling mean of funding rate over a lookback window."
_ERROR_LOOKBACK: Final[str] = "FEATURE-FUNDING-ROLLING-MEAN-001"
_ERROR_MISSING_RATE: Final[str] = "FEATURE-FUNDING-ROLLING-MEAN-002"


@dataclass(frozen=True, slots=True)
class FundingRollingMeanFeature(BaseFeature):
    """Rolling mean of the funding rate.

    Computes ``funding_rate.rolling_mean(lookback)`` and appends the result as
    ``funding_rolling_mean``. The first ``lookback - 1`` rows are null.

    Attributes:
        name: Stable feature identifier (``funding_rolling_mean``).
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
        """Append funding rolling mean without mutating ``frame``.

        Args:
            frame: Input DataFrame containing a ``funding_rate`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``funding_rolling_mean``.

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

        mean_expr = pl.col(_RATE_COLUMN).rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            mean_expr.alias(_OUTPUT_COLUMN)
        )

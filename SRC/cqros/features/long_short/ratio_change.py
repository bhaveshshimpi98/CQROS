"""CQROS long/short ratio change feature.

Purpose:
    Compute one-period long/short ratio change from raw long/short repository
    data.

Responsibilities:
    - Expose immutable ``RatioChangeFeature`` metadata
    - Append a ``ratio_change`` column using Polars expressions only
    - Fail fast when ``long_short_ratio`` is missing

Dependencies:
    ``polars``, ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``RatioChangeFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["RatioChangeFeature"]

_RATIO_COLUMN: Final[str] = "long_short_ratio"
_OUTPUT_COLUMN: Final[str] = "ratio_change"
_FEATURE_NAME: Final[str] = "ratio_change"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "long_short"
_FEATURE_DESCRIPTION: Final[str] = (
    "One-period change in long/short ratio as ratio minus previous ratio."
)
_ERROR_MISSING_RATIO: Final[str] = "FEATURE-RATIO-CHANGE-001"


@dataclass(frozen=True, slots=True)
class RatioChangeFeature(BaseFeature):
    """One-period absolute change in long/short ratio.

    Computes ``long_short_ratio - long_short_ratio.shift(1)`` and appends the
    result as ``ratio_change``. The first row is null. Missing values are not
    filled.

    Attributes:
        name: Stable feature identifier (``ratio_change``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``long_short``).
        description: Human-readable feature summary.
        required_columns: Input columns required by ``transform``.
        produced_columns: Output columns produced by ``transform``.
        lookback: Warm-up rows required for a fully defined value.
        dependencies: Upstream feature names (none for this feature).
    """

    name: str = _FEATURE_NAME
    version: str = _FEATURE_VERSION
    category: str = _FEATURE_CATEGORY
    description: str = _FEATURE_DESCRIPTION
    required_columns: tuple[str, ...] = (_RATIO_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 1
    dependencies: tuple[str, ...] = ()

    @property
    def warmup_rows(self) -> int:
        """``diff(1)`` leaves the first row undefined."""
        return self.lookback

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append ratio change without mutating ``frame``.

        Args:
            frame: Input DataFrame containing a ``long_short_ratio`` column.

        Returns:
            A new DataFrame with all original columns plus ``ratio_change``.

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
        change_expr = ratio - ratio.shift(1)  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            change_expr.alias(_OUTPUT_COLUMN)
        )

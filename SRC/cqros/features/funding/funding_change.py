"""CQROS funding rate change feature.

Purpose:
    Compute one-period funding rate change from raw funding repository data.

Responsibilities:
    - Expose immutable ``FundingChangeFeature`` metadata
    - Append a ``funding_change`` column using Polars expressions only
    - Fail fast when ``funding_rate`` is missing

Dependencies:
    ``polars``, ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``FundingChangeFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["FundingChangeFeature"]

_RATE_COLUMN: Final[str] = "funding_rate"
_OUTPUT_COLUMN: Final[str] = "funding_change"
_FEATURE_NAME: Final[str] = "funding_change"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "funding"
_FEATURE_DESCRIPTION: Final[str] = (
    "One-period change in funding rate as funding_rate minus previous funding_rate."
)
_ERROR_MISSING_RATE: Final[str] = "FEATURE-FUNDING-CHANGE-001"


@dataclass(frozen=True, slots=True)
class FundingChangeFeature(BaseFeature):
    """One-period absolute change in funding rate.

    Computes ``funding_rate - funding_rate.shift(1)`` and appends the result
    as ``funding_change``. The first row is null. Missing values are not filled.

    Attributes:
        name: Stable feature identifier (``funding_change``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``funding``).
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
    required_columns: tuple[str, ...] = (_RATE_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 1
    dependencies: tuple[str, ...] = ()

    @property
    def warmup_rows(self) -> int:
        """``diff(1)`` leaves the first row undefined."""
        return self.lookback

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append funding change without mutating ``frame``.

        Args:
            frame: Input DataFrame containing a ``funding_rate`` column.

        Returns:
            A new DataFrame with all original columns plus ``funding_change``.

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
        change_expr = rate - rate.shift(1)  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            change_expr.alias(_OUTPUT_COLUMN)
        )

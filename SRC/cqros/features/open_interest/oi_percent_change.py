"""CQROS open interest percent change feature.

Purpose:
    Compute one-period open interest percent change from raw open-interest
    repository data.

Responsibilities:
    - Expose immutable ``OIPercentChangeFeature`` metadata
    - Append an ``oi_percent_change`` column using Polars expressions only
    - Fail fast when ``open_interest`` is missing

Dependencies:
    ``polars``, ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``OIPercentChangeFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["OIPercentChangeFeature"]

_OI_COLUMN: Final[str] = "open_interest"
_OUTPUT_COLUMN: Final[str] = "oi_percent_change"
_FEATURE_NAME: Final[str] = "oi_percent_change"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "open_interest"
_FEATURE_DESCRIPTION: Final[str] = (
    "One-period percent change in open interest as (OI / previous OI) - 1."
)
_ERROR_MISSING_OI: Final[str] = "FEATURE-OI-PERCENT-CHANGE-001"


@dataclass(frozen=True, slots=True)
class OIPercentChangeFeature(BaseFeature):
    """One-period percent change in open interest.

    Computes ``(open_interest / open_interest.shift(1)) - 1`` and appends the
    result as ``oi_percent_change``. The first row is null. Missing values are
    not filled.

    Attributes:
        name: Stable feature identifier (``oi_percent_change``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``open_interest``).
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
    required_columns: tuple[str, ...] = (_OI_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 1
    dependencies: tuple[str, ...] = ()

    @property
    def warmup_rows(self) -> int:
        """``pct_change(1)`` leaves the first row undefined."""
        return self.lookback

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append OI percent change without mutating ``frame``.

        Args:
            frame: Input DataFrame containing an ``open_interest`` column.

        Returns:
            A new DataFrame with all original columns plus ``oi_percent_change``.

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
        pct_expr = (oi / oi.shift(1)) - 1  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            pct_expr.alias(_OUTPUT_COLUMN)
        )

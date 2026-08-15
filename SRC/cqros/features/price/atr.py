"""CQROS average true range (ATR) feature.

Purpose:
    Compute average true range from OHLC columns for research feature
    engineering.

Responsibilities:
    - Expose immutable ``ATRFeature`` metadata
    - Append an ``atr`` column using Polars expressions only
    - Fail fast on invalid lookback and missing OHLC columns

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``ATRFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["ATRFeature"]

_HIGH_COLUMN: Final[str] = "high"
_LOW_COLUMN: Final[str] = "low"
_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "atr"
_FEATURE_NAME: Final[str] = "atr"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "price"
_FEATURE_DESCRIPTION: Final[str] = (
    "Average true range as the rolling mean of true range over a lookback window."
)
_ERROR_LOOKBACK: Final[str] = "FEATURE-ATR-001"
_ERROR_MISSING_COLUMN: Final[str] = "FEATURE-ATR-002"


@dataclass(frozen=True, slots=True)
class ATRFeature(BaseFeature):
    """Average true range from high, low, and close.

    True range is ``max(high - low, |high - prev_close|, |low - prev_close|)``.
    ATR is the rolling mean of true range over ``lookback``. The first
    ``lookback - 1`` ATR rows are null. Missing values are never filled.

    Attributes:
        name: Stable feature identifier (``atr``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``price``).
        description: Human-readable feature summary.
        required_columns: Input columns required by ``transform``.
        produced_columns: Output columns produced by ``transform``.
        lookback: Rolling ATR window size (must be > 0).
        dependencies: Upstream feature names (none for this feature).
    """

    name: str = _FEATURE_NAME
    version: str = _FEATURE_VERSION
    category: str = _FEATURE_CATEGORY
    description: str = _FEATURE_DESCRIPTION
    required_columns: tuple[str, ...] = (_HIGH_COLUMN, _LOW_COLUMN, _CLOSE_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 14
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
        """Append ATR without mutating ``frame``.

        Args:
            frame: Input DataFrame containing ``high``, ``low``, and ``close``.

        Returns:
            A new DataFrame with all original columns plus ``atr``.

        Raises:
            FeatureExecutionError: If a required column is missing.
        """
        for column in self.required_columns:
            if column not in frame.columns:
                raise FeatureExecutionError(
                    f"required column missing: {column}",
                    error_code=_ERROR_MISSING_COLUMN,
                    details={
                        "feature": self.name,
                        "required_column": column,
                        "available_columns": tuple(frame.columns),
                    },
                )

        prev_close = pl.col(_CLOSE_COLUMN).shift(1)
        true_range = pl.max_horizontal(
            pl.col(_HIGH_COLUMN) - pl.col(_LOW_COLUMN),
            (pl.col(_HIGH_COLUMN) - prev_close).abs(),
            (pl.col(_LOW_COLUMN) - prev_close).abs(),
        )
        atr_expr = true_range.rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            atr_expr.alias(_OUTPUT_COLUMN)
        )

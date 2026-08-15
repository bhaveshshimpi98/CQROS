"""CQROS log returns feature.

Purpose:
    Compute logarithmic returns from the close price column for research
    feature engineering.

Responsibilities:
    - Expose immutable ``LogReturnsFeature`` metadata
    - Append a ``log_returns`` column using Polars expressions only
    - Fail fast when the required ``close`` column is missing

Dependencies:
    ``polars``, ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``LogReturnsFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["LogReturnsFeature"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "log_returns"
_FEATURE_NAME: Final[str] = "log_returns"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "price"
_FEATURE_DESCRIPTION: Final[str] = "Natural log returns computed as ln(close / previous close)."
_ERROR_MISSING_CLOSE: Final[str] = "FEATURE-LOG-RETURNS-001"


@dataclass(frozen=True, slots=True)
class LogReturnsFeature(BaseFeature):
    """Natural logarithmic returns from the close price.

    Computes ``ln(close / close.shift(1))`` and appends the result as
    ``log_returns``. The first row is null. Missing values are not filled.

    Attributes:
        name: Stable feature identifier (``log_returns``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``price``).
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
    required_columns: tuple[str, ...] = (_CLOSE_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 1
    dependencies: tuple[str, ...] = ()

    @property
    def warmup_rows(self) -> int:
        """``shift(1)`` leaves the first row undefined."""
        return self.lookback

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append log returns without mutating ``frame``.

        Args:
            frame: Input DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus ``log_returns``.

        Raises:
            FeatureExecutionError: If ``close`` is not present in ``frame``.
        """
        if _CLOSE_COLUMN not in frame.columns:
            raise FeatureExecutionError(
                f"required column missing: {_CLOSE_COLUMN}",
                error_code=_ERROR_MISSING_CLOSE,
                details={
                    "feature": self.name,
                    "required_column": _CLOSE_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        close = pl.col(_CLOSE_COLUMN)
        log_returns_expr = (
            close / close.shift(1)
        ).log()  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            log_returns_expr.alias(_OUTPUT_COLUMN)
        )

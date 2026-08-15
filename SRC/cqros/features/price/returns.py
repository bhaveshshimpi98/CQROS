"""CQROS simple percentage returns feature.

Purpose:
    Compute simple percentage returns from the close price column as the
    reference production feature implementation for the Feature Engine.

Responsibilities:
    - Expose immutable ``ReturnsFeature`` metadata
    - Append a ``returns`` column using Polars expressions only
    - Fail fast when the required ``close`` column is missing

Dependencies:
    ``polars``, ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``ReturnsFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["ReturnsFeature"]

_CLOSE_COLUMN: Final[str] = "close"
_RETURNS_COLUMN: Final[str] = "returns"
_FEATURE_NAME: Final[str] = "returns"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "price"
_FEATURE_DESCRIPTION: Final[str] = "Simple percentage returns computed from the close column."
_ERROR_MISSING_CLOSE: Final[str] = "FEATURE-RETURNS-001"


@dataclass(frozen=True, slots=True)
class ReturnsFeature(BaseFeature):
    """Simple percentage returns from the close price.

    Computes ``(close / close.shift(1)) - 1`` and appends the result as
    ``returns``. The first row is null. Missing values are not filled.

    Attributes:
        name: Stable feature identifier (``returns``).
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
    produced_columns: tuple[str, ...] = (_RETURNS_COLUMN,)
    lookback: int = 1
    dependencies: tuple[str, ...] = ()

    @property
    def warmup_rows(self) -> int:
        """``shift(1)`` leaves the first row undefined."""
        return self.lookback

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append simple percentage returns without mutating ``frame``.

        Args:
            frame: Input DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus ``returns``.

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
        returns_expr = (close / close.shift(1)) - 1  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            returns_expr.alias(_RETURNS_COLUMN)
        )

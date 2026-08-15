"""CQROS dollar volume feature.

Purpose:
    Compute dollar volume from close price and traded volume for research
    feature engineering.

Responsibilities:
    - Expose immutable ``DollarVolumeFeature`` metadata
    - Append a ``dollar_volume`` column using Polars expressions only
    - Fail fast when required columns are missing

Dependencies:
    ``polars``, ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``DollarVolumeFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["DollarVolumeFeature"]

_CLOSE_COLUMN: Final[str] = "close"
_VOLUME_COLUMN: Final[str] = "volume"
_OUTPUT_COLUMN: Final[str] = "dollar_volume"
_FEATURE_NAME: Final[str] = "dollar_volume"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "price"
_FEATURE_DESCRIPTION: Final[str] = "Dollar volume computed as close multiplied by volume."
_ERROR_MISSING_COLUMN: Final[str] = "FEATURE-DOLLAR-VOLUME-001"


@dataclass(frozen=True, slots=True)
class DollarVolumeFeature(BaseFeature):
    """Dollar volume from close price and base volume.

    Computes ``close * volume`` and appends the result as ``dollar_volume``.
    Missing values are not filled.

    Attributes:
        name: Stable feature identifier (``dollar_volume``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``price``).
        description: Human-readable feature summary.
        required_columns: Input columns required by ``transform``.
        produced_columns: Output columns produced by ``transform``.
        lookback: Warm-up rows required (none for this point feature).
        dependencies: Upstream feature names (none for this feature).
    """

    name: str = _FEATURE_NAME
    version: str = _FEATURE_VERSION
    category: str = _FEATURE_CATEGORY
    description: str = _FEATURE_DESCRIPTION
    required_columns: tuple[str, ...] = (_CLOSE_COLUMN, _VOLUME_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 0
    dependencies: tuple[str, ...] = ()

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append dollar volume without mutating ``frame``.

        Args:
            frame: Input DataFrame containing ``close`` and ``volume``.

        Returns:
            A new DataFrame with all original columns plus ``dollar_volume``.

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

        dollar_volume_expr = pl.col(_CLOSE_COLUMN) * pl.col(
            _VOLUME_COLUMN
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            dollar_volume_expr.alias(_OUTPUT_COLUMN)
        )

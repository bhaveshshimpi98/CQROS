"""CQROS long/short ratio momentum feature.

Purpose:
    Compute multi-period long/short ratio momentum from raw long/short
    repository data.

Responsibilities:
    - Expose immutable ``RatioMomentumFeature`` metadata
    - Append a ``ratio_momentum`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``long_short_ratio``

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``RatioMomentumFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["RatioMomentumFeature"]

_RATIO_COLUMN: Final[str] = "long_short_ratio"
_OUTPUT_COLUMN: Final[str] = "ratio_momentum"
_FEATURE_NAME: Final[str] = "ratio_momentum"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "long_short"
_FEATURE_DESCRIPTION: Final[str] = (
    "Long/short ratio momentum as the absolute change over a lookback window."
)
_ERROR_LOOKBACK: Final[str] = "FEATURE-RATIO-MOMENTUM-001"
_ERROR_MISSING_RATIO: Final[str] = "FEATURE-RATIO-MOMENTUM-002"


@dataclass(frozen=True, slots=True)
class RatioMomentumFeature(BaseFeature):
    """Multi-period absolute long/short ratio momentum.

    Computes ``long_short_ratio - long_short_ratio.shift(lookback)`` and
    appends the result as ``ratio_momentum``. The first ``lookback`` rows are
    null. Missing values are never filled.

    Attributes:
        name: Stable feature identifier (``ratio_momentum``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``long_short``).
        description: Human-readable feature summary.
        required_columns: Input columns required by ``transform``.
        produced_columns: Output columns produced by ``transform``.
        lookback: Momentum horizon in rows (must be > 0).
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

    @property
    def warmup_rows(self) -> int:
        """``shift(lookback)`` leaves the first ``lookback`` rows undefined."""
        return self.lookback

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
        """Append ratio momentum without mutating ``frame``.

        Args:
            frame: Input DataFrame containing a ``long_short_ratio`` column.

        Returns:
            A new DataFrame with all original columns plus ``ratio_momentum``.

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
        momentum_expr = ratio - ratio.shift(
            self.lookback
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            momentum_expr.alias(_OUTPUT_COLUMN)
        )

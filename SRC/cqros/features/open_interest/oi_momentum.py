"""CQROS open interest momentum feature.

Purpose:
    Compute multi-period open interest momentum from raw open-interest
    repository data.

Responsibilities:
    - Expose immutable ``OIMomentumFeature`` metadata
    - Append an ``oi_momentum`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``open_interest``

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.features.base.BaseFeature``, and
    ``cqros.features.exceptions.FeatureExecutionError``.

Public API:
    ``OIMomentumFeature``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.features.base import BaseFeature
from cqros.features.exceptions import FeatureExecutionError

__all__ = ["OIMomentumFeature"]

_OI_COLUMN: Final[str] = "open_interest"
_OUTPUT_COLUMN: Final[str] = "oi_momentum"
_FEATURE_NAME: Final[str] = "oi_momentum"
_FEATURE_VERSION: Final[str] = "1.0.0"
_FEATURE_CATEGORY: Final[str] = "open_interest"
_FEATURE_DESCRIPTION: Final[str] = (
    "Open interest momentum as percent change over a lookback window."
)
_ERROR_LOOKBACK: Final[str] = "FEATURE-OI-MOMENTUM-001"
_ERROR_MISSING_OI: Final[str] = "FEATURE-OI-MOMENTUM-002"


@dataclass(frozen=True, slots=True)
class OIMomentumFeature(BaseFeature):
    """Multi-period percent open interest momentum.

    Computes ``(open_interest / open_interest.shift(lookback)) - 1`` and
    appends the result as ``oi_momentum``. The first ``lookback`` rows are
    null. Missing values are never filled.

    Attributes:
        name: Stable feature identifier (``oi_momentum``).
        version: Feature formula version (``1.0.0``).
        category: Feature category (``open_interest``).
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
    required_columns: tuple[str, ...] = (_OI_COLUMN,)
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
        """Append OI momentum without mutating ``frame``.

        Args:
            frame: Input DataFrame containing an ``open_interest`` column.

        Returns:
            A new DataFrame with all original columns plus ``oi_momentum``.

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
        momentum_expr = (
            oi / oi.shift(self.lookback)
        ) - 1  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            momentum_expr.alias(_OUTPUT_COLUMN)
        )

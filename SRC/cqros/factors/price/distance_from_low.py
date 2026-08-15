"""CQROS distance-from-low research factor.

Purpose:
    Compute normalized distance of close from the rolling low as a pure
    price alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``DistanceFromLowFactor`` metadata
    - Append a ``distance_from_low`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``DistanceFromLowFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["DistanceFromLowFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "distance_from_low"
_FACTOR_NAME: Final[str] = "distance_from_low"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = "Normalized distance of close from the rolling lookback low."
_ERROR_LOOKBACK: Final[str] = "FACTOR-DISTANCE-FROM-LOW-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-DISTANCE-FROM-LOW-002"


@dataclass(frozen=True, slots=True)
class DistanceFromLowFactor(BaseFactor):
    """Distance-from-low alpha factor from the rolling close low.

    Computes ``(close - rolling_low) / rolling_low`` where
    ``rolling_low = close.rolling_min(lookback)`` and appends the result as
    ``distance_from_low``. Values are greater than or equal to zero. The
    first ``lookback - 1`` rows are null. Missing values are never filled.
    The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``distance_from_low``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling low window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 20

    def __post_init__(self) -> None:
        """Validate base metadata and require a strictly positive lookback.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback <= 0``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback < 1:
            raise ValidationError(
                "lookback must be an integer greater than 0",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append distance from low without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``distance_from_low``. The first ``lookback - 1`` rows of
            ``distance_from_low`` are null.

        Raises:
            FactorError: If ``close`` is not present in ``frame``.
        """
        if _CLOSE_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_CLOSE_COLUMN}",
                error_code=_ERROR_MISSING_CLOSE,
                details={
                    "factor": self.name,
                    "required_column": _CLOSE_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        close = pl.col(_CLOSE_COLUMN)
        rolling_low = close.rolling_min(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        distance_expr = (
            close - rolling_low
        ) / rolling_low  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            distance_expr.alias(_OUTPUT_COLUMN)
        )

"""CQROS EMA distance research factor.

Purpose:
    Compute normalized distance of close from its exponential moving average
    as a pure price alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``EMADistanceFactor`` metadata
    - Append a ``ema_distance`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``EMADistanceFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["EMADistanceFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "ema_distance"
_FACTOR_NAME: Final[str] = "ema_distance"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Normalized distance of close from its exponential moving average."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-EMA-DISTANCE-001"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-EMA-DISTANCE-002"


@dataclass(frozen=True, slots=True)
class EMADistanceFactor(BaseFactor):
    """EMA distance alpha factor from close versus exponential moving average.

    Computes ``(close - EMA) / EMA`` where EMA uses span ``lookback`` with
    ``adjust=False`` and ``min_samples=lookback``, and appends the result as
    ``ema_distance``. Returns null when EMA is zero. The first
    ``lookback - 1`` rows are null. Missing values are never filled. The
    input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``ema_distance``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: EMA span / warm-up window (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 20

    def __post_init__(self) -> None:
        """Validate base metadata and require lookback >= 2.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback < 2``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback < 2:
            raise ValidationError(
                "lookback must be an integer greater than or equal to 2",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append EMA distance without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus ``ema_distance``.
            The first ``lookback - 1`` rows of ``ema_distance`` are null.

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
        ema = close.ewm_mean(  # pyright: ignore[reportUnknownMemberType]
            span=self.lookback,
            adjust=False,
            min_samples=self.lookback,
        )
        distance_expr = (
            pl.when(ema != 0).then((close - ema) / ema).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            distance_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

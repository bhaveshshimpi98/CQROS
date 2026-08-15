"""CQROS VWAP distance research factor.

Purpose:
    Compute normalized distance of close from rolling-mean VWAP as a pure
    microstructure alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``VWAPDistanceFactor`` metadata
    - Append a ``vwap_distance`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``VWAPDistanceFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["VWAPDistanceFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_VWAP_COLUMN: Final[str] = "vwap"
_OUTPUT_COLUMN: Final[str] = "vwap_distance"
_FACTOR_NAME: Final[str] = "vwap_distance"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "microstructure"
_FACTOR_DESCRIPTION: Final[str] = (
    "Normalized distance of close from the rolling mean of provided VWAP."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-VWAP-DISTANCE-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-VWAP-DISTANCE-002"


@dataclass(frozen=True, slots=True)
class VWAPDistanceFactor(BaseFactor):
    """VWAP distance alpha factor from close versus rolling-mean VWAP.

    Computes ``(close - mean(vwap)) / mean(vwap)`` over ``lookback`` and
    appends the result as ``vwap_distance``. Returns null when the rolling
    VWAP mean is zero. Incomplete windows are null. Missing values are never
    filled. The input DataFrame is never mutated.

    This factor consumes the provided ``vwap`` column and does not recompute
    VWAP from OHLC.

    Attributes:
        name: Stable factor identifier (``vwap_distance``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``microstructure``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling VWAP mean window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN, _VWAP_COLUMN)
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
        """Append VWAP distance without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``close`` and ``vwap``.

        Returns:
            A new DataFrame with all original columns plus ``vwap_distance``.
            Incomplete windows are null.

        Raises:
            FactorError: If a required column is not present in ``frame``.
        """
        for column in self.required_features:
            if column not in frame.columns:
                raise FactorError(
                    f"required column missing: {column}",
                    error_code=_ERROR_MISSING_COLUMN,
                    details={
                        "factor": self.name,
                        "required_column": column,
                        "available_columns": tuple(frame.columns),
                    },
                )

        close = pl.col(_CLOSE_COLUMN)
        vwap_mean = pl.col(_VWAP_COLUMN).rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        distance_expr = (
            pl.when(vwap_mean != 0).then((close - vwap_mean) / vwap_mean).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            distance_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

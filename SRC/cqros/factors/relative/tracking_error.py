"""CQROS tracking error research factor.

Purpose:
    Compute rolling tracking error of an asset versus a benchmark return as
    a pure relative-value alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``TrackingErrorFactor`` metadata
    - Append a ``tracking_error`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``TrackingErrorFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["TrackingErrorFactor"]

_ASSET_COLUMN: Final[str] = "asset_return"
_BENCHMARK_COLUMN: Final[str] = "benchmark_return"
_OUTPUT_COLUMN: Final[str] = "tracking_error"
_FACTOR_NAME: Final[str] = "tracking_error"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "relative"
_FACTOR_DESCRIPTION: Final[str] = (
    "Tracking error as the rolling population standard deviation of "
    "asset-minus-benchmark returns."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-TRACKING-ERROR-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-TRACKING-ERROR-002"


@dataclass(frozen=True, slots=True)
class TrackingErrorFactor(BaseFactor):
    """Tracking error alpha factor from active-return volatility.

    Computes ``std(asset_return - benchmark_return, lookback)`` with
    population standard deviation (``ddof=0``) and appends the result as
    ``tracking_error``. Incomplete windows are null. Missing values are
    never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``tracking_error``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``relative``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling standard-deviation window size (must be >= 2).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_ASSET_COLUMN, _BENCHMARK_COLUMN)
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
        """Append tracking error without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``asset_return`` and
                ``benchmark_return`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``tracking_error``. Incomplete windows are null.

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

        active_return = pl.col(_ASSET_COLUMN) - pl.col(_BENCHMARK_COLUMN)
        tracking_expr = active_return.rolling_std(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback,
            ddof=0,
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            tracking_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

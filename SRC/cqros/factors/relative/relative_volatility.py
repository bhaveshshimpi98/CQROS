"""CQROS relative volatility research factor.

Purpose:
    Compute rolling relative volatility of an asset versus a benchmark
    return as a pure relative-value alpha factor for the Factor Research
    Engine.

Responsibilities:
    - Expose immutable ``RelativeVolatilityFactor`` metadata
    - Append a ``relative_volatility`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``RelativeVolatilityFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["RelativeVolatilityFactor"]

_ASSET_COLUMN: Final[str] = "asset_return"
_BENCHMARK_COLUMN: Final[str] = "benchmark_return"
_OUTPUT_COLUMN: Final[str] = "relative_volatility"
_FACTOR_NAME: Final[str] = "relative_volatility"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "relative"
_FACTOR_DESCRIPTION: Final[str] = (
    "Relative volatility as asset rolling return standard deviation divided "
    "by benchmark rolling return standard deviation."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-RELATIVE-VOLATILITY-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-RELATIVE-VOLATILITY-002"


@dataclass(frozen=True, slots=True)
class RelativeVolatilityFactor(BaseFactor):
    """Relative volatility alpha factor from rolling return std ratio.

    Computes ``std(asset_return) / std(benchmark_return)`` over ``lookback``
    with population standard deviation (``ddof=0``) and appends the result
    as ``relative_volatility``. Returns null when benchmark standard
    deviation is zero. Incomplete windows are null. Missing values are
    never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``relative_volatility``).
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
        """Append relative volatility without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``asset_return`` and
                ``benchmark_return`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``relative_volatility``. Incomplete windows are null.

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

        window = self.lookback
        asset_std = pl.col(_ASSET_COLUMN).rolling_std(  # pyright: ignore[reportUnknownMemberType]
            window_size=window,
            ddof=0,
        )
        bench_std = pl.col(
            _BENCHMARK_COLUMN
        ).rolling_std(  # pyright: ignore[reportUnknownMemberType]
            window_size=window,
            ddof=0,
        )
        relative_expr = (
            pl.when(bench_std != 0).then(asset_std / bench_std).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            relative_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

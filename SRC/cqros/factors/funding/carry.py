"""CQROS carry research factor.

Purpose:
    Compute rolling-mean funding rate times point-in-time basis as a pure
    funding alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``CarryFactor`` metadata
    - Append a ``carry`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``CarryFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["CarryFactor"]

_RATE_COLUMN: Final[str] = "funding_rate"
_MARK_COLUMN: Final[str] = "mark_price"
_INDEX_COLUMN: Final[str] = "index_price"
_OUTPUT_COLUMN: Final[str] = "carry"
_FACTOR_NAME: Final[str] = "carry"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "funding"
_FACTOR_DESCRIPTION: Final[str] = (
    "Carry as rolling mean funding rate multiplied by mark-versus-index basis."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-CARRY-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-CARRY-002"


@dataclass(frozen=True, slots=True)
class CarryFactor(BaseFactor):
    """Carry alpha factor from rolling-mean funding and point-in-time basis.

    Computes basis ``(mark_price - index_price) / index_price``, then
    ``rolling_mean(funding_rate, lookback) * basis``, and appends the result
    as ``carry``. Returns null when ``index_price`` is zero. Incomplete
    windows are null. Missing values are never filled. The input DataFrame
    is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``carry``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``funding``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Rolling funding-rate mean window size (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_RATE_COLUMN, _MARK_COLUMN, _INDEX_COLUMN)
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
        """Append carry without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``funding_rate``,
                ``mark_price``, and ``index_price`` columns.

        Returns:
            A new DataFrame with all original columns plus ``carry``.
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

        mark = pl.col(_MARK_COLUMN)
        index = pl.col(_INDEX_COLUMN)
        basis = pl.when(index == 0).then(None).otherwise((mark - index) / index)
        funding_mean = pl.col(
            _RATE_COLUMN
        ).rolling_mean(  # pyright: ignore[reportUnknownMemberType]
            window_size=self.lookback
        )
        carry_expr = funding_mean * basis
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            carry_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

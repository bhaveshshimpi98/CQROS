"""CQROS basis momentum research factor.

Purpose:
    Compute multi-period absolute basis momentum as a pure funding alpha
    factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``BasisMomentumFactor`` metadata
    - Append a ``basis_momentum`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``BasisMomentumFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["BasisMomentumFactor"]

_MARK_COLUMN: Final[str] = "mark_price"
_INDEX_COLUMN: Final[str] = "index_price"
_OUTPUT_COLUMN: Final[str] = "basis_momentum"
_FACTOR_NAME: Final[str] = "basis_momentum"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "funding"
_FACTOR_DESCRIPTION: Final[str] = (
    "Basis momentum as the absolute change in mark-versus-index basis over " "a lookback window."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-BASIS-MOMENTUM-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-BASIS-MOMENTUM-002"


@dataclass(frozen=True, slots=True)
class BasisMomentumFactor(BaseFactor):
    """Basis momentum alpha factor from absolute basis change.

    Computes basis ``(mark_price - index_price) / index_price``, then
    ``basis - basis.shift(lookback)``, and appends the result as
    ``basis_momentum``. Returns null when ``index_price`` is zero. The first
    ``lookback`` rows are null. Missing values are never filled. The input
    DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``basis_momentum``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``funding``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Momentum horizon in rows (must be > 0).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_MARK_COLUMN, _INDEX_COLUMN)
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
        """Append basis momentum without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``mark_price`` and
                ``index_price`` columns.

        Returns:
            A new DataFrame with all original columns plus
            ``basis_momentum``. The first ``lookback`` rows are null.

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
        momentum_expr = basis - basis.shift(  # pyright: ignore[reportUnknownMemberType]
            self.lookback
        )
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            momentum_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

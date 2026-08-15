"""CQROS basis research factor.

Purpose:
    Compute mark-versus-index basis as a pure funding alpha factor for the
    Factor Research Engine.

Responsibilities:
    - Expose immutable ``BasisFactor`` metadata
    - Append a ``basis`` column using Polars expressions only
    - Fail fast on invalid lookback and missing required columns
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``BasisFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["BasisFactor"]

_MARK_COLUMN: Final[str] = "mark_price"
_INDEX_COLUMN: Final[str] = "index_price"
_OUTPUT_COLUMN: Final[str] = "basis"
_FACTOR_NAME: Final[str] = "basis"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "funding"
_FACTOR_DESCRIPTION: Final[str] = "Perpetual basis as (mark_price - index_price) / index_price."
_ERROR_LOOKBACK: Final[str] = "FACTOR-BASIS-001"
_ERROR_MISSING_COLUMN: Final[str] = "FACTOR-BASIS-002"


@dataclass(frozen=True, slots=True)
class BasisFactor(BaseFactor):
    """Basis alpha factor from mark price versus index price.

    Computes ``(mark_price - index_price) / index_price`` and appends the
    result as ``basis``. Returns null when ``index_price`` is zero. Missing
    values are never filled. The input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``basis``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``funding``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Must be ``0``; basis is point-in-time.
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_MARK_COLUMN, _INDEX_COLUMN)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = 0

    def __post_init__(self) -> None:
        """Validate base metadata and require lookback equal to 0.

        Raises:
            ValidationError: If any metadata invariant is violated, including
                ``lookback != 0``.
        """
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)
        if self.lookback != 0:
            raise ValidationError(
                "lookback must be 0 for basis",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append basis without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing ``mark_price`` and
                ``index_price`` columns.

        Returns:
            A new DataFrame with all original columns plus ``basis``.

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
        basis_expr = (
            pl.when(index == 0).then(None).otherwise((mark - index) / index)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            basis_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

"""CQROS premium index research factor.

Purpose:
    Expose the current premium index as a pure funding alpha factor for the
    Factor Research Engine.

Responsibilities:
    - Expose immutable ``PremiumIndexFactor`` metadata
    - Append a ``premium_index_factor`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``premium_index``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``PremiumIndexFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["PremiumIndexFactor"]

_PREMIUM_COLUMN: Final[str] = "premium_index"
_OUTPUT_COLUMN: Final[str] = "premium_index_factor"
_FACTOR_NAME: Final[str] = "premium_index_factor"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "funding"
_FACTOR_DESCRIPTION: Final[str] = (
    "Current premium index as a point-in-time funding and derivatives signal."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-PREMIUM-INDEX-001"
_ERROR_MISSING_PREMIUM: Final[str] = "FACTOR-PREMIUM-INDEX-002"


@dataclass(frozen=True, slots=True)
class PremiumIndexFactor(BaseFactor):
    """Premium index alpha factor from the current premium index.

    Appends the ``premium_index`` column as ``premium_index_factor``. Missing
    values remain null and are never filled. The input DataFrame is never
    mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``premium_index_factor``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``funding``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Must be ``0``; premium index is point-in-time.
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_PREMIUM_COLUMN,)
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
                "lookback must be 0 for premium_index_factor",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append premium index factor without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``premium_index``
                column.

        Returns:
            A new DataFrame with all original columns plus
            ``premium_index_factor``.

        Raises:
            FactorError: If ``premium_index`` is not present in ``frame``.
        """
        if _PREMIUM_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_PREMIUM_COLUMN}",
                error_code=_ERROR_MISSING_PREMIUM,
                details={
                    "factor": self.name,
                    "required_column": _PREMIUM_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            pl.col(_PREMIUM_COLUMN).cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

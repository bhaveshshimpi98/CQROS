"""CQROS funding rate level research factor.

Purpose:
    Expose the current funding rate as a pure funding alpha factor for the
    Factor Research Engine.

Responsibilities:
    - Expose immutable ``FundingRateLevelFactor`` metadata
    - Append a ``funding_rate_level`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``funding_rate``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``FundingRateLevelFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["FundingRateLevelFactor"]

_RATE_COLUMN: Final[str] = "funding_rate"
_OUTPUT_COLUMN: Final[str] = "funding_rate_level"
_FACTOR_NAME: Final[str] = "funding_rate_level"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "funding"
_FACTOR_DESCRIPTION: Final[str] = "Current funding rate level as a point-in-time funding signal."
_ERROR_LOOKBACK: Final[str] = "FACTOR-FUNDING-RATE-LEVEL-001"
_ERROR_MISSING_RATE: Final[str] = "FACTOR-FUNDING-RATE-LEVEL-002"


@dataclass(frozen=True, slots=True)
class FundingRateLevelFactor(BaseFactor):
    """Funding rate level alpha factor from the current funding rate.

    Appends the ``funding_rate`` column as ``funding_rate_level``. Missing
    values remain null and are never filled. The input DataFrame is never
    mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``funding_rate_level``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``funding``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Must be ``0``; level is point-in-time.
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_RATE_COLUMN,)
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
                "lookback must be 0 for funding_rate_level",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append funding rate level without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``funding_rate``
                column.

        Returns:
            A new DataFrame with all original columns plus
            ``funding_rate_level``.

        Raises:
            FactorError: If ``funding_rate`` is not present in ``frame``.
        """
        if _RATE_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_RATE_COLUMN}",
                error_code=_ERROR_MISSING_RATE,
                details={
                    "factor": self.name,
                    "required_column": _RATE_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            pl.col(_RATE_COLUMN).cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

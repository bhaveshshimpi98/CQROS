"""CQROS open interest level research factor.

Purpose:
    Expose the current open interest as a pure open-interest alpha factor for
    the Factor Research Engine.

Responsibilities:
    - Expose immutable ``OpenInterestLevelFactor`` metadata
    - Append an ``open_interest_level`` column using Polars expressions only
    - Fail fast on invalid lookback and missing ``open_interest``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``OpenInterestLevelFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["OpenInterestLevelFactor"]

_OI_COLUMN: Final[str] = "open_interest"
_OUTPUT_COLUMN: Final[str] = "open_interest_level"
_FACTOR_NAME: Final[str] = "open_interest_level"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "open_interest"
_FACTOR_DESCRIPTION: Final[str] = (
    "Current open interest level as a point-in-time positioning signal."
)
_ERROR_LOOKBACK: Final[str] = "FACTOR-OPEN-INTEREST-LEVEL-001"
_ERROR_MISSING_OI: Final[str] = "FACTOR-OPEN-INTEREST-LEVEL-002"


@dataclass(frozen=True, slots=True)
class OpenInterestLevelFactor(BaseFactor):
    """Open interest level alpha factor from the current open interest.

    Appends the ``open_interest`` column as ``open_interest_level``. Missing
    values remain null and are never filled. The input DataFrame is never
    mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``open_interest_level``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``open_interest``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Must be ``0``; level is point-in-time.
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_OI_COLUMN,)
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
                "lookback must be 0 for open_interest_level",
                error_code=_ERROR_LOOKBACK,
                details={"parameter": "lookback", "value": self.lookback},
            )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append open interest level without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing an ``open_interest``
                column.

        Returns:
            A new DataFrame with all original columns plus
            ``open_interest_level``.

        Raises:
            FactorError: If ``open_interest`` is not present in ``frame``.
        """
        if _OI_COLUMN not in frame.columns:
            raise FactorError(
                f"required column missing: {_OI_COLUMN}",
                error_code=_ERROR_MISSING_OI,
                details={
                    "factor": self.name,
                    "required_column": _OI_COLUMN,
                    "available_columns": tuple(frame.columns),
                },
            )

        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            pl.col(_OI_COLUMN).cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )

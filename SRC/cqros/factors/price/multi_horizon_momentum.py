"""CQROS multi-horizon momentum research factor.

Purpose:
    Compute an equal-weighted average of pure price momentum across multiple
    horizons as an alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``MultiHorizonMomentumFactor`` metadata
    - Append a ``multi_horizon_momentum`` column using Polars expressions only
    - Fail fast on invalid horizons and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``MultiHorizonMomentumFactor``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["MultiHorizonMomentumFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "multi_horizon_momentum"
_FACTOR_NAME: Final[str] = "multi_horizon_momentum"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Equal-weighted average of pure price momentum across multiple horizons."
)
_DEFAULT_HORIZONS: Final[tuple[int, ...]] = (5, 10, 20, 50)
_ERROR_HORIZONS_EMPTY: Final[str] = "FACTOR-MULTI-HORIZON-MOMENTUM-001"
_ERROR_HORIZON_INVALID: Final[str] = "FACTOR-MULTI-HORIZON-MOMENTUM-002"
_ERROR_HORIZONS_TYPE: Final[str] = "FACTOR-MULTI-HORIZON-MOMENTUM-003"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-MULTI-HORIZON-MOMENTUM-004"


@dataclass(frozen=True, slots=True)
class MultiHorizonMomentumFactor(BaseFactor):
    """Multi-horizon momentum alpha factor from the close price.

    Computes the equal-weighted mean of
    ``(close / close.shift(horizon)) - 1`` for each horizon in ``horizons``
    and appends the result as ``multi_horizon_momentum``. The first
    ``max(horizons)`` rows are null. Missing values are never filled. The
    input DataFrame is never mutated.

    ``lookback`` is always set to ``max(horizons)``.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``multi_horizon_momentum``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Warm-up rows equal to ``max(horizons)``.
        horizons: Positive momentum horizons averaged by ``compute``.
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = max(_DEFAULT_HORIZONS)
    horizons: tuple[int, ...] = _DEFAULT_HORIZONS

    def __post_init__(self) -> None:
        """Validate horizons, derive lookback, and validate base metadata.

        Raises:
            ValidationError: If any metadata or horizons invariant is violated.
        """
        frozen_horizons = _freeze_positive_horizons(self.horizons)
        object.__setattr__(self, "horizons", frozen_horizons)
        object.__setattr__(self, "lookback", max(frozen_horizons))
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append multi-horizon momentum without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``multi_horizon_momentum``. The first ``max(horizons)`` rows of
            ``multi_horizon_momentum`` are null.

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
        momentum_exprs = [(close / close.shift(horizon)) - 1 for horizon in self.horizons]
        # Sum/divide propagates nulls so warm-up waits for the longest horizon.
        total = momentum_exprs[0]
        for momentum_expr in momentum_exprs[1:]:
            total = total + momentum_expr
        average_expr = total / len(momentum_exprs)  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            average_expr.alias(_OUTPUT_COLUMN)
        )


def _freeze_positive_horizons(value: object) -> tuple[int, ...]:
    """Validate and freeze a non-empty sequence of strictly positive horizons.

    Args:
        value: Candidate horizon sequence.

    Returns:
        An immutable tuple of validated positive integers.

    Raises:
        ValidationError: If ``value`` is empty, mistyped, or contains
            non-positive integers.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(
            "horizons must be a sequence of positive integers",
            error_code=_ERROR_HORIZONS_TYPE,
            details={"parameter": "horizons", "value_type": type(value).__name__},
        )

    sequence = cast(Sequence[object], value)
    if len(sequence) == 0:
        raise ValidationError(
            "horizons must contain at least one entry",
            error_code=_ERROR_HORIZONS_EMPTY,
            details={"parameter": "horizons"},
        )

    frozen: list[int] = []
    for index, entry in enumerate(sequence):
        if not isinstance(entry, int) or isinstance(entry, bool) or entry < 1:
            raise ValidationError(
                "horizons entries must be integers greater than 0",
                error_code=_ERROR_HORIZON_INVALID,
                details={"parameter": "horizons", "index": index, "value": entry},
            )
        frozen.append(entry)
    return tuple(frozen)

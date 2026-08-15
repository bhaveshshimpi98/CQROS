"""CQROS Percentage Price Oscillator research factor.

Purpose:
    Compute the Percentage Price Oscillator (PPO) from close prices as a pure
    price alpha factor for the Factor Research Engine.

Responsibilities:
    - Expose immutable ``PriceOscillatorFactor`` metadata
    - Append a ``price_oscillator`` column using Polars expressions only
    - Fail fast on invalid EMA spans and missing ``close``
    - Remain free of indicator logic, registry, pipeline, and research stats

Dependencies:
    ``polars``, ``cqros.core.exceptions.ValidationError``,
    ``cqros.factors.base.BaseFactor``, and
    ``cqros.factors.exceptions.FactorError``.

Public API:
    ``PriceOscillatorFactor``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.exceptions import ValidationError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError

__all__ = ["PriceOscillatorFactor"]

_CLOSE_COLUMN: Final[str] = "close"
_OUTPUT_COLUMN: Final[str] = "price_oscillator"
_FACTOR_NAME: Final[str] = "price_oscillator"
_FACTOR_VERSION: Final[str] = "1.0.0"
_FACTOR_CATEGORY: Final[str] = "price"
_FACTOR_DESCRIPTION: Final[str] = (
    "Percentage Price Oscillator from fast and slow exponential moving averages."
)
_DEFAULT_FAST_SPAN: Final[int] = 12
_DEFAULT_SLOW_SPAN: Final[int] = 26
_ERROR_FAST_SPAN: Final[str] = "FACTOR-PRICE-OSCILLATOR-001"
_ERROR_SLOW_SPAN: Final[str] = "FACTOR-PRICE-OSCILLATOR-002"
_ERROR_SPAN_ORDER: Final[str] = "FACTOR-PRICE-OSCILLATOR-003"
_ERROR_MISSING_CLOSE: Final[str] = "FACTOR-PRICE-OSCILLATOR-004"


@dataclass(frozen=True, slots=True)
class PriceOscillatorFactor(BaseFactor):
    """Percentage Price Oscillator alpha factor from dual EMAs.

    Computes
    ``100 * (EMA_fast - EMA_slow) / EMA_slow`` where each EMA uses
    ``adjust=False`` and ``min_samples`` equal to its span. Defaults follow
    the classic PPO configuration ``fast=12``, ``slow=26`` (signal period 9
    is not part of the oscillator output). Returns null when the slow EMA is
    zero. Incomplete windows are null. Missing values are never filled. The
    input DataFrame is never mutated.

    This is a research alpha factor, not a technical indicator.

    Attributes:
        name: Stable factor identifier (``price_oscillator``).
        version: Factor formula version (``1.0.0``).
        description: Human-readable factor summary.
        category: Factor category (``price``).
        required_features: Input features required by ``compute``.
        produced_columns: Output columns produced by ``compute``.
        lookback: Warm-up rows equal to ``slow_span``.
        fast_span: Fast EMA span (must be >= 2).
        slow_span: Slow EMA span (must be >= 2 and greater than ``fast_span``).
    """

    name: str = _FACTOR_NAME
    version: str = _FACTOR_VERSION
    description: str = _FACTOR_DESCRIPTION
    category: str = _FACTOR_CATEGORY
    required_features: tuple[str, ...] = (_CLOSE_COLUMN,)
    produced_columns: tuple[str, ...] = (_OUTPUT_COLUMN,)
    lookback: int = _DEFAULT_SLOW_SPAN
    fast_span: int = _DEFAULT_FAST_SPAN
    slow_span: int = _DEFAULT_SLOW_SPAN

    def __post_init__(self) -> None:
        """Validate EMA spans, derive lookback, and validate base metadata.

        Raises:
            ValidationError: If any metadata or span invariant is violated.
        """
        _require_span_at_least(
            self.fast_span,
            minimum=2,
            parameter="fast_span",
            error_code=_ERROR_FAST_SPAN,
        )
        _require_span_at_least(
            self.slow_span,
            minimum=2,
            parameter="slow_span",
            error_code=_ERROR_SLOW_SPAN,
        )
        if self.slow_span <= self.fast_span:
            raise ValidationError(
                "slow_span must be greater than fast_span",
                error_code=_ERROR_SPAN_ORDER,
                details={
                    "parameter": "slow_span",
                    "fast_span": self.fast_span,
                    "slow_span": self.slow_span,
                },
            )
        object.__setattr__(self, "lookback", self.slow_span)
        # Explicit base call: super() breaks with frozen slotted ABC dataclasses.
        BaseFactor.__post_init__(self)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Append Percentage Price Oscillator without mutating ``frame``.

        Args:
            frame: Input research DataFrame containing a ``close`` column.

        Returns:
            A new DataFrame with all original columns plus
            ``price_oscillator``. Incomplete windows are null.

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
        ema_fast = close.ewm_mean(  # pyright: ignore[reportUnknownMemberType]
            span=self.fast_span,
            adjust=False,
            min_samples=self.fast_span,
        )
        ema_slow = close.ewm_mean(  # pyright: ignore[reportUnknownMemberType]
            span=self.slow_span,
            adjust=False,
            min_samples=self.slow_span,
        )
        ppo_expr = (
            pl.when(ema_slow != 0).then(100.0 * (ema_fast - ema_slow) / ema_slow).otherwise(None)
        )  # pyright: ignore[reportUnknownMemberType]
        return frame.with_columns(  # pyright: ignore[reportUnknownMemberType]
            ppo_expr.cast(pl.Float64).alias(_OUTPUT_COLUMN)
        )


def _require_span_at_least(
    value: object,
    *,
    minimum: int,
    parameter: str,
    error_code: str,
) -> None:
    """Raise ``ValidationError`` when ``value`` is not an int >= ``minimum``."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValidationError(
            f"{parameter} must be an integer greater than or equal to {minimum}",
            error_code=error_code,
            details={"parameter": parameter, "value": value},
        )

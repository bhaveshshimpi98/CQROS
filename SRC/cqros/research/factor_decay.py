"""CQROS factor decay analysis.

Purpose:
    Measure how a factor's predictive power changes as the forward-return
    horizon increases by computing IC at each requested horizon.

Responsibilities:
    - Define immutable ``DecayPoint`` and ``FactorDecayResult`` value objects
    - Generate forward returns via ``ForwardReturnTarget``
    - Estimate IC via ``InformationCoefficient``
    - Derive an IC half-life from the decay curve
    - Remain free of trading, portfolio simulation, execution, and ML logic

Dependencies:
    ``polars``, ``cqros.core.exceptions.ResearchError``,
    ``cqros.research.information_coefficient``, and ``cqros.research.target``.

Public API:
    ``DecayPoint``, ``FactorDecayResult``, ``FactorDecayAnalyzer``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ResearchError
from cqros.research.information_coefficient import InformationCoefficient
from cqros.research.target import ForwardReturnTarget, TargetDefinition

__all__ = [
    "DecayPoint",
    "FactorDecayResult",
    "FactorDecayAnalyzer",
]

_DEFAULT_METHOD: Final[str] = "spearman"
_DEFAULT_PRICE_COLUMN: Final[str] = "close"
_DEFAULT_HORIZONS: Final[tuple[int, ...]] = (1, 2, 4, 8, 12, 24)
_HALF_LIFE_RATIO: Final[float] = 0.5
_TARGET_OUTPUT_COLUMN: Final[str] = "__cqros_decay_forward_return"

_ERROR_HORIZONS_EMPTY: Final[str] = "RESEARCH-DECAY-001"
_ERROR_HORIZON_INVALID: Final[str] = "RESEARCH-DECAY-002"
_ERROR_MISSING_FACTOR: Final[str] = "RESEARCH-DECAY-003"
_ERROR_MISSING_PRICE: Final[str] = "RESEARCH-DECAY-004"


@dataclass(frozen=True, slots=True)
class DecayPoint:
    """Immutable IC estimate at a single forward-return horizon.

    Attributes:
        horizon: Forward-return horizon in rows.
        observations: Number of paired non-null observations used.
        coefficient: Information coefficient at this horizon.
        p_value: Two-sided p-value for the coefficient.
    """

    horizon: int
    observations: int
    coefficient: float
    p_value: float


@dataclass(frozen=True, slots=True)
class FactorDecayResult:
    """Immutable factor decay analysis result.

    Attributes:
        factor_column: Factor column analyzed.
        price_column: Price column used to form forward returns.
        method: Correlation method used for IC estimation.
        points: IC estimates ordered by the requested horizons.
        half_life: First horizon whose absolute IC falls below half of the
            first horizon's absolute IC, or ``None`` when never reached.
    """

    factor_column: str
    price_column: str
    method: str
    points: tuple[DecayPoint, ...]
    half_life: int | None


class FactorDecayAnalyzer:
    """Statistical factor decay analyzer across forward-return horizons.

    For each horizon the analyzer builds a forward-return target, estimates
    the configured Information Coefficient against the factor, and derives
    an IC half-life from the resulting decay curve. The input DataFrame is
    never mutated.
    """

    __slots__ = ("_calculator",)

    def __init__(self, method: str = _DEFAULT_METHOD) -> None:
        """Initialize the decay analyzer.

        Args:
            method: Correlation method forwarded to ``InformationCoefficient``.
                Defaults to ``spearman``.

        Raises:
            ResearchError: If ``method`` is not supported by
                ``InformationCoefficient``.
        """
        self._calculator = InformationCoefficient(method=method)

    @property
    def method(self) -> str:
        """Return the configured IC correlation method."""
        return self._calculator.method

    def analyze(
        self,
        frame: pl.DataFrame,
        factor_column: str,
        price_column: str = _DEFAULT_PRICE_COLUMN,
        horizons: Sequence[int] = _DEFAULT_HORIZONS,
    ) -> FactorDecayResult:
        """Analyze factor IC decay across forward-return horizons.

        Args:
            frame: Input research DataFrame. Must not be mutated.
            factor_column: Factor column to evaluate.
            price_column: Price column used for forward returns. Defaults to
                ``close``.
            horizons: Positive forward-return horizons in rows. Defaults to
                ``(1, 2, 4, 8, 12, 24)``.

        Returns:
            An immutable ``FactorDecayResult``.

        Raises:
            ResearchError: If ``horizons`` is empty or contains a non-positive
                value, or if ``factor_column`` / ``price_column`` is missing.
        """
        validated_horizons = _validate_horizons(horizons)
        if factor_column not in frame.columns:
            raise ResearchError(
                f"required column missing: {factor_column}",
                error_code=_ERROR_MISSING_FACTOR,
                details={
                    "required_column": factor_column,
                    "role": "factor",
                    "available_columns": tuple(frame.columns),
                },
            )
        if price_column not in frame.columns:
            raise ResearchError(
                f"required column missing: {price_column}",
                error_code=_ERROR_MISSING_PRICE,
                details={
                    "required_column": price_column,
                    "role": "price",
                    "available_columns": tuple(frame.columns),
                },
            )

        points = tuple(
            _decay_point_for_horizon(
                frame=frame,
                factor_column=factor_column,
                price_column=price_column,
                horizon=horizon,
                calculator=self._calculator,
            )
            for horizon in validated_horizons
        )
        return FactorDecayResult(
            factor_column=factor_column,
            price_column=price_column,
            method=self._calculator.method,
            points=points,
            half_life=_compute_half_life(points),
        )


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    """Validate and freeze the requested forward-return horizons."""
    sequence = tuple(horizons)
    if len(sequence) == 0:
        raise ResearchError(
            "horizons must contain at least one positive integer",
            error_code=_ERROR_HORIZONS_EMPTY,
            details={"horizons": sequence},
        )
    for index, horizon in enumerate(sequence):
        if not isinstance(cast(object, horizon), int) or isinstance(horizon, bool) or horizon < 1:
            raise ResearchError(
                "horizons entries must be integers greater than or equal to 1",
                error_code=_ERROR_HORIZON_INVALID,
                details={"index": index, "value": horizon},
            )
    return sequence


def _decay_point_for_horizon(
    *,
    frame: pl.DataFrame,
    factor_column: str,
    price_column: str,
    horizon: int,
    calculator: InformationCoefficient,
) -> DecayPoint:
    """Compute one decay point by reusing target generation and IC estimation."""
    target = ForwardReturnTarget(
        TargetDefinition(
            name=f"decay_forward_return_{horizon}",
            horizon=horizon,
            price_column=price_column,
            output_column=_TARGET_OUTPUT_COLUMN,
        )
    )
    with_target = target.transform(frame)
    ic_result = calculator.compute(with_target, factor_column, _TARGET_OUTPUT_COLUMN)
    return DecayPoint(
        horizon=horizon,
        observations=ic_result.observations,
        coefficient=ic_result.coefficient,
        p_value=ic_result.p_value,
    )


def _compute_half_life(points: Sequence[DecayPoint]) -> int | None:
    """Return the first horizon whose absolute IC falls below half the baseline."""
    if not points:
        return None
    threshold = abs(points[0].coefficient) * _HALF_LIFE_RATIO
    for point in points:
        if abs(point.coefficient) < threshold:
            return point.horizon
    return None

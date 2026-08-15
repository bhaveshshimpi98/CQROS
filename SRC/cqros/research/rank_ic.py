"""CQROS Rank Information Coefficient (Rank IC) evaluator.

Purpose:
    Provide a dedicated research abstraction for Spearman rank correlation
    between a factor column and a forward-return target column.

Responsibilities:
    - Define immutable ``RankICResult`` value objects
    - Compute Rank IC and two-sided p-values via ``RankInformationCoefficient``
    - Delegate statistical estimation to the Spearman path of
      ``InformationCoefficient`` so validation and SciPy usage stay shared
    - Remain free of trading, signals, backtests, execution, and ML logic

Dependencies:
    ``polars`` and ``cqros.research.information_coefficient``.

Public API:
    ``RankICResult``, ``RankInformationCoefficient``

Notes:
    ``InformationCoefficient`` already supports Spearman. ``RankIC`` exists as
    the explicit contract used by reports and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from cqros.research.information_coefficient import InformationCoefficient

__all__ = [
    "RankICResult",
    "RankInformationCoefficient",
]


@dataclass(frozen=True, slots=True)
class RankICResult:
    """Immutable result of a Rank Information Coefficient calculation.

    Attributes:
        factor_column: Factor column name used in the calculation.
        target_column: Target column name used in the calculation.
        observations: Number of paired non-null observations used.
        coefficient: Spearman rank correlation coefficient.
        p_value: Two-sided p-value for the coefficient.
    """

    factor_column: str
    target_column: str
    observations: int
    coefficient: float
    p_value: float


class RankInformationCoefficient:
    """Dedicated Spearman Rank Information Coefficient calculator.

    Drops null pairs, requires at least two observations, and returns the
    Spearman coefficient with a two-sided p-value. The input DataFrame is
    never mutated.
    """

    __slots__ = ("_calculator",)

    def __init__(self) -> None:
        """Initialize a Rank IC calculator backed by Spearman IC."""
        self._calculator = InformationCoefficient(method="spearman")

    def compute(
        self,
        frame: pl.DataFrame,
        factor_column: str,
        target_column: str,
    ) -> RankICResult:
        """Compute the Rank Information Coefficient for two columns.

        Rows with a null in either column are dropped before estimation.
        At least two paired observations are required.

        Args:
            frame: Input research DataFrame. Must not be mutated.
            factor_column: Name of the factor column.
            target_column: Name of the forward-return target column.

        Returns:
            An immutable ``RankICResult``.

        Raises:
            ResearchError: If either column is missing or fewer than two
                paired non-null observations remain.
        """
        result = self._calculator.compute(frame, factor_column, target_column)
        return RankICResult(
            factor_column=result.factor_column,
            target_column=result.target_column,
            observations=result.observations,
            coefficient=result.coefficient,
            p_value=result.p_value,
        )

"""CQROS cross-sectional quantile analysis for factor research.

Purpose:
    Evaluate whether a factor ranks assets from worst to best by comparing
    forward returns across equal-frequency factor quantiles.

Responsibilities:
    - Define immutable ``QuantileStatistics`` and ``QuantileAnalysisResult``
    - Assign observations into equal-frequency factor quantiles
    - Aggregate return statistics per quantile
    - Report top-minus-bottom spread and mean-return monotonicity
    - Remain free of portfolio simulation, execution, and backtesting

Dependencies:
    ``polars`` and ``cqros.core.exceptions.ResearchError``.

Public API:
    ``QuantileStatistics``, ``QuantileAnalysisResult``, ``QuantileAnalyzer``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import polars as pl

from cqros.core.exceptions import ResearchError

__all__ = [
    "QuantileStatistics",
    "QuantileAnalysisResult",
    "QuantileAnalyzer",
]

_DEFAULT_QUANTILES: Final[int] = 5
_QUANTILE_COLUMN: Final[str] = "__cqros_factor_quantile"

_ERROR_QUANTILES_INVALID: Final[str] = "RESEARCH-QA-001"
_ERROR_MISSING_FACTOR: Final[str] = "RESEARCH-QA-002"
_ERROR_MISSING_TARGET: Final[str] = "RESEARCH-QA-003"
_ERROR_INSUFFICIENT_OBS: Final[str] = "RESEARCH-QA-004"


@dataclass(frozen=True, slots=True)
class QuantileStatistics:
    """Immutable return statistics for a single factor quantile.

    Attributes:
        quantile: Quantile index from ``1`` (lowest factor values) to ``N``.
        count: Number of observations in the quantile.
        mean_return: Mean forward return in the quantile.
        median_return: Median forward return in the quantile.
        std_return: Sample standard deviation of forward returns.
        min_return: Minimum forward return in the quantile.
        max_return: Maximum forward return in the quantile.
    """

    quantile: int
    count: int
    mean_return: float
    median_return: float
    std_return: float
    min_return: float
    max_return: float


@dataclass(frozen=True, slots=True)
class QuantileAnalysisResult:
    """Immutable result of a cross-sectional quantile analysis.

    Attributes:
        factor_column: Factor column name used for quantile assignment.
        target_column: Forward-return target column analyzed.
        quantiles: Number of equal-frequency quantiles requested.
        statistics: Per-quantile statistics ordered from ``1`` to ``N``.
        top_minus_bottom: Highest-quantile mean return minus lowest-quantile
            mean return.
        monotonic: ``True`` when mean returns are non-decreasing from
            quantile ``1`` through quantile ``N``.
    """

    factor_column: str
    target_column: str
    quantiles: int
    statistics: tuple[QuantileStatistics, ...]
    top_minus_bottom: float
    monotonic: bool


class QuantileAnalyzer:
    """Equal-frequency cross-sectional quantile analyzer.

    Observations are assigned into ``quantiles`` buckets by factor rank.
    Quantile ``1`` holds the lowest factor values and quantile ``N`` holds
    the highest. The input DataFrame is never mutated.
    """

    __slots__ = ("_quantiles",)

    def __init__(self, quantiles: int = _DEFAULT_QUANTILES) -> None:
        """Initialize the analyzer.

        Args:
            quantiles: Number of equal-frequency quantiles. Must be ``>= 2``.
                Defaults to ``5``.

        Raises:
            ResearchError: If ``quantiles`` is not an integer ``>= 2``.
        """
        if (
            not isinstance(cast(object, quantiles), int)
            or isinstance(quantiles, bool)
            or quantiles < 2
        ):
            raise ResearchError(
                "quantiles must be an integer greater than or equal to 2",
                error_code=_ERROR_QUANTILES_INVALID,
                details={"parameter": "quantiles", "value": quantiles},
            )
        self._quantiles = quantiles

    @property
    def quantiles(self) -> int:
        """Return the configured number of equal-frequency quantiles."""
        return self._quantiles

    def analyze(
        self,
        frame: pl.DataFrame,
        factor_column: str,
        target_column: str,
    ) -> QuantileAnalysisResult:
        """Analyze forward returns across factor quantiles.

        Rows with a null factor or target value are dropped. Remaining
        observations are assigned into equal-frequency quantiles by factor
        ordinal rank.

        Args:
            frame: Input research DataFrame. Must not be mutated.
            factor_column: Factor column used for ranking.
            target_column: Forward-return target column.

        Returns:
            An immutable ``QuantileAnalysisResult``.

        Raises:
            ResearchError: If a required column is missing or fewer than
                ``quantiles`` paired non-null observations remain.
        """
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
        if target_column not in frame.columns:
            raise ResearchError(
                f"required column missing: {target_column}",
                error_code=_ERROR_MISSING_TARGET,
                details={
                    "required_column": target_column,
                    "role": "target",
                    "available_columns": tuple(frame.columns),
                },
            )

        clean = frame.select(factor_column, target_column).drop_nulls()
        observations = clean.height
        if observations < self._quantiles:
            raise ResearchError(
                "insufficient observations for quantile analysis",
                error_code=_ERROR_INSUFFICIENT_OBS,
                details={
                    "factor_column": factor_column,
                    "target_column": target_column,
                    "observations": observations,
                    "minimum_observations": self._quantiles,
                    "quantiles": self._quantiles,
                },
            )

        assigned = clean.with_columns(
            _quantile_assignment_expr(factor_column, observations, self._quantiles).alias(
                _QUANTILE_COLUMN
            )
        )
        aggregated = (
            assigned.group_by(_QUANTILE_COLUMN)
            .agg(
                pl.len().alias("count"),
                pl.col(target_column).mean().alias("mean_return"),
                pl.col(target_column).median().alias("median_return"),
                pl.col(target_column).std().alias("std_return"),
                pl.col(target_column).min().alias("min_return"),
                pl.col(target_column).max().alias("max_return"),
            )
            .sort(_QUANTILE_COLUMN)
        )

        statistics = _statistics_from_frame(aggregated)
        means = tuple(item.mean_return for item in statistics)
        top_minus_bottom = means[-1] - means[0]
        monotonic = all(means[index] <= means[index + 1] for index in range(len(means) - 1))

        return QuantileAnalysisResult(
            factor_column=factor_column,
            target_column=target_column,
            quantiles=self._quantiles,
            statistics=statistics,
            top_minus_bottom=float(top_minus_bottom),
            monotonic=monotonic,
        )


def _quantile_assignment_expr(
    factor_column: str,
    observations: int,
    quantiles: int,
) -> pl.Expr:
    """Build an expression assigning equal-frequency quantile indices ``1..N``."""
    return (
        ((pl.col(factor_column).rank(method="ordinal") - 1) * quantiles) // observations + 1
    ).clip(1, quantiles)


def _statistics_from_frame(aggregated: pl.DataFrame) -> tuple[QuantileStatistics, ...]:
    """Build ordered ``QuantileStatistics`` from an aggregated Polars frame."""
    quantiles = aggregated.get_column(_QUANTILE_COLUMN).to_list()
    counts = aggregated.get_column("count").to_list()
    means = aggregated.get_column("mean_return").to_list()
    medians = aggregated.get_column("median_return").to_list()
    stds = aggregated.get_column("std_return").to_list()
    mins = aggregated.get_column("min_return").to_list()
    maxs = aggregated.get_column("max_return").to_list()

    return tuple(
        QuantileStatistics(
            quantile=int(quantile),
            count=int(count),
            mean_return=float(mean_return),
            median_return=float(median_return),
            std_return=float("nan") if std_return is None else float(std_return),
            min_return=float(min_return),
            max_return=float(max_return),
        )
        for (
            quantile,
            count,
            mean_return,
            median_return,
            std_return,
            min_return,
            max_return,
        ) in zip(quantiles, counts, means, medians, stds, mins, maxs, strict=True)
    )

"""CQROS research reporting summaries.

Purpose:
    Provide immutable, presentation-independent summaries of research results
    produced by ``FactorResearchRunner``.

Responsibilities:
    - Aggregate ``FactorResearchRunResult`` into immutable report models
    - Preserve existing leaderboard ordering
    - Summarize factors, symbols, timeframes, correlations, skips, and failures
    - Identify best and worst factors from already-computed metrics
    - Remain free of factor computation, metric recomputation, charts, HTML,
      markdown, serialization, storage, CLI, and file writing

Dependencies:
    The Python standard library, ``cqros.core.types``, and
    ``cqros.research.runner`` result types.

Public API:
    ``OverallStatistics``, ``LeaderboardSummary``, ``FactorSummary``,
    ``SymbolSummary``, ``TimeframeSummary``, ``CorrelationSummary``,
    ``SkippedFactorSummary``, ``FailedFactorSummary``, ``ResearchReport``,
    ``ResearchReportGenerator``
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from cqros.core.types import Symbol, Timeframe
from cqros.research.runner import (
    FactorLeaderboardEntry,
    FactorResearchRunResult,
    SkippedFactorEvaluation,
)

__all__ = [
    "CorrelationSummary",
    "FactorSummary",
    "FailedFactorSummary",
    "LeaderboardSummary",
    "OverallStatistics",
    "ResearchReport",
    "ResearchReportGenerator",
    "SkippedFactorSummary",
    "SymbolSummary",
    "TimeframeSummary",
]

_STAGE_COMPUTE: Final[str] = "compute"
_STAGE_ANALYZE: Final[str] = "analyze"


@dataclass(frozen=True, slots=True)
class OverallStatistics:
    """Immutable overall statistics for a factor research run.

    Attributes:
        total_assets: Number of symbols included in the run.
        total_factors: Number of distinct factor names observed.
        total_experiments: Number of successful experiment records stored.
        successful_evaluations: Number of successful factor evaluations.
        failed_evaluations: Number of analysis-stage failures.
        skipped_evaluations: Number of compute-stage skips.
        average_ic: Mean IC across successful leaderboard entries, or ``None``.
        average_rank_ic: Mean Rank IC across successful leaderboard entries,
            or ``None``.
    """

    total_assets: int
    total_factors: int
    total_experiments: int
    successful_evaluations: int
    failed_evaluations: int
    skipped_evaluations: int
    average_ic: float | None
    average_rank_ic: float | None


@dataclass(frozen=True, slots=True)
class LeaderboardSummary:
    """Immutable leaderboard summary preserving runner ordering.

    Attributes:
        entries: Leaderboard rows in the same order produced by the runner.
    """

    entries: tuple[FactorLeaderboardEntry, ...]


@dataclass(frozen=True, slots=True)
class FactorSummary:
    """Immutable per-factor aggregate summary across successful evaluations.

    Attributes:
        factor_name: Registered factor name.
        mean_ic: Mean information coefficient.
        mean_rank_ic: Mean rank information coefficient.
        mean_stability: Mean stability score.
        mean_quantile_spread: Mean top-minus-bottom quantile spread.
        mean_decay_half_life: Mean decay half-life over non-null observations,
            or ``None`` when no half-life is available.
        evaluated_assets: Number of distinct symbols with successful evaluations.
        failed_evaluations: Number of analysis-stage failures for this factor.
    """

    factor_name: str
    mean_ic: float
    mean_rank_ic: float
    mean_stability: float
    mean_quantile_spread: float
    mean_decay_half_life: float | None
    evaluated_assets: int
    failed_evaluations: int


@dataclass(frozen=True, slots=True)
class SymbolSummary:
    """Immutable per-symbol research summary.

    Attributes:
        symbol: Symbol identifier.
        evaluated_factors: Distinct factor names successfully evaluated.
        best_factor: Factor with the highest mean IC on this symbol, or
            ``None`` when no successful evaluations exist.
        average_ic: Mean IC across successful evaluations on this symbol.
        average_rank_ic: Mean Rank IC across successful evaluations.
    """

    symbol: Symbol
    evaluated_factors: tuple[str, ...]
    best_factor: str | None
    average_ic: float | None
    average_rank_ic: float | None


@dataclass(frozen=True, slots=True)
class TimeframeSummary:
    """Immutable per-timeframe research summary.

    Attributes:
        timeframe: Timeframe identifier.
        evaluated_factors: Distinct factor names successfully evaluated.
        best_factor: Factor with the highest mean IC on this timeframe, or
            ``None`` when no successful evaluations exist.
        average_ic: Mean IC across successful evaluations on this timeframe.
    """

    timeframe: Timeframe
    evaluated_factors: tuple[str, ...]
    best_factor: str | None
    average_ic: float | None


@dataclass(frozen=True, slots=True)
class CorrelationSummary:
    """Immutable highly correlated factor pair from experiment results.

    Attributes:
        factor_a: First factor column in the correlated pair.
        factor_b: Second factor column in the correlated pair.
        correlation: Absolute or signed correlation value reported by the
            experiment.
        symbol: Symbol context of the source experiment.
        timeframe: Timeframe context of the source experiment.
        target_horizon: Forward-return horizon of the source experiment.
    """

    factor_a: str
    factor_b: str
    correlation: float
    symbol: Symbol
    timeframe: Timeframe
    target_horizon: int


@dataclass(frozen=True, slots=True)
class SkippedFactorSummary:
    """Immutable summary of a compute-stage skipped factor evaluation.

    Attributes:
        factor_name: Factor that could not be computed.
        symbol: Symbol being evaluated when skipped.
        timeframe: Timeframe being evaluated when skipped.
        reason: Human-readable skip reason.
        error_code: Optional CQROS error code.
    """

    factor_name: str
    symbol: Symbol
    timeframe: Timeframe
    reason: str
    error_code: str | None


@dataclass(frozen=True, slots=True)
class FailedFactorSummary:
    """Immutable summary of an analysis-stage failed factor evaluation.

    Attributes:
        factor_name: Factor that failed during analysis.
        symbol: Symbol being evaluated when the failure occurred.
        timeframe: Timeframe being evaluated when the failure occurred.
        reason: Human-readable failure reason.
        error_code: Optional CQROS error code.
        target_horizon: Horizon associated with the failed analysis, if any.
    """

    factor_name: str
    symbol: Symbol
    timeframe: Timeframe
    reason: str
    error_code: str | None
    target_horizon: int | None


@dataclass(frozen=True, slots=True)
class ResearchReport:
    """Immutable presentation-independent research report.

    Attributes:
        overall: Aggregate run statistics.
        leaderboard: Leaderboard summary preserving runner ordering.
        factors: Per-factor aggregate summaries ordered by factor name.
        symbols: Per-symbol summaries ordered by symbol.
        timeframes: Per-timeframe summaries ordered by timeframe.
        correlations: Highly correlated pairs collected from experiments.
        skipped: Compute-stage skips.
        failed: Analysis-stage failures.
        best_by_information_coefficient: Factor with highest mean IC.
        best_by_rank_ic: Factor with highest mean Rank IC.
        best_by_stability: Factor with highest mean stability.
        best_by_decay_half_life: Factor with highest mean decay half-life.
        best_by_quantile_spread: Factor with highest mean quantile spread.
        worst_by_information_coefficient: Factor with lowest mean IC.
        worst_by_rank_ic: Factor with lowest mean Rank IC.
    """

    overall: OverallStatistics
    leaderboard: LeaderboardSummary
    factors: tuple[FactorSummary, ...]
    symbols: tuple[SymbolSummary, ...]
    timeframes: tuple[TimeframeSummary, ...]
    correlations: tuple[CorrelationSummary, ...]
    skipped: tuple[SkippedFactorSummary, ...]
    failed: tuple[FailedFactorSummary, ...]
    best_by_information_coefficient: str | None
    best_by_rank_ic: str | None
    best_by_stability: str | None
    best_by_decay_half_life: str | None
    best_by_quantile_spread: str | None
    worst_by_information_coefficient: str | None
    worst_by_rank_ic: str | None


class ResearchReportGenerator:
    """Pure reporting generator for ``FactorResearchRunResult`` summaries.

    The generator performs aggregation and ranking only. It never recomputes
    research metrics, mutates the input result, or produces presentation
    artifacts.
    """

    __slots__ = ()

    def generate(self, result: FactorResearchRunResult) -> ResearchReport:
        """Build an immutable research report from a runner result.

        Args:
            result: Immutable factor research run result. Must not be mutated.

        Returns:
            An immutable ``ResearchReport`` summarizing the run.
        """
        skipped_entries, failed_entries = _partition_skips(result.skipped)
        factor_summaries = _build_factor_summaries(result, failed_entries)
        symbol_summaries = _build_symbol_summaries(result)
        timeframe_summaries = _build_timeframe_summaries(result)
        correlations = _build_correlation_summaries(result)
        overall = _build_overall_statistics(
            result,
            factor_summaries=factor_summaries,
            skipped_count=len(skipped_entries),
            failed_count=len(failed_entries),
        )

        return ResearchReport(
            overall=overall,
            leaderboard=LeaderboardSummary(entries=result.leaderboard),
            factors=factor_summaries,
            symbols=symbol_summaries,
            timeframes=timeframe_summaries,
            correlations=correlations,
            skipped=skipped_entries,
            failed=failed_entries,
            best_by_information_coefficient=_best_factor(
                factor_summaries,
                key=lambda summary: summary.mean_ic,
            ),
            best_by_rank_ic=_best_factor(
                factor_summaries,
                key=lambda summary: summary.mean_rank_ic,
            ),
            best_by_stability=_best_factor(
                factor_summaries,
                key=lambda summary: summary.mean_stability,
            ),
            best_by_decay_half_life=_best_factor(
                factor_summaries,
                key=lambda summary: (
                    summary.mean_decay_half_life
                    if summary.mean_decay_half_life is not None
                    else float("-inf")
                ),
            ),
            best_by_quantile_spread=_best_factor(
                factor_summaries,
                key=lambda summary: summary.mean_quantile_spread,
            ),
            worst_by_information_coefficient=_worst_factor(
                factor_summaries,
                key=lambda summary: summary.mean_ic,
            ),
            worst_by_rank_ic=_worst_factor(
                factor_summaries,
                key=lambda summary: summary.mean_rank_ic,
            ),
        )


def _partition_skips(
    skipped: tuple[SkippedFactorEvaluation, ...],
) -> tuple[tuple[SkippedFactorSummary, ...], tuple[FailedFactorSummary, ...]]:
    """Split runner skips into compute skips and analysis failures."""
    skipped_summaries: list[SkippedFactorSummary] = []
    failed_summaries: list[FailedFactorSummary] = []
    for item in skipped:
        if item.stage == _STAGE_ANALYZE:
            failed_summaries.append(
                FailedFactorSummary(
                    factor_name=item.factor_name,
                    symbol=item.symbol,
                    timeframe=item.timeframe,
                    reason=item.reason,
                    error_code=item.error_code,
                    target_horizon=item.target_horizon,
                )
            )
        else:
            skipped_summaries.append(
                SkippedFactorSummary(
                    factor_name=item.factor_name,
                    symbol=item.symbol,
                    timeframe=item.timeframe,
                    reason=item.reason,
                    error_code=item.error_code,
                )
            )
    return tuple(skipped_summaries), tuple(failed_summaries)


def _build_factor_summaries(
    result: FactorResearchRunResult,
    failed_entries: tuple[FailedFactorSummary, ...],
) -> tuple[FactorSummary, ...]:
    """Aggregate successful leaderboard rows into per-factor summaries."""
    by_factor: dict[str, list[FactorLeaderboardEntry]] = defaultdict(list)
    for entry in result.leaderboard:
        by_factor[entry.factor_name].append(entry)

    failed_counts: dict[str, int] = defaultdict(int)
    for failure in failed_entries:
        failed_counts[failure.factor_name] += 1

    summaries: list[FactorSummary] = []
    for factor_name in sorted(by_factor):
        entries = by_factor[factor_name]
        half_lives = [
            float(entry.decay_half_life) for entry in entries if entry.decay_half_life is not None
        ]
        summaries.append(
            FactorSummary(
                factor_name=factor_name,
                mean_ic=statistics.fmean(entry.mean_ic for entry in entries),
                mean_rank_ic=statistics.fmean(entry.mean_rank_ic for entry in entries),
                mean_stability=statistics.fmean(entry.stability_score for entry in entries),
                mean_quantile_spread=statistics.fmean(entry.quantile_spread for entry in entries),
                mean_decay_half_life=(statistics.fmean(half_lives) if half_lives else None),
                evaluated_assets=len({entry.symbol for entry in entries}),
                failed_evaluations=failed_counts.get(factor_name, 0),
            )
        )
    return tuple(summaries)


def _build_symbol_summaries(
    result: FactorResearchRunResult,
) -> tuple[SymbolSummary, ...]:
    """Aggregate successful evaluations into per-symbol summaries."""
    by_symbol: dict[Symbol, list[FactorLeaderboardEntry]] = defaultdict(list)
    for entry in result.leaderboard:
        by_symbol[entry.symbol].append(entry)

    summaries: list[SymbolSummary] = []
    for symbol in result.symbols:
        entries = by_symbol.get(symbol, [])
        if not entries:
            summaries.append(
                SymbolSummary(
                    symbol=symbol,
                    evaluated_factors=(),
                    best_factor=None,
                    average_ic=None,
                    average_rank_ic=None,
                )
            )
            continue

        factor_means = _mean_ic_by_factor(entries)
        best_factor = max(factor_means.items(), key=lambda item: item[1])[0]
        summaries.append(
            SymbolSummary(
                symbol=symbol,
                evaluated_factors=tuple(sorted({entry.factor_name for entry in entries})),
                best_factor=best_factor,
                average_ic=statistics.fmean(entry.mean_ic for entry in entries),
                average_rank_ic=statistics.fmean(entry.mean_rank_ic for entry in entries),
            )
        )
    return tuple(summaries)


def _build_timeframe_summaries(
    result: FactorResearchRunResult,
) -> tuple[TimeframeSummary, ...]:
    """Aggregate successful evaluations into per-timeframe summaries."""
    by_timeframe: dict[Timeframe, list[FactorLeaderboardEntry]] = defaultdict(list)
    for entry in result.leaderboard:
        by_timeframe[entry.timeframe].append(entry)

    summaries: list[TimeframeSummary] = []
    for timeframe in result.timeframes:
        entries = by_timeframe.get(timeframe, [])
        if not entries:
            summaries.append(
                TimeframeSummary(
                    timeframe=timeframe,
                    evaluated_factors=(),
                    best_factor=None,
                    average_ic=None,
                )
            )
            continue

        factor_means = _mean_ic_by_factor(entries)
        best_factor = max(factor_means.items(), key=lambda item: item[1])[0]
        summaries.append(
            TimeframeSummary(
                timeframe=timeframe,
                evaluated_factors=tuple(sorted({entry.factor_name for entry in entries})),
                best_factor=best_factor,
                average_ic=statistics.fmean(entry.mean_ic for entry in entries),
            )
        )
    return tuple(summaries)


def _build_correlation_summaries(
    result: FactorResearchRunResult,
) -> tuple[CorrelationSummary, ...]:
    """Collect highly correlated pairs from stored experiment results."""
    summaries: list[CorrelationSummary] = []
    seen: set[tuple[str, str, str, str, int, float]] = set()
    for record in result.records:
        horizon = record.experiment_result.definition.target_horizon
        for factor_a, factor_b, correlation in record.experiment_result.highly_correlated_pairs:
            key = (
                record.symbol,
                record.timeframe,
                factor_a,
                factor_b,
                horizon,
                correlation,
            )
            if key in seen:
                continue
            seen.add(key)
            summaries.append(
                CorrelationSummary(
                    factor_a=factor_a,
                    factor_b=factor_b,
                    correlation=correlation,
                    symbol=record.symbol,
                    timeframe=record.timeframe,
                    target_horizon=horizon,
                )
            )
    return tuple(summaries)


def _build_overall_statistics(
    result: FactorResearchRunResult,
    *,
    factor_summaries: tuple[FactorSummary, ...],
    skipped_count: int,
    failed_count: int,
) -> OverallStatistics:
    """Build overall run statistics from the runner result."""
    factor_names = {summary.factor_name for summary in factor_summaries}
    for item in result.skipped:
        factor_names.add(item.factor_name)

    average_ic: float | None = None
    average_rank_ic: float | None = None
    if result.leaderboard:
        average_ic = statistics.fmean(entry.mean_ic for entry in result.leaderboard)
        average_rank_ic = statistics.fmean(entry.mean_rank_ic for entry in result.leaderboard)

    return OverallStatistics(
        total_assets=len(result.symbols),
        total_factors=len(factor_names),
        total_experiments=len(result.records),
        successful_evaluations=len(result.leaderboard),
        failed_evaluations=failed_count,
        skipped_evaluations=skipped_count,
        average_ic=average_ic,
        average_rank_ic=average_rank_ic,
    )


def _mean_ic_by_factor(
    entries: list[FactorLeaderboardEntry],
) -> dict[str, float]:
    """Return mean IC keyed by factor name for a subset of entries."""
    by_factor: dict[str, list[float]] = defaultdict(list)
    for entry in entries:
        by_factor[entry.factor_name].append(entry.mean_ic)
    return {factor_name: statistics.fmean(values) for factor_name, values in by_factor.items()}


def _best_factor(
    summaries: tuple[FactorSummary, ...],
    *,
    key: Callable[[FactorSummary], float],
) -> str | None:
    """Return the factor name maximizing ``key``, or ``None`` when empty."""
    if not summaries:
        return None
    return max(summaries, key=key).factor_name


def _worst_factor(
    summaries: tuple[FactorSummary, ...],
    *,
    key: Callable[[FactorSummary], float],
) -> str | None:
    """Return the factor name minimizing ``key``, or ``None`` when empty."""
    if not summaries:
        return None
    return min(summaries, key=key).factor_name

"""Unit tests for CQROS research reporting module."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from datetime import UTC, datetime

import pytest

from cqros.research.experiment import ExperimentDefinition, ExperimentResult
from cqros.research.factor_decay import FactorDecayResult
from cqros.research.factor_stability import FactorStabilityResult
from cqros.research.information_coefficient import InformationCoefficientResult
from cqros.research.quantile_analysis import QuantileAnalysisResult
from cqros.research.rank_ic import RankICResult
from cqros.research.report import (
    CorrelationSummary,
    FactorSummary,
    FailedFactorSummary,
    LeaderboardSummary,
    OverallStatistics,
    ResearchReport,
    ResearchReportGenerator,
    SkippedFactorSummary,
    SymbolSummary,
    TimeframeSummary,
)
from cqros.research.runner import (
    AssetExperimentRecord,
    FactorLeaderboardEntry,
    FactorResearchRunnerConfig,
    FactorResearchRunResult,
    SkippedFactorEvaluation,
)
from cqros.research.target import TargetDefinition


def _now() -> datetime:
    """Return a fixed UTC timestamp for immutable fixtures."""
    return datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


def _entry(
    *,
    rank: int,
    factor_name: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    horizon: int = 1,
    mean_ic: float,
    mean_rank_ic: float | None = None,
    stability_score: float = 0.5,
    decay_half_life: int | None = 4,
    quantile_spread: float = 0.01,
) -> FactorLeaderboardEntry:
    """Build a leaderboard entry fixture."""
    return FactorLeaderboardEntry(
        rank=rank,
        factor_name=factor_name,
        factor_column=factor_name,
        symbol=symbol,
        timeframe=timeframe,
        target_horizon=horizon,
        mean_ic=mean_ic,
        mean_rank_ic=mean_ic if mean_rank_ic is None else mean_rank_ic,
        stability_score=stability_score,
        decay_half_life=decay_half_life,
        quantile_spread=quantile_spread,
    )


def _experiment_result(
    *,
    name: str,
    horizon: int = 1,
    correlated_pairs: tuple[tuple[str, str, float], ...] = (),
) -> ExperimentResult:
    """Build a minimal ExperimentResult for correlation reporting tests."""
    now = _now()
    return ExperimentResult(
        definition=ExperimentDefinition(
            name=name,
            description="fixture",
            factor_columns=(name,),
            target_horizon=horizon,
        ),
        target=TargetDefinition(name="forward_return", horizon=horizon),
        information_coefficients=(
            InformationCoefficientResult(
                factor_column=name,
                target_column="forward_return",
                method="pearson",
                observations=10,
                coefficient=0.1,
                p_value=0.2,
            ),
        ),
        rank_information_coefficients=(
            RankICResult(
                factor_column=name,
                target_column="forward_return",
                observations=10,
                coefficient=0.1,
                p_value=0.2,
            ),
        ),
        quantile_results=(
            QuantileAnalysisResult(
                factor_column=name,
                target_column="forward_return",
                quantiles=5,
                statistics=(),
                top_minus_bottom=0.01,
                monotonic=True,
            ),
        ),
        decay_results=(
            FactorDecayResult(
                factor_column=name,
                price_column="close",
                method="pearson",
                points=(),
                half_life=4,
            ),
        ),
        stability_results=(
            FactorStabilityResult(
                factor_column=name,
                target_column="forward_return",
                method="pearson",
                window_size=10,
                windows=(),
                mean_ic=0.1,
                std_ic=0.1,
                min_ic=0.0,
                max_ic=0.2,
                stability_score=0.5,
            ),
        ),
        correlation_result=None,
        highly_correlated_pairs=correlated_pairs,
        started_at=now,
        completed_at=now,
        duration_seconds=0.0,
    )


def _run_result(
    *,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    timeframes: tuple[str, ...] = ("1h", "4h"),
    leaderboard: tuple[FactorLeaderboardEntry, ...] = (),
    records: tuple[AssetExperimentRecord, ...] = (),
    skipped: tuple[SkippedFactorEvaluation, ...] = (),
) -> FactorResearchRunResult:
    """Build a FactorResearchRunResult fixture."""
    now = _now()
    return FactorResearchRunResult(
        config=FactorResearchRunnerConfig(target_horizons=(1, 5)),
        symbols=symbols,
        timeframes=timeframes,
        records=records,
        leaderboard=leaderboard,
        skipped=skipped,
        started_at=now,
        completed_at=now,
        duration_seconds=1.0,
    )


def test_report_models_are_frozen() -> None:
    """All report models are immutable slotted dataclasses."""
    report = ResearchReportGenerator().generate(_run_result())
    for model in (
        report,
        report.overall,
        report.leaderboard,
        OverallStatistics,
        LeaderboardSummary,
        FactorSummary,
        SymbolSummary,
        TimeframeSummary,
        CorrelationSummary,
        SkippedFactorSummary,
        FailedFactorSummary,
        ResearchReport,
    ):
        assert is_dataclass(model if not isinstance(model, type) else model)

    with pytest.raises(FrozenInstanceError):
        report.best_by_information_coefficient = "x"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.overall.total_assets = 0  # type: ignore[misc]


def test_empty_run_produces_empty_summaries() -> None:
    """An empty runner result yields empty collections and null averages."""
    report = ResearchReportGenerator().generate(
        _run_result(symbols=("BTCUSDT",), timeframes=("1h",))
    )
    assert report.overall.total_assets == 1
    assert report.overall.total_factors == 0
    assert report.overall.total_experiments == 0
    assert report.overall.successful_evaluations == 0
    assert report.overall.failed_evaluations == 0
    assert report.overall.skipped_evaluations == 0
    assert report.overall.average_ic is None
    assert report.overall.average_rank_ic is None
    assert report.leaderboard.entries == ()
    assert report.factors == ()
    assert report.symbols == (
        SymbolSummary(
            symbol="BTCUSDT",
            evaluated_factors=(),
            best_factor=None,
            average_ic=None,
            average_rank_ic=None,
        ),
    )
    assert report.timeframes == (
        TimeframeSummary(
            timeframe="1h",
            evaluated_factors=(),
            best_factor=None,
            average_ic=None,
        ),
    )
    assert report.best_by_information_coefficient is None
    assert report.worst_by_rank_ic is None


def test_leaderboard_preserves_runner_ordering() -> None:
    """Leaderboard summary keeps the runner's existing ranking order."""
    leaderboard = (
        _entry(rank=1, factor_name="alpha", mean_ic=0.5),
        _entry(rank=2, factor_name="beta", mean_ic=0.2),
    )
    report = ResearchReportGenerator().generate(_run_result(leaderboard=leaderboard))
    assert report.leaderboard.entries == leaderboard
    assert [entry.rank for entry in report.leaderboard.entries] == [1, 2]


def test_per_factor_summaries_aggregate_metrics() -> None:
    """Factor summaries average metrics and count assets and failures."""
    leaderboard = (
        _entry(
            rank=1,
            factor_name="momentum",
            symbol="BTCUSDT",
            mean_ic=0.4,
            mean_rank_ic=0.3,
            stability_score=0.8,
            decay_half_life=8,
            quantile_spread=0.02,
        ),
        _entry(
            rank=2,
            factor_name="momentum",
            symbol="ETHUSDT",
            mean_ic=0.2,
            mean_rank_ic=0.1,
            stability_score=0.4,
            decay_half_life=4,
            quantile_spread=0.04,
        ),
        _entry(
            rank=3,
            factor_name="breakout",
            symbol="BTCUSDT",
            mean_ic=0.1,
            mean_rank_ic=0.05,
            stability_score=0.2,
            decay_half_life=None,
            quantile_spread=0.01,
        ),
    )
    skipped = (
        SkippedFactorEvaluation(
            factor_name="momentum",
            symbol="SOLUSDT",
            timeframe="1h",
            stage="analyze",
            reason="insufficient observations",
            error_code="RESEARCH-STABILITY-004",
            target_horizon=5,
        ),
    )
    report = ResearchReportGenerator().generate(
        _run_result(leaderboard=leaderboard, skipped=skipped)
    )

    assert len(report.factors) == 2
    breakout = report.factors[0]
    momentum = report.factors[1]
    assert breakout.factor_name == "breakout"
    assert momentum.factor_name == "momentum"
    assert momentum.mean_ic == pytest.approx(0.3)
    assert momentum.mean_rank_ic == pytest.approx(0.2)
    assert momentum.mean_stability == pytest.approx(0.6)
    assert momentum.mean_quantile_spread == pytest.approx(0.03)
    assert momentum.mean_decay_half_life == pytest.approx(6.0)
    assert momentum.evaluated_assets == 2
    assert momentum.failed_evaluations == 1
    assert breakout.mean_decay_half_life is None
    assert breakout.failed_evaluations == 0


def test_symbol_and_timeframe_summaries() -> None:
    """Symbol and timeframe summaries identify best factors and averages."""
    leaderboard = (
        _entry(
            rank=1,
            factor_name="momentum",
            symbol="BTCUSDT",
            timeframe="1h",
            mean_ic=0.5,
            mean_rank_ic=0.4,
        ),
        _entry(
            rank=2,
            factor_name="breakout",
            symbol="BTCUSDT",
            timeframe="1h",
            mean_ic=0.1,
            mean_rank_ic=0.05,
        ),
        _entry(
            rank=3,
            factor_name="momentum",
            symbol="ETHUSDT",
            timeframe="4h",
            mean_ic=0.2,
            mean_rank_ic=0.15,
        ),
    )
    report = ResearchReportGenerator().generate(
        _run_result(
            symbols=("BTCUSDT", "ETHUSDT"),
            timeframes=("1h", "4h"),
            leaderboard=leaderboard,
        )
    )

    btc = next(item for item in report.symbols if item.symbol == "BTCUSDT")
    eth = next(item for item in report.symbols if item.symbol == "ETHUSDT")
    assert btc.evaluated_factors == ("breakout", "momentum")
    assert btc.best_factor == "momentum"
    assert btc.average_ic == pytest.approx(0.3)
    assert btc.average_rank_ic == pytest.approx(0.225)
    assert eth.best_factor == "momentum"

    one_hour = next(item for item in report.timeframes if item.timeframe == "1h")
    four_hour = next(item for item in report.timeframes if item.timeframe == "4h")
    assert one_hour.best_factor == "momentum"
    assert one_hour.average_ic == pytest.approx(0.3)
    assert four_hour.evaluated_factors == ("momentum",)
    assert four_hour.average_ic == pytest.approx(0.2)


def test_best_and_worst_factors() -> None:
    """Best/worst factor selections use aggregated per-factor means."""
    leaderboard = (
        _entry(
            rank=1,
            factor_name="strong",
            mean_ic=0.6,
            mean_rank_ic=0.5,
            stability_score=0.9,
            decay_half_life=12,
            quantile_spread=0.08,
        ),
        _entry(
            rank=2,
            factor_name="weak",
            mean_ic=-0.2,
            mean_rank_ic=-0.3,
            stability_score=0.1,
            decay_half_life=2,
            quantile_spread=0.01,
        ),
    )
    report = ResearchReportGenerator().generate(_run_result(leaderboard=leaderboard))
    assert report.best_by_information_coefficient == "strong"
    assert report.best_by_rank_ic == "strong"
    assert report.best_by_stability == "strong"
    assert report.best_by_decay_half_life == "strong"
    assert report.best_by_quantile_spread == "strong"
    assert report.worst_by_information_coefficient == "weak"
    assert report.worst_by_rank_ic == "weak"


def test_skipped_and_failed_partitioning() -> None:
    """Compute skips and analysis failures are reported separately."""
    skipped = (
        SkippedFactorEvaluation(
            factor_name="bad_compute",
            symbol="BTCUSDT",
            timeframe="1h",
            stage="compute",
            reason="missing column",
            error_code="FACTOR-X-001",
        ),
        SkippedFactorEvaluation(
            factor_name="bad_analyze",
            symbol="ETHUSDT",
            timeframe="4h",
            stage="analyze",
            reason="insufficient observations",
            error_code="RESEARCH-IC-004",
            target_horizon=10,
        ),
    )
    report = ResearchReportGenerator().generate(_run_result(skipped=skipped))
    assert report.skipped == (
        SkippedFactorSummary(
            factor_name="bad_compute",
            symbol="BTCUSDT",
            timeframe="1h",
            reason="missing column",
            error_code="FACTOR-X-001",
        ),
    )
    assert report.failed == (
        FailedFactorSummary(
            factor_name="bad_analyze",
            symbol="ETHUSDT",
            timeframe="4h",
            reason="insufficient observations",
            error_code="RESEARCH-IC-004",
            target_horizon=10,
        ),
    )
    assert report.overall.skipped_evaluations == 1
    assert report.overall.failed_evaluations == 1
    assert report.overall.total_factors == 2


def test_correlation_summaries_from_experiment_results() -> None:
    """Highly correlated pairs are collected directly from ExperimentResult."""
    records = (
        AssetExperimentRecord(
            symbol="BTCUSDT",
            timeframe="1h",
            factor_name="momentum",
            experiment_result=_experiment_result(
                name="momentum",
                horizon=1,
                correlated_pairs=(("momentum", "breakout", 0.95),),
            ),
        ),
        AssetExperimentRecord(
            symbol="ETHUSDT",
            timeframe="4h",
            factor_name="breakout",
            experiment_result=_experiment_result(
                name="breakout",
                horizon=5,
                correlated_pairs=(("breakout", "recovery", 0.91),),
            ),
        ),
    )
    report = ResearchReportGenerator().generate(_run_result(records=records))
    assert report.correlations == (
        CorrelationSummary(
            factor_a="momentum",
            factor_b="breakout",
            correlation=0.95,
            symbol="BTCUSDT",
            timeframe="1h",
            target_horizon=1,
        ),
        CorrelationSummary(
            factor_a="breakout",
            factor_b="recovery",
            correlation=0.91,
            symbol="ETHUSDT",
            timeframe="4h",
            target_horizon=5,
        ),
    )


def test_overall_statistics_and_input_immutability() -> None:
    """Overall statistics use runner aggregates and never mutate inputs."""
    leaderboard = (
        _entry(rank=1, factor_name="momentum", mean_ic=0.4, mean_rank_ic=0.2),
        _entry(rank=2, factor_name="breakout", mean_ic=0.2, mean_rank_ic=0.1),
    )
    skipped = (
        SkippedFactorEvaluation(
            factor_name="failing",
            symbol="BTCUSDT",
            timeframe="1h",
            stage="compute",
            reason="forced",
            error_code="FACTOR-FAIL-001",
        ),
    )
    records = (
        AssetExperimentRecord(
            symbol="BTCUSDT",
            timeframe="1h",
            factor_name="momentum",
            experiment_result=_experiment_result(name="momentum"),
        ),
    )
    result = _run_result(
        symbols=("BTCUSDT", "ETHUSDT"),
        timeframes=("1h",),
        leaderboard=leaderboard,
        records=records,
        skipped=skipped,
    )
    original_leaderboard = result.leaderboard
    original_skipped = result.skipped

    report = ResearchReportGenerator().generate(result)

    assert report.overall.total_assets == 2
    assert report.overall.total_factors == 3
    assert report.overall.total_experiments == 1
    assert report.overall.successful_evaluations == 2
    assert report.overall.failed_evaluations == 0
    assert report.overall.skipped_evaluations == 1
    assert report.overall.average_ic == pytest.approx(0.3)
    assert report.overall.average_rank_ic == pytest.approx(0.15)
    assert result.leaderboard is original_leaderboard
    assert result.skipped is original_skipped
    assert result.leaderboard == original_leaderboard


def test_package_exports_report_api() -> None:
    """Report public types are exported from cqros.research."""
    import cqros.research as research_package

    for name in (
        "ResearchReport",
        "ResearchReportGenerator",
        "OverallStatistics",
        "LeaderboardSummary",
        "FactorSummary",
        "SymbolSummary",
        "TimeframeSummary",
        "CorrelationSummary",
        "SkippedFactorSummary",
        "FailedFactorSummary",
    ):
        assert name in research_package.__all__

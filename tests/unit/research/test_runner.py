"""Unit tests for CQROS Factor Research Runner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from cqros.core.exceptions import ExperimentError
from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorError
from cqros.factors.price import MomentumFactor
from cqros.factors.registry import FactorRegistry
from cqros.research.experiment import ExperimentDefinition, ExperimentResult
from cqros.research.factor_correlation import FactorCorrelationAnalyzer
from cqros.research.factor_decay import FactorDecayAnalyzer, FactorDecayResult
from cqros.research.factor_stability import FactorStabilityAnalyzer, FactorStabilityResult
from cqros.research.information_coefficient import (
    InformationCoefficient,
    InformationCoefficientResult,
)
from cqros.research.quantile_analysis import QuantileAnalysisResult, QuantileAnalyzer
from cqros.research.rank_ic import RankICResult, RankInformationCoefficient
from cqros.research.runner import (
    AssetExperimentRecord,
    FactorLeaderboardEntry,
    FactorResearchRunner,
    FactorResearchRunnerConfig,
    FactorResearchRunResult,
    SkippedFactorEvaluation,
    _build_leaderboard,
)
from cqros.research.target import TargetDefinition
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.storage.layout import StorageLayout


@dataclass(frozen=True, slots=True)
class _FailingFactor(BaseFactor):
    """Factor that always fails compute for skip-path tests."""

    name: str = "failing_factor"
    version: str = "1.0.0"
    description: str = "Always fails"
    category: str = "price"
    required_features: tuple[str, ...] = ("close",)
    produced_columns: tuple[str, ...] = ("failing_factor",)
    lookback: int = 1

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Raise FactorError unconditionally."""
        raise FactorError(
            "forced compute failure",
            error_code="FACTOR-FAIL-001",
            details={"factor": self.name},
        )


def _ohlcv_frame(rows: int = 80) -> pl.DataFrame:
    """Build a deterministic OHLCV-like frame with a close column."""
    close = [100.0]
    for index in range(rows - 1):
        shock = (((index * 47) % 13) - 6) / 80.0
        close.append(close[-1] * (1.0 + shock))
    return pl.DataFrame(
        {
            "open": close,
            "high": [value * 1.01 for value in close],
            "low": [value * 0.99 for value in close],
            "close": close,
            "volume": [float(1000 + index) for index in range(rows)],
        }
    )


def _config(**overrides: object) -> FactorResearchRunnerConfig:
    """Build a runner config tuned for small synthetic frames."""
    values: dict[str, object] = {
        "target_horizons": (1, 5),
        "decay_horizons": (1, 2),
        "stability_window": 20,
        "quantiles": 5,
    }
    values.update(overrides)
    return FactorResearchRunnerConfig(**values)  # type: ignore[arg-type]


def _registry(*factors: BaseFactor) -> FactorRegistry:
    """Build a registry populated with the supplied factors."""
    registry = FactorRegistry()
    registry.register_many(factors)
    return registry


def _mock_repository(
    frames_by_key: dict[tuple[str, str, int], pl.DataFrame] | None = None,
    *,
    existing: set[tuple[str, str]] | None = None,
) -> MagicMock:
    """Build a MarketDataRepository mock with multi-year load behavior."""
    repository = MagicMock()
    stored = frames_by_key if frames_by_key is not None else {}
    present = existing if existing is not None else {(key[0], key[1]) for key in stored}

    def load_ohlcv(**kwargs: Any) -> pl.DataFrame:
        key = (kwargs["symbol"], kwargs["timeframe"], kwargs["year"])
        if key not in stored:
            raise DatasetNotFoundError(
                "dataset not found",
                error_code="STORAGE-NOT-FOUND",
                details={"key": key},
            )
        return stored[key]

    def has_ohlcv(symbol: str, timeframe: str) -> bool:
        return (symbol, timeframe) in present

    repository.load_ohlcv.side_effect = load_ohlcv
    repository.has_ohlcv.side_effect = has_ohlcv
    return repository


def _runner(
    *,
    registry: FactorRegistry,
    repository: MagicMock | None = None,
    layout: StorageLayout | None = None,
    config: FactorResearchRunnerConfig | None = None,
    tmp_path: Path | None = None,
) -> FactorResearchRunner:
    """Build a FactorResearchRunner with real analyzers and mocked storage."""
    return FactorResearchRunner(
        repository=repository if repository is not None else _mock_repository(),
        registry=registry,
        layout=layout if layout is not None else StorageLayout(tmp_path or Path(".")),
        information_coefficient=InformationCoefficient(method="pearson"),
        rank_information_coefficient=RankInformationCoefficient(),
        quantile_analyzer=QuantileAnalyzer(quantiles=5),
        factor_decay_analyzer=FactorDecayAnalyzer(method="pearson"),
        factor_stability_analyzer=FactorStabilityAnalyzer(method="pearson"),
        factor_correlation_analyzer=FactorCorrelationAnalyzer(method="pearson"),
        config=config if config is not None else _config(),
    )


def test_config_and_result_types_are_frozen() -> None:
    """Runner config and result value objects are immutable dataclasses."""
    config = _config()
    assert is_dataclass(config)
    with pytest.raises(FrozenInstanceError):
        config.price_column = "open"  # type: ignore[misc]

    assert is_dataclass(FactorLeaderboardEntry)
    assert is_dataclass(AssetExperimentRecord)
    assert is_dataclass(SkippedFactorEvaluation)
    assert is_dataclass(FactorResearchRunResult)


def test_config_rejects_empty_and_invalid_horizons() -> None:
    """target_horizons must be a non-empty sequence of positive integers."""
    with pytest.raises(ExperimentError, match="at least one horizon") as empty_info:
        FactorResearchRunnerConfig(target_horizons=())
    assert empty_info.value.error_code == "RESEARCH-RUNNER-003"

    with pytest.raises(ExperimentError, match="greater than 0") as invalid_info:
        FactorResearchRunnerConfig(target_horizons=(1, 0, 5))
    assert invalid_info.value.error_code == "RESEARCH-RUNNER-004"


def test_run_requires_symbols_timeframes_and_registered_factors(tmp_path: Path) -> None:
    """Empty symbols, timeframes, or registry fail fast."""
    runner = _runner(registry=_registry(MomentumFactor(lookback=2)), tmp_path=tmp_path)
    with pytest.raises(ExperimentError, match="symbols") as symbols_info:
        runner.run(symbols=(), timeframes=("1h",))
    assert symbols_info.value.error_code == "RESEARCH-RUNNER-001"

    with pytest.raises(ExperimentError, match="timeframes") as timeframes_info:
        runner.run(symbols=("BTCUSDT",), timeframes=())
    assert timeframes_info.value.error_code == "RESEARCH-RUNNER-002"

    empty_runner = _runner(registry=FactorRegistry(), tmp_path=tmp_path)
    with pytest.raises(ExperimentError, match="no registered factors") as factors_info:
        empty_runner.run(symbols=("BTCUSDT",), timeframes=("1h",))
    assert factors_info.value.error_code == "RESEARCH-RUNNER-005"


def test_run_one_symbol_stores_experiment_results_and_leaderboard(tmp_path: Path) -> None:
    """Single-symbol runs store ExperimentResults and a ranked leaderboard."""
    frame = _ohlcv_frame(80)
    repository = _mock_repository({("BTCUSDT", "1h", 2024): frame})
    runner = _runner(
        registry=_registry(MomentumFactor(lookback=2)),
        repository=repository,
        tmp_path=tmp_path,
        config=_config(target_horizons=(1, 5), earliest_partition_year=2024),
    )

    result = runner.run(symbols=("BTCUSDT",), timeframes=("1h",))

    assert isinstance(result, FactorResearchRunResult)
    assert result.symbols == ("BTCUSDT",)
    assert result.timeframes == ("1h",)
    assert len(result.records) == 2
    assert all(isinstance(record.experiment_result, ExperimentResult) for record in result.records)
    assert len(result.leaderboard) == 2
    assert result.leaderboard[0].rank == 1
    assert result.leaderboard[1].rank == 2
    assert result.leaderboard[0].mean_ic >= result.leaderboard[1].mean_ic
    assert result.completed_at >= result.started_at
    assert result.duration_seconds >= 0.0
    with pytest.raises(FrozenInstanceError):
        result.duration_seconds = 0.0  # type: ignore[misc]


def test_run_multiple_symbols(tmp_path: Path) -> None:
    """Multiple symbols are evaluated independently."""
    frame = _ohlcv_frame(80)
    repository = _mock_repository(
        {
            ("BTCUSDT", "1h", 2024): frame,
            ("ETHUSDT", "1h", 2024): frame,
        }
    )
    runner = _runner(
        registry=_registry(MomentumFactor(lookback=2)),
        repository=repository,
        tmp_path=tmp_path,
        config=_config(target_horizons=(1,), earliest_partition_year=2024),
    )

    result = runner.run(symbols=("BTCUSDT", "ETHUSDT"), timeframes=("1h",))
    assert result.symbols == ("BTCUSDT", "ETHUSDT")
    assert {record.symbol for record in result.records} == {"BTCUSDT", "ETHUSDT"}
    assert len(result.records) == 2


def test_concatenates_multi_year_ohlcv_without_mutating_input(tmp_path: Path) -> None:
    """Multi-year partitions are concatenated and source frames stay intact."""
    year_a = _ohlcv_frame(40)
    year_b = _ohlcv_frame(40)
    original_a = year_a.clone()
    original_b = year_b.clone()
    repository = _mock_repository(
        {
            ("BTCUSDT", "1h", 2023): year_a,
            ("BTCUSDT", "1h", 2024): year_b,
        }
    )
    runner = _runner(
        registry=_registry(MomentumFactor(lookback=2)),
        repository=repository,
        tmp_path=tmp_path,
        config=_config(
            target_horizons=(1,),
            earliest_partition_year=2023,
            stability_window=20,
        ),
    )

    result = runner.run(symbols=("BTCUSDT",), timeframes=("1h",))
    assert len(result.records) == 1
    assert year_a.equals(original_a)
    assert year_b.equals(original_b)
    assert "momentum" not in year_a.columns
    assert "momentum" not in year_b.columns


def test_skips_failing_factor_and_continues(tmp_path: Path) -> None:
    """Factors that fail compute are skipped while others continue."""
    frame = _ohlcv_frame(80)
    repository = _mock_repository({("BTCUSDT", "1h", 2024): frame})
    runner = _runner(
        registry=_registry(_FailingFactor(), MomentumFactor(lookback=2)),
        repository=repository,
        tmp_path=tmp_path,
        config=_config(target_horizons=(1,), earliest_partition_year=2024),
    )

    result = runner.run(symbols=("BTCUSDT",), timeframes=("1h",))
    assert len(result.skipped) == 1
    assert result.skipped[0].factor_name == "failing_factor"
    assert result.skipped[0].stage == "compute"
    assert result.skipped[0].error_code == "FACTOR-FAIL-001"
    assert len(result.records) == 1
    assert result.records[0].factor_name == "momentum"


def test_skips_missing_ohlcv_symbol_and_continues(tmp_path: Path) -> None:
    """Symbols without OHLCV history are skipped without aborting the run."""
    frame = _ohlcv_frame(80)
    repository = _mock_repository({("BTCUSDT", "1h", 2024): frame})
    runner = _runner(
        registry=_registry(MomentumFactor(lookback=2)),
        repository=repository,
        tmp_path=tmp_path,
        config=_config(target_horizons=(1,), earliest_partition_year=2024),
    )

    result = runner.run(symbols=("BTCUSDT", "MISSING"), timeframes=("1h",))
    assert result.symbols == ("BTCUSDT", "MISSING")
    assert {record.symbol for record in result.records} == {"BTCUSDT"}


def test_skips_analysis_failure_and_continues(tmp_path: Path) -> None:
    """Analysis failures are skipped while other evaluations continue."""
    frame = _ohlcv_frame(30)
    repository = _mock_repository({("BTCUSDT", "1h", 2024): frame})
    runner = _runner(
        registry=_registry(MomentumFactor(lookback=2)),
        repository=repository,
        tmp_path=tmp_path,
        config=_config(
            target_horizons=(1, 5),
            earliest_partition_year=2024,
            stability_window=500,
        ),
    )

    result = runner.run(symbols=("BTCUSDT",), timeframes=("1h",))
    assert len(result.records) == 0
    assert len(result.skipped) >= 1
    assert all(item.stage == "analyze" for item in result.skipped)
    assert all(isinstance(item.error_code, str) for item in result.skipped)


def test_run_downloaded_universe_discovers_symbols(tmp_path: Path) -> None:
    """Universe mode discovers downloaded symbols from the storage layout."""
    frame = _ohlcv_frame(80)
    layout = StorageLayout(tmp_path)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        symbol_dir = tmp_path / "raw" / "ohlcv" / "binance" / "usdt_perpetual" / symbol / "1h"
        symbol_dir.mkdir(parents=True)
        (symbol_dir / "2024.parquet").write_text("placeholder", encoding="utf-8")

    repository = _mock_repository(
        {
            ("BTCUSDT", "1h", 2024): frame,
            ("ETHUSDT", "1h", 2024): frame,
        },
        existing={("BTCUSDT", "1h"), ("ETHUSDT", "1h")},
    )
    runner = _runner(
        registry=_registry(MomentumFactor(lookback=2)),
        repository=repository,
        layout=layout,
        config=_config(target_horizons=(1,), earliest_partition_year=2024),
    )

    result = runner.run_downloaded_universe(timeframes=("1h",))
    assert result.symbols == ("BTCUSDT", "ETHUSDT")
    assert len(result.records) == 2


def test_run_downloaded_universe_with_no_data_raises(tmp_path: Path) -> None:
    """Universe mode fails when no downloaded symbols exist."""
    runner = _runner(
        registry=_registry(MomentumFactor(lookback=2)),
        repository=_mock_repository(existing=set()),
        layout=StorageLayout(tmp_path),
        tmp_path=tmp_path,
    )
    with pytest.raises(ExperimentError, match="no downloaded OHLCV") as exc_info:
        runner.run_downloaded_universe(timeframes=("1h",))
    assert exc_info.value.error_code == "RESEARCH-RUNNER-006"


def test_leaderboard_sort_order_uses_requested_metrics() -> None:
    """Leaderboard ranking prefers higher IC, Rank IC, stability, half-life, spread."""

    def _record(
        *,
        name: str,
        ic: float,
        rank_ic: float,
        stability: float,
        half_life: int | None,
        spread: float,
    ) -> AssetExperimentRecord:
        definition = ExperimentDefinition(
            name=name,
            description="test",
            factor_columns=(name,),
            target_horizon=1,
        )
        now = datetime.now(UTC)
        result = ExperimentResult(
            definition=definition,
            target=TargetDefinition(name="t", horizon=1),
            information_coefficients=(
                InformationCoefficientResult(
                    factor_column=name,
                    target_column="forward_return",
                    method="pearson",
                    observations=10,
                    coefficient=ic,
                    p_value=0.1,
                ),
            ),
            rank_information_coefficients=(
                RankICResult(
                    factor_column=name,
                    target_column="forward_return",
                    observations=10,
                    coefficient=rank_ic,
                    p_value=0.1,
                ),
            ),
            quantile_results=(
                QuantileAnalysisResult(
                    factor_column=name,
                    target_column="forward_return",
                    quantiles=5,
                    statistics=(),
                    top_minus_bottom=spread,
                    monotonic=True,
                ),
            ),
            decay_results=(
                FactorDecayResult(
                    factor_column=name,
                    price_column="close",
                    method="pearson",
                    points=(),
                    half_life=half_life,
                ),
            ),
            stability_results=(
                FactorStabilityResult(
                    factor_column=name,
                    target_column="forward_return",
                    method="pearson",
                    window_size=10,
                    windows=(),
                    mean_ic=ic,
                    std_ic=0.1,
                    min_ic=ic,
                    max_ic=ic,
                    stability_score=stability,
                ),
            ),
            correlation_result=None,
            highly_correlated_pairs=(),
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
        )
        return AssetExperimentRecord(
            symbol="BTCUSDT",
            timeframe="1h",
            factor_name=name,
            experiment_result=result,
        )

    board = _build_leaderboard(
        (
            _record(
                name="low",
                ic=0.1,
                rank_ic=0.1,
                stability=0.5,
                half_life=2,
                spread=0.01,
            ),
            _record(
                name="high",
                ic=0.5,
                rank_ic=0.4,
                stability=0.9,
                half_life=8,
                spread=0.05,
            ),
        )
    )
    assert [entry.factor_name for entry in board] == ["high", "low"]
    assert [entry.rank for entry in board] == [1, 2]


def test_package_exports_runner_api() -> None:
    """Runner public types are exported from cqros.research."""
    import cqros.research as research_package

    for name in (
        "FactorResearchRunner",
        "FactorResearchRunnerConfig",
        "FactorResearchRunResult",
        "FactorLeaderboardEntry",
        "AssetExperimentRecord",
        "SkippedFactorEvaluation",
    ):
        assert name in research_package.__all__

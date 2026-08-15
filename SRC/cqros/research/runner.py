"""CQROS Factor Research Runner.

Purpose:
    Automatically evaluate every registered factor on downloaded market data
    across symbols, timeframes, and forward-return horizons.

Responsibilities:
    - Load historical OHLCV from ``MarketDataRepository``
    - Discover factors from ``FactorRegistry``
    - Compute factors without mutating input frames
    - Generate forward-return targets for configurable horizons
    - Run IC, Rank IC, quantile, decay, and stability analyses via
      ``ResearchExperiment``
    - Store immutable ``ExperimentResult`` objects
    - Produce a leaderboard ranked by research quality metrics
    - Skip factors that cannot be computed while continuing the run
    - Support one symbol, multiple symbols, and the downloaded universe

Dependencies:
    ``polars``, the Python standard library, ``cqros.core``,
    ``cqros.factors``, ``cqros.storage``, and composed research modules.

Public API:
    ``FactorResearchRunnerConfig``, ``SkippedFactorEvaluation``,
    ``AssetExperimentRecord``, ``FactorLeaderboardEntry``,
    ``FactorResearchRunResult``, ``FactorResearchRunner``
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, cast

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_RAW,
)
from cqros.core.exceptions import ExperimentError, ResearchError
from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.factors.exceptions import FactorError
from cqros.factors.interfaces import Factor
from cqros.factors.registry import FactorRegistry
from cqros.research.experiment import (
    ExperimentDefinition,
    ExperimentResult,
    ResearchExperiment,
)
from cqros.research.factor_correlation import FactorCorrelationAnalyzer
from cqros.research.factor_decay import FactorDecayAnalyzer
from cqros.research.factor_stability import FactorStabilityAnalyzer
from cqros.research.information_coefficient import InformationCoefficient
from cqros.research.quantile_analysis import QuantileAnalyzer
from cqros.research.rank_ic import RankInformationCoefficient
from cqros.research.target import ForwardReturnTarget, TargetDefinition
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.storage.layout import StorageLayout
from cqros.storage.repository import MarketDataRepository

__all__ = [
    "AssetExperimentRecord",
    "FactorLeaderboardEntry",
    "FactorResearchRunResult",
    "FactorResearchRunner",
    "FactorResearchRunnerConfig",
    "SkippedFactorEvaluation",
]

_logger = logging.getLogger(__name__)

_DATASET_OHLCV: Final[str] = "ohlcv"
_DEFAULT_PRICE_COLUMN: Final[str] = "close"
_DEFAULT_TARGET_HORIZONS: Final[tuple[int, ...]] = (1, 5, 10, 20)
_DEFAULT_IC_METHOD: Final[str] = "spearman"
_DEFAULT_QUANTILES: Final[int] = 5
_DEFAULT_DECAY_HORIZONS: Final[tuple[int, ...]] = (1, 2, 4, 8, 12, 24)
_DEFAULT_STABILITY_WINDOW: Final[int] = 500
_DEFAULT_CORRELATION_THRESHOLD: Final[float] = 0.90
_EARLIEST_PARTITION_YEAR: Final[int] = 2019

_ERROR_SYMBOLS_EMPTY: Final[str] = "RESEARCH-RUNNER-001"
_ERROR_TIMEFRAMES_EMPTY: Final[str] = "RESEARCH-RUNNER-002"
_ERROR_HORIZONS_EMPTY: Final[str] = "RESEARCH-RUNNER-003"
_ERROR_HORIZON_INVALID: Final[str] = "RESEARCH-RUNNER-004"
_ERROR_NO_FACTORS: Final[str] = "RESEARCH-RUNNER-005"
_ERROR_NO_OHLCV: Final[str] = "RESEARCH-RUNNER-006"


@dataclass(frozen=True, slots=True)
class FactorResearchRunnerConfig:
    """Immutable configuration for a factor research runner.

    Attributes:
        exchange: Exchange identifier used when loading OHLCV partitions.
        market: Market segment used when loading OHLCV partitions.
        price_column: Price column used for forward-return targets.
        target_horizons: Forward-return horizons evaluated for each factor.
        ic_method: IC method metadata recorded on experiment definitions.
        quantiles: Quantile count metadata recorded on experiment definitions.
        decay_horizons: Horizons forwarded to factor decay analysis.
        stability_window: Non-overlapping window size for stability analysis.
        correlation_threshold: Absolute correlation cutoff metadata.
        earliest_partition_year: First calendar year considered when loading
            multi-year OHLCV history.
    """

    exchange: Exchange = EXCHANGE_BINANCE
    market: Market = MARKET_USDT_PERPETUAL
    price_column: str = _DEFAULT_PRICE_COLUMN
    target_horizons: tuple[int, ...] = _DEFAULT_TARGET_HORIZONS
    ic_method: str = _DEFAULT_IC_METHOD
    quantiles: int = _DEFAULT_QUANTILES
    decay_horizons: tuple[int, ...] = _DEFAULT_DECAY_HORIZONS
    stability_window: int = _DEFAULT_STABILITY_WINDOW
    correlation_threshold: float = _DEFAULT_CORRELATION_THRESHOLD
    earliest_partition_year: int = _EARLIEST_PARTITION_YEAR

    def __post_init__(self) -> None:
        """Normalize sequences and validate horizon invariants."""
        horizons = tuple(self.target_horizons)
        if len(horizons) == 0:
            raise ExperimentError(
                "target_horizons must contain at least one horizon",
                error_code=_ERROR_HORIZONS_EMPTY,
                details={"parameter": "target_horizons"},
            )
        for index, horizon in enumerate(horizons):
            horizon_value = cast(object, horizon)
            if (
                not isinstance(horizon_value, int)
                or isinstance(horizon_value, bool)
                or horizon_value < 1
            ):
                raise ExperimentError(
                    "target_horizons entries must be integers greater than 0",
                    error_code=_ERROR_HORIZON_INVALID,
                    details={
                        "parameter": "target_horizons",
                        "index": index,
                        "value": horizon,
                    },
                )
        object.__setattr__(self, "target_horizons", horizons)
        object.__setattr__(self, "decay_horizons", tuple(self.decay_horizons))


@dataclass(frozen=True, slots=True)
class SkippedFactorEvaluation:
    """Immutable record of a factor skipped during a research run.

    Attributes:
        factor_name: Registered factor name that was skipped.
        symbol: Symbol being evaluated when the skip occurred.
        timeframe: Timeframe being evaluated when the skip occurred.
        stage: Pipeline stage that failed (``compute`` or ``analyze``).
        reason: Human-readable failure summary.
        error_code: Optional CQROS error code from the failure.
        target_horizon: Horizon associated with an analysis skip, if any.
    """

    factor_name: str
    symbol: Symbol
    timeframe: Timeframe
    stage: str
    reason: str
    error_code: str | None = None
    target_horizon: int | None = None


@dataclass(frozen=True, slots=True)
class AssetExperimentRecord:
    """Immutable binding of an experiment result to market context.

    Attributes:
        symbol: Symbol evaluated by the experiment.
        timeframe: Timeframe evaluated by the experiment.
        factor_name: Registered factor name evaluated by the experiment.
        experiment_result: Immutable research experiment result.
    """

    symbol: Symbol
    timeframe: Timeframe
    factor_name: str
    experiment_result: ExperimentResult


@dataclass(frozen=True, slots=True)
class FactorLeaderboardEntry:
    """Immutable leaderboard row for one factor evaluation context.

    Attributes:
        rank: One-based rank after sorting by research quality metrics.
        factor_name: Registered factor name.
        factor_column: Produced factor column evaluated.
        symbol: Symbol evaluated.
        timeframe: Timeframe evaluated.
        target_horizon: Forward-return horizon evaluated.
        mean_ic: Information coefficient for the evaluation.
        mean_rank_ic: Rank information coefficient for the evaluation.
        stability_score: Rolling stability score for the evaluation.
        decay_half_life: IC decay half-life in rows, or ``None``.
        quantile_spread: Top-minus-bottom quantile mean-return spread.
    """

    rank: int
    factor_name: str
    factor_column: str
    symbol: Symbol
    timeframe: Timeframe
    target_horizon: int
    mean_ic: float
    mean_rank_ic: float
    stability_score: float
    decay_half_life: int | None
    quantile_spread: float


@dataclass(frozen=True, slots=True)
class FactorResearchRunResult:
    """Immutable aggregate result of a factor research runner execution.

    Attributes:
        config: Runner configuration used for the run.
        symbols: Symbols evaluated.
        timeframes: Timeframes evaluated.
        records: Stored immutable experiment results with asset context.
        leaderboard: Ranked factor evaluations.
        skipped: Factors skipped due to compute or analysis failures.
        started_at: UTC timestamp when the run began.
        completed_at: UTC timestamp when the run finished.
        duration_seconds: Wall-clock duration in seconds.
    """

    config: FactorResearchRunnerConfig
    symbols: tuple[Symbol, ...]
    timeframes: tuple[Timeframe, ...]
    records: tuple[AssetExperimentRecord, ...]
    leaderboard: tuple[FactorLeaderboardEntry, ...]
    skipped: tuple[SkippedFactorEvaluation, ...]
    started_at: datetime
    completed_at: datetime
    duration_seconds: float


class FactorResearchRunner:
    """Dependency-injected orchestrator for multi-asset factor research.

    The runner loads OHLCV history, computes every registered factor, evaluates
    each successful factor across configured forward-return horizons, stores
    immutable ``ExperimentResult`` objects, and builds a ranked leaderboard.
    Statistical work is delegated to injected research analyzers. Input frames
    are never mutated.
    """

    __slots__ = (
        "_repository",
        "_registry",
        "_layout",
        "_information_coefficient",
        "_rank_information_coefficient",
        "_quantile_analyzer",
        "_factor_decay_analyzer",
        "_factor_stability_analyzer",
        "_factor_correlation_analyzer",
        "_config",
        "_logger",
    )

    def __init__(
        self,
        repository: MarketDataRepository,
        registry: FactorRegistry,
        layout: StorageLayout,
        information_coefficient: InformationCoefficient,
        rank_information_coefficient: RankInformationCoefficient,
        quantile_analyzer: QuantileAnalyzer,
        factor_decay_analyzer: FactorDecayAnalyzer,
        factor_stability_analyzer: FactorStabilityAnalyzer,
        factor_correlation_analyzer: FactorCorrelationAnalyzer,
        *,
        config: FactorResearchRunnerConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize with injected storage, registry, and research analyzers.

        Args:
            repository: Market-data repository used to load OHLCV partitions.
            registry: Factor registry providing factors to evaluate.
            layout: Storage layout used to discover the downloaded universe.
            information_coefficient: IC calculator.
            rank_information_coefficient: Rank IC calculator.
            quantile_analyzer: Quantile analyzer.
            factor_decay_analyzer: Factor decay analyzer.
            factor_stability_analyzer: Factor stability analyzer.
            factor_correlation_analyzer: Cross-factor correlation analyzer.
            config: Optional runner configuration. Defaults to institutional
                defaults including horizons ``(1, 5, 10, 20)``.
            logger: Optional logger instance.
        """
        self._repository = repository
        self._registry = registry
        self._layout = layout
        self._information_coefficient = information_coefficient
        self._rank_information_coefficient = rank_information_coefficient
        self._quantile_analyzer = quantile_analyzer
        self._factor_decay_analyzer = factor_decay_analyzer
        self._factor_stability_analyzer = factor_stability_analyzer
        self._factor_correlation_analyzer = factor_correlation_analyzer
        self._config = config if config is not None else FactorResearchRunnerConfig()
        self._logger = logger if logger is not None else _logger

    def run(
        self,
        *,
        symbols: Sequence[Symbol],
        timeframes: Sequence[Timeframe],
    ) -> FactorResearchRunResult:
        """Evaluate registered factors on the supplied symbols and timeframes.

        Args:
            symbols: One or more symbols to evaluate.
            timeframes: One or more timeframes to evaluate.

        Returns:
            An immutable ``FactorResearchRunResult`` containing stored
            experiment results, skipped-factor records, and a leaderboard.

        Raises:
            ExperimentError: If symbols, timeframes, or registered factors are
                empty.
        """
        return self._run(
            symbols=_require_non_empty_str_sequence(
                symbols,
                parameter="symbols",
                error_code=_ERROR_SYMBOLS_EMPTY,
            ),
            timeframes=_require_non_empty_str_sequence(
                timeframes,
                parameter="timeframes",
                error_code=_ERROR_TIMEFRAMES_EMPTY,
            ),
        )

    def run_downloaded_universe(
        self,
        *,
        timeframes: Sequence[Timeframe],
    ) -> FactorResearchRunResult:
        """Evaluate registered factors on every downloaded symbol.

        Symbols are discovered from the on-disk OHLCV layout for each
        requested timeframe and unioned across timeframes. A symbol is kept
        when at least one requested timeframe has downloaded OHLCV data.

        Args:
            timeframes: One or more timeframes to evaluate.

        Returns:
            An immutable ``FactorResearchRunResult``.

        Raises:
            ExperimentError: If timeframes are empty, no factors are
                registered, or no downloaded OHLCV symbols are found.
        """
        validated_timeframes = _require_non_empty_str_sequence(
            timeframes,
            parameter="timeframes",
            error_code=_ERROR_TIMEFRAMES_EMPTY,
        )
        discovered: set[Symbol] = set()
        for timeframe in validated_timeframes:
            discovered.update(self._discover_downloaded_symbols(timeframe))
        symbols = tuple(sorted(discovered))
        if len(symbols) == 0:
            raise ExperimentError(
                "no downloaded OHLCV symbols found for the requested timeframes",
                error_code=_ERROR_NO_OHLCV,
                details={
                    "timeframes": validated_timeframes,
                    "exchange": self._config.exchange,
                    "market": self._config.market,
                },
            )
        return self._run(symbols=symbols, timeframes=validated_timeframes)

    def _run(
        self,
        *,
        symbols: tuple[Symbol, ...],
        timeframes: tuple[Timeframe, ...],
    ) -> FactorResearchRunResult:
        """Execute the full multi-asset factor research workflow."""
        factors = self._registry.list()
        if len(factors) == 0:
            raise ExperimentError(
                "factor registry contains no registered factors",
                error_code=_ERROR_NO_FACTORS,
                details={"registry_size": 0},
            )

        started_at = datetime.now(UTC)
        records: list[AssetExperimentRecord] = []
        skipped: list[SkippedFactorEvaluation] = []

        for symbol in symbols:
            for timeframe in timeframes:
                frame = self._load_ohlcv_history(symbol=symbol, timeframe=timeframe)
                if frame is None:
                    self._logger.warning(
                        "Skipping symbol/timeframe with no OHLCV history",
                        extra={"symbol": symbol, "timeframe": timeframe},
                    )
                    continue

                computed_frame, computed_factors, compute_skips = self._compute_factors(
                    frame,
                    factors,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                skipped.extend(compute_skips)
                if len(computed_factors) == 0:
                    continue

                for horizon in self._config.target_horizons:
                    for factor in computed_factors:
                        record, analyze_skip = self._evaluate_factor(
                            computed_frame,
                            factor,
                            symbol=symbol,
                            timeframe=timeframe,
                            horizon=horizon,
                        )
                        if analyze_skip is not None:
                            skipped.append(analyze_skip)
                        if record is not None:
                            records.append(record)

        completed_at = datetime.now(UTC)
        leaderboard = _build_leaderboard(tuple(records))
        return FactorResearchRunResult(
            config=self._config,
            symbols=symbols,
            timeframes=timeframes,
            records=tuple(records),
            leaderboard=leaderboard,
            skipped=tuple(skipped),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
        )

    def _compute_factors(
        self,
        frame: pl.DataFrame,
        factors: Sequence[Factor],
        *,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> tuple[pl.DataFrame, tuple[Factor, ...], list[SkippedFactorEvaluation]]:
        """Compute registered factors, skipping failures without mutation."""
        working = frame
        computed: list[Factor] = []
        skipped: list[SkippedFactorEvaluation] = []

        for factor in factors:
            try:
                working = factor.compute(working)
            except FactorError as exc:
                skipped.append(
                    SkippedFactorEvaluation(
                        factor_name=factor.name,
                        symbol=symbol,
                        timeframe=timeframe,
                        stage="compute",
                        reason=str(exc),
                        error_code=exc.error_code,
                    )
                )
                self._logger.warning(
                    "Skipping factor that failed compute",
                    extra={
                        "factor": factor.name,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "error_code": exc.error_code,
                    },
                )
                continue
            computed.append(factor)

        return working, tuple(computed), skipped

    def _evaluate_factor(
        self,
        frame: pl.DataFrame,
        factor: Factor,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        horizon: int,
    ) -> tuple[AssetExperimentRecord | None, SkippedFactorEvaluation | None]:
        """Run research analyses for one factor and horizon."""
        factor_columns = tuple(factor.produced_columns)
        definition = ExperimentDefinition(
            name=f"{factor.name}__{symbol}__{timeframe}__h{horizon}",
            description=(
                f"Factor research evaluation for {factor.name} on {symbol} "
                f"{timeframe} with horizon {horizon}."
            ),
            factor_columns=factor_columns,
            price_column=self._config.price_column,
            target_horizon=horizon,
            ic_method=self._config.ic_method,
            quantiles=self._config.quantiles,
            decay_horizons=self._config.decay_horizons,
            stability_window=self._config.stability_window,
            correlation_threshold=self._config.correlation_threshold,
        )
        experiment = ResearchExperiment(
            forward_return_target=ForwardReturnTarget(
                TargetDefinition(
                    name=f"forward_return_h{horizon}",
                    horizon=horizon,
                    price_column=self._config.price_column,
                    output_column="forward_return",
                )
            ),
            information_coefficient=self._information_coefficient,
            rank_information_coefficient=self._rank_information_coefficient,
            quantile_analyzer=self._quantile_analyzer,
            factor_decay_analyzer=self._factor_decay_analyzer,
            factor_stability_analyzer=self._factor_stability_analyzer,
            factor_correlation_analyzer=self._factor_correlation_analyzer,
        )
        try:
            result = experiment.run(frame, definition)
        except ResearchError as exc:
            skip = SkippedFactorEvaluation(
                factor_name=factor.name,
                symbol=symbol,
                timeframe=timeframe,
                stage="analyze",
                reason=str(exc),
                error_code=exc.error_code,
                target_horizon=horizon,
            )
            self._logger.warning(
                "Skipping factor that failed analysis",
                extra={
                    "factor": factor.name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "horizon": horizon,
                    "error_code": exc.error_code,
                },
            )
            return None, skip

        return (
            AssetExperimentRecord(
                symbol=symbol,
                timeframe=timeframe,
                factor_name=factor.name,
                experiment_result=result,
            ),
            None,
        )

    def _load_ohlcv_history(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> pl.DataFrame | None:
        """Load and concatenate available OHLCV year partitions."""
        frames: list[pl.DataFrame] = []
        current_year = datetime.now(UTC).year
        for year in range(self._config.earliest_partition_year, current_year + 1):
            try:
                frame = self._repository.load_ohlcv(
                    exchange=self._config.exchange,
                    market=self._config.market,
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                )
            except DatasetNotFoundError:
                continue
            if frame.height == 0:
                continue
            frames.append(frame)

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        return pl.concat(frames, how="vertical")

    def _discover_downloaded_symbols(self, timeframe: Timeframe) -> tuple[Symbol, ...]:
        """Discover symbols with downloaded OHLCV for ``timeframe``."""
        base = (
            self._layout.root
            / STORAGE_DIR_RAW
            / _DATASET_OHLCV
            / self._config.exchange
            / self._config.market
        )
        if not base.is_dir():
            return ()

        symbols: list[Symbol] = []
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            symbol = entry.name
            if self._repository.has_ohlcv(symbol, timeframe):
                symbols.append(symbol)
        return tuple(symbols)


def _require_non_empty_str_sequence(
    value: object,
    *,
    parameter: str,
    error_code: str,
) -> tuple[str, ...]:
    """Validate and freeze a non-empty string sequence."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExperimentError(
            f"{parameter} must be a non-empty sequence of strings",
            error_code=error_code,
            details={"parameter": parameter, "value_type": type(value).__name__},
        )
    sequence = cast(Sequence[object], value)
    if len(sequence) == 0:
        raise ExperimentError(
            f"{parameter} must contain at least one entry",
            error_code=error_code,
            details={"parameter": parameter},
        )
    frozen: list[str] = []
    for index, entry in enumerate(sequence):
        if not isinstance(entry, str) or entry.strip() == "":
            raise ExperimentError(
                f"{parameter} entries must be non-empty strings",
                error_code=error_code,
                details={"parameter": parameter, "index": index, "value": entry},
            )
        frozen.append(entry)
    return tuple(frozen)


def _build_leaderboard(
    records: tuple[AssetExperimentRecord, ...],
) -> tuple[FactorLeaderboardEntry, ...]:
    """Build a leaderboard ranked by IC, Rank IC, stability, decay, and spread."""
    entries: list[FactorLeaderboardEntry] = []
    for record in records:
        result = record.experiment_result
        if len(result.information_coefficients) == 0:
            continue
        for index, ic_result in enumerate(result.information_coefficients):
            rank_ic = result.rank_information_coefficients[index]
            stability = result.stability_results[index]
            decay = result.decay_results[index]
            quantile = result.quantile_results[index]
            entries.append(
                FactorLeaderboardEntry(
                    rank=0,
                    factor_name=record.factor_name,
                    factor_column=ic_result.factor_column,
                    symbol=record.symbol,
                    timeframe=record.timeframe,
                    target_horizon=result.definition.target_horizon,
                    mean_ic=ic_result.coefficient,
                    mean_rank_ic=rank_ic.coefficient,
                    stability_score=stability.stability_score,
                    decay_half_life=decay.half_life,
                    quantile_spread=quantile.top_minus_bottom,
                )
            )

    entries.sort(
        key=lambda entry: (
            entry.mean_ic,
            entry.mean_rank_ic,
            entry.stability_score,
            entry.decay_half_life if entry.decay_half_life is not None else -1,
            entry.quantile_spread,
        ),
        reverse=True,
    )
    return tuple(
        FactorLeaderboardEntry(
            rank=rank,
            factor_name=entry.factor_name,
            factor_column=entry.factor_column,
            symbol=entry.symbol,
            timeframe=entry.timeframe,
            target_horizon=entry.target_horizon,
            mean_ic=entry.mean_ic,
            mean_rank_ic=entry.mean_rank_ic,
            stability_score=entry.stability_score,
            decay_half_life=entry.decay_half_life,
            quantile_spread=entry.quantile_spread,
        )
        for rank, entry in enumerate(entries, start=1)
    )

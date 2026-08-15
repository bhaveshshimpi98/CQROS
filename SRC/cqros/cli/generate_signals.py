"""CQROS signal-generation CLI.

Purpose:
    Provide an argparse-based production entry point that discovers prediction
    partitions, resolves a model artifact for framework identity, and executes
    ``SignalPipeline`` across the universe with bounded symbol concurrency.

Responsibilities:
    - Parse CLI arguments for signal dataset generation
    - Discover available prediction partitions
    - Resolve ``--model`` / ``--version`` through ``ModelArtifactRepository``
    - Resolve ``--policy`` through ``SignalPolicyRegistry``
    - Load prediction frames into ``SignalPipeline``
    - Execute ``SignalPipeline`` and persist via ``SignalRepository``
    - Honor ``--overwrite``, worker concurrency, and debug logging
    - Print deterministic progress and a final summary

Dependencies:
    ``argparse``, ``asyncio``, ``polars``, ``cqros.config``, ``cqros.core``,
    ``cqros.ml``, ``cqros.signals``, and ``cqros.storage``.

Public API:
    ``build_parser``, ``build_options``, ``build_default_policy``,
    ``build_regression_policy``, ``build_adaptive_regression_policy``,
    ``build_policy_registry``, ``build_signal_pipeline``, ``discover_work``,
    ``format_summary``, ``resolve_model_artifact``, ``run_generation``, and
    ``main``.

Notes:
    This module is a thin composition root. It does not implement signal
    semantics, threshold rules, schema validation, or repository filesystem
    walks beyond calling repository discovery and load APIs. Signal creation
    is delegated exclusively to ``SignalPipeline``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cqros.config.models import ResearchConfig
from cqros.core.constants import (
    DEFAULT_STORAGE_ROOT,
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_SIGNALS,
    SUPPORTED_TIMEFRAMES,
)
from cqros.core.exceptions import CQROSError, ValidationError
from cqros.core.types import Symbol, Timeframe
from cqros.ml.models import (
    Model,
    ModelArtifactRef,
    ModelArtifactRepository,
    ModelPersistence,
    ModelValidationError,
)
from cqros.signals import (
    AdaptiveRegressionSignalPolicy,
    ClassificationSignalPolicy,
    RegressionSignalPolicy,
    RegressionThresholds,
    RepositoryThresholdProvider,
    SignalPipeline,
    SignalPolicy,
    SignalPolicyRegistry,
)
from cqros.storage import (
    ParquetStore,
    PredictionPartitionRef,
    PredictionRepository,
    SignalPartitionRef,
    SignalRepository,
    StorageLayout,
    ThresholdRepository,
)

__all__ = [
    "DiscoveredWorkItem",
    "SignalGenerationOptions",
    "SignalGenerationSummary",
    "SignalTaskResult",
    "build_adaptive_regression_policy",
    "build_default_policy",
    "build_options",
    "build_parser",
    "build_policy_registry",
    "build_regression_policy",
    "build_signal_pipeline",
    "discover_work",
    "format_summary",
    "main",
    "resolve_model_artifact",
    "run_generation",
]

_logger = logging.getLogger(__name__)

_EXIT_SUCCESS: Final[int] = 0
_EXIT_FAILURE: Final[int] = 1

_DEFAULT_WORKER_COUNT: Final[int] = ResearchConfig().worker_count
_POLICY_CLASSIFICATION: Final[str] = "classification"
_POLICY_REGRESSION: Final[str] = "regression"
_POLICY_ADAPTIVE_REGRESSION: Final[str] = "adaptive_regression"
_DEFAULT_BUY_THRESHOLD: Final[float] = 0.01
_DEFAULT_SELL_THRESHOLD: Final[float] = -0.01
_DEFAULT_THRESHOLD_PROFILE: Final[str] = "Balanced"

_EXCHANGE: Final[str] = EXCHANGE_BINANCE
_MARKET: Final[str] = MARKET_USDT_PERPETUAL

_ERROR_WORKERS: Final[str] = "CLI-GENERATE-SIGNALS-001"
_ERROR_TIMEFRAME: Final[str] = "CLI-GENERATE-SIGNALS-002"
_ERROR_YEAR: Final[str] = "CLI-GENERATE-SIGNALS-003"
_ERROR_MODEL: Final[str] = "CLI-GENERATE-SIGNALS-004"
_ERROR_VERSION: Final[str] = "CLI-GENERATE-SIGNALS-005"
_ERROR_MODEL_NOT_FOUND: Final[str] = "CLI-GENERATE-SIGNALS-006"
_ERROR_MODEL_AMBIGUOUS: Final[str] = "CLI-GENERATE-SIGNALS-007"
_ERROR_PERSISTENCE: Final[str] = "CLI-GENERATE-SIGNALS-008"
_ERROR_POLICY: Final[str] = "CLI-GENERATE-SIGNALS-009"


@dataclass(frozen=True, slots=True)
class SignalGenerationOptions:
    """Immutable CLI options for signal dataset generation.

    Attributes:
        storage_root: Storage root containing ``predictions``, ``models``, and
            ``signals``.
        policy: Registry key of the signal policy to execute.
        model: Stable model identifier.
        version: Model version identifier.
        symbols: Optional symbol allowlist. ``None`` discovers all.
        timeframes: Optional timeframe allowlist. ``None`` discovers all.
        years: Optional year allowlist. ``None`` discovers all.
        overwrite: When ``True``, regenerate existing signal partitions.
        workers: Maximum concurrent symbols.
        verbose: When ``True``, enable INFO logging.
        debug: When ``True``, enable DEBUG logging and log full failure
            tracebacks.
    """

    storage_root: Path
    policy: str
    model: str
    version: str
    symbols: tuple[Symbol, ...] | None
    timeframes: tuple[Timeframe, ...] | None
    years: tuple[int, ...] | None
    overwrite: bool
    workers: int
    verbose: bool
    debug: bool


@dataclass(frozen=True, slots=True)
class DiscoveredWorkItem:
    """One discovered prediction partition group ready for signal generation.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Available bar interval.
        years: Calendar years with existing prediction parquet partitions.
    """

    symbol: Symbol
    timeframe: Timeframe
    years: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SignalTaskResult:
    """Immutable result for one symbol/timeframe/year generation task.

    Attributes:
        symbol: Tradeable symbol.
        timeframe: Bar interval.
        year: Calendar year of the partition.
        status: ``succeeded``, ``failed``, or ``skipped``.
        rows_generated: Output row count when succeeded.
        error_type: Exception type name when failed.
        error_message: Exception message when failed.
        error_code: Optional CQROS error code when failed.
    """

    symbol: Symbol
    timeframe: Timeframe
    year: int
    status: str
    rows_generated: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class SignalGenerationSummary:
    """Immutable aggregate summary for a signal generation run.

    Attributes:
        policy: Signal policy registry key used for generation.
        model: Stable model identifier used for generation.
        version: Model version used for generation.
        symbols_discovered: Unique symbols discovered from prediction storage.
        symbols_processed: Unique symbols for which generation was attempted.
        timeframes_processed: Unique timeframes attempted.
        successful_tasks: Count of succeeded partition tasks.
        failed_tasks: Count of failed partition tasks.
        skipped_tasks: Count of skipped existing partitions.
        rows_generated: Sum of output rows across successes.
        duration_seconds: Wall-clock generation duration.
        output_directory: Signals-tier output directory.
        failed_task_labels: Deterministic failed-task labels for reporting.
    """

    policy: str
    model: str
    version: str
    symbols_discovered: int
    symbols_processed: int
    timeframes_processed: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    rows_generated: int
    duration_seconds: float
    output_directory: Path
    failed_task_labels: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Create the signal-generation argument parser.

    Returns:
        Configured ``ArgumentParser`` for signal generation flags.
    """
    parser = argparse.ArgumentParser(
        prog="cqros-generate-signals",
        description=(
            "Generate CQROS signal datasets from discovered prediction "
            "partitions and an injected signal policy."
        ),
    )
    parser.add_argument(
        "--policy",
        dest="policy",
        required=True,
        metavar="NAME",
        help=(
            "Signal policy registry key " "(classification, regression, or adaptive_regression)."
        ),
    )
    parser.add_argument(
        "--model",
        dest="model",
        required=True,
        metavar="NAME",
        help="Stable model identifier to resolve from ModelArtifactRepository.",
    )
    parser.add_argument(
        "--version",
        dest="version",
        required=True,
        metavar="VERSION",
        help="Model version identifier to resolve from ModelArtifactRepository.",
    )
    parser.add_argument(
        "--symbols",
        dest="symbols",
        nargs="*",
        default=None,
        metavar="SYMBOL",
        help="Optional symbol allowlist (0..N values). Omit to discover all.",
    )
    parser.add_argument(
        "--timeframes",
        dest="timeframes",
        nargs="*",
        default=None,
        metavar="TIMEFRAME",
        help="Optional timeframe allowlist (0..N values). Omit to discover all.",
    )
    parser.add_argument(
        "--years",
        dest="years",
        nargs="*",
        default=None,
        metavar="YEAR",
        help="Optional calendar-year allowlist (0..N values). Omit to discover all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate signal partitions that already exist.",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=_DEFAULT_WORKER_COUNT,
        metavar="INT",
        help=("Maximum concurrent symbols " f"(default: {_DEFAULT_WORKER_COUNT})."),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Enable DEBUG logging and log complete failure tracebacks " "with logger.exception."),
    )
    return parser


def build_options(args: argparse.Namespace) -> SignalGenerationOptions:
    """Map parsed CLI arguments onto ``SignalGenerationOptions``.

    Args:
        args: Namespace produced by ``build_parser().parse_args(...)``.

    Returns:
        Immutable generation options.

    Raises:
        ValidationError: If ``workers`` is not positive or filters are invalid.
    """
    workers = int(args.workers)
    if workers <= 0:
        raise ValidationError(
            "workers must be greater than 0",
            error_code=_ERROR_WORKERS,
            details={"parameter": "workers", "value": workers},
        )

    policy = str(args.policy).strip()
    if policy == "":
        raise ValidationError(
            "policy must be a non-empty string",
            error_code=_ERROR_POLICY,
            details={"parameter": "policy", "value": args.policy},
        )

    model = str(args.model).strip()
    if model == "":
        raise ValidationError(
            "model must be a non-empty string",
            error_code=_ERROR_MODEL,
            details={"parameter": "model", "value": args.model},
        )
    version = str(args.version).strip()
    if version == "":
        raise ValidationError(
            "version must be a non-empty string",
            error_code=_ERROR_VERSION,
            details={"parameter": "version", "value": args.version},
        )

    return SignalGenerationOptions(
        storage_root=Path(DEFAULT_STORAGE_ROOT),
        policy=policy,
        model=model,
        version=version,
        symbols=_normalize_symbols(args.symbols),
        timeframes=_normalize_timeframes(args.timeframes),
        years=_normalize_years(args.years),
        overwrite=bool(args.overwrite),
        workers=workers,
        verbose=bool(args.verbose),
        debug=bool(args.debug),
    )


def resolve_model_artifact(
    model_repository: ModelArtifactRepository,
    *,
    model_name: str,
    version: str,
) -> ModelArtifactRef:
    """Resolve a unique trained model artifact for ``model_name`` / ``version``.

    Args:
        model_repository: Model artifact repository providing discovery APIs.
        model_name: Stable model identifier.
        version: Model version identifier.

    Returns:
        The unique matching ``ModelArtifactRef``.

    Raises:
        ValidationError: If no artifact matches or multiple frameworks match.
    """
    artifacts = model_repository.discover_artifacts(model_names=(model_name,))
    matches = tuple(item for item in artifacts if item.version == version)
    if len(matches) == 0:
        raise ValidationError(
            f"model artifact not found: {model_name}@{version}",
            error_code=_ERROR_MODEL_NOT_FOUND,
            details={"model": model_name, "version": version},
        )
    if len(matches) > 1:
        frameworks = tuple(item.framework for item in matches)
        raise ValidationError(
            f"ambiguous model artifact: {model_name}@{version}",
            error_code=_ERROR_MODEL_AMBIGUOUS,
            details={
                "model": model_name,
                "version": version,
                "frameworks": frameworks,
            },
        )
    return matches[0]


def build_default_policy() -> SignalPolicy:
    """Compose the default production signal policy for the CLI.

    Returns:
        ``ClassificationSignalPolicy`` with production class-label defaults.
    """
    return ClassificationSignalPolicy()


def build_regression_policy() -> SignalPolicy:
    """Compose the production regression signal policy for the CLI.

    Returns:
        ``RegressionSignalPolicy`` with production buy/sell thresholds.
    """
    return RegressionSignalPolicy(
        buy_threshold=_DEFAULT_BUY_THRESHOLD,
        sell_threshold=_DEFAULT_SELL_THRESHOLD,
    )


def build_adaptive_regression_policy(
    *,
    threshold_repository: ThresholdRepository | None = None,
    storage_root: Path | None = None,
    profile: str = _DEFAULT_THRESHOLD_PROFILE,
) -> SignalPolicy:
    """Compose the production adaptive regression signal policy for the CLI.

    Args:
        threshold_repository: Optional threshold repository. When ``None``, one
            is constructed from ``storage_root`` or ``DEFAULT_STORAGE_ROOT``.
        storage_root: Storage root used when ``threshold_repository`` is
            ``None``.
        profile: Threshold profile selected from stored partitions.

    Returns:
        ``AdaptiveRegressionSignalPolicy`` backed by
        ``RepositoryThresholdProvider`` with global-default fallback.
    """
    repository = threshold_repository
    if repository is None:
        root = storage_root if storage_root is not None else Path(DEFAULT_STORAGE_ROOT)
        repository = ThresholdRepository(StorageLayout(root), ParquetStore())
    provider = RepositoryThresholdProvider(
        repository,
        global_thresholds=RegressionThresholds(
            buy_threshold=_DEFAULT_BUY_THRESHOLD,
            sell_threshold=_DEFAULT_SELL_THRESHOLD,
        ),
        profile=profile,
    )
    return AdaptiveRegressionSignalPolicy(provider)


def build_policy_registry(
    *,
    policies: Mapping[str, SignalPolicy] | None = None,
    storage_root: Path | None = None,
    threshold_repository: ThresholdRepository | None = None,
) -> SignalPolicyRegistry:
    """Compose a registry with default or injected signal-policy implementations.

    Args:
        policies: Optional mapping of registry names to signal-policy instances.
            When ``None``, registers ``ClassificationSignalPolicy`` under
            ``classification``, ``RegressionSignalPolicy`` under ``regression``,
            and ``AdaptiveRegressionSignalPolicy`` under ``adaptive_regression``.
        storage_root: Optional storage root forwarded to the adaptive policy
            composition root when building default policies.
        threshold_repository: Optional threshold repository forwarded to the
            adaptive policy composition root when building default policies.

    Returns:
        Fully populated ``SignalPolicyRegistry``.
    """
    registry = SignalPolicyRegistry()
    if policies is None:
        registry.register(_POLICY_CLASSIFICATION, build_default_policy())
        registry.register(_POLICY_REGRESSION, build_regression_policy())
        registry.register(
            _POLICY_ADAPTIVE_REGRESSION,
            build_adaptive_regression_policy(
                storage_root=storage_root,
                threshold_repository=threshold_repository,
            ),
        )
    else:
        registry.register_many(policies)
    return registry


def build_signal_pipeline(
    options: SignalGenerationOptions,
    *,
    policy_registry: SignalPolicyRegistry | None = None,
    signal_repository: SignalRepository | None = None,
    logger: logging.Logger | None = None,
) -> SignalPipeline:
    """Compose ``SignalPipeline`` from injected policy registry and storage deps.

    Args:
        options: Immutable generation options providing the storage root.
        policy_registry: Optional policy registry. When ``None``, a default
            registry containing classification, regression, and adaptive
            regression policies is built.
        signal_repository: Optional signal repository. When ``None``, one is
            constructed from ``options.storage_root``.
        logger: Optional logger forwarded to the pipeline.

    Returns:
        Fully wired ``SignalPipeline``.
    """
    if signal_repository is None:
        layout = StorageLayout(options.storage_root)
        signal_repository = SignalRepository(layout, ParquetStore())
    if policy_registry is None:
        policy_registry = build_policy_registry()
    return SignalPipeline(
        signal_repository,
        policy_registry,
        logger=logger if logger is not None else _logger,
    )


def discover_work(
    prediction_repository: PredictionRepository,
    options: SignalGenerationOptions,
    *,
    framework: str,
) -> tuple[DiscoveredWorkItem, ...]:
    """Discover signal-ready prediction partitions matching CLI filters.

    Only prediction partitions that exist are scheduled. Missing prediction
    partitions are never invented. Partial signal datasets are never
    generated.

    Args:
        prediction_repository: Prediction repository providing discovery APIs.
        options: CLI filters for model, symbol, timeframe, and year.
        framework: Resolved model framework identifier.

    Returns:
        Deterministically ordered discovered work items.
    """
    partitions = prediction_repository.discover_partitions(
        framework=framework,
        model_names=(options.model,),
        versions=(options.version,),
        symbols=options.symbols,
        timeframes=options.timeframes,
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    return _group_partitions(partitions, year_filter=options.years)


def format_summary(summary: SignalGenerationSummary) -> str:
    """Render a deterministic signal-generation summary report.

    Args:
        summary: Aggregate generation summary.

    Returns:
        Multi-line summary text suitable for stdout.
    """
    lines = [
        "=====================================",
        "CQROS Signal Generation Summary",
        "=====================================",
        "",
        f"Policy: {summary.policy}",
        f"Model: {summary.model}",
        f"Version: {summary.version}",
        f"Symbols discovered: {summary.symbols_discovered}",
        f"Symbols processed: {summary.symbols_processed}",
        f"Timeframes processed: {summary.timeframes_processed}",
        f"Successful tasks: {summary.successful_tasks}",
        f"Failed tasks: {summary.failed_tasks}",
        f"Skipped tasks: {summary.skipped_tasks}",
        f"Rows generated: {summary.rows_generated}",
        f"Generation duration: {_format_duration(summary.duration_seconds)}",
        f"Output directory: {_format_output_directory(summary.output_directory)}",
    ]
    if summary.failed_task_labels:
        lines.extend(["", "Failed Tasks", ""])
        lines.extend(f"- {label}" for label in summary.failed_task_labels)
    return "\n".join(lines) + "\n"


async def main(argv: Sequence[str] | None = None) -> int:
    """Run the signal-generation CLI.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        ``0`` on completion; ``1`` when a fatal CLI error occurs or any task
        failed.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = build_options(args)
        _configure_logging(verbose=options.verbose, debug=options.debug)
        layout = StorageLayout(options.storage_root)
        datastore = ParquetStore()
        prediction_repository = PredictionRepository(layout, datastore)
        signal_repository = SignalRepository(layout, datastore)
        threshold_repository = ThresholdRepository(layout, datastore)
        model_repository = ModelArtifactRepository(layout, _DiscoveryPersistence())
        model_artifact = resolve_model_artifact(
            model_repository,
            model_name=options.model,
            version=options.version,
        )
        policy_registry = build_policy_registry(
            storage_root=options.storage_root,
            threshold_repository=threshold_repository,
        )
        # Fail fast when --policy is not registered before scheduling work.
        policy_registry.get(options.policy)
        pipeline = build_signal_pipeline(
            options,
            policy_registry=policy_registry,
            signal_repository=signal_repository,
        )
        work = discover_work(
            prediction_repository,
            options,
            framework=model_artifact.framework,
        )
        summary = await run_generation(
            pipeline=pipeline,
            prediction_repository=prediction_repository,
            signal_repository=signal_repository,
            options=options,
            work=work,
            framework=model_artifact.framework,
        )
    except CQROSError as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return _EXIT_FAILURE

    print(format_summary(summary), end="")
    return _EXIT_SUCCESS if summary.failed_tasks == 0 else _EXIT_FAILURE


async def run_generation(
    *,
    pipeline: SignalPipeline,
    prediction_repository: PredictionRepository,
    signal_repository: SignalRepository,
    options: SignalGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    framework: str,
) -> SignalGenerationSummary:
    """Execute discovered work through a bounded symbol worker pool.

    Args:
        pipeline: Injected signal pipeline.
        prediction_repository: Prediction partition repository.
        signal_repository: Signal partition repository.
        options: Immutable generation options.
        work: Discovered work items.
        framework: Resolved model framework identifier.

    Returns:
        Aggregate immutable summary.
    """
    started = time.perf_counter()
    output_directory = options.storage_root / STORAGE_DIR_SIGNALS

    if len(work) == 0:
        return SignalGenerationSummary(
            policy=options.policy,
            model=options.model,
            version=options.version,
            symbols_discovered=0,
            symbols_processed=0,
            timeframes_processed=0,
            successful_tasks=0,
            failed_tasks=0,
            skipped_tasks=0,
            rows_generated=0,
            duration_seconds=time.perf_counter() - started,
            output_directory=output_directory,
            failed_task_labels=(),
        )

    work_by_symbol = _group_work_by_symbol(work)
    results = await _run_worker_pool(
        pipeline=pipeline,
        prediction_repository=prediction_repository,
        signal_repository=signal_repository,
        work_by_symbol=work_by_symbol,
        worker_count=options.workers,
        overwrite=options.overwrite,
        debug=options.debug,
        framework=framework,
        policy_name=options.policy,
        model_name=options.model,
        model_version=options.version,
    )
    return _build_summary(
        options=options,
        work=work,
        results=results,
        duration_seconds=time.perf_counter() - started,
        output_directory=output_directory,
    )


def _configure_logging(*, verbose: bool, debug: bool) -> None:
    """Configure process logging for the CLI entry point."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("cqros").setLevel(level)


def _normalize_symbols(values: Sequence[str] | None) -> tuple[Symbol, ...] | None:
    """Validate and freeze optional symbol filters."""
    if values is None:
        return None
    normalized: list[Symbol] = []
    for symbol in values:
        stripped = symbol.strip()
        if stripped == "":
            continue
        if stripped not in normalized:
            normalized.append(stripped)
    return tuple(normalized) if normalized else None


def _normalize_timeframes(
    values: Sequence[str] | None,
) -> tuple[Timeframe, ...] | None:
    """Validate and freeze optional timeframe filters."""
    if values is None:
        return None
    normalized: list[Timeframe] = []
    for timeframe in values:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValidationError(
                f"unsupported timeframe: {timeframe}",
                error_code=_ERROR_TIMEFRAME,
                details={"parameter": "timeframes", "value": timeframe},
            )
        if timeframe not in normalized:
            normalized.append(timeframe)
    return tuple(normalized) if normalized else None


def _normalize_years(values: Sequence[str] | None) -> tuple[int, ...] | None:
    """Validate and freeze optional year filters."""
    if values is None:
        return None
    normalized: list[int] = []
    for raw in values:
        try:
            year = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"invalid year: {raw}",
                error_code=_ERROR_YEAR,
                details={"parameter": "years", "value": raw},
            ) from exc
        if year < 1:
            raise ValidationError(
                f"invalid year: {raw}",
                error_code=_ERROR_YEAR,
                details={"parameter": "years", "value": raw},
            )
        if year not in normalized:
            normalized.append(year)
    return tuple(sorted(normalized)) if normalized else None


def _group_partitions(
    partitions: Sequence[PredictionPartitionRef],
    *,
    year_filter: tuple[int, ...] | None,
) -> tuple[DiscoveredWorkItem, ...]:
    """Group prediction year partitions into symbol/timeframe work items."""
    year_allowlist = set(year_filter) if year_filter is not None else None
    grouped: dict[tuple[str, str], list[int]] = {}
    for partition in partitions:
        if year_allowlist is not None and partition.year not in year_allowlist:
            continue
        key = (partition.symbol, partition.timeframe)
        grouped.setdefault(key, []).append(partition.year)

    items: list[DiscoveredWorkItem] = []
    for (symbol, timeframe), years in grouped.items():
        items.append(
            DiscoveredWorkItem(
                symbol=symbol,
                timeframe=timeframe,
                years=tuple(sorted(years)),
            )
        )
    return tuple(
        sorted(
            items,
            key=lambda item: (item.symbol, item.timeframe),
        )
    )


def _group_work_by_symbol(
    work: Sequence[DiscoveredWorkItem],
) -> dict[Symbol, tuple[DiscoveredWorkItem, ...]]:
    """Group discovered work by symbol while preserving discovery order."""
    grouped: dict[Symbol, list[DiscoveredWorkItem]] = {}
    for item in work:
        grouped.setdefault(item.symbol, []).append(item)
    return {symbol: tuple(items) for symbol, items in grouped.items()}


async def _run_worker_pool(
    *,
    pipeline: SignalPipeline,
    prediction_repository: PredictionRepository,
    signal_repository: SignalRepository,
    work_by_symbol: Mapping[Symbol, Sequence[DiscoveredWorkItem]],
    worker_count: int,
    overwrite: bool,
    debug: bool,
    framework: str,
    policy_name: str,
    model_name: str,
    model_version: str,
) -> tuple[SignalTaskResult, ...]:
    """Drain symbols through a bounded asyncio worker pool."""
    symbols = tuple(work_by_symbol.keys())
    if len(symbols) == 0:
        return ()

    queue: asyncio.Queue[Symbol | None] = asyncio.Queue()
    for symbol in symbols:
        queue.put_nowait(symbol)
    for _ in range(worker_count):
        queue.put_nowait(None)

    collected: list[SignalTaskResult] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                results = await _generate_symbol_work(
                    pipeline=pipeline,
                    prediction_repository=prediction_repository,
                    signal_repository=signal_repository,
                    symbol=item,
                    work_items=work_by_symbol[item],
                    overwrite=overwrite,
                    debug=debug,
                    framework=framework,
                    policy_name=policy_name,
                    model_name=model_name,
                    model_version=model_version,
                )
                async with lock:
                    collected.extend(results)
            finally:
                queue.task_done()

    worker_tasks = [
        asyncio.create_task(worker(), name=f"generate-signals-worker-{index}")
        for index in range(worker_count)
    ]
    try:
        await asyncio.gather(*worker_tasks)
    finally:
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)

    return tuple(
        sorted(
            collected,
            key=lambda result: (result.symbol, result.timeframe, result.year),
        )
    )


async def _generate_symbol_work(
    *,
    pipeline: SignalPipeline,
    prediction_repository: PredictionRepository,
    signal_repository: SignalRepository,
    symbol: Symbol,
    work_items: Sequence[DiscoveredWorkItem],
    overwrite: bool,
    debug: bool,
    framework: str,
    policy_name: str,
    model_name: str,
    model_version: str,
) -> tuple[SignalTaskResult, ...]:
    """Generate signal datasets for every discovered year for one symbol."""
    results: list[SignalTaskResult] = []
    for item in work_items:
        for year in item.years:
            result = await asyncio.to_thread(
                _generate_partition,
                pipeline,
                prediction_repository,
                signal_repository,
                symbol=symbol,
                timeframe=item.timeframe,
                year=year,
                overwrite=overwrite,
                debug=debug,
                framework=framework,
                policy_name=policy_name,
                model_name=model_name,
                model_version=model_version,
            )
            _print_progress(result)
            results.append(result)
    return tuple(results)


def _generate_partition(
    pipeline: SignalPipeline,
    prediction_repository: PredictionRepository,
    signal_repository: SignalRepository,
    *,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    overwrite: bool,
    debug: bool,
    framework: str,
    policy_name: str,
    model_name: str,
    model_version: str,
) -> SignalTaskResult:
    """Generate one signal year partition synchronously."""
    if not overwrite and signal_repository.exists(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ):
        return SignalTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="skipped",
        )

    try:
        predictions = prediction_repository.load(
            framework=framework,
            model_name=model_name,
            model_version=model_version,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        partition_ref = SignalPartitionRef(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )
        output = pipeline.run(policy_name, predictions, partition_ref)
    except Exception as exc:
        _log_partition_failure(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            exc=exc,
            debug=debug,
        )
        return SignalTaskResult(
            symbol=symbol,
            timeframe=timeframe,
            year=year,
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            error_code=exc.error_code if isinstance(exc, CQROSError) else None,
        )

    return SignalTaskResult(
        symbol=symbol,
        timeframe=timeframe,
        year=year,
        status="succeeded",
        rows_generated=output.height,
    )


def _print_progress(result: SignalTaskResult) -> None:
    """Print a deterministic one-line progress record for a task result."""
    label = f"{result.symbol} {result.timeframe} {result.year}"
    if result.status == "succeeded":
        rows = result.rows_generated if result.rows_generated is not None else 0
        message = f"OK {label} rows={rows}"
    elif result.status == "skipped":
        message = f"SKIP {label}"
    else:
        error_type = result.error_type if result.error_type is not None else "Exception"
        message = f"FAIL {label} {error_type}"
    print(message, flush=True)


def _log_partition_failure(
    *,
    symbol: Symbol,
    timeframe: Timeframe,
    year: int,
    exc: BaseException,
    debug: bool,
) -> None:
    """Log a partition generation failure without aborting the run."""
    log_extra = {
        "symbol": symbol,
        "timeframe": timeframe,
        "year": year,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "error_code": exc.error_code if isinstance(exc, CQROSError) else None,
    }
    if debug:
        _logger.exception(
            "Failed signal generation partition; continuing",
            extra=log_extra,
            exc_info=True,
        )
    else:
        _logger.warning(
            "Failed signal generation partition; continuing",
            extra=log_extra,
        )


def _build_summary(
    *,
    options: SignalGenerationOptions,
    work: Sequence[DiscoveredWorkItem],
    results: Sequence[SignalTaskResult],
    duration_seconds: float,
    output_directory: Path,
) -> SignalGenerationSummary:
    """Aggregate task results into a generation report."""
    symbols_discovered = {item.symbol for item in work}
    symbols_processed: set[Symbol] = set()
    timeframes_processed: set[Timeframe] = set()
    successful_tasks = 0
    failed_tasks = 0
    skipped_tasks = 0
    rows_generated = 0
    failed_labels: set[str] = set()

    for result in results:
        symbols_processed.add(result.symbol)
        timeframes_processed.add(result.timeframe)
        if result.status == "succeeded":
            successful_tasks += 1
            if result.rows_generated is not None:
                rows_generated += result.rows_generated
        elif result.status == "skipped":
            skipped_tasks += 1
        else:
            failed_tasks += 1
            failed_labels.add(f"{result.symbol} {result.timeframe} {result.year}")

    return SignalGenerationSummary(
        policy=options.policy,
        model=options.model,
        version=options.version,
        symbols_discovered=len(symbols_discovered),
        symbols_processed=len(symbols_processed),
        timeframes_processed=len(timeframes_processed),
        successful_tasks=successful_tasks,
        failed_tasks=failed_tasks,
        skipped_tasks=skipped_tasks,
        rows_generated=rows_generated,
        duration_seconds=duration_seconds,
        output_directory=output_directory,
        failed_task_labels=tuple(sorted(failed_labels)),
    )


def _format_duration(duration_seconds: float) -> str:
    """Format duration seconds for the summary report."""
    return f"{duration_seconds:.3f}s"


def _format_output_directory(path: Path) -> str:
    """Format the output directory using POSIX separators."""
    return path.as_posix()


class _DiscoveryPersistence(ModelPersistence):
    """Composition-root persistence stub used only for artifact discovery.

    ``resolve_model_artifact`` relies on filesystem discovery and never loads
    model binaries. This stub satisfies the ``ModelArtifactRepository``
    constructor contract without implementing serialization.
    """

    def save(self, model: object, path: Path | str) -> None:
        """Reject save attempts; discovery never persists models."""
        raise ModelValidationError(
            "model persistence is not supported by the signal-generation CLI",
            error_code=_ERROR_PERSISTENCE,
            details={"path": str(path)},
        )

    def load(self, path: Path | str) -> Model:
        """Reject load attempts; discovery never loads models."""
        raise ModelValidationError(
            "model persistence is not supported by the signal-generation CLI",
            error_code=_ERROR_PERSISTENCE,
            details={"path": str(path)},
        )

    def exists(self, path: Path | str) -> bool:
        """Return whether a model binary exists at ``path``."""
        return Path(path).is_file()

    def delete(self, path: Path | str) -> None:
        """Reject delete attempts; discovery never deletes models."""
        raise ModelValidationError(
            "model persistence is not supported by the signal-generation CLI",
            error_code=_ERROR_PERSISTENCE,
            details={"path": str(path)},
        )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

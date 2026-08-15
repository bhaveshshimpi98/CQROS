"""CQROS Factor Research Engine generation orchestration pipeline.

Purpose:
    Wire the completed Factor Research Engine components into one
    deterministic generation flow that persists canonical factor datasets.

Responsibilities:
    - Accept a training DataFrame for one factors partition
    - Select executable factors through ``ExecutableFactorCatalog`` based on
      available input columns (no hard-coded allowlist)
    - Execute the injected ``FactorPipeline`` against only those factors by
      supplying a filtered registry
    - Build a ``Mapping[str, FactorMetadata]`` from executable factors
    - Execute the injected ``WideToLongFactorTransformer``
    - Validate the long frame against ``FACTOR_SCHEMA``
    - Sort the long frame by ``(symbol, timeframe, open_time, factor_name)``
      before persistence (``unpivot`` emits factor-major order)
    - Persist through ``FactorsRepository.save``
    - Return immutable generation statistics including registered /
      executable / skipped factor counts
    - Fail fast on duplicate factor names, missing metadata, schema
      validation failures, and repository save failures
    - Remain free of CLI, factor validation services, factor selection,
      research runners, new factor calculations, registry mutation, and
      storage-layout changes

Dependencies:
    ``polars``, ``cqros.core.types``, ``cqros.factors.exceptions``,
    ``cqros.factors.executable_catalog``, ``cqros.factors.metadata``,
    ``cqros.factors.pipeline``, ``cqros.factors.registry``,
    ``cqros.factors.repository``, ``cqros.factors.schema``, and
    ``cqros.factors.wide_to_long``.

Public API:
    ``FactorGenerationPipeline``, ``FactorGenerationStatistics``
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.factors.exceptions import (
    FactorError,
    FactorExecutionError,
    FactorRegistrationError,
    FactorValidationError,
)
from cqros.factors.executable_catalog import ExecutableFactorCatalog
from cqros.factors.interfaces import Factor
from cqros.factors.metadata import FactorMetadata
from cqros.factors.pipeline import FactorPipeline
from cqros.factors.registry import FactorRegistry
from cqros.factors.repository import FactorsRepository
from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.factors.wide_to_long import WideToLongFactorTransformer

__all__ = [
    "FactorGenerationPipeline",
    "FactorGenerationStatistics",
]

_ERROR_DUPLICATE_NAME: Final[str] = "FACTOR-GEN-001"
_ERROR_DUPLICATE_COLUMN: Final[str] = "FACTOR-GEN-002"
_ERROR_MISSING_METADATA: Final[str] = "FACTOR-GEN-003"
_ERROR_METADATA_NAME_MISMATCH: Final[str] = "FACTOR-GEN-004"
_ERROR_SCHEMA: Final[str] = "FACTOR-GEN-005"
_ERROR_SAVE: Final[str] = "FACTOR-GEN-006"
_ERROR_FRAME_TYPE: Final[str] = "FACTOR-GEN-007"

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FactorGenerationStatistics:
    """Immutable summary of one factor-generation pipeline run.

    Attributes:
        symbols_processed: Count of unique symbols for which generation was
            attempted.
        rows_generated: Number of long-format factor rows produced.
        factors_generated: Number of distinct factor columns generated.
        total_registered_factors: Count of factors in the full registry.
        executable_factors: Count of factors selected for execution.
        skipped_factors: Count of registered factors skipped because required
            columns were absent. Informational only; never a failure.
        generation_duration: Wall-clock seconds spent in ``run``.
        failed_symbols: Symbols that failed generation. Empty on success
            because this pipeline fails immediately on error.
        successful_symbols: Symbols that completed generation successfully.
    """

    symbols_processed: int
    rows_generated: int
    factors_generated: int
    total_registered_factors: int
    executable_factors: int
    skipped_factors: int
    generation_duration: float
    failed_symbols: tuple[Symbol, ...]
    successful_symbols: tuple[Symbol, ...]


class FactorGenerationPipeline:
    """Orchestrate training → filter → wide → long → repository persistence.

    All collaborators are injected. The pipeline never constructs a registry,
    transformer, or repository internally and never mutates the caller-supplied
    training DataFrame. Factor selection is performed by
    ``ExecutableFactorCatalog`` outside ``FactorPipeline`` compute logic.

    Args:
        registry: Full factor catalog (registered universe).
        pipeline: Wide-matrix execution engine. When ``None``, a
            ``FactorPipeline`` is constructed per run over the filtered
            executable registry.
        transformer: Wide-to-long ``FACTOR_SCHEMA`` converter.
        repository: Persistence facade for canonical factor partitions.
        executable_catalog: Optional catalog used to select executable
            factors. Defaults to ``ExecutableFactorCatalog(registry)``.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = (
        "_catalog",
        "_logger",
        "_pipeline",
        "_registry",
        "_repository",
        "_transformer",
    )

    _registry: FactorRegistry
    _pipeline: FactorPipeline | None
    _transformer: WideToLongFactorTransformer
    _repository: FactorsRepository
    _catalog: ExecutableFactorCatalog
    _logger: logging.Logger

    def __init__(
        self,
        registry: FactorRegistry,
        pipeline: FactorPipeline | None,
        transformer: WideToLongFactorTransformer,
        repository: FactorsRepository,
        *,
        executable_catalog: ExecutableFactorCatalog | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the generation pipeline with injected collaborators.

        Args:
            registry: Full factor catalog providing the registered universe.
            pipeline: Optional factor execution engine. When provided, it is
                used only as a test override after executable filtering builds
                metadata; production wiring passes ``None`` so each run
                constructs ``FactorPipeline`` over the filtered registry.
            transformer: Converter from wide matrix to ``FACTOR_SCHEMA``.
            repository: Repository used to persist the long factor frame.
            executable_catalog: Optional executable-factor selector.
            logger: Optional logger instance.
        """
        self._registry = registry
        self._pipeline = pipeline
        self._transformer = transformer
        self._repository = repository
        self._catalog = (
            executable_catalog
            if executable_catalog is not None
            else ExecutableFactorCatalog(registry)
        )
        self._logger = logger if logger is not None else _logger

    def run(
        self,
        frame: pl.DataFrame,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> FactorGenerationStatistics:
        """Generate, validate, and persist executable factors for one partition.

        Empty training frames return zeroed statistics without executing the
        factor pipeline, transformer, or repository. Non-empty frames:

        ``ExecutableFactorCatalog`` → filtered ``FactorPipeline`` → metadata
        map → ``WideToLongFactorTransformer`` → ``FACTOR_SCHEMA`` validation →
        ``FactorsRepository.save``.

        Factors whose ``required_features`` are absent from ``frame`` are
        skipped informationally and are never treated as failures.

        Args:
            frame: Training DataFrame containing primary keys and available
                features. Never mutated.
            manager: Order manager identifier for the persisted partition.
            exchange: Exchange identifier for the persisted partition.
            market: Market segment for the persisted partition.
            symbol: Tradeable symbol for the persisted partition.
            timeframe: Factor bar interval for the persisted partition.
            year: Calendar year of the persisted partition.

        Returns:
            Immutable ``FactorGenerationStatistics`` for the run.

        Raises:
            FactorValidationError: If ``frame`` is not a DataFrame, metadata
                is missing or mismatched, or ``FACTOR_SCHEMA`` validation
                fails.
            FactorRegistrationError: If the executable set exposes duplicate
                factor names or produced columns.
            FactorExecutionError: If repository persistence fails with a
                non-factor exception.
            FactorError: Propagated unchanged from collaborators.
        """
        started = time.perf_counter()
        validated = _require_dataframe(frame)
        registered_count = len(self._registry.list())

        if validated.height == 0:
            duration = time.perf_counter() - started
            self._logger.info(
                "Factor generation skipped for empty training frame",
                extra={
                    "manager": manager,
                    "exchange": exchange,
                    "market": market,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "total_registered_factors": registered_count,
                    "generation_duration": duration,
                },
            )
            return FactorGenerationStatistics(
                symbols_processed=0,
                rows_generated=0,
                factors_generated=0,
                total_registered_factors=registered_count,
                executable_factors=0,
                skipped_factors=0,
                generation_duration=duration,
                failed_symbols=(),
                successful_symbols=(),
            )

        available_columns = tuple(validated.columns)
        executable = self._catalog.get_executable_factors(available_columns)
        skipped = self._catalog.get_skipped_factors(available_columns)
        executable_count = len(executable)
        skipped_count = len(skipped)

        self._logger.info(
            "Starting factor generation",
            extra={
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
                "row_count": validated.height,
                "total_registered_factors": registered_count,
                "executable_factors": executable_count,
                "skipped_factors": skipped_count,
                "skipped_factor_names": tuple(factor.name for factor in skipped),
            },
        )

        metadata = _build_metadata_map(executable)
        filtered_registry = _registry_from_factors(executable)
        wide = self._execute_pipeline(filtered_registry, validated)
        factor_columns = _factor_columns(wide)
        _require_metadata_coverage(factor_columns, metadata)

        long_frame = self._transformer.transform(wide, metadata)
        schema_validated = _require_factor_schema(long_frame)
        # ``unpivot`` emits factor-major row order, which breaks global
        # ``open_time`` monotonicity required by FactorVerifier. Restore
        # canonical bar-major ordering once before persistence.
        ordered = schema_validated.sort(
            [*PRIMARY_KEY_COLUMNS, "factor_name"],
        )

        self._save_partition(
            ordered,
            manager=manager,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        )

        duration = time.perf_counter() - started
        statistics = FactorGenerationStatistics(
            symbols_processed=1,
            rows_generated=ordered.height,
            factors_generated=len(factor_columns),
            total_registered_factors=registered_count,
            executable_factors=executable_count,
            skipped_factors=skipped_count,
            generation_duration=duration,
            failed_symbols=(),
            successful_symbols=(symbol,),
        )
        self._logger.info(
            "Factor generation completed",
            extra={
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "symbol": symbol,
                "timeframe": timeframe,
                "year": year,
                "symbols_processed": statistics.symbols_processed,
                "rows_generated": statistics.rows_generated,
                "factors_generated": statistics.factors_generated,
                "total_registered_factors": statistics.total_registered_factors,
                "executable_factors": statistics.executable_factors,
                "skipped_factors": statistics.skipped_factors,
                "generation_duration": statistics.generation_duration,
            },
        )
        return statistics

    def _execute_pipeline(
        self,
        filtered_registry: FactorRegistry,
        frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """Run ``FactorPipeline`` over the executable filtered registry.

        Production uses a new ``FactorPipeline(filtered_registry)``. When an
        execution pipeline was injected (unit tests), that override is used
        instead so collaborators can be mocked without changing
        ``FactorPipeline`` compute logic.
        """
        if self._pipeline is not None:
            return self._pipeline.run(frame)
        return FactorPipeline(filtered_registry, logger=self._logger).run(frame)

    def _save_partition(
        self,
        frame: pl.DataFrame,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> None:
        """Persist ``frame`` through the injected repository.

        Raises:
            FactorError: Propagated unchanged from the repository.
            FactorExecutionError: If save raises a non-factor exception.
        """
        try:
            self._repository.save(
                frame,
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
        except FactorError:
            raise
        except Exception as exc:
            raise FactorExecutionError(
                "factor repository save failed",
                error_code=_ERROR_SAVE,
                details={
                    "manager": manager,
                    "exchange": exchange,
                    "market": market,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            ) from exc


def _registry_from_factors(factors: Sequence[Factor]) -> FactorRegistry:
    """Build a new registry containing only ``factors``."""
    registry = FactorRegistry()
    if factors:
        registry.register_many(factors)
    return registry


def _require_dataframe(frame: object) -> pl.DataFrame:
    """Raise when ``frame`` is not a Polars DataFrame.

    Raises:
        FactorValidationError: If ``frame`` is not a ``pl.DataFrame``.
    """
    if not isinstance(frame, pl.DataFrame):
        raise FactorValidationError(
            "training frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    return frame


def _build_metadata_map(factors: Sequence[Factor]) -> dict[str, FactorMetadata]:
    """Build a produced-column metadata map from executable ``factors``.

    Metadata is projected from each factor in ``factors``. The transformer
    receives only this map and never accesses the registry.

    Returns:
        Mapping from produced column name to ``FactorMetadata``.

    Raises:
        FactorRegistrationError: If duplicate factor names or produced
            columns are detected.
        FactorValidationError: If metadata ``name`` does not match a
            produced column key required by the transformer.
    """
    mapping: dict[str, FactorMetadata] = {}
    seen_names: set[str] = set()

    for factor in factors:
        name = factor.name
        if name in seen_names:
            raise FactorRegistrationError(
                f"duplicate factor name in catalog: {name}",
                error_code=_ERROR_DUPLICATE_NAME,
                details={"name": name},
            )
        seen_names.add(name)

        metadata = _factor_metadata(factor)
        for column in factor.produced_columns:
            if column in mapping:
                raise FactorRegistrationError(
                    f"duplicate produced column in catalog: {column}",
                    error_code=_ERROR_DUPLICATE_COLUMN,
                    details={
                        "column": column,
                        "name": name,
                        "owner": mapping[column].name,
                    },
                )
            if metadata.name != column:
                raise FactorValidationError(
                    "factor metadata name does not match produced column",
                    error_code=_ERROR_METADATA_NAME_MISMATCH,
                    details={
                        "produced_column": column,
                        "metadata_name": metadata.name,
                        "factor": name,
                    },
                )
            mapping[column] = metadata

    return mapping


def _factor_metadata(factor: Factor) -> FactorMetadata:
    """Project a registered factor into immutable metadata.

    Args:
        factor: Factor exposing the public metadata attribute contract.

    Returns:
        Immutable ``FactorMetadata`` snapshot for transformer enrichment.
    """
    return FactorMetadata(
        name=factor.name,
        version=factor.version,
        description=factor.description,
        category=factor.category,
        required_features=tuple(factor.required_features),
        produced_columns=tuple(factor.produced_columns),
        lookback=factor.lookback,
        factor_group=factor.factor_group,
        prediction_horizon=factor.prediction_horizon,
        enabled=factor.enabled,
        status=factor.status,
    )


def _factor_columns(frame: pl.DataFrame) -> list[str]:
    """Return non-primary-key columns in frame order."""
    primary = set(PRIMARY_KEY_COLUMNS)
    return [column for column in frame.columns if column not in primary]


def _require_metadata_coverage(
    factor_columns: Sequence[str],
    metadata: Mapping[str, FactorMetadata],
) -> None:
    """Raise when any wide factor column lacks matching metadata.

    Raises:
        FactorValidationError: If metadata is missing for any factor column.
    """
    missing = tuple(column for column in factor_columns if column not in metadata)
    if missing:
        raise FactorValidationError(
            "factor metadata is missing for one or more factor columns",
            error_code=_ERROR_MISSING_METADATA,
            details={
                "missing_factor_names": missing,
                "available_metadata_keys": tuple(sorted(metadata.keys())),
                "factor_columns": tuple(factor_columns),
            },
        )


def _require_factor_schema(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate and cast ``frame`` to ``FACTOR_SCHEMA``.

    Raises:
        FactorValidationError: If required columns are missing or casting fails.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            "generated factors frame is missing required columns",
            error_code=_ERROR_SCHEMA,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )
    try:
        return frame.select(list(CANONICAL_COLUMN_ORDER)).cast(FACTOR_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise FactorValidationError(
            "generated factors frame failed FACTOR_SCHEMA validation",
            error_code=_ERROR_SCHEMA,
            details={"reason": str(exc)},
        ) from exc

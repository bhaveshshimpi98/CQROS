"""CQROS Factor Validation Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical Factors and Labels
    panels into canonical factor-validation datasets through
    ``ValidationDatasetBuilder`` and registered ``FactorValidationEngine``
    implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``FactorValidationEngineRegistry``
    - Assemble the cross-sectional validation input through an injected
      ``ValidationDatasetBuilder`` (full-panel or memory-efficient spill)
    - Delegate validation-row assembly exclusively to an injected engine
    - Validate required factor-validation schema columns on the engine output
    - Reject missing or duplicate primary keys
    - Finalize outputs against ``FACTOR_VALIDATION_SCHEMA``
    - Remain free of validation algorithms, join logic, persistence,
      verification, exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.core.types``,
    ``cqros.factor_validation.dataset_builder``,
    ``cqros.factor_validation.engine``,
    ``cqros.factor_validation.exceptions``,
    ``cqros.factor_validation.memory_efficient``,
    ``cqros.factor_validation.registry``, and
    ``cqros.factor_validation.schema``.

Public API:
    ``FactorValidationPipeline``

Notes:
    Column contracts for engine output come exclusively from ``schema.py``
    (``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``,
    ``FACTOR_VALIDATION_SCHEMA``). Validation-input assembly belongs to
    ``ValidationDatasetBuilder``. Metric formulas remain exclusively in the
    engine. Memory-efficient execution only changes materialization schedule.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.factor_validation.dataset_builder import ValidationDatasetBuilder
from cqros.factor_validation.exceptions import FactorValidationError
from cqros.factor_validation.memory_efficient import (
    FactorValidationExecutionConfig,
    FactorValidationExecutionMode,
    MemoryEfficientFactorValidationRunner,
)
from cqros.factor_validation.registry import FactorValidationEngineRegistry
from cqros.factor_validation.schema import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_VALIDATION_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["FactorValidationPipeline"]

_ERROR_NAME_BLANK: Final[str] = "FVAL_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "FVAL_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "FVAL_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "FVAL_PIPE_MISSING_COLUMNS"
_ERROR_MISSING_PRIMARY_KEYS: Final[str] = "FVAL_PIPE_MISSING_PRIMARY_KEYS"
_ERROR_DUPLICATE_KEYS: Final[str] = "FVAL_PIPE_DUPLICATE_KEYS"
_ERROR_SCHEMA_CAST: Final[str] = "FVAL_PIPE_SCHEMA_CAST"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class FactorValidationPipeline:
    """Deterministic orchestrator for canonical factor-validation assembly.

    The pipeline resolves a registered ``FactorValidationEngine``, assembles
    the cross-sectional validation dataset through
    ``ValidationDatasetBuilder`` (or the memory-efficient spill runner),
    delegates validation-row generation, and finalizes the result to
    ``FACTOR_VALIDATION_SCHEMA``. Validation semantics remain exclusively in
    the engine. Join logic remains exclusively in the builder.

    Args:
        registry: Registry used to resolve factor-validation-engine
            implementations.
        builder: Builder that loads Factors and Labels and assembles the
            engine input dataset.
        execution_config: Optional execution-mode configuration. Defaults to
            memory-efficient factor-identity batching.
    """

    __slots__ = ("_builder", "_execution_config", "_registry")

    _registry: FactorValidationEngineRegistry
    _builder: ValidationDatasetBuilder
    _execution_config: FactorValidationExecutionConfig

    def __init__(
        self,
        registry: FactorValidationEngineRegistry,
        builder: ValidationDatasetBuilder,
        *,
        execution_config: FactorValidationExecutionConfig | None = None,
    ) -> None:
        """Initialize the pipeline with a registry and validation dataset builder.

        Args:
            registry: Registry containing ``FactorValidationEngine``
                implementations.
            builder: Assembles the Factors+Labels validation input dataset.
            execution_config: Optional full-panel vs memory-efficient settings.
        """
        self._registry = registry
        self._builder = builder
        self._execution_config = (
            execution_config if execution_config is not None else FactorValidationExecutionConfig()
        )

    def run(
        self,
        engine_name: str,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
        symbols: Sequence[Symbol] | None = None,
    ) -> pl.DataFrame:
        """Assemble a validation panel, run an engine, and finalize output.

        ``engine_name`` is validated and resolved first. Depending on
        ``execution_config.mode``, either the full panel is materialized in
        RAM or the memory-efficient spill runner validates factor-identity
        batches through the same engine. The engine output is checked against
        ``REQUIRED_COLUMNS`` / ``CANONICAL_COLUMN_ORDER``, rejected when
        primary keys are missing or duplicated, reordered to
        ``CANONICAL_COLUMN_ORDER``, and cast to ``FACTOR_VALIDATION_SCHEMA``.

        Args:
            engine_name: Registry key of the factor validation engine to
                execute.
            manager: Order manager identifier for the Factors partitions.
            exchange: Exchange identifier.
            market: Market segment.
            timeframe: Bar interval.
            year: Calendar year of the panel.
            symbols: Optional symbol allowlist forwarded to the builder.
                ``None`` includes every symbol with both Factors and Labels.

        Returns:
            A new DataFrame containing the finalized factor validation rows.

        Raises:
            FactorValidationError: If ``engine_name`` is invalid, the engine
                is unknown, dataset assembly fails, or the engine output fails
                factor-validation-schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        engine = self._registry.get(validated_name)
        if self._execution_config.mode is FactorValidationExecutionMode.MEMORY_EFFICIENT:
            runner = MemoryEfficientFactorValidationRunner(
                self._builder,
                engine,
                self._execution_config,
            )
            created = runner.run(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
                symbols=symbols,
            )
        else:
            validation_dataset = self._builder.build(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
                symbols=symbols,
            )
            created = engine.build(validation_dataset)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorValidationError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise FactorValidationError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_primary_key_columns(frame)
    _require_factor_validation_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    try:
        return ordered.cast(FACTOR_VALIDATION_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise FactorValidationError(
            "engine output cannot be cast to FACTOR_VALIDATION_SCHEMA",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise FactorValidationError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_factor_validation_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required factor-validation-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            "factor validation schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_primary_key_columns(frame: pl.DataFrame) -> None:
    """Raise when any primary-key column is missing from ``frame``."""
    missing = [column for column in PRIMARY_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            "factor validation frame is missing primary key columns",
            error_code=_ERROR_MISSING_PRIMARY_KEYS,
            details={
                "missing_columns": tuple(missing),
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``."""
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise FactorValidationError(
            "factor validation frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )

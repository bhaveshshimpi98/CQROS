"""CQROS Factor Research Engine execution pipeline.

Purpose:
    Execute every registered production factor independently against a
    training DataFrame and merge produced columns into one wide factor
    matrix.

Responsibilities:
    - Validate primary-key and required-feature columns on the input frame
    - Load the production factor catalog via ``build_default_registry`` when
      no registry is injected
    - Align each factor's input via ``FactorInputPartition`` using only that
      factor's declared ``required_features``
    - Execute every registered factor exactly once on its aligned frame
    - Left-join produced columns onto the original primary-key timeline
    - Collect produced columns and merge them with primary-key columns
    - Fail fast on duplicate names, duplicate produced columns, missing
      required features, compute failures, and duplicate merged columns
    - Preserve input DataFrame immutability
    - Remain free of CLI, repository, storage, validation services,
      verification, wide-to-long conversion, and research orchestration

Dependencies:
    ``polars``, ``cqros.factors.default_registry``,
    ``cqros.factors.exceptions``, ``cqros.factors.input_partition``,
    ``cqros.factors.interfaces.Factor``, ``cqros.factors.registry``, and
    ``cqros.factors.schema``.

Public API:
    ``FactorPipeline``
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.factors.default_registry import build_default_registry
from cqros.factors.exceptions import (
    FactorExecutionError,
    FactorRegistrationError,
    FactorValidationError,
)
from cqros.factors.input_partition import (
    KNOWN_FACTOR_INPUT_FEATURES,
    FactorInputPartition,
)
from cqros.factors.interfaces import Factor
from cqros.factors.registry import FactorRegistry
from cqros.factors.schema import PRIMARY_KEY_COLUMNS

__all__ = ["FactorPipeline"]

_ERROR_MISSING_PRIMARY_KEY: Final[str] = "FACTOR-PIPE-001"
_ERROR_MISSING_FEATURES: Final[str] = "FACTOR-PIPE-002"
_ERROR_DUPLICATE_NAME: Final[str] = "FACTOR-PIPE-003"
_ERROR_DUPLICATE_COLUMN: Final[str] = "FACTOR-PIPE-004"
_ERROR_EXECUTION: Final[str] = "FACTOR-PIPE-005"
_ERROR_MISSING_PRODUCED: Final[str] = "FACTOR-PIPE-006"
_ERROR_MERGED_DUPLICATE: Final[str] = "FACTOR-PIPE-007"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_logger = logging.getLogger(__name__)


class FactorPipeline:
    """Deterministic execution engine for the production factor catalog.

    Every registered factor is aligned and computed independently from the
    original training DataFrame using ``FactorInputPartition``. Factor
    outputs are never chained: produced columns are left-joined onto the
    original primary-key timeline and merged into one wide matrix. The
    caller-supplied input frame is never mutated.

    Args:
        registry: Optional factor catalog. When omitted, the production
            catalog is loaded through ``build_default_registry``.
        input_partition: Optional factor-input partition helper. Defaults to
            a new ``FactorInputPartition``.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_input_partition", "_logger", "_registry")

    _registry: FactorRegistry
    _input_partition: FactorInputPartition
    _logger: logging.Logger

    def __init__(
        self,
        registry: FactorRegistry | None = None,
        *,
        input_partition: FactorInputPartition | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the pipeline with an optional factor registry.

        Args:
            registry: Factor catalog used for execution. Defaults to the
                production catalog from ``build_default_registry``.
            input_partition: Optional dependency-aware input aligner.
            logger: Optional logger instance.
        """
        self._registry = registry if registry is not None else build_default_registry()
        self._input_partition = (
            input_partition if input_partition is not None else FactorInputPartition(logger=logger)
        )
        self._logger = logger if logger is not None else _logger

    @property
    def registry(self) -> FactorRegistry:
        """Return the factor registry used by this pipeline."""
        return self._registry

    def run(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Execute every registered factor and return one wide factor matrix.

        Validation runs before execution. Each factor is aligned through
        ``FactorInputPartition`` using only its declared dependencies, then
        computed on that aligned frame. Produced columns are left-joined onto
        the original primary-key timeline and merged with
        ``PRIMARY_KEY_COLUMNS`` in deterministic registry order. The original
        ``frame`` is never mutated.

        Args:
            frame: Training DataFrame containing primary keys and every
                required feature column. Companion columns may contain leading
                nulls; factor-specific alignment drops only rows required by
                each factor.

        Returns:
            A new DataFrame with ``symbol``, ``timeframe``, ``open_time``,
            and every generated factor column. Height matches ``frame``.

        Raises:
            FactorValidationError: If primary keys or required features are
                missing, a factor omits a declared produced column, or the
                merged frame contains duplicate column names.
            FactorRegistrationError: If duplicate factor names or produced
                columns are detected in the catalog.
            FactorExecutionError: If any factor ``compute`` raises.
        """
        factors = self._registry.list()
        _require_primary_key_columns(frame)
        _require_unique_factor_names(factors)
        produced_order = _require_unique_produced_columns(factors)
        _require_required_features(frame, factors)

        self._logger.info(
            "Executing factor pipeline",
            extra={
                "factor_count": len(factors),
                "row_count": frame.height,
                "produced_column_count": len(produced_order),
            },
        )

        produced_frames = [self._compute_produced_columns(factor, frame) for factor in factors]
        merged = _merge_wide_matrix(frame, produced_frames, produced_order)

        self._logger.info(
            "Factor pipeline completed",
            extra={
                "factor_count": len(factors),
                "row_count": merged.height,
                "column_count": merged.width,
            },
        )
        return merged

    def _compute_produced_columns(self, factor: Factor, frame: pl.DataFrame) -> pl.DataFrame:
        """Align, execute one factor, and return produced columns on ``frame`` keys.

        Args:
            factor: Factor to execute.
            frame: Original training DataFrame shared by every factor.

        Returns:
            DataFrame containing only ``factor.produced_columns`` with the same
            row count as ``frame``. Bars before the factor's dependency
            boundary are null.

        Raises:
            FactorExecutionError: If ``compute`` raises any exception.
            FactorValidationError: If a declared produced column is missing
                from the compute result or dependency alignment fails.
        """
        allow_non_raw = not set(factor.required_features).issubset(KNOWN_FACTOR_INPUT_FEATURES)
        aligned = self._input_partition.align_frame(
            frame,
            factor.required_features,
            allow_non_raw=allow_non_raw,
        )
        try:
            computed = factor.compute(aligned)
        except Exception as exc:
            raise FactorExecutionError(
                f"factor compute failed: {factor.name}",
                error_code=_ERROR_EXECUTION,
                details={
                    "factor": factor.name,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            ) from exc

        missing = [column for column in factor.produced_columns if column not in computed.columns]
        if missing:
            raise FactorValidationError(
                f"factor compute omitted produced columns: {factor.name}",
                error_code=_ERROR_MISSING_PRODUCED,
                details={
                    "factor": factor.name,
                    "missing_columns": tuple(missing),
                    "produced_columns": tuple(factor.produced_columns),
                    "available_columns": tuple(computed.columns),
                },
            )
        produced_values = computed.select(list(factor.produced_columns))
        if aligned.height == frame.height:
            return produced_values
        keyed = aligned.select(_PRIMARY_KEY_LIST).with_columns(produced_values)
        joined = frame.select(_PRIMARY_KEY_LIST).join(
            keyed,
            on=_PRIMARY_KEY_LIST,
            how="left",
        )
        return joined.select(list(factor.produced_columns))


def _require_primary_key_columns(frame: pl.DataFrame) -> None:
    """Raise when any factor-matrix primary-key column is missing.

    Raises:
        FactorValidationError: If one or more ``PRIMARY_KEY_COLUMNS`` are absent.
    """
    missing = [column for column in PRIMARY_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            "training frame is missing primary key columns",
            error_code=_ERROR_MISSING_PRIMARY_KEY,
            details={
                "missing_columns": tuple(missing),
                "required_columns": PRIMARY_KEY_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_required_features(frame: pl.DataFrame, factors: Sequence[Factor]) -> None:
    """Raise when any registered factor is missing a required feature.

    Raises:
        FactorValidationError: If one or more required feature columns are absent.
    """
    required: set[str] = set()
    for factor in factors:
        required.update(factor.required_features)
    missing = tuple(sorted(column for column in required if column not in frame.columns))
    if missing:
        raise FactorValidationError(
            "training frame is missing required factor features",
            error_code=_ERROR_MISSING_FEATURES,
            details={
                "missing_columns": missing,
                "required_columns": tuple(sorted(required)),
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_factor_names(factors: Sequence[Factor]) -> None:
    """Raise when the catalog contains duplicate factor names.

    Raises:
        FactorRegistrationError: If any factor name appears more than once.
    """
    seen: set[str] = set()
    for factor in factors:
        name = factor.name
        if name in seen:
            raise FactorRegistrationError(
                f"duplicate factor name in catalog: {name}",
                error_code=_ERROR_DUPLICATE_NAME,
                details={"name": name},
            )
        seen.add(name)


def _require_unique_produced_columns(factors: Sequence[Factor]) -> tuple[str, ...]:
    """Validate produced-column uniqueness and return deterministic order.

    Returns:
        Produced column names in registry factor order, preserving each
        factor's ``produced_columns`` declaration order.

    Raises:
        FactorRegistrationError: If any produced column is claimed twice.
    """
    ordered: list[str] = []
    owners: dict[str, str] = {}
    for factor in factors:
        for column in factor.produced_columns:
            owner = owners.get(column)
            if owner is not None:
                raise FactorRegistrationError(
                    f"duplicate produced column in catalog: {column}",
                    error_code=_ERROR_DUPLICATE_COLUMN,
                    details={
                        "column": column,
                        "name": factor.name,
                        "owner": owner,
                    },
                )
            owners[column] = factor.name
            ordered.append(column)
    return tuple(ordered)


def _merge_wide_matrix(
    frame: pl.DataFrame,
    produced_frames: Sequence[pl.DataFrame],
    produced_order: Sequence[str],
) -> pl.DataFrame:
    """Merge primary keys with produced factor columns into one wide frame.

    Args:
        frame: Original training DataFrame providing primary-key columns.
        produced_frames: Per-factor frames containing only produced columns.
        produced_order: Deterministic factor column order.

    Returns:
        Wide factor matrix with primary keys followed by ``produced_order``.

    Raises:
        FactorValidationError: If the merged frame contains duplicate columns.
    """
    candidate_columns = [*PRIMARY_KEY_COLUMNS, *produced_order]
    if len(candidate_columns) != len(set(candidate_columns)):
        duplicates = tuple(
            sorted({name for name in candidate_columns if candidate_columns.count(name) > 1})
        )
        raise FactorValidationError(
            "merged factor matrix contains duplicate column names",
            error_code=_ERROR_MERGED_DUPLICATE,
            details={
                "duplicate_columns": duplicates,
                "available_columns": tuple(candidate_columns),
            },
        )

    pieces: list[pl.DataFrame] = [frame.select(_PRIMARY_KEY_LIST), *produced_frames]
    merged = pl.concat(pieces, how="horizontal_extend")
    return merged.select([*_PRIMARY_KEY_LIST, *produced_order])

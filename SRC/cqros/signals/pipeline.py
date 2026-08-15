"""CQROS Signals package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical prediction datasets into
    canonical signal datasets, including prediction-frame validation, policy
    resolution, policy delegation, merged-schema finalization, and persistence
    through ``SignalRepository``.

Responsibilities:
    - Validate policy names and resolve policies from ``SignalPolicyRegistry``
    - Validate canonical prediction DataFrame structure
    - Delegate signal creation exclusively to the resolved ``SignalPolicy``
    - Validate required Signal schema columns on the policy output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged signal schema
    - Persist the partition through an injected ``SignalRepository``
    - Preserve prediction-frame immutability
    - Remain free of trading strategy, threshold logic, prediction
      generation, verification, and CLI logic

Dependencies:
    ``polars``, ``cqros.predictions.schema``, ``cqros.signals.exceptions``,
    ``cqros.signals.registry``, ``cqros.signals.schema``, and
    ``cqros.storage.signal_repository``.

Public API:
    ``SignalPipeline``
"""

from __future__ import annotations

import logging
from typing import Final

import polars as pl

from cqros.predictions.schema import (
    REQUIRED_COLUMNS as PREDICTION_REQUIRED_COLUMNS,
)
from cqros.signals.exceptions import SignalValidationError
from cqros.signals.registry import SignalPolicyRegistry
from cqros.signals.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_SIGNAL_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)
from cqros.storage.signal_repository import SignalPartitionRef, SignalRepository

__all__ = ["SignalPipeline"]

_ERROR_NAME_BLANK: Final[str] = "SIGNAL-PIPE-005"
_ERROR_INVALID_PREDICTION_FRAME: Final[str] = "SIGNAL-PIPE-001"
_ERROR_MISSING_SIGNAL_COLUMNS: Final[str] = "SIGNAL-PIPE-002"
_ERROR_DUPLICATE_KEYS: Final[str] = "SIGNAL-PIPE-003"
_ERROR_MISSING_PREDICTION_COLUMNS: Final[str] = "SIGNAL-PIPE-004"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_logger = logging.getLogger(__name__)


class SignalPipeline:
    """Deterministic orchestrator for canonical signal dataset assembly.

    The pipeline resolves a registered ``SignalPolicy``, validates a canonical
    prediction DataFrame, delegates signal creation, finalizes the result to
    the canonical merged signal schema, and persists the partition through
    ``SignalRepository``. Signal semantics remain exclusively in the policy.
    The caller-supplied prediction frame is never mutated.

    Args:
        repository: Persistence facade for merged signal partitions.
        policy_registry: Registry used to resolve signal-policy implementations.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_logger", "_policy_registry", "_repository")

    _repository: SignalRepository
    _policy_registry: SignalPolicyRegistry
    _logger: logging.Logger

    def __init__(
        self,
        repository: SignalRepository,
        policy_registry: SignalPolicyRegistry,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the pipeline with a repository and policy registry.

        Args:
            repository: Repository used to persist finalized partitions.
            policy_registry: Registry containing ``SignalPolicy``
                implementations.
            logger: Optional logger instance.
        """
        self._repository = repository
        self._policy_registry = policy_registry
        self._logger = logger if logger is not None else _logger

    def run(
        self,
        policy_name: str,
        predictions: pl.DataFrame,
        partition_ref: SignalPartitionRef,
    ) -> pl.DataFrame:
        """Resolve a policy, generate, finalize, and persist a signal partition.

        ``policy_name`` is validated and resolved first. ``predictions`` is
        validated next. Signal creation is then delegated to
        ``SignalPolicy.generate``. The policy output is checked against
        ``REQUIRED_COLUMNS``, rejected when primary keys are duplicated,
        reordered to ``CANONICAL_COLUMN_ORDER``, cast to
        ``MERGED_SIGNAL_SCHEMA``, and saved through ``SignalRepository``. The
        original ``predictions`` frame is never mutated.

        Args:
            policy_name: Registry key of the signal policy to execute.
            predictions: Canonical prediction dataset.
            partition_ref: Partition identity used for persistence.

        Returns:
            A new DataFrame containing the finalized merged signal matrix.

        Raises:
            SignalValidationError: If ``policy_name`` is invalid, the policy is
                unknown, ``predictions`` is invalid, required schema columns
                are missing, or primary keys are duplicated.
        """
        validated_name = _require_policy_name(policy_name)
        policy = self._policy_registry.get(validated_name)
        frame = _require_prediction_frame(predictions)
        _require_unique_primary_keys(frame, context="prediction")
        generated = policy.generate(frame)
        finalized = _finalize(generated)

        self._logger.debug(
            "Persisting merged signal partition",
            extra={
                "policy": validated_name,
                "exchange": partition_ref.exchange,
                "market": partition_ref.market,
                "symbol": partition_ref.symbol,
                "timeframe": partition_ref.timeframe,
                "year": partition_ref.year,
                "rows": finalized.height,
                "columns": finalized.width,
            },
        )
        self._repository.save(
            finalized,
            exchange=partition_ref.exchange,
            market=partition_ref.market,
            symbol=partition_ref.symbol,
            timeframe=partition_ref.timeframe,
            year=partition_ref.year,
        )
        self._logger.info(
            "Persisted merged signal partition",
            extra={
                "policy": validated_name,
                "exchange": partition_ref.exchange,
                "market": partition_ref.market,
                "symbol": partition_ref.symbol,
                "timeframe": partition_ref.timeframe,
                "year": partition_ref.year,
                "rows": finalized.height,
                "columns": finalized.width,
            },
        )
        return finalized


def _finalize(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting.

    Args:
        frame: Frame produced by ``SignalPolicy.generate``.

    Returns:
        Finalized merged signal DataFrame.

    Raises:
        SignalValidationError: If required columns are missing or primary
            keys are duplicated.
    """
    _require_signal_schema_columns(frame)
    _require_unique_primary_keys(frame, context="signal")
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_SIGNAL_SCHEMA)


def _require_policy_name(name: object) -> str:
    """Validate and return a non-blank policy name.

    Args:
        name: Candidate policy registry key.

    Returns:
        The validated name string.

    Raises:
        SignalValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise SignalValidationError(
            "policy_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"policy_name": name},
        )
    return name


def _require_prediction_frame(predictions: object) -> pl.DataFrame:
    """Raise when ``predictions`` fails structural integrity checks.

    Args:
        predictions: Candidate canonical prediction dataset.

    Returns:
        ``predictions`` as a DataFrame after structural checks.

    Raises:
        SignalValidationError: If the value is not a non-empty DataFrame or
            required prediction-schema columns are missing.
    """
    if not isinstance(predictions, pl.DataFrame):
        raise SignalValidationError(
            "predictions must be a polars DataFrame",
            error_code=_ERROR_INVALID_PREDICTION_FRAME,
            details={
                "actual_type": type(predictions).__name__,
            },
        )
    if predictions.height == 0:
        raise SignalValidationError(
            "predictions must contain at least one row",
            error_code=_ERROR_INVALID_PREDICTION_FRAME,
            details={
                "rows": predictions.height,
            },
        )

    missing = [
        column for column in PREDICTION_REQUIRED_COLUMNS if column not in predictions.columns
    ]
    if missing:
        raise SignalValidationError(
            "prediction frame is missing required columns",
            error_code=_ERROR_MISSING_PREDICTION_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": PREDICTION_REQUIRED_COLUMNS,
                "available_columns": tuple(predictions.columns),
            },
        )
    return predictions


def _require_signal_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing.

    Raises:
        SignalValidationError: If one or more ``REQUIRED_COLUMNS`` are absent.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SignalValidationError(
            "merged signal schema is missing required columns",
            error_code=_ERROR_MISSING_SIGNAL_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_primary_keys(
    frame: pl.DataFrame,
    *,
    context: str,
) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``.

    Args:
        frame: DataFrame containing primary-key columns.
        context: Frame role used in the error message (``prediction`` or
            ``signal``).

    Raises:
        SignalValidationError: If any primary-key combination appears more
            than once.
    """
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise SignalValidationError(
            f"{context} frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )

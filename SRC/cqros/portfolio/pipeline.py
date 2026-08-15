"""CQROS Portfolio package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical signal datasets into
    canonical portfolio datasets through registered ``PortfolioOptimizer``
    implementations.

Responsibilities:
    - Validate optimizer names and resolve optimizers from
      ``PortfolioOptimizerRegistry``
    - Validate canonical signal DataFrame structure
    - Delegate allocation exclusively to an injected optimizer
    - Validate required Portfolio schema columns on the optimizer output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged portfolio schema
    - Preserve signal-frame immutability
    - Remain free of allocation algorithms, persistence, verification, and
      CLI logic

Dependencies:
    ``polars``, ``cqros.portfolio.exceptions``, ``cqros.portfolio.interfaces``,
    ``cqros.portfolio.registry``, and ``cqros.portfolio.schema``.

Public API:
    ``PortfolioPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.portfolio.exceptions import PortfolioValidationError
from cqros.portfolio.interfaces import validate_signals_frame
from cqros.portfolio.registry import PortfolioOptimizerRegistry
from cqros.portfolio.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_PORTFOLIO_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["PortfolioPipeline"]

_ERROR_NAME_BLANK: Final[str] = "PORTFOLIO_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "PORTFOLIO_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "PORTFOLIO_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "PORTFOLIO_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "PORTFOLIO_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class PortfolioPipeline:
    """Deterministic orchestrator for canonical portfolio dataset assembly.

    The pipeline resolves a registered ``PortfolioOptimizer``, validates a
    canonical signal DataFrame, delegates optimization, and finalizes the
    result to the canonical merged portfolio schema. Allocation semantics
    remain exclusively in the optimizer. The caller-supplied signal frame is
    never mutated.

    Args:
        optimizer_registry: Registry used to resolve optimizer implementations.
    """

    __slots__ = ("_optimizer_registry",)

    _optimizer_registry: PortfolioOptimizerRegistry

    def __init__(self, optimizer_registry: PortfolioOptimizerRegistry) -> None:
        """Initialize the pipeline with an optimizer registry.

        Args:
            optimizer_registry: Registry containing ``PortfolioOptimizer``
                implementations.
        """
        self._optimizer_registry = optimizer_registry

    def run(self, optimizer_name: str, signals: pl.DataFrame) -> pl.DataFrame:
        """Resolve an optimizer and produce a finalized portfolio frame.

        ``optimizer_name`` is validated and resolved first. ``signals`` is
        validated through ``validate_signals_frame``. Optimization is then
        delegated to ``PortfolioOptimizer.optimize``. The optimizer output is
        checked against ``REQUIRED_COLUMNS``, rejected when primary keys are
        duplicated, reordered to ``CANONICAL_COLUMN_ORDER``, and cast to
        ``MERGED_PORTFOLIO_SCHEMA``. The original ``signals`` frame is never
        mutated.

        Args:
            optimizer_name: Registry key of the optimizer to execute.
            signals: Canonical signal dataset.

        Returns:
            A new DataFrame containing the finalized merged portfolio matrix.

        Raises:
            PortfolioValidationError: If ``optimizer_name`` is invalid, the
                optimizer is unknown, ``signals`` is invalid, or the optimizer
                output fails portfolio-schema finalization.
        """
        validated_name = _require_optimizer_name(optimizer_name)
        optimizer = self._optimizer_registry.get(validated_name)
        frame = validate_signals_frame(signals)
        optimized = optimizer.optimize(frame)
        return _finalize(optimized)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting.

    Args:
        frame: Frame produced by ``PortfolioOptimizer.optimize``.

    Returns:
        Finalized merged portfolio DataFrame.

    Raises:
        PortfolioValidationError: If ``frame`` is not a non-empty DataFrame,
            required columns are missing, or primary keys are duplicated.
    """
    if not isinstance(frame, pl.DataFrame):
        raise PortfolioValidationError(
            "optimizer output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise PortfolioValidationError(
            "optimizer output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_portfolio_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_PORTFOLIO_SCHEMA)


def _require_optimizer_name(name: object) -> str:
    """Validate and return a non-blank optimizer name.

    Args:
        name: Candidate optimizer registry key.

    Returns:
        The validated name string.

    Raises:
        PortfolioValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise PortfolioValidationError(
            "optimizer_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"optimizer_name": name},
        )
    return name


def _require_portfolio_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing.

    Args:
        frame: Candidate portfolio DataFrame.

    Raises:
        PortfolioValidationError: If one or more ``REQUIRED_COLUMNS`` are
            absent.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PortfolioValidationError(
            "merged portfolio schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``.

    Args:
        frame: DataFrame containing primary-key columns.

    Raises:
        PortfolioValidationError: If any primary-key combination appears more
            than once.
    """
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise PortfolioValidationError(
            "portfolio frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )

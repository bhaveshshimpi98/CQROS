"""CQROS Pyramiding Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical open-position inputs
    into canonical pyramiding recommendation datasets through registered
    ``PyramidingEngine`` implementations.

Responsibilities:
    - Validate engine names and resolve engines from ``PyramidingRegistry``
    - Validate canonical input DataFrame structure
    - Delegate pyramiding evaluation exclusively to an injected engine
    - Validate required pyramiding schema columns on the engine output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged pyramiding schema
    - Preserve input-frame immutability
    - Remain free of pyramiding algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.pyramiding.engine``, ``cqros.pyramiding.exceptions``,
    ``cqros.pyramiding.registry``, and ``cqros.pyramiding.schema``.

Public API:
    ``PyramidingPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.pyramiding.engine import (
    validate_accounting_frame,
    validate_market_price_frame,
    validate_portfolio_risk_frame,
    validate_position_frame,
    validate_trade_management_frame,
)
from cqros.pyramiding.exceptions import PyramidingValidationError
from cqros.pyramiding.registry import PyramidingRegistry
from cqros.pyramiding.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_PYRAMIDING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["PyramidingPipeline"]

_ERROR_NAME_BLANK: Final[str] = "PYR_PIPE_NAME_BLANK"
_ERROR_MANAGER_BLANK: Final[str] = "PYR_PIPE_MANAGER_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "PYR_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "PYR_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "PYR_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "PYR_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_DEFAULT_ENGINE: Final[str] = "simple"


class PyramidingPipeline:
    """Deterministic orchestrator for canonical pyramiding assembly.

    The pipeline resolves a registered ``PyramidingEngine``, validates
    canonical input DataFrames, delegates evaluation, and finalizes the
    result to the canonical merged pyramiding schema. Pyramiding semantics
    remain exclusively in the engine. The caller-supplied frames are never
    mutated.

    Args:
        registry: Registry used to resolve pyramiding-engine implementations.
    """

    __slots__ = ("_registry",)

    _registry: PyramidingRegistry

    def __init__(self, registry: PyramidingRegistry) -> None:
        """Initialize the pipeline with a pyramiding engine registry.

        Args:
            registry: Registry containing ``PyramidingEngine`` implementations.
        """
        self._registry = registry

    def run(
        self,
        positions: pl.DataFrame,
        accounting: pl.DataFrame,
        portfolio_risk: pl.DataFrame,
        trade_management: pl.DataFrame,
        market_prices: pl.DataFrame,
        *,
        manager: str,
        engine_name: str = _DEFAULT_ENGINE,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized pyramiding frame.

        ``engine_name`` is validated and resolved first. ``manager`` is
        validated as a non-blank lineage identity. Input frames are validated
        through their respective frame validators. Evaluation is then
        delegated to ``PyramidingEngine.evaluate``. The engine output is
        checked against ``REQUIRED_COLUMNS``, rejected when primary keys are
        duplicated, reordered to ``CANONICAL_COLUMN_ORDER``, and cast to
        ``MERGED_PYRAMIDING_SCHEMA``. The original frames are never mutated.

        Args:
            positions: Canonical position dataset.
            accounting: Canonical accounting dataset.
            portfolio_risk: Canonical portfolio-risk dataset.
            trade_management: Canonical trade-management dataset.
            market_prices: Market price dataset keyed by open_time.
            manager: Order manager identity stamped onto every decision row
                and used for partition lineage.
            engine_name: Registry key of the pyramiding engine to execute.
                Defaults to ``simple``.

        Returns:
            A new DataFrame containing the finalized merged pyramiding matrix.

        Raises:
            PyramidingValidationError: If ``engine_name`` is invalid, the
                engine is unknown, ``manager`` is blank, inputs are invalid,
                or the engine output fails schema finalization.
        """
        validated_name = _require_engine_name(engine_name)
        validated_manager = _require_manager(manager)
        engine = self._registry.get(validated_name)
        positions_frame = validate_position_frame(positions)
        accounting_frame = validate_accounting_frame(accounting)
        risk_frame = validate_portfolio_risk_frame(portfolio_risk)
        tm_frame = validate_trade_management_frame(trade_management)
        prices_frame = validate_market_price_frame(market_prices)
        created = engine.evaluate(
            positions_frame,
            accounting_frame,
            risk_frame,
            tm_frame,
            prices_frame,
            manager=validated_manager,
        )
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise PyramidingValidationError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise PyramidingValidationError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_pyramiding_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_PYRAMIDING_SCHEMA)


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise PyramidingValidationError(
            "engine_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine_name": name},
        )
    return name


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise PyramidingValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_pyramiding_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PyramidingValidationError(
            "merged pyramiding schema is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Raise when primary-key combinations are duplicated in ``frame``."""
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise PyramidingValidationError(
            "pyramiding frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )

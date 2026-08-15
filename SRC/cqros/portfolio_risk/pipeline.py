"""CQROS Portfolio Risk Manager package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical accounting datasets into
    canonical portfolio risk decision datasets through registered
    ``PortfolioRiskManager`` implementations.

Responsibilities:
    - Validate manager names and resolve managers from
      ``PortfolioRiskManagerRegistry``
    - Validate canonical accounting and position DataFrame structure
    - Delegate portfolio-risk evaluation exclusively to an injected manager
    - Validate required portfolio-risk schema columns on the manager output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged portfolio-risk schema
    - Preserve accounting-frame and position-frame immutability
    - Remain free of portfolio-risk algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.portfolio_risk.manager``,
    ``cqros.portfolio_risk.exceptions``, ``cqros.portfolio_risk.registry``,
    and ``cqros.portfolio_risk.schema``.

Public API:
    ``PortfolioRiskPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.portfolio_risk.exceptions import PortfolioRiskValidationError
from cqros.portfolio_risk.manager import (
    validate_accounting_frame,
    validate_position_frame,
)
from cqros.portfolio_risk.registry import PortfolioRiskManagerRegistry
from cqros.portfolio_risk.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_PORTFOLIO_RISK_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["PortfolioRiskPipeline"]

_ERROR_NAME_BLANK: Final[str] = "PRISK_PIPE_NAME_BLANK"
_ERROR_MANAGER_BLANK: Final[str] = "PRISK_PIPE_MANAGER_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "PRISK_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "PRISK_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "PRISK_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "PRISK_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_DEFAULT_RISK_MANAGER: Final[str] = "simple"


class PortfolioRiskPipeline:
    """Deterministic orchestrator for canonical portfolio-risk dataset assembly.

    The pipeline resolves a registered ``PortfolioRiskManager``, validates
    canonical accounting and position DataFrames, delegates evaluation, and
    finalizes the result to the canonical merged portfolio-risk schema.
    Portfolio-risk semantics remain exclusively in the manager. The
    caller-supplied frames are never mutated.

    Args:
        manager_registry: Registry used to resolve portfolio-risk-manager
            implementations.
    """

    __slots__ = ("_manager_registry",)

    _manager_registry: PortfolioRiskManagerRegistry

    def __init__(self, manager_registry: PortfolioRiskManagerRegistry) -> None:
        """Initialize the pipeline with a portfolio risk manager registry.

        Args:
            manager_registry: Registry containing ``PortfolioRiskManager``
                implementations.
        """
        self._manager_registry = manager_registry

    def run(
        self,
        accounting: pl.DataFrame,
        positions: pl.DataFrame,
        *,
        manager: str,
        risk_manager_name: str = _DEFAULT_RISK_MANAGER,
    ) -> pl.DataFrame:
        """Resolve a manager and produce a finalized portfolio-risk frame.

        ``risk_manager_name`` is validated and resolved first. ``manager`` is
        validated as a non-blank lineage identity. ``accounting`` and
        ``positions`` are validated through their respective frame validators.
        Evaluation is then delegated to ``PortfolioRiskManager.evaluate``. The
        manager output is checked against ``REQUIRED_COLUMNS``, rejected when
        primary keys are duplicated, reordered to ``CANONICAL_COLUMN_ORDER``,
        and cast to ``MERGED_PORTFOLIO_RISK_SCHEMA``. The original frames are
        never mutated.

        Args:
            accounting: Canonical accounting dataset.
            positions: Canonical position dataset.
            manager: Order manager identity stamped onto every risk row and
                used for partition lineage.
            risk_manager_name: Registry key of the portfolio risk manager to
                execute. Defaults to ``simple``.

        Returns:
            A new DataFrame containing the finalized merged portfolio-risk
            matrix.

        Raises:
            PortfolioRiskValidationError: If ``risk_manager_name`` is invalid,
                the manager is unknown, ``manager`` is blank, inputs are
                invalid, or the manager output fails schema finalization.
        """
        validated_name = _require_risk_manager_name(risk_manager_name)
        validated_manager = _require_manager(manager)
        risk_manager = self._manager_registry.get(validated_name)
        accounting_frame = validate_accounting_frame(accounting)
        positions_frame = validate_position_frame(positions)
        created = risk_manager.evaluate(
            accounting_frame,
            positions_frame,
            manager=validated_manager,
        )
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise PortfolioRiskValidationError(
            "manager output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise PortfolioRiskValidationError(
            "manager output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_portfolio_risk_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_PORTFOLIO_RISK_SCHEMA)


def _require_risk_manager_name(name: object) -> str:
    """Validate and return a non-blank risk-manager name."""
    if not isinstance(name, str) or name.strip() == "":
        raise PortfolioRiskValidationError(
            "risk_manager_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"risk_manager_name": name},
        )
    return name


def _require_manager(manager: object) -> str:
    """Validate and return a non-blank manager identity."""
    if not isinstance(manager, str) or manager.strip() == "":
        raise PortfolioRiskValidationError(
            "manager must be a non-blank string",
            error_code=_ERROR_MANAGER_BLANK,
            details={"manager": manager},
        )
    return manager


def _require_portfolio_risk_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PortfolioRiskValidationError(
            "merged portfolio-risk schema is missing required columns",
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
        raise PortfolioRiskValidationError(
            "portfolio-risk frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )

"""CQROS Risk Management package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical portfolio datasets into
    canonical risk-decision datasets through registered ``RiskManager``
    implementations.

Responsibilities:
    - Validate policy names and resolve managers from ``RiskPolicyRegistry``
    - Validate canonical portfolio DataFrame structure
    - Delegate evaluation exclusively to an injected risk manager
    - Validate required Risk Decision schema columns on the manager output
    - Reject duplicate primary keys
    - Finalize outputs against the canonical merged risk schema
    - Preserve portfolio-frame immutability
    - Remain free of risk calculations, persistence, verification, and CLI
      logic

Dependencies:
    ``polars``, ``cqros.risk.exceptions``, ``cqros.risk.interfaces``,
    ``cqros.risk.registry``, and ``cqros.risk.schema``.

Public API:
    ``RiskPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.risk.exceptions import RiskValidationError
from cqros.risk.interfaces import validate_portfolio_frame
from cqros.risk.registry import RiskPolicyRegistry
from cqros.risk.schema import (
    CANONICAL_COLUMN_ORDER,
    MERGED_RISK_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["RiskPipeline"]

_ERROR_NAME_BLANK: Final[str] = "RISK_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "RISK_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "RISK_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "RISK_PIPE_MISSING_COLUMNS"
_ERROR_DUPLICATE_KEYS: Final[str] = "RISK_PIPE_DUPLICATE_KEYS"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class RiskPipeline:
    """Deterministic orchestrator for canonical risk-decision dataset assembly.

    The pipeline resolves a registered ``RiskManager``, validates a canonical
    portfolio DataFrame, delegates evaluation, and finalizes the result to the
    canonical merged risk schema. Risk semantics remain exclusively in the
    manager. The caller-supplied portfolio frame is never mutated.

    Args:
        policy_registry: Registry used to resolve risk-manager implementations.
    """

    __slots__ = ("_policy_registry",)

    _policy_registry: RiskPolicyRegistry

    def __init__(self, policy_registry: RiskPolicyRegistry) -> None:
        """Initialize the pipeline with a risk policy registry.

        Args:
            policy_registry: Registry containing ``RiskManager``
                implementations.
        """
        self._policy_registry = policy_registry

    def run(self, policy_name: str, portfolios: pl.DataFrame) -> pl.DataFrame:
        """Resolve a risk manager and produce a finalized risk-decision frame.

        ``policy_name`` is validated and resolved first. ``portfolios`` is
        validated through ``validate_portfolio_frame``. Evaluation is then
        delegated to ``RiskManager.evaluate``. The manager output is checked
        against ``REQUIRED_COLUMNS``, rejected when primary keys are
        duplicated, reordered to ``CANONICAL_COLUMN_ORDER``, and cast to
        ``MERGED_RISK_SCHEMA``. The original ``portfolios`` frame is never
        mutated.

        Args:
            policy_name: Registry key of the risk manager to execute.
            portfolios: Canonical portfolio dataset.

        Returns:
            A new DataFrame containing the finalized merged risk-decision
            matrix.

        Raises:
            RiskValidationError: If ``policy_name`` is invalid, the policy is
                unknown, ``portfolios`` is invalid, or the manager output fails
                risk-schema finalization.
        """
        validated_name = _require_policy_name(policy_name)
        policy = self._policy_registry.get(validated_name)
        frame = validate_portfolio_frame(portfolios)
        evaluated = policy.evaluate(frame)
        return _finalize(evaluated)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting.

    Args:
        frame: Frame produced by ``RiskManager.evaluate``.

    Returns:
        Finalized merged risk-decision DataFrame.

    Raises:
        RiskValidationError: If ``frame`` is not a non-empty DataFrame,
            required columns are missing, or primary keys are duplicated.
    """
    if not isinstance(frame, pl.DataFrame):
        raise RiskValidationError(
            "policy output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise RiskValidationError(
            "policy output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_risk_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(MERGED_RISK_SCHEMA)


def _require_policy_name(name: object) -> str:
    """Validate and return a non-blank policy name.

    Args:
        name: Candidate policy registry key.

    Returns:
        The validated name string.

    Raises:
        RiskValidationError: If ``name`` is not a non-blank string.
    """
    if not isinstance(name, str) or name.strip() == "":
        raise RiskValidationError(
            "policy_name must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"policy_name": name},
        )
    return name


def _require_risk_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required merged-schema column is missing.

    Args:
        frame: Candidate risk-decision DataFrame.

    Raises:
        RiskValidationError: If one or more ``REQUIRED_COLUMNS`` are absent.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise RiskValidationError(
            "merged risk schema is missing required columns",
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
        RiskValidationError: If any primary-key combination appears more than
            once.
    """
    unique_keys = frame.select(_PRIMARY_KEY_LIST).n_unique()
    if unique_keys != frame.height:
        raise RiskValidationError(
            "risk frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )

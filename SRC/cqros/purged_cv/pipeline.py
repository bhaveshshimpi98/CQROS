"""CQROS Purged Cross Validation Engine package pipeline.

Purpose:
    Orchestrate deterministic conversion of canonical Walk-Forward
    datasets into canonical purged cross-validation datasets through
    registered ``PurgedCVEngine`` implementations.

Responsibilities:
    - Validate engine names and resolve engines from
      ``PurgedCVEngineRegistry``
    - Validate canonical Walk-Forward DataFrame structure
    - Delegate purged-CV-row assembly exclusively to an injected engine
    - Validate required purged-CV schema columns on the engine output
    - Reject missing or duplicate primary keys
    - Finalize outputs against ``PURGED_CV_SCHEMA``
    - Preserve Walk-Forward-frame immutability
    - Remain free of purge algorithms, persistence, verification,
      exchange APIs, and CLI logic

Dependencies:
    ``polars``, ``cqros.purged_cv.engine``,
    ``cqros.purged_cv.exceptions``,
    ``cqros.purged_cv.registry``, and
    ``cqros.purged_cv.schema``.

Public API:
    ``PurgedCVPipeline``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.purged_cv.engine import SimplePurgedCVEngine, validate_walk_forward_frame
from cqros.purged_cv.exceptions import PurgedCVError
from cqros.purged_cv.registry import PurgedCVEngineRegistry
from cqros.purged_cv.schema import (
    CANONICAL_COLUMN_ORDER,
    PRIMARY_KEY_COLUMNS,
    PURGED_CV_SCHEMA,
    REQUIRED_COLUMNS,
)

__all__ = ["PurgedCVPipeline"]

_ERROR_NAME_BLANK: Final[str] = "PCV_PIPE_NAME_BLANK"
_ERROR_INVALID_OUTPUT: Final[str] = "PCV_PIPE_INVALID_OUTPUT"
_ERROR_OUTPUT_EMPTY: Final[str] = "PCV_PIPE_OUTPUT_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "PCV_PIPE_MISSING_COLUMNS"
_ERROR_MISSING_PRIMARY_KEYS: Final[str] = "PCV_PIPE_MISSING_PRIMARY_KEYS"
_ERROR_DUPLICATE_KEYS: Final[str] = "PCV_PIPE_DUPLICATE_KEYS"
_ERROR_SCHEMA_CAST: Final[str] = "PCV_PIPE_SCHEMA_CAST"

_DEFAULT_ENGINE_NAME: Final[str] = "simple"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)


class PurgedCVPipeline:
    """Deterministic orchestrator for canonical purged-CV assembly.

    The pipeline resolves a registered ``PurgedCVEngine``, validates a
    canonical Walk-Forward DataFrame, delegates purged-CV-row generation,
    and finalizes the result to ``PURGED_CV_SCHEMA``. Purge and embargo
    semantics remain exclusively in the engine. The caller-supplied
    Walk-Forward frame is never mutated.

    Args:
        registry: Registry used to resolve purged-CV-engine implementations.
            When ``None``, a registry containing ``SimplePurgedCVEngine``
            under ``\"simple\"`` is created.
    """

    __slots__ = ("_registry",)

    _registry: PurgedCVEngineRegistry

    def __init__(self, registry: PurgedCVEngineRegistry | None = None) -> None:
        """Initialize the pipeline with a purged-CV engine registry.

        Args:
            registry: Registry containing ``PurgedCVEngine`` implementations.
                When ``None``, registers ``SimplePurgedCVEngine`` as
                ``\"simple\"``.
        """
        if registry is None:
            registry = PurgedCVEngineRegistry()
            registry.register(_DEFAULT_ENGINE_NAME, SimplePurgedCVEngine())
        self._registry = registry

    def build(
        self,
        frame: pl.DataFrame,
        *,
        engine: str = _DEFAULT_ENGINE_NAME,
        **kwargs: object,
    ) -> pl.DataFrame:
        """Resolve an engine and produce a finalized purged-CV frame.

        ``engine`` is validated and resolved first. ``frame`` is validated
        through ``validate_walk_forward_frame``. Purged-CV generation is then
        delegated to ``PurgedCVEngine.build``. The engine output is checked
        against ``REQUIRED_COLUMNS`` / ``CANONICAL_COLUMN_ORDER``, rejected
        when primary keys are missing or duplicated, reordered to
        ``CANONICAL_COLUMN_ORDER``, and cast to ``PURGED_CV_SCHEMA``.
        The original Walk-Forward frame is never mutated.

        Args:
            frame: Canonical Walk-Forward dataset.
            engine: Registry key of the purged-CV engine to execute.
                Defaults to ``\"simple\"``.
            **kwargs: Reserved for forward-compatible orchestration options.
                Ignored by this implementation.

        Returns:
            A new DataFrame containing the finalized purged-CV rows.

        Raises:
            PurgedCVError: If ``engine`` is invalid, the engine is unknown,
                ``frame`` is invalid, or the engine output fails
                purged-CV-schema finalization.
        """
        del kwargs  # Reserved; orchestration does not forward engine options.
        validated_name = _require_engine_name(engine)
        resolved = self._registry.get(validated_name)
        validated_frame = validate_walk_forward_frame(frame)
        created = resolved.build(validated_frame)
        return _finalize(created)


def _finalize(frame: object) -> pl.DataFrame:
    """Apply schema checks, uniqueness checks, ordering, and casting."""
    if not isinstance(frame, pl.DataFrame):
        raise PurgedCVError(
            "engine output must be a polars DataFrame",
            error_code=_ERROR_INVALID_OUTPUT,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise PurgedCVError(
            "engine output must contain at least one row",
            error_code=_ERROR_OUTPUT_EMPTY,
            details={"rows": frame.height},
        )
    _require_primary_key_columns(frame)
    _require_purged_cv_schema_columns(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    try:
        return ordered.cast(PURGED_CV_SCHEMA)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as exc:
        raise PurgedCVError(
            "engine output cannot be cast to PURGED_CV_SCHEMA",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc


def _require_engine_name(name: object) -> str:
    """Validate and return a non-blank engine name."""
    if not isinstance(name, str) or name.strip() == "":
        raise PurgedCVError(
            "engine must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"engine": name},
        )
    return name


def _require_purged_cv_schema_columns(frame: pl.DataFrame) -> None:
    """Raise when any required purged-CV-schema column is missing."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise PurgedCVError(
            "purged cv schema is missing required columns",
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
        raise PurgedCVError(
            "purged cv frame is missing primary key columns",
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
        raise PurgedCVError(
            "purged cv frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )

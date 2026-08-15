"""CQROS Factor Research Engine wide-to-long factor transformer.

Purpose:
    Convert a wide factor matrix produced by ``FactorPipeline.run`` into a
    long-format DataFrame that conforms to ``FACTOR_SCHEMA``, enriching each
    factor row from an injected metadata map.

Responsibilities:
    - Validate primary-key columns on the wide input frame
    - Detect duplicate factor column names
    - Unpivot every non-primary-key column into ``factor_name`` /
      ``factor_value`` rows using vectorized Polars operations
    - Enrich rows from a caller-supplied ``Mapping[str, FactorMetadata]``
    - Fail fast when metadata is missing for any factor column
    - Cast and validate the result against ``FACTOR_SCHEMA``
    - Remain free of registry access, factor execution, persistence, CLI,
      storage, and research orchestration

Dependencies:
    ``polars``, ``cqros.factors.exceptions``, ``cqros.factors.metadata``,
    and ``cqros.factors.schema``.

Public API:
    ``WideToLongFactorTransformer``
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Final

import polars as pl

from cqros.factors.exceptions import FactorValidationError
from cqros.factors.metadata import FactorMetadata
from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
)

__all__ = ["WideToLongFactorTransformer"]

_ERROR_MISSING_PRIMARY_KEY: Final[str] = "FACTOR-W2L-001"
_ERROR_DUPLICATE_COLUMNS: Final[str] = "FACTOR-W2L-002"
_ERROR_MISSING_METADATA: Final[str] = "FACTOR-W2L-003"
_ERROR_METADATA_NAME_MISMATCH: Final[str] = "FACTOR-W2L-004"
_ERROR_SCHEMA_CAST: Final[str] = "FACTOR-W2L-005"
_ERROR_FRAME_TYPE: Final[str] = "FACTOR-W2L-006"

_PRIMARY_KEY_LIST: Final[list[str]] = list(PRIMARY_KEY_COLUMNS)

_logger = logging.getLogger(__name__)


class WideToLongFactorTransformer:
    """Convert a wide factor matrix into a ``FACTOR_SCHEMA`` long frame.

    The transformer is registry-independent. Callers inject a metadata map
    keyed by ``factor_name``. Metadata values are never fabricated.

    Args:
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_logger",)

    _logger: logging.Logger

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize the transformer.

        Args:
            logger: Optional logger instance.
        """
        self._logger = logger if logger is not None else _logger

    def transform(
        self,
        frame: pl.DataFrame,
        metadata: Mapping[str, FactorMetadata],
    ) -> pl.DataFrame:
        """Unpivot ``frame`` and enrich rows into ``FACTOR_SCHEMA``.

        Args:
            frame: Wide factor matrix with ``PRIMARY_KEY_COLUMNS`` plus one
                column per generated factor.
            metadata: Mapping from ``factor_name`` to ``FactorMetadata``.
                Every factor column in ``frame`` must have an entry.

        Returns:
            A new DataFrame conforming to ``FACTOR_SCHEMA``. One input row
            with ``N`` factor columns expands to ``N`` output rows.

        Raises:
            FactorValidationError: If ``frame`` is not a DataFrame, primary
                keys are missing, factor columns are duplicated, metadata is
                missing or mismatched, or the result fails ``FACTOR_SCHEMA``
                validation.
        """
        validated = _require_dataframe(frame)
        _require_unique_columns(validated)
        _require_primary_key_columns(validated)
        factor_columns = _factor_columns(validated)
        _require_metadata_coverage(factor_columns, metadata)

        self._logger.info(
            "Transforming wide factor matrix to FACTOR_SCHEMA",
            extra={
                "row_count": validated.height,
                "factor_column_count": len(factor_columns),
            },
        )

        if not factor_columns:
            empty = pl.DataFrame(schema=FACTOR_SCHEMA)
            self._logger.info(
                "Wide-to-long transform completed with no factor columns",
                extra={"row_count": 0, "factor_column_count": 0},
            )
            return empty

        long_frame = validated.unpivot(
            index=_PRIMARY_KEY_LIST,
            on=factor_columns,
            variable_name="factor_name",
            value_name="factor_value",
        )
        meta_frame = _metadata_frame(factor_columns, metadata)
        enriched = long_frame.join(meta_frame, on="factor_name", how="left")
        result = _require_factor_schema(enriched)

        self._logger.info(
            "Wide-to-long transform completed",
            extra={
                "row_count": result.height,
                "factor_column_count": len(factor_columns),
                "column_count": result.width,
            },
        )
        return result


def _require_dataframe(frame: object) -> pl.DataFrame:
    """Raise when ``frame`` is not a Polars DataFrame.

    Raises:
        FactorValidationError: If ``frame`` is not a ``pl.DataFrame``.
    """
    if not isinstance(frame, pl.DataFrame):
        raise FactorValidationError(
            "wide factor frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    return frame


def _require_unique_columns(frame: pl.DataFrame) -> None:
    """Raise when the wide frame contains duplicate column names.

    Raises:
        FactorValidationError: If any column name appears more than once.
    """
    columns = list(frame.columns)
    if len(columns) == len(set(columns)):
        return
    duplicates = tuple(sorted({name for name in columns if columns.count(name) > 1}))
    raise FactorValidationError(
        "wide factor frame contains duplicate column names",
        error_code=_ERROR_DUPLICATE_COLUMNS,
        details={
            "duplicate_columns": duplicates,
            "available_columns": tuple(columns),
        },
    )


def _require_primary_key_columns(frame: pl.DataFrame) -> None:
    """Raise when any factor-matrix primary-key column is missing.

    Raises:
        FactorValidationError: If one or more ``PRIMARY_KEY_COLUMNS`` are absent.
    """
    missing = [column for column in PRIMARY_KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            "wide factor frame is missing primary key columns",
            error_code=_ERROR_MISSING_PRIMARY_KEY,
            details={
                "missing_columns": tuple(missing),
                "required_columns": PRIMARY_KEY_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _factor_columns(frame: pl.DataFrame) -> list[str]:
    """Return non-primary-key columns in frame order."""
    primary = set(PRIMARY_KEY_COLUMNS)
    return [column for column in frame.columns if column not in primary]


def _require_metadata_coverage(
    factor_columns: Sequence[str],
    metadata: Mapping[str, FactorMetadata],
) -> None:
    """Raise when any factor column lacks matching metadata.

    Raises:
        FactorValidationError: If metadata is missing or ``name`` mismatches
            the map key.
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

    mismatched = tuple(column for column in factor_columns if metadata[column].name != column)
    if mismatched:
        raise FactorValidationError(
            "factor metadata name does not match map key",
            error_code=_ERROR_METADATA_NAME_MISMATCH,
            details={
                "mismatched_factor_names": mismatched,
                "expected_names": {column: metadata[column].name for column in mismatched},
            },
        )


def _metadata_frame(
    factor_columns: Sequence[str],
    metadata: Mapping[str, FactorMetadata],
) -> pl.DataFrame:
    """Build a compact metadata frame for vectorized join enrichment.

    Loops over factor columns only (catalog scale), never over data rows.
    """
    return pl.DataFrame(
        {
            "factor_name": list(factor_columns),
            "factor_version": [metadata[column].version for column in factor_columns],
            "factor_category": [metadata[column].category for column in factor_columns],
            "factor_group": [metadata[column].factor_group for column in factor_columns],
            "lookback": [metadata[column].lookback for column in factor_columns],
            "prediction_horizon": [
                metadata[column].prediction_horizon for column in factor_columns
            ],
            "enabled": [metadata[column].enabled for column in factor_columns],
            "status": [metadata[column].status.value for column in factor_columns],
        }
    )


def _require_factor_schema(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate and cast ``frame`` to ``FACTOR_SCHEMA``.

    Raises:
        FactorValidationError: If required columns are missing or casting fails.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise FactorValidationError(
            "transformed factors frame is missing required columns",
            error_code=_ERROR_SCHEMA_CAST,
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
            "transformed factors frame failed FACTOR_SCHEMA cast",
            error_code=_ERROR_SCHEMA_CAST,
            details={"reason": str(exc)},
        ) from exc

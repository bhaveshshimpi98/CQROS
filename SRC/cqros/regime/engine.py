"""CQROS Regime Engine contracts and implementation.

Purpose:
    Convert an Alpha dataset into a deterministic regime DataFrame
    conforming to ``REGIME_SCHEMA``.

Responsibilities:
    - Define ``RegimeEngine`` as the shared regime-generation contract
    - Provide ``SimpleRegimeEngine`` for placeholder regime-row assembly
    - Validate Alpha DataFrame structure
    - Restrict participation to rows where ``status`` is ``PASS`` and
      ``alpha_score`` is finite and non-null
    - Emit one deterministic regime row per surviving alpha row
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.regime.exceptions``, and ``cqros.regime.schema``.

Public API:
    ``RegimeEngine``, ``SimpleRegimeEngine``, ``REGIME_INPUT_COLUMNS``,
    ``validate_alpha_frame``
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.regime.exceptions import RegimeError
from cqros.regime.schema import CANONICAL_COLUMN_ORDER, REGIME_SCHEMA, RegimeStatus

__all__ = [
    "REGIME_INPUT_COLUMNS",
    "RegimeEngine",
    "SimpleRegimeEngine",
    "validate_alpha_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "REGIME_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "REGIME_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "REGIME_MISSING_COLUMNS"
_ERROR_NO_ALPHA: Final[str] = "REGIME_NO_ALPHA"

_ID_SEPARATOR: Final[str] = "|"
_PASS_STATUS: Final[str] = "PASS"
_REGIME_PROBABILITY: Final[float] = 1.0
_REGIME_TYPE: Final[str] = "UNKNOWN"
_REGIME_VERSION: Final[str] = "1.0.0"

# Alpha columns required to assemble a regime row.
REGIME_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "factor_set_id",
    "alpha_model",
    "alpha_version",
    "symbol",
    "timeframe",
    "prediction_time",
    "alpha_score",
    "status",
)


@runtime_checkable
class RegimeEngine(Protocol):
    """Structural contract for converting alpha into regime rows.

    Implementations own regime-generation semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, alpha: pl.DataFrame) -> pl.DataFrame:
        """Convert an Alpha dataset into regime rows.

        Args:
            alpha: Canonical Alpha dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``REGIME_SCHEMA``.
        """
        ...


class SimpleRegimeEngine:
    """Generate deterministic placeholder regime rows from alpha.

    Only rows with ``status == PASS`` and a finite non-null ``alpha_score``
    participate. Every surviving alpha row receives one regime row with
    ``regime_type = UNKNOWN``, ``regime_probability = 1.0``,
    ``regime_score = alpha_score``, and ``regime_version = 1.0.0``. No
    machine learning, HMM, clustering, or statistical regime detection is
    performed in this placeholder implementation; future engines may replace
    this implementation without changing the pipeline contract.

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        Output is fully deterministic for identical inputs.
        ``prediction_score`` is not part of the Alpha → Regime contract.
    """

    __slots__ = ()

    def build(self, alpha: pl.DataFrame) -> pl.DataFrame:
        """Convert an Alpha dataset into regime rows.

        Args:
            alpha: Canonical Alpha dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``REGIME_SCHEMA``.

        Raises:
            RegimeError: If the input fails structural validation, required
                columns are missing, or no PASS rows with finite non-null
                ``alpha_score`` remain for regime generation.
        """
        frame = validate_alpha_frame(alpha)
        _require_columns(frame, REGIME_INPUT_COLUMNS, "alpha")
        return _build_regime_rows(frame)


def validate_alpha_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate Alpha dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        RegimeError: If ``frame`` is not a Polars DataFrame or contains no
            rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise RegimeError(
            "alpha frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={
                "dataset": "alpha",
                "actual_type": type(frame).__name__,
            },
        )
    if frame.height == 0:
        raise RegimeError(
            "alpha frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "alpha", "rows": frame.height},
        )
    return frame


def _build_regime_rows(alpha: pl.DataFrame) -> pl.DataFrame:
    """Assemble canonical regime rows from Alpha."""
    surviving = alpha.filter(
        (pl.col("status") == _PASS_STATUS)
        & pl.col("alpha_score").is_not_null()
        & pl.col("alpha_score").is_finite()
    )
    if surviving.height == 0:
        raise RegimeError(
            "alpha frame contains no PASS rows with finite non-null alpha_score",
            error_code=_ERROR_NO_ALPHA,
            details={
                "dataset": "alpha",
                "rows": alpha.height,
                "surviving_rows": 0,
            },
        )

    pass_status = RegimeStatus.PASS.value
    regime_id = (
        pl.col("factor_set_id")
        + pl.lit(_ID_SEPARATOR)
        + pl.col("alpha_model")
        + pl.lit(_ID_SEPARATOR)
        + pl.col("alpha_version")
    )

    scored = surviving.with_columns(
        regime_id.alias("regime_id"),
        pl.col("factor_set_id"),
        regime_id.alias("alpha_id"),
        pl.col("symbol"),
        pl.col("timeframe"),
        pl.col("prediction_time").cast(pl.Datetime("ms")).alias("regime_time"),
        pl.lit(_REGIME_TYPE).alias("regime_type"),
        pl.lit(_REGIME_PROBABILITY).cast(pl.Float64).alias("regime_probability"),
        pl.col("alpha_score").cast(pl.Float64).alias("regime_score"),
        pl.lit(_REGIME_VERSION).alias("regime_version"),
        pl.lit(pass_status).alias("status"),
        pl.lit([], dtype=pl.List(pl.String)).alias("metadata"),
    )

    return scored.select(list(CANONICAL_COLUMN_ORDER)).cast(REGIME_SCHEMA)


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RegimeError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )

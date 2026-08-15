"""CQROS Models Engine contracts and implementation.

Purpose:
    Convert a Regime dataset into a deterministic models DataFrame
    conforming to ``MODELS_SCHEMA`` (Research Model Ledger).

Responsibilities:
    - Define ``ModelEngine`` as the shared model-generation contract
    - Provide ``SimpleModelEngine`` for placeholder model-row assembly
    - Validate Regime DataFrame structure
    - Restrict participation to rows where ``status`` is ``PASS`` and
      ``regime_score`` is finite and non-null
    - Map ``regime_score`` onto ``validation_score`` (ledger model score)
    - Preserve regime context in ``model_metadata`` without inventing
      predictive performance
    - Emit one deterministic model row per surviving regime row
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.models.exceptions``, and ``cqros.models.schema``.

Public API:
    ``ModelEngine``, ``SimpleModelEngine``, ``MODEL_INPUT_COLUMNS``,
    ``validate_regime_frame``

Notes:
    ``regime_probability`` is contextual classification confidence and is
    never used as the ledger model score. ``validation_score`` is the
    established MODELS_SCHEMA field that carries ``regime_score``.
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.models.exceptions import ModelError
from cqros.models.schema import CANONICAL_COLUMN_ORDER, MODELS_SCHEMA, ModelStatus

__all__ = [
    "MODEL_INPUT_COLUMNS",
    "ModelEngine",
    "SimpleModelEngine",
    "validate_regime_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "MODEL_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "MODEL_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "MODEL_MISSING_COLUMNS"
_ERROR_NO_REGIMES: Final[str] = "MODEL_NO_REGIMES"

_ID_SEPARATOR: Final[str] = "|"
_MODEL_TYPE: Final[str] = "baseline"
_MODEL_VERSION: Final[str] = "1.0.0"
_PASS_STATUS: Final[str] = "PASS"
_PREDICTION_HORIZON: Final[int] = 1

# Canonical Regime columns required to assemble a Research Model Ledger row.
MODEL_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "regime_id",
    "factor_set_id",
    "alpha_id",
    "symbol",
    "timeframe",
    "regime_time",
    "regime_type",
    "regime_probability",
    "regime_score",
    "regime_version",
    "status",
)


@runtime_checkable
class ModelEngine(Protocol):
    """Structural contract for converting regime into model rows.

    Implementations own model-generation semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, regime: pl.DataFrame) -> pl.DataFrame:
        """Convert a Regime dataset into model rows.

        Args:
            regime: Canonical Regime dataset. Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``MODELS_SCHEMA``.
        """
        ...


class SimpleModelEngine:
    """Generate deterministic Research Model Ledger rows from regime.

    Only rows with ``status == PASS`` and a finite non-null ``regime_score``
    participate. Every surviving regime row receives one ledger row with
    ``model_type = baseline``, ``model_version = 1.0.0``,
    ``prediction_horizon = 1``, and ``validation_score = regime_score``.
    ``regime_probability`` is preserved in ``model_metadata`` and is never
    used as the model score. No machine learning, fitting, inference, or
    optimization is performed in this placeholder implementation; future
    engines may replace this implementation without changing the pipeline
    contract.

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        Output is fully deterministic for identical inputs.
    """

    __slots__ = ()

    def build(self, regime: pl.DataFrame) -> pl.DataFrame:
        """Convert a Regime dataset into model rows.

        Args:
            regime: Canonical Regime dataset. Must not be mutated.

        Returns:
            A new DataFrame matching ``MODELS_SCHEMA``.

        Raises:
            ModelError: If the input fails structural validation, required
                columns are missing, or no PASS rows with finite non-null
                ``regime_score`` remain for model generation.
        """
        frame = validate_regime_frame(regime)
        _require_columns(frame, MODEL_INPUT_COLUMNS, "regime")
        return _build_model_rows(frame)


def validate_regime_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate Regime dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        ModelError: If ``frame`` is not a Polars DataFrame or contains no
            rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise ModelError(
            "regime frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={
                "dataset": "regime",
                "actual_type": type(frame).__name__,
            },
        )
    if frame.height == 0:
        raise ModelError(
            "regime frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "regime", "rows": frame.height},
        )
    return frame


def _build_model_rows(regime: pl.DataFrame) -> pl.DataFrame:
    """Assemble canonical Research Model Ledger rows from Regime."""
    surviving = regime.filter(
        (pl.col("status") == _PASS_STATUS)
        & pl.col("regime_score").is_not_null()
        & pl.col("regime_score").is_finite()
    )
    if surviving.height == 0:
        raise ModelError(
            "regime frame contains no PASS rows with finite non-null regime_score",
            error_code=_ERROR_NO_REGIMES,
            details={
                "dataset": "regime",
                "rows": regime.height,
                "surviving_rows": 0,
            },
        )

    pass_status = ModelStatus.PASS.value
    metadata = pl.concat_list(
        [
            pl.lit("factor_set_id=") + pl.col("factor_set_id").cast(pl.String),
            pl.lit("alpha_id=") + pl.col("alpha_id").cast(pl.String),
            pl.lit("regime_type=") + pl.col("regime_type").cast(pl.String),
            pl.lit("regime_probability=") + pl.col("regime_probability").cast(pl.String),
            pl.lit("regime_version=") + pl.col("regime_version").cast(pl.String),
        ]
    )

    scored = surviving.with_columns(
        (pl.col("regime_id") + pl.lit(_ID_SEPARATOR) + pl.lit(_MODEL_TYPE)).alias("model_id"),
        pl.col("regime_id"),
        pl.col("symbol"),
        pl.col("timeframe"),
        pl.col("regime_time").cast(pl.Int64).alias("training_time"),
        pl.lit(_MODEL_TYPE).alias("model_type"),
        pl.lit(_MODEL_VERSION).alias("model_version"),
        pl.lit(_PREDICTION_HORIZON).cast(pl.Int32).alias("prediction_horizon"),
        pl.col("regime_score").cast(pl.Float64).alias("validation_score"),
        pl.col("alpha_id").alias("feature_set_id"),
        metadata.alias("model_metadata"),
        pl.lit(pass_status).alias("status"),
    )

    return scored.select(list(CANONICAL_COLUMN_ORDER)).cast(MODELS_SCHEMA)


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ModelError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )

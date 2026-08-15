"""CQROS Walk-Forward evaluation-result schemas.

Purpose:
    Define the canonical columnar contracts for Walk-Forward evaluation
    artifacts that persist Labels ``future_return_1`` against fold windows
    without mutating the Walk-Forward ledger.

Responsibilities:
    - Declare observation-level evaluation-result columns and dtypes
    - Declare fold-level, factor-level, and panel-summary metric columns
    - Document metrics that remain null under ``SimpleWalkForwardEngine``
    - Remain free of fold math, persistence, CLI, and trading logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    Observation, fold, factor, and summary schema constants, partition
    and status enumerations, and unavailable-metric documentation.

Notes:
    ``SimpleWalkForwardEngine`` evaluates the selected-factor evaluation
    stream; it does not emit a predictive model forecast. Therefore
    ``prediction``, ``residual``, and ``correct`` are always null in the
    observation artifact. Annualized ``oos_sharpe`` and equity-curve
    ``oos_max_drawdown`` are also null: annualization and drawdown
    conventions live in the performance layer, and Walk-Forward evaluation
    must not invent a conflicting bar-frequency definition or import that
    upward dependency.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "EVALUATION_FACTOR_METRIC_COLUMNS",
    "EVALUATION_FACTOR_METRIC_SCHEMA",
    "EVALUATION_FOLD_METRIC_COLUMNS",
    "EVALUATION_FOLD_METRIC_SCHEMA",
    "EVALUATION_OBSERVATION_COLUMNS",
    "EVALUATION_OBSERVATION_SCHEMA",
    "EVALUATION_PARTITION_VALUES",
    "EVALUATION_SUMMARY_COLUMNS",
    "EVALUATION_SUMMARY_SCHEMA",
    "FACTOR_METRIC_DTYPES",
    "FOLD_METRIC_DTYPES",
    "OBSERVATION_DTYPES",
    "OBSERVATION_PRIMARY_KEY_COLUMNS",
    "SUMMARY_DTYPES",
    "UNAVAILABLE_METRIC_NOTES",
    "WalkForwardEvaluationPartition",
    "WalkForwardEvaluationStatus",
    "evaluation_partition_values",
    "evaluation_status_values",
]

# Metrics intentionally left null with explanation (not failures).
UNAVAILABLE_METRIC_NOTES: Final[MappingProxyType[str, str]] = MappingProxyType(
    {
        "prediction": (
            "SimpleWalkForwardEngine evaluates the selected-factor stream and "
            "does not produce a predictive forecast; prediction is null."
        ),
        "residual": ("residual requires prediction; left null because no forecast exists."),
        "correct": (
            "correct/hit-rate requires prediction; left null because no forecast " "exists."
        ),
        "oos_sharpe": (
            "Annualized Sharpe conventions live in the performance layer "
            "(_periods_per_year from equity timestamps). Walk-Forward evaluation "
            "does not invent a bar-frequency annualization factor and does not "
            "import performance (upward dependency)."
        ),
        "oos_max_drawdown": (
            "Project max_drawdown is defined on equity curves in the performance "
            "layer. Walk-Forward evaluation must not invent a conflicting "
            "return-stream drawdown definition."
        ),
    }
)


class WalkForwardEvaluationPartition(str, Enum):  # noqa: UP042
    """Train versus out-of-sample designation for an evaluation observation."""

    TRAIN = "TRAIN"
    OOS = "OOS"


class WalkForwardEvaluationStatus(str, Enum):  # noqa: UP042
    """Status for an evaluation observation or aggregate row."""

    PASS = "PASS"
    FAIL = "FAIL"


def evaluation_partition_values() -> tuple[str, ...]:
    """Return every evaluation partition string value."""
    return (
        WalkForwardEvaluationPartition.TRAIN.value,
        WalkForwardEvaluationPartition.OOS.value,
    )


def evaluation_status_values() -> tuple[str, ...]:
    """Return every evaluation status string value."""
    return (
        WalkForwardEvaluationStatus.PASS.value,
        WalkForwardEvaluationStatus.FAIL.value,
    )


EVALUATION_PARTITION_VALUES: Final[tuple[str, ...]] = evaluation_partition_values()

OBSERVATION_PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "engine",
    "timeframe",
    "year",
    "fold_id",
    "symbol",
    "observation_time",
    "factor_name",
    "factor_version",
    "partition",
)

EVALUATION_OBSERVATION_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "engine",
    "symbol",
    "timeframe",
    "year",
    "fold_id",
    "observation_time",
    "factor_name",
    "factor_version",
    "selected",
    "partition",
    "future_return_1",
    "factor_value",
    "selection_ic",
    "selected_direction",
    "orientation_policy",
    "prediction",
    "residual",
    "correct",
    "status",
)

OBSERVATION_DTYPES: Final = MappingProxyType(
    {
        "manager": pl.String,
        "engine": pl.String,
        "symbol": pl.String,
        "timeframe": pl.String,
        "year": pl.Int32,
        "fold_id": pl.Int32,
        "observation_time": pl.Int64,
        "factor_name": pl.String,
        "factor_version": pl.String,
        "selected": pl.Boolean,
        "partition": pl.String,
        "future_return_1": pl.Float64,
        "factor_value": pl.Float64,
        "selection_ic": pl.Float64,
        "selected_direction": pl.Int8,
        "orientation_policy": pl.String,
        "prediction": pl.Float64,
        "residual": pl.Float64,
        "correct": pl.Boolean,
        "status": pl.String,
    }
)

EVALUATION_OBSERVATION_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, OBSERVATION_DTYPES[column]) for column in EVALUATION_OBSERVATION_COLUMNS]
)

EVALUATION_FOLD_METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "engine",
    "symbol",
    "timeframe",
    "year",
    "fold_id",
    "train_rows",
    "oos_rows",
    "oos_non_null_returns",
    "oos_return_mean",
    "oos_return_std",
    "oos_return_min",
    "oos_return_max",
    "oos_positive_rate",
    "oos_cumulative_return",
    "oos_sharpe",
    "oos_max_drawdown",
    "status",
)

FOLD_METRIC_DTYPES: Final = MappingProxyType(
    {
        "manager": pl.String,
        "engine": pl.String,
        "symbol": pl.String,
        "timeframe": pl.String,
        "year": pl.Int32,
        "fold_id": pl.Int32,
        "train_rows": pl.Int64,
        "oos_rows": pl.Int64,
        "oos_non_null_returns": pl.Int64,
        "oos_return_mean": pl.Float64,
        "oos_return_std": pl.Float64,
        "oos_return_min": pl.Float64,
        "oos_return_max": pl.Float64,
        "oos_positive_rate": pl.Float64,
        "oos_cumulative_return": pl.Float64,
        "oos_sharpe": pl.Float64,
        "oos_max_drawdown": pl.Float64,
        "status": pl.String,
    }
)

EVALUATION_FOLD_METRIC_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, FOLD_METRIC_DTYPES[column]) for column in EVALUATION_FOLD_METRIC_COLUMNS]
)

EVALUATION_FACTOR_METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "engine",
    "symbol",
    "timeframe",
    "year",
    "factor_name",
    "factor_version",
    "fold_id",
    "train_rows",
    "oos_rows",
    "oos_return_mean",
    "oos_return_std",
    "oos_positive_rate",
    "oos_cumulative_return",
    "raw_oos_ic",
    "oriented_oos_ic",
    "oos_ic",
    "selection_ic",
    "selected_direction",
    "orientation_policy",
)

FACTOR_METRIC_DTYPES: Final = MappingProxyType(
    {
        "manager": pl.String,
        "engine": pl.String,
        "symbol": pl.String,
        "timeframe": pl.String,
        "year": pl.Int32,
        "factor_name": pl.String,
        "factor_version": pl.String,
        "fold_id": pl.Int32,
        "train_rows": pl.Int64,
        "oos_rows": pl.Int64,
        "oos_return_mean": pl.Float64,
        "oos_return_std": pl.Float64,
        "oos_positive_rate": pl.Float64,
        "oos_cumulative_return": pl.Float64,
        "raw_oos_ic": pl.Float64,
        "oriented_oos_ic": pl.Float64,
        "oos_ic": pl.Float64,
        "selection_ic": pl.Float64,
        "selected_direction": pl.Int8,
        "orientation_policy": pl.String,
    }
)

EVALUATION_FACTOR_METRIC_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, FACTOR_METRIC_DTYPES[column]) for column in EVALUATION_FACTOR_METRIC_COLUMNS]
)

EVALUATION_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "engine",
    "symbol",
    "timeframe",
    "year",
    "folds",
    "train_rows",
    "oos_rows",
    "oos_non_null_returns",
    "oos_return_mean",
    "oos_return_std",
    "oos_positive_rate",
    "oos_cumulative_return",
    "oos_sharpe",
    "oos_max_drawdown",
    "unique_selected_factors",
    "status",
    "error",
)

SUMMARY_DTYPES: Final = MappingProxyType(
    {
        "manager": pl.String,
        "engine": pl.String,
        "symbol": pl.String,
        "timeframe": pl.String,
        "year": pl.Int32,
        "folds": pl.Int64,
        "train_rows": pl.Int64,
        "oos_rows": pl.Int64,
        "oos_non_null_returns": pl.Int64,
        "oos_return_mean": pl.Float64,
        "oos_return_std": pl.Float64,
        "oos_positive_rate": pl.Float64,
        "oos_cumulative_return": pl.Float64,
        "oos_sharpe": pl.Float64,
        "oos_max_drawdown": pl.Float64,
        "unique_selected_factors": pl.Int64,
        "status": pl.String,
        "error": pl.String,
    }
)

EVALUATION_SUMMARY_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, SUMMARY_DTYPES[column]) for column in EVALUATION_SUMMARY_COLUMNS]
)

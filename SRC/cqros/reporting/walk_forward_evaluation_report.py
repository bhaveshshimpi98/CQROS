"""CQROS Walk-Forward evaluation CSV reporter.

Purpose:
    Emit deterministic CSV reports from Walk-Forward evaluation artifacts
    without mutating lake parquet files.

Responsibilities:
    - Write all / folds / factors / global summary CSVs
    - Aggregate global metrics across panel summaries
    - Remain free of evaluation fold math and parquet mutation

Dependencies:
    ``polars``, ``cqros.core``, and walk-forward evaluation schemas.

Public API:
    ``WalkForwardEvaluationReporter``, report column constants, and
    ``DEFAULT_OUTPUT_ROOT``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import FILE_EXTENSION_CSV
from cqros.walk_forward.evaluation_schema import (
    EVALUATION_FACTOR_METRIC_COLUMNS,
    EVALUATION_FOLD_METRIC_COLUMNS,
    EVALUATION_SUMMARY_COLUMNS,
    UNAVAILABLE_METRIC_NOTES,
)

__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "EVALUATION_ALL_CSV_NAME",
    "EVALUATION_FACTORS_CSV_NAME",
    "EVALUATION_FOLDS_CSV_NAME",
    "EVALUATION_GLOBAL_CSV_NAME",
    "GLOBAL_SUMMARY_COLUMNS",
    "WalkForwardEvaluationReporter",
]

DEFAULT_OUTPUT_ROOT: Final[Path] = Path("reports") / "walk_forward"

EVALUATION_ALL_CSV_NAME: Final[str] = f"walk_forward_evaluation_all{FILE_EXTENSION_CSV}"
EVALUATION_FOLDS_CSV_NAME: Final[str] = f"walk_forward_evaluation_folds{FILE_EXTENSION_CSV}"
EVALUATION_FACTORS_CSV_NAME: Final[str] = f"walk_forward_evaluation_factors{FILE_EXTENSION_CSV}"
EVALUATION_GLOBAL_CSV_NAME: Final[str] = f"walk_forward_evaluation_global{FILE_EXTENSION_CSV}"

GLOBAL_SUMMARY_COLUMNS: Final[tuple[str, ...]] = ("metric", "value")


class WalkForwardEvaluationReporter:
    """Write Walk-Forward evaluation CSV reports to an output directory."""

    __slots__ = ("_output_root",)

    def __init__(self, output_root: Path | None = None) -> None:
        """Initialize the reporter.

        Args:
            output_root: Directory receiving CSV reports. Defaults to
                ``reports/walk_forward``.
        """
        self._output_root = output_root if output_root is not None else DEFAULT_OUTPUT_ROOT

    @property
    def output_root(self) -> Path:
        """Return the configured report output directory."""
        return self._output_root

    def write_reports(
        self,
        *,
        summaries: pl.DataFrame,
        fold_metrics: pl.DataFrame,
        factor_metrics: pl.DataFrame,
    ) -> dict[str, Path]:
        """Persist the four evaluation CSV reports.

        Args:
            summaries: Panel summary rows (one per evaluated partition).
            fold_metrics: Fold-level metric rows.
            factor_metrics: Factor-level metric rows.

        Returns:
            Mapping of report label to written path.
        """
        self._output_root.mkdir(parents=True, exist_ok=True)
        all_path = self._output_root / EVALUATION_ALL_CSV_NAME
        folds_path = self._output_root / EVALUATION_FOLDS_CSV_NAME
        factors_path = self._output_root / EVALUATION_FACTORS_CSV_NAME
        global_path = self._output_root / EVALUATION_GLOBAL_CSV_NAME

        summary_frame = _ensure_columns(summaries, EVALUATION_SUMMARY_COLUMNS)
        fold_frame = _ensure_columns(fold_metrics, EVALUATION_FOLD_METRIC_COLUMNS)
        factor_frame = _ensure_columns(factor_metrics, EVALUATION_FACTOR_METRIC_COLUMNS)
        global_frame = build_global_summary(summary_frame)

        summary_frame.write_csv(all_path)
        fold_frame.write_csv(folds_path)
        factor_frame.write_csv(factors_path)
        global_frame.write_csv(global_path)
        return {
            "all": all_path,
            "folds": folds_path,
            "factors": factors_path,
            "global": global_path,
        }


def build_global_summary(summaries: pl.DataFrame) -> pl.DataFrame:
    """Build a metric/value global summary across panel summaries."""
    if summaries.height == 0:
        rows: list[dict[str, object]] = [
            {"metric": "panels", "value": "0"},
            {"metric": "status", "value": "FAIL"},
            {
                "metric": "unavailable_oos_sharpe",
                "value": UNAVAILABLE_METRIC_NOTES["oos_sharpe"],
            },
            {
                "metric": "unavailable_oos_max_drawdown",
                "value": UNAVAILABLE_METRIC_NOTES["oos_max_drawdown"],
            },
            {
                "metric": "unavailable_prediction",
                "value": UNAVAILABLE_METRIC_NOTES["prediction"],
            },
        ]
        return pl.DataFrame(rows).select(list(GLOBAL_SUMMARY_COLUMNS))

    folds = int(summaries["folds"].sum())
    train_rows = int(summaries["train_rows"].sum())
    oos_rows = int(summaries["oos_rows"].sum())
    oos_non_null = int(summaries["oos_non_null_returns"].sum())
    unique_factors = int(summaries["unique_selected_factors"].sum())
    statuses = summaries["status"].to_list()
    status = "PASS" if statuses and all(item == "PASS" for item in statuses) else "FAIL"
    mean_return = summaries["oos_return_mean"].drop_nulls().mean()
    rows = [
        {"metric": "panels", "value": str(summaries.height)},
        {"metric": "folds", "value": str(folds)},
        {"metric": "train_rows", "value": str(train_rows)},
        {"metric": "oos_rows", "value": str(oos_rows)},
        {"metric": "oos_non_null_returns", "value": str(oos_non_null)},
        {
            "metric": "oos_return_mean",
            "value": "" if mean_return is None else f"{float(mean_return):.12g}",
        },
        {"metric": "unique_selected_factors", "value": str(unique_factors)},
        {"metric": "oos_sharpe", "value": ""},
        {"metric": "oos_max_drawdown", "value": ""},
        {"metric": "status", "value": status},
        {
            "metric": "unavailable_oos_sharpe",
            "value": UNAVAILABLE_METRIC_NOTES["oos_sharpe"],
        },
        {
            "metric": "unavailable_oos_max_drawdown",
            "value": UNAVAILABLE_METRIC_NOTES["oos_max_drawdown"],
        },
        {
            "metric": "unavailable_prediction",
            "value": UNAVAILABLE_METRIC_NOTES["prediction"],
        },
    ]
    return pl.DataFrame(rows).select(list(GLOBAL_SUMMARY_COLUMNS))


def _ensure_columns(frame: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Select report columns in canonical order, filling missing with null."""
    if frame.height == 0:
        return pl.DataFrame({column: [] for column in columns})
    expressions: list[pl.Expr] = []
    for column in columns:
        if column in frame.columns:
            expressions.append(pl.col(column))
        else:
            expressions.append(pl.lit(None).alias(column))
    return frame.select(expressions)

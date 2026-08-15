"""CQROS factor orientation diagnostic CSV reporter.

Purpose:
    Emit deterministic orientation diagnostic summaries for Factor Selection,
    Walk-Forward evaluation, and Purged-CV evaluation without mutating lake
    parquet artifacts.

Responsibilities:
    - Aggregate selected-direction counts and mean raw / oriented IC metrics
    - Write timeframe-year orientation summary CSVs under the existing report
      hierarchy
    - Remain free of selection math, fold construction, and upward ML imports

Dependencies:
    ``polars``, ``cqros.core.constants``, and the Python standard library.

Public API:
    ``FactorOrientationReporter``, column constants, and summary builders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import FILE_EXTENSION_CSV

__all__ = [
    "FACTOR_ORIENTATION_SUMMARY_COLUMNS",
    "FACTOR_ORIENTATION_SUMMARY_CSV_NAME",
    "FactorOrientationReporter",
    "ORIENTATION_FACTOR_DETAIL_COLUMNS",
    "build_factor_orientation_details",
    "build_orientation_summary",
]

DEFAULT_FACTOR_SELECTION_ROOT: Final[Path] = Path("reports") / "factor_selection"
DEFAULT_WALK_FORWARD_ROOT: Final[Path] = Path("reports") / "walk_forward"
DEFAULT_PURGED_CV_ROOT: Final[Path] = Path("reports") / "purged_cv"

FACTOR_ORIENTATION_SUMMARY_CSV_NAME: Final[str] = f"factor_orientation_summary{FILE_EXTENSION_CSV}"
FACTOR_ORIENTATION_DETAIL_CSV_NAME: Final[str] = f"factor_orientation_factors{FILE_EXTENSION_CSV}"

FACTOR_ORIENTATION_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "selected_factor_count",
    "positive_direction_count",
    "negative_direction_count",
    "mean_raw_selection_ic",
    "mean_oriented_selection_ic",
    "mean_raw_oos_ic",
    "mean_oriented_oos_ic",
)

ORIENTATION_FACTOR_DETAIL_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "exchange",
    "market",
    "timeframe",
    "year",
    "factor_name",
    "factor_version",
    "fold_id",
    "selection_ic",
    "selected_direction",
    "orientation_policy",
    "oriented_selection_ic",
    "raw_oos_ic",
    "oriented_oos_ic",
)

_SORT_SUMMARY: Final[tuple[str, ...]] = ("timeframe", "year")
_SORT_DETAIL: Final[tuple[str, ...]] = (
    "manager",
    "exchange",
    "market",
    "timeframe",
    "year",
    "factor_name",
    "factor_version",
    "fold_id",
)


class FactorOrientationReporter:
    """Write orientation diagnostic CSVs for one report root."""

    __slots__ = ("_output_root",)

    def __init__(self, output_root: Path) -> None:
        """Initialize the reporter.

        Args:
            output_root: Directory receiving orientation CSV reports.
        """
        self._output_root = output_root

    @property
    def output_root(self) -> Path:
        """Return the configured report output directory."""
        return self._output_root

    def write_reports(
        self,
        *,
        summary: pl.DataFrame,
        factor_details: pl.DataFrame | None = None,
    ) -> dict[str, Path]:
        """Persist orientation summary and optional factor-detail CSVs."""
        self._output_root.mkdir(parents=True, exist_ok=True)
        summary_path = self._output_root / FACTOR_ORIENTATION_SUMMARY_CSV_NAME
        summary.select(list(FACTOR_ORIENTATION_SUMMARY_COLUMNS)).write_csv(summary_path)
        written: dict[str, Path] = {"summary": summary_path}
        if factor_details is not None:
            detail_path = self._output_root / FACTOR_ORIENTATION_DETAIL_CSV_NAME
            factor_details.select(list(ORIENTATION_FACTOR_DETAIL_COLUMNS)).write_csv(detail_path)
            written["factors"] = detail_path
        return written


def build_orientation_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """Build timeframe/year orientation summary rows from factor metrics.

    Expected optional columns:
        ``timeframe``, ``year``, ``selected_direction``, ``selection_ic``,
        ``raw_oos_ic``, ``oriented_oos_ic``.

    Missing optional metric columns yield null means rather than fabricated
    values.
    """
    empty_schema = {
        "timeframe": pl.String,
        "year": pl.Int32,
        "selected_factor_count": pl.Int64,
        "positive_direction_count": pl.Int64,
        "negative_direction_count": pl.Int64,
        "mean_raw_selection_ic": pl.Float64,
        "mean_oriented_selection_ic": pl.Float64,
        "mean_raw_oos_ic": pl.Float64,
        "mean_oriented_oos_ic": pl.Float64,
    }
    if frame.height == 0:
        return pl.DataFrame(schema=empty_schema)

    working = frame
    if "selected_direction" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Int8).alias("selected_direction"))
    if "selection_ic" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Float64).alias("selection_ic"))
    if "raw_oos_ic" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Float64).alias("raw_oos_ic"))
    if "oriented_oos_ic" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Float64).alias("oriented_oos_ic"))
    if "year" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Int32).alias("year"))

    oriented_selection = pl.col("selection_ic") * pl.col("selected_direction").cast(pl.Float64)
    finite_raw_oos = (
        pl.when(pl.col("raw_oos_ic").is_nan()).then(None).otherwise(pl.col("raw_oos_ic"))
    )
    finite_oriented_oos = (
        pl.when(pl.col("oriented_oos_ic").is_nan()).then(None).otherwise(pl.col("oriented_oos_ic"))
    )

    # Direction / selection-IC counts are unique selected factors. OOS IC means
    # average all factor-fold metric rows (Walk-Forward may have many folds).
    has_factor_keys = {"factor_name", "factor_version"}.issubset(working.columns)
    if has_factor_keys:
        factors = (
            working.select(
                [
                    "timeframe",
                    "year",
                    "factor_name",
                    "factor_version",
                    "selection_ic",
                    "selected_direction",
                ]
            )
            .sort(["timeframe", "year", "factor_name", "factor_version"])
            .unique(
                subset=["timeframe", "year", "factor_name", "factor_version"],
                keep="first",
                maintain_order=True,
            )
        )
        direction_summary = (
            factors.group_by(["timeframe", "year"], maintain_order=True)
            .agg(
                [
                    pl.len().alias("selected_factor_count"),
                    (pl.col("selected_direction") == 1).sum().alias("positive_direction_count"),
                    (pl.col("selected_direction") == -1).sum().alias("negative_direction_count"),
                    pl.col("selection_ic").mean().alias("mean_raw_selection_ic"),
                    oriented_selection.mean().alias("mean_oriented_selection_ic"),
                ]
            )
            .sort(list(_SORT_SUMMARY))
        )
        oos_summary = (
            working.sort(["timeframe", "year", "factor_name", "factor_version"])
            .group_by(["timeframe", "year"], maintain_order=True)
            .agg(
                [
                    finite_raw_oos.mean().alias("mean_raw_oos_ic"),
                    finite_oriented_oos.mean().alias("mean_oriented_oos_ic"),
                ]
            )
            .sort(list(_SORT_SUMMARY))
        )
        return (
            direction_summary.join(oos_summary, on=["timeframe", "year"], how="left")
            .select(list(FACTOR_ORIENTATION_SUMMARY_COLUMNS))
            .sort(list(_SORT_SUMMARY))
        )

    return (
        working.sort(["timeframe", "year"])
        .group_by(["timeframe", "year"], maintain_order=True)
        .agg(
            [
                pl.len().alias("selected_factor_count"),
                (pl.col("selected_direction") == 1).sum().alias("positive_direction_count"),
                (pl.col("selected_direction") == -1).sum().alias("negative_direction_count"),
                pl.col("selection_ic").mean().alias("mean_raw_selection_ic"),
                oriented_selection.mean().alias("mean_oriented_selection_ic"),
                finite_raw_oos.mean().alias("mean_raw_oos_ic"),
                finite_oriented_oos.mean().alias("mean_oriented_oos_ic"),
            ]
        )
        .select(list(FACTOR_ORIENTATION_SUMMARY_COLUMNS))
        .sort(list(_SORT_SUMMARY))
    )


def build_factor_orientation_details(
    frame: pl.DataFrame,
    *,
    manager: str = "",
    exchange: str = "",
    market: str = "",
) -> pl.DataFrame:
    """Normalize factor-level orientation detail rows for CSV export."""
    if frame.height == 0:
        return pl.DataFrame(
            schema={column: pl.String for column in ORIENTATION_FACTOR_DETAIL_COLUMNS}
        ).clear()

    working = frame
    defaults: dict[str, object] = {
        "manager": manager,
        "exchange": exchange,
        "market": market,
        "fold_id": None,
        "selection_ic": None,
        "selected_direction": None,
        "orientation_policy": None,
        "raw_oos_ic": None,
        "oriented_oos_ic": None,
    }
    for column, value in defaults.items():
        if column not in working.columns:
            if column in {"fold_id", "selected_direction"}:
                working = working.with_columns(pl.lit(value, dtype=pl.Int32).alias(column))
            elif column in {"selection_ic", "raw_oos_ic", "oriented_oos_ic"}:
                working = working.with_columns(pl.lit(value, dtype=pl.Float64).alias(column))
            else:
                working = working.with_columns(pl.lit(value, dtype=pl.String).alias(column))
    if "year" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Int32).alias("year"))
    if "timeframe" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.String).alias("timeframe"))
    if "factor_name" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.String).alias("factor_name"))
    if "factor_version" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.String).alias("factor_version"))

    return (
        working.with_columns(
            (pl.col("selection_ic") * pl.col("selected_direction").cast(pl.Float64)).alias(
                "oriented_selection_ic"
            )
        )
        .select(list(ORIENTATION_FACTOR_DETAIL_COLUMNS))
        .sort(list(_SORT_DETAIL))
    )

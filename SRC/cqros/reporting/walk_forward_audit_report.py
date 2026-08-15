"""CQROS Walk-Forward consolidated CSV audit reporter.

Purpose:
    Provide a read-only reporting layer that discovers production Walk-Forward
    parquet partitions, aggregates audit metrics, and emits deterministic CSV
    reports without mutating lake artifacts or the Walk-Forward engine.

Responsibilities:
    - Discover manager/timeframe/year partitions under ``walk_forward``
    - Load raw parquet schemas without casting or silent repair
    - Aggregate per-symbol (when present), timeframe, year, manager, engine
    - Emit detail, timeframe-summary, and global-summary CSV reports
    - Record schema and integrity failures in ``status`` / ``error``
    - Remain free of Walk-Forward fold math, Alpha/Regime/Predictions/
      Signals/ML imports, and parquet mutation

Dependencies:
    ``hashlib``, ``logging``, ``polars``, ``cqros.core``,
    ``cqros.reporting.exceptions``, and ``cqros.storage``.

Public API:
    ``WalkForwardAuditReporter``, ``DETAIL_COLUMNS``,
    ``TIMEFRAME_SUMMARY_COLUMNS``, ``DEFAULT_OUTPUT_ROOT``,
    ``ALL_TIMEFRAMES_CSV_NAME``, ``TIMEFRAME_SUMMARY_CSV_NAME``,
    ``GLOBAL_SUMMARY_CSV_NAME``, ``aggregate_partition_frame``,
    ``build_timeframe_summary``, ``build_global_summary``,
    ``format_discovery_table``, ``forbidden_import_violations``.

Notes:
    Missing evaluation-input columns (for example ``future_return_1``,
    ``selected``, ``factor_name``) are never fabricated. Metrics that
    require absent columns are reported as null and the partition row is
    marked ``FAIL`` with a descriptive ``error``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    FILE_EXTENSION_CSV,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_WALK_FORWARD,
)
from cqros.core.types import Exchange, Market, Timeframe
from cqros.reporting.exceptions import ReportingValidationError
from cqros.storage.layout import StorageLayout
from cqros.walk_forward.schema import PRIMARY_KEY_COLUMNS, WalkForwardStatus

__all__ = [
    "ALL_TIMEFRAMES_CSV_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "DETAIL_COLUMNS",
    "GLOBAL_SUMMARY_CSV_NAME",
    "TIMEFRAME_SUMMARY_COLUMNS",
    "TIMEFRAME_SUMMARY_CSV_NAME",
    "WalkForwardAuditReporter",
    "aggregate_partition_frame",
    "build_global_summary",
    "build_timeframe_summary",
    "forbidden_import_violations",
    "format_discovery_table",
]

_logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT: Final[Path] = Path("reports") / "walk_forward"

ALL_TIMEFRAMES_CSV_NAME: Final[str] = f"walk_forward_all_timeframes{FILE_EXTENSION_CSV}"
TIMEFRAME_SUMMARY_CSV_NAME: Final[str] = f"walk_forward_timeframe_summary{FILE_EXTENSION_CSV}"
GLOBAL_SUMMARY_CSV_NAME: Final[str] = f"walk_forward_global_summary{FILE_EXTENSION_CSV}"

DETAIL_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "engine",
    "symbol",
    "timeframe",
    "year",
    "rows",
    "selected_rows",
    "pass_rows",
    "fail_rows",
    "unique_factors",
    "unique_factor_versions",
    "unique_folds",
    "train_rows",
    "oos_rows",
    "first_open_time",
    "last_open_time",
    "first_selection_time",
    "last_selection_time",
    "future_return_1_non_null",
    "future_return_1_null",
    "future_return_1_mean",
    "future_return_1_std",
    "future_return_1_min",
    "future_return_1_max",
    "oos_future_return_mean",
    "oos_future_return_std",
    "oos_future_return_min",
    "oos_future_return_max",
    "selected_factor_memberships",
    "status",
    "error",
)

TIMEFRAME_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "years",
    "symbols",
    "rows",
    "selected_rows",
    "pass_rows",
    "fail_rows",
    "unique_factors",
    "unique_factor_versions",
    "unique_folds",
    "future_return_1_non_null",
    "future_return_1_mean",
    "future_return_1_std",
    "oos_rows",
    "oos_future_return_mean",
    "oos_future_return_std",
    "status",
)

_STATUS_PASS: Final[str] = "PASS"
_STATUS_FAIL: Final[str] = "FAIL"

_FOLD_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "fold_id",
    "fold",
    "fold_index",
    "wf_fold",
)

_PARTITION_COLUMN_CANDIDATES: Final[tuple[str, ...]] = (
    "is_oos",
    "partition",
    "split",
    "set",
    "sample",
)

_OOS_PARTITION_VALUES: Final[frozenset[str]] = frozenset(
    {"oos", "test", "out_of_sample", "out-of-sample"}
)
_TRAIN_PARTITION_VALUES: Final[frozenset[str]] = frozenset(
    {"train", "training", "in_sample", "in-sample"}
)

_FORBIDDEN_IMPORT_MODULES: Final[tuple[str, ...]] = (
    "cqros.alpha",
    "cqros.regime",
    "cqros.predictions",
    "cqros.signals",
    "cqros.ml",
    "cqros.models",
)

_ERROR_NO_PARTITIONS: Final[str] = "REPORT-WALK-FORWARD-AUDIT-001"
_ERROR_OUTPUT_DIR: Final[str] = "REPORT-WALK-FORWARD-AUDIT-002"
_ERROR_ENGINE: Final[str] = "REPORT-WALK-FORWARD-AUDIT-003"

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL


@dataclass(frozen=True, slots=True)
class WalkForwardAuditPaths:
    """Output paths for the consolidated Walk-Forward audit CSVs."""

    detail: Path
    timeframe_summary: Path
    global_summary: Path


@dataclass(frozen=True, slots=True)
class WalkForwardAuditResult:
    """Artifacts produced by a Walk-Forward audit run."""

    detail: pl.DataFrame
    timeframe_summary: pl.DataFrame
    global_summary: pl.DataFrame
    paths: WalkForwardAuditPaths
    parquet_hashes_before: Mapping[str, str]
    parquet_hashes_after: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _PartitionRef:
    """Minimal discovered partition identity for audit reporting."""

    manager: str
    timeframe: Timeframe
    year: int


class WalkForwardAuditReporter:
    """Read-only consolidated Walk-Forward CSV audit reporter.

    Discovers partitions under ``walk_forward``, aggregates audit metrics from
    the raw parquet bytes, and writes deterministic CSV reports. Never writes
    parquet and never imports Alpha/Regime/Predictions/Signals/ML packages.

    Args:
        layout: Canonical path composer for the data lake.
        output_dir: Directory that receives the three CSV reports.
        engine: Engine label recorded in the detail CSV. Walk-Forward
            partitions do not persist engine identity; callers supply the
            label used for lineage in the report only.
        exchange: Exchange identifier for discovery.
        market: Market segment for discovery.
        logger: Optional logger instance.
    """

    __slots__ = (
        "_engine",
        "_exchange",
        "_layout",
        "_logger",
        "_market",
        "_output_dir",
    )

    _layout: StorageLayout
    _output_dir: Path
    _engine: str
    _exchange: Exchange
    _market: Market
    _logger: logging.Logger

    def __init__(
        self,
        layout: StorageLayout,
        *,
        output_dir: Path = DEFAULT_OUTPUT_ROOT,
        engine: str = "unknown",
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the reporter with injected layout and output path."""
        if not engine.strip():
            raise ReportingValidationError(
                "engine label must be a non-empty string",
                error_code=_ERROR_ENGINE,
                details={"engine": engine},
            )
        self._layout = layout
        self._output_dir = Path(output_dir)
        self._engine = engine.strip()
        self._exchange = exchange
        self._market = market
        self._logger = logger if logger is not None else _logger

    def collect(self) -> tuple[pl.DataFrame, tuple[Path, ...], Mapping[str, str]]:
        """Discover partitions and aggregate the detail audit frame.

        Returns:
            ``(detail_frame, parquet_paths, hashes_before)``.

        Raises:
            ReportingValidationError: If no partitions exist.
        """
        partitions = self._discover_partitions()
        if not partitions:
            raise ReportingValidationError(
                "no Walk-Forward partitions discovered under storage root",
                error_code=_ERROR_NO_PARTITIONS,
                details={
                    "storage_root": str(self._layout.root),
                    "tier": STORAGE_DIR_WALK_FORWARD,
                },
            )

        paths_before = tuple(path for _, path in partitions)
        hashes_before = _hash_files(paths_before)

        detail_rows: list[dict[str, object]] = []
        for ref, path in partitions:
            frame = pl.read_parquet(path)
            detail_rows.extend(
                aggregate_partition_frame(
                    frame,
                    manager=ref.manager,
                    engine=self._engine,
                    timeframe=ref.timeframe,
                    year=ref.year,
                )
            )
        return _finalize_detail_frame(detail_rows), paths_before, hashes_before

    def emit(
        self,
        detail: pl.DataFrame,
        parquet_paths: Sequence[Path],
        hashes_before: Mapping[str, str],
    ) -> WalkForwardAuditResult:
        """Write CSV reports for an already aggregated detail frame."""
        timeframe_summary = build_timeframe_summary(detail)
        global_summary = build_global_summary(detail)
        output_paths = self._write_reports(detail, timeframe_summary, global_summary)
        hashes_after = _hash_files(parquet_paths)
        if dict(hashes_after) != dict(hashes_before):
            raise ReportingValidationError(
                "Walk-Forward production parquet hashes changed during audit",
                error_code=_ERROR_OUTPUT_DIR,
                details={
                    "before": dict(hashes_before),
                    "after": dict(hashes_after),
                },
            )
        self._logger.info(
            "Wrote Walk-Forward audit reports",
            extra={
                "detail_rows": detail.height,
                "timeframes": detail.select("timeframe").n_unique(),
                "output_dir": str(self._output_dir),
            },
        )
        return WalkForwardAuditResult(
            detail=detail,
            timeframe_summary=timeframe_summary,
            global_summary=global_summary,
            paths=output_paths,
            parquet_hashes_before=dict(hashes_before),
            parquet_hashes_after=hashes_after,
        )

    def run(self) -> WalkForwardAuditResult:
        """Discover partitions, aggregate metrics, and write CSV reports."""
        detail, parquet_paths, hashes_before = self.collect()
        return self.emit(detail, parquet_paths, hashes_before)

    def _discover_partitions(self) -> tuple[tuple[_PartitionRef, Path], ...]:
        """Discover Walk-Forward year partitions and their absolute paths."""
        root = self._layout.root / STORAGE_DIR_WALK_FORWARD
        if not root.is_dir():
            return ()

        items: list[tuple[_PartitionRef, Path]] = []
        for manager_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            exchange_dir = manager_dir / self._exchange / self._market
            if not exchange_dir.is_dir():
                continue
            for timeframe_dir in sorted(path for path in exchange_dir.iterdir() if path.is_dir()):
                for parquet_path in sorted(timeframe_dir.glob("*.parquet")):
                    year = _parse_year_stem(parquet_path.stem)
                    if year is None:
                        continue
                    ref = _PartitionRef(
                        manager=manager_dir.name,
                        timeframe=timeframe_dir.name,
                        year=year,
                    )
                    items.append((ref, parquet_path))

        return tuple(
            sorted(
                items,
                key=lambda item: (
                    item[0].manager,
                    item[0].timeframe,
                    item[0].year,
                    str(item[1]),
                ),
            )
        )

    def _write_reports(
        self,
        detail: pl.DataFrame,
        timeframe_summary: pl.DataFrame,
        global_summary: pl.DataFrame,
    ) -> WalkForwardAuditPaths:
        """Write the three deterministic CSV reports."""
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ReportingValidationError(
                "unable to create Walk-Forward audit output directory",
                error_code=_ERROR_OUTPUT_DIR,
                details={"output_dir": str(self._output_dir)},
            ) from exc

        paths = WalkForwardAuditPaths(
            detail=self._output_dir / ALL_TIMEFRAMES_CSV_NAME,
            timeframe_summary=self._output_dir / TIMEFRAME_SUMMARY_CSV_NAME,
            global_summary=self._output_dir / GLOBAL_SUMMARY_CSV_NAME,
        )
        _write_csv(detail, paths.detail)
        _write_csv(timeframe_summary, paths.timeframe_summary)
        _write_csv(global_summary, paths.global_summary)
        return paths


def aggregate_partition_frame(
    frame: pl.DataFrame,
    *,
    manager: str,
    engine: str,
    timeframe: Timeframe,
    year: int,
) -> list[dict[str, object]]:
    """Aggregate one Walk-Forward partition into one or more audit rows.

    When a ``symbol`` column is present, one audit row is emitted per symbol.
    Otherwise a single cross-sectional row is emitted with an empty symbol.
    Schema problems are never repaired; they are recorded in ``error`` with
    ``status=FAIL``.
    """
    if "symbol" in frame.columns:
        symbols = sorted(
            str(value) for value in frame.get_column("symbol").drop_nulls().unique().to_list()
        )
        rows: list[dict[str, object]] = []
        for symbol in symbols:
            symbol_frame = frame.filter(pl.col("symbol").cast(pl.String) == symbol)
            rows.append(
                _aggregate_symbol_frame(
                    symbol_frame,
                    manager=manager,
                    engine=engine,
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                )
            )
        return rows

    return [
        _aggregate_symbol_frame(
            frame,
            manager=manager,
            engine=engine,
            symbol="",
            timeframe=timeframe,
            year=year,
        )
    ]


def build_timeframe_summary(detail: pl.DataFrame) -> pl.DataFrame:
    """Build one summary row per timeframe from the detail audit frame."""
    if detail.height == 0:
        return pl.DataFrame(schema={column: pl.Null for column in TIMEFRAME_SUMMARY_COLUMNS})

    rows: list[dict[str, object]] = []
    for timeframe in detail.get_column("timeframe").unique(maintain_order=True).to_list():
        group = detail.filter(pl.col("timeframe") == timeframe)
        status = (
            _STATUS_PASS
            if group.filter(pl.col("status") != _STATUS_PASS).height == 0
            else _STATUS_FAIL
        )
        rows.append(
            {
                "timeframe": timeframe,
                "years": group.select("year").n_unique(),
                "symbols": _count_nonempty_symbols(group),
                "rows": _sum_int(group, "rows"),
                "selected_rows": _sum_optional_int(group, "selected_rows"),
                "pass_rows": _sum_int(group, "pass_rows"),
                "fail_rows": _sum_int(group, "fail_rows"),
                "unique_factors": _sum_optional_int(group, "unique_factors"),
                "unique_factor_versions": _sum_optional_int(group, "unique_factor_versions"),
                "unique_folds": _sum_optional_int(group, "unique_folds"),
                "future_return_1_non_null": _sum_optional_int(group, "future_return_1_non_null"),
                "future_return_1_mean": _weighted_mean(
                    group, "future_return_1_mean", "future_return_1_non_null"
                ),
                "future_return_1_std": _pooled_std(
                    group,
                    "future_return_1_std",
                    "future_return_1_mean",
                    "future_return_1_non_null",
                ),
                "oos_rows": _sum_optional_int(group, "oos_rows"),
                "oos_future_return_mean": _weighted_mean(
                    group, "oos_future_return_mean", "oos_rows"
                ),
                "oos_future_return_std": _pooled_std(
                    group,
                    "oos_future_return_std",
                    "oos_future_return_mean",
                    "oos_rows",
                ),
                "status": status,
            }
        )
    return (
        pl.DataFrame(rows)
        .select(list(TIMEFRAME_SUMMARY_COLUMNS))
        .sort("timeframe", maintain_order=True)
    )


def build_global_summary(detail: pl.DataFrame) -> pl.DataFrame:
    """Build the metric/value global summary frame from detail rows."""
    metrics: list[dict[str, object]] = [
        {"metric": "total_timeframes", "value": detail.select("timeframe").n_unique()},
        {"metric": "total_years", "value": detail.select("year").n_unique()},
        {"metric": "total_symbols", "value": _count_nonempty_symbols(detail)},
        {"metric": "total_rows", "value": _sum_int(detail, "rows")},
        {"metric": "total_selected_rows", "value": _sum_optional_int(detail, "selected_rows")},
        {"metric": "total_pass_rows", "value": _sum_int(detail, "pass_rows")},
        {"metric": "total_fail_rows", "value": _sum_int(detail, "fail_rows")},
        {
            "metric": "total_unique_factors",
            "value": _sum_optional_int(detail, "unique_factors"),
        },
        {
            "metric": "total_unique_factor_versions",
            "value": _sum_optional_int(detail, "unique_factor_versions"),
        },
        {"metric": "total_folds", "value": _sum_optional_int(detail, "unique_folds")},
        {
            "metric": "total_non_null_future_return_1",
            "value": _sum_optional_int(detail, "future_return_1_non_null"),
        },
        {
            "metric": "global_future_return_1_mean",
            "value": _weighted_mean(detail, "future_return_1_mean", "future_return_1_non_null"),
        },
        {
            "metric": "global_future_return_1_std",
            "value": _pooled_std(
                detail,
                "future_return_1_std",
                "future_return_1_mean",
                "future_return_1_non_null",
            ),
        },
        {"metric": "total_oos_rows", "value": _sum_optional_int(detail, "oos_rows")},
        {
            "metric": "global_oos_future_return_mean",
            "value": _weighted_mean(detail, "oos_future_return_mean", "oos_rows"),
        },
        {
            "metric": "global_oos_future_return_std",
            "value": _pooled_std(
                detail,
                "oos_future_return_std",
                "oos_future_return_mean",
                "oos_rows",
            ),
        },
    ]
    return pl.DataFrame(metrics).select(["metric", "value"])


def format_discovery_table(detail: pl.DataFrame) -> str:
    """Format the concise discovery table printed before report generation."""
    header = "TIMEFRAME | YEAR | SYMBOLS | ROWS | SELECTED | PASS | FAIL"
    if detail.height == 0:
        return header

    lines = [header]
    keys = detail.select(["timeframe", "year"]).unique(maintain_order=True)
    for timeframe, year in keys.iter_rows():
        group = detail.filter((pl.col("timeframe") == timeframe) & (pl.col("year") == year))
        selected = _sum_optional_int(group, "selected_rows")
        selected_display = "NULL" if selected is None else str(selected)
        lines.append(
            f"{timeframe} | {year} | {_count_nonempty_symbols(group)} | "
            f"{_sum_int(group, 'rows')} | {selected_display} | "
            f"{_sum_int(group, 'pass_rows')} | {_sum_int(group, 'fail_rows')}"
        )
    return "\n".join(lines)


def forbidden_import_violations(source: str) -> tuple[str, ...]:
    """Return forbidden import module names referenced by ``source``."""
    violations: list[str] = []
    for module in _FORBIDDEN_IMPORT_MODULES:
        patterns = (
            f"import {module}",
            f"from {module} ",
            f"from {module}.",
        )
        if any(pattern in source for pattern in patterns):
            violations.append(module)
    return tuple(violations)


def _aggregate_symbol_frame(
    frame: pl.DataFrame,
    *,
    manager: str,
    engine: str,
    symbol: str,
    timeframe: Timeframe,
    year: int,
) -> dict[str, object]:
    """Aggregate metrics and integrity checks for one symbol slice."""
    errors: list[str] = []
    rows = frame.height

    errors.extend(_check_duplicate_primary_observations(frame))
    errors.extend(_check_symbol_isolation(frame, expected_symbol=symbol))
    errors.extend(_check_timeframe_isolation(frame, expected_timeframe=timeframe))
    errors.extend(_check_year_isolation(frame, expected_year=year))
    errors.extend(_check_future_return_1(frame))

    fold_column = _detect_fold_column(frame.columns)
    partition = _detect_partition_masks(frame)

    selected_rows = _count_selected_rows(frame)
    pass_rows, fail_rows = _count_status_rows(frame)
    unique_factors = _unique_count(frame, "factor_name")
    unique_factor_versions = _unique_factor_version_count(frame)
    unique_folds: int | None
    if fold_column is None:
        unique_folds = None
        errors.append("fold column not found in Walk-Forward schema")
    else:
        unique_folds = frame.select(fold_column).n_unique()

    train_rows: int | None = None
    oos_rows: int | None = None
    if partition is not None:
        train_mask, oos_mask = partition
        train_rows = int(frame.filter(train_mask).height)
        oos_rows = int(frame.filter(oos_mask).height)

    first_open_time, last_open_time = _minmax(frame, "open_time")
    first_selection_time, last_selection_time = _selection_time_bounds(frame)

    future_stats = _return_stats(frame, "future_return_1", mask=None)
    if partition is None:
        oos_stats = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "non_null": None,
            "null": None,
        }
    else:
        oos_stats = _return_stats(frame, "future_return_1", mask=partition[1])

    selected_factor_memberships = _count_selected_factor_memberships(frame)
    status = _STATUS_PASS if not errors else _STATUS_FAIL
    return {
        "manager": manager,
        "engine": engine,
        "symbol": symbol,
        "timeframe": timeframe,
        "year": int(year),
        "rows": rows,
        "selected_rows": selected_rows,
        "pass_rows": pass_rows,
        "fail_rows": fail_rows,
        "unique_factors": unique_factors,
        "unique_factor_versions": unique_factor_versions,
        "unique_folds": unique_folds,
        "train_rows": train_rows,
        "oos_rows": oos_rows,
        "first_open_time": first_open_time,
        "last_open_time": last_open_time,
        "first_selection_time": first_selection_time,
        "last_selection_time": last_selection_time,
        "future_return_1_non_null": future_stats["non_null"],
        "future_return_1_null": future_stats["null"],
        "future_return_1_mean": future_stats["mean"],
        "future_return_1_std": future_stats["std"],
        "future_return_1_min": future_stats["min"],
        "future_return_1_max": future_stats["max"],
        "oos_future_return_mean": oos_stats["mean"],
        "oos_future_return_std": oos_stats["std"],
        "oos_future_return_min": oos_stats["min"],
        "oos_future_return_max": oos_stats["max"],
        "selected_factor_memberships": selected_factor_memberships,
        "status": status,
        "error": "; ".join(errors) if errors else "",
    }


def _check_duplicate_primary_observations(frame: pl.DataFrame) -> list[str]:
    """Return errors when primary-key duplicates exist."""
    key_columns = [column for column in PRIMARY_KEY_COLUMNS if column in frame.columns]
    if len(key_columns) != len(PRIMARY_KEY_COLUMNS):
        if frame.height != frame.unique().height:
            return ["duplicate primary observations detected"]
        return ["primary key columns incomplete for duplicate detection"]

    if frame.height != frame.unique(subset=list(PRIMARY_KEY_COLUMNS)).height:
        return ["duplicate primary observations detected on " f"({', '.join(PRIMARY_KEY_COLUMNS)})"]
    return []


def _check_symbol_isolation(frame: pl.DataFrame, *, expected_symbol: str) -> list[str]:
    """Return errors when symbol isolation is violated."""
    if "symbol" not in frame.columns:
        return []
    values = {str(value) for value in frame.get_column("symbol").drop_nulls().unique().to_list()}
    if expected_symbol == "":
        if len(values) > 1:
            return ["symbol isolation violated: multiple symbols in partition slice"]
        return []
    if values != {expected_symbol}:
        return [
            "symbol isolation violated: " f"expected={expected_symbol!r} actual={sorted(values)!r}"
        ]
    return []


def _check_timeframe_isolation(
    frame: pl.DataFrame,
    *,
    expected_timeframe: Timeframe,
) -> list[str]:
    """Return errors when timeframe values disagree with the partition path."""
    if "timeframe" not in frame.columns:
        return ["timeframe column missing; cannot verify timeframe isolation"]
    values = {str(value) for value in frame.get_column("timeframe").drop_nulls().unique().to_list()}
    if values != {str(expected_timeframe)}:
        return [
            "timeframe isolation violated: "
            f"expected={expected_timeframe!r} actual={sorted(values)!r}"
        ]
    return []


def _check_year_isolation(frame: pl.DataFrame, *, expected_year: int) -> list[str]:
    """Return errors when timestamp years disagree with the partition year."""
    timestamp_columns = [
        column
        for column in (
            "open_time",
            "selection_time",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        )
        if column in frame.columns
    ]
    if not timestamp_columns:
        return []

    years: set[int] = set()
    for column in timestamp_columns:
        for value in frame.get_column(column).drop_nulls().to_list():
            try:
                timestamp_ms = int(value)
            except (TypeError, ValueError):
                continue
            years.add(datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).year)
    if years and years != {int(expected_year)}:
        return ["year isolation violated: " f"expected={expected_year!r} actual={sorted(years)!r}"]
    return []


def _check_future_return_1(frame: pl.DataFrame) -> list[str]:
    """Return errors when ``future_return_1`` is missing or would be fabricated."""
    if "future_return_1" not in frame.columns:
        return [
            "future_return_1 missing from Walk-Forward artifact "
            "(evaluation-input target is not persisted on the ledger; "
            "refusing to fabricate values)"
        ]
    return []


def _count_selected_rows(frame: pl.DataFrame) -> int | None:
    """Count rows where ``selected == True`` when the column exists."""
    if "selected" not in frame.columns:
        return None
    selected = frame.get_column("selected")
    if selected.dtype == pl.Boolean:
        return int(selected.fill_null(False).sum())
    as_string = selected.cast(pl.String).str.to_lowercase()
    return int(as_string.is_in(["true", "1", "t", "yes", "y"]).fill_null(False).sum())


def _count_selected_factor_memberships(frame: pl.DataFrame) -> int | None:
    """Count selected factor memberships, not merely output rows."""
    if "selected" in frame.columns:
        return _count_selected_rows(frame)
    if "selected_factors" in frame.columns:
        series = frame.get_column("selected_factors").cast(pl.Float64, strict=False)
        if series.null_count() == series.len():
            return None
        return int(series.fill_null(0).sum())
    return None


def _count_status_rows(frame: pl.DataFrame) -> tuple[int, int]:
    """Count PASS and non-PASS rows from the status column."""
    if "status" not in frame.columns:
        return 0, frame.height
    pass_status = WalkForwardStatus.PASS.value
    pass_rows = frame.filter(pl.col("status").cast(pl.String) == pass_status).height
    return pass_rows, frame.height - pass_rows


def _unique_count(frame: pl.DataFrame, column: str) -> int | None:
    """Return unique non-null count for ``column`` or null when absent."""
    if column not in frame.columns:
        return None
    return int(frame.select(column).n_unique())


def _unique_factor_version_count(frame: pl.DataFrame) -> int | None:
    """Return unique ``(factor_name, factor_version)`` count when present."""
    if "factor_name" not in frame.columns or "factor_version" not in frame.columns:
        return None
    return int(frame.select(["factor_name", "factor_version"]).unique().height)


def _detect_fold_column(columns: Sequence[str]) -> str | None:
    """Detect the fold identifier column from the artifact schema."""
    column_set = set(columns)
    for candidate in _FOLD_COLUMN_CANDIDATES:
        if candidate in column_set:
            return candidate
    return None


def _detect_partition_masks(
    frame: pl.DataFrame,
) -> tuple[pl.Expr, pl.Expr] | None:
    """Detect explicit train/OOS partition masks when present."""
    for column in _PARTITION_COLUMN_CANDIDATES:
        if column not in frame.columns:
            continue
        dtype = frame.schema[column]
        if column == "is_oos" or dtype == pl.Boolean:
            oos_mask = pl.col(column).fill_null(False).cast(pl.Boolean)
            return ~oos_mask, oos_mask
        normalized = pl.col(column).cast(pl.String).str.strip_chars().str.to_lowercase()
        oos_mask = normalized.is_in(list(_OOS_PARTITION_VALUES))
        train_mask = normalized.is_in(list(_TRAIN_PARTITION_VALUES))
        if frame.filter(oos_mask | train_mask).height > 0:
            return train_mask, oos_mask
    return None


def _selection_time_bounds(frame: pl.DataFrame) -> tuple[int | None, int | None]:
    """Return first/last selection times from explicit columns only."""
    if "selection_time" in frame.columns:
        return _minmax(frame, "selection_time")
    lower_columns = [column for column in ("train_start", "test_start") if column in frame.columns]
    upper_columns = [column for column in ("train_end", "test_end") if column in frame.columns]
    first: int | None = None
    last: int | None = None
    if lower_columns:
        minimums = [
            frame.get_column(column).cast(pl.Int64, strict=False).drop_nulls().min()
            for column in lower_columns
        ]
        valid = [_as_int(value) for value in minimums if value is not None]
        filtered = [value for value in valid if value is not None]
        first = min(filtered) if filtered else None
    if upper_columns:
        maximums = [
            frame.get_column(column).cast(pl.Int64, strict=False).drop_nulls().max()
            for column in upper_columns
        ]
        valid = [_as_int(value) for value in maximums if value is not None]
        filtered = [value for value in valid if value is not None]
        last = max(filtered) if filtered else None
    return first, last


def _return_stats(
    frame: pl.DataFrame,
    column: str,
    *,
    mask: pl.Expr | None,
) -> dict[str, float | int | None]:
    """Compute non-null return statistics for ``column``."""
    empty: dict[str, float | int | None] = {
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "non_null": None,
        "null": None,
    }
    if column not in frame.columns:
        return empty

    scoped = frame if mask is None else frame.filter(mask)
    numeric = scoped.get_column(column).cast(pl.Float64, strict=False)
    non_null = numeric.drop_nulls()
    return {
        "non_null": int(non_null.len()),
        "null": int(numeric.len() - non_null.len()),
        "mean": _as_float(non_null.mean()) if non_null.len() > 0 else None,
        "std": _as_float(non_null.std(ddof=1)) if non_null.len() >= 2 else None,
        "min": _as_float(non_null.min()) if non_null.len() > 0 else None,
        "max": _as_float(non_null.max()) if non_null.len() > 0 else None,
    }


def _minmax(frame: pl.DataFrame, column: str) -> tuple[int | None, int | None]:
    """Return min/max integer bounds for ``column`` when present."""
    if column not in frame.columns:
        return None, None
    series = frame.get_column(column).cast(pl.Int64, strict=False).drop_nulls()
    if series.len() == 0:
        return None, None
    return _as_int(series.min()), _as_int(series.max())


def _finalize_detail_frame(rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    """Materialize, order, and sort the detail audit frame."""
    detail = pl.DataFrame(list(rows)).select(list(DETAIL_COLUMNS))
    return detail.sort(
        ["manager", "engine", "symbol", "timeframe", "year"],
        maintain_order=True,
    )


def _write_csv(frame: pl.DataFrame, path: Path) -> None:
    """Write a deterministic UTF-8 CSV with stable null representation."""
    frame.write_csv(path, null_value="")


def _hash_files(paths: Sequence[Path]) -> dict[str, str]:
    """Return SHA-256 hashes keyed by normalized path strings."""
    hashes: dict[str, str] = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        hashes[str(path.resolve())] = digest.hexdigest()
    return hashes


def _parse_year_stem(stem: str) -> int | None:
    """Parse a ``{year}.parquet`` stem into an integer year."""
    if not stem.isdigit():
        return None
    year = int(stem)
    if year < 1970 or year > 2100:
        return None
    return year


def _count_nonempty_symbols(frame: pl.DataFrame) -> int:
    """Count distinct non-empty symbol values."""
    if "symbol" not in frame.columns:
        return 0
    symbols = frame.get_column("symbol").cast(pl.String).drop_nulls()
    return int(symbols.filter(symbols != "").n_unique())


def _sum_int(frame: pl.DataFrame, column: str) -> int:
    """Sum integer-like values, treating nulls as zero."""
    total = frame.get_column(column).cast(pl.Float64, strict=False).fill_null(0).sum()
    converted = _as_int(total)
    return 0 if converted is None else converted


def _sum_optional_int(frame: pl.DataFrame, column: str) -> int | None:
    """Sum optional integers, returning null when every value is null."""
    series = frame.get_column(column).cast(pl.Float64, strict=False)
    if series.null_count() == series.len():
        return None
    return _as_int(series.fill_null(0).sum())


def _weighted_mean(frame: pl.DataFrame, value_column: str, weight_column: str) -> float | None:
    """Return a weight-aware mean, or null when weights are unavailable."""
    values = frame.get_column(value_column).cast(pl.Float64, strict=False)
    weights = frame.get_column(weight_column).cast(pl.Float64, strict=False)
    mask = values.is_not_null() & weights.is_not_null() & (weights > 0)
    if mask.sum() == 0:
        return None
    total_weight = _as_float(weights.filter(mask).sum())
    if total_weight is None or total_weight <= 0.0:
        return None
    weighted_sum = _as_float((values.filter(mask) * weights.filter(mask)).sum())
    if weighted_sum is None:
        return None
    return weighted_sum / total_weight


def _pooled_std(
    frame: pl.DataFrame,
    std_column: str,
    mean_column: str,
    weight_column: str,
) -> float | None:
    """Approximate a pooled std from subgroup stats; null when incomplete."""
    stds = frame.get_column(std_column).cast(pl.Float64, strict=False)
    means = frame.get_column(mean_column).cast(pl.Float64, strict=False)
    weights = frame.get_column(weight_column).cast(pl.Float64, strict=False)
    mask = stds.is_not_null() & means.is_not_null() & weights.is_not_null() & (weights > 1)
    if mask.sum() == 0:
        return None
    if int(mask.sum()) == 1:
        return _as_float(stds.filter(mask).item())
    global_mean = _weighted_mean(frame, mean_column, weight_column)
    if global_mean is None:
        return None
    weight_total = _as_float(weights.filter(mask).sum())
    if weight_total is None or weight_total <= 0.0:
        return None
    variance_sum = _as_float(
        (
            weights.filter(mask)
            * (stds.filter(mask) ** 2 + (means.filter(mask) - global_mean) ** 2)
        ).sum()
    )
    if variance_sum is None:
        return None
    weighted_variance = variance_sum / weight_total
    if weighted_variance < 0.0:
        return None
    return weighted_variance**0.5


def _as_int(value: object) -> int | None:
    """Convert a Polars scalar to ``int`` when numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return int(value)
    return None


def _as_float(value: object) -> float | None:
    """Convert a Polars scalar to ``float`` when numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return value
    return None

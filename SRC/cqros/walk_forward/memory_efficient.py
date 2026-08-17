"""Bounded-memory physical execution for canonical Walk Forward.

The executor preserves the full-panel join, ordering, row-index fold, score,
and aggregate semantics while bounding retained input data to one symbol at a
time. Symbol-local canonical evaluation shards are externally merged into a
run-scoped Parquet stream, assigned deterministic row ordinals, evaluated in
bounded row windows, and removed on success or failure.
"""

from __future__ import annotations

import heapq
import logging
import shutil
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import uuid4

import polars as pl
from polars.testing import assert_frame_equal

from cqros.core.types import Exchange, Market, Symbol, Timeframe
from cqros.factors.repository import FactorsRepository
from cqros.factors.schema import (
    FACTOR_SCHEMA,
)
from cqros.factors.schema import (
    REQUIRED_COLUMNS as FACTOR_REQUIRED_COLUMNS,
)
from cqros.labels.schema import (
    MERGED_LABEL_SCHEMA,
)
from cqros.labels.schema import (
    REQUIRED_COLUMNS as LABEL_REQUIRED_COLUMNS,
)
from cqros.storage.label_repository import LabelRepository
from cqros.storage.layout import StorageLayout
from cqros.walk_forward.engine import (
    SimpleWalkForwardEngine,
    _AggregateMetrics,
    _compute_aggregate_metrics_frame,
    build_walk_forward_fold,
)
from cqros.walk_forward.evaluation_input import (
    TARGET_COLUMN,
    WALK_FORWARD_EVALUATION_COLUMNS,
    assemble_walk_forward_symbol_input,
)
from cqros.walk_forward.exceptions import WalkForwardError
from cqros.walk_forward.schema import (
    CANONICAL_COLUMN_ORDER,
    WALK_FORWARD_SCHEMA,
    WalkForwardStatus,
)

__all__ = [
    "FULL_PANEL_EXECUTION_MODE",
    "MEMORY_EFFICIENT_EXECUTION_MODE",
    "MemoryEfficientExecutionConfig",
    "MemoryEfficientWalkForwardExecutor",
    "assert_walk_forward_equivalent",
]

FULL_PANEL_EXECUTION_MODE: Final[str] = "full_panel"
MEMORY_EFFICIENT_EXECUTION_MODE: Final[str] = "memory_efficient"

_DEFAULT_MEMORY_BUDGET_MB: Final[int] = 256
_MEBIBYTE: Final[int] = 1024 * 1024
_MIN_CURSOR_BATCH_ROWS: Final[int] = 256
_MAX_CURSOR_BATCH_ROWS: Final[int] = 65_536
_OUTPUT_BATCH_ROWS: Final[int] = 8_192
_ROW_ORDINAL: Final[str] = "row_ordinal"
_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "selection_time",
    "symbol",
    "factor_name",
    "factor_version",
)
_FACTOR_PROJECT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    "factor_name",
    "factor_version",
    "factor_value",
)
_LABEL_PROJECT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "open_time",
    TARGET_COLUMN,
)
_ENGINE_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "selection_time",
    "selected",
    TARGET_COLUMN,
)
_AGGREGATE_COLUMNS: Final[tuple[str, ...]] = (
    "train_score",
    "test_score",
    "overfit_gap",
    "status",
)
_ERROR_CONFIG: Final[str] = "WF_MEMORY_CONFIG"
_ERROR_EMPTY_JOIN: Final[str] = "WF_EVAL_EMPTY_JOIN"
_ERROR_SCHEMA: Final[str] = "WF_MEMORY_SOURCE_SCHEMA"

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryEfficientExecutionConfig:
    """Configuration for bounded Walk Forward physical execution.

    Args:
        spill_parent: Parent for unique run-scoped temporary directories.
        memory_budget_mb: External-merge cursor budget. This bounds batches,
            not production output schema or research semantics.
    """

    spill_parent: Path
    memory_budget_mb: int = _DEFAULT_MEMORY_BUDGET_MB

    def __post_init__(self) -> None:
        """Validate immutable execution-only configuration."""
        object.__setattr__(self, "spill_parent", Path(self.spill_parent))
        _require_positive_memory_budget(self.memory_budget_mb)


class _ParquetRowCursor:
    """Bounded sequential row cursor over one sorted Parquet shard."""

    __slots__ = ("_batches", "_rows")

    _batches: Iterator[pl.DataFrame]
    _rows: Iterator[dict[str, object]]

    def __init__(self, path: Path, *, batch_rows: int) -> None:
        self._batches = (
            pl.scan_parquet(path)
            .select(list(WALK_FORWARD_EVALUATION_COLUMNS))
            .collect_batches(
                chunk_size=batch_rows,
                maintain_order=True,
                engine="streaming",
            )
        )
        self._rows = iter(())

    def next_row(self) -> dict[str, object] | None:
        """Return the next row, loading at most one configured batch."""
        try:
            return next(self._rows)
        except StopIteration:
            try:
                batch = next(self._batches)
            except StopIteration:
                return None
            self._rows = batch.iter_rows(named=True)
            return next(self._rows, None)


class MemoryEfficientWalkForwardExecutor:
    """Execute canonical Walk Forward with bounded retained symbol data.

    The executor creates only run-scoped symbol shards, a sorted ordinal run,
    and a raw-fold spill beneath ``config.spill_parent``. Cleanup occurs in a
    ``finally`` block for normal completion, exceptions, and
    ``KeyboardInterrupt``. No canonical artifact is written by this class.
    """

    __slots__ = (
        "_config",
        "_engine",
        "_factors_repository",
        "_label_repository",
        "_layout",
        "_logger",
    )

    def __init__(
        self,
        layout: StorageLayout,
        factors_repository: FactorsRepository,
        label_repository: LabelRepository,
        engine: SimpleWalkForwardEngine,
        config: MemoryEfficientExecutionConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the executor with injected storage and engine contracts."""
        self._layout = layout
        self._factors_repository = factors_repository
        self._label_repository = label_repository
        self._engine = engine
        self._config = config
        self._logger = logger if logger is not None else _logger

    def execute(
        self,
        factor_selection: pl.DataFrame,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
        symbols: Sequence[Symbol] | None = None,
    ) -> pl.DataFrame:
        """Build one canonical partition without retaining the full input panel."""
        run_directory = self._config.spill_parent / (
            f"{manager}-{exchange}-{market}-{timeframe}-{year}-{uuid4().hex}"
        )
        run_directory.mkdir(parents=True, exist_ok=False)
        self._logger.info(
            "Starting memory-efficient walk-forward execution",
            extra={
                "execution_mode": MEMORY_EFFICIENT_EXECUTION_MODE,
                "memory_budget_mb": self._config.memory_budget_mb,
                "timeframe": timeframe,
                "year": year,
            },
        )
        try:
            panel_symbols = self._resolve_panel_symbols(
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
                symbols=symbols,
            )
            selection = factor_selection.select(
                [
                    "factor_name",
                    "factor_version",
                    "timeframe",
                    "selected",
                    "selection_ic",
                    "selected_direction",
                    "orientation_policy",
                ]
            )
            shards = self._write_symbol_shards(
                run_directory,
                selection,
                panel_symbols=panel_symbols,
                manager=manager,
                exchange=exchange,
                market=market,
                timeframe=timeframe,
                year=year,
            )
            if not shards:
                raise WalkForwardError(
                    "Factor Selection join produced no matching observation rows",
                    error_code=_ERROR_EMPTY_JOIN,
                    details={"timeframe": timeframe, "year": year},
                )
            sorted_runs = self._merge_sorted_shards(
                shards,
                run_directory / "sorted_ordinal_run",
            )
            return self._build_folds(sorted_runs, run_directory / "raw_folds")
        finally:
            shutil.rmtree(run_directory, ignore_errors=True)

    def _resolve_panel_symbols(
        self,
        *,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
        symbols: Sequence[Symbol] | None,
    ) -> tuple[Symbol, ...]:
        """Resolve the canonical sorted intersection of Factors and Labels."""
        if symbols is None:
            candidates = tuple(
                partition.symbol
                for partition in self._factors_repository.discover_partitions(
                    managers=(manager,),
                    timeframes=(timeframe,),
                    exchange=exchange,
                    market=market,
                )
                if partition.year == year
            )
        else:
            candidates = tuple(symbol for symbol in symbols if symbol.strip())
        resolved = tuple(
            symbol
            for symbol in sorted(set(candidates))
            if self._factors_repository.exists(
                manager=manager,
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
            and self._label_repository.exists(
                exchange=exchange,
                market=market,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
        )
        if not resolved:
            raise WalkForwardError(
                "no symbols with both Factors and Labels for walk-forward panel",
                error_code="WF_EVAL_EMPTY_PANEL",
                details={"timeframe": timeframe, "year": year},
            )
        return resolved

    def _write_symbol_shards(
        self,
        run_directory: Path,
        selection: pl.DataFrame,
        *,
        panel_symbols: tuple[Symbol, ...],
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> tuple[Path, ...]:
        """Project, join, sort, and spill one symbol at a time."""
        shard_directory = run_directory / "symbol_shards"
        shard_directory.mkdir()
        shards: list[Path] = []
        for index, symbol in enumerate(panel_symbols):
            factors_path = self._layout.factors_path(
                manager, exchange, market, symbol, timeframe, year
            )
            labels_path = self._layout.label_path(exchange, market, symbol, timeframe, year)
            factors = _read_projected(
                factors_path,
                required=FACTOR_REQUIRED_COLUMNS,
                projected=_FACTOR_PROJECT_COLUMNS,
                schema=FACTOR_SCHEMA,
                side="factors",
            )
            labels = _read_projected(
                labels_path,
                required=LABEL_REQUIRED_COLUMNS,
                projected=_LABEL_PROJECT_COLUMNS,
                schema=MERGED_LABEL_SCHEMA,
                side="labels",
            )
            shard = assemble_walk_forward_symbol_input(selection, factors, labels)
            if shard.height == 0:
                continue
            shard_path = shard_directory / f"{index:06d}.parquet"
            shard.write_parquet(shard_path, compression="zstd")
            shards.append(shard_path)
        return tuple(shards)

    def _merge_sorted_shards(
        self,
        shards: tuple[Path, ...],
        target: Path,
    ) -> tuple[Path, ...]:
        """Perform a deterministic bounded k-way merge and assign row ordinals."""
        target.mkdir()
        budget_bytes = self._config.memory_budget_mb * _MEBIBYTE
        batch_rows = max(
            _MIN_CURSOR_BATCH_ROWS,
            min(
                _MAX_CURSOR_BATCH_ROWS,
                budget_bytes // max(len(shards), 1) // 1024,
            ),
        )
        cursors = [_ParquetRowCursor(path, batch_rows=batch_rows) for path in shards]
        heap: list[tuple[tuple[object, ...], int, dict[str, object]]] = []
        for source_index, cursor in enumerate(cursors):
            row = cursor.next_row()
            if row is not None:
                heapq.heappush(heap, (_row_sort_key(row), source_index, row))

        parts: list[Path] = []
        output_rows: list[dict[str, object]] = []
        ordinal = 0
        while heap:
            _, source_index, row = heapq.heappop(heap)
            emitted = dict(row)
            emitted[_ROW_ORDINAL] = ordinal
            output_rows.append(emitted)
            ordinal += 1
            if len(output_rows) >= _OUTPUT_BATCH_ROWS:
                parts.append(_write_rows(target, output_rows, part=len(parts)))
                output_rows.clear()
            next_row = cursors[source_index].next_row()
            if next_row is not None:
                heapq.heappush(
                    heap,
                    (_row_sort_key(next_row), source_index, next_row),
                )
        if output_rows:
            parts.append(_write_rows(target, output_rows, part=len(parts)))
        return tuple(parts)

    def _build_folds(
        self,
        sorted_runs: tuple[Path, ...],
        raw_fold_path: Path,
    ) -> pl.DataFrame:
        """Evaluate row-ordinal folds, then apply aggregates in a second pass."""
        raw_fold_path.mkdir()
        required = self._engine.train_window + self._engine.test_window
        window: deque[dict[str, object]] = deque(maxlen=required)
        short_rows: list[dict[str, object]] = []
        next_start = 0
        fold_id = 1
        raw_parts: list[Path] = []
        raw_rows: list[dict[str, object]] = []
        total_rows = 0
        for sorted_run in sorted_runs:
            batches = (
                pl.scan_parquet(sorted_run)
                .select(list(_ENGINE_COLUMNS))
                .collect_batches(
                    chunk_size=_OUTPUT_BATCH_ROWS,
                    maintain_order=True,
                    engine="streaming",
                )
            )
            for batch in batches:
                for row in batch.iter_rows(named=True):
                    total_rows += 1
                    window.append(row)
                    if fold_id == 1 and total_rows < required:
                        short_rows.append(row)
                    if total_rows < next_start + required:
                        continue
                    group = pl.DataFrame(
                        list(window),
                        schema={
                            "timeframe": pl.String,
                            "selection_time": pl.Int64,
                            "selected": pl.Boolean,
                            TARGET_COLUMN: pl.Float64,
                        },
                    )
                    raw_rows.append(
                        build_walk_forward_fold(
                            group,
                            fold_id=fold_id,
                            train_start_index=0,
                            train_end_index=self._engine.train_window,
                            test_start_index=self._engine.train_window,
                            test_end_index=required,
                        )
                    )
                    fold_id += 1
                    next_start += self._engine.step_size
                    if raw_rows and len(raw_rows) >= _OUTPUT_BATCH_ROWS:
                        raw_parts.append(
                            _write_fold_rows(
                                raw_fold_path,
                                raw_rows,
                                part=len(raw_parts),
                            )
                        )
                        raw_rows.clear()
        if fold_id == 1:
            group = pl.DataFrame(
                short_rows,
                schema={
                    "timeframe": pl.String,
                    "selection_time": pl.Int64,
                    "selected": pl.Boolean,
                    TARGET_COLUMN: pl.Float64,
                },
            )
            return self._engine.build(group)
        if raw_rows:
            raw_parts.append(
                _write_fold_rows(
                    raw_fold_path,
                    raw_rows,
                    part=len(raw_parts),
                )
            )

        aggregates = _aggregate_metrics_from_parts(raw_parts)
        return _apply_aggregate_metrics_to_parts(raw_parts, aggregates)


def assert_walk_forward_equivalent(
    canonical: pl.DataFrame,
    memory_efficient: pl.DataFrame,
) -> None:
    """Require exact canonical schema, ordering, values, and floating bits."""
    assert_frame_equal(
        memory_efficient,
        canonical,
        check_row_order=True,
        check_column_order=True,
        check_dtypes=True,
        check_exact=True,
    )


def _read_projected(
    path: Path,
    *,
    required: tuple[str, ...],
    projected: tuple[str, ...],
    schema: pl.Schema,
    side: str,
) -> pl.DataFrame:
    """Validate stored columns, then materialize only the execution projection."""
    stored = pl.Schema(pl.read_parquet_schema(path))
    missing = tuple(column for column in required if column not in stored)
    if missing:
        raise WalkForwardError(
            f"{side} source is missing canonical columns",
            error_code=_ERROR_SCHEMA,
            details={"side": side, "missing_columns": missing, "path": str(path)},
        )
    projected_schema = pl.Schema([(column, schema[column]) for column in projected])
    return pl.read_parquet(path, columns=list(projected)).cast(projected_schema)


def _require_positive_memory_budget(value: object) -> int:
    """Return a positive integer memory budget or raise a CQROS error."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WalkForwardError(
            "memory_budget_mb must be a positive integer",
            error_code=_ERROR_CONFIG,
            details={"memory_budget_mb": value},
        )
    return value


def _row_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    """Return the strict canonical evaluation ordering key."""
    return tuple(row[column] for column in _SORT_COLUMNS)


def _write_rows(
    target: Path,
    rows: list[dict[str, object]],
    *,
    part: int,
) -> Path:
    """Append one deterministic merged-run batch."""
    schema = {
        "symbol": pl.String,
        "timeframe": pl.String,
        "open_time": pl.Int64,
        "factor_name": pl.String,
        "factor_version": pl.String,
        "factor_value": pl.Float64,
        "selected": pl.Boolean,
        "selection_time": pl.Int64,
        "selection_ic": pl.Float64,
        "selected_direction": pl.Int8,
        "orientation_policy": pl.String,
        TARGET_COLUMN: pl.Float64,
        _ROW_ORDINAL: pl.UInt64,
    }
    path = target / f"part-{part:08d}.parquet"
    pl.DataFrame(rows, schema=schema).write_parquet(path, compression="zstd")
    return path


def _write_fold_rows(
    target: Path,
    rows: list[dict[str, object]],
    *,
    part: int,
) -> Path:
    """Append one raw-fold batch using the canonical output schema."""
    path = target / f"part-{part:08d}.parquet"
    (
        pl.DataFrame(rows, schema=WALK_FORWARD_SCHEMA)
        .select(list(CANONICAL_COLUMN_ORDER))
        .write_parquet(path, compression="zstd")
    )
    return path


def _aggregate_metrics_from_parts(raw_parts: Sequence[Path]) -> _AggregateMetrics:
    """Replay the canonical PASS-only aggregate over spilled raw folds.

    The spilled fold columns are reassembled as the same single-chunk frame
    the full-panel engine reduces, so every floating-point bit is reproduced.
    """
    aggregate_frame = (
        pl.read_parquet(raw_parts, columns=list(_AGGREGATE_COLUMNS))
        .select(list(_AGGREGATE_COLUMNS))
        .rechunk()
    )
    return _compute_aggregate_metrics_frame(
        aggregate_frame.filter(pl.col("status") == WalkForwardStatus.PASS.value)
    )


def _apply_aggregate_metrics_to_parts(
    raw_parts: Sequence[Path],
    aggregates: _AggregateMetrics,
) -> pl.DataFrame:
    """Build the canonical output frame from spilled raw folds in bounded batches."""
    return (
        pl.scan_parquet(raw_parts)
        .with_columns(
            pl.lit(aggregates.mean_train_score, dtype=pl.Float64).alias("train_score"),
            pl.lit(aggregates.mean_test_score, dtype=pl.Float64).alias("test_score"),
            pl.lit(aggregates.walk_forward_stability, dtype=pl.Float64).alias("overfit_gap"),
        )
        .select(list(CANONICAL_COLUMN_ORDER))
        .cast(WALK_FORWARD_SCHEMA)
        .collect(engine="streaming")
    )

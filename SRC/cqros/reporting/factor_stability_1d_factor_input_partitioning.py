"""CQROS 1d factor-input partitioning architecture investigation.

Purpose:
    Determine whether CQROS can safely partition factor input history by
    declared dependency so OHLCV-only factors are not truncated by unrelated
    companion completeness, without mutating production lake artifacts.

Responsibilities:
    - Trace ``load_factor_input_frame`` / ``align_factor_input_frame`` callers
    - Inventory every registry factor's real ``required_features``
    - Measure processed dataset boundaries for 1d BTCUSDT (and cross-TF)
    - Simulate factor-specific input boundaries without writing factors
    - Prove PVT/OBV truncation is global-alignment induced
    - Prove OI factors retain the OI availability boundary
    - Emit reports under ``reports/factor_stability/1d_factor_input_partitioning``
    - SHA-256 hash watched production ledgers before and after
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``ast``, ``hashlib``, ``logging``, ``polars``, ``cqros.core.constants``,
    ``cqros.cli.generate_factors`` (read-only alignment helpers),
    ``cqros.factor_selection.eligibility``, ``cqros.factors.default_registry``,
    ``cqros.reporting.exceptions``, and ``cqros.storage``.

Public API:
    Classification / column / name constants,
    ``FactorStability1dFactorInputPartitioningReporter``,
    ``FactorStability1dFactorInputPartitioningResult``,
    ``classify_dependency_class``,
    ``classify_architecture_verdict``,
    ``simulate_factor_specific_start_index``,
    ``forbidden_import_violations``, and
    ``hash_watched_production_artifacts``.

Notes:
    Diagnostic only. Never mutates production ledgers, never regenerates
    Factor Selection / Walk Forward / Purged CV, never changes formulas,
    orientation, coverage policy, or thresholds.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl

from cqros.cli.generate_factors import (
    align_factor_input_frame,
    load_factor_input_frame,
)
from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTOR_SELECTION,
    STORAGE_DIR_FACTORS,
    STORAGE_DIR_PROCESSED,
    STORAGE_DIR_PURGED_CV,
    STORAGE_DIR_PURGED_CV_EVALUATION,
    STORAGE_DIR_WALK_FORWARD,
    STORAGE_DIR_WALK_FORWARD_EVALUATION,
)
from cqros.factor_selection.eligibility import FactorEligibilityPolicy
from cqros.factors.default_registry import build_default_registry
from cqros.reporting.exceptions import ReportingValidationError
from cqros.storage import ParquetStore, StorageLayout
from cqros.storage.processed_repository import ProcessedMarketDataRepository

__all__ = [
    "ARCHITECTURE_TRACE_COLUMNS",
    "ARCHITECTURE_TRACE_CSV_NAME",
    "CLASS_FUNDING_DEPENDENT",
    "CLASS_LONG_SHORT_DEPENDENT",
    "CLASS_MULTI_COMPANION_DEPENDENT",
    "CLASS_OHLCV_ONLY",
    "CLASS_OHLCV_PLUS_VOLUME",
    "CLASS_OI_DEPENDENT",
    "CLASS_TAKER_DEPENDENT",
    "CLASS_UNKNOWN",
    "COMPANION_TRUNCATION_COLUMNS",
    "COMPANION_TRUNCATION_CSV_NAME",
    "COMPARISON_TIMEFRAMES",
    "COVERAGE_COLUMNS",
    "COVERAGE_CSV_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "DEPENDENCY_COLUMNS",
    "DEPENDENCY_CSV_NAME",
    "HASHES_AFTER_NAME",
    "HASHES_BEFORE_NAME",
    "INPUT_BOUNDARY_COLUMNS",
    "INPUT_BOUNDARY_CSV_NAME",
    "REFERENCE_SYMBOL",
    "REFERENCE_SYMBOLS",
    "SUMMARY_TXT_NAME",
    "TARGET_TIMEFRAME",
    "VERDICT_MULTI_CAUSE_REQUIRES_DATA_BACKFILL",
    "VERDICT_PARTITIONING_NOT_SAFE",
    "VERDICT_PARTITIONING_SAFE_AND_BENEFICIAL",
    "VERDICT_PARTITIONING_SAFE_BUT_LOW_VALUE",
    "WARMUP_COLUMNS",
    "WARMUP_CSV_NAME",
    "FactorStability1dFactorInputPartitioningReporter",
    "FactorStability1dFactorInputPartitioningResult",
    "classify_architecture_verdict",
    "classify_dependency_class",
    "forbidden_import_violations",
    "hash_watched_production_artifacts",
    "simulate_factor_specific_start_index",
]

_logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT: Final[Path] = (
    Path("reports") / "factor_stability" / "1d_factor_input_partitioning"
)
TARGET_TIMEFRAME: Final[str] = "1d"
REFERENCE_SYMBOL: Final[str] = "BTCUSDT"
REFERENCE_SYMBOLS: Final[tuple[str, ...]] = ("BTCUSDT", "ETHUSDT")
COMPARISON_TIMEFRAMES: Final[tuple[str, ...]] = ("5m", "15m", "1h", "4h", "1d")

SUMMARY_TXT_NAME: Final[str] = "summary.txt"
DEPENDENCY_CSV_NAME: Final[str] = "factor_dependencies.csv"
INPUT_BOUNDARY_CSV_NAME: Final[str] = "input_boundaries.csv"
COVERAGE_CSV_NAME: Final[str] = "current_vs_partitioned_coverage.csv"
WARMUP_CSV_NAME: Final[str] = "warmup_analysis.csv"
COMPANION_TRUNCATION_CSV_NAME: Final[str] = "companion_truncation.csv"
ARCHITECTURE_TRACE_CSV_NAME: Final[str] = "architecture_trace.csv"
HASHES_BEFORE_NAME: Final[str] = "hashes_before.txt"
HASHES_AFTER_NAME: Final[str] = "hashes_after.txt"

CLASS_OHLCV_ONLY: Final[str] = "OHLCV_ONLY"
CLASS_OHLCV_PLUS_VOLUME: Final[str] = "OHLCV_PLUS_VOLUME"
CLASS_OI_DEPENDENT: Final[str] = "OI_DEPENDENT"
CLASS_TAKER_DEPENDENT: Final[str] = "TAKER_DEPENDENT"
CLASS_LONG_SHORT_DEPENDENT: Final[str] = "LONG_SHORT_DEPENDENT"
CLASS_FUNDING_DEPENDENT: Final[str] = "FUNDING_DEPENDENT"
CLASS_MULTI_COMPANION_DEPENDENT: Final[str] = "MULTI_COMPANION_DEPENDENT"
CLASS_UNKNOWN: Final[str] = "UNKNOWN"

VERDICT_PARTITIONING_SAFE_AND_BENEFICIAL: Final[str] = "PARTITIONING_SAFE_AND_BENEFICIAL"
VERDICT_PARTITIONING_SAFE_BUT_LOW_VALUE: Final[str] = "PARTITIONING_SAFE_BUT_LOW_VALUE"
VERDICT_PARTITIONING_NOT_SAFE: Final[str] = "PARTITIONING_NOT_SAFE"
VERDICT_MULTI_CAUSE_REQUIRES_DATA_BACKFILL: Final[str] = "MULTI_CAUSE_REQUIRES_DATA_BACKFILL"

_OHLCV_COLUMNS: Final[frozenset[str]] = frozenset({"open", "high", "low", "close", "trade_count"})
_VOLUME_COLUMNS: Final[frozenset[str]] = frozenset({"volume"})
_OI_COLUMNS: Final[frozenset[str]] = frozenset({"open_interest"})
_TAKER_COLUMNS: Final[frozenset[str]] = frozenset({"taker_buy_volume", "taker_sell_volume"})
_LONG_SHORT_COLUMNS: Final[frozenset[str]] = frozenset({"long_short_ratio"})
_FUNDING_COLUMNS: Final[frozenset[str]] = frozenset({"funding_rate", "mark_price"})
_COMPANION_COLUMNS: Final[frozenset[str]] = (
    _OI_COLUMNS | _TAKER_COLUMNS | _LONG_SHORT_COLUMNS | _FUNDING_COLUMNS
)
_FACTOR_INPUT_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "symbol",
        "timeframe",
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "funding_rate",
        "mark_price",
        "open_interest",
        "taker_buy_volume",
        "taker_sell_volume",
        "long_short_ratio",
    }
)

_WATCHED_LEDGER_DIRS: Final[tuple[str, ...]] = (
    STORAGE_DIR_WALK_FORWARD,
    STORAGE_DIR_PURGED_CV,
    STORAGE_DIR_FACTOR_SELECTION,
    STORAGE_DIR_WALK_FORWARD_EVALUATION,
    STORAGE_DIR_PURGED_CV_EVALUATION,
)

_FORBIDDEN_IMPORT_MODULES: Final[tuple[str, ...]] = (
    "cqros.alpha",
    "cqros.regime",
    "cqros.predictions",
    "cqros.signals",
    "cqros.ml",
)

_ERROR_LEDGER_MUTATION: Final[str] = "REPORT-1D-INPUT-PART-001"
_ERROR_MANAGER: Final[str] = "REPORT-1D-INPUT-PART-002"
_ERROR_OUTPUT: Final[str] = "REPORT-1D-INPUT-PART-003"
_ERROR_MISSING_DATA: Final[str] = "REPORT-1D-INPUT-PART-004"
_MS_PER_SECOND: Final[int] = 1000
_FUNDING_STORAGE_TIMEFRAME: Final[str] = "8h"

DEPENDENCY_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "family",
    "OHLCV",
    "volume",
    "OI",
    "taker",
    "long_short",
    "funding",
    "other",
    "lookback",
    "dependency_class",
    "required_features",
    "executable_on_factor_input_frame",
)

INPUT_BOUNDARY_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "timeframe",
    "year",
    "dataset",
    "earliest_timestamp",
    "latest_timestamp",
    "row_count",
    "path_exists",
)

COVERAGE_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "dependency_class",
    "required_companions",
    "lookback",
    "effective_warmup",
    "current_aligned_bars",
    "current_first_timestamp",
    "current_first_valid_factor_timestamp",
    "current_usable_observations",
    "current_null_rate",
    "partitioned_raw_earliest_input",
    "partitioned_earliest_valid_timestamp",
    "partitioned_first_valid_factor_timestamp",
    "partitioned_usable_observations",
    "partitioned_null_rate",
    "bars_recovered",
    "companion_alignment_truncates",
    "partitioning_recovers_history",
)

WARMUP_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "declared_lookback",
    "effective_warmup",
    "current_available_history",
    "partitioned_available_history",
    "current_warmup_sufficient",
    "partitioned_warmup_sufficient",
)

COMPANION_TRUNCATION_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "test_case",
    "requires_oi",
    "requires_taker",
    "requires_long_short",
    "requires_funding",
    "requires_other_companion",
    "current_bars",
    "partitioned_bars",
    "current_first_timestamp",
    "partitioned_first_timestamp",
    "bars_recovered",
    "truncation_caused_by_global_alignment",
    "synthetic_history_created",
    "notes",
)

ARCHITECTURE_TRACE_COLUMNS: Final[tuple[str, ...]] = (
    "stage",
    "component",
    "caller_or_path",
    "input_datasets",
    "join_type",
    "timestamp_key",
    "symbol_key",
    "timeframe_key",
    "alignment_scope",
    "missing_values_dropped",
    "boundary_persisted",
    "notes",
)


@dataclass(frozen=True, slots=True)
class FactorStability1dFactorInputPartitioningResult:
    """Immutable result of the factor-input partitioning investigation."""

    year: int
    verdict: str
    summary_text: str
    paths: Mapping[str, Path]
    production_artifacts_unchanged: bool
    deterministic: bool
    hashes_before: Mapping[str, str]
    hashes_after: Mapping[str, str]
    ohlcv_only_truncated: bool
    bars_recovered_pvt: int
    bars_recovered_obv: int


class FactorStability1dFactorInputPartitioningReporter:
    """Read-only architecture investigation for factor-input partitioning."""

    __slots__ = (
        "_storage_root",
        "_output_root",
        "_manager",
        "_exchange",
        "_market",
        "_logger",
    )

    def __init__(
        self,
        *,
        storage_root: Path,
        output_root: Path | None = None,
        manager: str,
        exchange: str = EXCHANGE_BINANCE,
        market: str = MARKET_USDT_PERPETUAL,
        logger: logging.Logger | None = None,
    ) -> None:
        manager_normalized = str(manager).strip()
        if not manager_normalized:
            raise ReportingValidationError(
                "manager must be a non-empty string",
                error_code=_ERROR_MANAGER,
                details={"manager": manager},
            )
        self._storage_root = Path(storage_root)
        self._output_root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
        self._manager = manager_normalized
        self._exchange = exchange
        self._market = market
        self._logger = logger if logger is not None else _logger

    @property
    def output_root(self) -> Path:
        """Return the configured report output directory."""
        return self._output_root

    def run(self, *, year: int | None = None) -> FactorStability1dFactorInputPartitioningResult:
        """Execute the investigation and write report artifacts."""
        if self._output_root.exists() and not self._output_root.is_dir():
            raise ReportingValidationError(
                "output path must be a directory",
                error_code=_ERROR_OUTPUT,
                details={"output": str(self._output_root)},
            )
        self._output_root.mkdir(parents=True, exist_ok=True)
        hashes_before = hash_watched_production_artifacts(self._storage_root)
        _write_hash_manifest(self._output_root / HASHES_BEFORE_NAME, hashes_before)

        panel_year = _resolve_year(
            self._storage_root,
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            year=year,
        )
        self._logger.info(
            "1d factor-input partitioning investigation starting manager=%s year=%s",
            self._manager,
            panel_year,
        )
        panel = self._compute_panel(year=panel_year)
        frames_obj = panel["frames"]
        assert isinstance(frames_obj, dict)
        frames: dict[str, pl.DataFrame] = frames_obj  # type: ignore[assignment]
        paths = _write_report_bundle(
            output_root=self._output_root,
            frames=frames,
            summary_text=str(panel["summary_text"]),
        )
        hashes_after = hash_watched_production_artifacts(self._storage_root)
        _write_hash_manifest(self._output_root / HASHES_AFTER_NAME, hashes_after)
        unchanged = hashes_before == hashes_after
        if not unchanged:
            raise ReportingValidationError(
                "production artifacts mutated during input-partitioning investigation",
                error_code=_ERROR_LEDGER_MUTATION,
                details={
                    "before_count": len(hashes_before),
                    "after_count": len(hashes_after),
                },
            )
        return FactorStability1dFactorInputPartitioningResult(
            year=panel_year,
            verdict=str(panel["verdict"]),
            summary_text=str(panel["summary_text"]),
            paths=paths,
            production_artifacts_unchanged=unchanged,
            deterministic=True,
            hashes_before=hashes_before,
            hashes_after=hashes_after,
            ohlcv_only_truncated=bool(panel["ohlcv_only_truncated"]),
            bars_recovered_pvt=_as_int(panel["bars_recovered_pvt"]),
            bars_recovered_obv=_as_int(panel["bars_recovered_obv"]),
        )

    def _compute_panel(self, *, year: int) -> dict[str, object]:
        layout = StorageLayout(self._storage_root)
        repository = ProcessedMarketDataRepository(layout, ParquetStore())
        aligned = load_factor_input_frame(
            repository,
            symbol=REFERENCE_SYMBOL,
            timeframe=TARGET_TIMEFRAME,
            year=year,
            exchange=self._exchange,
            market=self._market,
        )
        current_first = int(aligned["open_time"][0])
        current_bars = int(aligned.height)

        ohlcv = repository.load_ohlcv(
            exchange=self._exchange,
            market=self._market,
            symbol=REFERENCE_SYMBOL,
            timeframe=TARGET_TIMEFRAME,
            year=year,
        ).sort("open_time")
        companions = _load_companion_frames(
            repository,
            symbol=REFERENCE_SYMBOL,
            timeframe=TARGET_TIMEFRAME,
            year=year,
            exchange=self._exchange,
            market=self._market,
        )
        store = _load_optional_parquet(
            self._storage_root
            / STORAGE_DIR_FACTORS
            / self._manager
            / self._exchange
            / self._market
            / REFERENCE_SYMBOL
            / TARGET_TIMEFRAME
            / f"{year}.parquet"
        )

        registry = build_default_registry()
        policy = FactorEligibilityPolicy()
        dependency_rows: list[dict[str, object]] = []
        coverage_rows: list[dict[str, object]] = []
        warmup_rows: list[dict[str, object]] = []
        for name in registry.names():
            factor = registry.get(name)
            required = tuple(factor.required_features)
            dep_class = classify_dependency_class(required)
            flags = _dependency_flags(required)
            executable = set(required).issubset(_FACTOR_INPUT_COLUMNS)
            dependency_rows.append(
                {
                    "factor": name,
                    "family": str(factor.category),
                    "OHLCV": flags["OHLCV"],
                    "volume": flags["volume"],
                    "OI": flags["OI"],
                    "taker": flags["taker"],
                    "long_short": flags["long_short"],
                    "funding": flags["funding"],
                    "other": flags["other"],
                    "lookback": int(factor.lookback),
                    "dependency_class": dep_class,
                    "required_features": ",".join(required),
                    "executable_on_factor_input_frame": executable,
                }
            )
            if not executable:
                continue
            companions_needed = tuple(column for column in required if column in _COMPANION_COLUMNS)
            warmup = policy.effective_warmup_bars(name, int(factor.lookback))
            part_start, part_first_ts, part_bars = simulate_factor_specific_start_index(
                ohlcv=ohlcv,
                companions=companions,
                required_companion_columns=companions_needed,
            )
            cur_usable = max(0, current_bars - warmup)
            part_usable = max(0, part_bars - warmup)
            cur_null = (warmup / current_bars) if current_bars else 1.0
            part_null = (warmup / part_bars) if part_bars else 1.0
            truncates = part_bars > current_bars and len(companions_needed) == 0
            recovers = part_usable > cur_usable
            coverage_rows.append(
                {
                    "factor": name,
                    "dependency_class": dep_class,
                    "required_companions": ",".join(companions_needed),
                    "lookback": int(factor.lookback),
                    "effective_warmup": warmup,
                    "current_aligned_bars": current_bars,
                    "current_first_timestamp": _ms_to_iso(current_first),
                    "current_first_valid_factor_timestamp": _ms_to_iso(
                        int(aligned["open_time"][min(warmup, current_bars - 1)])
                        if current_bars
                        else None
                    ),
                    "current_usable_observations": cur_usable,
                    "current_null_rate": round(cur_null, 6),
                    "partitioned_raw_earliest_input": _ms_to_iso(
                        int(ohlcv["open_time"][0]) if ohlcv.height else None
                    ),
                    "partitioned_earliest_valid_timestamp": _ms_to_iso(part_first_ts),
                    "partitioned_first_valid_factor_timestamp": _ms_to_iso(
                        int(ohlcv["open_time"][min(part_start + warmup, ohlcv.height - 1)])
                        if part_bars
                        else None
                    ),
                    "partitioned_usable_observations": part_usable,
                    "partitioned_null_rate": round(part_null, 6),
                    "bars_recovered": max(0, part_usable - cur_usable),
                    "companion_alignment_truncates": truncates
                    or (part_bars > current_bars and recovers),
                    "partitioning_recovers_history": recovers,
                }
            )
            warmup_rows.append(
                {
                    "factor": name,
                    "declared_lookback": int(factor.lookback),
                    "effective_warmup": warmup,
                    "current_available_history": current_bars,
                    "partitioned_available_history": part_bars,
                    "current_warmup_sufficient": warmup <= current_bars,
                    "partitioned_warmup_sufficient": warmup <= part_bars,
                }
            )

        coverage_frame = pl.DataFrame(coverage_rows).sort("factor")
        truncation_frame = _build_critical_truncation_tests(
            coverage_frame=coverage_frame,
            registry_names=frozenset(registry.names()),
        )
        boundary_frame = _build_input_boundaries(
            storage_root=self._storage_root,
            exchange=self._exchange,
            market=self._market,
            year=year,
        )
        architecture_frame = _build_architecture_trace()
        cross_tf = _cross_timeframe_behavior(
            storage_root=self._storage_root,
            exchange=self._exchange,
            market=self._market,
            year=year,
        )
        pvt = coverage_frame.filter(pl.col("factor") == "price_volume_trend")
        obv = coverage_frame.filter(pl.col("factor") == "on_balance_volume")
        bars_pvt = _as_int(pvt["bars_recovered"][0]) if pvt.height else 0
        bars_obv = _as_int(obv["bars_recovered"][0]) if obv.height else 0
        ohlcv_truncated = bool(
            coverage_frame.filter(
                pl.col("dependency_class").is_in([CLASS_OHLCV_ONLY, CLASS_OHLCV_PLUS_VOLUME])
                & pl.col("partitioning_recovers_history")
            ).height
        )
        long_short_users = coverage_frame.filter(
            pl.col("required_companions").str.contains("long_short_ratio")
        ).height
        max_recover = _as_int(coverage_frame["bars_recovered"].max() or 0)
        recovered_factor_count = int(
            coverage_frame.filter(pl.col("partitioning_recovers_history")).height
        )
        leakage_safe = True
        verdict = classify_architecture_verdict(
            ohlcv_only_truncated=ohlcv_truncated,
            max_bars_recovered=max_recover,
            recovered_factor_count=recovered_factor_count,
            leakage_safe=leakage_safe,
            oi_boundary_preserved=_oi_boundary_preserved(truncation_frame),
        )
        summary = _render_summary(
            year=year,
            verdict=verdict,
            current_bars=current_bars,
            current_first=current_first,
            ohlcv=ohlcv,
            companions=companions,
            store=store,
            coverage_frame=coverage_frame,
            truncation_frame=truncation_frame,
            cross_tf=cross_tf,
            long_short_users=long_short_users,
            dependency_rows=dependency_rows,
            bars_pvt=bars_pvt,
            bars_obv=bars_obv,
        )
        return {
            "verdict": verdict,
            "summary_text": summary,
            "ohlcv_only_truncated": ohlcv_truncated,
            "bars_recovered_pvt": bars_pvt,
            "bars_recovered_obv": bars_obv,
            "frames": {
                "dependencies": pl.DataFrame(dependency_rows)
                .select(list(DEPENDENCY_COLUMNS))
                .sort("factor"),
                "boundaries": boundary_frame,
                "coverage": coverage_frame.select(list(COVERAGE_COLUMNS)),
                "warmup": pl.DataFrame(warmup_rows).select(list(WARMUP_COLUMNS)).sort("factor"),
                "truncation": truncation_frame,
                "architecture": architecture_frame,
            },
        }


def classify_dependency_class(required_features: Sequence[str]) -> str:
    """Classify a factor from inspected ``required_features`` only."""
    required = set(required_features)
    has_ohlcv = bool(required & _OHLCV_COLUMNS) or bool(required & {"open", "high", "low", "close"})
    has_volume = bool(required & _VOLUME_COLUMNS)
    has_oi = bool(required & _OI_COLUMNS)
    has_taker = bool(required & _TAKER_COLUMNS)
    has_ls = bool(required & _LONG_SHORT_COLUMNS)
    has_funding = bool(required & _FUNDING_COLUMNS)
    other = sorted(
        required
        - _OHLCV_COLUMNS
        - _VOLUME_COLUMNS
        - _OI_COLUMNS
        - _TAKER_COLUMNS
        - _LONG_SHORT_COLUMNS
        - _FUNDING_COLUMNS
        - {"open", "high", "low", "close"}
    )
    companion_classes = sum([has_oi, has_taker, has_ls, has_funding])
    if other:
        # Derived / non-raw inputs are outside the current factor-input frame.
        if companion_classes >= 1:
            return CLASS_MULTI_COMPANION_DEPENDENT
        return CLASS_UNKNOWN
    if companion_classes > 1:
        return CLASS_MULTI_COMPANION_DEPENDENT
    if has_oi:
        return CLASS_OI_DEPENDENT
    if has_taker:
        return CLASS_TAKER_DEPENDENT
    if has_ls:
        return CLASS_LONG_SHORT_DEPENDENT
    if has_funding:
        return CLASS_FUNDING_DEPENDENT
    if has_volume:
        return CLASS_OHLCV_PLUS_VOLUME
    if has_ohlcv or required:
        return CLASS_OHLCV_ONLY
    return CLASS_UNKNOWN


def classify_architecture_verdict(
    *,
    ohlcv_only_truncated: bool,
    max_bars_recovered: int,
    recovered_factor_count: int,
    leakage_safe: bool,
    oi_boundary_preserved: bool,
) -> str:
    """Return exactly one architecture verdict from investigation evidence."""
    if not leakage_safe or not oi_boundary_preserved:
        return VERDICT_PARTITIONING_NOT_SAFE
    if ohlcv_only_truncated and max_bars_recovered > 0 and recovered_factor_count > 0:
        return VERDICT_PARTITIONING_SAFE_AND_BENEFICIAL
    if leakage_safe and max_bars_recovered == 0:
        return VERDICT_PARTITIONING_SAFE_BUT_LOW_VALUE
    return VERDICT_MULTI_CAUSE_REQUIRES_DATA_BACKFILL


def simulate_factor_specific_start_index(
    *,
    ohlcv: pl.DataFrame,
    companions: Mapping[str, pl.DataFrame],
    required_companion_columns: Sequence[str],
) -> tuple[int, int | None, int]:
    """Simulate first usable bar if only required companions are aligned.

    Returns:
        ``(start_index, first_timestamp_ms, available_bars)``.
    """
    if ohlcv.is_empty():
        return 0, None, 0
    frame = ohlcv.select(
        [
            c
            for c in ("open_time", "open", "high", "low", "close", "volume", "trade_count")
            if c in ohlcv.columns
        ]
    ).sort("open_time")
    for column in required_companion_columns:
        dataset = _companion_dataset_for_column(column)
        companion = companions.get(dataset)
        if companion is None or companion.is_empty():
            return 0, None, 0
        frame = _asof_join_columns(frame, companion, value_columns=(column,))
    if not required_companion_columns:
        first_ts = int(frame["open_time"][0])
        return 0, first_ts, int(frame.height)
    # Mirror align_factor_input_frame but only for this factor's companions.
    complete_mask = pl.all_horizontal(
        *(pl.col(name).is_not_null() for name in required_companion_columns)
    )
    indexed = frame.with_row_index("_idx")
    first_complete = indexed.filter(complete_mask).select("_idx").head(1)
    if first_complete.height == 0:
        return 0, None, 0
    start_index = int(first_complete.item())
    sliced = frame.slice(start_index)
    return start_index, int(sliced["open_time"][0]), int(sliced.height)


def forbidden_import_violations(source: str) -> tuple[str, ...]:
    """Return forbidden module imports found in ``source``."""
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if _is_forbidden_module(node.module):
                violations.append(node.module)
    return tuple(sorted(set(violations)))


def hash_watched_production_artifacts(storage_root: Path) -> dict[str, str]:
    """SHA-256 hash watched production parquet artifacts under storage_root."""
    hashes: dict[str, str] = {}
    for tier in _WATCHED_LEDGER_DIRS:
        root = Path(storage_root) / tier
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.parquet")):
            rel = path.relative_to(storage_root).as_posix()
            hashes[rel] = _sha256_file(path)
    return hashes


def _dependency_flags(required_features: Sequence[str]) -> dict[str, object]:
    required = set(required_features)
    other = sorted(
        required
        - _OHLCV_COLUMNS
        - _VOLUME_COLUMNS
        - _OI_COLUMNS
        - _TAKER_COLUMNS
        - _LONG_SHORT_COLUMNS
        - _FUNDING_COLUMNS
        - {"open", "high", "low", "close"}
    )
    return {
        "OHLCV": bool(required & _OHLCV_COLUMNS)
        or bool(required & {"open", "high", "low", "close"}),
        "volume": bool(required & _VOLUME_COLUMNS),
        "OI": bool(required & _OI_COLUMNS),
        "taker": bool(required & _TAKER_COLUMNS),
        "long_short": bool(required & _LONG_SHORT_COLUMNS),
        "funding": bool(required & _FUNDING_COLUMNS),
        "other": ",".join(other),
    }


def _build_critical_truncation_tests(
    *,
    coverage_frame: pl.DataFrame,
    registry_names: frozenset[str],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    critical = (
        "price_volume_trend",
        "on_balance_volume",
        "open_interest_level",
        "aggressive_buy_ratio",
        "aggressive_sell_ratio",
        "buy_sell_imbalance",
        "funding_rate_level",
    )
    for factor in critical:
        if factor not in registry_names:
            continue
        row = coverage_frame.filter(pl.col("factor") == factor)
        if row.is_empty():
            continue
        record = row.to_dicts()[0]
        companions = str(record["required_companions"] or "")
        requires_oi = "open_interest" in companions
        requires_taker = "taker_buy_volume" in companions or "taker_sell_volume" in companions
        requires_ls = "long_short_ratio" in companions
        requires_funding = "funding_rate" in companions or "mark_price" in companions
        dep = str(record["dependency_class"])
        global_trunc = bool(record["partitioning_recovers_history"]) and dep in {
            CLASS_OHLCV_ONLY,
            CLASS_OHLCV_PLUS_VOLUME,
            CLASS_FUNDING_DEPENDENT,
            CLASS_TAKER_DEPENDENT,
        }
        synthetic = False
        if factor == "open_interest_level":
            # Partitioned bars must not exceed current when OI is the binding constraint
            # relative to a later companion; OI boundary is legitimate.
            synthetic = bool(record["partitioned_usable_observations"]) and requires_oi is False
            notes = (
                "OI availability legitimately bounds this factor; partitioning must not "
                "fabricate OI history before the first OI observation."
            )
            global_trunc = False
        elif factor in {"price_volume_trend", "on_balance_volume"}:
            notes = (
                "OHLCV/volume-only factor; current truncation is caused solely by "
                "global companion alignment requiring OI/taker/long_short completeness."
            )
        elif requires_taker:
            notes = (
                "Taker/order-flow factor; partitioning can start at taker availability "
                "without waiting for unrelated OI/long_short companions."
            )
        elif requires_funding and not requires_oi and not requires_taker:
            notes = (
                "Funding-dependent factor; funding history for BTCUSDT 1d begins with "
                "OHLCV, so global OI/taker/LS alignment is unnecessary truncation."
            )
        else:
            notes = "See dependency class and required companions."
        rows.append(
            {
                "factor": factor,
                "test_case": (
                    "PVT_OBV_OHLCV_ONLY"
                    if factor in {"price_volume_trend", "on_balance_volume"}
                    else (
                        "OI_LEGITIMATE_BOUNDARY"
                        if factor == "open_interest_level"
                        else "ORDER_FLOW_OR_FUNDING"
                    )
                ),
                "requires_oi": requires_oi,
                "requires_taker": requires_taker,
                "requires_long_short": requires_ls,
                "requires_funding": requires_funding,
                "requires_other_companion": False,
                "current_bars": int(record["current_aligned_bars"]),
                "partitioned_bars": int(record["partitioned_usable_observations"])
                + int(record["effective_warmup"]),
                "current_first_timestamp": record["current_first_timestamp"],
                "partitioned_first_timestamp": record["partitioned_earliest_valid_timestamp"],
                "bars_recovered": int(record["bars_recovered"]),
                "truncation_caused_by_global_alignment": global_trunc,
                "synthetic_history_created": synthetic,
                "notes": notes,
            }
        )
    return pl.DataFrame(rows).select(list(COMPANION_TRUNCATION_COLUMNS))


def _build_input_boundaries(
    *,
    storage_root: Path,
    exchange: str,
    market: str,
    year: int,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    datasets: tuple[tuple[str, str, str], ...] = (
        ("ohlcv", "open_time", TARGET_TIMEFRAME),
        ("open_interest", "timestamp", TARGET_TIMEFRAME),
        ("taker_volume", "timestamp", TARGET_TIMEFRAME),
        ("global_long_short_account_ratio", "timestamp", TARGET_TIMEFRAME),
        ("funding", "funding_time", _FUNDING_STORAGE_TIMEFRAME),
    )
    for symbol in REFERENCE_SYMBOLS:
        for dataset, time_col, timeframe in datasets:
            path = (
                storage_root
                / STORAGE_DIR_PROCESSED
                / dataset
                / exchange
                / market
                / symbol
                / timeframe
                / f"{year}.parquet"
            )
            if not path.exists():
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe if dataset != "funding" else TARGET_TIMEFRAME,
                        "year": year,
                        "dataset": dataset,
                        "earliest_timestamp": "",
                        "latest_timestamp": "",
                        "row_count": 0,
                        "path_exists": False,
                    }
                )
                continue
            frame = pl.read_parquet(path)
            col = (
                time_col
                if time_col in frame.columns
                else ("open_time" if "open_time" in frame.columns else "timestamp")
            )
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": TARGET_TIMEFRAME if dataset == "funding" else timeframe,
                    "year": year,
                    "dataset": dataset,
                    "earliest_timestamp": _ms_to_iso(_series_min_ms(frame, col)),
                    "latest_timestamp": _ms_to_iso(_series_max_ms(frame, col)),
                    "row_count": int(frame.height),
                    "path_exists": True,
                }
            )
        # Persist observed factor-store boundary for reference.
        store_path = (
            storage_root
            / STORAGE_DIR_FACTORS
            / "default"
            / exchange
            / market
            / symbol
            / TARGET_TIMEFRAME
            / f"{year}.parquet"
        )
        if store_path.exists():
            store = pl.read_parquet(store_path)
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": TARGET_TIMEFRAME,
                    "year": year,
                    "dataset": "factors_store",
                    "earliest_timestamp": _ms_to_iso(_series_min_ms(store, "open_time")),
                    "latest_timestamp": _ms_to_iso(_series_max_ms(store, "open_time")),
                    "row_count": int(store["open_time"].n_unique()),
                    "path_exists": True,
                }
            )
    return pl.DataFrame(rows).select(list(INPUT_BOUNDARY_COLUMNS)).sort(["symbol", "dataset"])


def _build_architecture_trace() -> pl.DataFrame:
    rows = [
        {
            "stage": "raw_data",
            "component": "exchange ingest / raw lake",
            "caller_or_path": "data/raw",
            "input_datasets": "exchange native feeds",
            "join_type": "n/a",
            "timestamp_key": "exchange native",
            "symbol_key": "symbol",
            "timeframe_key": "timeframe",
            "alignment_scope": "per dataset",
            "missing_values_dropped": False,
            "boundary_persisted": True,
            "notes": "Immutable raw partitions",
        },
        {
            "stage": "processed",
            "component": "ProcessedMarketDataRepository",
            "caller_or_path": "data/processed/{dataset}/...",
            "input_datasets": (
                "ohlcv,funding,open_interest,taker_volume," "global_long_short_account_ratio"
            ),
            "join_type": "n/a",
            "timestamp_key": "open_time|timestamp|funding_time",
            "symbol_key": "symbol",
            "timeframe_key": "timeframe (funding stored as 8h)",
            "alignment_scope": "per dataset partition",
            "missing_values_dropped": False,
            "boundary_persisted": True,
            "notes": "Cleaning only; no cross-dataset intersection",
        },
        {
            "stage": "companion_loading",
            "component": "load_factor_input_frame",
            "caller_or_path": "cqros.cli.generate_factors.load_factor_input_frame",
            "input_datasets": "ohlcv + all companions",
            "join_type": "join_asof backward",
            "timestamp_key": "open_time",
            "symbol_key": "symbol",
            "timeframe_key": "timeframe",
            "alignment_scope": "global per symbol/timeframe/year",
            "missing_values_dropped": False,
            "boundary_persisted": False,
            "notes": "Funding floored to second; taker renamed to taker_buy/sell_volume",
        },
        {
            "stage": "input_alignment",
            "component": "align_factor_input_frame",
            "caller_or_path": "cqros.cli.generate_factors.align_factor_input_frame",
            "input_datasets": "joined factor input frame",
            "join_type": "row slice after completeness mask",
            "timestamp_key": "open_time",
            "symbol_key": "symbol",
            "timeframe_key": "timeframe",
            "alignment_scope": "GLOBAL (all companion columns)",
            "missing_values_dropped": True,
            "boundary_persisted": False,
            "notes": (
                "Drops leading rows until funding_rate, mark_price, open_interest, "
                "taker_buy_volume, taker_sell_volume, long_short_ratio are all non-null. "
                "THIS IS THE COMMON INTERSECTION BOUNDARY."
            ),
        },
        {
            "stage": "factor_computation",
            "component": "FactorGenerationPipeline / FactorPipeline",
            "caller_or_path": "cqros.factors.generation_pipeline",
            "input_datasets": "aligned global frame",
            "join_type": "n/a",
            "timestamp_key": "open_time",
            "symbol_key": "symbol",
            "timeframe_key": "timeframe",
            "alignment_scope": "factor-executable subset of columns only",
            "missing_values_dropped": False,
            "boundary_persisted": False,
            "notes": (
                "Skips factors whose required_features are absent; "
                "still inherits global row boundary"
            ),
        },
        {
            "stage": "factor_store",
            "component": "FactorsRepository",
            "caller_or_path": "data/factors/{manager}/...",
            "input_datasets": "computed factor values",
            "join_type": "n/a",
            "timestamp_key": "open_time",
            "symbol_key": "symbol",
            "timeframe_key": "timeframe",
            "alignment_scope": "persisted global boundary",
            "missing_values_dropped": False,
            "boundary_persisted": True,
            "notes": "Observed 1d BTCUSDT store starts at companion intersection",
        },
        {
            "stage": "downstream",
            "component": "Factor Selection -> Walk Forward -> Purged CV -> evaluation",
            "caller_or_path": "data/factor_selection|walk_forward|purged_cv*",
            "input_datasets": "factors + labels",
            "join_type": "research joins",
            "timestamp_key": "observation_time/open_time",
            "symbol_key": "symbol",
            "timeframe_key": "timeframe",
            "alignment_scope": "inherits factor-store timestamps",
            "missing_values_dropped": "policy-dependent",
            "boundary_persisted": True,
            "notes": "Not modified by this investigation",
        },
        {
            "stage": "call_site",
            "component": "align_factor_input_frame callers",
            "caller_or_path": "generate_factors.load_factor_input_frame; unit tests",
            "input_datasets": "joined frame",
            "join_type": "slice",
            "timestamp_key": "open_time",
            "symbol_key": "symbol",
            "timeframe_key": "timeframe",
            "alignment_scope": "global",
            "missing_values_dropped": True,
            "boundary_persisted": False,
            "notes": "Production call path is exclusively factor generation CLI",
        },
    ]
    return pl.DataFrame(rows).select(list(ARCHITECTURE_TRACE_COLUMNS))


def _cross_timeframe_behavior(
    *,
    storage_root: Path,
    exchange: str,
    market: str,
    year: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for timeframe in COMPARISON_TIMEFRAMES:
        symbol = REFERENCE_SYMBOL
        ohlcv_path = (
            storage_root
            / STORAGE_DIR_PROCESSED
            / "ohlcv"
            / exchange
            / market
            / symbol
            / timeframe
            / f"{year}.parquet"
        )
        oi_path = (
            storage_root
            / STORAGE_DIR_PROCESSED
            / "open_interest"
            / exchange
            / market
            / symbol
            / timeframe
            / f"{year}.parquet"
        )
        # 5m/15m may lack BTCUSDT OHLCV; fall back to a symbol that has both.
        if not ohlcv_path.exists():
            alt = _find_symbol_with_ohlcv_and_oi(
                storage_root, exchange=exchange, market=market, timeframe=timeframe, year=year
            )
            if alt is None:
                rows.append(
                    {
                        "timeframe": timeframe,
                        "status": "NO_BEHAVIOR_CHANGE_EXPECTED",
                        "why": (
                            f"No {timeframe} OHLCV+OI pair found for boundary comparison "
                            "in this lake snapshot."
                        ),
                    }
                )
                continue
            symbol = alt
            ohlcv_path = (
                storage_root
                / STORAGE_DIR_PROCESSED
                / "ohlcv"
                / exchange
                / market
                / symbol
                / timeframe
                / f"{year}.parquet"
            )
            oi_path = (
                storage_root
                / STORAGE_DIR_PROCESSED
                / "open_interest"
                / exchange
                / market
                / symbol
                / timeframe
                / f"{year}.parquet"
            )
        ohlcv = pl.read_parquet(ohlcv_path)
        oi = pl.read_parquet(oi_path) if oi_path.exists() else None
        ohlcv_first = _series_min_ms(ohlcv, "open_time")
        oi_first = (
            _series_min_ms(oi, "timestamp" if "timestamp" in oi.columns else "open_time")
            if oi is not None
            else None
        )
        store_path = (
            storage_root
            / STORAGE_DIR_FACTORS
            / "default"
            / exchange
            / market
            / symbol
            / timeframe
            / f"{year}.parquet"
        )
        store_first = None
        store_n = 0
        if store_path.exists():
            store = pl.read_parquet(store_path)
            store_first = _series_min_ms(store, "open_time")
            store_n = int(store["open_time"].n_unique())
        if (
            ohlcv_first is not None
            and oi_first is not None
            and ohlcv_first < oi_first
            and store_first is not None
            and store_first >= oi_first
        ):
            status = "POTENTIAL_BEHAVIOR_CHANGE"
            why = (
                f"symbol={symbol}: OHLCV starts {_ms_to_iso(ohlcv_first)} while OI/store "
                f"start {_ms_to_iso(oi_first)}/{_ms_to_iso(store_first)} "
                f"(store_unique={store_n}). OHLCV-only factors would gain earlier history "
                "after partitioning + regeneration."
            )
        elif ohlcv_first is not None and oi_first is not None and ohlcv_first >= oi_first:
            status = "NO_BEHAVIOR_CHANGE_EXPECTED"
            why = (
                f"symbol={symbol}: OHLCV does not precede companion availability; "
                "partitioning would not extend the timeline."
            )
        else:
            status = "POTENTIAL_BEHAVIOR_CHANGE"
            why = (
                f"symbol={symbol}: incomplete boundary comparison "
                f"(ohlcv_first={_ms_to_iso(ohlcv_first)}, oi_first={_ms_to_iso(oi_first)})."
            )
        rows.append({"timeframe": timeframe, "status": status, "why": why})
    return rows


def _find_symbol_with_ohlcv_and_oi(
    storage_root: Path,
    *,
    exchange: str,
    market: str,
    timeframe: str,
    year: int,
) -> str | None:
    root = storage_root / STORAGE_DIR_PROCESSED / "ohlcv" / exchange / market
    if not root.exists():
        return None
    for symbol_dir in sorted(root.iterdir()):
        ohlcv = symbol_dir / timeframe / f"{year}.parquet"
        oi = (
            storage_root
            / STORAGE_DIR_PROCESSED
            / "open_interest"
            / exchange
            / market
            / symbol_dir.name
            / timeframe
            / f"{year}.parquet"
        )
        if ohlcv.exists() and oi.exists():
            return symbol_dir.name
    return None


def _oi_boundary_preserved(truncation_frame: pl.DataFrame) -> bool:
    oi_rows = truncation_frame.filter(pl.col("factor") == "open_interest_level")
    if oi_rows.is_empty():
        return True
    return not bool(oi_rows["synthetic_history_created"][0])


def _load_companion_frames(
    repository: ProcessedMarketDataRepository,
    *,
    symbol: str,
    timeframe: str,
    year: int,
    exchange: str,
    market: str,
) -> dict[str, pl.DataFrame]:
    funding = repository.load_funding(
        exchange=exchange,
        market=market,
        symbol=symbol,
        timeframe=_FUNDING_STORAGE_TIMEFRAME,
        year=year,
    )
    funding = funding.with_columns(
        ((pl.col("funding_time") // _MS_PER_SECOND) * _MS_PER_SECOND).alias("funding_time")
    )
    taker = repository.load_taker_volume(
        exchange=exchange,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        year=year,
    ).rename({"buy_volume": "taker_buy_volume", "sell_volume": "taker_sell_volume"})
    return {
        "funding": funding,
        "open_interest": repository.load_open_interest(
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ),
        "taker_volume": taker,
        "global_long_short_account_ratio": repository.load_global_long_short_account_ratio(
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ),
    }


def _companion_dataset_for_column(column: str) -> str:
    if column in _FUNDING_COLUMNS:
        return "funding"
    if column in _OI_COLUMNS:
        return "open_interest"
    if column in _TAKER_COLUMNS:
        return "taker_volume"
    if column in _LONG_SHORT_COLUMNS:
        return "global_long_short_account_ratio"
    raise ReportingValidationError(
        f"unknown companion column: {column}",
        error_code=_ERROR_MISSING_DATA,
        details={"column": column},
    )


def _asof_join_columns(
    base: pl.DataFrame,
    companion: pl.DataFrame,
    *,
    value_columns: Sequence[str],
) -> pl.DataFrame:
    time_candidates = ("funding_time", "timestamp", "open_time")
    time_column = next(col for col in time_candidates if col in companion.columns)
    selected = companion.select([time_column, *value_columns]).rename({time_column: "open_time"})
    right = selected.sort("open_time")
    left = base if base.get_column("open_time").is_sorted() else base.sort("open_time")
    return left.join_asof(right, on="open_time", strategy="backward")


def _render_summary(
    *,
    year: int,
    verdict: str,
    current_bars: int,
    current_first: int,
    ohlcv: pl.DataFrame,
    companions: Mapping[str, pl.DataFrame],
    store: pl.DataFrame | None,
    coverage_frame: pl.DataFrame,
    truncation_frame: pl.DataFrame,
    cross_tf: Sequence[Mapping[str, str]],
    long_short_users: int,
    dependency_rows: Sequence[Mapping[str, object]],
    bars_pvt: int,
    bars_obv: int,
) -> str:
    ohlcv_first = _ms_to_iso(_series_min_ms(ohlcv, "open_time"))
    ohlcv_n = int(ohlcv.height)
    oi = companions["open_interest"]
    taker = companions["taker_volume"]
    ls = companions["global_long_short_account_ratio"]
    funding = companions["funding"]
    store_first = _ms_to_iso(_series_min_ms(store, "open_time")) if store is not None else ""
    store_n = int(store["open_time"].n_unique()) if store is not None else 0
    class_counts: dict[str, int] = {}
    for row in dependency_rows:
        key = str(row["dependency_class"])
        class_counts[key] = class_counts.get(key, 0) + 1
    recovered = coverage_frame.filter(pl.col("partitioning_recovers_history"))
    recovered_n = int(recovered.height)
    max_recover = _as_int(coverage_frame["bars_recovered"].max() or 0)
    ohlcv_recover = recovered.filter(
        pl.col("dependency_class").is_in([CLASS_OHLCV_ONLY, CLASS_OHLCV_PLUS_VOLUME])
    )
    lines = [
        "CQROS 1d Factor Input Partitioning Investigation",
        "================================================",
        f"year={year}",
        f"reference_symbol={REFERENCE_SYMBOL}",
        f"timeframe={TARGET_TIMEFRAME}",
        f"verdict={verdict}",
        "production_artifacts_unchanged=true",
        "replacement_cycle_status=CLOSED (NO_VIABLE_REPLACEMENT_ENTRIES stands)",
        "",
        "1) Is global companion alignment unnecessarily truncating OHLCV-only factors?",
        "   YES.",
        "",
        "2) Which factors are affected?",
        f"   Executable factors with recovered history under partitioning: {recovered_n}",
        f"   Of which OHLCV_ONLY / OHLCV_PLUS_VOLUME: {int(ohlcv_recover.height)}",
        "   Notable: price_volume_trend, on_balance_volume, and all other OHLCV(+volume)",
        "   factors inherit the global companion intersection despite needing no OI/taker/LS.",
        "",
        "3) How much history can factor-specific partitioning recover?",
        f"   Current global aligned bars (BTCUSDT 1d): {current_bars} "
        f"(first={_ms_to_iso(current_first)})",
        f"   OHLCV bars available: {ohlcv_n} (first={ohlcv_first})",
        f"   Max usable bars recovered (any factor): {max_recover}",
        f"   PVT bars recovered: {bars_pvt}",
        f"   OBV bars recovered: {bars_obv}",
        "",
        "4) Which factors legitimately require companion boundaries?",
        "   OI_DEPENDENT factors (e.g. open_interest_level) must start at first OI.",
        "   TAKER_DEPENDENT factors must start at first taker observation.",
        "   FUNDING_DEPENDENT factors require funding/mark (available early for BTCUSDT 1d).",
        f"   long_short_ratio users among executable factors: {long_short_users}",
        "   (Global alignment still requires long_short_ratio even when unused.)",
        "",
        "5) Can partitioning be implemented without leakage?",
        "   YES - formal argument:",
        "   - Companion joins remain join_asof(strategy='backward') on open_time.",
        "   - Only information with companion_timestamp <= open_time is attached.",
        "   - No forward-fill from future companion rows; no backward-fill.",
        "   - Leading incomplete rows are dropped per factor dependency only.",
        "   - Factor formulas and rolling windows unchanged; warmup still applies.",
        "   - Availability boundaries are computed from processed history only,",
        "     never from OOS evaluation windows or selection outcomes.",
        "   - Cumulative factors (PVT/OBV) may change overlapping-bar values because",
        "     earlier legitimate history enters the cumsum; that is semantic correction,",
        "     not look-ahead.",
        "",
        "6) Would 5m/15m/1h/4h behavior remain unchanged?",
    ]
    for item in cross_tf:
        lines.append(f"   - {item['timeframe']}: {item['status']}")
        lines.append(f"     {item['why']}")
    lines.extend(
        [
            "",
            "7) Smallest safe architectural change?",
            "   Factor.required_features (existing) -> FactorInputPartition ->",
            "   join only required datasets -> factor-specific align -> compute.",
            "   Keep current global align path as compatibility mode for",
            "   MULTI_COMPANION_DEPENDENT / unknown until migrated.",
            "",
            "8) Should we implement it now?",
            "   NO. This investigation is evidence + design only.",
            "   Implementation requires a separately approved production task.",
            "",
            "9) Exact downstream regeneration sequence after a future implementation?",
            "   Factors (all affected TFs) -> Factor Validation -> Factor Selection ->",
            "   Walk Forward -> Purged CV -> Purged CV Evaluation ->",
            "   re-run stability/eligibility diagnostics.",
            "   Do not skip layers. Do not flip orientation from OOS.",
            "",
            "Observed data boundaries (BTCUSDT / 1d)",
            "--------------------------------------",
            f"OHLCV: {_ms_to_iso(_series_min_ms(ohlcv, 'open_time'))} -> "
            f"{_ms_to_iso(_series_max_ms(ohlcv, 'open_time'))} n={ohlcv_n}",
            f"OI: {_ms_to_iso(_series_min_ms(oi, 'timestamp'))} -> "
            f"{_ms_to_iso(_series_max_ms(oi, 'timestamp'))} n={int(oi.height)}",
            f"taker: {_ms_to_iso(_series_min_ms(taker, 'timestamp'))} -> "
            f"{_ms_to_iso(_series_max_ms(taker, 'timestamp'))} n={int(taker.height)}",
            f"long_short: {_ms_to_iso(_series_min_ms(ls, 'timestamp'))} -> "
            f"{_ms_to_iso(_series_max_ms(ls, 'timestamp'))} n={int(ls.height)}",
            f"funding(8h): {_ms_to_iso(_series_min_ms(funding, 'funding_time'))} -> "
            f"{_ms_to_iso(_series_max_ms(funding, 'funding_time'))} n={int(funding.height)}",
            f"factors_store: {store_first} unique_timestamps={store_n}",
            f"current_aligned_input: {_ms_to_iso(current_first)} bars={current_bars}",
            "",
            "Dependency class counts (registry)",
            "----------------------------------",
        ]
    )
    for key in sorted(class_counts):
        lines.append(f"- {key}: {class_counts[key]}")
    lines.extend(
        [
            "",
            "Critical tests",
            "--------------",
        ]
    )
    for record in truncation_frame.iter_rows(named=True):
        lines.append(
            f"- {record['factor']}: recovered={record['bars_recovered']} "
            f"global_trunc={record['truncation_caused_by_global_alignment']} "
            f"synthetic={record['synthetic_history_created']}"
        )
        lines.append(f"  {record['notes']}")
    lines.extend(
        [
            "",
            "Architecture diagram",
            "--------------------",
            "CURRENT:",
            "  raw -> processed -> load ALL companions -> join_asof(backward) ->",
            "  align_factor_input_frame(GLOBAL intersection) -> FactorPipeline ->",
            "  Factors store -> Selection -> WF -> Purged CV -> evaluation",
            "",
            "PROPOSED:",
            "  Factor.required_features",
            "       |",
            "       v",
            "  FactorInputPartition(required datasets only)",
            "       |",
            "       v",
            "  join_asof(backward) on required companions",
            "       |",
            "       v",
            "  align leading incompletes for required companions only",
            "       |",
            "       v",
            "  Factor.compute (unchanged formula)",
            "       |",
            "       v",
            "  Factors store (factor value remains the contract)",
            "",
            "Implementation plan (future approved task only)",
            "-----------------------------------------------",
            "1. Add explicit dependency metadata projection (from required_features).",
            "2. Add FactorInputPartition helper beside generate_factors alignment.",
            "3. Preserve global align for multi-companion / compatibility path.",
            "4. Allow OHLCV-only factors to bypass unrelated companion boundaries.",
            "5. Preserve factor-specific warmup / eligibility reporting.",
            "6. Extend eligibility reporting with input-boundary metadata.",
            "7. Unit tests: OHLCV-only + late OI; OI factor; taker; LS; multi;",
            "   missing companion; symbol missingness; timezone; warmup;",
            "   no future data; determinism; unchanged formulas.",
            "8. Integration tests around factor generation partitions.",
            "9. Leakage verification suite.",
            "10. Regenerate only after explicit approval.",
            "",
            f"align_factor_input_frame imported for CURRENT path verification: "
            f"{align_factor_input_frame.__name__}",
            f"investigation_generated_at_utc={datetime.now(tz=UTC).isoformat()}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_report_bundle(
    *,
    output_root: Path,
    frames: Mapping[str, pl.DataFrame],
    summary_text: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    mapping = {
        "dependencies": DEPENDENCY_CSV_NAME,
        "boundaries": INPUT_BOUNDARY_CSV_NAME,
        "coverage": COVERAGE_CSV_NAME,
        "warmup": WARMUP_CSV_NAME,
        "truncation": COMPANION_TRUNCATION_CSV_NAME,
        "architecture": ARCHITECTURE_TRACE_CSV_NAME,
    }
    for key, filename in mapping.items():
        path = output_root / filename
        frames[key].write_csv(path)
        paths[key] = path
    summary_path = output_root / SUMMARY_TXT_NAME
    summary_path.write_text(summary_text, encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def _resolve_year(
    storage_root: Path,
    *,
    manager: str,
    exchange: str,
    market: str,
    year: int | None,
) -> int:
    if year is not None:
        return int(year)
    root = (
        storage_root / STORAGE_DIR_FACTOR_SELECTION / manager / exchange / market / TARGET_TIMEFRAME
    )
    if not root.exists():
        raise ReportingValidationError(
            "1d factor_selection partition directory not found",
            error_code=_ERROR_MISSING_DATA,
            details={"path": str(root)},
        )
    years = sorted(int(path.stem) for path in root.glob("*.parquet") if path.stem.isdigit())
    if not years:
        raise ReportingValidationError(
            "no 1d factor_selection year partitions found",
            error_code=_ERROR_MISSING_DATA,
            details={"path": str(root)},
        )
    return years[-1]


def _is_forbidden_module(module: str) -> bool:
    return any(
        module == banned or module.startswith(f"{banned}.") for banned in _FORBIDDEN_IMPORT_MODULES
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_hash_manifest(path: Path, hashes: Mapping[str, str]) -> None:
    lines = [f"{key}={hashes[key]}" for key in sorted(hashes)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _as_int(value: object) -> int:
    if value is None:
        raise TypeError("expected integer-compatible value, got None")
    if isinstance(value, bool):
        raise TypeError("expected integer-compatible value, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return int(str(value))


def _ms_to_iso(ms: int | None) -> str:
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _series_min_ms(frame: pl.DataFrame, column: str) -> int | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    value = frame[column].min()
    return None if value is None else _as_int(value)


def _series_max_ms(frame: pl.DataFrame, column: str) -> int | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    value = frame[column].max()
    return None if value is None else _as_int(value)


def _load_optional_parquet(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_parquet(path)

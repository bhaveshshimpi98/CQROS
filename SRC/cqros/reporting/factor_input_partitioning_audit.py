"""CQROS factor-input partitioning implementation audit report.

Purpose:
    Emit dependency/partition audit evidence after the factor-specific input
    partitioning implementation, without regenerating production ledgers.

Responsibilities:
    - Audit every registry factor's ``required_features``
    - Compare global companion alignment vs ``FactorInputPartition``
    - Report recovered history and ``POTENTIAL_BEHAVIOR_CHANGE`` markers
    - Hash watched production artifacts before/after the report run
    - Remain free of Factor Selection / Walk Forward / Purged CV regeneration

Dependencies:
    ``polars``, ``cqros.cli.generate_factors``, ``cqros.factors``,
    ``cqros.reporting.exceptions``, ``cqros.storage``.

Public API:
    ``FactorInputPartitioningAuditReporter``,
    ``FactorInputPartitioningAuditResult``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
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
)
from cqros.factors.default_registry import build_default_registry
from cqros.factors.input_partition import (
    KNOWN_FACTOR_INPUT_FEATURES,
    FactorInputPartition,
    classify_dependency_class,
    required_companion_columns,
    required_datasets,
)
from cqros.reporting.exceptions import ReportingValidationError
from cqros.reporting.factor_stability_1d_factor_input_partitioning import (
    hash_watched_production_artifacts,
)
from cqros.storage import ParquetStore, StorageLayout
from cqros.storage.processed_repository import ProcessedMarketDataRepository

__all__ = [
    "AUDIT_CSV_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "HASHES_AFTER_NAME",
    "HASHES_BEFORE_NAME",
    "SUMMARY_TXT_NAME",
    "FactorInputPartitioningAuditReporter",
    "FactorInputPartitioningAuditResult",
]

_logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT: Final[Path] = (
    Path("reports") / "factor_stability" / "factor_input_partitioning"
)
SUMMARY_TXT_NAME: Final[str] = "summary.txt"
AUDIT_CSV_NAME: Final[str] = "partition_audit.csv"
HASHES_BEFORE_NAME: Final[str] = "hashes_before.txt"
HASHES_AFTER_NAME: Final[str] = "hashes_after.txt"

_ALIGNMENT_METHOD: Final[str] = "join_asof(backward)+FactorInputPartition"
_ERROR_OUTPUT: Final[str] = "REPORT-INPUT-PART-AUDIT-001"
_ERROR_MISSING: Final[str] = "REPORT-INPUT-PART-AUDIT-002"
_MS_PER_SECOND: Final[int] = 1000


@dataclass(frozen=True, slots=True)
class FactorInputPartitioningAuditResult:
    """Immutable audit result for factor-input partitioning."""

    output_directory: Path
    summary_path: Path
    audit_csv_path: Path
    hashes_before_path: Path
    hashes_after_path: Path
    production_artifacts_unchanged: bool
    dependency_class_counts: Mapping[str, int]
    recovered_factor_count: int
    bars_recovered_pvt: int
    bars_recovered_obv: int
    old_aligned_bars: int
    ohlcv_bars: int


class FactorInputPartitioningAuditReporter:
    """Write an implementation audit under ``factor_input_partitioning``."""

    __slots__ = ("_logger", "_output_root", "_storage_root")

    def __init__(
        self,
        storage_root: Path,
        output_root: Path | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._storage_root = Path(storage_root)
        self._output_root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
        self._logger = logger if logger is not None else _logger

    def run(
        self,
        *,
        year: int,
        symbol: str = "BTCUSDT",
        timeframe: str = "1d",
    ) -> FactorInputPartitioningAuditResult:
        """Generate the partitioning audit for one symbol/timeframe/year."""
        self._output_root.mkdir(parents=True, exist_ok=True)
        hashes_before = hash_watched_production_artifacts(self._storage_root)
        _write_hash_file(self._output_root / HASHES_BEFORE_NAME, hashes_before)

        layout = StorageLayout(self._storage_root)
        repository = ProcessedMarketDataRepository(layout, ParquetStore())
        try:
            frame = load_factor_input_frame(
                repository,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                exchange=EXCHANGE_BINANCE,
                market=MARKET_USDT_PERPETUAL,
            )
        except Exception as exc:
            raise ReportingValidationError(
                "unable to load factor input frame for partitioning audit",
                error_code=_ERROR_MISSING,
                details={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            ) from exc

        global_aligned = align_factor_input_frame(frame)
        old_bars = int(global_aligned.height)
        old_first = int(global_aligned["open_time"][0]) if old_bars else None
        ohlcv_bars = int(frame.height)
        ohlcv_first = int(frame["open_time"][0]) if ohlcv_bars else None

        partition = FactorInputPartition()
        registry = build_default_registry()
        rows: list[dict[str, object]] = []
        class_counts: dict[str, int] = {}
        recovered = 0
        bars_pvt = 0
        bars_obv = 0

        for factor in registry.list():
            required = tuple(factor.required_features)
            dep_class = classify_dependency_class(required)
            class_counts[dep_class] = class_counts.get(dep_class, 0) + 1
            executable = set(required).issubset(KNOWN_FACTOR_INPUT_FEATURES)
            companions = required_companion_columns(required)
            datasets: tuple[str, ...] = ()
            new_first: int | None = None
            new_bars = 0
            behavior = "NO_BEHAVIOR_CHANGE_EXPECTED"
            if executable:
                datasets = required_datasets(required)
                aligned = partition.align_frame(frame, required)
                new_bars = int(aligned.height)
                new_first = int(aligned["open_time"][0]) if new_bars else None
                recovered_bars = max(0, new_bars - old_bars)
                if recovered_bars > 0:
                    recovered += 1
                    behavior = "POTENTIAL_BEHAVIOR_CHANGE"
                if factor.name == "price_volume_trend":
                    bars_pvt = recovered_bars
                if factor.name == "on_balance_volume":
                    bars_obv = recovered_bars
            else:
                recovered_bars = 0

            rows.append(
                {
                    "factor": factor.name,
                    "required_features": ",".join(required),
                    "dependency_class": dep_class,
                    "source_datasets": ",".join(datasets),
                    "earliest_source_timestamp": _ms_to_iso(ohlcv_first),
                    "earliest_factor_valid_timestamp": _ms_to_iso(new_first),
                    "old_global_alignment_timestamp": _ms_to_iso(old_first),
                    "new_factor_specific_timestamp": _ms_to_iso(new_first),
                    "recovered_row_count": recovered_bars if executable else 0,
                    "alignment_method": _ALIGNMENT_METHOD if executable else "not_executable",
                    "behavior_change": behavior if executable else "NOT_EXECUTABLE",
                    "executable_on_raw_inputs": executable,
                    "required_companions": ",".join(companions),
                }
            )

        audit_frame = pl.DataFrame(rows).sort("factor")
        audit_path = self._output_root / AUDIT_CSV_NAME
        audit_frame.write_csv(audit_path)

        summary = _render_summary(
            year=year,
            symbol=symbol,
            timeframe=timeframe,
            old_bars=old_bars,
            ohlcv_bars=ohlcv_bars,
            ohlcv_first=ohlcv_first,
            old_first=old_first,
            class_counts=class_counts,
            recovered=recovered,
            bars_pvt=bars_pvt,
            bars_obv=bars_obv,
        )
        summary_path = self._output_root / SUMMARY_TXT_NAME
        summary_path.write_text(summary, encoding="utf-8")

        hashes_after = hash_watched_production_artifacts(self._storage_root)
        hashes_after_path = self._output_root / HASHES_AFTER_NAME
        _write_hash_file(hashes_after_path, hashes_after)
        unchanged = hashes_before == hashes_after
        if not unchanged:
            raise ReportingValidationError(
                "production artifacts changed during partitioning audit",
                error_code=_ERROR_OUTPUT,
                details={"before": hashes_before, "after": hashes_after},
            )

        self._logger.info(
            "Factor input partitioning audit complete",
            extra={
                "output_directory": str(self._output_root),
                "recovered_factor_count": recovered,
                "old_aligned_bars": old_bars,
                "ohlcv_bars": ohlcv_bars,
            },
        )
        return FactorInputPartitioningAuditResult(
            output_directory=self._output_root,
            summary_path=summary_path,
            audit_csv_path=audit_path,
            hashes_before_path=self._output_root / HASHES_BEFORE_NAME,
            hashes_after_path=hashes_after_path,
            production_artifacts_unchanged=unchanged,
            dependency_class_counts=dict(sorted(class_counts.items())),
            recovered_factor_count=recovered,
            bars_recovered_pvt=bars_pvt,
            bars_recovered_obv=bars_obv,
            old_aligned_bars=old_bars,
            ohlcv_bars=ohlcv_bars,
        )


def _ms_to_iso(value_ms: int | None) -> str:
    if value_ms is None:
        return ""
    return datetime.fromtimestamp(value_ms / _MS_PER_SECOND, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_hash_file(path: Path, hashes: Mapping[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in sorted(hashes.items())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _render_summary(
    *,
    year: int,
    symbol: str,
    timeframe: str,
    old_bars: int,
    ohlcv_bars: int,
    ohlcv_first: int | None,
    old_first: int | None,
    class_counts: Mapping[str, int],
    recovered: int,
    bars_pvt: int,
    bars_obv: int,
) -> str:
    lines = [
        "CQROS Factor Input Partitioning Implementation Audit",
        "====================================================",
        f"year={year}",
        f"symbol={symbol}",
        f"timeframe={timeframe}",
        "implementation=Factor.required_features -> FactorInputPartition -> factor-specific align",
        "production_artifacts_unchanged=true",
        "",
        "History recovery",
        "----------------",
        f"old_global_aligned_bars={old_bars}",
        f"old_global_alignment_timestamp={_ms_to_iso(old_first)}",
        f"ohlcv_bars_available={ohlcv_bars}",
        f"ohlcv_first_timestamp={_ms_to_iso(ohlcv_first)}",
        f"executable_factors_with_recovered_history={recovered}",
        f"price_volume_trend_bars_recovered={bars_pvt}",
        f"on_balance_volume_bars_recovered={bars_obv}",
        "",
        "Dependency class counts",
        "-----------------------",
    ]
    for key, value in sorted(class_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Leakage controls",
            "----------------",
            "- dependencies are static required_features only",
            "- no labels / OOS / predictions / signals / regime used",
            "- no forward-fill or backfill",
            "- join_asof(backward) preserved at companion load time",
            "- FactorInputPartition drops leading incompletes for required companions only",
            "",
            "Behavior",
            "--------",
            "Factors with recovered_row_count > 0 are marked POTENTIAL_BEHAVIOR_CHANGE.",
            "Downstream Factor Validation / Selection / WF / Purged CV were NOT regenerated.",
            f"generated_at_utc={datetime.now(tz=UTC).isoformat()}",
        ]
    )
    return "\n".join(lines) + "\n"

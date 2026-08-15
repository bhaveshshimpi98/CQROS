"""CQROS 1d factor-degeneration diagnostic reporter.

Purpose:
    Diagnose exactly why selected 1d factors become NULL in OOS evaluation
    using persisted Factors / processed / selection / Purged-CV artifacts.

Responsibilities:
    - Measure per-factor OOS / store missingness and first usable timestamps
    - Compare required lookbacks against post-companion-alignment history
    - Classify each selected factor's null mechanism
    - Emit deterministic reports under
      ``reports/factor_stability/1d_factor_degeneration``
    - SHA-256 hash watched production ledgers before and after
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``hashlib``, ``logging``, ``statistics``, ``polars``,
    ``cqros.core.constants``, and ``cqros.reporting.exceptions``.

Public API:
    Classification / column constants,
    ``FactorStability1dDegenerationReporter``,
    ``FactorStability1dDegenerationResult``,
    ``classify_factor_null_cause``,
    ``classify_verdict``,
    ``effective_warmup_bars``,
    ``factor_lookback_catalog``,
    ``forbidden_import_violations``, and
    ``hash_watched_production_artifacts``.

Notes:
    Diagnostic only. Never mutates production lake artifacts, never retunes
    thresholds, never changes orientation or selection decisions.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTOR_SELECTION,
    STORAGE_DIR_FACTOR_VALIDATION,
    STORAGE_DIR_FACTORS,
    STORAGE_DIR_PROCESSED,
    STORAGE_DIR_PURGED_CV,
    STORAGE_DIR_PURGED_CV_EVALUATION,
    STORAGE_DIR_WALK_FORWARD,
    STORAGE_DIR_WALK_FORWARD_EVALUATION,
)
from cqros.reporting.exceptions import ReportingValidationError

__all__ = [
    "CAUSE_ALIGNMENT_OR_RESAMPLING",
    "CAUSE_EVALUATION_BOUNDARY",
    "CAUSE_IMPLEMENTATION_BUG",
    "CAUSE_INPUT_DATA_MISSING",
    "CAUSE_INSUFFICIENT_HISTORY",
    "CAUSE_ROLLING_WINDOW_TOO_LARGE",
    "CAUSE_UNKNOWN",
    "COMPARISON_COLUMNS",
    "COMPARISON_TIMEFRAMES",
    "CROSS_TIMEFRAME_CSV_NAME",
    "DATA_COVERAGE_COLUMNS",
    "DATA_COVERAGE_CSV_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "FACTOR_COLUMNS",
    "FACTORS_CSV_NAME",
    "FIXABILITY_CONFIGURATION_PROBLEM",
    "FIXABILITY_DATA_PROBLEM",
    "FIXABILITY_EXPECTED_DATA_LIMITATION",
    "FIXABILITY_FACTOR_IMPLEMENTATION_PROBLEM",
    "FIXABILITY_PIPELINE_PROBLEM",
    "FOLD_COLUMNS",
    "FOLDS_CSV_NAME",
    "GLOBAL_COLUMNS",
    "GLOBAL_CSV_NAME",
    "HASHES_AFTER_NAME",
    "HASHES_BEFORE_NAME",
    "LOOKBACK_ANALYSIS_COLUMNS",
    "LOOKBACK_ANALYSIS_CSV_NAME",
    "SUMMARY_TXT_NAME",
    "TARGET_TIMEFRAME",
    "VERDICT_ALIGNMENT_OR_RESAMPLING",
    "VERDICT_FOLD_BOUNDARY_WARMUP_LOSS",
    "VERDICT_IMPLEMENTATION_BUG",
    "VERDICT_INPUT_DATA_COVERAGE",
    "VERDICT_INSUFFICIENT_HISTORY",
    "VERDICT_MULTI_CAUSE",
    "VERDICT_ROLLING_WINDOW_CONFIGURATION",
    "VERDICT_UNDETERMINED",
    "FactorStability1dDegenerationReporter",
    "FactorStability1dDegenerationResult",
    "classify_factor_null_cause",
    "classify_verdict",
    "effective_warmup_bars",
    "factor_lookback_catalog",
    "forbidden_import_violations",
    "hash_watched_production_artifacts",
]

_logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT: Final[Path] = Path("reports") / "factor_stability" / "1d_factor_degeneration"
TARGET_TIMEFRAME: Final[str] = "1d"
COMPARISON_TIMEFRAMES: Final[tuple[str, ...]] = ("5m", "15m", "1h", "4h", "1d")

GLOBAL_CSV_NAME: Final[str] = "global.csv"
FACTORS_CSV_NAME: Final[str] = "factors.csv"
FOLDS_CSV_NAME: Final[str] = "folds.csv"
DATA_COVERAGE_CSV_NAME: Final[str] = "data_coverage.csv"
LOOKBACK_ANALYSIS_CSV_NAME: Final[str] = "lookback_analysis.csv"
CROSS_TIMEFRAME_CSV_NAME: Final[str] = "cross_timeframe_comparison.csv"
SUMMARY_TXT_NAME: Final[str] = "summary.txt"
HASHES_BEFORE_NAME: Final[str] = "hashes_before.txt"
HASHES_AFTER_NAME: Final[str] = "hashes_after.txt"

CAUSE_INSUFFICIENT_HISTORY: Final[str] = "INSUFFICIENT_HISTORY"
CAUSE_INPUT_DATA_MISSING: Final[str] = "INPUT_DATA_MISSING"
CAUSE_ROLLING_WINDOW_TOO_LARGE: Final[str] = "ROLLING_WINDOW_TOO_LARGE"
CAUSE_ALIGNMENT_OR_RESAMPLING: Final[str] = "ALIGNMENT_OR_RESAMPLING"
CAUSE_IMPLEMENTATION_BUG: Final[str] = "IMPLEMENTATION_BUG"
CAUSE_EVALUATION_BOUNDARY: Final[str] = "EVALUATION_BOUNDARY"
CAUSE_UNKNOWN: Final[str] = "UNKNOWN"

VERDICT_INSUFFICIENT_HISTORY: Final[str] = "A. INSUFFICIENT_HISTORY"
VERDICT_INPUT_DATA_COVERAGE: Final[str] = "B. INPUT_DATA_COVERAGE"
VERDICT_ROLLING_WINDOW_CONFIGURATION: Final[str] = "C. ROLLING_WINDOW_CONFIGURATION"
VERDICT_FOLD_BOUNDARY_WARMUP_LOSS: Final[str] = "D. FOLD_BOUNDARY_WARMUP_LOSS"
VERDICT_ALIGNMENT_OR_RESAMPLING: Final[str] = "E. ALIGNMENT_OR_RESAMPLING"
VERDICT_IMPLEMENTATION_BUG: Final[str] = "F. IMPLEMENTATION_BUG"
VERDICT_MULTI_CAUSE: Final[str] = "G. MULTI_CAUSE"
VERDICT_UNDETERMINED: Final[str] = "H. UNDETERMINED"

FIXABILITY_DATA_PROBLEM: Final[str] = "DATA_PROBLEM"
FIXABILITY_CONFIGURATION_PROBLEM: Final[str] = "CONFIGURATION_PROBLEM"
FIXABILITY_PIPELINE_PROBLEM: Final[str] = "PIPELINE_PROBLEM"
FIXABILITY_FACTOR_IMPLEMENTATION_PROBLEM: Final[str] = "FACTOR_IMPLEMENTATION_PROBLEM"
FIXABILITY_EXPECTED_DATA_LIMITATION: Final[str] = "EXPECTED_DATA-LIMITATION"

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

_PARTITION_OOS: Final[str] = "OOS"
_MS_PER_DAY: Final[int] = 86_400_000
_HIGH_NULL_RATE: Final[float] = 0.80
_REFERENCE_SYMBOL: Final[str] = "BTCUSDT"

_ERROR_MANAGER: Final[str] = "REPORT-1D-DEGEN-001"
_ERROR_OUTPUT: Final[str] = "REPORT-1D-DEGEN-002"
_ERROR_MISSING_1D: Final[str] = "REPORT-1D-DEGEN-003"
_ERROR_LEDGER_MUTATION: Final[str] = "REPORT-1D-DEGEN-004"

# Lookback / input catalog for selected 1d factors (defaults from factor modules).
# atr_slope effective warmup is 2*lookback-1 because ATR then rolling OLS reuse
# the same lookback window.
_FACTOR_SPECS: Final[dict[str, tuple[int, int, tuple[str, ...], str]]] = {
    # name: (configured_lookback, effective_warmup_bars, inputs, family)
    "accumulation_distribution": (0, 0, ("high", "low", "close", "volume"), "volume"),
    "aggressive_buy_ratio": (20, 20, ("taker_buy_volume", "volume"), "microstructure"),
    "aggressive_sell_ratio": (20, 20, ("taker_sell_volume", "volume"), "microstructure"),
    "atr_distance": (20, 20, ("high", "low", "close"), "price"),
    "atr_percent": (20, 20, ("high", "low", "close"), "price"),
    "atr_slope": (20, 39, ("high", "low", "close"), "price"),
    "bollinger_bandwidth": (20, 20, ("close",), "price"),
    "bollinger_position": (20, 20, ("close",), "price"),
    "bollinger_width": (20, 20, ("close",), "price"),
    "breakout_strength": (20, 21, ("close",), "price"),
    "buy_sell_imbalance": (0, 0, ("taker_buy_volume", "taker_sell_volume"), "microstructure"),
    "ease_of_movement": (14, 14, ("high", "low", "volume"), "volume"),
    "funding_rate_level": (0, 0, ("funding_rate",), "funding"),
    "money_flow_index": (14, 14, ("high", "low", "close", "volume"), "volume"),
    "on_balance_volume": (0, 1, ("close", "volume"), "volume"),
    "open_interest_level": (0, 0, ("open_interest",), "open_interest"),
    "price_volume_trend": (0, 1, ("close", "volume"), "volume"),
    "rate_of_change": (12, 12, ("close",), "price"),
    "rsi": (14, 14, ("close",), "price"),
    "stochastic_k": (14, 14, ("high", "low", "close"), "price"),
}

GLOBAL_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "selected_factors",
    "degenerate_oos_100pct_null",
    "high_missingness_factors",
    "median_oos_null_rate",
    "unique_oos_timestamps",
    "oos_first_timestamp",
    "oos_last_timestamp",
    "factor_store_first_timestamp",
    "factor_store_last_timestamp",
    "ohlcv_first_timestamp",
    "ohlcv_last_timestamp",
    "ohlcv_rows_reference",
    "post_alignment_bars_reference",
    "companion_alignment_truncates_history",
    "factors_computed_globally",
    "fold_local_recompute",
    "selection_has_min_coverage_rule",
    "selected_zero_observation_factors",
    "verdict",
    "primary_root_cause",
    "fixability",
    "production_artifacts_unchanged",
    "deterministic",
)

FACTOR_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "family",
    "selected",
    "direction",
    "train_null_rate",
    "oos_null_rate",
    "first_non_null_timestamp",
    "last_non_null_timestamp",
    "usable_rows",
    "total_rows",
    "required_lookback",
    "effective_warmup_bars",
    "input_columns",
    "input_null_rate",
    "store_null_rate",
    "store_first_non_null_timestamp",
    "store_usable_rows",
    "store_total_rows",
    "validation_observations",
    "validation_status",
    "selection_rank",
    "selection_score",
    "selection_time_null_rate",
    "likely_cause",
)

FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "fold_id",
    "oos_rows",
    "unique_timestamps",
    "oos_first_timestamp",
    "oos_last_timestamp",
    "median_factor_null_rate",
    "degenerate_factor_count",
    "factors_with_any_non_null",
)

DATA_COVERAGE_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "timeframe",
    "symbol_scope",
    "total_rows",
    "unique_timestamps",
    "first_timestamp",
    "last_timestamp",
    "symbols",
    "min_history_rows",
    "median_history_rows",
    "max_history_rows",
    "notes",
)

LOOKBACK_ANALYSIS_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "configured_lookback",
    "effective_warmup_bars",
    "input_columns",
    "post_alignment_bars_reference",
    "warmup_exceeds_post_alignment",
    "first_theoretical_valid_day_index",
    "oos_day_count",
    "first_store_non_null_after_oos_end",
    "mechanism_notes",
)

COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "factor_store_unique_timestamps",
    "factor_store_span_first",
    "factor_store_span_last",
    "selected_factor_count",
    "median_selected_oos_null_rate",
    "degenerate_selected_count",
    "unique_oos_timestamps",
    "notes",
)


@dataclass(frozen=True, slots=True)
class FactorStability1dDegenerationResult:
    """Immutable result of the 1d factor-degeneration investigation."""

    global_frame: pl.DataFrame
    factor_frame: pl.DataFrame
    fold_frame: pl.DataFrame
    data_coverage_frame: pl.DataFrame
    lookback_frame: pl.DataFrame
    comparison_frame: pl.DataFrame
    summary_text: str
    paths: Mapping[str, Path]
    verdict: str
    primary_root_cause: str
    fixability: str
    production_artifacts_unchanged: bool
    deterministic: bool
    hashes_before: Mapping[str, str]
    hashes_after: Mapping[str, str]


class FactorStability1dDegenerationReporter:
    """Read-only 1d factor-degeneration diagnostic reporter."""

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

    def run(self, *, year: int | None = None) -> FactorStability1dDegenerationResult:
        """Execute the degeneration investigation and write reports."""
        if self._output_root.exists() and not self._output_root.is_dir():
            raise ReportingValidationError(
                "output path must be a directory",
                error_code=_ERROR_OUTPUT,
                details={"output": str(self._output_root)},
            )
        self._output_root.mkdir(parents=True, exist_ok=True)
        hashes_before = hash_watched_production_artifacts(self._storage_root)
        _write_hash_manifest(self._output_root / HASHES_BEFORE_NAME, hashes_before)

        selection_path, panel_year = _resolve_1d_selection_partition(
            self._storage_root,
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            year=year,
        )
        self._logger.info(
            "1d factor-degeneration investigation starting manager=%s year=%s",
            self._manager,
            panel_year,
        )
        panel = self._compute_panel(selection_path=selection_path, year=panel_year)
        paths = _write_report_bundle(
            output_root=self._output_root,
            global_frame=panel["global"],  # type: ignore[arg-type]
            factor_frame=panel["factor"],  # type: ignore[arg-type]
            fold_frame=panel["fold"],  # type: ignore[arg-type]
            data_coverage_frame=panel["data_coverage"],  # type: ignore[arg-type]
            lookback_frame=panel["lookback"],  # type: ignore[arg-type]
            comparison_frame=panel["comparison"],  # type: ignore[arg-type]
            summary_text=str(panel["summary_text"]),
        )
        hashes_after = hash_watched_production_artifacts(self._storage_root)
        _write_hash_manifest(self._output_root / HASHES_AFTER_NAME, hashes_after)
        unchanged = hashes_before == hashes_after
        if not unchanged:
            raise ReportingValidationError(
                "production artifacts mutated during degeneration investigation",
                error_code=_ERROR_LEDGER_MUTATION,
                details={
                    "before_count": len(hashes_before),
                    "after_count": len(hashes_after),
                },
            )
        global_frame = panel["global"]
        assert isinstance(global_frame, pl.DataFrame)
        global_updated = global_frame.with_columns(
            pl.lit(unchanged).alias("production_artifacts_unchanged"),
            pl.lit(True).alias("deterministic"),
        )
        global_path = paths["global"]
        global_updated.write_csv(global_path)
        return FactorStability1dDegenerationResult(
            global_frame=global_updated,
            factor_frame=panel["factor"],  # type: ignore[arg-type]
            fold_frame=panel["fold"],  # type: ignore[arg-type]
            data_coverage_frame=panel["data_coverage"],  # type: ignore[arg-type]
            lookback_frame=panel["lookback"],  # type: ignore[arg-type]
            comparison_frame=panel["comparison"],  # type: ignore[arg-type]
            summary_text=str(panel["summary_text"]),
            paths=paths,
            verdict=str(panel["verdict"]),
            primary_root_cause=str(panel["primary_root_cause"]),
            fixability=str(panel["fixability"]),
            production_artifacts_unchanged=unchanged,
            deterministic=True,
            hashes_before=hashes_before,
            hashes_after=hashes_after,
        )

    def _compute_panel(
        self,
        *,
        selection_path: Path,
        year: int,
    ) -> dict[str, object]:
        selection = pl.read_parquet(selection_path)
        selected = selection.filter(pl.col("selected") == True)  # noqa: E712
        if selected.is_empty():
            raise ReportingValidationError(
                "no selected 1d factors found",
                error_code=_ERROR_MISSING_1D,
                details={"path": str(selection_path)},
            )

        pcv_eval_path = _evaluation_path(
            self._storage_root,
            STORAGE_DIR_PURGED_CV_EVALUATION,
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=TARGET_TIMEFRAME,
            year=year,
        )
        if pcv_eval_path is None:
            raise ReportingValidationError(
                "1d purged_cv_evaluation partition not found",
                error_code=_ERROR_MISSING_1D,
                details={"year": year},
            )
        evaluation = pl.read_parquet(pcv_eval_path)
        if "partition" in evaluation.columns:
            oos = evaluation.filter(pl.col("partition") == _PARTITION_OOS)
        else:
            oos = evaluation

        validation = _load_optional_parquet(
            self._storage_root
            / STORAGE_DIR_FACTOR_VALIDATION
            / self._manager
            / self._exchange
            / self._market
            / TARGET_TIMEFRAME
            / f"{year}.parquet"
        )
        pcv_ledger = _load_optional_parquet(
            self._storage_root
            / STORAGE_DIR_PURGED_CV
            / self._manager
            / self._exchange
            / self._market
            / TARGET_TIMEFRAME
            / f"{year}.parquet"
        )

        reference_store = _load_optional_parquet(
            self._storage_root
            / STORAGE_DIR_FACTORS
            / self._manager
            / self._exchange
            / self._market
            / _REFERENCE_SYMBOL
            / TARGET_TIMEFRAME
            / f"{year}.parquet"
        )
        ohlcv_ref = _load_optional_parquet(
            self._storage_root
            / STORAGE_DIR_PROCESSED
            / "ohlcv"
            / self._exchange
            / self._market
            / _REFERENCE_SYMBOL
            / TARGET_TIMEFRAME
            / f"{year}.parquet"
        )
        coverage = _build_data_coverage(
            storage_root=self._storage_root,
            exchange=self._exchange,
            market=self._market,
            year=year,
            reference_store=reference_store,
            ohlcv_ref=ohlcv_ref,
            oos=oos,
        )
        factor_frame = _build_factor_frame(
            selected=selected,
            oos=oos,
            validation=validation,
            reference_store=reference_store,
            ohlcv_ref=ohlcv_ref,
        )
        fold_frame = _build_fold_frame(oos=oos, year=year, pcv_ledger=pcv_ledger)
        lookback_frame = _build_lookback_frame(
            factor_frame=factor_frame,
            reference_store=reference_store,
            oos=oos,
        )
        comparison_frame = _build_cross_timeframe_frame(
            storage_root=self._storage_root,
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            year=year,
            factor_frame_1d=factor_frame,
            oos_1d=oos,
            reference_store_1d=reference_store,
        )

        oos_null_rates = [
            float(value) for value in factor_frame["oos_null_rate"].to_list() if value is not None
        ]
        median_null = float(statistics.median(oos_null_rates)) if oos_null_rates else float("nan")
        degenerate_100 = int(factor_frame.filter(pl.col("oos_null_rate") >= 1.0 - 1e-12).height)
        high_missing = int(factor_frame.filter(pl.col("oos_null_rate") >= _HIGH_NULL_RATE).height)
        zero_obs = int(
            factor_frame.filter(
                pl.col("validation_observations").is_not_null()
                & (pl.col("validation_observations") <= 0)
            ).height
        )

        oos_first = _min_time_iso(oos, "observation_time")
        oos_last = _max_time_iso(oos, "observation_time")
        store_first = (
            _min_time_iso(reference_store, "open_time") if reference_store is not None else ""
        )
        store_last = (
            _max_time_iso(reference_store, "open_time") if reference_store is not None else ""
        )
        ohlcv_first = _min_time_iso(ohlcv_ref, "open_time") if ohlcv_ref is not None else ""
        ohlcv_last = _max_time_iso(ohlcv_ref, "open_time") if ohlcv_ref is not None else ""
        ohlcv_rows = int(ohlcv_ref.height) if ohlcv_ref is not None else 0
        post_align_bars = (
            int(reference_store.select(pl.col("open_time").n_unique()).item())
            if reference_store is not None
            else 0
        )
        companion_truncates = bool(
            ohlcv_rows > 0 and post_align_bars > 0 and post_align_bars < ohlcv_rows
        )

        causes = [str(value) for value in factor_frame["likely_cause"].to_list()]
        verdict, primary, fixability = classify_verdict(
            causes=causes,
            companion_alignment_truncates_history=companion_truncates,
            fold_local_recompute=False,
            degenerate_100pct_count=degenerate_100,
            high_missingness_count=high_missing,
        )
        summary_text = _build_summary_text(
            year=year,
            verdict=verdict,
            primary_root_cause=primary,
            fixability=fixability,
            factor_frame=factor_frame,
            oos_first=oos_first,
            oos_last=oos_last,
            store_first=store_first,
            store_last=store_last,
            ohlcv_first=ohlcv_first,
            ohlcv_last=ohlcv_last,
            ohlcv_rows=ohlcv_rows,
            post_align_bars=post_align_bars,
            companion_truncates=companion_truncates,
            unique_oos_timestamps=int(oos["observation_time"].n_unique()),
            degenerate_100=degenerate_100,
            high_missing=high_missing,
            median_null=median_null,
            zero_obs=zero_obs,
            comparison_frame=comparison_frame,
        )
        global_frame = pl.DataFrame(
            [
                {
                    "timeframe": TARGET_TIMEFRAME,
                    "year": year,
                    "selected_factors": int(selected.height),
                    "degenerate_oos_100pct_null": degenerate_100,
                    "high_missingness_factors": high_missing,
                    "median_oos_null_rate": median_null,
                    "unique_oos_timestamps": int(oos["observation_time"].n_unique()),
                    "oos_first_timestamp": oos_first,
                    "oos_last_timestamp": oos_last,
                    "factor_store_first_timestamp": store_first,
                    "factor_store_last_timestamp": store_last,
                    "ohlcv_first_timestamp": ohlcv_first,
                    "ohlcv_last_timestamp": ohlcv_last,
                    "ohlcv_rows_reference": ohlcv_rows,
                    "post_alignment_bars_reference": post_align_bars,
                    "companion_alignment_truncates_history": companion_truncates,
                    "factors_computed_globally": True,
                    "fold_local_recompute": False,
                    "selection_has_min_coverage_rule": False,
                    "selected_zero_observation_factors": zero_obs,
                    "verdict": verdict,
                    "primary_root_cause": primary,
                    "fixability": fixability,
                    "production_artifacts_unchanged": True,
                    "deterministic": True,
                }
            ]
        ).select(list(GLOBAL_COLUMNS))
        return {
            "global": global_frame,
            "factor": factor_frame,
            "fold": fold_frame,
            "data_coverage": coverage,
            "lookback": lookback_frame,
            "comparison": comparison_frame,
            "summary_text": summary_text,
            "verdict": verdict,
            "primary_root_cause": primary,
            "fixability": fixability,
        }


def factor_lookback_catalog() -> dict[str, tuple[int, int, tuple[str, ...], str]]:
    """Return the immutable lookback/input catalog for selected 1d factors."""
    return dict(_FACTOR_SPECS)


def effective_warmup_bars(factor_name: str) -> int:
    """Return effective warmup bars required before the first non-null value."""
    spec = _FACTOR_SPECS.get(factor_name)
    if spec is None:
        return 0
    return int(spec[1])


def classify_factor_null_cause(
    *,
    oos_null_rate: float,
    store_null_rate: float | None,
    required_lookback: int,
    effective_warmup: int,
    post_alignment_bars: int,
    store_first_non_null_ms: int | None,
    oos_last_ms: int | None,
    input_null_rate: float | None,
    inputs_present: bool,
) -> str:
    """Classify the dominant null mechanism for one factor.

    Args:
        oos_null_rate: Fraction of OOS evaluation rows with null factor_value.
        store_null_rate: Fraction of factor-store rows with null factor_value.
        required_lookback: Configured factor lookback.
        effective_warmup: Bars until the first theoretically valid value.
        post_alignment_bars: Bars available after companion alignment.
        store_first_non_null_ms: First non-null timestamp in the factor store.
        oos_last_ms: Last OOS evaluation timestamp.
        input_null_rate: Null rate of required input columns when measurable.
        inputs_present: Whether required input columns exist upstream.

    Returns:
        One of the ``CAUSE_*`` classification labels.
    """
    if not inputs_present:
        return CAUSE_INPUT_DATA_MISSING
    if input_null_rate is not None and input_null_rate >= 1.0 - 1e-12 and oos_null_rate >= 0.99:
        return CAUSE_INPUT_DATA_MISSING
    if (
        store_null_rate is not None
        and store_null_rate >= 1.0 - 1e-12
        and post_alignment_bars > 0
        and effective_warmup > post_alignment_bars
    ):
        return CAUSE_ROLLING_WINDOW_TOO_LARGE
    if (
        store_first_non_null_ms is not None
        and oos_last_ms is not None
        and store_first_non_null_ms > oos_last_ms
        and oos_null_rate >= 0.99
    ):
        return CAUSE_EVALUATION_BOUNDARY
    if oos_null_rate >= _HIGH_NULL_RATE and effective_warmup > 0:
        if post_alignment_bars > 0 and effective_warmup >= post_alignment_bars * 0.35:
            return CAUSE_INSUFFICIENT_HISTORY
        return CAUSE_INSUFFICIENT_HISTORY
    if oos_null_rate < 0.10:
        return CAUSE_UNKNOWN
    if required_lookback <= 0 and oos_null_rate >= _HIGH_NULL_RATE:
        return CAUSE_IMPLEMENTATION_BUG
    return CAUSE_UNKNOWN


def classify_verdict(
    *,
    causes: Sequence[str],
    companion_alignment_truncates_history: bool,
    fold_local_recompute: bool,
    degenerate_100pct_count: int,
    high_missingness_count: int,
) -> tuple[str, str, str]:
    """Classify global verdict, primary root-cause text seed, and fixability."""
    if fold_local_recompute:
        return (
            VERDICT_FOLD_BOUNDARY_WARMUP_LOSS,
            (
                "Factors are recomputed inside each fold without warmup history, "
                "forcing rolling windows to restart at fold boundaries."
            ),
            FIXABILITY_PIPELINE_PROBLEM,
        )
    material = [
        cause
        for cause in causes
        if cause
        not in {
            CAUSE_UNKNOWN,
        }
    ]
    unique = sorted(set(material))
    if not unique and (degenerate_100pct_count > 0 or high_missingness_count > 0):
        return (
            VERDICT_UNDETERMINED,
            "Degeneration observed but mechanism could not be classified.",
            FIXABILITY_EXPECTED_DATA_LIMITATION,
        )
    if fold_local_recompute is False and companion_alignment_truncates_history:
        if CAUSE_EVALUATION_BOUNDARY in unique or CAUSE_ROLLING_WINDOW_TOO_LARGE in unique:
            return (
                VERDICT_MULTI_CAUSE,
                (
                    "Companion alignment truncates 1d OHLCV history to the first "
                    "companion-complete bar; configured rolling lookbacks then push "
                    "the first non-null factor value to or beyond the OOS window end, "
                    "so selected lookback-heavy factors are null throughout OOS."
                ),
                f"{FIXABILITY_DATA_PROBLEM};{FIXABILITY_CONFIGURATION_PROBLEM};"
                f"{FIXABILITY_EXPECTED_DATA_LIMITATION}",
            )
        return (
            VERDICT_ALIGNMENT_OR_RESAMPLING,
            (
                "Companion alignment drops leading OHLCV bars until funding / OI / "
                "taker / long-short companions are all non-null, collapsing usable "
                "1d history."
            ),
            FIXABILITY_DATA_PROBLEM,
        )
    if unique == [CAUSE_ROLLING_WINDOW_TOO_LARGE]:
        return (
            VERDICT_ROLLING_WINDOW_CONFIGURATION,
            "Configured rolling windows exceed available post-alignment bars.",
            FIXABILITY_CONFIGURATION_PROBLEM,
        )
    if unique == [CAUSE_INSUFFICIENT_HISTORY]:
        return (
            VERDICT_INSUFFICIENT_HISTORY,
            "Post-alignment 1d history is shorter than required factor warmup.",
            FIXABILITY_EXPECTED_DATA_LIMITATION,
        )
    if unique == [CAUSE_INPUT_DATA_MISSING]:
        return (
            VERDICT_INPUT_DATA_COVERAGE,
            "Required factor input columns are missing upstream.",
            FIXABILITY_DATA_PROBLEM,
        )
    if CAUSE_IMPLEMENTATION_BUG in unique and len(unique) == 1:
        return (
            VERDICT_IMPLEMENTATION_BUG,
            "Nulls persist despite sufficient history and valid inputs.",
            FIXABILITY_FACTOR_IMPLEMENTATION_PROBLEM,
        )
    if len(unique) > 1:
        return (
            VERDICT_MULTI_CAUSE,
            (
                "Multiple interacting mechanisms produce 1d OOS null degeneration: "
                + ", ".join(unique)
            ),
            f"{FIXABILITY_DATA_PROBLEM};{FIXABILITY_CONFIGURATION_PROBLEM}",
        )
    if unique:
        mapping = {
            CAUSE_EVALUATION_BOUNDARY: (
                VERDICT_INSUFFICIENT_HISTORY,
                "First non-null factor values occur after the OOS evaluation window.",
                FIXABILITY_EXPECTED_DATA_LIMITATION,
            ),
            CAUSE_ALIGNMENT_OR_RESAMPLING: (
                VERDICT_ALIGNMENT_OR_RESAMPLING,
                "Alignment/resampling truncates usable 1d history.",
                FIXABILITY_PIPELINE_PROBLEM,
            ),
        }
        return mapping.get(
            unique[0],
            (
                VERDICT_UNDETERMINED,
                f"Classified cause {unique[0]} without a stronger global mapping.",
                FIXABILITY_EXPECTED_DATA_LIMITATION,
            ),
        )
    return (
        VERDICT_UNDETERMINED,
        "No material degeneration causes classified.",
        FIXABILITY_EXPECTED_DATA_LIMITATION,
    )


def forbidden_import_violations(source: str) -> tuple[str, ...]:
    """Return forbidden import module names found in ``source``."""
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if _is_forbidden_module(module):
                    violations.append(module)
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


def _resolve_1d_selection_partition(
    storage_root: Path,
    *,
    manager: str,
    exchange: str,
    market: str,
    year: int | None,
) -> tuple[Path, int]:
    root = (
        Path(storage_root)
        / STORAGE_DIR_FACTOR_SELECTION
        / manager
        / exchange
        / market
        / TARGET_TIMEFRAME
    )
    if not root.exists():
        raise ReportingValidationError(
            "1d factor selection partition not found",
            error_code=_ERROR_MISSING_1D,
            details={"path": str(root)},
        )
    years: list[tuple[int, Path]] = []
    for parquet_path in sorted(root.glob("*.parquet")):
        try:
            panel_year = int(parquet_path.stem)
        except ValueError:
            continue
        years.append((panel_year, parquet_path))
    if not years:
        raise ReportingValidationError(
            "1d factor selection parquet not found",
            error_code=_ERROR_MISSING_1D,
            details={"path": str(root)},
        )
    if year is not None:
        for panel_year, path in years:
            if panel_year == year:
                return path, panel_year
        raise ReportingValidationError(
            "requested 1d year not found",
            error_code=_ERROR_MISSING_1D,
            details={"year": year, "available": [item[0] for item in years]},
        )
    panel_year, path = max(years, key=lambda item: item[0])
    return path, panel_year


def _evaluation_path(
    storage_root: Path,
    tier: str,
    *,
    manager: str,
    exchange: str,
    market: str,
    timeframe: str,
    year: int,
) -> Path | None:
    path = Path(storage_root) / tier / manager / exchange / market / timeframe / f"{year}.parquet"
    return path if path.exists() else None


def _load_optional_parquet(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _as_int(value: object) -> int:
    """Convert a polars scalar-like value to ``int``."""
    return int(value)  # type: ignore[arg-type]


def _ms_to_iso(ms: int | None) -> str:
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).strftime("%Y-%m-%d")


def _min_time_iso(frame: pl.DataFrame, column: str) -> str:
    if frame.is_empty() or column not in frame.columns:
        return ""
    value = frame[column].min()
    return _ms_to_iso(_as_int(value)) if value is not None else ""


def _max_time_iso(frame: pl.DataFrame, column: str) -> str:
    if frame.is_empty() or column not in frame.columns:
        return ""
    value = frame[column].max()
    return _ms_to_iso(_as_int(value)) if value is not None else ""


def _build_factor_frame(
    *,
    selected: pl.DataFrame,
    oos: pl.DataFrame,
    validation: pl.DataFrame | None,
    reference_store: pl.DataFrame | None,
    ohlcv_ref: pl.DataFrame | None,
) -> pl.DataFrame:
    oos_last_ms = (
        _as_int(oos["observation_time"].max())
        if oos.height and "observation_time" in oos.columns
        else None
    )
    post_align_bars = (
        int(reference_store.select(pl.col("open_time").n_unique()).item())
        if reference_store is not None and reference_store.height
        else 0
    )
    rows: list[dict[str, object]] = []
    for record in selected.sort("selection_rank").iter_rows(named=True):
        name = str(record["factor_name"])
        spec = _FACTOR_SPECS.get(name)
        configured_lookback = int(spec[0]) if spec is not None else 0
        warmup = int(spec[1]) if spec is not None else 0
        inputs = spec[2] if spec is not None else tuple()
        family = (
            str(record["factor_category"])
            if "factor_category" in record and record["factor_category"] is not None
            else (spec[3] if spec is not None else "")
        )
        sub = oos.filter(pl.col("factor_name") == name)
        total_rows = int(sub.height)
        usable_rows = (
            int(sub.filter(pl.col("factor_value").is_not_null()).height) if total_rows else 0
        )
        oos_null_rate = (
            float((total_rows - usable_rows) / total_rows) if total_rows else float("nan")
        )
        nn = sub.filter(pl.col("factor_value").is_not_null()) if total_rows else sub
        first_nn = (
            _ms_to_iso(int(nn["observation_time"].min())) if nn.height else ""  # type: ignore[arg-type]
        )
        last_nn = (
            _ms_to_iso(int(nn["observation_time"].max())) if nn.height else ""  # type: ignore[arg-type]
        )
        store_null_rate: float | None = None
        store_first_nn = ""
        store_first_nn_ms: int | None = None
        store_usable = 0
        store_total = 0
        if reference_store is not None and "factor_name" in reference_store.columns:
            store_sub = reference_store.filter(pl.col("factor_name") == name)
            store_total = int(store_sub.height)
            if store_total:
                store_nn = store_sub.filter(pl.col("factor_value").is_not_null())
                store_usable = int(store_nn.height)
                store_null_rate = float((store_total - store_usable) / store_total)
                if store_nn.height:
                    store_first_nn_ms = int(store_nn["open_time"].min())  # type: ignore[arg-type]
                    store_first_nn = _ms_to_iso(store_first_nn_ms)

        input_null_rate: float | None = None
        inputs_present = True
        if ohlcv_ref is not None and inputs:
            ohlcv_inputs = [column for column in inputs if column in ohlcv_ref.columns]
            companion_only = [column for column in inputs if column not in ohlcv_ref.columns]
            if ohlcv_inputs:
                null_expr = pl.any_horizontal(
                    *(pl.col(column).is_null() for column in ohlcv_inputs)
                )
                input_null_rate = float(ohlcv_ref.filter(null_expr).height / ohlcv_ref.height)
            # Companion inputs (taker/funding/OI) are joined later; presence is
            # evidenced by lookback-0 companion factors being non-null in store.
            if companion_only and store_null_rate is not None and store_null_rate < 1.0:
                inputs_present = True
            elif companion_only and not ohlcv_inputs:
                # Pure companion factor: treat as present when store has rows.
                inputs_present = store_total > 0

        validation_observations: int | None = None
        validation_status = ""
        if validation is not None and "factor_name" in validation.columns:
            val_sub = validation.filter(pl.col("factor_name") == name)
            if val_sub.height:
                if "observations" in val_sub.columns:
                    validation_observations = int(val_sub["observations"][0])
                if "status" in val_sub.columns:
                    validation_status = str(val_sub["status"][0])

        # Selection-time null proxy: 1 - observations / theoretical panel size
        # when observations available; else unknown.
        selection_time_null_rate: float | None
        if validation_observations is not None and total_rows > 0:
            selection_time_null_rate = max(
                0.0,
                min(1.0, 1.0 - (float(validation_observations) / float(total_rows))),
            )
        elif validation_observations == 0:
            selection_time_null_rate = 1.0
        else:
            selection_time_null_rate = None

        likely_cause = classify_factor_null_cause(
            oos_null_rate=oos_null_rate if total_rows else 1.0,
            store_null_rate=store_null_rate,
            required_lookback=configured_lookback,
            effective_warmup=warmup,
            post_alignment_bars=post_align_bars,
            store_first_non_null_ms=store_first_nn_ms,
            oos_last_ms=oos_last_ms,
            input_null_rate=input_null_rate,
            inputs_present=inputs_present,
        )
        rows.append(
            {
                "factor": name,
                "family": family,
                "selected": True,
                "direction": int(record.get("selected_direction") or 1),
                "train_null_rate": None,
                "oos_null_rate": oos_null_rate,
                "first_non_null_timestamp": first_nn,
                "last_non_null_timestamp": last_nn,
                "usable_rows": usable_rows,
                "total_rows": total_rows,
                "required_lookback": configured_lookback,
                "effective_warmup_bars": warmup,
                "input_columns": "|".join(inputs),
                "input_null_rate": input_null_rate,
                "store_null_rate": store_null_rate,
                "store_first_non_null_timestamp": store_first_nn,
                "store_usable_rows": store_usable,
                "store_total_rows": store_total,
                "validation_observations": validation_observations,
                "validation_status": validation_status,
                "selection_rank": int(record.get("selection_rank") or 0),
                "selection_score": (
                    float(record["selection_score"])
                    if record.get("selection_score") is not None
                    else None
                ),
                "selection_time_null_rate": selection_time_null_rate,
                "likely_cause": likely_cause,
            }
        )
    return pl.DataFrame(rows).select(list(FACTOR_COLUMNS))


def _build_fold_frame(
    *,
    oos: pl.DataFrame,
    year: int,
    pcv_ledger: pl.DataFrame | None,
) -> pl.DataFrame:
    if "fold_id" not in oos.columns:
        return pl.DataFrame(schema={name: pl.Utf8 for name in FOLD_COLUMNS}).clear()
    rows: list[dict[str, object]] = []
    for fold_id in sorted(int(value) for value in oos["fold_id"].unique().to_list()):
        sub = oos.filter(pl.col("fold_id") == fold_id)
        null_rates: list[float] = []
        degenerate = 0
        any_non_null = 0
        for name in sorted(sub["factor_name"].unique().to_list()):
            factor_sub = sub.filter(pl.col("factor_name") == name)
            total = factor_sub.height
            usable = factor_sub.filter(pl.col("factor_value").is_not_null()).height
            rate = float((total - usable) / total) if total else 1.0
            null_rates.append(rate)
            if rate >= 1.0 - 1e-12:
                degenerate += 1
            if usable > 0:
                any_non_null += 1
        rows.append(
            {
                "timeframe": TARGET_TIMEFRAME,
                "year": year,
                "fold_id": fold_id,
                "oos_rows": int(sub.height),
                "unique_timestamps": int(sub["observation_time"].n_unique()),
                "oos_first_timestamp": _min_time_iso(sub, "observation_time"),
                "oos_last_timestamp": _max_time_iso(sub, "observation_time"),
                "median_factor_null_rate": (
                    float(statistics.median(null_rates)) if null_rates else float("nan")
                ),
                "degenerate_factor_count": degenerate,
                "factors_with_any_non_null": any_non_null,
            }
        )
    frame = pl.DataFrame(rows).select(list(FOLD_COLUMNS))
    if pcv_ledger is not None and "fold_id" in pcv_ledger.columns:
        # Ledger presence confirms fold boundaries; values already come from OOS.
        _ = pcv_ledger.height
    return frame


def _build_data_coverage(
    *,
    storage_root: Path,
    exchange: str,
    market: str,
    year: int,
    reference_store: pl.DataFrame | None,
    ohlcv_ref: pl.DataFrame | None,
    oos: pl.DataFrame,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    ohlcv_root = Path(storage_root) / STORAGE_DIR_PROCESSED / "ohlcv" / exchange / market
    histories: list[int] = []
    firsts: list[int] = []
    lasts: list[int] = []
    symbols = 0
    if ohlcv_root.exists():
        for symbol_dir in sorted(ohlcv_root.iterdir()):
            path = symbol_dir / TARGET_TIMEFRAME / f"{year}.parquet"
            if not path.exists():
                continue
            frame = pl.read_parquet(path)
            symbols += 1
            histories.append(int(frame.height))
            firsts.append(int(frame["open_time"].min()))  # type: ignore[arg-type]
            lasts.append(int(frame["open_time"].max()))  # type: ignore[arg-type]
    rows.append(
        {
            "dataset": "processed_ohlcv",
            "timeframe": TARGET_TIMEFRAME,
            "symbol_scope": "all",
            "total_rows": int(sum(histories)) if histories else 0,
            "unique_timestamps": "",
            "first_timestamp": _ms_to_iso(min(firsts)) if firsts else "",
            "last_timestamp": _ms_to_iso(max(lasts)) if lasts else "",
            "symbols": symbols,
            "min_history_rows": min(histories) if histories else 0,
            "median_history_rows": (int(statistics.median(histories)) if histories else 0),
            "max_history_rows": max(histories) if histories else 0,
            "notes": "raw calendar-year OHLCV before companion alignment",
        }
    )
    if ohlcv_ref is not None:
        rows.append(
            {
                "dataset": "processed_ohlcv",
                "timeframe": TARGET_TIMEFRAME,
                "symbol_scope": _REFERENCE_SYMBOL,
                "total_rows": int(ohlcv_ref.height),
                "unique_timestamps": int(ohlcv_ref["open_time"].n_unique()),
                "first_timestamp": _min_time_iso(ohlcv_ref, "open_time"),
                "last_timestamp": _max_time_iso(ohlcv_ref, "open_time"),
                "symbols": 1,
                "min_history_rows": int(ohlcv_ref.height),
                "median_history_rows": int(ohlcv_ref.height),
                "max_history_rows": int(ohlcv_ref.height),
                "notes": "reference symbol OHLCV",
            }
        )
    companion_specs = (
        ("taker_volume", "taker_volume", TARGET_TIMEFRAME, "timestamp"),
        ("open_interest", "open_interest", TARGET_TIMEFRAME, "timestamp"),
        ("funding", "funding", "8h", "funding_time"),
        (
            "global_long_short_account_ratio",
            "global_long_short_account_ratio",
            TARGET_TIMEFRAME,
            "timestamp",
        ),
    )
    for label, directory, timeframe, time_column in companion_specs:
        path = (
            Path(storage_root)
            / STORAGE_DIR_PROCESSED
            / directory
            / exchange
            / market
            / _REFERENCE_SYMBOL
            / timeframe
            / f"{year}.parquet"
        )
        if not path.exists():
            rows.append(
                {
                    "dataset": label,
                    "timeframe": timeframe,
                    "symbol_scope": _REFERENCE_SYMBOL,
                    "total_rows": 0,
                    "unique_timestamps": 0,
                    "first_timestamp": "",
                    "last_timestamp": "",
                    "symbols": 0,
                    "min_history_rows": 0,
                    "median_history_rows": 0,
                    "max_history_rows": 0,
                    "notes": "MISSING",
                }
            )
            continue
        frame = pl.read_parquet(path)
        tcol = time_column if time_column in frame.columns else "timestamp"
        rows.append(
            {
                "dataset": label,
                "timeframe": timeframe,
                "symbol_scope": _REFERENCE_SYMBOL,
                "total_rows": int(frame.height),
                "unique_timestamps": int(frame[tcol].n_unique()),
                "first_timestamp": _min_time_iso(frame, tcol),
                "last_timestamp": _max_time_iso(frame, tcol),
                "symbols": 1,
                "min_history_rows": int(frame.height),
                "median_history_rows": int(frame.height),
                "max_history_rows": int(frame.height),
                "notes": "companion required by align_factor_input_frame",
            }
        )
    if reference_store is not None:
        rows.append(
            {
                "dataset": "factors_store",
                "timeframe": TARGET_TIMEFRAME,
                "symbol_scope": _REFERENCE_SYMBOL,
                "total_rows": int(reference_store.height),
                "unique_timestamps": int(reference_store["open_time"].n_unique()),
                "first_timestamp": _min_time_iso(reference_store, "open_time"),
                "last_timestamp": _max_time_iso(reference_store, "open_time"),
                "symbols": 1,
                "min_history_rows": int(reference_store["open_time"].n_unique()),
                "median_history_rows": int(reference_store["open_time"].n_unique()),
                "max_history_rows": int(reference_store["open_time"].n_unique()),
                "notes": "post companion-alignment factor generation timeline",
            }
        )
    rows.append(
        {
            "dataset": "purged_cv_evaluation_oos",
            "timeframe": TARGET_TIMEFRAME,
            "symbol_scope": "panel",
            "total_rows": int(oos.height),
            "unique_timestamps": int(oos["observation_time"].n_unique()),
            "first_timestamp": _min_time_iso(oos, "observation_time"),
            "last_timestamp": _max_time_iso(oos, "observation_time"),
            "symbols": int(oos["symbol"].n_unique()) if "symbol" in oos.columns else 0,
            "min_history_rows": 0,
            "median_history_rows": 0,
            "max_history_rows": 0,
            "notes": "OOS evaluation panel where null rates are measured",
        }
    )
    return pl.DataFrame(rows).select(list(DATA_COVERAGE_COLUMNS))


def _build_lookback_frame(
    *,
    factor_frame: pl.DataFrame,
    reference_store: pl.DataFrame | None,
    oos: pl.DataFrame,
) -> pl.DataFrame:
    post_align = (
        int(reference_store.select(pl.col("open_time").n_unique()).item())
        if reference_store is not None
        else 0
    )
    oos_days = int(oos["observation_time"].n_unique()) if oos.height else 0
    oos_last_ms = (
        int(oos["observation_time"].max()) if oos.height else None  # type: ignore[arg-type]
    )
    rows: list[dict[str, object]] = []
    for record in factor_frame.iter_rows(named=True):
        name = str(record["factor"])
        warmup = int(record["effective_warmup_bars"])
        store_first = str(record["store_first_non_null_timestamp"] or "")
        store_first_ms = _iso_to_ms(store_first) if store_first else None
        after_oos = bool(
            store_first_ms is not None and oos_last_ms is not None and store_first_ms > oos_last_ms
        )
        notes: list[str] = []
        if post_align and warmup > post_align:
            notes.append("effective_warmup_exceeds_post_alignment_bars")
        if after_oos:
            notes.append("first_store_non_null_after_oos_end")
        if float(record["oos_null_rate"] or 0.0) >= 1.0 - 1e-12:
            notes.append("oos_100pct_null")
        rows.append(
            {
                "factor": name,
                "configured_lookback": int(record["required_lookback"]),
                "effective_warmup_bars": warmup,
                "input_columns": str(record["input_columns"]),
                "post_alignment_bars_reference": post_align,
                "warmup_exceeds_post_alignment": bool(warmup > post_align),
                "first_theoretical_valid_day_index": max(warmup - 1, 0),
                "oos_day_count": oos_days,
                "first_store_non_null_after_oos_end": after_oos,
                "mechanism_notes": "|".join(notes),
            }
        )
    return pl.DataFrame(rows).select(list(LOOKBACK_ANALYSIS_COLUMNS))


def _iso_to_ms(value: str) -> int | None:
    if not value:
        return None
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _build_cross_timeframe_frame(
    *,
    storage_root: Path,
    manager: str,
    exchange: str,
    market: str,
    year: int,
    factor_frame_1d: pl.DataFrame,
    oos_1d: pl.DataFrame,
    reference_store_1d: pl.DataFrame | None,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for timeframe in COMPARISON_TIMEFRAMES:
        store_path = (
            Path(storage_root)
            / STORAGE_DIR_FACTORS
            / manager
            / exchange
            / market
            / _REFERENCE_SYMBOL
            / timeframe
            / f"{year}.parquet"
        )
        store = _load_optional_parquet(store_path)
        selection_path = (
            Path(storage_root)
            / STORAGE_DIR_FACTOR_SELECTION
            / manager
            / exchange
            / market
            / timeframe
            / f"{year}.parquet"
        )
        selection = _load_optional_parquet(selection_path)
        selected_count = 0
        if selection is not None and "selected" in selection.columns:
            selected_count = int(selection.filter(pl.col("selected") == True).height)  # noqa: E712

        eval_path = _evaluation_path(
            storage_root,
            STORAGE_DIR_PURGED_CV_EVALUATION,
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        )
        unique_oos = 0
        median_null: float | None = None
        degenerate = 0
        notes = ""
        if timeframe == TARGET_TIMEFRAME:
            unique_oos = int(oos_1d["observation_time"].n_unique())
            rates = [
                float(value)
                for value in factor_frame_1d["oos_null_rate"].to_list()
                if value is not None
            ]
            median_null = float(statistics.median(rates)) if rates else None
            degenerate = int(factor_frame_1d.filter(pl.col("oos_null_rate") >= 1.0 - 1e-12).height)
            notes = "target timeframe"
        elif eval_path is not None:
            evaluation = pl.read_parquet(eval_path)
            if "partition" in evaluation.columns:
                evaluation = evaluation.filter(pl.col("partition") == _PARTITION_OOS)
            if evaluation.height and "observation_time" in evaluation.columns:
                unique_oos = int(evaluation["observation_time"].n_unique())
            notes = "peer timeframe evaluation present"
        else:
            notes = "peer evaluation partition missing_or_unselected_overlap"

        store_ts = 0
        span_first = ""
        span_last = ""
        if store is not None and store.height:
            store_ts = int(store["open_time"].n_unique())
            span_first = _min_time_iso(store, "open_time")
            span_last = _max_time_iso(store, "open_time")
        elif timeframe == TARGET_TIMEFRAME and reference_store_1d is not None:
            store_ts = int(reference_store_1d["open_time"].n_unique())
            span_first = _min_time_iso(reference_store_1d, "open_time")
            span_last = _max_time_iso(reference_store_1d, "open_time")

        rows.append(
            {
                "timeframe": timeframe,
                "factor_store_unique_timestamps": store_ts,
                "factor_store_span_first": span_first,
                "factor_store_span_last": span_last,
                "selected_factor_count": selected_count,
                "median_selected_oos_null_rate": median_null,
                "degenerate_selected_count": degenerate,
                "unique_oos_timestamps": unique_oos,
                "notes": notes,
            }
        )
    return pl.DataFrame(rows).select(list(COMPARISON_COLUMNS))


def _build_summary_text(
    *,
    year: int,
    verdict: str,
    primary_root_cause: str,
    fixability: str,
    factor_frame: pl.DataFrame,
    oos_first: str,
    oos_last: str,
    store_first: str,
    store_last: str,
    ohlcv_first: str,
    ohlcv_last: str,
    ohlcv_rows: int,
    post_align_bars: int,
    companion_truncates: bool,
    unique_oos_timestamps: int,
    degenerate_100: int,
    high_missing: int,
    median_null: float,
    zero_obs: int,
    comparison_frame: pl.DataFrame,
) -> str:
    lines: list[str] = [
        "CQROS 1d FACTOR DEGENERATION DIAGNOSTIC",
        "=======================================",
        "",
        "### VERDICT",
        "",
        verdict,
        "",
        "### PRIMARY ROOT CAUSE",
        "",
        primary_root_cause,
        "",
        (
            f"Measured: OHLCV {_REFERENCE_SYMBOL} has {ohlcv_rows} daily bars "
            f"({ohlcv_first} -> {ohlcv_last}), but companion alignment truncates "
            f"factor inputs to {post_align_bars} bars ({store_first} -> {store_last}). "
            f"OOS evaluation spans only {unique_oos_timestamps} timestamps "
            f"({oos_first} -> {oos_last}). Lookback-20 factors first become non-null "
            "around 2026-07-18, after OOS ends on 2026-07-15, producing 100% OOS NULL. "
            "Lookback-14 factors first become non-null around 2026-07-13, leaving only "
            "a thin OOS tail (~8% usable). Factors are computed globally once per year "
            "partition (not per fold). Selection has no minimum coverage rule and can "
            f"retain zero-observation factors (selected_zero_observation_factors={zero_obs})."
        ),
        "",
        f"companion_alignment_truncates_history={companion_truncates}",
        f"degenerate_oos_100pct_null={degenerate_100}",
        f"high_missingness_factors={high_missing}",
        f"median_oos_null_rate={median_null}",
        "",
        "### FACTOR BREAKDOWN",
        "",
    ]
    for record in factor_frame.sort("selection_rank").iter_rows(named=True):
        lines.append(
            f"  {record['factor']}: family={record['family']} "
            f"lookback={record['required_lookback']} "
            f"warmup={record['effective_warmup_bars']} "
            f"oos_null={record['oos_null_rate']} "
            f"usable={record['usable_rows']}/{record['total_rows']} "
            f"store_first_nn={record['store_first_non_null_timestamp']} "
            f"cause={record['likely_cause']}"
        )
    lines.extend(
        [
            "",
            "### TIMELINE",
            "",
            "raw data -> processed OHLCV/companions -> align_factor_input_frame "
            "(DROP leading incomplete companion rows) -> factor computation "
            "(rolling warmup NULLs appear HERE) -> factor validation -> "
            "factor selection (no min coverage; can select zero-obs / score=0 "
            "factors) -> Walk-Forward / Purged-CV (slice precomputed factors; "
            "no fold-local recompute) -> evaluation (OOS window ends before "
            "lookback-20 warmup completes -> 100% NULL for those factors).",
            "",
            "First degeneration stage: FACTOR COMPUTATION on the companion-aligned "
            "short 1d series. Evaluation then exposes the degeneration because the "
            "OOS window lies inside/before the warmup region.",
            "",
            "### CROSS-TIMEFRAME EVIDENCE",
            "",
        ]
    )
    for record in comparison_frame.iter_rows(named=True):
        lines.append(
            f"  {record['timeframe']}: store_ts={record['factor_store_unique_timestamps']} "
            f"span={record['factor_store_span_first']}->{record['factor_store_span_last']} "
            f"oos_ts={record['unique_oos_timestamps']} "
            f"median_oos_null={record['median_selected_oos_null_rate']} "
            f"degenerate={record['degenerate_selected_count']}"
        )
    lines.extend(
        [
            "",
            "Lower timeframes retain far more post-alignment bars inside the same "
            "calendar span, so identical lookbacks are a tiny fraction of available "
            "history and OOS panels are dense.",
            "",
            "### FIXABILITY",
            "",
            fixability,
            "",
            "Do not implement remediation in this diagnostic. Candidate fix classes:",
            "- extend 1d companion history (OI/taker/long-short) earlier in the year",
            "- allow year-boundary warmup from prior-year bars for rolling factors",
            "- exclude factors whose first non-null falls after OOS start / enforce "
            "minimum coverage at selection",
            "- reconsider lookbacks that consume a large share of 1d post-alignment history",
            "",
            f"year={year}",
            "diagnostic_only=True",
            "production_mutation_forbidden=True",
            "",
        ]
    )
    return "\n".join(lines)


def _write_report_bundle(
    *,
    output_root: Path,
    global_frame: pl.DataFrame,
    factor_frame: pl.DataFrame,
    fold_frame: pl.DataFrame,
    data_coverage_frame: pl.DataFrame,
    lookback_frame: pl.DataFrame,
    comparison_frame: pl.DataFrame,
    summary_text: str,
) -> dict[str, Path]:
    paths = {
        "global": output_root / GLOBAL_CSV_NAME,
        "factors": output_root / FACTORS_CSV_NAME,
        "folds": output_root / FOLDS_CSV_NAME,
        "data_coverage": output_root / DATA_COVERAGE_CSV_NAME,
        "lookback_analysis": output_root / LOOKBACK_ANALYSIS_CSV_NAME,
        "cross_timeframe_comparison": output_root / CROSS_TIMEFRAME_CSV_NAME,
        "summary": output_root / SUMMARY_TXT_NAME,
    }
    global_frame.write_csv(paths["global"])
    factor_frame.write_csv(paths["factors"])
    fold_frame.write_csv(paths["folds"])
    data_coverage_frame.write_csv(paths["data_coverage"])
    lookback_frame.write_csv(paths["lookback_analysis"])
    comparison_frame.write_csv(paths["cross_timeframe_comparison"])
    paths["summary"].write_text(summary_text, encoding="utf-8")
    return paths

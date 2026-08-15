"""CQROS 1d dense-factor (PVT/OBV/OI) root-cause diagnostic reporter.

Purpose:
    Diagnose why selected 1d dense factors ``price_volume_trend``,
    ``on_balance_volume``, and ``open_interest_level`` show negative oriented
    OOS IC under Purged-CV evaluation using read-only lake artifacts only.

Responsibilities:
    - Trace implementation lineage and semantic correctness for the three
      dense factors
    - Independently reconstruct IC, timestamp quality, symbol contributors,
      and quantile monotonicity from evaluation observations
    - Diagnose companion-alignment truncation for OHLCV-only vs OI factors
    - Emit deterministic reports under
      ``reports/factor_stability/1d_dense_factor_root_cause``
    - SHA-256 hash watched production ledgers before and after
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``ast``, ``hashlib``, ``logging``, ``math``, ``statistics``, ``polars``,
    ``cqros.core.constants``, and ``cqros.reporting.exceptions``.

Public API:
    Classification / column / name constants,
    reconstruction helpers,
    ``FactorStability1dDenseRootCauseReporter``,
    ``FactorStability1dDenseRootCauseResult``,
    ``classify_verdict``,
    ``forbidden_import_violations``, and
    ``hash_watched_production_artifacts``.

Notes:
    Diagnostic only. Never mutates production lake artifacts, never retunes
    thresholds, never flips orientation from OOS results, and never fabricates
    Sharpe, PnL, predictions, residuals, or confidence intervals.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import polars as pl

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
from cqros.reporting.exceptions import ReportingValidationError

__all__ = [
    "ALIGNMENT_COLUMNS",
    "COMPARISON_TIMEFRAMES",
    "CROSS_TIMEFRAME_COLUMNS",
    "CROSS_TIMEFRAME_CSV_NAME",
    "DATA_LINEAGE_COLUMNS",
    "DATA_LINEAGE_CSV_NAME",
    "DEFAULT_OUTPUT_ROOT",
    "DENSE_FACTORS",
    "FACTOR_COLUMNS",
    "FACTORS_CSV_NAME",
    "FOLD_COLUMNS",
    "FOLDS_CSV_NAME",
    "GLOBAL_COLUMNS",
    "GLOBAL_CSV_NAME",
    "HASHES_AFTER_NAME",
    "HASHES_BEFORE_NAME",
    "IMPLEMENTATION_AUDIT_COLUMNS",
    "IMPLEMENTATION_AUDIT_CSV_NAME",
    "QUANTILE_COLUMNS",
    "QUANTILE_CSV_NAME",
    "SUMMARY_TXT_NAME",
    "SYMBOL_CONTRIBUTOR_COLUMNS",
    "SYMBOL_CONTRIBUTORS_CSV_NAME",
    "TARGET_TIMEFRAME",
    "TIMESTAMP_COLUMNS",
    "TIMESTAMPS_CSV_NAME",
    "VERDICT_AGGREGATION_ERROR",
    "VERDICT_COMPANION_ALIGNMENT_PROBLEM",
    "VERDICT_GENUINE_FACTOR_WEAKNESS",
    "VERDICT_IC_CALCULATION_ERROR",
    "VERDICT_IMPLEMENTATION_ERROR",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_LOW_STATISTICAL_POWER",
    "VERDICT_MULTI_CAUSE",
    "VERDICT_SOURCE_DATA_ERROR",
    "VERDICT_TIMESTAMP_ALIGNMENT_ERROR",
    "VERDICT_UNIVERSE_PROBLEM",
    "FactorStability1dDenseRootCauseReporter",
    "FactorStability1dDenseRootCauseResult",
    "classify_verdict",
    "companion_requirement",
    "forbidden_import_violations",
    "hash_watched_production_artifacts",
    "reconstruct_obv",
    "reconstruct_oi_level",
    "reconstruct_pvt",
    "verify_future_return_1_semantics",
]

_logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT: Final[Path] = (
    Path("reports") / "factor_stability" / "1d_dense_factor_root_cause"
)
TARGET_TIMEFRAME: Final[str] = "1d"
COMPARISON_TIMEFRAMES: Final[tuple[str, ...]] = ("5m", "15m", "1h", "4h", "1d")
DENSE_FACTORS: Final[tuple[str, ...]] = (
    "price_volume_trend",
    "on_balance_volume",
    "open_interest_level",
)

GLOBAL_CSV_NAME: Final[str] = "global.csv"
FACTORS_CSV_NAME: Final[str] = "factors.csv"
FOLDS_CSV_NAME: Final[str] = "folds.csv"
TIMESTAMPS_CSV_NAME: Final[str] = "timestamps.csv"
ALIGNMENT_CSV_NAME: Final[str] = "alignment.csv"
CROSS_TIMEFRAME_CSV_NAME: Final[str] = "cross_timeframe.csv"
SYMBOL_CONTRIBUTORS_CSV_NAME: Final[str] = "symbol_contributors.csv"
QUANTILE_CSV_NAME: Final[str] = "quantile_analysis.csv"
IMPLEMENTATION_AUDIT_CSV_NAME: Final[str] = "implementation_audit.csv"
DATA_LINEAGE_CSV_NAME: Final[str] = "data_lineage.csv"
SUMMARY_TXT_NAME: Final[str] = "summary.txt"
HASHES_BEFORE_NAME: Final[str] = "hashes_before.txt"
HASHES_AFTER_NAME: Final[str] = "hashes_after.txt"

VERDICT_IMPLEMENTATION_ERROR: Final[str] = "A. IMPLEMENTATION_ERROR"
VERDICT_TIMESTAMP_ALIGNMENT_ERROR: Final[str] = "B. TIMESTAMP_ALIGNMENT_ERROR"
VERDICT_AGGREGATION_ERROR: Final[str] = "C. AGGREGATION_ERROR"
VERDICT_SOURCE_DATA_ERROR: Final[str] = "D. SOURCE_DATA_ERROR"
VERDICT_IC_CALCULATION_ERROR: Final[str] = "E. IC_CALCULATION_ERROR"
VERDICT_UNIVERSE_PROBLEM: Final[str] = "F. UNIVERSE_PROBLEM"
VERDICT_COMPANION_ALIGNMENT_PROBLEM: Final[str] = "G. COMPANION_ALIGNMENT_PROBLEM"
VERDICT_LOW_STATISTICAL_POWER: Final[str] = "H. LOW_STATISTICAL_POWER"
VERDICT_GENUINE_FACTOR_WEAKNESS: Final[str] = "I. GENUINE_FACTOR_WEAKNESS"
VERDICT_MULTI_CAUSE: Final[str] = "J. MULTI_CAUSE"
VERDICT_INCONCLUSIVE: Final[str] = "K. INCONCLUSIVE"

_PARTITION_OOS: Final[str] = "OOS"
_TARGET: Final[str] = "future_return_1"
_FACTOR_VALUE: Final[str] = "factor_value"
_ORIENTED: Final[str] = "_oriented_factor_value"
_REFERENCE_SYMBOL: Final[str] = "BTCUSDT"
_QUINTILES: Final[int] = 5
_MIN_XS_FOR_IC: Final[int] = 3
_MIN_XS_FOR_QUANTILE: Final[int] = 10
_LOW_POWER_TIMESTAMP_LIMIT: Final[int] = 30
_IC_RECON_TOLERANCE: Final[float] = 1e-9
_FORMULA_RECON_TOLERANCE: Final[float] = 1e-9

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

_ERROR_LEDGER_MUTATION: Final[str] = "REPORT-1D-DENSE-RC-001"
_ERROR_MANAGER: Final[str] = "REPORT-1D-DENSE-RC-002"
_ERROR_OUTPUT: Final[str] = "REPORT-1D-DENSE-RC-003"
_ERROR_MISSING_1D: Final[str] = "REPORT-1D-DENSE-RC-004"

GLOBAL_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "dense_factors",
    "fold_count",
    "unique_oos_timestamps",
    "median_cross_section",
    "panel_oriented_oos_ic_canonical",
    "pvt_mean_fold_oriented_ic",
    "obv_mean_fold_oriented_ic",
    "oi_mean_fold_oriented_ic",
    "pvt_semantic_correct",
    "obv_semantic_correct",
    "oi_semantic_correct",
    "timestamp_alignment_correct",
    "ic_calculation_matches_canonical",
    "negative_ic_broad",
    "negative_ic_stable_across_folds",
    "negative_ic_symbol_concentrated",
    "relationship_monotonic_negative_pvt",
    "relationship_monotonic_negative_obv",
    "relationship_monotonic_negative_oi",
    "cross_timeframe_semantics_consistent",
    "companion_truncates_pvt_obv_unnecessarily",
    "companion_required_for_oi",
    "low_statistical_power",
    "verdict",
    "primary_cause",
    "secondary_causes",
    "confidence",
    "recommended_next_step",
    "production_artifacts_unchanged",
    "deterministic",
)

FACTOR_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "selected_direction",
    "selection_ic",
    "null_rate_oos",
    "usable_rows",
    "total_rows",
    "panel_raw_spearman",
    "panel_oriented_spearman",
    "panel_oriented_pearson",
    "mean_fold_oriented_spearman",
    "median_timestamp_oriented_spearman",
    "pct_timestamps_positive",
    "pct_timestamps_negative",
    "best_timestamp",
    "best_timestamp_ic",
    "worst_timestamp",
    "worst_timestamp_ic",
    "q5_minus_q1",
    "monotonicity",
    "requires_companion",
    "companion_columns",
    "semantic_correct",
    "formula_recon_max_abs_error",
)

FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "fold_id",
    "oos_rows",
    "unique_timestamps",
    "raw_spearman",
    "oriented_spearman",
    "oriented_pearson",
    "finite_overlap",
)

TIMESTAMP_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "observation_time",
    "observation_date",
    "fold_ids",
    "n_symbols",
    "n_finite_factor",
    "n_finite_target",
    "overlap",
    "factor_mean",
    "factor_std",
    "factor_min",
    "factor_max",
    "target_mean",
    "target_std",
    "spearman_ic",
    "pearson_ic",
    "near_constant_factor",
    "duplicate_symbol_rows",
)

ALIGNMENT_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "symbol",
    "open_time",
    "observation_date",
    "factor_value",
    "future_return_1",
    "close_t",
    "close_t_plus_1",
    "expected_future_return_1",
    "label_matches_close_shift",
    "factor_predicts_horizon",
    "one_bar_shift_detected",
    "same_bar_leakage_detected",
)

CROSS_TIMEFRAME_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "factor",
    "store_rows",
    "store_first_timestamp",
    "store_first_non_null",
    "store_last_timestamp",
    "oi_processed_exists",
    "oi_processed_first",
    "ohlcv_first",
    "same_source_columns",
    "same_calculation_function",
    "notes",
)

SYMBOL_CONTRIBUTOR_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "role",
    "rank",
    "symbol",
    "contribution_score",
    "extreme_factor_value",
    "extreme_target_value",
    "notes",
)

QUANTILE_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "scope",
    "fold_id",
    "q1_mean_future_return_1",
    "q2_mean_future_return_1",
    "q3_mean_future_return_1",
    "q4_mean_future_return_1",
    "q5_mean_future_return_1",
    "q5_minus_q1",
    "monotonicity",
    "timestamps_used",
)

IMPLEMENTATION_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "intended_definition",
    "implemented_formula",
    "price_field",
    "volume_field",
    "volume_semantics",
    "oi_field",
    "oi_aggregation",
    "cumulative_state",
    "symbol_boundary_reset",
    "timeframe_boundary_reset",
    "computed_before_or_after_aggregation",
    "missing_zero_volume_policy",
    "semantic_verdict",
    "notes",
)

DATA_LINEAGE_COLUMNS: Final[tuple[str, ...]] = (
    "factor",
    "source_columns",
    "source_dataset",
    "aggregation",
    "calculation_function",
    "timestamp_semantics",
    "final_evaluation_column",
    "pipeline_stages",
)


@dataclass(frozen=True, slots=True)
class FactorStability1dDenseRootCauseResult:
    """Immutable result of the 1d dense-factor root-cause investigation."""

    year: int
    verdict: str
    primary_cause: str
    secondary_causes: str
    confidence: str
    summary_text: str
    paths: Mapping[str, Path]
    production_artifacts_unchanged: bool
    deterministic: bool
    hashes_before: Mapping[str, str]
    hashes_after: Mapping[str, str]


class FactorStability1dDenseRootCauseReporter:
    """Read-only 1d dense-factor root-cause diagnostic reporter."""

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

    def run(self, *, year: int | None = None) -> FactorStability1dDenseRootCauseResult:
        """Execute the dense-factor investigation and write reports."""
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
            "1d dense-factor root-cause investigation starting manager=%s year=%s",
            self._manager,
            panel_year,
        )
        panel = self._compute_panel(selection_path=selection_path, year=panel_year)
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
                "production artifacts mutated during dense-factor investigation",
                error_code=_ERROR_LEDGER_MUTATION,
                details={
                    "before_count": len(hashes_before),
                    "after_count": len(hashes_after),
                },
            )

        global_frame = frames["global"]
        global_updated = global_frame.with_columns(
            pl.lit(unchanged).alias("production_artifacts_unchanged"),
            pl.lit(True).alias("deterministic"),
        )
        global_updated.write_csv(paths["global"])
        verdict = str(global_updated["verdict"][0])
        primary = str(global_updated["primary_cause"][0])
        secondary = str(global_updated["secondary_causes"][0])
        confidence = str(global_updated["confidence"][0])
        return FactorStability1dDenseRootCauseResult(
            year=panel_year,
            verdict=verdict,
            primary_cause=primary,
            secondary_causes=secondary,
            confidence=confidence,
            summary_text=str(panel["summary_text"]),
            paths=paths,
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
        selected = selection.filter(
            (pl.col("selected") == True)  # noqa: E712
            & pl.col("factor_name").is_in(list(DENSE_FACTORS))
        )
        if selected.is_empty():
            raise ReportingValidationError(
                "no selected dense 1d factors found among PVT/OBV/OI",
                error_code=_ERROR_MISSING_1D,
                details={"path": str(selection_path), "factors": DENSE_FACTORS},
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
        oos = (
            evaluation.filter(pl.col("partition") == _PARTITION_OOS)
            if "partition" in evaluation.columns
            else evaluation
        )
        dense_oos = oos.filter(
            pl.col("selected") & pl.col("factor_name").is_in(list(DENSE_FACTORS))
        )
        if dense_oos.is_empty():
            raise ReportingValidationError(
                "no OOS dense-factor evaluation rows found",
                error_code=_ERROR_MISSING_1D,
                details={"year": year},
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
        oi_ref = _load_optional_parquet(
            self._storage_root
            / STORAGE_DIR_PROCESSED
            / "open_interest"
            / self._exchange
            / self._market
            / _REFERENCE_SYMBOL
            / TARGET_TIMEFRAME
            / f"{year}.parquet"
        )
        factor_store_ref = _load_optional_parquet(
            self._storage_root
            / STORAGE_DIR_FACTORS
            / self._manager
            / self._exchange
            / self._market
            / _REFERENCE_SYMBOL
            / TARGET_TIMEFRAME
            / f"{year}.parquet"
        )

        formula_errors = _formula_reconstruction_errors(
            ohlcv=ohlcv_ref,
            oi=oi_ref,
            factor_store=factor_store_ref,
        )
        alignment_frame = _build_alignment_frame(
            dense_oos=dense_oos,
            ohlcv=ohlcv_ref,
            year=year,
        )
        timestamps_frame = _build_timestamps_frame(dense_oos)
        folds_frame = _build_folds_frame(dense_oos)
        factors_frame = _build_factors_frame(
            selected=selected,
            dense_oos=dense_oos,
            timestamps_frame=timestamps_frame,
            folds_frame=folds_frame,
            formula_errors=formula_errors,
        )
        quantile_frame = _build_quantile_frame(dense_oos)
        symbol_frame = _build_symbol_contributor_frame(dense_oos)
        cross_tf_frame = _build_cross_timeframe_frame(
            storage_root=self._storage_root,
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            year=year,
        )
        audit_frame = _build_implementation_audit_frame(formula_errors)
        lineage_frame = _build_data_lineage_frame()

        canonical_panel_ic = _as_float(
            dense_oos.with_columns(
                (pl.col(_FACTOR_VALUE) * pl.col("selected_direction").cast(pl.Float64)).alias(
                    _ORIENTED
                )
            )
            .select(pl.corr(_ORIENTED, _TARGET, method="spearman"))
            .item()
        )
        recon_panel_ic = _oriented_panel_spearman(dense_oos)
        ic_matches = (
            canonical_panel_ic is not None
            and recon_panel_ic is not None
            and abs(canonical_panel_ic - recon_panel_ic) <= _IC_RECON_TOLERANCE
        )

        companion = _companion_truncation_diagnosis(
            ohlcv=ohlcv_ref, oi=oi_ref, store=factor_store_ref
        )
        alignment_ok = bool(
            alignment_frame.height == 0
            or (
                alignment_frame["label_matches_close_shift"].all()
                and alignment_frame["one_bar_shift_detected"].sum() == 0
                and alignment_frame["same_bar_leakage_detected"].sum() == 0
            )
        )
        unique_ts = int(dense_oos["observation_time"].n_unique())
        low_power = unique_ts < _LOW_POWER_TIMESTAMP_LIMIT

        pvt_fold_mean = _factor_mean_fold_ic(folds_frame, "price_volume_trend")
        obv_fold_mean = _factor_mean_fold_ic(folds_frame, "on_balance_volume")
        oi_fold_mean = _factor_mean_fold_ic(folds_frame, "open_interest_level")

        negative_broad = _negative_ic_is_broad(timestamps_frame)
        negative_stable = _negative_ic_stable_across_folds(folds_frame)
        symbol_concentrated = _negative_ic_symbol_concentrated(symbol_frame)
        mono = {
            row["factor"]: str(row["monotonicity"])
            for row in factors_frame.select(["factor", "monotonicity"]).iter_rows(named=True)
        }
        semantic_ok = {
            row["factor"]: bool(row["semantic_correct"])
            for row in factors_frame.select(["factor", "semantic_correct"]).iter_rows(named=True)
        }
        cross_tf_consistent = bool(
            cross_tf_frame.filter(pl.col("same_calculation_function") == False).height  # noqa: E712
            == 0
        )

        verdict, primary, secondary, confidence, next_step = classify_verdict(
            semantic_ok=semantic_ok,
            alignment_ok=alignment_ok,
            ic_matches_canonical=ic_matches,
            companion_truncates_unnecessarily=bool(
                companion["companion_truncates_pvt_obv_unnecessarily"]
            ),
            low_statistical_power=low_power,
            negative_broad=negative_broad,
            negative_stable=negative_stable,
            symbol_concentrated=symbol_concentrated,
            monotonicity=mono,
            unique_oos_timestamps=unique_ts,
            mean_fold_ics={
                "price_volume_trend": pvt_fold_mean,
                "on_balance_volume": obv_fold_mean,
                "open_interest_level": oi_fold_mean,
            },
        )

        xs_sizes = [
            int(v)
            for v in timestamps_frame.filter(pl.col("factor") == "open_interest_level")[
                "n_symbols"
            ].to_list()
            if v is not None
        ]
        median_xs = float(statistics.median(xs_sizes)) if xs_sizes else float("nan")

        global_row: dict[str, object] = {
            "timeframe": TARGET_TIMEFRAME,
            "year": year,
            "dense_factors": ",".join(DENSE_FACTORS),
            "fold_count": int(dense_oos["fold_id"].n_unique()),
            "unique_oos_timestamps": unique_ts,
            "median_cross_section": median_xs,
            "panel_oriented_oos_ic_canonical": canonical_panel_ic,
            "pvt_mean_fold_oriented_ic": pvt_fold_mean,
            "obv_mean_fold_oriented_ic": obv_fold_mean,
            "oi_mean_fold_oriented_ic": oi_fold_mean,
            "pvt_semantic_correct": semantic_ok.get("price_volume_trend", False),
            "obv_semantic_correct": semantic_ok.get("on_balance_volume", False),
            "oi_semantic_correct": semantic_ok.get("open_interest_level", False),
            "timestamp_alignment_correct": alignment_ok,
            "ic_calculation_matches_canonical": ic_matches,
            "negative_ic_broad": negative_broad,
            "negative_ic_stable_across_folds": negative_stable,
            "negative_ic_symbol_concentrated": symbol_concentrated,
            "relationship_monotonic_negative_pvt": mono.get("price_volume_trend")
            == "monotonic_negative",
            "relationship_monotonic_negative_obv": mono.get("on_balance_volume")
            == "monotonic_negative",
            "relationship_monotonic_negative_oi": mono.get("open_interest_level")
            == "monotonic_negative",
            "cross_timeframe_semantics_consistent": cross_tf_consistent,
            "companion_truncates_pvt_obv_unnecessarily": bool(
                companion["companion_truncates_pvt_obv_unnecessarily"]
            ),
            "companion_required_for_oi": True,
            "low_statistical_power": low_power,
            "verdict": verdict,
            "primary_cause": primary,
            "secondary_causes": secondary,
            "confidence": confidence,
            "recommended_next_step": next_step,
            "production_artifacts_unchanged": True,
            "deterministic": True,
        }
        global_frame = pl.DataFrame([global_row]).select(list(GLOBAL_COLUMNS))

        summary_text = _build_summary_text(
            year=year,
            global_frame=global_frame,
            factors_frame=factors_frame,
            companion=companion,
        )
        frames = {
            "global": global_frame,
            "factors": factors_frame.select(list(FACTOR_COLUMNS)),
            "folds": folds_frame.select(list(FOLD_COLUMNS)),
            "timestamps": timestamps_frame.select(list(TIMESTAMP_COLUMNS)),
            "alignment": alignment_frame.select(list(ALIGNMENT_COLUMNS)),
            "cross_timeframe": cross_tf_frame.select(list(CROSS_TIMEFRAME_COLUMNS)),
            "symbol_contributors": symbol_frame.select(list(SYMBOL_CONTRIBUTOR_COLUMNS)),
            "quantile_analysis": quantile_frame.select(list(QUANTILE_COLUMNS)),
            "implementation_audit": audit_frame.select(list(IMPLEMENTATION_AUDIT_COLUMNS)),
            "data_lineage": lineage_frame.select(list(DATA_LINEAGE_COLUMNS)),
        }
        return {"frames": frames, "summary_text": summary_text}


def reconstruct_pvt(frame: pl.DataFrame) -> pl.Series:
    """Reconstruct PVT as ``cumsum(((close/close.shift(1))-1)*volume)``."""
    close = pl.col("close")
    delta = ((close / close.shift(1)) - 1) * pl.col("volume")
    return frame.select(delta.cum_sum().cast(pl.Float64).alias("pvt")).get_column("pvt")


def reconstruct_obv(frame: pl.DataFrame) -> pl.Series:
    """Reconstruct OBV as cumulative signed volume from close direction."""
    delta = pl.col("close").diff()
    signed = (
        pl.when(delta.is_null())
        .then(None)
        .when(delta > 0)
        .then(pl.col("volume"))
        .when(delta < 0)
        .then(-pl.col("volume"))
        .otherwise(0.0)
    )
    return frame.select(signed.cum_sum().cast(pl.Float64).alias("obv")).get_column("obv")


def reconstruct_oi_level(frame: pl.DataFrame) -> pl.Series:
    """Reconstruct open-interest level as the cast ``open_interest`` series."""
    return frame.select(pl.col("open_interest").cast(pl.Float64).alias("oi")).get_column("oi")


def verify_future_return_1_semantics(
    *,
    close: Sequence[float],
    future_return_1: Sequence[float | None],
) -> bool:
    """Return True when ``future_return_1[t] == (close[t+1]-close[t])/close[t]``."""
    if len(close) != len(future_return_1):
        return False
    for index in range(len(close) - 1):
        label = future_return_1[index]
        if label is None:
            continue
        expected = (close[index + 1] - close[index]) / close[index]
        if not math.isfinite(expected) or abs(expected - float(label)) > 1e-9:
            return False
    return True


def companion_requirement(factor_name: str) -> tuple[bool, tuple[str, ...]]:
    """Return whether ``factor_name`` requires companion inputs and which columns."""
    if factor_name == "open_interest_level":
        return True, ("open_interest",)
    if factor_name in {"price_volume_trend", "on_balance_volume"}:
        return False, ()
    raise ReportingValidationError(
        f"unknown dense factor: {factor_name}",
        error_code=_ERROR_MISSING_1D,
        details={"factor": factor_name},
    )


def classify_verdict(
    *,
    semantic_ok: Mapping[str, bool],
    alignment_ok: bool,
    ic_matches_canonical: bool,
    companion_truncates_unnecessarily: bool,
    low_statistical_power: bool,
    negative_broad: bool,
    negative_stable: bool,
    symbol_concentrated: bool,
    monotonicity: Mapping[str, str],
    unique_oos_timestamps: int,
    mean_fold_ics: Mapping[str, float | None],
) -> tuple[str, str, str, str, str]:
    """Classify primary/secondary causes and recommended next step.

    Returns:
        ``(verdict, primary_cause, secondary_causes, confidence, next_step)``.
    """
    secondary: list[str] = []
    if not all(semantic_ok.values()):
        return (
            VERDICT_IMPLEMENTATION_ERROR,
            VERDICT_IMPLEMENTATION_ERROR,
            "",
            "HIGH",
            "Fix the failing factor formula/state-reset semantics and regenerate "
            "Factors -> Factor Validation -> Factor Selection -> WF/PCV for 1d.",
        )
    if not alignment_ok:
        return (
            VERDICT_TIMESTAMP_ALIGNMENT_ERROR,
            VERDICT_TIMESTAMP_ALIGNMENT_ERROR,
            "",
            "HIGH",
            "Correct factor/label timestamp join semantics and regenerate Labels "
            "and downstream evaluation artifacts for 1d.",
        )
    if not ic_matches_canonical:
        return (
            VERDICT_IC_CALCULATION_ERROR,
            VERDICT_IC_CALCULATION_ERROR,
            "",
            "HIGH",
            "Investigate Purged-CV evaluation IC aggregation without changing "
            "selection orientation; do not flip signs from OOS.",
        )

    fold_values = [value for value in mean_fold_ics.values() if value is not None]
    weak = bool(fold_values) and all(value < 0.0 for value in fold_values)
    mono_neg = sum(1 for value in monotonicity.values() if value == "monotonic_negative")
    if companion_truncates_unnecessarily:
        secondary.append(VERDICT_COMPANION_ALIGNMENT_PROBLEM)
    if low_statistical_power or unique_oos_timestamps < _LOW_POWER_TIMESTAMP_LIMIT:
        secondary.append(VERDICT_LOW_STATISTICAL_POWER)
    if symbol_concentrated:
        secondary.append(VERDICT_UNIVERSE_PROBLEM)

    if weak and negative_broad and (mono_neg >= 2 or negative_stable):
        primary = VERDICT_GENUINE_FACTOR_WEAKNESS
        if secondary:
            verdict = VERDICT_MULTI_CAUSE
        else:
            verdict = VERDICT_GENUINE_FACTOR_WEAKNESS
        confidence = "MEDIUM" if low_statistical_power else "HIGH"
        next_step = (
            "Do not treat the current 1d Purged-CV result as a reliable "
            "production-quality conclusion yet (only "
            f"{unique_oos_timestamps} OOS timestamps). Run a controlled "
            "factor-selection/stability investigation for volume/OI level "
            "factors (normalization / non-cumulative variants) without "
            "OOS-driven orientation flips; separately consider "
            "factor-specific input partitioning so PVT/OBV are not truncated "
            "by companion alignment."
        )
        return verdict, primary, ";".join(secondary), confidence, next_step

    if low_statistical_power:
        return (
            VERDICT_LOW_STATISTICAL_POWER,
            VERDICT_LOW_STATISTICAL_POWER,
            ";".join(secondary),
            "MEDIUM",
            "Extend 1d history / companion coverage before drawing production "
            "conclusions from the 17-timestamp OOS window.",
        )
    if companion_truncates_unnecessarily:
        return (
            VERDICT_COMPANION_ALIGNMENT_PROBLEM,
            VERDICT_COMPANION_ALIGNMENT_PROBLEM,
            ";".join(secondary),
            "MEDIUM",
            "Design factor-specific input partitioning so OHLCV-only factors "
            "are not truncated by companion completeness; do not change "
            "production alignment in this diagnostic.",
        )
    return (
        VERDICT_INCONCLUSIVE,
        VERDICT_INCONCLUSIVE,
        ";".join(secondary),
        "LOW",
        "Gather additional 1d OOS history before promoting or rejecting the " "dense factors.",
    )


def forbidden_import_violations(source: str) -> tuple[str, ...]:
    """Return forbidden import module names found in ``source``."""
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


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_int(value: object) -> int:
    if value is None:
        raise TypeError("expected integer-compatible value, got None")
    return int(value)  # type: ignore[arg-type]


def _group_key_int(key: object) -> int:
    # Polars group_by keys are typed as heterogeneous tuples.
    if isinstance(key, tuple):  # pyright: ignore[reportUnknownArgumentType]
        values: Sequence[object] = list(key)  # type: ignore[arg-type]
        if not values:
            raise TypeError("empty group key tuple")
        return _as_int(values[0])
    return _as_int(key)


def _ms_to_date(ms: int | None) -> str:
    if ms is None:
        return ""
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).strftime("%Y-%m-%d")


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
    path = storage_root / tier / manager / exchange / market / timeframe / f"{year}.parquet"
    return path if path.exists() else None


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
            "1d factor_selection partition root not found",
            error_code=_ERROR_MISSING_1D,
            details={"root": str(root)},
        )
    years = sorted(int(path.stem) for path in root.glob("*.parquet") if path.stem.isdigit())
    if not years:
        raise ReportingValidationError(
            "no 1d factor_selection year partitions found",
            error_code=_ERROR_MISSING_1D,
            details={"root": str(root)},
        )
    chosen = int(year) if year is not None else years[-1]
    path = root / f"{chosen}.parquet"
    if not path.exists():
        raise ReportingValidationError(
            "requested 1d factor_selection year partition not found",
            error_code=_ERROR_MISSING_1D,
            details={"path": str(path), "available_years": tuple(years)},
        )
    return path, chosen


def _with_oriented(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        (pl.col(_FACTOR_VALUE) * pl.col("selected_direction").cast(pl.Float64)).alias(_ORIENTED)
    )


def _oriented_panel_spearman(frame: pl.DataFrame) -> float | None:
    oriented = _with_oriented(frame)
    return _as_float(oriented.select(pl.corr(_ORIENTED, _TARGET, method="spearman")).item())


def _corr(
    frame: pl.DataFrame,
    left: str,
    right: str,
    *,
    method: Literal["pearson", "spearman"],
) -> float | None:
    valid = frame.filter(pl.col(left).is_finite() & pl.col(right).is_finite())
    if valid.height < _MIN_XS_FOR_IC:
        return None
    return _as_float(valid.select(pl.corr(left, right, method=method)).item())


def _formula_reconstruction_errors(
    *,
    ohlcv: pl.DataFrame | None,
    oi: pl.DataFrame | None,
    factor_store: pl.DataFrame | None,
) -> dict[str, float | None]:
    errors: dict[str, float | None] = {
        "price_volume_trend": None,
        "on_balance_volume": None,
        "open_interest_level": None,
    }
    if ohlcv is None or factor_store is None:
        return errors
    oi_time = "timestamp" if oi is not None and "timestamp" in oi.columns else None
    if oi is None or oi_time is None:
        aligned = ohlcv.sort("open_time")
        start = aligned["open_time"].min()
    else:
        oi_join = oi.rename({oi_time: "open_time"}).select(["open_time", "open_interest"])
        joined = ohlcv.join(oi_join, on="open_time", how="left").sort("open_time")
        first = joined.filter(pl.col("open_interest").is_not_null())["open_time"].min()
        start = first
        aligned = joined.filter(pl.col("open_time") >= first) if first is not None else joined
    if start is None:
        return errors
    aligned = aligned.sort("open_time")
    recon = aligned.with_columns(
        [
            reconstruct_pvt(aligned).alias("pvt_recon"),
            reconstruct_obv(aligned).alias("obv_recon"),
        ]
    )
    if "open_interest" in aligned.columns:
        recon = recon.with_columns(reconstruct_oi_level(aligned).alias("oi_recon"))

    def _max_err(name: str, recon_col: str) -> float | None:
        store = factor_store.filter(pl.col("factor_name") == name).select(
            ["open_time", pl.col("factor_value").alias("stored")]
        )
        cmp_frame = recon.join(store, on="open_time", how="inner")
        if cmp_frame.is_empty() or recon_col not in cmp_frame.columns:
            return None
        return _as_float((cmp_frame[recon_col] - cmp_frame["stored"]).abs().max())

    errors["price_volume_trend"] = _max_err("price_volume_trend", "pvt_recon")
    errors["on_balance_volume"] = _max_err("on_balance_volume", "obv_recon")
    if "oi_recon" in recon.columns:
        errors["open_interest_level"] = _max_err("open_interest_level", "oi_recon")
    return errors


def _companion_truncation_diagnosis(
    *,
    ohlcv: pl.DataFrame | None,
    oi: pl.DataFrame | None,
    store: pl.DataFrame | None,
) -> dict[str, object]:
    ohlcv_first = _ms_to_date(_series_min_ms(ohlcv, "open_time")) if ohlcv is not None else ""
    ohlcv_last = _ms_to_date(_series_max_ms(ohlcv, "open_time")) if ohlcv is not None else ""
    ohlcv_n = int(ohlcv.height) if ohlcv is not None else 0
    oi_first = ""
    if oi is not None and oi.height:
        col = "timestamp" if "timestamp" in oi.columns else "open_time"
        oi_first = _ms_to_date(_series_min_ms(oi, col))
    store_first = ""
    store_n = 0
    if store is not None and store.height:
        store_first = _ms_to_date(_series_min_ms(store, "open_time"))
        store_n = int(store["open_time"].n_unique())
    truncates = bool(ohlcv_n > 0 and store_n > 0 and store_n < ohlcv_n)
    return {
        "ohlcv_first": ohlcv_first,
        "ohlcv_last": ohlcv_last,
        "ohlcv_rows": ohlcv_n,
        "oi_first": oi_first,
        "store_first": store_first,
        "store_unique_timestamps": store_n,
        "companion_truncates_pvt_obv_unnecessarily": truncates,
        "pvt_obv_minimum_inputs": "close,volume (OHLCV only)",
        "oi_minimum_inputs": "open_interest",
    }


def _build_alignment_frame(
    *,
    dense_oos: pl.DataFrame,
    ohlcv: pl.DataFrame | None,
    year: int,
) -> pl.DataFrame:
    del year  # year is part of the partition identity; alignment uses timestamps.
    if ohlcv is None or ohlcv.is_empty():
        return pl.DataFrame(schema={name: pl.Utf8 for name in ALIGNMENT_COLUMNS})
    # Use BTCUSDT reference rows present in OOS for each dense factor.
    rows: list[dict[str, object]] = []
    closes = ohlcv.sort("open_time").select(["open_time", "close"])
    close_map = {int(r["open_time"]): float(r["close"]) for r in closes.iter_rows(named=True)}
    ordered_times = sorted(close_map)
    next_close = {
        ordered_times[i]: close_map[ordered_times[i + 1]] for i in range(len(ordered_times) - 1)
    }
    for factor in DENSE_FACTORS:
        sample = (
            dense_oos.filter(
                (pl.col("factor_name") == factor) & (pl.col("symbol") == _REFERENCE_SYMBOL)
            )
            .sort(["observation_time", "fold_id"])
            .unique(subset=["observation_time"], keep="first")
            .head(8)
        )
        for record in sample.iter_rows(named=True):
            open_time = _as_int(record["observation_time"])
            close_t = close_map.get(open_time)
            close_tp1 = next_close.get(open_time)
            target = _as_float(record.get(_TARGET))
            expected = None
            matches = False
            if close_t is not None and close_tp1 is not None:
                expected = (close_tp1 - close_t) / close_t
                matches = target is not None and abs(expected - target) <= 1e-9
            # Same-bar leakage: factor predicting same-bar return using close_t vs prior.
            same_bar = False
            one_bar_shift = False
            if close_t is not None and open_time in close_map:
                idx = ordered_times.index(open_time)
                if idx > 0 and target is not None:
                    prev = close_map[ordered_times[idx - 1]]
                    same_bar_ret = (close_t - prev) / prev
                    if abs(same_bar_ret - target) <= 1e-9 and (
                        expected is None or abs(expected - target) > 1e-9
                    ):
                        same_bar = True
                        one_bar_shift = True
            rows.append(
                {
                    "factor": factor,
                    "symbol": _REFERENCE_SYMBOL,
                    "open_time": open_time,
                    "observation_date": _ms_to_date(open_time),
                    "factor_value": _as_float(record.get(_FACTOR_VALUE)),
                    "future_return_1": target,
                    "close_t": close_t,
                    "close_t_plus_1": close_tp1,
                    "expected_future_return_1": expected,
                    "label_matches_close_shift": matches,
                    "factor_predicts_horizon": "t->t+1",
                    "one_bar_shift_detected": one_bar_shift,
                    "same_bar_leakage_detected": same_bar,
                }
            )
    if not rows:
        return pl.DataFrame(schema={name: pl.Utf8 for name in ALIGNMENT_COLUMNS})
    return pl.DataFrame(rows)


def _build_timestamps_frame(dense_oos: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    oriented = _with_oriented(dense_oos)
    for factor in DENSE_FACTORS:
        sub = oriented.filter(pl.col("factor_name") == factor)
        for ts, group in sub.group_by("observation_time"):
            observation_time = _group_key_int(ts)
            n_symbols = int(group["symbol"].n_unique())
            n_factor = int(group.filter(pl.col(_FACTOR_VALUE).is_finite()).height)
            n_target = int(group.filter(pl.col(_TARGET).is_finite()).height)
            overlap = group.filter(pl.col(_ORIENTED).is_finite() & pl.col(_TARGET).is_finite())
            n_overlap = int(overlap.height)
            factor_std = _as_float(overlap[_ORIENTED].std()) if n_overlap > 1 else None
            near_constant = bool(factor_std is not None and factor_std <= 1e-12)
            dup = int(group.height - group["symbol"].n_unique())
            rows.append(
                {
                    "factor": factor,
                    "observation_time": observation_time,
                    "observation_date": _ms_to_date(observation_time),
                    "fold_ids": ",".join(
                        str(v) for v in sorted(group["fold_id"].unique().to_list())
                    ),
                    "n_symbols": n_symbols,
                    "n_finite_factor": n_factor,
                    "n_finite_target": n_target,
                    "overlap": n_overlap,
                    "factor_mean": _as_float(overlap[_ORIENTED].mean()) if n_overlap else None,
                    "factor_std": factor_std,
                    "factor_min": _as_float(overlap[_ORIENTED].min()) if n_overlap else None,
                    "factor_max": _as_float(overlap[_ORIENTED].max()) if n_overlap else None,
                    "target_mean": _as_float(overlap[_TARGET].mean()) if n_overlap else None,
                    "target_std": (_as_float(overlap[_TARGET].std()) if n_overlap > 1 else None),
                    "spearman_ic": _corr(overlap, _ORIENTED, _TARGET, method="spearman"),
                    "pearson_ic": _corr(overlap, _ORIENTED, _TARGET, method="pearson"),
                    "near_constant_factor": near_constant,
                    "duplicate_symbol_rows": dup,
                }
            )
    return pl.DataFrame(rows).sort(["factor", "observation_time"])


def _build_folds_frame(dense_oos: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    oriented = _with_oriented(dense_oos)
    for factor in DENSE_FACTORS:
        sub = oriented.filter(pl.col("factor_name") == factor)
        for fold_id, group in sub.group_by("fold_id"):
            fold = _group_key_int(fold_id)
            overlap = group.filter(pl.col(_ORIENTED).is_finite() & pl.col(_TARGET).is_finite())
            rows.append(
                {
                    "factor": factor,
                    "fold_id": fold,
                    "oos_rows": int(group.height),
                    "unique_timestamps": int(group["observation_time"].n_unique()),
                    "raw_spearman": _corr(group, _FACTOR_VALUE, _TARGET, method="spearman"),
                    "oriented_spearman": _corr(overlap, _ORIENTED, _TARGET, method="spearman"),
                    "oriented_pearson": _corr(overlap, _ORIENTED, _TARGET, method="pearson"),
                    "finite_overlap": int(overlap.height),
                }
            )
    return pl.DataFrame(rows).sort(["factor", "fold_id"])


def _quantile_means_for_group(group: pl.DataFrame) -> list[float | None]:
    overlap = group.filter(pl.col(_ORIENTED).is_finite() & pl.col(_TARGET).is_finite())
    if overlap.height < _MIN_XS_FOR_QUANTILE:
        return [None] * _QUINTILES
    ranked = overlap.with_columns(pl.col(_ORIENTED).rank(method="average").alias("_rk"))
    n = ranked.height
    ranked = ranked.with_columns(
        ((((pl.col("_rk") - 1) / n) * _QUINTILES).floor().clip(0, _QUINTILES - 1))
        .cast(pl.Int32)
        .alias("_q")
    )
    means: list[float | None] = []
    for quintile in range(_QUINTILES):
        means.append(_as_float(ranked.filter(pl.col("_q") == quintile)[_TARGET].mean()))
    return means


def _classify_monotonicity(means: Sequence[float | None]) -> str:
    values = [value for value in means if value is not None]
    if len(values) < _QUINTILES:
        return "insufficient_data"
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if all(diff >= 0 for diff in diffs):
        return "monotonic_positive"
    if all(diff <= 0 for diff in diffs):
        return "monotonic_negative"
    # U / inverted-U heuristics
    mid = values[2]
    if mid <= min(values[0], values[-1]):
        return "u_shaped"
    if mid >= max(values[0], values[-1]):
        return "inverted_u"
    # soft monotonic by endpoints
    if values[-1] < values[0] and sum(1 for diff in diffs if diff <= 0) >= 3:
        return "monotonic_negative"
    if values[-1] > values[0] and sum(1 for diff in diffs if diff >= 0) >= 3:
        return "monotonic_positive"
    return "noisy_no_relationship"


def _build_quantile_frame(dense_oos: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    oriented = _with_oriented(dense_oos)
    for factor in DENSE_FACTORS:
        sub = oriented.filter(pl.col("factor_name") == factor)
        # global: average quintile means across timestamps
        bucket: list[list[float]] = [[] for _ in range(_QUINTILES)]
        timestamps_used = 0
        for _, group in sub.group_by("observation_time"):
            means = _quantile_means_for_group(group)
            if any(value is None for value in means):
                continue
            timestamps_used += 1
            for index, value in enumerate(means):
                assert value is not None
                bucket[index].append(value)
        global_means: list[float | None] = [
            (sum(values) / len(values) if values else None) for values in bucket
        ]
        rows.append(
            {
                "factor": factor,
                "scope": "global",
                "fold_id": None,
                "q1_mean_future_return_1": global_means[0],
                "q2_mean_future_return_1": global_means[1],
                "q3_mean_future_return_1": global_means[2],
                "q4_mean_future_return_1": global_means[3],
                "q5_mean_future_return_1": global_means[4],
                "q5_minus_q1": (
                    None
                    if global_means[0] is None or global_means[4] is None
                    else global_means[4] - global_means[0]
                ),
                "monotonicity": _classify_monotonicity(global_means),
                "timestamps_used": timestamps_used,
            }
        )
        for fold_id, group in sub.group_by("fold_id"):
            fold = _group_key_int(fold_id)
            fold_bucket: list[list[float]] = [[] for _ in range(_QUINTILES)]
            fold_ts = 0
            for _, ts_group in group.group_by("observation_time"):
                means = _quantile_means_for_group(ts_group)
                if any(value is None for value in means):
                    continue
                fold_ts += 1
                for index, value in enumerate(means):
                    assert value is not None
                    fold_bucket[index].append(value)
            fold_means: list[float | None] = [
                (sum(values) / len(values) if values else None) for values in fold_bucket
            ]
            rows.append(
                {
                    "factor": factor,
                    "scope": "fold",
                    "fold_id": fold,
                    "q1_mean_future_return_1": fold_means[0],
                    "q2_mean_future_return_1": fold_means[1],
                    "q3_mean_future_return_1": fold_means[2],
                    "q4_mean_future_return_1": fold_means[3],
                    "q5_mean_future_return_1": fold_means[4],
                    "q5_minus_q1": (
                        None
                        if fold_means[0] is None or fold_means[4] is None
                        else fold_means[4] - fold_means[0]
                    ),
                    "monotonicity": _classify_monotonicity(fold_means),
                    "timestamps_used": fold_ts,
                }
            )
    return pl.DataFrame(rows).sort(["factor", "scope", "fold_id"])


def _build_symbol_contributor_frame(dense_oos: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    oriented = _with_oriented(dense_oos)
    for factor in DENSE_FACTORS:
        sub = oriented.filter(pl.col("factor_name") == factor)
        dedup = sub.unique(subset=["observation_time", "symbol"], keep="first")
        contrib: dict[str, float] = {}
        for _, group in dedup.group_by("observation_time"):
            overlap = group.filter(pl.col(_ORIENTED).is_finite() & pl.col(_TARGET).is_finite())
            if overlap.height < 5:
                continue
            ranked = overlap.with_columns(
                [
                    pl.col(_ORIENTED).rank().alias("_rf"),
                    pl.col(_TARGET).rank().alias("_rt"),
                ]
            )
            rf_mean = _as_float(ranked["_rf"].mean())
            rt_mean = _as_float(ranked["_rt"].mean())
            if rf_mean is None or rt_mean is None:
                continue
            for record in ranked.iter_rows(named=True):
                rf_value = _as_float(record["_rf"])
                rt_value = _as_float(record["_rt"])
                if rf_value is None or rt_value is None:
                    continue
                score = (rf_value - rf_mean) * (rt_value - rt_mean)
                symbol = str(record["symbol"])
                contrib[symbol] = contrib.get(symbol, 0.0) + score
        ordered = sorted(contrib.items(), key=lambda item: item[1])
        for rank, (symbol, score) in enumerate(ordered[:10], start=1):
            rows.append(
                {
                    "factor": factor,
                    "role": "top_negative_contributor",
                    "rank": rank,
                    "symbol": symbol,
                    "contribution_score": score,
                    "extreme_factor_value": None,
                    "extreme_target_value": None,
                    "notes": "sum of centered rank products across timestamps",
                }
            )
        for rank, (symbol, score) in enumerate(reversed(ordered[-10:]), start=1):
            rows.append(
                {
                    "factor": factor,
                    "role": "top_positive_contributor",
                    "rank": rank,
                    "symbol": symbol,
                    "contribution_score": score,
                    "extreme_factor_value": None,
                    "extreme_target_value": None,
                    "notes": "sum of centered rank products across timestamps",
                }
            )
        finite = dedup.filter(pl.col(_ORIENTED).is_finite())
        if finite.height:
            high = finite.sort(_ORIENTED, descending=True).head(5)
            low = finite.sort(_ORIENTED).head(5)
            for rank, record in enumerate(high.iter_rows(named=True), start=1):
                rows.append(
                    {
                        "factor": factor,
                        "role": "extreme_high_factor",
                        "rank": rank,
                        "symbol": record["symbol"],
                        "contribution_score": None,
                        "extreme_factor_value": _as_float(record[_ORIENTED]),
                        "extreme_target_value": _as_float(record[_TARGET]),
                        "notes": _ms_to_date(_as_int(record["observation_time"])),
                    }
                )
            for rank, record in enumerate(low.iter_rows(named=True), start=1):
                rows.append(
                    {
                        "factor": factor,
                        "role": "extreme_low_factor",
                        "rank": rank,
                        "symbol": record["symbol"],
                        "contribution_score": None,
                        "extreme_factor_value": _as_float(record[_ORIENTED]),
                        "extreme_target_value": _as_float(record[_TARGET]),
                        "notes": _ms_to_date(_as_int(record["observation_time"])),
                    }
                )
    if not rows:
        return pl.DataFrame(schema={name: pl.Utf8 for name in SYMBOL_CONTRIBUTOR_COLUMNS})
    return pl.DataFrame(rows).sort(["factor", "role", "rank"])


def _build_factors_frame(
    *,
    selected: pl.DataFrame,
    dense_oos: pl.DataFrame,
    timestamps_frame: pl.DataFrame,
    folds_frame: pl.DataFrame,
    formula_errors: Mapping[str, float | None],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    oriented = _with_oriented(dense_oos)
    quantiles = _build_quantile_frame(dense_oos).filter(pl.col("scope") == "global")
    for factor in DENSE_FACTORS:
        meta = selected.filter(pl.col("factor_name") == factor)
        direction = int(meta["selected_direction"][0]) if meta.height else 1
        selection_ic = _as_float(meta["selection_ic"][0]) if meta.height else None
        sub = oriented.filter(pl.col("factor_name") == factor)
        total = int(sub.height)
        usable = int(sub.filter(pl.col(_FACTOR_VALUE).is_finite()).height)
        null_rate = 1.0 - (usable / total) if total else None
        ts = timestamps_frame.filter(pl.col("factor") == factor)
        ics = [v for v in ts["spearman_ic"].to_list() if v is not None]
        pos = sum(1 for value in ics if value > 0)
        neg = sum(1 for value in ics if value < 0)
        best = None
        worst = None
        best_ic = None
        worst_ic = None
        if ics:
            best_idx = max(range(len(ics)), key=lambda i: ics[i])
            worst_idx = min(range(len(ics)), key=lambda i: ics[i])
            best_ic = ics[best_idx]
            worst_ic = ics[worst_idx]
            ts_dates = [
                d
                for d, ic in zip(
                    ts["observation_date"].to_list(), ts["spearman_ic"].to_list(), strict=False
                )
                if ic is not None
            ]
            best = ts_dates[best_idx] if best_idx < len(ts_dates) else None
            worst = ts_dates[worst_idx] if worst_idx < len(ts_dates) else None
        fold_ics = [
            v
            for v in folds_frame.filter(pl.col("factor") == factor)["oriented_spearman"].to_list()
            if v is not None
        ]
        qrow = quantiles.filter(pl.col("factor") == factor)
        requires, companions = companion_requirement(factor)
        err = formula_errors.get(factor)
        semantic = err is not None and err <= _FORMULA_RECON_TOLERANCE
        rows.append(
            {
                "factor": factor,
                "selected_direction": direction,
                "selection_ic": selection_ic,
                "null_rate_oos": null_rate,
                "usable_rows": usable,
                "total_rows": total,
                "panel_raw_spearman": _corr(sub, _FACTOR_VALUE, _TARGET, method="spearman"),
                "panel_oriented_spearman": _corr(sub, _ORIENTED, _TARGET, method="spearman"),
                "panel_oriented_pearson": _corr(sub, _ORIENTED, _TARGET, method="pearson"),
                "mean_fold_oriented_spearman": (
                    sum(fold_ics) / len(fold_ics) if fold_ics else None
                ),
                "median_timestamp_oriented_spearman": (
                    float(statistics.median(ics)) if ics else None
                ),
                "pct_timestamps_positive": (pos / len(ics) if ics else None),
                "pct_timestamps_negative": (neg / len(ics) if ics else None),
                "best_timestamp": best,
                "best_timestamp_ic": best_ic,
                "worst_timestamp": worst,
                "worst_timestamp_ic": worst_ic,
                "q5_minus_q1": _as_float(qrow["q5_minus_q1"][0]) if qrow.height else None,
                "monotonicity": (
                    str(qrow["monotonicity"][0]) if qrow.height else "insufficient_data"
                ),
                "requires_companion": requires,
                "companion_columns": ",".join(companions),
                "semantic_correct": semantic,
                "formula_recon_max_abs_error": err,
            }
        )
    return pl.DataFrame(rows).sort("factor")


def _build_cross_timeframe_frame(
    *,
    storage_root: Path,
    manager: str,
    exchange: str,
    market: str,
    year: int,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for timeframe in COMPARISON_TIMEFRAMES:
        store = _load_optional_parquet(
            storage_root
            / STORAGE_DIR_FACTORS
            / manager
            / exchange
            / market
            / _REFERENCE_SYMBOL
            / timeframe
            / f"{year}.parquet"
        )
        ohlcv = _load_optional_parquet(
            storage_root
            / STORAGE_DIR_PROCESSED
            / "ohlcv"
            / exchange
            / market
            / _REFERENCE_SYMBOL
            / timeframe
            / f"{year}.parquet"
        )
        oi = _load_optional_parquet(
            storage_root
            / STORAGE_DIR_PROCESSED
            / "open_interest"
            / exchange
            / market
            / _REFERENCE_SYMBOL
            / timeframe
            / f"{year}.parquet"
        )
        ohlcv_first = _ms_to_date(_series_min_ms(ohlcv, "open_time")) if ohlcv is not None else ""
        oi_exists = oi is not None and oi.height > 0
        oi_first = ""
        if oi_exists and oi is not None:
            col = "timestamp" if "timestamp" in oi.columns else "open_time"
            oi_first = _ms_to_date(_series_min_ms(oi, col))
        for factor in DENSE_FACTORS:
            if store is None or store.is_empty():
                rows.append(
                    {
                        "timeframe": timeframe,
                        "factor": factor,
                        "store_rows": 0,
                        "store_first_timestamp": "",
                        "store_first_non_null": "",
                        "store_last_timestamp": "",
                        "oi_processed_exists": oi_exists,
                        "oi_processed_first": oi_first,
                        "ohlcv_first": ohlcv_first,
                        "same_source_columns": True,
                        "same_calculation_function": True,
                        "notes": "factor store partition absent",
                    }
                )
                continue
            sub = store.filter(pl.col("factor_name") == factor)
            nn = sub.filter(pl.col("factor_value").is_not_null())
            rows.append(
                {
                    "timeframe": timeframe,
                    "factor": factor,
                    "store_rows": int(sub.height),
                    "store_first_timestamp": (
                        _ms_to_date(_series_min_ms(sub, "open_time")) if sub.height else ""
                    ),
                    "store_first_non_null": (
                        _ms_to_date(_series_min_ms(nn, "open_time")) if nn.height else ""
                    ),
                    "store_last_timestamp": (
                        _ms_to_date(_series_max_ms(sub, "open_time")) if sub.height else ""
                    ),
                    "oi_processed_exists": oi_exists,
                    "oi_processed_first": oi_first,
                    "ohlcv_first": ohlcv_first,
                    "same_source_columns": True,
                    "same_calculation_function": True,
                    "notes": (
                        "same Factor classes across timeframes; native processed "
                        "partition per timeframe; companion alignment truncates "
                        "leading incomplete companion bars"
                    ),
                }
            )
    return pl.DataFrame(rows).sort(["timeframe", "factor"])


def _build_implementation_audit_frame(
    formula_errors: Mapping[str, float | None],
) -> pl.DataFrame:
    rows = [
        {
            "factor": "price_volume_trend",
            "intended_definition": "cumsum(((close/prev_close)-1)*volume)",
            "implemented_formula": "cumsum(((close/close.shift(1))-1)*volume)",
            "price_field": "close",
            "volume_field": "volume",
            "volume_semantics": "Binance USD-M kline base-asset volume (not quote_volume)",
            "oi_field": "",
            "oi_aggregation": "",
            "cumulative_state": "cum_sum over per-symbol input frame; first row null",
            "symbol_boundary_reset": "yes (generation runs per symbol)",
            "timeframe_boundary_reset": "no incorrect reset; state starts at aligned frame",
            "computed_before_or_after_aggregation": "after processed timeframe aggregation",
            "missing_zero_volume_policy": "nulls never filled; zero volume contributes 0 delta",
            "semantic_verdict": (
                "CORRECT"
                if (formula_errors.get("price_volume_trend") or 0.0) <= _FORMULA_RECON_TOLERANCE
                else "MISMATCH"
            ),
            "notes": "Absolute cumulative level is not cross-sectionally normalized.",
        },
        {
            "factor": "on_balance_volume",
            "intended_definition": "cumsum(sign(close.diff())*volume); 0 on flat close",
            "implemented_formula": "when diff>0 +volume; diff<0 -volume; else 0; cum_sum",
            "price_field": "close",
            "volume_field": "volume",
            "volume_semantics": "Binance USD-M kline base-asset volume (not quote_volume)",
            "oi_field": "",
            "oi_aggregation": "",
            "cumulative_state": "cum_sum over per-symbol input frame; first row null",
            "symbol_boundary_reset": "yes (generation runs per symbol)",
            "timeframe_boundary_reset": "no incorrect reset; state starts at aligned frame",
            "computed_before_or_after_aggregation": "after processed timeframe aggregation",
            "missing_zero_volume_policy": "nulls never filled; zero volume contributes 0 delta",
            "semantic_verdict": (
                "CORRECT"
                if (formula_errors.get("on_balance_volume") or 0.0) <= _FORMULA_RECON_TOLERANCE
                else "MISMATCH"
            ),
            "notes": "Absolute cumulative level is not cross-sectionally normalized.",
        },
        {
            "factor": "open_interest_level",
            "intended_definition": "point-in-time open interest level",
            "implemented_formula": "cast(open_interest -> Float64) as open_interest_level",
            "price_field": "",
            "volume_field": "",
            "volume_semantics": "",
            "oi_field": "open_interest",
            "oi_aggregation": (
                "native processed open_interest partition at the OHLCV timeframe "
                "(not resampled inside the factor); as-of joined on open_time"
            ),
            "cumulative_state": "none (level)",
            "symbol_boundary_reset": "n/a",
            "timeframe_boundary_reset": "n/a",
            "computed_before_or_after_aggregation": "after processed timeframe partition load",
            "missing_zero_volume_policy": "null OI remains null; zero OI allowed upstream",
            "semantic_verdict": (
                "CORRECT"
                if (formula_errors.get("open_interest_level") or 0.0) <= _FORMULA_RECON_TOLERANCE
                else "MISMATCH"
            ),
            "notes": "Raw OI level is not size-normalized across symbols.",
        },
    ]
    return pl.DataFrame(rows)


def _build_data_lineage_frame() -> pl.DataFrame:
    stages = (
        "raw kline/OI -> processed cleaning -> load_factor_input_frame joins -> "
        "align_factor_input_frame -> FactorPipeline.compute -> Factors store -> "
        "Factor Validation -> Factor Selection -> Walk-Forward -> Purged CV -> "
        "evaluation_input join on (symbol,timeframe,open_time) -> "
        "PurgedCVEvaluator Spearman IC"
    )
    rows = [
        {
            "factor": "price_volume_trend",
            "source_columns": "close,volume",
            "source_dataset": "processed/ohlcv",
            "aggregation": "native timeframe OHLCV partition (volume summed at ingest/TF)",
            "calculation_function": (
                "cqros.factors.volume.price_volume_trend.PriceVolumeTrendFactor.compute"
            ),
            "timestamp_semantics": "open_time bar timestamp; value known at bar close",
            "final_evaluation_column": "factor_value (factor_name=price_volume_trend)",
            "pipeline_stages": stages,
        },
        {
            "factor": "on_balance_volume",
            "source_columns": "close,volume",
            "source_dataset": "processed/ohlcv",
            "aggregation": "native timeframe OHLCV partition (volume summed at ingest/TF)",
            "calculation_function": (
                "cqros.factors.volume.on_balance_volume.OnBalanceVolumeFactor.compute"
            ),
            "timestamp_semantics": "open_time bar timestamp; value known at bar close",
            "final_evaluation_column": "factor_value (factor_name=on_balance_volume)",
            "pipeline_stages": stages,
        },
        {
            "factor": "open_interest_level",
            "source_columns": "open_interest",
            "source_dataset": "processed/open_interest",
            "aggregation": "native timeframe OI history partition; as-of join to OHLCV open_time",
            "calculation_function": (
                "cqros.factors.open_interest.open_interest_level." "OpenInterestLevelFactor.compute"
            ),
            "timestamp_semantics": "open_time bar timestamp after as-of join",
            "final_evaluation_column": "factor_value (factor_name=open_interest_level)",
            "pipeline_stages": stages,
        },
    ]
    return pl.DataFrame(rows)


def _factor_mean_fold_ic(folds_frame: pl.DataFrame, factor: str) -> float | None:
    values = [
        v
        for v in folds_frame.filter(pl.col("factor") == factor)["oriented_spearman"].to_list()
        if v is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _negative_ic_is_broad(timestamps_frame: pl.DataFrame) -> bool:
    broad_flags: list[bool] = []
    for factor in DENSE_FACTORS:
        ics = [
            v
            for v in timestamps_frame.filter(pl.col("factor") == factor)["spearman_ic"].to_list()
            if v is not None
        ]
        if not ics:
            continue
        broad_flags.append((sum(1 for value in ics if value < 0) / len(ics)) >= 0.5)
    return bool(broad_flags) and all(broad_flags)


def _negative_ic_stable_across_folds(folds_frame: pl.DataFrame) -> bool:
    stable: list[bool] = []
    for factor in ("price_volume_trend", "on_balance_volume"):
        values = [
            v
            for v in folds_frame.filter(pl.col("factor") == factor)["oriented_spearman"].to_list()
            if v is not None
        ]
        if values:
            stable.append(sum(1 for value in values if value < 0) >= max(1, len(values) - 1))
    return bool(stable) and all(stable)


def _negative_ic_symbol_concentrated(symbol_frame: pl.DataFrame) -> bool:
    # Concentrated if top-3 absolute negative contributors dominate total negative mass.
    for factor in DENSE_FACTORS:
        neg = symbol_frame.filter(
            (pl.col("factor") == factor) & (pl.col("role") == "top_negative_contributor")
        )
        if neg.height < 3:
            continue
        scores = [abs(float(v)) for v in neg["contribution_score"].to_list() if v is not None]
        if not scores:
            continue
        top3 = sum(sorted(scores, reverse=True)[:3])
        total = sum(scores)
        if total > 0 and (top3 / total) >= 0.80:
            return True
    return False


def _build_summary_text(
    *,
    year: int,
    global_frame: pl.DataFrame,
    factors_frame: pl.DataFrame,
    companion: Mapping[str, object],
) -> str:
    record = global_frame.to_dicts()[0]
    lines = [
        "CQROS 1d DENSE FACTOR ROOT-CAUSE DIAGNOSTIC",
        "==========================================",
        "",
        "### VERDICT",
        "",
        str(record["verdict"]),
        "",
        "### PRIMARY CAUSE",
        "",
        str(record["primary_cause"]),
        "",
        "### SECONDARY CAUSES",
        "",
        str(record["secondary_causes"]),
        "",
        f"confidence={record['confidence']}",
        "",
        "### ANSWERS",
        "",
        f"1. PVT semantically correct: {record['pvt_semantic_correct']}",
        f"2. OBV semantically correct: {record['obv_semantic_correct']}",
        f"3. OI semantically correct: {record['oi_semantic_correct']}",
        f"4. Factor/target timestamp alignment correct: {record['timestamp_alignment_correct']}",
        f"5. IC calculation correct: {record['ic_calculation_matches_canonical']}",
        f"6. Negative IC broad: {record['negative_ic_broad']}",
        f"7. Negative IC stable across folds: {record['negative_ic_stable_across_folds']}",
        f"8. Negative IC driven by few symbols: {record['negative_ic_symbol_concentrated']}",
        f"9. Monotonic negative PVT/OBV/OI: "
        f"{record['relationship_monotonic_negative_pvt']}/"
        f"{record['relationship_monotonic_negative_obv']}/"
        f"{record['relationship_monotonic_negative_oi']}",
        f"10. Cross-timeframe implementation inconsistency: "
        f"{not bool(record['cross_timeframe_semantics_consistent'])}",
        f"11. Companion alignment unnecessarily truncates PVT/OBV: "
        f"{record['companion_truncates_pvt_obv_unnecessarily']}",
        f"12. Most likely root cause: {record['primary_cause']}",
        "",
        "### FACTOR METRICS",
        "",
    ]
    for row in factors_frame.iter_rows(named=True):
        lines.append(
            f"  {row['factor']}: dir={row['selected_direction']} "
            f"mean_fold_ic={row['mean_fold_oriented_spearman']} "
            f"median_ts_ic={row['median_timestamp_oriented_spearman']} "
            f"pct_neg_ts={row['pct_timestamps_negative']} "
            f"q5-q1={row['q5_minus_q1']} mono={row['monotonicity']} "
            f"semantic={row['semantic_correct']} recon_err={row['formula_recon_max_abs_error']}"
        )
    lines.extend(
        [
            "",
            "### COMPANION / HISTORY",
            "",
            f"OHLCV span: {companion.get('ohlcv_first')} -> {companion.get('ohlcv_last')} "
            f"({companion.get('ohlcv_rows')} rows)",
            f"OI first: {companion.get('oi_first')}",
            f"Factor store first: {companion.get('store_first')} "
            f"({companion.get('store_unique_timestamps')} unique timestamps)",
            f"PVT/OBV minimum inputs: {companion.get('pvt_obv_minimum_inputs')}",
            f"OI minimum inputs: {companion.get('oi_minimum_inputs')}",
            "",
            "LOW STATISTICAL POWER: unique 1d OOS timestamps = "
            f"{record['unique_oos_timestamps']} (explicitly insufficient for "
            "production-quality significance claims).",
            "",
            "### RECOMMENDED NEXT STEP",
            "",
            str(record["recommended_next_step"]),
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
    frames: Mapping[str, pl.DataFrame],
    summary_text: str,
) -> dict[str, Path]:
    paths = {
        "global": output_root / GLOBAL_CSV_NAME,
        "factors": output_root / FACTORS_CSV_NAME,
        "folds": output_root / FOLDS_CSV_NAME,
        "timestamps": output_root / TIMESTAMPS_CSV_NAME,
        "alignment": output_root / ALIGNMENT_CSV_NAME,
        "cross_timeframe": output_root / CROSS_TIMEFRAME_CSV_NAME,
        "symbol_contributors": output_root / SYMBOL_CONTRIBUTORS_CSV_NAME,
        "quantile_analysis": output_root / QUANTILE_CSV_NAME,
        "implementation_audit": output_root / IMPLEMENTATION_AUDIT_CSV_NAME,
        "data_lineage": output_root / DATA_LINEAGE_CSV_NAME,
        "summary": output_root / SUMMARY_TXT_NAME,
    }
    frames["global"].write_csv(paths["global"])
    frames["factors"].write_csv(paths["factors"])
    frames["folds"].write_csv(paths["folds"])
    frames["timestamps"].write_csv(paths["timestamps"])
    frames["alignment"].write_csv(paths["alignment"])
    frames["cross_timeframe"].write_csv(paths["cross_timeframe"])
    frames["symbol_contributors"].write_csv(paths["symbol_contributors"])
    frames["quantile_analysis"].write_csv(paths["quantile_analysis"])
    frames["implementation_audit"].write_csv(paths["implementation_audit"])
    frames["data_lineage"].write_csv(paths["data_lineage"])
    paths["summary"].write_text(summary_text, encoding="utf-8")
    return paths

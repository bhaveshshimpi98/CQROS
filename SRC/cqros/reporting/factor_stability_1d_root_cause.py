"""CQROS 1d Factor Stability root-cause investigation reporter.

Purpose:
    Provide a read-only root-cause investigation for the 1d timeframe
    ``MIXED_STABILITY`` / negative oriented OOS IC result using persisted
    Factor Selection, Walk-Forward, and Purged-CV artifacts only.

Responsibilities:
    - Discover the 1d selection panel and five Purged-CV folds
    - Compute factor-, fold-, distribution-, and cross-timeframe diagnostics
    - Classify a single primary root-cause category with secondary causes
    - Emit deterministic CSV / TXT reports under ``reports/factor_stability``
    - Hash production ledgers before/after and require immutability
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``ast``, ``hashlib``, ``logging``, ``math``, ``statistics``, ``polars``,
    ``cqros.core.constants``, and ``cqros.reporting.exceptions``.

Public API:
    Classification helpers, column/name constants,
    ``FactorStability1dRootCauseReporter``,
    ``FactorStability1dRootCauseResult``, and
    ``forbidden_import_violations``.

Notes:
    Orientation is inherited only from persisted selection / evaluation
    metadata. This reporter never mutates ledgers, never retunes thresholds,
    and never reconstructs TRAIN factor values from the Factors lake when
    TRAIN evaluation partitions are absent.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTOR_SELECTION,
    STORAGE_DIR_PURGED_CV,
    STORAGE_DIR_PURGED_CV_EVALUATION,
    STORAGE_DIR_WALK_FORWARD,
    STORAGE_DIR_WALK_FORWARD_EVALUATION,
)
from cqros.reporting.exceptions import ReportingValidationError

__all__ = [
    "COMPARISON_COLUMNS",
    "COMPARISON_TIMEFRAMES",
    "DEFAULT_OUTPUT_ROOT",
    "DISTRIBUTION_COLUMNS",
    "FACTOR_COLUMNS",
    "FOLD_COLUMNS",
    "GLOBAL_COLUMNS",
    "HASHES_AFTER_NAME",
    "HASHES_BEFORE_NAME",
    "PRIMARY_CROSS_SECTIONAL_BREADTH_LIMITATION",
    "PRIMARY_FACTOR_DEGENERATION",
    "PRIMARY_FACTOR_REDUNDANCY",
    "PRIMARY_GENUINE_TIMEFRAME_SIGNAL_WEAKNESS",
    "PRIMARY_INSUFFICIENT_EVIDENCE",
    "PRIMARY_SELECTION_INSTABILITY",
    "PRIMARY_STATISTICAL_POWER_LIMITATION",
    "PRIMARY_TARGET_DISTRIBUTION_PROBLEM",
    "PRIMARY_TIMEFRAME_ALIGNMENT_OR_HORIZON_PROBLEM",
    "PRIMARY_TRAIN_OOS_DEGRADATION",
    "ROOT_CAUSE_COMPARISON_CSV_NAME",
    "ROOT_CAUSE_DISTRIBUTION_CSV_NAME",
    "ROOT_CAUSE_FACTORS_CSV_NAME",
    "ROOT_CAUSE_FOLDS_CSV_NAME",
    "ROOT_CAUSE_GLOBAL_CSV_NAME",
    "ROOT_CAUSE_SUMMARY_TXT_NAME",
    "TARGET_TIMEFRAME",
    "FactorStability1dRootCauseReporter",
    "FactorStability1dRootCauseResult",
    "classify_primary_root_cause",
    "forbidden_import_violations",
    "hash_watched_production_artifacts",
]

_logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT: Final[Path] = Path("reports") / "factor_stability"
TARGET_TIMEFRAME: Final[str] = "1d"
COMPARISON_TIMEFRAMES: Final[tuple[str, ...]] = ("5m", "15m", "1h", "4h", "1d")

ROOT_CAUSE_GLOBAL_CSV_NAME: Final[str] = "1d_root_cause_global.csv"
ROOT_CAUSE_FOLDS_CSV_NAME: Final[str] = "1d_root_cause_folds.csv"
ROOT_CAUSE_FACTORS_CSV_NAME: Final[str] = "1d_root_cause_factors.csv"
ROOT_CAUSE_DISTRIBUTION_CSV_NAME: Final[str] = "1d_root_cause_distribution.csv"
ROOT_CAUSE_COMPARISON_CSV_NAME: Final[str] = "1d_root_cause_comparison.csv"
ROOT_CAUSE_SUMMARY_TXT_NAME: Final[str] = "1d_root_cause_summary.txt"
HASHES_BEFORE_NAME: Final[str] = "1d_root_cause_hashes_before.txt"
HASHES_AFTER_NAME: Final[str] = "1d_root_cause_hashes_after.txt"

PRIMARY_STATISTICAL_POWER_LIMITATION: Final[str] = "A. STATISTICAL_POWER_LIMITATION"
PRIMARY_CROSS_SECTIONAL_BREADTH_LIMITATION: Final[str] = "B. CROSS_SECTIONAL_BREADTH_LIMITATION"
PRIMARY_TARGET_DISTRIBUTION_PROBLEM: Final[str] = "C. TARGET_DISTRIBUTION_PROBLEM"
PRIMARY_FACTOR_DEGENERATION: Final[str] = "D. FACTOR_DEGENERATION"
PRIMARY_FACTOR_REDUNDANCY: Final[str] = "E. FACTOR_REDUNDANCY"
PRIMARY_SELECTION_INSTABILITY: Final[str] = "F. SELECTION_INSTABILITY"
PRIMARY_TRAIN_OOS_DEGRADATION: Final[str] = "G. TRAIN_OOS_DEGRADATION"
PRIMARY_TIMEFRAME_ALIGNMENT_OR_HORIZON_PROBLEM: Final[str] = (
    "H. TIMEFRAME_ALIGNMENT_OR_HORIZON_PROBLEM"
)
PRIMARY_GENUINE_TIMEFRAME_SIGNAL_WEAKNESS: Final[str] = "I. GENUINE_TIMEFRAME_SIGNAL_WEAKNESS"
PRIMARY_INSUFFICIENT_EVIDENCE: Final[str] = "J. INSUFFICIENT_EVIDENCE"

_FACTOR_VALUE: Final[str] = "factor_value"
_TARGET: Final[str] = "future_return_1"
_ORIENTED_ALIAS: Final[str] = "_oriented_factor_value"
_PARTITION_OOS: Final[str] = "OOS"
_PARTITION_TRAIN: Final[str] = "TRAIN"

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

_REDUNDANCY_SPEARMAN_THRESHOLD: Final[float] = 0.90
_DEGENERATE_NULL_RATE: Final[float] = 0.80
_LOW_VARIANCE_UNIQUE: Final[int] = 2
_BOOTSTRAP_REPLICATES: Final[int] = 200
_BOOTSTRAP_SEED: Final[int] = 42

_ERROR_LEDGER_MUTATION: Final[str] = "REPORT-1D-ROOT-CAUSE-001"
_ERROR_MANAGER: Final[str] = "REPORT-1D-ROOT-CAUSE-002"
_ERROR_OUTPUT: Final[str] = "REPORT-1D-ROOT-CAUSE-003"
_ERROR_MISSING_1D: Final[str] = "REPORT-1D-ROOT-CAUSE-004"

GLOBAL_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "selected_factors",
    "tested_factors",
    "fold_count",
    "is_ic",
    "oos_raw_ic",
    "oos_oriented_ic",
    "pcv_raw_ic",
    "pcv_oriented_ic",
    "positive_oos_factor_count",
    "negative_oos_factor_count",
    "degenerate_factor_count",
    "high_missingness_factor_count",
    "median_null_rate",
    "unique_oos_timestamps",
    "median_cross_section",
    "min_cross_section",
    "max_cross_section",
    "oos_rows_per_fold_median",
    "ic_standard_error",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_ci_width",
    "target_std",
    "target_positive_rate",
    "redundant_group_count",
    "redundancy_status",
    "effective_independent_factors",
    "selection_ratio",
    "degradation_mean",
    "degradation_median",
    "train_positive_oos_negative_count",
    "negative_fold_count",
    "fold_stability",
    "factor_stability",
    "cross_sectional_breadth",
    "statistical_power",
    "target_distribution",
    "factor_degeneracy",
    "factor_redundancy",
    "selection_stability",
    "train_oos_degradation",
    "timeframe_alignment",
    "primary_classification",
    "secondary_causes",
    "evidence_confidence",
    "leakage",
    "production_artifacts_unchanged",
    "deterministic",
)

FOLD_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "fold_id",
    "train_rows",
    "oos_rows",
    "raw_oos_mean_return",
    "raw_oos_ic",
    "oriented_oos_ic",
    "positive_factor_count",
    "negative_factor_count",
    "median_factor_ic",
    "dispersion_factor_ic",
    "factors_positive_oos_ic",
    "factors_negative_oos_ic",
    "unique_timestamps",
    "median_cross_section",
)

# Explicit dtypes avoid Polars dict-inference failures when early folds have
# null numeric fields and later folds introduce floats (regenerated corpora).
FOLD_SCHEMA: Final[dict[str, pl.DataType]] = {
    "timeframe": pl.String,
    "year": pl.Int64,
    "fold_id": pl.Int64,
    "train_rows": pl.Int64,
    "oos_rows": pl.Int64,
    "raw_oos_mean_return": pl.Float64,
    "raw_oos_ic": pl.Float64,
    "oriented_oos_ic": pl.Float64,
    "positive_factor_count": pl.Int64,
    "negative_factor_count": pl.Int64,
    "median_factor_ic": pl.Float64,
    "dispersion_factor_ic": pl.Float64,
    "factors_positive_oos_ic": pl.Int64,
    "factors_negative_oos_ic": pl.Int64,
    "unique_timestamps": pl.Int64,
    "median_cross_section": pl.Float64,
}

FACTOR_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "factor_name",
    "factor_version",
    "factor_family",
    "selection_rank",
    "selection_strength",
    "orientation_direction",
    "orientation_policy",
    "raw_is_ic",
    "oriented_is_ic",
    "raw_oos_ic_by_fold",
    "oriented_oos_ic_by_fold",
    "mean_raw_oos_ic",
    "mean_oriented_oos_ic",
    "median_oriented_oos_ic",
    "std_oriented_oos_ic",
    "min_oriented_oos_ic",
    "max_oriented_oos_ic",
    "positive_oos_folds",
    "negative_oos_folds",
    "fraction_positive_oos_folds",
    "oriented_oos_minus_is",
    "null_rate",
    "variance",
    "unique_values",
    "cross_sectional_dispersion_median",
    "degenerate_flag",
    "low_variance_flag",
    "high_missingness_flag",
)

DISTRIBUTION_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "metric_group",
    "metric_name",
    "metric_value",
    "fold_id",
)

COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "selected_factors",
    "tested_factors",
    "selection_ratio",
    "wf_raw_oos_ic",
    "wf_oriented_oos_ic",
    "pcv_raw_oos_ic",
    "pcv_oriented_oos_ic",
    "unique_oos_timestamps",
    "median_cross_section",
    "min_cross_section",
    "max_cross_section",
    "oos_observation_rows",
    "target_mean",
    "target_std",
    "target_positive_rate",
    "median_factor_null_rate",
    "degenerate_factor_count",
    "ic_standard_error",
    "bootstrap_ci_width",
)


@dataclass(frozen=True, slots=True)
class FactorStability1dRootCauseResult:
    """Immutable result of a 1d root-cause investigation."""

    global_frame: pl.DataFrame
    fold_frame: pl.DataFrame
    factor_frame: pl.DataFrame
    distribution_frame: pl.DataFrame
    comparison_frame: pl.DataFrame
    summary_text: str
    paths: Mapping[str, Path]
    primary_classification: str
    secondary_causes: tuple[str, ...]
    production_artifacts_unchanged: bool
    deterministic: bool
    hashes_before: Mapping[str, str]
    hashes_after: Mapping[str, str]


class FactorStability1dRootCauseReporter:
    """Read-only 1d root-cause investigation reporter.

    Args:
        storage_root: Lake root containing selection / evaluation tiers.
        output_root: Directory receiving deterministic report artifacts.
        manager: Order-manager identity used for partition discovery.
        exchange: Exchange partition label.
        market: Market partition label.
        engine: Engine label used when filtering evaluation rows if present.
        logger: Injected logger.
    """

    __slots__ = (
        "_storage_root",
        "_output_root",
        "_manager",
        "_exchange",
        "_market",
        "_engine",
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
        engine: str = "simple",
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
        self._engine = engine
        self._logger = logger if logger is not None else _logger

    @property
    def output_root(self) -> Path:
        """Return the configured report output directory."""
        return self._output_root

    def run(self, *, year: int | None = None) -> FactorStability1dRootCauseResult:
        """Execute the 1d root-cause investigation and write reports.

        Args:
            year: Optional year filter. ``None`` uses the latest discovered
                1d Factor Selection year.

        Returns:
            Immutable investigation result.

        Raises:
            ReportingValidationError: When 1d artifacts are missing or
                production artifacts mutate during the run.
        """
        if self._output_root.exists() and not self._output_root.is_dir():
            raise ReportingValidationError(
                "output path must be a directory",
                error_code=_ERROR_OUTPUT,
                details={"output": str(self._output_root)},
            )

        self._output_root.mkdir(parents=True, exist_ok=True)
        hashes_before = hash_watched_production_artifacts(self._storage_root)
        _write_hash_manifest(
            self._output_root / HASHES_BEFORE_NAME,
            hashes_before,
        )

        selection_path, panel_year = _resolve_1d_selection_partition(
            self._storage_root,
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            year=year,
        )
        self._logger.info(
            "1d root-cause investigation starting manager=%s year=%s",
            self._manager,
            panel_year,
        )

        panel = self._compute_panel(selection_path=selection_path, year=panel_year)

        paths = _write_report_bundle(
            output_root=self._output_root,
            global_frame=panel["global"],  # type: ignore[arg-type]
            fold_frame=panel["fold"],  # type: ignore[arg-type]
            factor_frame=panel["factor"],  # type: ignore[arg-type]
            distribution_frame=panel["distribution"],  # type: ignore[arg-type]
            comparison_frame=panel["comparison"],  # type: ignore[arg-type]
            summary_text=str(panel["summary_text"]),
        )

        hashes_after = hash_watched_production_artifacts(self._storage_root)
        _write_hash_manifest(self._output_root / HASHES_AFTER_NAME, hashes_after)
        unchanged = hashes_before == hashes_after
        if not unchanged:
            raise ReportingValidationError(
                "production artifacts changed during 1d root-cause investigation",
                error_code=_ERROR_LEDGER_MUTATION,
                details={
                    "before_count": len(hashes_before),
                    "after_count": len(hashes_after),
                },
            )

        global_frame = panel["global"]
        assert isinstance(global_frame, pl.DataFrame)
        fold_frame = panel["fold"]
        assert isinstance(fold_frame, pl.DataFrame)
        factor_frame = panel["factor"]
        assert isinstance(factor_frame, pl.DataFrame)
        distribution_frame = panel["distribution"]
        assert isinstance(distribution_frame, pl.DataFrame)
        comparison_frame = panel["comparison"]
        assert isinstance(comparison_frame, pl.DataFrame)

        global_row = global_frame.to_dicts()[0] if global_frame.height else {}
        primary = str(global_row.get("primary_classification", PRIMARY_INSUFFICIENT_EVIDENCE))
        secondary_raw = str(global_row.get("secondary_causes") or "")
        secondary = tuple(part for part in secondary_raw.split("|") if part)

        global_frame = global_frame.with_columns(
            [
                pl.lit(True).alias("production_artifacts_unchanged"),
                pl.lit(True).alias("deterministic"),
            ]
        )
        _write_csv(global_frame, paths["global"])
        summary_text = _render_summary(
            global_row={
                **global_row,
                "production_artifacts_unchanged": True,
                "deterministic": True,
            },
            fold_frame=fold_frame,
            factor_frame=factor_frame,
        )
        paths["summary"].write_text(summary_text, encoding="utf-8", newline="\n")

        self._logger.info(
            "1d root-cause investigation complete primary=%s unchanged=%s",
            primary,
            unchanged,
        )
        return FactorStability1dRootCauseResult(
            global_frame=global_frame,
            fold_frame=fold_frame,
            factor_frame=factor_frame,
            distribution_frame=distribution_frame,
            comparison_frame=comparison_frame,
            summary_text=summary_text,
            paths=paths,
            primary_classification=primary,
            secondary_causes=secondary,
            production_artifacts_unchanged=unchanged,
            deterministic=True,
            hashes_before=hashes_before,
            hashes_after=hashes_after,
        )

    def _compute_panel(self, *, selection_path: Path, year: int) -> dict[str, object]:
        selection = pl.read_parquet(selection_path)
        if "selected" in selection.columns:
            selected = selection.filter(pl.col("selected"))
        else:
            selected = selection
        tested_factors = int(selection.height)
        selected_factors = int(selected.height)

        wf_obs = _load_optional_parquet(
            _evaluation_path(
                self._storage_root,
                STORAGE_DIR_WALK_FORWARD_EVALUATION,
                manager=self._manager,
                exchange=self._exchange,
                market=self._market,
                timeframe=TARGET_TIMEFRAME,
                year=year,
            )
        )
        pcv_obs = _load_optional_parquet(
            _evaluation_path(
                self._storage_root,
                STORAGE_DIR_PURGED_CV_EVALUATION,
                manager=self._manager,
                exchange=self._exchange,
                market=self._market,
                timeframe=TARGET_TIMEFRAME,
                year=year,
            )
        )
        pcv_ledger = _load_optional_parquet(
            _evaluation_path(
                self._storage_root,
                STORAGE_DIR_PURGED_CV,
                manager=self._manager,
                exchange=self._exchange,
                market=self._market,
                timeframe=TARGET_TIMEFRAME,
                year=year,
            )
        )
        wf_ledger = _load_optional_parquet(
            _evaluation_path(
                self._storage_root,
                STORAGE_DIR_WALK_FORWARD,
                manager=self._manager,
                exchange=self._exchange,
                market=self._market,
                timeframe=TARGET_TIMEFRAME,
                year=year,
            )
        )

        primary_obs = pcv_obs if pcv_obs is not None and pcv_obs.height > 0 else wf_obs
        # Large panels skip materializing purged-CV evaluation observations.
        # Fall back to Walk-Forward evaluation rows remapped onto PCV fold
        # time bounds so fold_id means the five CV folds, not WF windows.
        if (
            (pcv_obs is None or pcv_obs.height == 0)
            and pcv_ledger is not None
            and pcv_ledger.height > 0
            and wf_obs is not None
            and wf_obs.height > 0
        ):
            primary_obs = _observations_aligned_to_pcv_folds(wf_obs, pcv_ledger)
        factor_fold = _compute_factor_fold_ics(primary_obs, engine=self._engine)
        wf_factor_fold = _compute_factor_fold_ics(wf_obs, engine=self._engine)
        pcv_factor_fold = _compute_factor_fold_ics(
            primary_obs if (pcv_obs is None or pcv_obs.height == 0) else pcv_obs,
            engine=self._engine,
        )

        factor_frame = _build_factor_frame(
            selected=selected,
            factor_fold=factor_fold,
            observations=primary_obs,
            engine=self._engine,
            timeframe=TARGET_TIMEFRAME,
            year=year,
        )
        fold_frame = _build_fold_frame(
            factor_fold=factor_fold,
            observations=primary_obs,
            ledger=pcv_ledger if pcv_ledger is not None else wf_ledger,
            engine=self._engine,
            timeframe=TARGET_TIMEFRAME,
            year=year,
        )
        distribution_frame = _build_distribution_frame(
            observations=primary_obs,
            engine=self._engine,
            timeframe=TARGET_TIMEFRAME,
            year=year,
        )
        comparison_frame = _build_comparison_frame(
            storage_root=self._storage_root,
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            engine=self._engine,
            year=year,
        )
        redundancy = _analyze_redundancy(primary_obs, selected=selected, engine=self._engine)
        alignment = _assess_timeframe_alignment(
            observations=primary_obs,
            wf_ledger=wf_ledger,
            comparison_obs=_load_optional_parquet(
                _evaluation_path(
                    self._storage_root,
                    STORAGE_DIR_WALK_FORWARD_EVALUATION,
                    manager=self._manager,
                    exchange=self._exchange,
                    market=self._market,
                    timeframe="4h",
                    year=year,
                )
            ),
            engine=self._engine,
        )
        global_frame = _build_global_frame(
            selected=selected,
            selection=selection,
            factor_frame=factor_frame,
            fold_frame=fold_frame,
            distribution_frame=distribution_frame,
            comparison_frame=comparison_frame,
            wf_factor_fold=wf_factor_fold,
            pcv_factor_fold=pcv_factor_fold,
            observations=primary_obs,
            redundancy=redundancy,
            alignment=alignment,
            engine=self._engine,
            timeframe=TARGET_TIMEFRAME,
            year=year,
            tested_factors=tested_factors,
            selected_factors=selected_factors,
        )
        summary_text = _render_summary(
            global_row=global_frame.to_dicts()[0] if global_frame.height else {},
            fold_frame=fold_frame,
            factor_frame=factor_frame,
        )
        return {
            "global": _stabilize_floats(global_frame),
            "fold": _stabilize_floats(fold_frame),
            "factor": _stabilize_floats(factor_frame),
            "distribution": _stabilize_floats(distribution_frame),
            "comparison": _stabilize_floats(comparison_frame),
            "summary_text": summary_text,
        }


def classify_primary_root_cause(
    *,
    selected_factors: int,
    fold_count: int,
    degenerate_factor_count: int,
    high_missingness_factor_count: int,
    median_null_rate: float | None,
    unique_oos_timestamps: int | None,
    comparison_median_timestamps: float | None,
    median_cross_section: float | None,
    comparison_median_cross_section: float | None,
    target_std: float | None,
    comparison_median_target_std: float | None,
    redundant_group_count: int,
    redundancy_status: str,
    selection_ratio: float | None,
    comparison_selection_ratio: float | None,
    train_positive_oos_negative_count: int,
    degradation_median: float | None,
    negative_fold_count: int,
    oos_oriented_ic: float | None,
    bootstrap_ci_low: float | None,
    bootstrap_ci_high: float | None,
    alignment_issue: bool,
) -> tuple[str, tuple[str, ...], str]:
    """Classify one primary root cause and ordered secondary causes.

    Returns:
        ``(primary, secondary_causes, evidence_confidence)``.
    """
    if selected_factors <= 0 or fold_count <= 0:
        return PRIMARY_INSUFFICIENT_EVIDENCE, (), "LOW"

    secondaries: list[str] = []

    degenerate_ratio = (
        float(degenerate_factor_count) / float(selected_factors) if selected_factors else 0.0
    )
    high_miss_ratio = (
        float(high_missingness_factor_count) / float(selected_factors) if selected_factors else 0.0
    )
    power_limited = (
        unique_oos_timestamps is not None
        and comparison_median_timestamps is not None
        and comparison_median_timestamps > 0
        and unique_oos_timestamps < (0.25 * comparison_median_timestamps)
    )
    breadth_limited = (
        median_cross_section is not None
        and comparison_median_cross_section is not None
        and comparison_median_cross_section > 0
        and median_cross_section < (0.5 * comparison_median_cross_section)
    )
    target_problem = (
        target_std is not None
        and comparison_median_target_std is not None
        and comparison_median_target_std > 0
        and target_std > (5.0 * comparison_median_target_std)
    )
    selection_aggressive = (
        selection_ratio is not None
        and comparison_selection_ratio is not None
        and selection_ratio > (1.5 * comparison_selection_ratio)
    )
    degradation_dominant = (
        selected_factors > 0 and train_positive_oos_negative_count > (selected_factors / 2.0)
    ) or (
        degradation_median is not None
        and math.isfinite(degradation_median)
        and degradation_median < -0.02
    )
    ci_includes_zero = (
        bootstrap_ci_low is not None
        and bootstrap_ci_high is not None
        and bootstrap_ci_low <= 0.0 <= bootstrap_ci_high
    )
    persistent_negative = (
        oos_oriented_ic is not None
        and oos_oriented_ic < 0.0
        and fold_count > 0
        and negative_fold_count > (fold_count / 2.0)
    )

    if alignment_issue:
        secondaries.append(PRIMARY_TIMEFRAME_ALIGNMENT_OR_HORIZON_PROBLEM)
    if (
        degenerate_ratio >= 0.25
        or high_miss_ratio >= 0.4
        or (median_null_rate is not None and median_null_rate >= _DEGENERATE_NULL_RATE)
    ):
        secondaries.append(PRIMARY_FACTOR_DEGENERATION)
    if power_limited or ci_includes_zero:
        secondaries.append(PRIMARY_STATISTICAL_POWER_LIMITATION)
    if breadth_limited:
        secondaries.append(PRIMARY_CROSS_SECTIONAL_BREADTH_LIMITATION)
    if target_problem:
        secondaries.append(PRIMARY_TARGET_DISTRIBUTION_PROBLEM)
    if redundancy_status != "REDUNDANCY_ANALYSIS_UNAVAILABLE" and redundant_group_count > 0:
        secondaries.append(PRIMARY_FACTOR_REDUNDANCY)
    if selection_aggressive:
        secondaries.append(PRIMARY_SELECTION_INSTABILITY)
    if degradation_dominant:
        secondaries.append(PRIMARY_TRAIN_OOS_DEGRADATION)
    if persistent_negative:
        secondaries.append(PRIMARY_GENUINE_TIMEFRAME_SIGNAL_WEAKNESS)

    # Primary preference: concrete panel defects before residual weakness.
    if degenerate_ratio >= 0.25 or (
        median_null_rate is not None and median_null_rate >= _DEGENERATE_NULL_RATE
    ):
        primary = PRIMARY_FACTOR_DEGENERATION
        confidence = "HIGH"
    elif alignment_issue and degenerate_ratio >= 0.1:
        primary = PRIMARY_TIMEFRAME_ALIGNMENT_OR_HORIZON_PROBLEM
        confidence = "MEDIUM"
    elif power_limited and ci_includes_zero:
        primary = PRIMARY_STATISTICAL_POWER_LIMITATION
        confidence = "HIGH"
    elif breadth_limited:
        primary = PRIMARY_CROSS_SECTIONAL_BREADTH_LIMITATION
        confidence = "MEDIUM"
    elif target_problem:
        primary = PRIMARY_TARGET_DISTRIBUTION_PROBLEM
        confidence = "MEDIUM"
    elif degradation_dominant:
        primary = PRIMARY_TRAIN_OOS_DEGRADATION
        confidence = "HIGH"
    elif redundancy_status != "REDUNDANCY_ANALYSIS_UNAVAILABLE" and redundant_group_count >= max(
        2, selected_factors // 4
    ):
        primary = PRIMARY_FACTOR_REDUNDANCY
        confidence = "MEDIUM"
    elif selection_aggressive:
        primary = PRIMARY_SELECTION_INSTABILITY
        confidence = "MEDIUM"
    elif persistent_negative:
        primary = PRIMARY_GENUINE_TIMEFRAME_SIGNAL_WEAKNESS
        confidence = "HIGH"
    elif power_limited:
        primary = PRIMARY_STATISTICAL_POWER_LIMITATION
        confidence = "MEDIUM"
    else:
        primary = PRIMARY_INSUFFICIENT_EVIDENCE
        confidence = "LOW"

    secondary = tuple(cause for cause in secondaries if cause != primary)
    return primary, secondary, confidence


def forbidden_import_violations(source_path: Path) -> tuple[str, ...]:
    """Return forbidden import module names found via AST inspection."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
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
) -> Path:
    return Path(storage_root) / tier / manager / exchange / market / timeframe / f"{year}.parquet"


def _compute_factor_fold_ics(observations: pl.DataFrame | None, *, engine: str) -> pl.DataFrame:
    schema = {
        "factor_name": pl.String,
        "factor_version": pl.String,
        "fold_id": pl.Int32,
        "raw_oos_ic": pl.Float64,
        "oriented_oos_ic": pl.Float64,
        "selected_direction": pl.Int32,
        "selection_ic": pl.Float64,
        "orientation_policy": pl.String,
        "oos_rows": pl.Int64,
    }
    working = _filter_oos_selected(observations, engine=engine)
    required = {_FACTOR_VALUE, _TARGET, "factor_name", "factor_version", "fold_id"}
    if working is None or working.height == 0 or not required.issubset(working.columns):
        return pl.DataFrame(schema=schema)
    working = _ensure_orientation_columns(working)
    oriented = working.with_columns(
        (pl.col(_FACTOR_VALUE) * pl.col("selected_direction").cast(pl.Float64)).alias(
            _ORIENTED_ALIAS
        )
    )
    aggregated = (
        oriented.group_by(["factor_name", "factor_version", "fold_id"], maintain_order=True)
        .agg(
            [
                pl.len().alias("oos_rows"),
                pl.corr(_FACTOR_VALUE, _TARGET, method="spearman").alias("raw_oos_ic"),
                pl.corr(_ORIENTED_ALIAS, _TARGET, method="spearman").alias("oriented_oos_ic"),
                pl.col("selected_direction").first().alias("selected_direction"),
                pl.col("selection_ic").first().alias("selection_ic"),
                pl.col("orientation_policy").first().alias("orientation_policy"),
            ]
        )
        .sort(["factor_name", "factor_version", "fold_id"])
        .select(list(schema))
    )
    return _nan_to_null(aggregated, ("raw_oos_ic", "oriented_oos_ic", "selection_ic"))


def _build_factor_frame(
    *,
    selected: pl.DataFrame,
    factor_fold: pl.DataFrame,
    observations: pl.DataFrame | None,
    engine: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    if selected.height == 0:
        return _empty(FACTOR_COLUMNS)
    oos = _filter_oos_selected(observations, engine=engine)
    rows: list[dict[str, object]] = []
    selected_sorted = selected.sort(
        ["selection_rank", "factor_name", "factor_version"]
        if "selection_rank" in selected.columns
        else ["factor_name", "factor_version"]
    )
    for item in selected_sorted.iter_rows(named=True):
        factor_name = str(item["factor_name"])
        factor_version = str(item["factor_version"])
        direction = _as_int(item.get("selected_direction"))
        raw_is = _as_float(item.get("selection_ic"))
        oriented_is = None
        if raw_is is not None and direction is not None:
            oriented_is = raw_is * float(direction)
        folds = factor_fold.filter(
            (pl.col("factor_name") == factor_name) & (pl.col("factor_version") == factor_version)
        ).sort("fold_id")
        raw_by_fold = [_as_float(value) for value in folds["raw_oos_ic"].to_list()]
        oriented_by_fold = [_as_float(value) for value in folds["oriented_oos_ic"].to_list()]
        finite_oriented = [
            value for value in oriented_by_fold if value is not None and math.isfinite(value)
        ]
        mean_raw = _mean(raw_by_fold)
        mean_oriented = _mean(oriented_by_fold)
        median_oriented = statistics.median(finite_oriented) if finite_oriented else None
        std_oriented = _std(finite_oriented)
        positive = sum(1 for value in finite_oriented if value > 0.0)
        negative = sum(1 for value in finite_oriented if value < 0.0)
        total_signed = positive + negative
        fraction_positive = (positive / total_signed) if total_signed > 0 else None
        degradation = None
        if mean_oriented is not None and oriented_is is not None:
            degradation = mean_oriented - oriented_is

        factor_obs = (
            oos.filter(
                (pl.col("factor_name") == factor_name)
                & (pl.col("factor_version") == factor_version)
            )
            if oos is not None
            else None
        )
        null_rate, variance, unique_values, cs_dispersion = _factor_value_stats(factor_obs)
        degenerate = bool(null_rate is not None and null_rate >= 1.0 - 1e-12) or bool(
            unique_values is not None and unique_values <= 1
        )
        low_variance = bool(
            unique_values is not None and unique_values <= _LOW_VARIANCE_UNIQUE
        ) or bool(variance is not None and variance <= 0.0)
        high_missing = bool(null_rate is not None and null_rate >= _DEGENERATE_NULL_RATE)
        selection_strength = _as_float(item.get("selection_score"))
        if selection_strength is None:
            selection_strength = abs(raw_is) if raw_is is not None else None
        rows.append(
            {
                "timeframe": timeframe,
                "year": year,
                "factor_name": factor_name,
                "factor_version": factor_version,
                "factor_family": item.get("factor_category"),
                "selection_rank": _as_int(item.get("selection_rank")),
                "selection_strength": selection_strength,
                "orientation_direction": direction,
                "orientation_policy": item.get("orientation_policy"),
                "raw_is_ic": raw_is,
                "oriented_is_ic": oriented_is,
                "raw_oos_ic_by_fold": _format_pipe_list(raw_by_fold),
                "oriented_oos_ic_by_fold": _format_pipe_list(oriented_by_fold),
                "mean_raw_oos_ic": mean_raw,
                "mean_oriented_oos_ic": mean_oriented,
                "median_oriented_oos_ic": median_oriented,
                "std_oriented_oos_ic": std_oriented,
                "min_oriented_oos_ic": min(finite_oriented) if finite_oriented else None,
                "max_oriented_oos_ic": max(finite_oriented) if finite_oriented else None,
                "positive_oos_folds": positive,
                "negative_oos_folds": negative,
                "fraction_positive_oos_folds": fraction_positive,
                "oriented_oos_minus_is": degradation,
                "null_rate": null_rate,
                "variance": variance,
                "unique_values": unique_values,
                "cross_sectional_dispersion_median": cs_dispersion,
                "degenerate_flag": degenerate,
                "low_variance_flag": low_variance,
                "high_missingness_flag": high_missing,
            }
        )
    return _sort_frame(pl.DataFrame(rows), FACTOR_COLUMNS)


def _build_fold_frame(
    *,
    factor_fold: pl.DataFrame,
    observations: pl.DataFrame | None,
    ledger: pl.DataFrame | None,
    engine: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    fold_ids: list[int] = []
    # Prefer canonical CV ledger folds when present. Walk-forward evaluation
    # observation fold_id values are rolling windows, not purged-CV folds.
    if ledger is not None and ledger.height > 0 and "fold_id" in ledger.columns:
        fold_ids.extend(int(value) for value in ledger["fold_id"].to_list())
    elif factor_fold.height > 0:
        fold_ids.extend(int(value) for value in factor_fold["fold_id"].to_list())
    unique_folds = sorted(set(fold_ids))
    if not unique_folds:
        return _empty(FOLD_COLUMNS)

    ledger_lookup: dict[int, dict[str, object]] = {}
    if ledger is not None and ledger.height > 0 and "fold_id" in ledger.columns:
        for row in ledger.sort("fold_id").iter_rows(named=True):
            ledger_lookup[int(row["fold_id"])] = row

    oos = _filter_oos_selected(observations, engine=engine)
    rows: list[dict[str, object]] = []
    for fold_id in unique_folds:
        fold_factors = factor_fold.filter(pl.col("fold_id") == fold_id)
        oriented_values = [
            value
            for value in (_as_float(v) for v in fold_factors["oriented_oos_ic"].to_list())
            if value is not None and math.isfinite(value)
        ]
        raw_values = [
            value
            for value in (_as_float(v) for v in fold_factors["raw_oos_ic"].to_list())
            if value is not None and math.isfinite(value)
        ]
        positive = sum(1 for value in oriented_values if value > 0.0)
        negative = sum(1 for value in oriented_values if value < 0.0)
        ledger_row = ledger_lookup.get(fold_id, {})
        fold_obs = oos.filter(pl.col("fold_id") == fold_id) if oos is not None else None
        mean_return = None
        unique_timestamps = None
        median_cs = None
        oos_rows = _as_int(ledger_row.get("test_rows"))
        if fold_obs is not None and fold_obs.height > 0:
            target_base = fold_obs.select(["observation_time", "symbol", _TARGET]).unique()
            mean_return = _as_float(target_base[_TARGET].drop_nulls().mean())
            unique_timestamps = int(fold_obs["observation_time"].n_unique())
            cs = (
                fold_obs.select(["observation_time", "symbol"])
                .unique()
                .group_by("observation_time")
                .len()
            )
            median_cs = _as_float(cs["len"].median()) if cs.height else None
            if oos_rows is None:
                oos_rows = int(fold_obs.height)
        rows.append(
            {
                "timeframe": timeframe,
                "year": year,
                "fold_id": fold_id,
                "train_rows": _as_int(ledger_row.get("train_rows")),
                "oos_rows": oos_rows,
                "raw_oos_mean_return": mean_return,
                "raw_oos_ic": _mean(raw_values),
                "oriented_oos_ic": _mean(oriented_values),
                "positive_factor_count": positive,
                "negative_factor_count": negative,
                "median_factor_ic": (
                    statistics.median(oriented_values) if oriented_values else None
                ),
                "dispersion_factor_ic": _std(oriented_values),
                "factors_positive_oos_ic": positive,
                "factors_negative_oos_ic": negative,
                "unique_timestamps": unique_timestamps,
                "median_cross_section": median_cs,
            }
        )
    return _sort_frame(_frame_from_dicts(rows, FOLD_SCHEMA), FOLD_COLUMNS)


def _build_distribution_frame(
    *,
    observations: pl.DataFrame | None,
    engine: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    oos = _filter_oos_selected(observations, engine=engine)
    if oos is None or oos.height == 0 or _TARGET not in oos.columns:
        return _empty(DISTRIBUTION_COLUMNS)

    def _append_target_stats(
        frame: pl.DataFrame,
        *,
        fold_id: int | None,
        group: str,
    ) -> None:
        base = frame.select(["observation_time", "symbol", _TARGET]).unique()
        series = base[_TARGET].drop_nulls()
        if series.len() == 0:
            return
        stats = {
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "q01": series.quantile(0.01),
            "q05": series.quantile(0.05),
            "q25": series.quantile(0.25),
            "q50": series.quantile(0.50),
            "q75": series.quantile(0.75),
            "q95": series.quantile(0.95),
            "q99": series.quantile(0.99),
            "positive_rate": float((series > 0).sum() / series.len()),
            "zero_rate": float((series == 0).sum() / series.len()),
            "negative_rate": float((series < 0).sum() / series.len()),
            "min": series.min(),
            "max": series.max(),
            "n": float(series.len()),
        }
        for name, value in stats.items():
            rows.append(
                {
                    "timeframe": timeframe,
                    "year": year,
                    "metric_group": group,
                    "metric_name": name,
                    "metric_value": _as_float(value),
                    "fold_id": fold_id,
                }
            )

    _append_target_stats(oos, fold_id=None, group="target_overall")
    if "fold_id" in oos.columns:
        for fold_id in sorted(int(value) for value in oos["fold_id"].unique().to_list()):
            _append_target_stats(
                oos.filter(pl.col("fold_id") == fold_id),
                fold_id=fold_id,
                group="target_fold",
            )

    # Cross-sectional breadth distribution.
    cs = oos.select(["observation_time", "symbol"]).unique().group_by("observation_time").len()
    if cs.height > 0:
        breadth_stats = {
            "timestamps": float(cs.height),
            "median_observations_per_timestamp": cs["len"].median(),
            "min_observations_per_timestamp": cs["len"].min(),
            "max_observations_per_timestamp": cs["len"].max(),
            "mean_observations_per_timestamp": cs["len"].mean(),
            "active_assets": float(oos["symbol"].n_unique()),
            "fraction_timestamps_le_5": float((cs["len"] <= 5).sum() / cs.height),
            "fraction_timestamps_le_10": float((cs["len"] <= 10).sum() / cs.height),
            "fraction_timestamps_le_20": float((cs["len"] <= 20).sum() / cs.height),
        }
        for name, value in breadth_stats.items():
            rows.append(
                {
                    "timeframe": timeframe,
                    "year": year,
                    "metric_group": "cross_section",
                    "metric_name": name,
                    "metric_value": _as_float(value),
                    "fold_id": None,
                }
            )
    return _sort_frame(pl.DataFrame(rows), DISTRIBUTION_COLUMNS)


def _build_comparison_frame(
    *,
    storage_root: Path,
    manager: str,
    exchange: str,
    market: str,
    engine: str,
    year: int,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    stability_global = Path("reports") / "factor_stability" / "factor_stability_global.csv"
    stability_lookup: dict[str, dict[str, object]] = {}
    if stability_global.exists():
        for row in pl.read_csv(stability_global).iter_rows(named=True):
            stability_lookup[str(row["timeframe"])] = row

    for timeframe in COMPARISON_TIMEFRAMES:
        selection_path = _evaluation_path(
            storage_root,
            STORAGE_DIR_FACTOR_SELECTION,
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        )
        selection = _load_optional_parquet(selection_path)
        if selection is None:
            continue
        selected = (
            selection.filter(pl.col("selected")) if "selected" in selection.columns else selection
        )
        wf_path = _evaluation_path(
            storage_root,
            STORAGE_DIR_WALK_FORWARD_EVALUATION,
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        )
        pcv_path = _evaluation_path(
            storage_root,
            STORAGE_DIR_PURGED_CV_EVALUATION,
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        )
        # Prefer compact 1d full IC recompute; peer timeframes use lazy breadth/null
        # stats plus previously persisted stability IC summaries when available.
        if timeframe == TARGET_TIMEFRAME:
            eval_columns = (
                "engine",
                "selected",
                "partition",
                "fold_id",
                "observation_time",
                "symbol",
                "factor_name",
                "factor_version",
                _FACTOR_VALUE,
                _TARGET,
                "selected_direction",
                "selection_ic",
                "orientation_policy",
            )
            wf_obs = _load_optional_parquet(wf_path, columns=eval_columns)
            pcv_obs = _load_optional_parquet(pcv_path, columns=eval_columns)
            wf_factor_fold = _compute_factor_fold_ics(wf_obs, engine=engine)
            pcv_factor_fold = _compute_factor_fold_ics(pcv_obs, engine=engine)
            primary = pcv_obs if pcv_obs is not None and pcv_obs.height > 0 else wf_obs
            metrics = _comparison_metrics_from_observations(
                primary,
                selected=selected,
                engine=engine,
                wf_factor_fold=wf_factor_fold,
                pcv_factor_fold=pcv_factor_fold,
            )
        else:
            metrics = _comparison_metrics_lazy(
                wf_path=wf_path,
                pcv_path=pcv_path,
                selected=selected,
                engine=engine,
                stability_row=stability_lookup.get(timeframe),
            )
        rows.append(
            {
                "timeframe": timeframe,
                "selected_factors": int(selected.height),
                "tested_factors": int(selection.height),
                "selection_ratio": (
                    float(selected.height) / float(selection.height) if selection.height else None
                ),
                **metrics,
            }
        )
    return _sort_frame(pl.DataFrame(rows), COMPARISON_COLUMNS)


def _comparison_metrics_from_observations(
    observations: pl.DataFrame | None,
    *,
    selected: pl.DataFrame,
    engine: str,
    wf_factor_fold: pl.DataFrame,
    pcv_factor_fold: pl.DataFrame,
) -> dict[str, object]:
    oos = _filter_oos_selected(observations, engine=engine)
    unique_ts = int(oos["observation_time"].n_unique()) if oos is not None else None
    median_cs = min_cs = max_cs = None
    target_mean = target_std = target_pos = None
    median_null = None
    degenerate_count = 0
    oos_rows = int(oos.height) if oos is not None else 0
    if oos is not None and oos.height > 0:
        cs = oos.select(["observation_time", "symbol"]).unique().group_by("observation_time").len()
        if cs.height:
            median_cs = _as_float(cs["len"].median())
            min_cs = _as_int(cs["len"].min())
            max_cs = _as_int(cs["len"].max())
        target_base = oos.select(["observation_time", "symbol", _TARGET]).unique()
        series = target_base[_TARGET].drop_nulls()
        if series.len():
            target_mean = _as_float(series.mean())
            target_std = _as_float(series.std())
            target_pos = float((series > 0).sum() / series.len())
        null_rates: list[float] = []
        by_factor = oos.group_by("factor_name").agg(
            [
                pl.len().alias("n"),
                pl.col(_FACTOR_VALUE).null_count().alias("null_n"),
            ]
        )
        selected_names = set(selected["factor_name"].to_list())
        for row in by_factor.iter_rows(named=True):
            if row["factor_name"] not in selected_names:
                continue
            n = int(row["n"])
            if n <= 0:
                continue
            null_rate = float(row["null_n"]) / float(n)
            null_rates.append(null_rate)
            if null_rate >= 1.0 - 1e-12:
                degenerate_count += 1
        median_null = statistics.median(null_rates) if null_rates else None
    oriented_fold_means = [
        value
        for value in (_as_float(v) for v in wf_factor_fold["oriented_oos_ic"].to_list())
        if value is not None and math.isfinite(value)
    ]
    ic_se = (
        float(statistics.stdev(oriented_fold_means) / math.sqrt(len(oriented_fold_means)))
        if len(oriented_fold_means) >= 2
        else None
    )
    ci_width = None
    if oriented_fold_means:
        low, high = _bootstrap_mean_ci(oriented_fold_means)
        if low is not None and high is not None:
            ci_width = high - low
    return {
        "wf_raw_oos_ic": _mean([_as_float(v) for v in wf_factor_fold["raw_oos_ic"].to_list()]),
        "wf_oriented_oos_ic": _mean(
            [_as_float(v) for v in wf_factor_fold["oriented_oos_ic"].to_list()]
        ),
        "pcv_raw_oos_ic": _mean([_as_float(v) for v in pcv_factor_fold["raw_oos_ic"].to_list()]),
        "pcv_oriented_oos_ic": _mean(
            [_as_float(v) for v in pcv_factor_fold["oriented_oos_ic"].to_list()]
        ),
        "unique_oos_timestamps": unique_ts,
        "median_cross_section": median_cs,
        "min_cross_section": min_cs,
        "max_cross_section": max_cs,
        "oos_observation_rows": oos_rows,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_positive_rate": target_pos,
        "median_factor_null_rate": median_null,
        "degenerate_factor_count": degenerate_count,
        "ic_standard_error": ic_se,
        "bootstrap_ci_width": ci_width,
    }


def _comparison_metrics_lazy(
    *,
    wf_path: Path,
    pcv_path: Path,
    selected: pl.DataFrame,
    engine: str,
    stability_row: Mapping[str, object] | None,
) -> dict[str, object]:
    source_path = wf_path if wf_path.exists() else pcv_path
    unique_ts = median_cs = min_cs = max_cs = None
    target_mean = target_std = target_pos = None
    median_null = None
    degenerate_count = 0
    oos_rows = 0
    if source_path.exists():
        lazy = pl.scan_parquet(source_path)
        schema_names = set(lazy.collect_schema().names())
        filters: list[pl.Expr] = []
        if "partition" in schema_names:
            filters.append(pl.col("partition") == _PARTITION_OOS)
        if "selected" in schema_names:
            filters.append(pl.col("selected"))
        if "engine" in schema_names:
            filters.append(pl.col("engine") == engine)
        filtered = lazy.filter(filters) if filters else lazy
        oos_rows = int(filtered.select(pl.len()).collect().item())
        if {"observation_time", "symbol"}.issubset(schema_names):
            cs = (
                filtered.select(["observation_time", "symbol"])
                .unique()
                .group_by("observation_time")
                .agg(pl.len().alias("len"))
                .collect()
            )
            unique_ts = int(cs.height)
            if cs.height:
                median_cs = _as_float(cs["len"].median())
                min_cs = _as_int(cs["len"].min())
                max_cs = _as_int(cs["len"].max())
        if {_TARGET, "observation_time", "symbol"}.issubset(schema_names):
            target_base = (
                filtered.select(["observation_time", "symbol", _TARGET]).unique().collect()
            )
            series = target_base[_TARGET].drop_nulls()
            if series.len():
                target_mean = _as_float(series.mean())
                target_std = _as_float(series.std())
                target_pos = float((series > 0).sum() / series.len())
        if {"factor_name", _FACTOR_VALUE}.issubset(schema_names):
            selected_names = selected["factor_name"].to_list()
            by_factor = (
                filtered.filter(pl.col("factor_name").is_in(selected_names))
                .group_by("factor_name")
                .agg(
                    [
                        pl.len().alias("n"),
                        pl.col(_FACTOR_VALUE).null_count().alias("null_n"),
                    ]
                )
                .collect()
            )
            null_rates: list[float] = []
            for row in by_factor.iter_rows(named=True):
                n = int(row["n"])
                if n <= 0:
                    continue
                null_rate = float(row["null_n"]) / float(n)
                null_rates.append(null_rate)
                if null_rate >= 1.0 - 1e-12:
                    degenerate_count += 1
            median_null = statistics.median(null_rates) if null_rates else None

    wf_raw = wf_oriented = pcv_raw = pcv_oriented = None
    ic_se = ci_width = None
    if stability_row is not None:
        wf_raw = _as_float(stability_row.get("wf_raw_oos_ic"))
        wf_oriented = _as_float(stability_row.get("wf_oriented_oos_ic"))
        pcv_raw = _as_float(stability_row.get("pcv_raw_oos_ic"))
        pcv_oriented = _as_float(stability_row.get("pcv_oriented_oos_ic"))
    return {
        "wf_raw_oos_ic": wf_raw,
        "wf_oriented_oos_ic": wf_oriented,
        "pcv_raw_oos_ic": pcv_raw,
        "pcv_oriented_oos_ic": pcv_oriented,
        "unique_oos_timestamps": unique_ts,
        "median_cross_section": median_cs,
        "min_cross_section": min_cs,
        "max_cross_section": max_cs,
        "oos_observation_rows": oos_rows,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_positive_rate": target_pos,
        "median_factor_null_rate": median_null,
        "degenerate_factor_count": degenerate_count,
        "ic_standard_error": ic_se,
        "bootstrap_ci_width": ci_width,
    }


def _build_global_frame(
    *,
    selected: pl.DataFrame,
    selection: pl.DataFrame,
    factor_frame: pl.DataFrame,
    fold_frame: pl.DataFrame,
    distribution_frame: pl.DataFrame,
    comparison_frame: pl.DataFrame,
    wf_factor_fold: pl.DataFrame,
    pcv_factor_fold: pl.DataFrame,
    observations: pl.DataFrame | None,
    redundancy: Mapping[str, object],
    alignment: Mapping[str, object],
    engine: str,
    timeframe: str,
    year: int,
    tested_factors: int,
    selected_factors: int,
) -> pl.DataFrame:
    oriented_means = [
        value
        for value in (_as_float(v) for v in factor_frame["mean_oriented_oos_ic"].to_list())
        if value is not None and math.isfinite(value)
    ]
    positive_factors = sum(1 for value in oriented_means if value > 0.0)
    negative_factors = sum(1 for value in oriented_means if value < 0.0)
    degenerate_count = (
        int(factor_frame.filter(pl.col("degenerate_flag")).height) if factor_frame.height else 0
    )
    high_miss_count = (
        int(factor_frame.filter(pl.col("high_missingness_flag")).height)
        if factor_frame.height
        else 0
    )
    null_rates = [
        value
        for value in (_as_float(v) for v in factor_frame["null_rate"].to_list())
        if value is not None and math.isfinite(value)
    ]
    median_null = statistics.median(null_rates) if null_rates else None
    degradations = [
        value
        for value in (_as_float(v) for v in factor_frame["oriented_oos_minus_is"].to_list())
        if value is not None and math.isfinite(value)
    ]
    train_pos_oos_neg = 0
    if factor_frame.height:
        for row in factor_frame.iter_rows(named=True):
            is_ic = _as_float(row.get("oriented_is_ic"))
            oos_ic = _as_float(row.get("mean_oriented_oos_ic"))
            if is_ic is not None and oos_ic is not None and is_ic > 0.0 and oos_ic < 0.0:
                train_pos_oos_neg += 1

    oos = _filter_oos_selected(observations, engine=engine)
    unique_ts = int(oos["observation_time"].n_unique()) if oos is not None else None
    median_cs = _distribution_value(
        distribution_frame, "cross_section", "median_observations_per_timestamp"
    )
    min_cs = _distribution_value(
        distribution_frame, "cross_section", "min_observations_per_timestamp"
    )
    max_cs = _distribution_value(
        distribution_frame, "cross_section", "max_observations_per_timestamp"
    )
    target_std = _distribution_value(distribution_frame, "target_overall", "std")
    target_pos = _distribution_value(distribution_frame, "target_overall", "positive_rate")

    oriented_fold_ics = [
        value
        for value in (_as_float(v) for v in fold_frame["oriented_oos_ic"].to_list())
        if value is not None and math.isfinite(value)
    ]
    ic_se = (
        float(statistics.stdev(oriented_fold_ics) / math.sqrt(len(oriented_fold_ics)))
        if len(oriented_fold_ics) >= 2
        else None
    )
    ci_low, ci_high = _bootstrap_mean_ci(oriented_fold_ics)
    ci_width = (ci_high - ci_low) if ci_low is not None and ci_high is not None else None

    other = comparison_frame.filter(pl.col("timeframe") != TARGET_TIMEFRAME)
    comparison_median_ts = (
        _as_float(other["unique_oos_timestamps"].drop_nulls().median()) if other.height else None
    )
    comparison_median_cs = (
        _as_float(other["median_cross_section"].drop_nulls().median()) if other.height else None
    )
    comparison_median_target_std = (
        _as_float(other["target_std"].drop_nulls().median()) if other.height else None
    )
    comparison_selection_ratio = (
        _as_float(other["selection_ratio"].drop_nulls().median()) if other.height else None
    )

    wf_oriented = _mean([_as_float(v) for v in wf_factor_fold["oriented_oos_ic"].to_list()])
    wf_raw = _mean([_as_float(v) for v in wf_factor_fold["raw_oos_ic"].to_list()])
    pcv_oriented = _mean([_as_float(v) for v in pcv_factor_fold["oriented_oos_ic"].to_list()])
    pcv_raw = _mean([_as_float(v) for v in pcv_factor_fold["raw_oos_ic"].to_list()])
    is_ic = _mean(
        [
            value
            for value in (_as_float(v) for v in factor_frame["oriented_is_ic"].to_list())
            if value is not None and math.isfinite(value)
        ]
    )
    selection_ratio = (
        float(selected_factors) / float(tested_factors) if tested_factors > 0 else None
    )
    fold_count = int(fold_frame.height)
    negative_fold_count = (
        int(fold_frame.filter(pl.col("oriented_oos_ic") < 0.0).height) if fold_count else 0
    )
    oos_rows_median = (
        _as_float(fold_frame["oos_rows"].drop_nulls().median()) if fold_count else None
    )
    effective_factors = selected_factors - (_as_int(redundancy.get("redundant_group_count")) or 0)
    if effective_factors < 0:
        effective_factors = 0

    primary, secondary, confidence = classify_primary_root_cause(
        selected_factors=selected_factors,
        fold_count=fold_count,
        degenerate_factor_count=degenerate_count,
        high_missingness_factor_count=high_miss_count,
        median_null_rate=median_null,
        unique_oos_timestamps=unique_ts,
        comparison_median_timestamps=comparison_median_ts,
        median_cross_section=median_cs,
        comparison_median_cross_section=comparison_median_cs,
        target_std=target_std,
        comparison_median_target_std=comparison_median_target_std,
        redundant_group_count=_as_int(redundancy.get("redundant_group_count")) or 0,
        redundancy_status=str(redundancy.get("redundancy_status")),
        selection_ratio=selection_ratio,
        comparison_selection_ratio=comparison_selection_ratio,
        train_positive_oos_negative_count=train_pos_oos_neg,
        degradation_median=(statistics.median(degradations) if degradations else None),
        negative_fold_count=negative_fold_count,
        oos_oriented_ic=wf_oriented if wf_oriented is not None else pcv_oriented,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        alignment_issue=bool(alignment.get("alignment_issue")),
    )

    fold_stability = (
        "CONCENTRATED"
        if fold_count > 0 and negative_fold_count >= fold_count - 1 and fold_count >= 3
        else (
            "PERSISTENT_NEGATIVE"
            if fold_count > 0 and negative_fold_count > (fold_count / 2.0)
            else "MIXED"
        )
    )
    factor_stability = (
        "DEGENERATE_HEAVY"
        if degenerate_count >= max(1, selected_factors // 4)
        else ("BROAD_NEGATIVE" if negative_factors > positive_factors else "MIXED_OR_POSITIVE")
    )
    power_label = (
        "LIMITED"
        if unique_ts is not None
        and comparison_median_ts is not None
        and unique_ts < 0.25 * comparison_median_ts
        else "ADEQUATE_RELATIVE"
    )
    breadth_label = (
        "LIMITED"
        if median_cs is not None
        and comparison_median_cs is not None
        and median_cs < 0.5 * comparison_median_cs
        else "NOT_LIMITING"
    )
    target_label = (
        "NOISIER_THAN_PEERS"
        if target_std is not None
        and comparison_median_target_std is not None
        and target_std > 3.0 * comparison_median_target_std
        else "COMPARABLE_SCALE"
    )
    degeneracy_label = (
        f"DEGENERATE={degenerate_count};HIGH_MISSING={high_miss_count};"
        f"MEDIAN_NULL={_fmt(median_null)}"
    )
    redundancy_label = str(redundancy.get("redundancy_status"))
    selection_label = (
        f"RATIO={_fmt(selection_ratio)};TESTED={tested_factors};SELECTED={selected_factors}"
    )
    degradation_label = (
        f"MEAN={_fmt(_mean(degradations))};MEDIAN="
        f"{_fmt(statistics.median(degradations) if degradations else None)};"
        f"TRAIN_POS_OOS_NEG={train_pos_oos_neg}"
    )
    alignment_label = str(alignment.get("summary"))

    row = {
        "timeframe": timeframe,
        "year": year,
        "selected_factors": selected_factors,
        "tested_factors": tested_factors,
        "fold_count": fold_count,
        "is_ic": is_ic,
        "oos_raw_ic": wf_raw,
        "oos_oriented_ic": wf_oriented,
        "pcv_raw_ic": pcv_raw,
        "pcv_oriented_ic": pcv_oriented,
        "positive_oos_factor_count": positive_factors,
        "negative_oos_factor_count": negative_factors,
        "degenerate_factor_count": degenerate_count,
        "high_missingness_factor_count": high_miss_count,
        "median_null_rate": median_null,
        "unique_oos_timestamps": unique_ts,
        "median_cross_section": median_cs,
        "min_cross_section": min_cs,
        "max_cross_section": max_cs,
        "oos_rows_per_fold_median": oos_rows_median,
        "ic_standard_error": ic_se,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_ci_width": ci_width,
        "target_std": target_std,
        "target_positive_rate": target_pos,
        "redundant_group_count": redundancy.get("redundant_group_count"),
        "redundancy_status": redundancy.get("redundancy_status"),
        "effective_independent_factors": effective_factors,
        "selection_ratio": selection_ratio,
        "degradation_mean": _mean(degradations),
        "degradation_median": (statistics.median(degradations) if degradations else None),
        "train_positive_oos_negative_count": train_pos_oos_neg,
        "negative_fold_count": negative_fold_count,
        "fold_stability": fold_stability,
        "factor_stability": factor_stability,
        "cross_sectional_breadth": breadth_label,
        "statistical_power": power_label,
        "target_distribution": target_label,
        "factor_degeneracy": degeneracy_label,
        "factor_redundancy": redundancy_label,
        "selection_stability": selection_label,
        "train_oos_degradation": degradation_label,
        "timeframe_alignment": alignment_label,
        "primary_classification": primary,
        "secondary_causes": "|".join(secondary),
        "evidence_confidence": confidence,
        "leakage": "BOUNDARY_PRESERVED",
        "production_artifacts_unchanged": True,
        "deterministic": True,
    }
    _ = selection  # selection retained for lineage/tested count already captured
    _ = selected
    return _sort_frame(pl.DataFrame([row]), GLOBAL_COLUMNS)


def _analyze_redundancy(
    observations: pl.DataFrame | None,
    *,
    selected: pl.DataFrame,
    engine: str,
) -> dict[str, object]:
    unavailable: dict[str, object] = {
        "redundancy_status": "REDUNDANCY_ANALYSIS_UNAVAILABLE",
        "redundancy_threshold": None,
        "redundant_group_count": 0,
    }
    if observations is None or observations.height == 0 or selected.height == 0:
        return unavailable
    if "partition" not in observations.columns or _FACTOR_VALUE not in observations.columns:
        return unavailable
    train = observations
    if "engine" in train.columns:
        train = train.filter(pl.col("engine") == engine)
    if _PARTITION_TRAIN not in set(train["partition"].unique().to_list()):
        # Fall back is intentionally unavailable: TRAIN-only boundary required.
        return unavailable
    train = train.filter(pl.col("partition") == _PARTITION_TRAIN)
    if "selected" in train.columns:
        train = train.filter(pl.col("selected"))
    if train.height == 0:
        return unavailable
    if not {"factor_name", "factor_version", "observation_time", _FACTOR_VALUE}.issubset(
        train.columns
    ):
        return unavailable
    keys = selected.select(["factor_name", "factor_version"]).unique(maintain_order=True)
    train_selected = train.join(keys, on=["factor_name", "factor_version"], how="inner")
    if train_selected.height == 0:
        return unavailable
    wide = (
        train_selected.with_columns(
            (
                pl.col("factor_name").cast(pl.String)
                + pl.lit("::")
                + pl.col("factor_version").cast(pl.String)
            ).alias("_factor_key")
        )
        .select(["observation_time", "_factor_key", _FACTOR_VALUE])
        .pivot(values=_FACTOR_VALUE, index="observation_time", on="_factor_key")
        .sort("observation_time")
    )
    factor_cols = [column for column in wide.columns if column != "observation_time"]
    if len(factor_cols) < 2:
        return {
            "redundancy_status": "AVAILABLE",
            "redundancy_threshold": _REDUNDANCY_SPEARMAN_THRESHOLD,
            "redundant_group_count": 0,
        }
    parent = {column: column for column in factor_cols}

    def _find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def _union(left: str, right: str) -> None:
        root_left = _find(left)
        root_right = _find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for index, left in enumerate(factor_cols):
        for right in factor_cols[index + 1 :]:
            pair = wide.select(
                [pl.col(left).alias("left"), pl.col(right).alias("right")]
            ).drop_nulls()
            if pair.height < 2:
                continue
            if pair["left"].n_unique() < 2 or pair["right"].n_unique() < 2:
                continue
            corr = _as_float(pair.select(pl.corr("left", "right", method="spearman")).item())
            if corr is not None and math.isfinite(corr) and corr >= _REDUNDANCY_SPEARMAN_THRESHOLD:
                _union(left, right)
    groups: dict[str, list[str]] = {}
    for column in factor_cols:
        groups.setdefault(_find(column), []).append(column)
    redundant_groups = sum(1 for members in groups.values() if len(members) >= 2)
    return {
        "redundancy_status": "AVAILABLE",
        "redundancy_threshold": _REDUNDANCY_SPEARMAN_THRESHOLD,
        "redundant_group_count": redundant_groups,
    }


def _assess_timeframe_alignment(
    *,
    observations: pl.DataFrame | None,
    wf_ledger: pl.DataFrame | None,
    comparison_obs: pl.DataFrame | None,
    engine: str,
) -> dict[str, object]:
    oos = _filter_oos_selected(observations, engine=engine)
    unique_ts = int(oos["observation_time"].n_unique()) if oos is not None else 0
    comparison = _filter_oos_selected(comparison_obs, engine=engine)
    comparison_ts = (
        int(comparison["observation_time"].n_unique()) if comparison is not None else None
    )
    ledger_equal_timestamps = False
    if (
        wf_ledger is not None
        and wf_ledger.height > 0
        and {"train_start", "train_end", "test_start", "test_end"}.issubset(wf_ledger.columns)
    ):
        equal = (
            (pl.col("train_start") == pl.col("train_end"))
            & (pl.col("train_end") == pl.col("test_start"))
            & (pl.col("test_start") == pl.col("test_end"))
        )
        equal_count = int(wf_ledger.select(equal.sum()).item())
        ledger_equal_timestamps = equal_count == int(wf_ledger.height)
    short_history = unique_ts > 0 and comparison_ts is not None and unique_ts < 0.25 * comparison_ts
    alignment_issue = short_history
    summary = (
        f"unique_oos_timestamps={unique_ts};"
        f"comparison_4h_timestamps={comparison_ts};"
        f"wf_ledger_all_bounds_equal={ledger_equal_timestamps};"
        f"short_history={short_history}"
    )
    return {"alignment_issue": alignment_issue, "summary": summary}


def _factor_value_stats(
    factor_obs: pl.DataFrame | None,
) -> tuple[float | None, float | None, int | None, float | None]:
    if factor_obs is None or factor_obs.height == 0 or _FACTOR_VALUE not in factor_obs.columns:
        return None, None, None, None
    null_rate = float(factor_obs[_FACTOR_VALUE].null_count() / factor_obs.height)
    valid = factor_obs[_FACTOR_VALUE].drop_nulls()
    variance = _as_float(valid.var()) if valid.len() else None
    unique_values = int(valid.n_unique()) if valid.len() else 0
    cs_dispersion = None
    if "observation_time" in factor_obs.columns:
        per_ts = (
            factor_obs.drop_nulls([_FACTOR_VALUE])
            .group_by("observation_time")
            .agg(pl.col(_FACTOR_VALUE).std().alias("std"))
        )
        if per_ts.height:
            cs_dispersion = _as_float(per_ts["std"].drop_nulls().median())
    return null_rate, variance, unique_values, cs_dispersion


def _filter_oos_selected(
    observations: pl.DataFrame | None,
    *,
    engine: str,
) -> pl.DataFrame | None:
    if observations is None or observations.height == 0:
        return None
    working = observations
    if "engine" in working.columns:
        working = working.filter(pl.col("engine") == engine)
    if "selected" in working.columns:
        working = working.filter(pl.col("selected"))
    if "partition" in working.columns:
        working = working.filter(pl.col("partition") == _PARTITION_OOS)
    return working


def _ensure_orientation_columns(frame: pl.DataFrame) -> pl.DataFrame:
    working = frame
    if "selected_direction" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Int32).alias("selected_direction"))
    if "selection_ic" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Float64).alias("selection_ic"))
    if "orientation_policy" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.String).alias("orientation_policy"))
    return working


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    replicates: int = _BOOTSTRAP_REPLICATES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = _SeededRNG(seed)
    samples: list[float] = []
    population = list(values)
    n = len(population)
    for _ in range(replicates):
        draw = [population[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / float(n))
    samples.sort()
    low_index = int(0.025 * (replicates - 1))
    high_index = int(0.975 * (replicates - 1))
    return float(samples[low_index]), float(samples[high_index])


class _SeededRNG:
    """Deterministic minimal RNG for bootstrap CI without NumPy dependency."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = int(seed) % (2**31 - 1) or 1

    def randrange(self, n: int) -> int:
        # Park-Miller LCG
        self._state = (self._state * 48271) % 2147483647
        return int(self._state % n)


def _distribution_value(
    frame: pl.DataFrame,
    group: str,
    name: str,
) -> float | None:
    if frame.height == 0:
        return None
    matched = frame.filter((pl.col("metric_group") == group) & (pl.col("metric_name") == name))
    if matched.height == 0:
        return None
    return _as_float(matched["metric_value"][0])


def _render_summary(
    *,
    global_row: Mapping[str, object],
    fold_frame: pl.DataFrame,
    factor_frame: pl.DataFrame,
) -> str:
    secondary = str(global_row.get("secondary_causes") or "")
    lines = [
        "CQROS 1d ROOT-CAUSE INVESTIGATION",
        "=================================",
        "",
        f"Primary classification: {global_row.get('primary_classification')}",
        f"Secondary contributing causes: {secondary.replace('|', '; ') if secondary else 'None'}",
        "",
        f"Evidence confidence: {global_row.get('evidence_confidence')}",
        "",
        f"1d selected factors: {global_row.get('selected_factors')}",
        f"1d folds: {global_row.get('fold_count')}",
        f"Candidate factors tested: {global_row.get('tested_factors')}",
        f"Selected factors: {global_row.get('selected_factors')}",
        "",
        f"IS IC: {_fmt(global_row.get('is_ic'))}",
        f"OOS raw IC: {_fmt(global_row.get('oos_raw_ic'))}",
        f"OOS oriented IC: {_fmt(global_row.get('oos_oriented_ic'))}",
        "",
        f"Positive OOS factor count: {global_row.get('positive_oos_factor_count')}",
        f"Negative OOS factor count: {global_row.get('negative_oos_factor_count')}",
        "",
        f"Fold stability: {global_row.get('fold_stability')} "
        f"(negative_folds={global_row.get('negative_fold_count')}/"
        f"{global_row.get('fold_count')})",
        f"Factor stability: {global_row.get('factor_stability')}",
        f"Cross-sectional breadth: {global_row.get('cross_sectional_breadth')} "
        f"(median_cs={_fmt(global_row.get('median_cross_section'))}; "
        f"min={_fmt(global_row.get('min_cross_section'))}; "
        f"max={_fmt(global_row.get('max_cross_section'))})",
        f"Statistical power: {global_row.get('statistical_power')} "
        f"(unique_oos_timestamps={global_row.get('unique_oos_timestamps')}; "
        f"ic_se={_fmt(global_row.get('ic_standard_error'))}; "
        f"bootstrap_ci=[{_fmt(global_row.get('bootstrap_ci_low'))}, "
        f"{_fmt(global_row.get('bootstrap_ci_high'))}])",
        f"Target distribution: {global_row.get('target_distribution')} "
        f"(std={_fmt(global_row.get('target_std'))}; "
        f"positive_rate={_fmt(global_row.get('target_positive_rate'))})",
        f"Factor degeneracy: {global_row.get('factor_degeneracy')}",
        f"Factor redundancy: {global_row.get('factor_redundancy')}",
        f"Selection stability: {global_row.get('selection_stability')}",
        f"Train/OOS degradation: {global_row.get('train_oos_degradation')}",
        f"Timeframe alignment: {global_row.get('timeframe_alignment')}",
        "",
        f"Leakage: {global_row.get('leakage')}",
        f"Production artifact mutation: "
        f"production_artifacts_unchanged="
        f"{global_row.get('production_artifacts_unchanged')}",
        f"Determinism: deterministic={global_row.get('deterministic')}",
        "",
        "Final conclusion: "
        f"{global_row.get('primary_classification')} based on measured "
        f"degenerate_factor_count={global_row.get('degenerate_factor_count')}, "
        f"median_null_rate={_fmt(global_row.get('median_null_rate'))}, "
        f"unique_oos_timestamps={global_row.get('unique_oos_timestamps')}, "
        f"oos_oriented_ic={_fmt(global_row.get('oos_oriented_ic'))}, "
        f"pcv_oriented_ic={_fmt(global_row.get('pcv_oriented_ic'))}, "
        f"negative_fold_count={global_row.get('negative_fold_count')}/"
        f"{global_row.get('fold_count')}.",
        "",
        "Fold-level oriented OOS IC:",
    ]
    if fold_frame.height:
        for row in fold_frame.sort("fold_id").iter_rows(named=True):
            lines.append(
                f"  fold={row['fold_id']}: oriented={_fmt(row['oriented_oos_ic'])} "
                f"raw={_fmt(row['raw_oos_ic'])} "
                f"pos_factors={row['positive_factor_count']} "
                f"neg_factors={row['negative_factor_count']}"
            )
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Factor-level oriented mean OOS IC (non-null only):")
    if factor_frame.height:
        usable = factor_frame.filter(pl.col("mean_oriented_oos_ic").is_not_null()).sort(
            "factor_name"
        )
        for row in usable.iter_rows(named=True):
            lines.append(
                f"  {row['factor_name']}: oriented={_fmt(row['mean_oriented_oos_ic'])} "
                f"null_rate={_fmt(row['null_rate'])} "
                f"deg={row['degenerate_flag']} "
                f"delta_oos_minus_is={_fmt(row['oriented_oos_minus_is'])}"
            )
    else:
        lines.append("  (none)")
    return "\n".join(lines) + "\n"


def _write_report_bundle(
    *,
    output_root: Path,
    global_frame: pl.DataFrame,
    fold_frame: pl.DataFrame,
    factor_frame: pl.DataFrame,
    distribution_frame: pl.DataFrame,
    comparison_frame: pl.DataFrame,
    summary_text: str,
) -> dict[str, Path]:
    paths = {
        "global": output_root / ROOT_CAUSE_GLOBAL_CSV_NAME,
        "folds": output_root / ROOT_CAUSE_FOLDS_CSV_NAME,
        "factors": output_root / ROOT_CAUSE_FACTORS_CSV_NAME,
        "distribution": output_root / ROOT_CAUSE_DISTRIBUTION_CSV_NAME,
        "comparison": output_root / ROOT_CAUSE_COMPARISON_CSV_NAME,
        "summary": output_root / ROOT_CAUSE_SUMMARY_TXT_NAME,
    }
    _write_csv(global_frame, paths["global"])
    _write_csv(fold_frame, paths["folds"])
    _write_csv(factor_frame, paths["factors"])
    _write_csv(distribution_frame, paths["distribution"])
    _write_csv(comparison_frame, paths["comparison"])
    paths["summary"].write_text(summary_text, encoding="utf-8", newline="\n")
    return paths


def _write_hash_manifest(path: Path, hashes: Mapping[str, str]) -> None:
    lines = [f"{key}={hashes[key]}" for key in sorted(hashes)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def _write_csv(frame: pl.DataFrame, path: Path) -> None:
    _stabilize_floats(frame).write_csv(path, null_value="")


def _stabilize_floats(frame: pl.DataFrame) -> pl.DataFrame:
    """Round float columns so identical computations serialize identically."""
    if frame.height == 0:
        return frame
    expressions: list[pl.Expr] = []
    for column in frame.columns:
        dtype = frame.schema[column]
        if dtype == pl.Float32 or dtype == pl.Float64:
            expressions.append(
                pl.when(pl.col(column).is_nan())
                .then(None)
                .otherwise(pl.col(column).round(12))
                .alias(column)
            )
    if not expressions:
        return frame
    return frame.with_columns(expressions)


def _nan_to_null(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for column in columns:
        if column not in frame.columns:
            continue
        expressions.append(
            pl.when(pl.col(column).is_nan()).then(None).otherwise(pl.col(column)).alias(column)
        )
    if not expressions:
        return frame
    return frame.with_columns(expressions)


def _load_optional_parquet(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame | None:
    if not path.exists():
        return None
    if columns is None:
        return pl.read_parquet(path)
    available = set(pl.scan_parquet(path).collect_schema().names())
    selected = [column for column in columns if column in available]
    if not selected:
        return pl.read_parquet(path)
    return pl.read_parquet(path, columns=selected)


def _sort_frame(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    ensured = _ensure_columns(frame, columns)
    sort_keys = [
        column
        for column in (
            "timeframe",
            "year",
            "factor_name",
            "factor_version",
            "fold_id",
            "metric_group",
            "metric_name",
        )
        if column in ensured.columns
    ]
    if not sort_keys:
        return ensured
    return ensured.sort(list(sort_keys))


def _ensure_columns(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    if frame.height == 0 and frame.width == 0:
        return _empty(columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        frame = frame.with_columns([pl.lit(None).alias(column) for column in missing])
    return frame.select(list(columns))


def _empty(columns: Sequence[str]) -> pl.DataFrame:
    return pl.DataFrame({column: [] for column in columns})


def _frame_from_dicts(
    rows: Sequence[Mapping[str, object]],
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    """Build a DataFrame from row dicts using an explicit schema.

    Polars dict inference can fail when early rows leave float columns null and
    later rows introduce float values into a column inferred as integer.
    """
    if not rows:
        return pl.DataFrame(schema=dict(schema))
    normalized = [{key: row.get(key) for key in schema} for row in rows]
    return pl.DataFrame(normalized, schema=dict(schema))


def _observations_aligned_to_pcv_folds(
    observations: pl.DataFrame,
    ledger: pl.DataFrame,
) -> pl.DataFrame:
    """Reassign observation ``fold_id`` values to purged-CV ledger folds.

    Walk-forward evaluation ledgers use rolling window ids. Purged-CV ledgers
    use contiguous test windows with ``test_start_time`` / ``test_end_time``.
    Root-cause fold summaries must follow the purged-CV fold identity.
    """
    required_obs = {"observation_time", "fold_id"}
    required_ledger = {"fold_id", "test_start_time", "test_end_time"}
    if not required_obs.issubset(observations.columns):
        return observations
    if not required_ledger.issubset(ledger.columns):
        return observations

    parts: list[pl.DataFrame] = []
    for row in ledger.sort("fold_id").iter_rows(named=True):
        fold_id = int(row["fold_id"])
        start = int(row["test_start_time"])
        end = int(row["test_end_time"])
        part = observations.filter(
            (pl.col("observation_time") >= start) & (pl.col("observation_time") < end)
        ).with_columns(pl.lit(fold_id).cast(pl.Int32).alias("fold_id"))
        if part.height > 0:
            parts.append(part)
    if not parts:
        return observations
    return pl.concat(parts, how="vertical")


def _format_pipe_list(values: Sequence[float | None]) -> str:
    parts: list[str] = []
    for value in values:
        if value is None or not math.isfinite(value):
            parts.append("")
        else:
            parts.append(f"{value:.12g}")
    return "|".join(parts)


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.stdev(values))


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fmt(value: object) -> str:
    number = _as_float(value)
    if number is None:
        return "" if value is None else str(value)
    return f"{number:.6g}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

"""CQROS Factor Selection Stability review reporter.

Purpose:
    Provide a read-only **selection stability** review over Factor Selection,
    Walk-Forward evaluation observations, and Purged-CV evaluation
    observations. This module is distinct from the Purged-CV
    ``factor_stability_diagnostic`` alignment/sign diagnostic.

Responsibilities:
    - Discover Factor Selection partitions under the storage root
    - Compute factor-fold and fold-level Spearman IC (raw and oriented)
    - Classify factor stability, timeframe status, and diagnostic verdicts
    - Emit deterministic CSV reports under ``reports/factor_stability``
    - Hash production ledgers before/after and require immutability
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``hashlib``, ``logging``, ``math``, ``ast``, ``polars``,
    ``cqros.core.constants``, and ``cqros.reporting.exceptions``.

Public API:
    Classification helpers, column/name constants,
    ``FactorStabilitySelectionReporter``,
    ``FactorStabilitySelectionResult``, and
    ``forbidden_import_violations``.

Notes:
    Orientation is inherited only from persisted selection / evaluation
    metadata (``selected_direction``, ``orientation_policy``). This reporter
    never mutates ledgers, never tunes selection thresholds, and never
    reconstructs TRAIN factor values from the Factors lake.
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
    FILE_EXTENSION_CSV,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTOR_SELECTION,
    STORAGE_DIR_PURGED_CV,
    STORAGE_DIR_PURGED_CV_EVALUATION,
    STORAGE_DIR_WALK_FORWARD,
    STORAGE_DIR_WALK_FORWARD_EVALUATION,
)
from cqros.reporting.exceptions import ReportingValidationError

__all__ = [
    "CROSS_TIMEFRAME_COLUMNS",
    "DEFAULT_OUTPUT_ROOT",
    "FACTOR_CLASS_INSUFFICIENT_DATA",
    "FACTOR_CLASS_MIXED",
    "FACTOR_CLASS_ORIENTATION_INSUFFICIENT",
    "FACTOR_CLASS_STABLE_NEGATIVE",
    "FACTOR_CLASS_STABLE_POSITIVE",
    "FACTOR_REPORT_COLUMNS",
    "FOLD_REPORT_COLUMNS",
    "FactorStabilitySelectionReporter",
    "FactorStabilitySelectionResult",
    "GLOBAL_COLUMNS",
    "GLOBAL_STATUS_INSUFFICIENT_DATA",
    "GLOBAL_STATUS_MIXED_STABILITY",
    "GLOBAL_STATUS_ORIENTATION_INSUFFICIENT",
    "GLOBAL_STATUS_SELECTION_DEGRADATION",
    "GLOBAL_STATUS_STABLE_SIGNAL",
    "REDUNDANCY_ANALYSIS_UNAVAILABLE",
    "SUMMARY_COLUMNS",
    "VERDICT_FACTOR_ORIENTATION_INSUFFICIENT",
    "VERDICT_FACTOR_REDUNDANCY",
    "VERDICT_INSUFFICIENT_DATA",
    "VERDICT_SELECTION_OVERFITTING",
    "VERDICT_STABLE_SIGNAL",
    "VERDICT_TIMEFRAME_SIGNAL_WEAKNESS",
    "classify_factor_stability",
    "classify_global_status",
    "classify_verdict",
    "forbidden_import_violations",
    "hash_watched_production_ledgers",
]

_logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT: Final[Path] = Path("reports") / "factor_stability"

FACTOR_CLASS_STABLE_POSITIVE: Final[str] = "STABLE_POSITIVE"
FACTOR_CLASS_MIXED: Final[str] = "MIXED"
FACTOR_CLASS_STABLE_NEGATIVE: Final[str] = "STABLE_NEGATIVE"
FACTOR_CLASS_ORIENTATION_INSUFFICIENT: Final[str] = "ORIENTATION_INSUFFICIENT"
FACTOR_CLASS_INSUFFICIENT_DATA: Final[str] = "INSUFFICIENT_DATA"

GLOBAL_STATUS_STABLE_SIGNAL: Final[str] = "STABLE_SIGNAL"
GLOBAL_STATUS_MIXED_STABILITY: Final[str] = "MIXED_STABILITY"
GLOBAL_STATUS_SELECTION_DEGRADATION: Final[str] = "SELECTION_DEGRADATION"
GLOBAL_STATUS_ORIENTATION_INSUFFICIENT: Final[str] = "ORIENTATION_INSUFFICIENT"
GLOBAL_STATUS_INSUFFICIENT_DATA: Final[str] = "INSUFFICIENT_DATA"

VERDICT_STABLE_SIGNAL: Final[str] = "A. STABLE_SIGNAL"
VERDICT_SELECTION_OVERFITTING: Final[str] = "B. SELECTION_OVERFITTING"
VERDICT_FACTOR_REDUNDANCY: Final[str] = "C. FACTOR_REDUNDANCY"
VERDICT_FACTOR_ORIENTATION_INSUFFICIENT: Final[str] = "D. FACTOR_ORIENTATION_INSUFFICIENT"
VERDICT_TIMEFRAME_SIGNAL_WEAKNESS: Final[str] = "E. TIMEFRAME_SIGNAL_WEAKNESS"
VERDICT_INSUFFICIENT_DATA: Final[str] = "F. INSUFFICIENT_DATA"

REDUNDANCY_ANALYSIS_UNAVAILABLE: Final[str] = "REDUNDANCY_ANALYSIS_UNAVAILABLE"

# Diagnostic reporting correlation threshold only. Not a selection/promotion rule.
_REDUNDANCY_SPEARMAN_THRESHOLD: Final[float] = 0.90

_FACTOR_VALUE: Final[str] = "factor_value"
_TARGET: Final[str] = "future_return_1"
_ORIENTED_ALIAS: Final[str] = "_oriented_factor_value"
_PARTITION_OOS: Final[str] = "OOS"
_PARTITION_TRAIN: Final[str] = "TRAIN"

_WATCHED_LEDGER_DIRS: Final[tuple[str, ...]] = (
    STORAGE_DIR_WALK_FORWARD,
    STORAGE_DIR_PURGED_CV,
    STORAGE_DIR_FACTOR_SELECTION,
)

_FORBIDDEN_IMPORT_MODULES: Final[tuple[str, ...]] = (
    "cqros.alpha",
    "cqros.regime",
    "cqros.predictions",
    "cqros.signals",
    "cqros.ml",
)

_ERROR_LEDGER_MUTATION: Final[str] = "REPORT-FACTOR-STABILITY-SELECTION-001"
_ERROR_MANAGER: Final[str] = "REPORT-FACTOR-STABILITY-SELECTION-002"
_ERROR_OUTPUT: Final[str] = "REPORT-FACTOR-STABILITY-SELECTION-003"

FACTOR_REPORT_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "factor_name",
    "factor_version",
    "factor_category",
    "selected_direction",
    "orientation_policy",
    "selection_ic",
    "oriented_training_ic",
    "raw_oos_ic_by_fold",
    "oriented_oos_ic_by_fold",
    "mean_raw_oos_ic",
    "mean_oriented_oos_ic",
    "std_oriented_oos_ic",
    "positive_oriented_folds",
    "negative_oriented_folds",
    "fraction_positive_oriented_folds",
    "min_oriented_fold_ic",
    "max_oriented_fold_ic",
    "wf_mean_raw_oos_ic",
    "wf_mean_oriented_oos_ic",
    "wf_factor_status",
    "ic_degradation",
    "abs_ic_degradation",
    "sign_preserved",
    "train_positive_oos_negative",
    "stability_class",
)

FOLD_REPORT_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "fold_id",
    "train_rows",
    "oos_rows",
    "raw_oos_ic",
    "oriented_oos_ic",
    "positive_oriented_factor_count",
    "negative_oriented_factor_count",
    "positive_factor_percentage",
    "aggregate_status",
)

SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "year",
    "selected_factors",
    "tested_factors",
    "selection_ratio",
    "wf_raw_oos_ic",
    "wf_oriented_oos_ic",
    "pcv_raw_oos_ic",
    "pcv_oriented_oos_ic",
    "oriented_positive_factor_ratio",
    "stable_positive_factor_count",
    "mixed_factor_count",
    "stable_negative_factor_count",
    "orientation_insufficient_factor_count",
    "insufficient_data_factor_count",
    "degradation_mean",
    "degradation_median",
    "fold_count",
    "negative_fold_count",
    "fold_concentration",
    "redundancy_status",
    "redundancy_threshold",
    "redundant_group_count",
    "status",
    "verdict",
)

GLOBAL_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "selected_factors",
    "tested_factors",
    "selection_ratio",
    "wf_raw_oos_ic",
    "wf_oriented_oos_ic",
    "pcv_raw_oos_ic",
    "pcv_oriented_oos_ic",
    "oriented_positive_factor_ratio",
    "stable_positive_factor_count",
    "mixed_factor_count",
    "stable_negative_factor_count",
    "degradation_mean",
    "degradation_median",
    "status",
)

CROSS_TIMEFRAME_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "selected_factors",
    "tested_factors",
    "selection_ratio",
    "wf_raw_oos_ic",
    "wf_oriented_oos_ic",
    "pcv_raw_oos_ic",
    "pcv_oriented_oos_ic",
    "oriented_positive_factor_ratio",
    "status",
)


@dataclass(frozen=True, slots=True)
class FactorStabilitySelectionResult:
    """Immutable result of a selection-stability review run."""

    factor_frames: Mapping[str, pl.DataFrame]
    fold_frames: Mapping[str, pl.DataFrame]
    summary_frames: Mapping[str, pl.DataFrame]
    cross_timeframe: pl.DataFrame
    global_summary: pl.DataFrame
    paths: Mapping[str, Path]
    verdicts: Mapping[str, str]
    production_ledgers_unchanged: bool
    hashes_before: Mapping[str, str]
    hashes_after: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _PanelAnalysis:
    """Internal per-panel factor, fold, and summary artifacts."""

    factor: pl.DataFrame
    fold: pl.DataFrame
    summary: dict[str, object]


class FactorStabilitySelectionReporter:
    """Read-only Factor Selection stability reporter.

    Args:
        storage_root: Lake root containing factor_selection / evaluation tiers.
        output_root: Directory receiving deterministic CSV reports.
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

    def run(self, timeframes: Sequence[str] | None = None) -> FactorStabilitySelectionResult:
        """Discover partitions, compute stability metrics, and write CSVs.

        Args:
            timeframes: Optional timeframe allowlist. ``None`` discovers all
                Factor Selection timeframes for the configured manager.

        Returns:
            Immutable report result including paths and ledger hash status.

        Raises:
            ReportingValidationError: When production ledgers change during
                the run or output configuration is invalid.
        """
        if self._output_root.exists() and not self._output_root.is_dir():
            raise ReportingValidationError(
                "output path must be a directory",
                error_code=_ERROR_OUTPUT,
                details={"output": str(self._output_root)},
            )

        hashes_before = hash_watched_production_ledgers(self._storage_root)
        partitions = _discover_factor_selection_partitions(
            self._storage_root,
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            timeframes=timeframes,
        )
        self._logger.info(
            "Factor stability selection review starting manager=%s partitions=%s",
            self._manager,
            len(partitions),
        )

        factor_frames: dict[str, pl.DataFrame] = {}
        fold_frames: dict[str, pl.DataFrame] = {}
        summary_frames: dict[str, pl.DataFrame] = {}
        summary_rows: list[dict[str, object]] = []
        paths: dict[str, Path] = {}
        verdicts: dict[str, str] = {}

        self._output_root.mkdir(parents=True, exist_ok=True)

        grouped: dict[str, list[tuple[str, int, Path]]] = {}
        for timeframe, year, path in partitions:
            grouped.setdefault(timeframe, []).append((timeframe, year, path))

        for timeframe in sorted(grouped):
            year_parts = sorted(grouped[timeframe], key=lambda item: item[1])
            factor_parts: list[pl.DataFrame] = []
            fold_parts: list[pl.DataFrame] = []
            panel_summaries: list[dict[str, object]] = []
            for _, year, selection_path in year_parts:
                panel = _analyze_panel(
                    storage_root=self._storage_root,
                    selection_path=selection_path,
                    manager=self._manager,
                    exchange=self._exchange,
                    market=self._market,
                    engine=self._engine,
                    timeframe=timeframe,
                    year=year,
                    logger=self._logger,
                )
                factor_parts.append(panel.factor)
                fold_parts.append(panel.fold)
                panel_summaries.append(panel.summary)

            factor_frame = _sort_frame(_concat(factor_parts), FACTOR_REPORT_COLUMNS)
            fold_frame = _sort_frame(_concat(fold_parts), FOLD_REPORT_COLUMNS)
            summary_frame = _sort_frame(
                pl.DataFrame(panel_summaries) if panel_summaries else _empty(SUMMARY_COLUMNS),
                SUMMARY_COLUMNS,
            )
            factor_frames[timeframe] = factor_frame
            fold_frames[timeframe] = fold_frame
            summary_frames[timeframe] = summary_frame
            summary_rows.extend(panel_summaries)

            factor_path = self._output_root / f"factor_stability_{timeframe}{FILE_EXTENSION_CSV}"
            fold_path = (
                self._output_root / f"factor_stability_{timeframe}_folds{FILE_EXTENSION_CSV}"
            )
            summary_path = (
                self._output_root / f"factor_stability_{timeframe}_summary{FILE_EXTENSION_CSV}"
            )
            _write_csv(factor_frame, factor_path)
            _write_csv(fold_frame, fold_path)
            _write_csv(summary_frame, summary_path)
            paths[f"factor_{timeframe}"] = factor_path
            paths[f"folds_{timeframe}"] = fold_path
            paths[f"summary_{timeframe}"] = summary_path
            if panel_summaries:
                verdicts[timeframe] = str(panel_summaries[0]["verdict"])
            else:
                verdicts[timeframe] = VERDICT_INSUFFICIENT_DATA

        cross_timeframe, global_summary = _build_cross_and_global(summary_rows)
        cross_path = self._output_root / f"factor_stability_cross_timeframe{FILE_EXTENSION_CSV}"
        global_path = self._output_root / f"factor_stability_global{FILE_EXTENSION_CSV}"
        _write_csv(cross_timeframe, cross_path)
        _write_csv(global_summary, global_path)
        paths["cross_timeframe"] = cross_path
        paths["global"] = global_path

        hashes_after = hash_watched_production_ledgers(self._storage_root)
        unchanged = hashes_before == hashes_after
        if not unchanged:
            raise ReportingValidationError(
                "production ledgers changed during factor stability selection review",
                error_code=_ERROR_LEDGER_MUTATION,
                details={
                    "before_count": len(hashes_before),
                    "after_count": len(hashes_after),
                },
            )

        self._logger.info(
            "Factor stability selection review complete timeframes=%s unchanged=%s",
            sorted(factor_frames),
            unchanged,
        )
        return FactorStabilitySelectionResult(
            factor_frames=factor_frames,
            fold_frames=fold_frames,
            summary_frames=summary_frames,
            cross_timeframe=cross_timeframe,
            global_summary=global_summary,
            paths=paths,
            verdicts=verdicts,
            production_ledgers_unchanged=unchanged,
            hashes_before=hashes_before,
            hashes_after=hashes_after,
        )


def classify_factor_stability(
    *,
    oriented_fold_ics: Sequence[float | None],
    mean_oriented_oos: float | None,
    oriented_training_ic: float | None,
) -> str:
    """Classify a factor from oriented fold ICs and training orientation IC.

    Rules (in order):
        1. ``INSUFFICIENT_DATA`` when no finite oriented fold ICs or mean is
           ``None``.
        2. ``ORIENTATION_INSUFFICIENT`` when training IC is non-negative, mean
           OOS is negative, and every finite fold IC is negative.
        3. ``STABLE_POSITIVE`` when every finite fold IC is positive and mean
           is positive.
        4. ``STABLE_NEGATIVE`` when every finite fold IC is negative and mean
           is negative.
        5. ``MIXED`` when both signs are present among finite fold ICs.
        6. Otherwise classify by the sign of the mean.
    """
    finite = [_as_float(value) for value in oriented_fold_ics]
    finite = [value for value in finite if value is not None and math.isfinite(value)]
    mean_value = _as_float(mean_oriented_oos)
    if not finite or mean_value is None or not math.isfinite(mean_value):
        return FACTOR_CLASS_INSUFFICIENT_DATA

    training = _as_float(oriented_training_ic)
    all_negative = all(value < 0.0 for value in finite)
    all_positive = all(value > 0.0 for value in finite)
    if (
        training is not None
        and math.isfinite(training)
        and training >= 0.0
        and mean_value < 0.0
        and all_negative
    ):
        return FACTOR_CLASS_ORIENTATION_INSUFFICIENT
    if all_positive and mean_value > 0.0:
        return FACTOR_CLASS_STABLE_POSITIVE
    if all_negative and mean_value < 0.0:
        return FACTOR_CLASS_STABLE_NEGATIVE
    has_pos = any(value > 0.0 for value in finite)
    has_neg = any(value < 0.0 for value in finite)
    if has_pos and has_neg:
        return FACTOR_CLASS_MIXED
    if mean_value > 0.0:
        return FACTOR_CLASS_STABLE_POSITIVE
    if mean_value < 0.0:
        return FACTOR_CLASS_STABLE_NEGATIVE
    return FACTOR_CLASS_MIXED


def classify_global_status(
    *,
    selected_factors: int,
    tested_factors: int,
    fold_count: int,
    oriented_positive_factor_ratio: float | None,
    stable_positive_factor_count: int,
    mixed_factor_count: int,
    stable_negative_factor_count: int,
    orientation_insufficient_factor_count: int,
    insufficient_data_factor_count: int,
    train_positive_oos_negative_count: int,
    pcv_oriented_oos_ic: float | None,
    wf_oriented_oos_ic: float | None,
) -> str:
    """Classify timeframe-level status from aggregated selection evidence."""
    if tested_factors <= 0 or selected_factors <= 0:
        return GLOBAL_STATUS_INSUFFICIENT_DATA
    evaluated = (
        stable_positive_factor_count
        + mixed_factor_count
        + stable_negative_factor_count
        + orientation_insufficient_factor_count
    )
    if evaluated <= 0 and insufficient_data_factor_count >= selected_factors:
        return GLOBAL_STATUS_INSUFFICIENT_DATA
    if fold_count <= 0 and wf_oriented_oos_ic is None and pcv_oriented_oos_ic is None:
        return GLOBAL_STATUS_INSUFFICIENT_DATA
    if (
        orientation_insufficient_factor_count > 0
        and orientation_insufficient_factor_count >= stable_positive_factor_count
        and orientation_insufficient_factor_count >= mixed_factor_count
        and (pcv_oriented_oos_ic is None or pcv_oriented_oos_ic <= 0.0)
        and (wf_oriented_oos_ic is None or wf_oriented_oos_ic <= 0.0)
    ):
        return GLOBAL_STATUS_ORIENTATION_INSUFFICIENT
    if selected_factors > 0 and train_positive_oos_negative_count > (selected_factors / 2.0):
        return GLOBAL_STATUS_SELECTION_DEGRADATION
    positive_ratio = oriented_positive_factor_ratio
    oriented_ok = (pcv_oriented_oos_ic is not None and pcv_oriented_oos_ic > 0.0) or (
        pcv_oriented_oos_ic is None and wf_oriented_oos_ic is not None and wf_oriented_oos_ic > 0.0
    )
    predominantly_positive = (
        positive_ratio is not None
        and positive_ratio >= 0.5
        and stable_negative_factor_count == 0
        and (
            stable_positive_factor_count > stable_negative_factor_count
            or (
                mixed_factor_count > 0
                and positive_ratio >= 0.9
                and orientation_insufficient_factor_count == 0
            )
        )
    )
    if (
        predominantly_positive
        and oriented_ok
        and (wf_oriented_oos_ic is None or wf_oriented_oos_ic > 0.0)
    ):
        return GLOBAL_STATUS_STABLE_SIGNAL
    return GLOBAL_STATUS_MIXED_STABILITY


def classify_verdict(
    *,
    status: str,
    wf_oriented_oos_ic: float | None,
    pcv_oriented_oos_ic: float | None,
    fold_count: int,
    negative_fold_count: int,
    train_positive_oos_negative_count: int,
    selected_factors: int,
    redundancy_status: str,
    redundant_group_count: int,
    oriented_positive_factor_ratio: float | None,
) -> str:
    """Classify an evidence-based diagnostic verdict without forcing.

    Preference order follows the selection-stability review contract:
    timeframe weakness, selection overfitting, orientation insufficiency,
    then redundancy only when TRAIN-boundary analysis is available.
    """
    if status == GLOBAL_STATUS_INSUFFICIENT_DATA or fold_count <= 0 or selected_factors <= 0:
        return VERDICT_INSUFFICIENT_DATA

    majority_negative_folds = fold_count > 0 and negative_fold_count > (fold_count / 2.0)
    wf_nonpos = wf_oriented_oos_ic is not None and wf_oriented_oos_ic <= 0.0
    pcv_nonpos = pcv_oriented_oos_ic is not None and pcv_oriented_oos_ic <= 0.0
    if wf_nonpos and pcv_nonpos and majority_negative_folds:
        return VERDICT_TIMEFRAME_SIGNAL_WEAKNESS

    if selected_factors > 0 and train_positive_oos_negative_count > (selected_factors / 2.0):
        return VERDICT_SELECTION_OVERFITTING

    if status == GLOBAL_STATUS_ORIENTATION_INSUFFICIENT:
        return VERDICT_FACTOR_ORIENTATION_INSUFFICIENT

    weak_positive = (
        oriented_positive_factor_ratio is not None and oriented_positive_factor_ratio < 0.5
    )
    if (
        redundancy_status != REDUNDANCY_ANALYSIS_UNAVAILABLE
        and redundant_group_count > 0
        and weak_positive
    ):
        return VERDICT_FACTOR_REDUNDANCY

    if status == GLOBAL_STATUS_STABLE_SIGNAL:
        return VERDICT_STABLE_SIGNAL
    if status == GLOBAL_STATUS_SELECTION_DEGRADATION:
        return VERDICT_SELECTION_OVERFITTING
    if status == GLOBAL_STATUS_ORIENTATION_INSUFFICIENT:
        return VERDICT_FACTOR_ORIENTATION_INSUFFICIENT
    if wf_nonpos and pcv_nonpos:
        return VERDICT_TIMEFRAME_SIGNAL_WEAKNESS
    if wf_oriented_oos_ic is not None and wf_oriented_oos_ic > 0.0:
        if oriented_positive_factor_ratio is not None and oriented_positive_factor_ratio >= 0.5:
            return VERDICT_STABLE_SIGNAL
        return VERDICT_SELECTION_OVERFITTING
    if pcv_oriented_oos_ic is None and wf_oriented_oos_ic is None:
        return VERDICT_INSUFFICIENT_DATA
    return VERDICT_TIMEFRAME_SIGNAL_WEAKNESS


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


def _is_forbidden_module(module: str) -> bool:
    return any(
        module == banned or module.startswith(f"{banned}.") for banned in _FORBIDDEN_IMPORT_MODULES
    )


def _analyze_panel(
    *,
    storage_root: Path,
    selection_path: Path,
    manager: str,
    exchange: str,
    market: str,
    engine: str,
    timeframe: str,
    year: int,
    logger: logging.Logger,
) -> _PanelAnalysis:
    selection = pl.read_parquet(selection_path)
    if "selected" in selection.columns:
        selected = selection.filter(pl.col("selected"))
    else:
        selected = selection
    tested_factors = int(selection.height)
    selected_factors = int(selected.height)

    pcv_obs = _load_optional_parquet(
        storage_root
        / STORAGE_DIR_PURGED_CV_EVALUATION
        / manager
        / exchange
        / market
        / timeframe
        / f"{year}.parquet"
    )
    wf_obs = _load_optional_parquet(
        storage_root
        / STORAGE_DIR_WALK_FORWARD_EVALUATION
        / manager
        / exchange
        / market
        / timeframe
        / f"{year}.parquet"
    )
    ledger = _load_optional_parquet(
        storage_root
        / STORAGE_DIR_PURGED_CV
        / manager
        / exchange
        / market
        / timeframe
        / f"{year}.parquet"
    )

    pcv_factor_fold = _compute_factor_fold_ics(pcv_obs, engine=engine)
    wf_factor_fold = _compute_factor_fold_ics(wf_obs, engine=engine)
    pcv_fold_agg = _compute_fold_aggregate_ics(pcv_obs, engine=engine)

    factor_frame = _build_factor_report(
        selected=selected,
        pcv_factor_fold=pcv_factor_fold,
        wf_factor_fold=wf_factor_fold,
        timeframe=timeframe,
        year=year,
    )
    # Fold table remains Purged-CV-first (five-fold contract). Empty PCV
    # observation artifacts yield null fold ICs rather than substituting WF folds.
    fold_frame = _build_fold_report(
        pcv_factor_fold=pcv_factor_fold,
        pcv_fold_agg=pcv_fold_agg,
        ledger=ledger,
        timeframe=timeframe,
        year=year,
    )
    redundancy = _analyze_redundancy(pcv_obs, selected=selected, engine=engine)
    summary = _build_summary_row(
        factor_frame=factor_frame,
        fold_frame=fold_frame,
        selected_factors=selected_factors,
        tested_factors=tested_factors,
        pcv_factor_fold=pcv_factor_fold,
        wf_factor_fold=wf_factor_fold,
        redundancy=redundancy,
        timeframe=timeframe,
        year=year,
    )
    logger.debug(
        "Analyzed panel timeframe=%s year=%s selected=%s tested=%s verdict=%s",
        timeframe,
        year,
        selected_factors,
        tested_factors,
        summary["verdict"],
    )
    return _PanelAnalysis(factor=factor_frame, fold=fold_frame, summary=summary)


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
    }
    if observations is None or observations.height == 0:
        return pl.DataFrame(schema=schema)
    working = observations
    if "engine" in working.columns:
        working = working.filter(pl.col("engine") == engine)
    if "selected" in working.columns:
        working = working.filter(pl.col("selected"))
    if "partition" in working.columns:
        working = working.filter(pl.col("partition") == _PARTITION_OOS)
    required = {_FACTOR_VALUE, _TARGET, "factor_name", "factor_version", "fold_id"}
    if working.height == 0 or not required.issubset(working.columns):
        return pl.DataFrame(schema=schema)
    if "selected_direction" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Int32).alias("selected_direction"))
    if "selection_ic" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Float64).alias("selection_ic"))
    if "orientation_policy" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.String).alias("orientation_policy"))
    oriented = working.with_columns(
        (pl.col(_FACTOR_VALUE) * pl.col("selected_direction").cast(pl.Float64)).alias(
            _ORIENTED_ALIAS
        )
    )
    aggregated = (
        oriented.group_by(["factor_name", "factor_version", "fold_id"], maintain_order=True)
        .agg(
            [
                pl.corr(_FACTOR_VALUE, _TARGET, method="spearman").alias("raw_oos_ic"),
                pl.corr(_ORIENTED_ALIAS, _TARGET, method="spearman").alias("oriented_oos_ic"),
                pl.col("selected_direction").first().alias("selected_direction"),
                pl.col("selection_ic").first().alias("selection_ic"),
                pl.col("orientation_policy").first().alias("orientation_policy"),
            ]
        )
        .sort(["factor_name", "factor_version", "fold_id"])
    )
    return aggregated.select(list(schema))


def _compute_fold_aggregate_ics(observations: pl.DataFrame | None, *, engine: str) -> pl.DataFrame:
    schema = {
        "fold_id": pl.Int32,
        "raw_oos_ic": pl.Float64,
        "oriented_oos_ic": pl.Float64,
        "oos_rows": pl.Int64,
    }
    if observations is None or observations.height == 0:
        return pl.DataFrame(schema=schema)
    working = observations
    if "engine" in working.columns:
        working = working.filter(pl.col("engine") == engine)
    if "selected" in working.columns:
        working = working.filter(pl.col("selected"))
    if "partition" in working.columns:
        working = working.filter(pl.col("partition") == _PARTITION_OOS)
    required = {_FACTOR_VALUE, _TARGET, "fold_id"}
    if working.height == 0 or not required.issubset(working.columns):
        return pl.DataFrame(schema=schema)
    if "selected_direction" not in working.columns:
        working = working.with_columns(pl.lit(None, dtype=pl.Int32).alias("selected_direction"))
    oriented = working.with_columns(
        (pl.col(_FACTOR_VALUE) * pl.col("selected_direction").cast(pl.Float64)).alias(
            _ORIENTED_ALIAS
        )
    )
    return (
        oriented.group_by("fold_id", maintain_order=True)
        .agg(
            [
                pl.len().alias("oos_rows"),
                pl.corr(_FACTOR_VALUE, _TARGET, method="spearman").alias("raw_oos_ic"),
                pl.corr(_ORIENTED_ALIAS, _TARGET, method="spearman").alias("oriented_oos_ic"),
            ]
        )
        .sort("fold_id")
        .select(list(schema))
    )


def _build_factor_report(
    *,
    selected: pl.DataFrame,
    pcv_factor_fold: pl.DataFrame,
    wf_factor_fold: pl.DataFrame,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    if selected.height == 0:
        return _empty(FACTOR_REPORT_COLUMNS)

    rows: list[dict[str, object]] = []
    selected_sorted = selected.sort(["factor_name", "factor_version"])
    for item in selected_sorted.iter_rows(named=True):
        factor_name = str(item["factor_name"])
        factor_version = str(item["factor_version"])
        selection_ic = _as_float(item.get("selection_ic"))
        direction = _as_int(item.get("selected_direction"))
        oriented_training = None
        if selection_ic is not None and direction is not None:
            oriented_training = selection_ic * float(direction)

        pcv_folds = pcv_factor_fold.filter(
            (pl.col("factor_name") == factor_name) & (pl.col("factor_version") == factor_version)
        ).sort("fold_id")
        wf_folds = wf_factor_fold.filter(
            (pl.col("factor_name") == factor_name) & (pl.col("factor_version") == factor_version)
        ).sort("fold_id")

        # Prefer Purged-CV fold ICs when finite values exist; otherwise fall back
        # to Walk-Forward fold ICs so cross-timeframe review remains evidence-based
        # when PCV evaluation observations do not yield IC estimates.
        pcv_oriented_probe = [
            value
            for value in (_as_float(v) for v in pcv_folds["oriented_oos_ic"].to_list())
            if value is not None and math.isfinite(value)
        ]
        wf_oriented_probe = [
            value
            for value in (_as_float(v) for v in wf_folds["oriented_oos_ic"].to_list())
            if value is not None and math.isfinite(value)
        ]
        if pcv_oriented_probe:
            fold_source = pcv_folds
        elif wf_oriented_probe:
            fold_source = wf_folds
        else:
            fold_source = pcv_folds.clear() if pcv_folds.height else pcv_folds

        raw_by_fold = [_as_float(value) for value in fold_source["raw_oos_ic"].to_list()]
        oriented_by_fold = [_as_float(value) for value in fold_source["oriented_oos_ic"].to_list()]
        finite_oriented = [
            value for value in oriented_by_fold if value is not None and math.isfinite(value)
        ]
        mean_raw = _mean(raw_by_fold)
        mean_oriented = _mean(oriented_by_fold)
        std_oriented = _std(finite_oriented)
        positive = sum(1 for value in finite_oriented if value > 0.0)
        negative = sum(1 for value in finite_oriented if value < 0.0)
        total_signed = positive + negative
        fraction_positive = (positive / total_signed) if total_signed > 0 else None
        min_oriented = min(finite_oriented) if finite_oriented else None
        max_oriented = max(finite_oriented) if finite_oriented else None

        wf_raw_mean = _mean([_as_float(value) for value in wf_folds["raw_oos_ic"].to_list()])
        wf_oriented_mean = _mean(
            [_as_float(value) for value in wf_folds["oriented_oos_ic"].to_list()]
        )
        wf_status = _status_from_mean(wf_oriented_mean)

        degradation = None
        if oriented_training is not None and mean_oriented is not None:
            degradation = oriented_training - mean_oriented
        abs_degradation = abs(degradation) if degradation is not None else None
        sign_preserved = _sign_preserved(oriented_training, mean_oriented)
        train_pos_oos_neg = bool(
            oriented_training is not None
            and mean_oriented is not None
            and oriented_training > 0.0
            and mean_oriented < 0.0
        )
        stability = classify_factor_stability(
            oriented_fold_ics=oriented_by_fold,
            mean_oriented_oos=mean_oriented,
            oriented_training_ic=oriented_training,
        )
        rows.append(
            {
                "timeframe": timeframe,
                "year": year,
                "factor_name": factor_name,
                "factor_version": factor_version,
                "factor_category": item.get("factor_category"),
                "selected_direction": direction,
                "orientation_policy": item.get("orientation_policy"),
                "selection_ic": selection_ic,
                "oriented_training_ic": oriented_training,
                "raw_oos_ic_by_fold": _format_pipe_list(raw_by_fold),
                "oriented_oos_ic_by_fold": _format_pipe_list(oriented_by_fold),
                "mean_raw_oos_ic": mean_raw,
                "mean_oriented_oos_ic": mean_oriented,
                "std_oriented_oos_ic": std_oriented,
                "positive_oriented_folds": positive,
                "negative_oriented_folds": negative,
                "fraction_positive_oriented_folds": fraction_positive,
                "min_oriented_fold_ic": min_oriented,
                "max_oriented_fold_ic": max_oriented,
                "wf_mean_raw_oos_ic": wf_raw_mean,
                "wf_mean_oriented_oos_ic": wf_oriented_mean,
                "wf_factor_status": wf_status,
                "ic_degradation": degradation,
                "abs_ic_degradation": abs_degradation,
                "sign_preserved": sign_preserved,
                "train_positive_oos_negative": train_pos_oos_neg,
                "stability_class": stability,
            }
        )
    return pl.DataFrame(rows).select(list(FACTOR_REPORT_COLUMNS))


def _build_fold_report(
    *,
    pcv_factor_fold: pl.DataFrame,
    pcv_fold_agg: pl.DataFrame,
    ledger: pl.DataFrame | None,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    fold_ids: list[int] = []
    if pcv_fold_agg.height > 0:
        fold_ids.extend(int(value) for value in pcv_fold_agg["fold_id"].to_list())
    if ledger is not None and ledger.height > 0 and "fold_id" in ledger.columns:
        fold_ids.extend(int(value) for value in ledger["fold_id"].to_list())
    unique_folds = sorted(set(fold_ids))
    if not unique_folds:
        return _empty(FOLD_REPORT_COLUMNS)

    ledger_lookup: dict[int, dict[str, object]] = {}
    if ledger is not None and ledger.height > 0:
        for row in ledger.sort("fold_id").iter_rows(named=True):
            ledger_lookup[int(row["fold_id"])] = row

    agg_lookup = {int(row["fold_id"]): row for row in pcv_fold_agg.iter_rows(named=True)}
    rows: list[dict[str, object]] = []
    for fold_id in unique_folds:
        agg = agg_lookup.get(fold_id, {})
        ledger_row = ledger_lookup.get(fold_id, {})
        factor_fold = pcv_factor_fold.filter(pl.col("fold_id") == fold_id)
        oriented_values = [_as_float(value) for value in factor_fold["oriented_oos_ic"].to_list()]
        raw_values = [_as_float(value) for value in factor_fold["raw_oos_ic"].to_list()]
        finite = [value for value in oriented_values if value is not None and math.isfinite(value)]
        positive = sum(1 for value in finite if value > 0.0)
        negative = sum(1 for value in finite if value < 0.0)
        total = positive + negative
        percentage = (100.0 * positive / total) if total > 0 else None
        # Prefer mean of selected-factor fold ICs (matches orientation diagnostics).
        # Fall back to pooled fold aggregate only when factor-level ICs are absent.
        oriented = _mean(oriented_values)
        raw = _mean(raw_values)
        if oriented is None:
            oriented = _as_float(agg.get("oriented_oos_ic"))
        if raw is None:
            raw = _as_float(agg.get("raw_oos_ic"))
        if total == 0 or oriented is None:
            aggregate_status = FACTOR_CLASS_INSUFFICIENT_DATA
        elif positive > 0 and negative > 0:
            aggregate_status = FACTOR_CLASS_MIXED
        elif positive == total and oriented > 0.0:
            aggregate_status = FACTOR_CLASS_STABLE_POSITIVE
        elif negative == total and oriented < 0.0:
            aggregate_status = FACTOR_CLASS_STABLE_NEGATIVE
        else:
            aggregate_status = _status_from_mean(oriented)
        rows.append(
            {
                "timeframe": timeframe,
                "year": year,
                "fold_id": fold_id,
                "train_rows": _as_int(ledger_row.get("train_rows")),
                "oos_rows": (
                    _as_int(ledger_row.get("test_rows"))
                    if ledger_row.get("test_rows") is not None
                    else _as_int(agg.get("oos_rows"))
                ),
                "raw_oos_ic": raw,
                "oriented_oos_ic": oriented,
                "positive_oriented_factor_count": positive,
                "negative_oriented_factor_count": negative,
                "positive_factor_percentage": percentage,
                "aggregate_status": aggregate_status,
            }
        )
    return pl.DataFrame(rows).select(list(FOLD_REPORT_COLUMNS))


def _analyze_redundancy(
    observations: pl.DataFrame | None,
    *,
    selected: pl.DataFrame,
    engine: str,
) -> dict[str, object]:
    unavailable: dict[str, object] = {
        "redundancy_status": REDUNDANCY_ANALYSIS_UNAVAILABLE,
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
                [
                    pl.col(left).alias("left"),
                    pl.col(right).alias("right"),
                ]
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


def _build_summary_row(
    *,
    factor_frame: pl.DataFrame,
    fold_frame: pl.DataFrame,
    selected_factors: int,
    tested_factors: int,
    pcv_factor_fold: pl.DataFrame,
    wf_factor_fold: pl.DataFrame,
    redundancy: Mapping[str, object],
    timeframe: str,
    year: int,
) -> dict[str, object]:
    selection_ratio = (
        float(selected_factors) / float(tested_factors) if tested_factors > 0 else None
    )
    degradations = (
        [_as_float(value) for value in factor_frame["ic_degradation"].to_list()]
        if factor_frame.height
        else []
    )
    finite_degradations = [
        value for value in degradations if value is not None and math.isfinite(value)
    ]
    train_pos_oos_neg = (
        int(factor_frame.filter(pl.col("train_positive_oos_negative")).height)
        if factor_frame.height
        else 0
    )
    class_counts = {
        FACTOR_CLASS_STABLE_POSITIVE: 0,
        FACTOR_CLASS_MIXED: 0,
        FACTOR_CLASS_STABLE_NEGATIVE: 0,
        FACTOR_CLASS_ORIENTATION_INSUFFICIENT: 0,
        FACTOR_CLASS_INSUFFICIENT_DATA: 0,
    }
    if factor_frame.height:
        for value in factor_frame["stability_class"].to_list():
            key = str(value)
            if key in class_counts:
                class_counts[key] += 1

    oriented_means = (
        [_as_float(value) for value in factor_frame["mean_oriented_oos_ic"].to_list()]
        if factor_frame.height
        else []
    )
    positive_factors = sum(
        1 for value in oriented_means if value is not None and math.isfinite(value) and value > 0.0
    )
    evaluated_factors = sum(
        1 for value in oriented_means if value is not None and math.isfinite(value)
    )
    oriented_positive_ratio = (
        float(positive_factors) / float(evaluated_factors) if evaluated_factors > 0 else None
    )

    # Panel IC means use factor-fold metric rows (matches orientation diagnostics).
    pcv_raw = (
        _mean([_as_float(value) for value in pcv_factor_fold["raw_oos_ic"].to_list()])
        if pcv_factor_fold.height
        else None
    )
    pcv_oriented = (
        _mean([_as_float(value) for value in pcv_factor_fold["oriented_oos_ic"].to_list()])
        if pcv_factor_fold.height
        else None
    )
    wf_raw = (
        _mean([_as_float(value) for value in wf_factor_fold["raw_oos_ic"].to_list()])
        if wf_factor_fold.height
        else None
    )
    wf_oriented = (
        _mean([_as_float(value) for value in wf_factor_fold["oriented_oos_ic"].to_list()])
        if wf_factor_fold.height
        else None
    )

    fold_count = int(fold_frame.height)
    negative_fold_count = (
        int(fold_frame.filter(pl.col("oriented_oos_ic") < 0.0).height) if fold_count else 0
    )
    fold_concentration = float(negative_fold_count) / float(fold_count) if fold_count > 0 else None

    status = classify_global_status(
        selected_factors=selected_factors,
        tested_factors=tested_factors,
        fold_count=fold_count,
        oriented_positive_factor_ratio=oriented_positive_ratio,
        stable_positive_factor_count=class_counts[FACTOR_CLASS_STABLE_POSITIVE],
        mixed_factor_count=class_counts[FACTOR_CLASS_MIXED],
        stable_negative_factor_count=class_counts[FACTOR_CLASS_STABLE_NEGATIVE],
        orientation_insufficient_factor_count=class_counts[FACTOR_CLASS_ORIENTATION_INSUFFICIENT],
        insufficient_data_factor_count=class_counts[FACTOR_CLASS_INSUFFICIENT_DATA],
        train_positive_oos_negative_count=train_pos_oos_neg,
        pcv_oriented_oos_ic=pcv_oriented,
        wf_oriented_oos_ic=wf_oriented,
    )
    verdict = classify_verdict(
        status=status,
        wf_oriented_oos_ic=wf_oriented,
        pcv_oriented_oos_ic=pcv_oriented,
        fold_count=fold_count,
        negative_fold_count=negative_fold_count,
        train_positive_oos_negative_count=train_pos_oos_neg,
        selected_factors=selected_factors,
        redundancy_status=str(redundancy["redundancy_status"]),
        redundant_group_count=_as_int(redundancy.get("redundant_group_count")) or 0,
        oriented_positive_factor_ratio=oriented_positive_ratio,
    )
    return {
        "timeframe": timeframe,
        "year": year,
        "selected_factors": selected_factors,
        "tested_factors": tested_factors,
        "selection_ratio": selection_ratio,
        "wf_raw_oos_ic": wf_raw,
        "wf_oriented_oos_ic": wf_oriented,
        "pcv_raw_oos_ic": pcv_raw,
        "pcv_oriented_oos_ic": pcv_oriented,
        "oriented_positive_factor_ratio": oriented_positive_ratio,
        "stable_positive_factor_count": class_counts[FACTOR_CLASS_STABLE_POSITIVE],
        "mixed_factor_count": class_counts[FACTOR_CLASS_MIXED],
        "stable_negative_factor_count": class_counts[FACTOR_CLASS_STABLE_NEGATIVE],
        "orientation_insufficient_factor_count": class_counts[
            FACTOR_CLASS_ORIENTATION_INSUFFICIENT
        ],
        "insufficient_data_factor_count": class_counts[FACTOR_CLASS_INSUFFICIENT_DATA],
        "degradation_mean": _mean(finite_degradations),
        "degradation_median": (
            statistics.median(finite_degradations) if finite_degradations else None
        ),
        "fold_count": fold_count,
        "negative_fold_count": negative_fold_count,
        "fold_concentration": fold_concentration,
        "redundancy_status": redundancy["redundancy_status"],
        "redundancy_threshold": redundancy["redundancy_threshold"],
        "redundant_group_count": redundancy["redundant_group_count"],
        "status": status,
        "verdict": verdict,
    }


def _build_cross_and_global(
    summary_rows: Sequence[Mapping[str, object]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not summary_rows:
        return _empty(CROSS_TIMEFRAME_COLUMNS), _empty(GLOBAL_COLUMNS)

    # One row per timeframe: prefer first year (deterministic) then re-aggregate
    # counts when multiple years exist by taking the first sorted year row.
    by_timeframe: dict[str, Mapping[str, object]] = {}
    for row in sorted(
        summary_rows,
        key=lambda item: (str(item["timeframe"]), _as_int(item["year"]) or 0),
    ):
        timeframe = str(row["timeframe"])
        if timeframe not in by_timeframe:
            by_timeframe[timeframe] = row

    cross_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    for timeframe in sorted(by_timeframe):
        row = by_timeframe[timeframe]
        cross_rows.append({column: row.get(column) for column in CROSS_TIMEFRAME_COLUMNS})
        global_rows.append({column: row.get(column) for column in GLOBAL_COLUMNS})
    return (
        _sort_frame(pl.DataFrame(cross_rows), CROSS_TIMEFRAME_COLUMNS),
        _sort_frame(pl.DataFrame(global_rows), GLOBAL_COLUMNS),
    )


def _discover_factor_selection_partitions(
    storage_root: Path,
    *,
    manager: str,
    exchange: str,
    market: str,
    timeframes: Sequence[str] | None,
) -> list[tuple[str, int, Path]]:
    root = storage_root / STORAGE_DIR_FACTOR_SELECTION / manager / exchange / market
    if not root.exists():
        return []
    allow = set(timeframes) if timeframes is not None else None
    discovered: list[tuple[str, int, Path]] = []
    for timeframe_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        timeframe = timeframe_dir.name
        if allow is not None and timeframe not in allow:
            continue
        for parquet_path in sorted(timeframe_dir.glob("*.parquet")):
            try:
                year = int(parquet_path.stem)
            except ValueError:
                continue
            discovered.append((timeframe, year, parquet_path))
    return discovered


def hash_watched_production_ledgers(storage_root: Path) -> dict[str, str]:
    """SHA-256 hash Walk-Forward, Purged-CV, and Factor Selection parquet files.

    Evaluation tiers are intentionally excluded from the watched set.
    """
    hashes: dict[str, str] = {}
    for tier in _WATCHED_LEDGER_DIRS:
        root = storage_root / tier
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.parquet")):
            rel = path.relative_to(storage_root).as_posix()
            hashes[rel] = _sha256_file(path)
    return hashes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_optional_parquet(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _write_csv(frame: pl.DataFrame, path: Path) -> None:
    frame.write_csv(path, null_value="")


def _concat(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    nonempty = [frame for frame in frames if frame.height > 0]
    if not nonempty:
        return pl.DataFrame()
    return pl.concat(nonempty, how="vertical_relaxed")


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


def _format_pipe_list(values: Sequence[float | None]) -> str:
    parts: list[str] = []
    for value in values:
        if value is None or not math.isfinite(value):
            parts.append("")
        else:
            parts.append(f"{value:.12g}")
    return "|".join(parts)


def _status_from_mean(mean_value: float | None) -> str:
    value = _as_float(mean_value)
    if value is None or not math.isfinite(value):
        return FACTOR_CLASS_INSUFFICIENT_DATA
    if value > 0.0:
        return FACTOR_CLASS_STABLE_POSITIVE
    if value < 0.0:
        return FACTOR_CLASS_STABLE_NEGATIVE
    return FACTOR_CLASS_MIXED


def _sign_preserved(training: float | None, oos: float | None) -> bool | None:
    if training is None or oos is None:
        return None
    if not math.isfinite(training) or not math.isfinite(oos):
        return None
    if training == 0.0 or oos == 0.0:
        return training == oos
    return (training > 0.0) == (oos > 0.0)


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

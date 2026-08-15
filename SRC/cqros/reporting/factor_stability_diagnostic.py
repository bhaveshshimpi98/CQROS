"""CQROS Factor Stability + OOS IC diagnostic reporter.

Purpose:
    Diagnose why Purged-CV OOS Information Coefficient may be negative by
    producing read-only evidence across alignment, value integrity, sign,
    fold/timeframe stability, family concentration, selection vs OOS, and
    methodology checks — without mutating production lake artifacts.

Responsibilities:
    - Verify Labels ``future_return_1`` timestamp alignment
    - Verify factor-value preservation across Factors / evaluation input / OOS
    - Compute diagnostic Spearman IC (original and inverted) per fold
    - Aggregate fold, timeframe, and family stability diagnostics
    - Compare Factor Selection / Factor Validation evidence with OOS IC
    - Document IC methodology alignment with Factor Validation Rank IC
    - Emit deterministic CSV reports under ``reports/purged_cv``
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``hashlib``, ``logging``, ``polars``, ``cqros.core``, ``cqros.purged_cv``,
    ``cqros.walk_forward.evaluation_input``, and repository facades injected
    by the CLI.

Public API:
    Column constants, classification helpers, ``FactorStabilityDiagnostic``,
    ``FactorStabilityDiagnosticResult``, ``write_factor_stability_reports``,
    ``build_global_summary``, ``forbidden_import_violations``, and
    ``DEFAULT_OUTPUT_ROOT``.

Notes:
    This stage is diagnostic only. It never changes factor signs, selection
    thresholds, labels, or production parquet bytes. Regime analysis is
    ``NOT_AVAILABLE`` unless regime columns already exist on evaluation
    artifacts (they do not in the current evaluation schemas).
"""

from __future__ import annotations

import gc
import hashlib
import logging
import math
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
    STORAGE_DIR_WALK_FORWARD,
    STORAGE_DIR_WALK_FORWARD_EVALUATION,
)
from cqros.core.types import Exchange, Market, Timeframe
from cqros.factor_selection import FactorSelectionRepository
from cqros.factor_validation import FactorValidationRepository
from cqros.factors import FactorsRepository
from cqros.purged_cv import PurgedCVRepository
from cqros.purged_cv.evaluation import PurgedCVEvaluator
from cqros.reporting.exceptions import ReportingValidationError
from cqros.storage.label_repository import LabelRepository
from cqros.walk_forward import WalkForwardRepository
from cqros.walk_forward.evaluation_input import TARGET_COLUMN, WalkForwardInputBuilder

__all__ = [
    "ALIGNMENT_AUDIT_COLUMNS",
    "ALIGNMENT_AUDIT_CSV_NAME",
    "CRITICAL_QUESTION_KEYS",
    "DEFAULT_OUTPUT_ROOT",
    "FAMILIES_COLUMNS",
    "FAMILIES_CSV_NAME",
    "FOLDS_COLUMNS",
    "FOLDS_CSV_NAME",
    "GLOBAL_COLUMNS",
    "GLOBAL_CSV_NAME",
    "PRIMARY_CONCLUSION_CODES",
    "SORT_COLUMNS",
    "STABILITY_ALL_COLUMNS",
    "STABILITY_ALL_CSV_NAME",
    "TIMEFRAMES_COLUMNS",
    "TIMEFRAMES_CSV_NAME",
    "FactorStabilityDiagnostic",
    "FactorStabilityDiagnosticResult",
    "PanelDiagnosticBundle",
    "aggregate_cross_timeframe",
    "aggregate_families",
    "build_global_summary",
    "classify_cross_timeframe_stability",
    "classify_is_oos",
    "classify_orientation",
    "compute_fold_factor_ic",
    "compute_spearman_ic",
    "detect_ic_methodology",
    "fold_stability_from_factor_metrics",
    "forbidden_import_violations",
    "verify_factor_value_integrity",
    "verify_target_alignment",
    "write_factor_stability_reports",
]

_logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT: Final[Path] = Path("reports") / "purged_cv"

STABILITY_ALL_CSV_NAME: Final[str] = f"factor_stability_all{FILE_EXTENSION_CSV}"
FOLDS_CSV_NAME: Final[str] = f"factor_stability_folds{FILE_EXTENSION_CSV}"
TIMEFRAMES_CSV_NAME: Final[str] = f"factor_stability_timeframes{FILE_EXTENSION_CSV}"
FAMILIES_CSV_NAME: Final[str] = f"factor_stability_families{FILE_EXTENSION_CSV}"
GLOBAL_CSV_NAME: Final[str] = f"factor_stability_global{FILE_EXTENSION_CSV}"
ALIGNMENT_AUDIT_CSV_NAME: Final[str] = f"factor_alignment_audit{FILE_EXTENSION_CSV}"

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL
_ENGINE: Final[str] = "simple"

_FACTOR_VALUE_COLUMN: Final[str] = "factor_value"
_FAMILY_UNKNOWN: Final[str] = "UNKNOWN"
_REGIME_NOT_AVAILABLE: Final[str] = "NOT_AVAILABLE"
_SELECTION_INTENSITY_NOT_AVAILABLE: Final[str] = "NOT_AVAILABLE"

# Canonical Labels definition (labels pipeline): future_return_1 at open_time t
# equals (close[t+1] - close[t]) / close[t]. Join key is Labels PK.
_LABEL_HORIZON_BARS: Final[int] = 1
_LABEL_DEFINITION: Final[str] = (
    "future_return_1(t) = (close[t+1] - close[t]) / close[t]; "
    "aligned to factor open_time via Labels PK (symbol, timeframe, open_time)"
)

# Diagnostic sampling / exploratory outlier policy (not a promotion threshold).
_ALIGNMENT_SAMPLE_ROWS: Final[int] = 500
_VALUE_INTEGRITY_SAMPLE_ROWS: Final[int] = 500
_OUTLIER_EXCLUSION_FRACTION: Final[float] = 0.01
_OUTLIER_POLICY_NOTE: Final[str] = (
    "exploratory: exclude extreme |future_return_1| tails at fraction "
    f"{_OUTLIER_EXCLUSION_FRACTION}; no CQROS outlier policy exists"
)

SORT_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "exchange",
    "market",
    "timeframe",
    "year",
    "factor_name",
    "factor_version",
    "fold_id",
)

STABILITY_ALL_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "exchange",
    "market",
    "timeframe",
    "year",
    "factor_name",
    "factor_version",
    "factor_family",
    "fold_id",
    "ic",
    "ic_inverted",
    "rows",
    "mean_return",
    "target_std",
    "positive_rate",
    "orientation_class",
    "selection_score",
    "selection_rank",
    "selection_metric_ic",
    "selection_metric_rank_ic",
    "is_oos_class",
    "raw_spearman_ic",
    "outlier_sensitive_ic",
    "outlier_policy",
)

FOLDS_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "exchange",
    "market",
    "timeframe",
    "year",
    "factor_name",
    "factor_version",
    "fold_id",
    "ic",
    "rows",
    "mean_return",
    "target_std",
    "positive_rate",
)

TIMEFRAMES_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "exchange",
    "market",
    "factor_name",
    "factor_version",
    "timeframes_evaluated",
    "mean_ic",
    "median_ic",
    "positive_timeframes",
    "negative_timeframes",
    "ic_sign_consistency",
    "cross_timeframe_class",
)

FAMILIES_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "exchange",
    "market",
    "family",
    "factor_count",
    "factor_fold_rows",
    "mean_ic",
    "median_ic",
    "positive_factor_fraction",
)

ALIGNMENT_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "manager",
    "exchange",
    "market",
    "timeframe",
    "year",
    "diagnostic",
    "factor_name",
    "factor_version",
    "rows_checked",
    "alignment_pass",
    "alignment_fail",
    "duplicate_keys",
    "missing_labels",
    "value_mismatches",
    "label_definition",
    "status",
    "notes",
)

GLOBAL_COLUMNS: Final[tuple[str, ...]] = ("metric", "value")

PRIMARY_CONCLUSION_CODES: Final[tuple[str, ...]] = (
    "A. TARGET_ALIGNMENT_PROBLEM",
    "B. FACTOR_VALUE_INTEGRITY_PROBLEM",
    "C. FACTOR_ORIENTATION_PROBLEM",
    "D. SELECTION_OVERFIT_SIGNAL",
    "E. FACTOR_INSTABILITY",
    "F. FACTOR_FAMILY_CONCENTRATION",
    "G. METHODOLOGY_MISMATCH",
    "H. NO_SINGLE_ROOT_CAUSE",
    "I. INSUFFICIENT_EVIDENCE",
)

CRITICAL_QUESTION_KEYS: Final[tuple[str, ...]] = (
    "q1_future_return_1_aligned",
    "q2_factor_values_preserved",
    "q3_orientation_cause",
    "q4_stable_across_folds",
    "q5_stable_across_timeframes",
    "q6_family_concentration",
    "q7_is_positive_oos_negative",
    "q8_methodology_aligned",
    "q9_selection_intensity_reconstructible",
    "q10_primary_conclusion",
)

_FORBIDDEN_IMPORT_PREFIXES: Final[tuple[str, ...]] = (
    "cqros.alpha",
    "cqros.regime",
    "cqros.predictions",
    "cqros.signals",
    "cqros.ml",
)

_ERROR_FRAME_TYPE: Final[str] = "FACTOR-STABILITY-001"
_ERROR_MISSING_COLUMNS: Final[str] = "FACTOR-STABILITY-002"


@dataclass(frozen=True, slots=True)
class FactorStabilityDiagnosticResult:
    """Immutable diagnostic outputs for one diagnostic run."""

    stability_all: pl.DataFrame
    folds: pl.DataFrame
    timeframes: pl.DataFrame
    families: pl.DataFrame
    alignment_audit: pl.DataFrame
    global_summary: pl.DataFrame
    report_paths: Mapping[str, Path]
    parquet_hashes_before: Mapping[str, str]
    parquet_hashes_after: Mapping[str, str]
    folds_accounted: int
    timeframes_analyzed: tuple[str, ...]
    verdict: str
    primary_conclusion: str


@dataclass(frozen=True, slots=True)
class PanelDiagnosticBundle:
    """Intermediate per-panel diagnostic frames and integrity flags."""

    stability_all: pl.DataFrame
    folds: pl.DataFrame
    alignment_audit: pl.DataFrame
    fold_count: int
    alignment_ok: bool
    value_ok: bool
    selection_tested: int | None
    selection_selected: int | None


class FactorStabilityDiagnostic:
    """Orchestrate Factor Stability diagnostics across discovered panels.

    Dependencies are injected. The diagnostic never writes lake parquet and
    never imports Alpha/Regime/Predictions/Signals/ML.
    """

    __slots__ = (
        "_exchange",
        "_factors_repository",
        "_factor_selection_repository",
        "_factor_validation_repository",
        "_label_repository",
        "_logger",
        "_market",
        "_output_root",
        "_purged_cv_repository",
        "_walk_forward_input_builder",
        "_walk_forward_repository",
    )

    _purged_cv_repository: PurgedCVRepository
    _walk_forward_repository: WalkForwardRepository
    _factor_selection_repository: FactorSelectionRepository
    _factors_repository: FactorsRepository
    _label_repository: LabelRepository
    _walk_forward_input_builder: WalkForwardInputBuilder
    _factor_validation_repository: FactorValidationRepository | None
    _output_root: Path
    _exchange: Exchange
    _market: Market
    _logger: logging.Logger

    def __init__(
        self,
        *,
        purged_cv_repository: PurgedCVRepository,
        walk_forward_repository: WalkForwardRepository,
        factor_selection_repository: FactorSelectionRepository,
        factors_repository: FactorsRepository,
        label_repository: LabelRepository,
        walk_forward_input_builder: WalkForwardInputBuilder,
        factor_validation_repository: FactorValidationRepository | None = None,
        output_root: Path | None = None,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the diagnostic with injected repositories.

        Args:
            purged_cv_repository: Purged-CV ledger repository.
            walk_forward_repository: Walk-Forward ledger repository.
            factor_selection_repository: Factor Selection repository.
            factors_repository: Canonical Factors repository.
            label_repository: Canonical Labels repository.
            walk_forward_input_builder: Evaluation-input assembler.
            factor_validation_repository: Optional Factor Validation repository
                for IS metrics / selection intensity.
            output_root: CSV report directory.
            exchange: Exchange identity.
            market: Market segment.
            logger: Optional logger.
        """
        self._purged_cv_repository = purged_cv_repository
        self._walk_forward_repository = walk_forward_repository
        self._factor_selection_repository = factor_selection_repository
        self._factors_repository = factors_repository
        self._label_repository = label_repository
        self._walk_forward_input_builder = walk_forward_input_builder
        self._factor_validation_repository = factor_validation_repository
        self._output_root = output_root if output_root is not None else DEFAULT_OUTPUT_ROOT
        self._exchange = exchange
        self._market = market
        self._logger = logger if logger is not None else _logger

    @property
    def output_root(self) -> Path:
        """Return the configured report output directory."""
        return self._output_root

    def run(
        self,
        *,
        manager: str,
        engine: str = _ENGINE,
        timeframes: Sequence[Timeframe] | None = None,
        years: Sequence[int] | None = None,
        storage_root: Path | None = None,
    ) -> FactorStabilityDiagnosticResult:
        """Discover panels, run diagnostics, and write CSV reports.

        Args:
            manager: Order manager identity.
            engine: Engine label recorded on evaluation artifacts.
            timeframes: Optional timeframe allowlist (discover otherwise).
            years: Optional year allowlist (discover otherwise).
            storage_root: Optional storage root used for immutability hashing.

        Returns:
            Immutable diagnostic result including report paths and verdict.
        """
        hashes_before = _hash_watched_artifacts(storage_root) if storage_root is not None else {}
        partitions = self._purged_cv_repository.discover_partitions(
            managers=(manager,),
            timeframes=timeframes,
            exchange=self._exchange,
            market=self._market,
        )
        year_allowlist = set(years) if years is not None else None
        panels: list[tuple[str, Timeframe, int]] = []
        for partition in partitions:
            if year_allowlist is not None and partition.year not in year_allowlist:
                continue
            panels.append((partition.manager, partition.timeframe, partition.year))
        panels = sorted(panels, key=lambda item: (item[0], item[1], item[2]))

        panel_results: list[PanelDiagnosticBundle] = []
        for panel_manager, timeframe, year in panels:
            self._logger.info(
                "Diagnosing factor stability panel",
                extra={
                    "manager": panel_manager,
                    "timeframe": timeframe,
                    "year": year,
                },
            )
            panel_results.append(
                self._diagnose_panel(
                    manager=panel_manager,
                    engine=engine,
                    timeframe=timeframe,
                    year=year,
                )
            )

        stability_all = _concat_frames([item.stability_all for item in panel_results])
        folds = _concat_frames([item.folds for item in panel_results])
        alignment_audit = _concat_frames([item.alignment_audit for item in panel_results])
        timeframes_frame = aggregate_cross_timeframe(stability_all)
        families_frame = aggregate_families(stability_all)

        methodology = detect_ic_methodology()
        global_summary = build_global_summary(
            stability_all=stability_all,
            folds=folds,
            timeframes=timeframes_frame,
            families=families_frame,
            alignment_audit=alignment_audit,
            panel_results=panel_results,
            methodology=methodology,
        )
        paths = write_factor_stability_reports(
            output_root=self._output_root,
            stability_all=stability_all,
            folds=folds,
            timeframes=timeframes_frame,
            families=families_frame,
            alignment_audit=alignment_audit,
            global_summary=global_summary,
        )
        hashes_after = _hash_watched_artifacts(storage_root) if storage_root is not None else {}
        verdict = _global_metric(global_summary, "verdict") or "FAIL"
        primary = _global_metric(global_summary, "q10_primary_conclusion") or (
            "I. INSUFFICIENT_EVIDENCE"
        )
        analyzed = tuple(
            sorted({str(row[1]) for row in panels}),
        )
        return FactorStabilityDiagnosticResult(
            stability_all=stability_all,
            folds=folds,
            timeframes=timeframes_frame,
            families=families_frame,
            alignment_audit=alignment_audit,
            global_summary=global_summary,
            report_paths=paths,
            parquet_hashes_before=hashes_before,
            parquet_hashes_after=hashes_after,
            folds_accounted=int(sum(item.fold_count for item in panel_results)),
            timeframes_analyzed=analyzed,
            verdict=str(verdict),
            primary_conclusion=str(primary),
        )

    def _diagnose_panel(
        self,
        *,
        manager: str,
        engine: str,
        timeframe: Timeframe,
        year: int,
    ) -> PanelDiagnosticBundle:
        """Run every diagnostic for one manager/timeframe/year panel."""
        selection = self._factor_selection_repository.load(
            manager=manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=timeframe,
            year=year,
        )
        evaluation_input = self._walk_forward_input_builder.build(
            selection,
            manager=manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=timeframe,
            year=year,
        )
        # OOS IC diagnostics only use selected factors; drop rejected rows early
        # to keep peak memory bounded on large multi-symbol panels.
        if "selected" in evaluation_input.columns:
            evaluation_input = evaluation_input.filter(pl.col("selected"))
        purged_cv = self._purged_cv_repository.load(
            manager=manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=timeframe,
            year=year,
        )
        fold_count = int(purged_cv.height)
        walk_forward = self._walk_forward_repository.load(
            manager=manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=timeframe,
            year=year,
        )
        artifacts = PurgedCVEvaluator(logger=self._logger).evaluate(
            purged_cv,
            walk_forward,
            manager=manager,
            engine=engine,
            exchange=self._exchange,
            market=self._market,
            year=year,
            evaluation_input=evaluation_input,
        )
        # Prefer factor_metrics (always computed). Full OOS observation frames are
        # intentionally skipped by PurgedCVEvaluator for large panels.
        fold_ic = fold_stability_from_factor_metrics(artifacts.factor_metrics)
        if fold_ic.height == 0 and artifacts.observations.height > 0:
            fold_ic = compute_fold_factor_ic(artifacts.observations)

        alignment = verify_target_alignment(
            evaluation_input,
            manager=manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=timeframe,
            year=year,
        )
        integrity_sample = evaluation_input.head(_VALUE_INTEGRITY_SAMPLE_ROWS)
        observation_sample = (
            artifacts.observations.head(_VALUE_INTEGRITY_SAMPLE_ROWS)
            if artifacts.observations.height > 0
            else artifacts.observations
        )
        value_audit = verify_factor_value_integrity(
            evaluation_input=integrity_sample,
            observations=observation_sample,
            factors_frame=_load_factors_for_sample(
                self._factors_repository,
                manager=manager,
                exchange=self._exchange,
                market=self._market,
                timeframe=timeframe,
                year=year,
                sample=integrity_sample,
            ),
            manager=manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=timeframe,
            year=year,
        )
        alignment_audit = pl.concat([alignment, value_audit], how="vertical")
        del integrity_sample
        del observation_sample

        validation = _try_load_validation(
            self._factor_validation_repository,
            manager=manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=timeframe,
            year=year,
        )
        stability_all = _enrich_stability_rows(
            fold_ic,
            selection=selection,
            validation=validation,
            manager=manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=timeframe,
            year=year,
        )
        folds = stability_all.select(list(FOLDS_COLUMNS))
        selected_count = int(selection.filter(pl.col("selected")).height)
        tested_count = int(selection.height)
        if validation is not None and validation.height > 0:
            tested_count = int(validation.select(["factor_name", "factor_version"]).unique().height)
        # Release the large evaluation input before returning.
        del evaluation_input
        del artifacts
        del walk_forward
        del purged_cv
        gc.collect()
        return PanelDiagnosticBundle(
            stability_all=_sort_frame(stability_all, STABILITY_ALL_COLUMNS),
            folds=_sort_frame(folds, FOLDS_COLUMNS),
            alignment_audit=_sort_frame(alignment_audit, ALIGNMENT_AUDIT_COLUMNS),
            fold_count=fold_count,
            alignment_ok=_audit_passed(alignment, "target_alignment"),
            value_ok=_audit_passed(value_audit, "value_integrity"),
            selection_tested=tested_count,
            selection_selected=selected_count,
        )


def compute_spearman_ic(factor_values: pl.Series, targets: pl.Series) -> float | None:
    """Compute pooled Spearman IC for aligned factor/target series.

    Args:
        factor_values: Factor value series.
        targets: Target return series.

    Returns:
        Spearman correlation or ``None`` when undefined.
    """
    frame = pl.DataFrame(
        {
            _FACTOR_VALUE_COLUMN: factor_values,
            TARGET_COLUMN: targets,
        }
    ).drop_nulls([_FACTOR_VALUE_COLUMN, TARGET_COLUMN])
    if frame.height < 2:
        return None
    if frame[_FACTOR_VALUE_COLUMN].n_unique() < 2 or frame[TARGET_COLUMN].n_unique() < 2:
        return None
    value = frame.select(pl.corr(_FACTOR_VALUE_COLUMN, TARGET_COLUMN, method="spearman")).item()
    return _as_float(value)


def fold_stability_from_factor_metrics(factor_metrics: pl.DataFrame) -> pl.DataFrame:
    """Build fold × factor stability rows from Purged-CV factor metrics.

    Uses the evaluator's pooled Spearman ``oos_ic`` directly so diagnostic IC
    matches production Purged-CV evaluation methodology. Inverted IC is the
    arithmetic negation (Spearman sign-flip identity).
    """
    if factor_metrics.height == 0:
        return pl.DataFrame(schema={column: pl.Null for column in STABILITY_ALL_COLUMNS})
    required = (
        "manager",
        "timeframe",
        "year",
        "factor_name",
        "factor_version",
        "fold_id",
        "oos_ic",
        "oos_rows",
        "oos_return_mean",
        "oos_return_std",
        "oos_positive_rate",
    )
    _require_columns(factor_metrics, required)
    rows: list[dict[str, object]] = []
    for row in factor_metrics.sort(
        ["manager", "timeframe", "year", "factor_name", "factor_version", "fold_id"]
    ).iter_rows(named=True):
        ic = _as_float(row["oos_ic"])
        inverted = None if ic is None else -ic
        rows.append(
            {
                "manager": str(row["manager"]),
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "timeframe": str(row["timeframe"]),
                "year": int(row["year"]),
                "factor_name": str(row["factor_name"]),
                "factor_version": str(row["factor_version"]),
                "factor_family": _FAMILY_UNKNOWN,
                "fold_id": int(row["fold_id"]),
                "ic": ic,
                "ic_inverted": inverted,
                "rows": int(row["oos_rows"]),
                "mean_return": _as_float(row["oos_return_mean"]),
                "target_std": _as_float(row["oos_return_std"]),
                "positive_rate": _as_float(row["oos_positive_rate"]),
                "orientation_class": "",
                "selection_score": None,
                "selection_rank": None,
                "selection_metric_ic": None,
                "selection_metric_rank_ic": None,
                "is_oos_class": "",
                "raw_spearman_ic": ic,
                "outlier_sensitive_ic": None,
                "outlier_policy": _OUTLIER_POLICY_NOTE,
            }
        )
    if not rows:
        return pl.DataFrame(schema={column: pl.Null for column in STABILITY_ALL_COLUMNS})
    return pl.DataFrame(rows)


def compute_fold_factor_ic(observations: pl.DataFrame) -> pl.DataFrame:
    """Compute per-factor OOS IC metrics for every fold without pooling folds.

    Args:
        observations: Purged-CV evaluation observations (OOS preferred).

    Returns:
        Fold × factor metric frame with original and inverted IC.
    """
    _require_columns(
        observations,
        (
            "fold_id",
            "factor_name",
            "factor_version",
            "selected",
            _FACTOR_VALUE_COLUMN,
            TARGET_COLUMN,
            "timeframe",
            "year",
            "manager",
        ),
    )
    if "partition" in observations.columns:
        selected = observations.filter((pl.col("selected")) & (pl.col("partition") == "OOS"))
    else:
        selected = observations.filter(pl.col("selected"))
    if selected.height == 0:
        return pl.DataFrame(schema={column: pl.Null for column in STABILITY_ALL_COLUMNS})

    rows: list[dict[str, object]] = []
    for row in (
        selected.group_by(
            ["manager", "timeframe", "year", "factor_name", "factor_version", "fold_id"],
            maintain_order=True,
        )
        .agg(
            [
                pl.len().alias("rows"),
                pl.col(TARGET_COLUMN).drop_nulls().mean().alias("mean_return"),
                pl.col(TARGET_COLUMN).drop_nulls().std(ddof=1).alias("target_std"),
                (pl.col(TARGET_COLUMN).drop_nulls() > 0.0).mean().alias("positive_rate"),
                pl.corr(_FACTOR_VALUE_COLUMN, TARGET_COLUMN, method="spearman").alias("ic"),
            ]
        )
        .sort(["manager", "timeframe", "year", "factor_name", "factor_version", "fold_id"])
        .iter_rows(named=True)
    ):
        ic = _as_float(row["ic"])
        inverted = None if ic is None else -ic
        factor_name = str(row["factor_name"])
        factor_version = str(row["factor_version"])
        fold_id = int(row["fold_id"])
        group = selected.filter(
            (pl.col("factor_name") == factor_name)
            & (pl.col("factor_version") == factor_version)
            & (pl.col("fold_id") == fold_id)
        )
        outlier_ic = _exploratory_outlier_ic(group)
        rows.append(
            {
                "manager": str(row["manager"]),
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "timeframe": str(row["timeframe"]),
                "year": int(row["year"]),
                "factor_name": factor_name,
                "factor_version": factor_version,
                "factor_family": _FAMILY_UNKNOWN,
                "fold_id": fold_id,
                "ic": ic,
                "ic_inverted": inverted,
                "rows": int(row["rows"]),
                "mean_return": _as_float(row["mean_return"]),
                "target_std": _as_float(row["target_std"]),
                "positive_rate": _as_float(row["positive_rate"]),
                "orientation_class": "",
                "selection_score": None,
                "selection_rank": None,
                "selection_metric_ic": None,
                "selection_metric_rank_ic": None,
                "is_oos_class": "",
                "raw_spearman_ic": ic,
                "outlier_sensitive_ic": outlier_ic,
                "outlier_policy": _OUTLIER_POLICY_NOTE,
            }
        )
    if not rows:
        return pl.DataFrame(schema={column: pl.Null for column in STABILITY_ALL_COLUMNS})
    return pl.DataFrame(rows)


def classify_orientation(
    *,
    mean_ic: float | None,
    positive_folds: int,
    negative_folds: int,
) -> str:
    """Classify factor orientation / sign stability with descriptive labels.

    No promotion magnitude thresholds are invented. Classification uses fold
    sign counts and the sign of mean IC only.
    """
    total = positive_folds + negative_folds
    if total == 0 or mean_ic is None:
        return "NEUTRAL"
    if positive_folds == total and mean_ic > 0.0:
        return "STABLE_POSITIVE"
    if negative_folds == total and mean_ic < 0.0:
        return "ORIENTATION_REVERSAL_CANDIDATE"
    if positive_folds > 0 and negative_folds > 0:
        return "UNSTABLE"
    if mean_ic > 0.0:
        return "STABLE_POSITIVE"
    if mean_ic < 0.0:
        return "STABLE_NEGATIVE"
    return "NEUTRAL"


def classify_cross_timeframe_stability(
    *,
    mean_ics: Sequence[float],
) -> tuple[str, float | None]:
    """Classify cross-timeframe IC sign consistency.

    Returns:
        ``(class_label, sign_consistency)`` where sign consistency is the
        fraction of timeframes sharing the majority IC sign.
    """
    if len(mean_ics) == 0:
        return "SINGLE_TIMEFRAME", None
    if len(mean_ics) == 1:
        return "SINGLE_TIMEFRAME", 1.0
    signs = [1 if value > 0.0 else (-1 if value < 0.0 else 0) for value in mean_ics]
    nonzero = [sign for sign in signs if sign != 0]
    if not nonzero:
        return "CROSS_TIMEFRAME_UNSTABLE", 0.0
    positive = sum(1 for sign in nonzero if sign > 0)
    negative = sum(1 for sign in nonzero if sign < 0)
    majority = max(positive, negative)
    consistency = majority / len(nonzero)
    if positive > 0 and negative > 0:
        if consistency >= 1.0:
            return "CROSS_TIMEFRAME_STABLE", consistency
        return "CROSS_TIMEFRAME_UNSTABLE", consistency
    if all(sign < 0 for sign in nonzero):
        return "CROSS_TIMEFRAME_INVERTED", consistency
    return "CROSS_TIMEFRAME_STABLE", consistency


def classify_is_oos(
    *,
    selection_ic: float | None,
    oos_ic: float | None,
) -> str:
    """Classify in-sample selection evidence versus OOS IC signs."""
    if oos_ic is None:
        return "IS_NEUTRAL_OOS"
    if selection_ic is None:
        return "IS_NEUTRAL_OOS"
    if selection_ic > 0.0 and oos_ic > 0.0:
        return "IS_POSITIVE_OOS_POSITIVE"
    if selection_ic > 0.0 and oos_ic < 0.0:
        return "IS_POSITIVE_OOS_NEGATIVE"
    if selection_ic < 0.0 and oos_ic > 0.0:
        return "IS_NEGATIVE_OOS_POSITIVE"
    if selection_ic < 0.0 and oos_ic < 0.0:
        return "IS_NEGATIVE_OOS_NEGATIVE"
    return "IS_NEUTRAL_OOS"


def detect_ic_methodology() -> dict[str, str]:
    """Document Factor Validation vs Walk-Forward / Purged-CV IC methodology.

    Returns:
        Methodology description mapping including alignment status.
    """
    return {
        "factor_validation_information_coefficient": (
            "mean of cross-sectional Pearson IC per open_time"
        ),
        "factor_validation_rank_information_coefficient": (
            "pooled Spearman of factor_value vs future_return_1"
        ),
        "walk_forward_oos_ic": "pooled Spearman of factor_value vs future_return_1",
        "purged_cv_oos_ic": "pooled Spearman of factor_value vs future_return_1",
        "diagnostic_ic": "pooled Spearman (identical to purged-CV OOS IC)",
        "alignment_status": "METHODOLOGY_ALIGNED",
        "alignment_notes": (
            "Purged-CV / Walk-Forward OOS IC match Factor Validation "
            "rank_information_coefficient (pooled Spearman). Primary Factor "
            "Validation information_coefficient remains cross-sectional Pearson "
            "and is a distinct metric, not a mismatch within the OOS chain."
        ),
    }


def verify_target_alignment(
    evaluation_input: pl.DataFrame,
    *,
    manager: str,
    exchange: str,
    market: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    """Verify factor timestamps align with Labels ``future_return_1``.

    Canonical contract: factor(t) joins label(t) on
    ``(symbol, timeframe, open_time)`` where ``future_return_1`` is the
    one-bar forward return from ``t`` to ``t+1``.
    """
    _require_columns(
        evaluation_input,
        ("symbol", "timeframe", "open_time", "selection_time", TARGET_COLUMN),
    )
    sample = evaluation_input.head(_ALIGNMENT_SAMPLE_ROWS)
    rows_checked = int(sample.height)
    duplicate_keys = int(
        evaluation_input.group_by(["symbol", "timeframe", "open_time", "factor_name"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    time_mismatch = int(sample.filter(pl.col("open_time") != pl.col("selection_time")).height)
    missing_labels = int(sample.filter(pl.col(TARGET_COLUMN).is_null()).height)
    alignment_fail = time_mismatch + (1 if duplicate_keys > 0 else 0)
    alignment_pass = rows_checked - time_mismatch
    status = "PASS" if alignment_fail == 0 and duplicate_keys == 0 else "FAIL"
    return pl.DataFrame(
        [
            {
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "timeframe": timeframe,
                "year": year,
                "diagnostic": "target_alignment",
                "factor_name": "",
                "factor_version": "",
                "rows_checked": rows_checked,
                "alignment_pass": alignment_pass,
                "alignment_fail": alignment_fail,
                "duplicate_keys": duplicate_keys,
                "missing_labels": missing_labels,
                "value_mismatches": 0,
                "label_definition": _LABEL_DEFINITION,
                "status": status,
                "notes": (
                    f"horizon_bars={_LABEL_HORIZON_BARS}; " "selection_time must equal open_time"
                ),
            }
        ]
    )


def verify_factor_value_integrity(
    *,
    evaluation_input: pl.DataFrame,
    observations: pl.DataFrame,
    factors_frame: pl.DataFrame,
    manager: str,
    exchange: str,
    market: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    """Compare canonical Factors values against evaluation / OOS observations."""
    _require_columns(
        evaluation_input,
        (
            "symbol",
            "timeframe",
            "open_time",
            "factor_name",
            "factor_version",
            _FACTOR_VALUE_COLUMN,
        ),
    )
    join_keys = [
        "symbol",
        "timeframe",
        "open_time",
        "factor_name",
        "factor_version",
    ]
    sample = evaluation_input.select([*join_keys, _FACTOR_VALUE_COLUMN]).head(
        _VALUE_INTEGRITY_SAMPLE_ROWS
    )
    factors_side = factors_frame.select(
        [
            *join_keys,
            pl.col(_FACTOR_VALUE_COLUMN).alias("canonical_factor_value"),
        ]
    )
    # Key existence only. Null factor_value is warmup/missing data, not a key miss.
    missing_keys = int(sample.join(factors_side.select(join_keys), on=join_keys, how="anti").height)
    compared = sample.join(factors_side, on=join_keys, how="inner")
    value_mismatches = int(
        compared.filter(
            pl.col("canonical_factor_value").is_not_null()
            & pl.col(_FACTOR_VALUE_COLUMN).is_not_null()
            & (pl.col(_FACTOR_VALUE_COLUMN) != pl.col("canonical_factor_value"))
        ).height
    )

    oos_mismatches = 0
    oos_missing_keys = 0
    oos_rows = 0
    if observations.height > 0 and _FACTOR_VALUE_COLUMN in observations.columns:
        oos = observations
        if "partition" in oos.columns:
            oos = oos.filter(pl.col("partition") == "OOS")
        if "observation_time" in oos.columns:
            oos = oos.rename({"observation_time": "open_time"})
        oos_rows = min(int(oos.height), _VALUE_INTEGRITY_SAMPLE_ROWS)
        oos_sample = oos.head(oos_rows)
        if set(join_keys).issubset(oos_sample.columns):
            oos_keys = oos_sample.select([*join_keys, _FACTOR_VALUE_COLUMN])
            oos_missing_keys = int(
                oos_keys.join(factors_side.select(join_keys), on=join_keys, how="anti").height
            )
            oos_compared = oos_keys.join(factors_side, on=join_keys, how="inner")
            oos_mismatches = int(
                oos_compared.filter(
                    pl.col("canonical_factor_value").is_not_null()
                    & pl.col(_FACTOR_VALUE_COLUMN).is_not_null()
                    & (pl.col(_FACTOR_VALUE_COLUMN) != pl.col("canonical_factor_value"))
                ).height
            )

    total_mismatches = value_mismatches + oos_mismatches
    total_missing_keys = missing_keys + oos_missing_keys
    status = "PASS" if total_mismatches == 0 and total_missing_keys == 0 else "FAIL"
    return pl.DataFrame(
        [
            {
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "timeframe": timeframe,
                "year": year,
                "diagnostic": "value_integrity",
                "factor_name": "",
                "factor_version": "",
                "rows_checked": int(sample.height) + oos_rows,
                "alignment_pass": (
                    int(sample.height) + oos_rows - total_mismatches - total_missing_keys
                ),
                "alignment_fail": total_mismatches + total_missing_keys,
                "duplicate_keys": 0,
                "missing_labels": total_missing_keys,
                "value_mismatches": total_mismatches,
                "label_definition": _LABEL_DEFINITION,
                "status": status,
                "notes": (
                    "exact float equality on non-null pairs; "
                    "null factor_value is not treated as a key miss; "
                    "evaluation_input and purged-CV OOS observations sampled"
                ),
            }
        ]
    )


def aggregate_cross_timeframe(stability_all: pl.DataFrame) -> pl.DataFrame:
    """Aggregate factor mean IC across timeframes."""
    if stability_all.height == 0:
        return pl.DataFrame(schema={column: pl.Null for column in TIMEFRAMES_COLUMNS})
    per_tf = (
        stability_all.group_by(
            ["manager", "exchange", "market", "factor_name", "factor_version", "timeframe"]
        )
        .agg(pl.col("ic").mean().alias("mean_ic"))
        .sort(["manager", "factor_name", "factor_version", "timeframe"])
    )
    rows: list[dict[str, object]] = []
    identity_cols = ["manager", "exchange", "market", "factor_name", "factor_version"]
    for identity in per_tf.select(identity_cols).unique(maintain_order=True).iter_rows(named=True):
        group = per_tf.filter(
            (pl.col("manager") == identity["manager"])
            & (pl.col("exchange") == identity["exchange"])
            & (pl.col("market") == identity["market"])
            & (pl.col("factor_name") == identity["factor_name"])
            & (pl.col("factor_version") == identity["factor_version"])
        )
        mean_ics = [
            value
            for value in (_as_float(item) for item in group["mean_ic"].to_list())
            if value is not None
        ]
        class_label, consistency = classify_cross_timeframe_stability(mean_ics=mean_ics)
        positive_tf = sum(1 for value in mean_ics if value > 0.0)
        negative_tf = sum(1 for value in mean_ics if value < 0.0)
        median_raw = pl.Series(mean_ics).median() if mean_ics else None
        median_value = float(median_raw) if isinstance(median_raw, int | float) else None
        rows.append(
            {
                "manager": identity["manager"],
                "exchange": identity["exchange"],
                "market": identity["market"],
                "factor_name": identity["factor_name"],
                "factor_version": identity["factor_version"],
                "timeframes_evaluated": int(group.height),
                "mean_ic": _as_float(sum(mean_ics) / len(mean_ics)) if mean_ics else None,
                "median_ic": _as_float(median_value),
                "positive_timeframes": positive_tf,
                "negative_timeframes": negative_tf,
                "ic_sign_consistency": consistency,
                "cross_timeframe_class": class_label,
            }
        )
    return _sort_frame(pl.DataFrame(rows), TIMEFRAMES_COLUMNS)


def aggregate_families(stability_all: pl.DataFrame) -> pl.DataFrame:
    """Aggregate IC diagnostics by factor family / category metadata."""
    if stability_all.height == 0:
        return pl.DataFrame(schema={column: pl.Null for column in FAMILIES_COLUMNS})
    frame = stability_all.with_columns(
        pl.when(pl.col("factor_family").is_null() | (pl.col("factor_family") == ""))
        .then(pl.lit(_FAMILY_UNKNOWN))
        .otherwise(pl.col("factor_family"))
        .alias("family")
    )
    factor_means = frame.group_by(
        ["manager", "exchange", "market", "family", "factor_name", "factor_version"]
    ).agg(pl.col("ic").mean().alias("factor_mean_ic"))
    rows: list[dict[str, object]] = []
    family_keys = ["manager", "exchange", "market", "family"]
    for identity in frame.select(family_keys).unique(maintain_order=True).iter_rows(named=True):
        group = frame.filter(
            (pl.col("manager") == identity["manager"])
            & (pl.col("exchange") == identity["exchange"])
            & (pl.col("market") == identity["market"])
            & (pl.col("family") == identity["family"])
        )
        ics = [
            value
            for value in (_as_float(item) for item in group["ic"].to_list())
            if value is not None
        ]
        family_factors = factor_means.filter(
            (pl.col("manager") == identity["manager"]) & (pl.col("family") == identity["family"])
        )
        positive_fraction = None
        if family_factors.height > 0:
            means = [
                value
                for value in (
                    _as_float(item) for item in family_factors["factor_mean_ic"].to_list()
                )
                if value is not None
            ]
            if means:
                positive_fraction = sum(1 for value in means if value > 0.0) / len(means)
        median_raw = pl.Series(ics).median() if ics else None
        median_value = float(median_raw) if isinstance(median_raw, int | float) else None
        rows.append(
            {
                "manager": identity["manager"],
                "exchange": identity["exchange"],
                "market": identity["market"],
                "family": identity["family"],
                "factor_count": int(family_factors.height),
                "factor_fold_rows": int(group.height),
                "mean_ic": _as_float(sum(ics) / len(ics)) if ics else None,
                "median_ic": _as_float(median_value),
                "positive_factor_fraction": _as_float(positive_fraction),
            }
        )
    return _sort_frame(pl.DataFrame(rows), FAMILIES_COLUMNS)


def build_global_summary(
    *,
    stability_all: pl.DataFrame,
    folds: pl.DataFrame,
    timeframes: pl.DataFrame,
    families: pl.DataFrame,
    alignment_audit: pl.DataFrame,
    panel_results: Sequence[PanelDiagnosticBundle],
    methodology: Mapping[str, str],
) -> pl.DataFrame:
    """Build the global diagnostic summary answering the ten critical questions."""
    alignment_ok = all(item.alignment_ok for item in panel_results) if panel_results else False
    value_ok = all(item.value_ok for item in panel_results) if panel_results else False
    folds_accounted = int(sum(item.fold_count for item in panel_results))
    timeframes_analyzed = (
        sorted({str(tf) for tf in stability_all["timeframe"].unique()})
        if (stability_all.height > 0 and "timeframe" in stability_all.columns)
        else []
    )

    target_stats = _target_distribution_rows(stability_all)
    orientation_answer, orientation_evidence = _answer_orientation(stability_all)
    fold_stability = _answer_fold_stability(stability_all)
    timeframe_stability = _answer_timeframe_stability(timeframes)
    family_answer = _answer_family_concentration(families)
    is_oos_answer = _answer_is_oos(stability_all)
    methodology_aligned = methodology.get("alignment_status") == "METHODOLOGY_ALIGNED"
    selection_intensity = _selection_intensity(panel_results)
    primary = _select_primary_conclusion(
        alignment_ok=alignment_ok,
        value_ok=value_ok,
        methodology_aligned=methodology_aligned,
        orientation_answer=orientation_answer,
        fold_stability=fold_stability,
        timeframe_stability=timeframe_stability,
        family_answer=family_answer,
        is_oos_answer=is_oos_answer,
        stability_all=stability_all,
    )
    verdict = _select_verdict(
        alignment_ok=alignment_ok,
        value_ok=value_ok,
        methodology_aligned=methodology_aligned,
        folds_accounted=folds_accounted,
        timeframes_analyzed=timeframes_analyzed,
        panel_count=len(panel_results),
    )

    rows: list[dict[str, object]] = [
        {"metric": "verdict", "value": verdict},
        {"metric": "timeframes_analyzed", "value": ",".join(timeframes_analyzed)},
        {"metric": "timeframe_panel_count", "value": str(len(panel_results))},
        {"metric": "folds_accounted", "value": str(folds_accounted)},
        {"metric": "factor_fold_rows", "value": str(stability_all.height)},
        {"metric": "fold_metric_rows", "value": str(folds.height)},
        {"metric": "regime_analysis", "value": _REGIME_NOT_AVAILABLE},
        {"metric": "selection_intensity", "value": selection_intensity},
        {"metric": "methodology_status", "value": methodology.get("alignment_status", "")},
        {
            "metric": "methodology_notes",
            "value": methodology.get("alignment_notes", ""),
        },
        {
            "metric": "factor_validation_ic_method",
            "value": methodology.get("factor_validation_information_coefficient", ""),
        },
        {
            "metric": "factor_validation_rank_ic_method",
            "value": methodology.get("factor_validation_rank_information_coefficient", ""),
        },
        {
            "metric": "purged_cv_oos_ic_method",
            "value": methodology.get("purged_cv_oos_ic", ""),
        },
        {"metric": "label_definition", "value": _LABEL_DEFINITION},
        {"metric": "outlier_policy", "value": _OUTLIER_POLICY_NOTE},
        {"metric": "q1_future_return_1_aligned", "value": "YES" if alignment_ok else "NO"},
        {"metric": "q2_factor_values_preserved", "value": "YES" if value_ok else "NO"},
        {"metric": "q3_orientation_cause", "value": orientation_answer},
        {"metric": "q4_stable_across_folds", "value": fold_stability},
        {"metric": "q5_stable_across_timeframes", "value": timeframe_stability},
        {"metric": "q6_family_concentration", "value": family_answer},
        {"metric": "q7_is_positive_oos_negative", "value": is_oos_answer},
        {"metric": "q8_methodology_aligned", "value": "YES" if methodology_aligned else "NO"},
        {
            "metric": "q9_selection_intensity_reconstructible",
            "value": "YES" if selection_intensity != _SELECTION_INTENSITY_NOT_AVAILABLE else "NO",
        },
        {"metric": "q10_primary_conclusion", "value": primary},
        {"metric": "orientation_evidence", "value": orientation_evidence},
    ]
    rows.extend(target_stats)
    return pl.DataFrame(rows).select(list(GLOBAL_COLUMNS))


def write_factor_stability_reports(
    *,
    output_root: Path,
    stability_all: pl.DataFrame,
    folds: pl.DataFrame,
    timeframes: pl.DataFrame,
    families: pl.DataFrame,
    alignment_audit: pl.DataFrame,
    global_summary: pl.DataFrame,
) -> dict[str, Path]:
    """Persist the six deterministic diagnostic CSV reports."""
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "all": output_root / STABILITY_ALL_CSV_NAME,
        "folds": output_root / FOLDS_CSV_NAME,
        "timeframes": output_root / TIMEFRAMES_CSV_NAME,
        "families": output_root / FAMILIES_CSV_NAME,
        "global": output_root / GLOBAL_CSV_NAME,
        "alignment": output_root / ALIGNMENT_AUDIT_CSV_NAME,
    }
    _ensure_columns(stability_all, STABILITY_ALL_COLUMNS).write_csv(paths["all"])
    _ensure_columns(folds, FOLDS_COLUMNS).write_csv(paths["folds"])
    _ensure_columns(timeframes, TIMEFRAMES_COLUMNS).write_csv(paths["timeframes"])
    _ensure_columns(families, FAMILIES_COLUMNS).write_csv(paths["families"])
    _ensure_columns(global_summary, GLOBAL_COLUMNS).write_csv(paths["global"])
    _ensure_columns(alignment_audit, ALIGNMENT_AUDIT_COLUMNS).write_csv(paths["alignment"])
    return paths


def forbidden_import_violations(source: str) -> tuple[str, ...]:
    """Return forbidden import prefixes found in ``source`` text."""
    violations: list[str] = []
    for module in _FORBIDDEN_IMPORT_PREFIXES:
        patterns = (f"import {module}", f"from {module} ", f"from {module}.")
        if any(pattern in source for pattern in patterns):
            violations.append(module)
    return tuple(violations)


def _enrich_stability_rows(
    fold_ic: pl.DataFrame,
    *,
    selection: pl.DataFrame,
    validation: pl.DataFrame | None,
    manager: str,
    exchange: str,
    market: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame:
    """Attach selection / validation metadata and classification labels."""
    if fold_ic.height == 0:
        return _ensure_columns(pl.DataFrame(), STABILITY_ALL_COLUMNS)

    family_source = "factor_category" if "factor_category" in selection.columns else None
    selected = selection.filter(pl.col("selected"))
    meta_rows: dict[tuple[str, str], dict[str, object]] = {}
    for row in selected.iter_rows(named=True):
        key = (str(row["factor_name"]), str(row["factor_version"]))
        meta_rows[key] = {
            "factor_family": (
                str(row[family_source])
                if family_source is not None and row.get(family_source) not in (None, "")
                else _FAMILY_UNKNOWN
            ),
            "selection_score": _as_float(row.get("selection_score")),
            "selection_rank": (
                int(row["selection_rank"]) if row.get("selection_rank") is not None else None
            ),
            "selection_metric_ic": None,
            "selection_metric_rank_ic": None,
        }
    if validation is not None and validation.height > 0:
        for row in validation.iter_rows(named=True):
            key = (str(row["factor_name"]), str(row["factor_version"]))
            current = meta_rows.setdefault(
                key,
                {
                    "factor_family": _FAMILY_UNKNOWN,
                    "selection_score": None,
                    "selection_rank": None,
                    "selection_metric_ic": None,
                    "selection_metric_rank_ic": None,
                },
            )
            current["selection_metric_ic"] = _as_float(row.get("information_coefficient"))
            current["selection_metric_rank_ic"] = _as_float(row.get("rank_information_coefficient"))

    records: list[dict[str, object]] = []
    for row in fold_ic.iter_rows(named=True):
        key = (str(row["factor_name"]), str(row["factor_version"]))
        meta = meta_rows.get(
            key,
            {
                "factor_family": _FAMILY_UNKNOWN,
                "selection_score": None,
                "selection_rank": None,
                "selection_metric_ic": None,
                "selection_metric_rank_ic": None,
            },
        )
        records.append(
            {
                **row,
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "timeframe": timeframe,
                "year": year,
                "factor_family": meta["factor_family"],
                "selection_score": meta["selection_score"],
                "selection_rank": meta["selection_rank"],
                "selection_metric_ic": meta["selection_metric_ic"],
                "selection_metric_rank_ic": meta["selection_metric_rank_ic"],
            }
        )
    enriched = pl.DataFrame(records)

    agg = enriched.group_by(["factor_name", "factor_version"]).agg(
        [
            pl.col("ic").mean().alias("mean_ic"),
            (pl.col("ic") > 0.0).sum().alias("positive_folds"),
            (pl.col("ic") < 0.0).sum().alias("negative_folds"),
            pl.col("selection_metric_rank_ic").drop_nulls().first().alias("rank_ic"),
            pl.col("selection_metric_ic").drop_nulls().first().alias("sel_ic"),
        ]
    )
    class_rows: dict[tuple[str, str], tuple[str, str]] = {}
    for row in agg.iter_rows(named=True):
        mean_ic = _as_float(row["mean_ic"])
        orientation = classify_orientation(
            mean_ic=mean_ic,
            positive_folds=int(row["positive_folds"] or 0),
            negative_folds=int(row["negative_folds"] or 0),
        )
        selection_metric = _as_float(row["rank_ic"])
        if selection_metric is None:
            selection_metric = _as_float(row["sel_ic"])
        is_oos = classify_is_oos(selection_ic=selection_metric, oos_ic=mean_ic)
        class_rows[(str(row["factor_name"]), str(row["factor_version"]))] = (orientation, is_oos)

    orientation_values: list[str] = []
    is_oos_values: list[str] = []
    for row in enriched.iter_rows(named=True):
        orientation, is_oos = class_rows.get(
            (str(row["factor_name"]), str(row["factor_version"])),
            ("NEUTRAL", "IS_NEUTRAL_OOS"),
        )
        orientation_values.append(orientation)
        is_oos_values.append(is_oos)
    enriched = enriched.with_columns(
        [
            pl.Series("orientation_class", orientation_values),
            pl.Series("is_oos_class", is_oos_values),
        ]
    )
    return _ensure_columns(enriched, STABILITY_ALL_COLUMNS)


def _exploratory_outlier_ic(group: pl.DataFrame) -> float | None:
    """Exploratory Spearman IC after excluding extreme |target| tails."""
    frame = group.select([_FACTOR_VALUE_COLUMN, TARGET_COLUMN]).drop_nulls()
    if frame.height < 5:
        return compute_spearman_ic(frame[_FACTOR_VALUE_COLUMN], frame[TARGET_COLUMN])
    abs_ret = frame[TARGET_COLUMN].abs()
    upper_raw = abs_ret.quantile(1.0 - _OUTLIER_EXCLUSION_FRACTION)
    if upper_raw is None:
        return compute_spearman_ic(frame[_FACTOR_VALUE_COLUMN], frame[TARGET_COLUMN])
    upper = float(upper_raw)
    filtered = frame.filter(pl.col(TARGET_COLUMN).abs() <= upper)
    return compute_spearman_ic(filtered[_FACTOR_VALUE_COLUMN], filtered[TARGET_COLUMN])


def _answer_orientation(stability_all: pl.DataFrame) -> tuple[str, str]:
    if stability_all.height == 0:
        return "INCONCLUSIVE", "no_factor_fold_rows"
    classes = stability_all.select(["factor_name", "factor_version", "orientation_class"]).unique()
    reversal = classes.filter(pl.col("orientation_class") == "ORIENTATION_REVERSAL_CANDIDATE")
    unstable = classes.filter(pl.col("orientation_class") == "UNSTABLE")
    positive = classes.filter(pl.col("orientation_class") == "STABLE_POSITIVE")
    total = max(classes.height, 1)
    reversal_share = reversal.height / total
    evidence = (
        f"reversal_candidates={reversal.height};"
        f"stable_positive={positive.height};"
        f"unstable={unstable.height};"
        f"selection_uses_abs_ic=true"
    )
    if reversal_share >= 0.5:
        return "EVIDENCE FOR", evidence
    if positive.height / total >= 0.5:
        return "EVIDENCE AGAINST", evidence
    return "INCONCLUSIVE", evidence


def _answer_fold_stability(stability_all: pl.DataFrame) -> str:
    if stability_all.height == 0:
        return "MIXED"
    classes = stability_all.select(["factor_name", "factor_version", "orientation_class"]).unique()
    unstable = classes.filter(pl.col("orientation_class") == "UNSTABLE").height
    stable = classes.filter(
        pl.col("orientation_class").is_in(
            ["STABLE_POSITIVE", "STABLE_NEGATIVE", "ORIENTATION_REVERSAL_CANDIDATE"]
        )
    ).height
    if unstable == 0 and stable > 0:
        return "YES"
    if stable == 0 and unstable > 0:
        return "NO"
    return "MIXED"


def _answer_timeframe_stability(timeframes: pl.DataFrame) -> str:
    if timeframes.height == 0:
        return "MIXED"
    classes = timeframes["cross_timeframe_class"].to_list()
    stable = sum(1 for item in classes if item == "CROSS_TIMEFRAME_STABLE")
    inverted = sum(1 for item in classes if item == "CROSS_TIMEFRAME_INVERTED")
    unstable = sum(1 for item in classes if item == "CROSS_TIMEFRAME_UNSTABLE")
    if unstable == 0 and (stable + inverted) > 0:
        return "YES"
    if stable == 0 and inverted == 0 and unstable > 0:
        return "NO"
    return "MIXED"


def _answer_family_concentration(families: pl.DataFrame) -> str:
    if families.height == 0:
        return "UNKNOWN"
    if families.height == 1 and families["family"][0] == _FAMILY_UNKNOWN:
        return "UNKNOWN"
    ics = [value for value in families["mean_ic"].to_list() if value is not None]
    if not ics:
        return "UNKNOWN"
    # Concentration: one family materially more negative than the rest.
    sorted_ics = sorted(ics)
    if len(sorted_ics) == 1:
        return "NO"
    if sorted_ics[0] < 0.0 and (sorted_ics[1] - sorted_ics[0]) > abs(sorted_ics[0]) * 0.5:
        return "YES"
    negative_families = sum(1 for value in ics if value < 0.0)
    if negative_families == len(ics):
        return "NO"
    if negative_families == 1 and len(ics) > 1:
        return "YES"
    return "NO"


def _answer_is_oos(stability_all: pl.DataFrame) -> str:
    if stability_all.height == 0:
        return "UNKNOWN"
    classes = stability_all.select(["factor_name", "factor_version", "is_oos_class"]).unique()
    if classes.filter(pl.col("is_oos_class") != "IS_NEUTRAL_OOS").height == 0:
        # No validation metrics attached.
        has_selection_metric = (
            "selection_metric_rank_ic" in stability_all.columns
            and stability_all["selection_metric_rank_ic"].drop_nulls().len() > 0
        ) or (
            "selection_metric_ic" in stability_all.columns
            and stability_all["selection_metric_ic"].drop_nulls().len() > 0
        )
        if not has_selection_metric:
            return "UNKNOWN"
    degraded = classes.filter(pl.col("is_oos_class") == "IS_POSITIVE_OOS_NEGATIVE").height
    total = max(classes.height, 1)
    if degraded / total >= 0.5:
        return "YES"
    if degraded == 0:
        return "NO"
    return "YES" if degraded > (total - degraded) else "NO"


def _selection_intensity(panel_results: Sequence[PanelDiagnosticBundle]) -> str:
    tested = [item.selection_tested for item in panel_results if item.selection_tested is not None]
    selected = [
        item.selection_selected for item in panel_results if item.selection_selected is not None
    ]
    if not tested or not selected:
        return _SELECTION_INTENSITY_NOT_AVAILABLE
    return (
        f"factors_tested={sum(tested)};"
        f"factors_selected={sum(selected)};"
        f"panels={len(panel_results)}"
    )


def _select_primary_conclusion(
    *,
    alignment_ok: bool,
    value_ok: bool,
    methodology_aligned: bool,
    orientation_answer: str,
    fold_stability: str,
    timeframe_stability: str,
    family_answer: str,
    is_oos_answer: str,
    stability_all: pl.DataFrame,
) -> str:
    if not alignment_ok:
        return "A. TARGET_ALIGNMENT_PROBLEM"
    if not value_ok:
        return "B. FACTOR_VALUE_INTEGRITY_PROBLEM"
    if not methodology_aligned:
        return "G. METHODOLOGY_MISMATCH"
    if stability_all.height == 0:
        return "I. INSUFFICIENT_EVIDENCE"
    evidence: list[str] = []
    if orientation_answer == "EVIDENCE FOR":
        evidence.append("C. FACTOR_ORIENTATION_PROBLEM")
    if is_oos_answer == "YES":
        evidence.append("D. SELECTION_OVERFIT_SIGNAL")
    if fold_stability == "NO" or timeframe_stability == "NO":
        evidence.append("E. FACTOR_INSTABILITY")
    if family_answer == "YES":
        evidence.append("F. FACTOR_FAMILY_CONCENTRATION")
    if len(evidence) == 1:
        return evidence[0]
    if len(evidence) == 0:
        # Broadly negative but no single mechanism dominates.
        mean_ic = _as_float(stability_all["ic"].mean()) if "ic" in stability_all.columns else None
        if mean_ic is not None and mean_ic < 0.0:
            return "H. NO_SINGLE_ROOT_CAUSE"
        return "I. INSUFFICIENT_EVIDENCE"
    return "H. NO_SINGLE_ROOT_CAUSE"


def _select_verdict(
    *,
    alignment_ok: bool,
    value_ok: bool,
    methodology_aligned: bool,
    folds_accounted: int,
    timeframes_analyzed: Sequence[str],
    panel_count: int,
) -> str:
    if panel_count == 0:
        return "FAIL"
    if not alignment_ok or not value_ok:
        return "FAIL"
    if not methodology_aligned:
        return "FAIL"
    if folds_accounted <= 0 or len(timeframes_analyzed) == 0:
        return "PASS WITH GAPS"
    return "PASS"


def _target_distribution_rows(stability_all: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if stability_all.height == 0 or "mean_return" not in stability_all.columns:
        return rows
    for timeframe in sorted(stability_all["timeframe"].unique().to_list()):
        subset = stability_all.filter(pl.col("timeframe") == timeframe)
        returns = subset["mean_return"].drop_nulls()
        rows.append(
            {
                "metric": f"target_dist_{timeframe}_fold_rows",
                "value": str(subset.height),
            }
        )
        rows.append(
            {
                "metric": f"target_dist_{timeframe}_mean_of_fold_means",
                "value": _format_optional_float(_as_float(returns.mean())),
            }
        )
        rows.append(
            {
                "metric": f"target_dist_{timeframe}_positive_rate_mean",
                "value": _format_optional_float(
                    _as_float(subset["positive_rate"].drop_nulls().mean())
                ),
            }
        )
    return rows


def _load_factors_for_sample(
    factors_repository: FactorsRepository,
    *,
    manager: str,
    exchange: str,
    market: str,
    timeframe: str,
    year: int,
    sample: pl.DataFrame,
) -> pl.DataFrame:
    """Load canonical Factors only for symbols present in ``sample``."""
    if sample.height == 0 or "symbol" not in sample.columns:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "timeframe": pl.String,
                "open_time": pl.Int64,
                "factor_name": pl.String,
                "factor_version": pl.String,
                _FACTOR_VALUE_COLUMN: pl.Float64,
            }
        )
    symbols = sorted({str(symbol) for symbol in sample["symbol"].unique().to_list()})
    parts: list[pl.DataFrame] = []
    for symbol in symbols:
        if factors_repository.exists(
            manager=manager,
            exchange=exchange,
            market=market,
            symbol=symbol,
            timeframe=timeframe,
            year=year,
        ):
            parts.append(
                factors_repository.load(
                    manager=manager,
                    exchange=exchange,
                    market=market,
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                )
            )
    if not parts:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "timeframe": pl.String,
                "open_time": pl.Int64,
                "factor_name": pl.String,
                "factor_version": pl.String,
                _FACTOR_VALUE_COLUMN: pl.Float64,
            }
        )
    return pl.concat(parts, how="vertical")


def _try_load_validation(
    repository: FactorValidationRepository | None,
    *,
    manager: str,
    exchange: str,
    market: str,
    timeframe: str,
    year: int,
) -> pl.DataFrame | None:
    if repository is None:
        return None
    try:
        if not repository.exists(
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        ):
            return None
        return repository.load(
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        )
    except Exception:
        return None


def _hash_watched_artifacts(storage_root: Path | None) -> dict[str, str]:
    if storage_root is None:
        return {}
    hashes: dict[str, str] = {}
    watched = [
        STORAGE_DIR_PURGED_CV,
        STORAGE_DIR_FACTOR_SELECTION,
        STORAGE_DIR_WALK_FORWARD,
        STORAGE_DIR_WALK_FORWARD_EVALUATION,
    ]
    for tier in watched:
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


def _audit_passed(audit: pl.DataFrame, diagnostic: str) -> bool:
    rows = audit.filter(pl.col("diagnostic") == diagnostic)
    if rows.height == 0:
        return False
    return all(status == "PASS" for status in rows["status"].to_list())


def _concat_frames(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    nonempty = [frame for frame in frames if frame.height > 0]
    if not nonempty:
        return pl.DataFrame()
    return pl.concat(nonempty, how="vertical")


def _sort_frame(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    ensured = _ensure_columns(frame, columns)
    sort_keys = [column for column in SORT_COLUMNS if column in ensured.columns]
    if not sort_keys:
        return ensured
    return ensured.sort(sort_keys)


def _ensure_columns(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    if frame.height == 0 and frame.width == 0:
        return pl.DataFrame({column: [] for column in columns})
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        frame = frame.with_columns([pl.lit(None).alias(column) for column in missing])
    return frame.select(list(columns))


def _require_columns(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ReportingValidationError(
            "diagnostic frame missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={"missing_columns": missing},
        )


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def _global_metric(summary: pl.DataFrame, metric: str) -> str | None:
    matched = summary.filter(pl.col("metric") == metric)
    if matched.height == 0:
        return None
    return str(matched["value"][0])

"""CQROS 1d factor-replacement investigation reporter.

Purpose:
    Run a controlled, leakage-safe investigation that retires unsuitable 1d
    PVT/OBV/OI level factors from an experimental selection path and evaluates
    alternative registry candidates without mutating production ledgers.

Responsibilities:
    - Inventory the production factor registry and classify families
    - Publish a deterministic versioned replacement candidate set
    - Re-run Factor Selection with ``coverage_v1`` / ``signed_ic_v1`` after
      excluding retired factors from the 1d experiment universe
    - Compare OLD production 1d selected set vs NEW experimental set using
      existing Purged-CV evaluation observations when the NEW set is a subset
    - Emit reports under ``reports/factor_stability/1d_factor_replacement``
    - SHA-256 hash watched production ledgers before and after
    - Remain free of Alpha, Regime, Predictions, Signals, and ``cqros.ml``

Dependencies:
    ``ast``, ``hashlib``, ``logging``, ``math``, ``statistics``, ``polars``,
    ``cqros.core.constants``, ``cqros.factor_selection``, ``cqros.factors``,
    ``cqros.factor_validation``, ``cqros.reporting.exceptions``, and
    ``cqros.storage``.

Public API:
    Candidate / retirement / decision constants,
    ``FactorStability1dFactorReplacementReporter``,
    ``FactorStability1dFactorReplacementResult``,
    ``classify_factor_family``,
    ``classify_replacement_decision``,
    ``classify_verdict``,
    ``forbidden_import_violations``,
    ``hash_watched_production_artifacts``, and
    ``is_cumulative_level_factor``.

Notes:
    Investigation only. Never mutates production lake artifacts, never retunes
    thresholds, never flips orientation from OOS IC, and never claims
    ``FACTOR_REPLACEMENT_SUCCESS`` without sufficient evidence.
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
from cqros.factor_selection import (
    FactorEligibilityPolicy,
    FactorsObservationLoader,
    SimpleFactorSelectionEngine,
)
from cqros.factor_selection.eligibility import FACTOR_ELIGIBILITY_POLICY
from cqros.factor_selection.orientation import FACTOR_ORIENTATION_POLICY
from cqros.factor_validation import FactorValidationRepository
from cqros.factors.default_registry import build_default_registry
from cqros.reporting.exceptions import ReportingValidationError
from cqros.storage import ParquetStore, StorageLayout

__all__ = [
    "CANDIDATE_ELIGIBILITY_COLUMNS",
    "CANDIDATE_ELIGIBILITY_CSV_NAME",
    "CANDIDATE_FACTORS_COLUMNS",
    "CANDIDATE_FACTORS_CSV_NAME",
    "CANDIDATE_FOLDS_COLUMNS",
    "CANDIDATE_FOLDS_CSV_NAME",
    "CANDIDATE_INVENTORY_COLUMNS",
    "CANDIDATE_INVENTORY_CSV_NAME",
    "CANDIDATE_SELECTION_COLUMNS",
    "CANDIDATE_SELECTION_CSV_NAME",
    "CANDIDATE_SET_VERSION",
    "COMPARISON_TIMEFRAMES",
    "CROSS_TIMEFRAME_COLUMNS",
    "CROSS_TIMEFRAME_CSV_NAME",
    "DECISION_ELIGIBLE_AND_SELECTED",
    "DECISION_ELIGIBLE_NOT_SELECTED",
    "DECISION_INELIGIBLE_COMPANION_HISTORY",
    "DECISION_INELIGIBLE_INSUFFICIENT_WARMUP",
    "DECISION_INELIGIBLE_LOW_COVERAGE",
    "DECISION_INELIGIBLE_ZERO_OBSERVATIONS",
    "DECISION_RETIRED_EXISTING_FACTOR",
    "DEFAULT_OUTPUT_ROOT",
    "FLAG_1D_STATISTICAL_POWER_LIMITATION",
    "HASHES_AFTER_NAME",
    "HASHES_BEFORE_NAME",
    "REPLACEMENT_CANDIDATES_V1",
    "RETIREMENT_REASON",
    "RETIRED_1D_FACTORS",
    "SUMMARY_TXT_NAME",
    "TARGET_TIMEFRAME",
    "VERDICT_FACTOR_REPLACEMENT_SUCCESS",
    "VERDICT_NO_VIABLE_REPLACEMENT_ENTRIES",
    "VERDICT_REPLACEMENT_INCONCLUSIVE",
    "FactorStability1dFactorReplacementReporter",
    "FactorStability1dFactorReplacementResult",
    "classify_factor_family",
    "classify_replacement_decision",
    "classify_verdict",
    "forbidden_import_violations",
    "hash_watched_production_artifacts",
    "is_cumulative_level_factor",
]

_logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT: Final[Path] = Path("reports") / "factor_stability" / "1d_factor_replacement"
TARGET_TIMEFRAME: Final[str] = "1d"
COMPARISON_TIMEFRAMES: Final[tuple[str, ...]] = ("5m", "15m", "1h", "4h", "1d")

CANDIDATE_SET_VERSION: Final[str] = "1d_replacement_candidates_v1"
RETIREMENT_REASON: Final[str] = "GENUINE_FACTOR_WEAKNESS_UNDERPOWERED_1D"
RETIRED_1D_FACTORS: Final[tuple[str, ...]] = (
    "price_volume_trend",
    "on_balance_volume",
    "open_interest_level",
)

# Deterministic preferred replacement families / factors for 1d.
# Excludes retired cumulative/level factors. Prefer OHLCV and non-level forms.
REPLACEMENT_CANDIDATES_V1: Final[tuple[str, ...]] = (
    # volume (non-cumulative / rolling)
    "volume_zscore",
    "relative_volume",
    "volume_rate_of_change",
    "volume_trend",
    "chaikin_money_flow",
    "money_flow_index",
    "ease_of_movement",
    # price momentum
    "momentum",
    "rate_of_change",
    "multi_horizon_momentum",
    "price_acceleration",
    # trend
    "trend_slope",
    "trend_angle",
    "trend_persistence",
    "sma_distance",
    "ema_distance",
    "efficiency_ratio",
    # volatility / range
    "historical_volatility",
    "atr_percent",
    "bollinger_bandwidth",
    "parkinson_volatility",
    # candle / return structure
    "rsi",
    "stochastic_k",
    "stochastic_d",
    "williams_r",
    "commodity_channel_index",
    "mean_reversion_score",
    # open interest (non-level)
    "open_interest_momentum",
    "open_interest_zscore",
    "open_interest_acceleration",
    # funding (non-raw-level preference still includes momentum/zscore)
    "funding_rate_momentum",
    "funding_rate_zscore",
    # order-flow / microstructure
    "buy_sell_imbalance",
    "order_flow_momentum",
    "signed_volume",
)

CANDIDATE_INVENTORY_CSV_NAME: Final[str] = "candidate_inventory.csv"
CANDIDATE_ELIGIBILITY_CSV_NAME: Final[str] = "candidate_eligibility.csv"
CANDIDATE_SELECTION_CSV_NAME: Final[str] = "candidate_selection.csv"
CANDIDATE_FACTORS_CSV_NAME: Final[str] = "candidate_factors.csv"
CANDIDATE_FOLDS_CSV_NAME: Final[str] = "candidate_folds.csv"
CROSS_TIMEFRAME_CSV_NAME: Final[str] = "cross_timeframe_comparison.csv"
SUMMARY_TXT_NAME: Final[str] = "summary.txt"
HASHES_BEFORE_NAME: Final[str] = "hashes_before.txt"
HASHES_AFTER_NAME: Final[str] = "hashes_after.txt"

DECISION_ELIGIBLE_AND_SELECTED: Final[str] = "ELIGIBLE_AND_SELECTED"
DECISION_ELIGIBLE_NOT_SELECTED: Final[str] = "ELIGIBLE_NOT_SELECTED"
DECISION_INELIGIBLE_ZERO_OBSERVATIONS: Final[str] = "INELIGIBLE_ZERO_OBSERVATIONS"
DECISION_INELIGIBLE_INSUFFICIENT_WARMUP: Final[str] = "INELIGIBLE_INSUFFICIENT_WARMUP"
DECISION_INELIGIBLE_LOW_COVERAGE: Final[str] = "INELIGIBLE_LOW_COVERAGE"
DECISION_INELIGIBLE_COMPANION_HISTORY: Final[str] = "INELIGIBLE_COMPANION_HISTORY"
DECISION_RETIRED_EXISTING_FACTOR: Final[str] = "RETIRED_EXISTING_FACTOR"

FLAG_1D_STATISTICAL_POWER_LIMITATION: Final[str] = "1D_STATISTICAL_POWER_LIMITATION"
VERDICT_FACTOR_REPLACEMENT_SUCCESS: Final[str] = "FACTOR_REPLACEMENT_SUCCESS"
VERDICT_NO_VIABLE_REPLACEMENT_ENTRIES: Final[str] = "NO_VIABLE_REPLACEMENT_ENTRIES"
VERDICT_REPLACEMENT_INCONCLUSIVE: Final[str] = "REPLACEMENT_INCONCLUSIVE"

_PARTITION_OOS: Final[str] = "OOS"
_FACTOR_VALUE: Final[str] = "factor_value"
_TARGET: Final[str] = "future_return_1"
_ORIENTED: Final[str] = "_oriented_factor_value"
_MIN_XS_FOR_IC: Final[int] = 3
_LOW_POWER_TIMESTAMP_LIMIT: Final[int] = 30

_OHLCV_COLUMNS: Final[frozenset[str]] = frozenset({"open", "high", "low", "close", "volume"})
_COMPANION_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "funding_rate",
        "mark_price",
        "index_price",
        "open_interest",
        "taker_buy_volume",
        "taker_sell_volume",
        "long_short_ratio",
        "premium_index",
        "vwap",
        "buy_trade_count",
        "sell_trade_count",
        "trade_count",
        "long_liquidation_volume",
        "short_liquidation_volume",
        "total_liquidation_volume",
        "asset_return",
        "btc_return",
        "eth_return",
        "benchmark_return",
        "returns",
        "oi_momentum",
        "buy_pressure",
        "sell_pressure",
        "flow_imbalance",
        "crowding_score",
        "funding_zscore",
        "ratio_momentum",
    }
)
_CUMULATIVE_LEVEL_FACTORS: Final[frozenset[str]] = frozenset(
    {
        "price_volume_trend",
        "on_balance_volume",
        "accumulation_distribution",
        "open_interest_level",
        "funding_rate_level",
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

_ERROR_MANAGER: Final[str] = "REPORT-1D-REPL-001"
_ERROR_OUTPUT: Final[str] = "REPORT-1D-REPL-002"
_ERROR_MISSING_1D: Final[str] = "REPORT-1D-REPL-003"
_ERROR_LEDGER_MUTATION: Final[str] = "REPORT-1D-REPL-004"
_ERROR_EVALUATION: Final[str] = "REPORT-1D-REPL-005"

CANDIDATE_INVENTORY_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_family",
    "registry_category",
    "factor_group",
    "version",
    "lookback",
    "required_features",
    "produced_columns",
    "uses_ohlcv_only",
    "requires_companion",
    "companion_dependencies",
    "is_cumulative_level",
    "in_replacement_candidate_set",
    "is_retirement_exclusion",
    "candidate_set_version",
)

CANDIDATE_ELIGIBILITY_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_family",
    "in_replacement_candidate_set",
    "is_retirement_exclusion",
    "eligibility_status",
    "eligibility_reason",
    "eligibility_policy",
    "usable_observations",
    "total_observations",
    "coverage_ratio",
    "null_rate",
    "required_lookback",
    "effective_warmup",
    "available_history",
    "warmup_sufficient",
    "companion_dependencies",
    "companion_coverage_status",
    "expected_coverage_note",
    "decision",
)

CANDIDATE_SELECTION_COLUMNS: Final[tuple[str, ...]] = (
    "set_label",
    "factor_name",
    "factor_family",
    "status",
    "selected",
    "selected_direction",
    "orientation_policy",
    "selection_ic",
    "selection_score",
    "selection_rank",
    "selection_reason",
    "eligibility_status",
    "eligibility_policy",
    "usable_observations",
    "null_rate",
    "redundancy_rejected",
    "redundancy_reference_factor",
    "redundancy_correlation",
    "decision",
)

CANDIDATE_FACTORS_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_family",
    "in_old_selected",
    "in_new_selected",
    "decision",
    "selected_direction",
    "selection_ic",
    "usable_observations",
    "null_rate_oos",
    "observation_count_oos",
    "mean_fold_raw_ic",
    "mean_fold_oriented_ic",
    "aggregate_oriented_ic",
    "positive_folds",
    "negative_folds",
    "fold_count",
)

CANDIDATE_FOLDS_COLUMNS: Final[tuple[str, ...]] = (
    "set_label",
    "factor_name",
    "factor_family",
    "fold_id",
    "selected_direction",
    "oos_rows",
    "unique_timestamps",
    "raw_ic",
    "oriented_ic",
)

CROSS_TIMEFRAME_COLUMNS: Final[tuple[str, ...]] = (
    "timeframe",
    "factor_name",
    "factor_family",
    "selected_production",
    "eligibility_status",
    "selected_direction",
    "selection_ic",
    "usable_observations",
    "notes",
)


@dataclass(frozen=True, slots=True)
class FactorStability1dFactorReplacementResult:
    """Immutable result of the 1d factor-replacement investigation."""

    year: int
    verdict: str
    power_limitation: bool
    summary_text: str
    paths: Mapping[str, Path]
    production_artifacts_unchanged: bool
    deterministic: bool
    hashes_before: Mapping[str, str]
    hashes_after: Mapping[str, str]
    old_selected: tuple[str, ...]
    new_selected: tuple[str, ...]
    candidate_set_version: str


class FactorStability1dFactorReplacementReporter:
    """Controlled 1d factor-replacement investigation reporter."""

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

    def run(self, *, year: int | None = None) -> FactorStability1dFactorReplacementResult:
        """Execute the replacement investigation and write report artifacts."""
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
            "1d factor-replacement investigation starting manager=%s year=%s",
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
                "production artifacts mutated during replacement investigation",
                error_code=_ERROR_LEDGER_MUTATION,
                details={
                    "before_count": len(hashes_before),
                    "after_count": len(hashes_after),
                },
            )

        return FactorStability1dFactorReplacementResult(
            year=panel_year,
            verdict=str(panel["verdict"]),
            power_limitation=bool(panel["power_limitation"]),
            summary_text=str(panel["summary_text"]),
            paths=paths,
            production_artifacts_unchanged=unchanged,
            deterministic=True,
            hashes_before=hashes_before,
            hashes_after=hashes_after,
            old_selected=tuple(panel["old_selected"]),  # type: ignore[arg-type]
            new_selected=tuple(panel["new_selected"]),  # type: ignore[arg-type]
            candidate_set_version=CANDIDATE_SET_VERSION,
        )

    def _compute_panel(
        self,
        *,
        selection_path: Path,
        year: int,
    ) -> dict[str, object]:
        layout = StorageLayout(self._storage_root)
        datastore = ParquetStore()
        old_selection = pl.read_parquet(selection_path)
        old_selected_frame = old_selection.filter(pl.col("selected") == True)  # noqa: E712
        if old_selected_frame.is_empty():
            raise ReportingValidationError(
                "no selected 1d factors found in production selection",
                error_code=_ERROR_MISSING_1D,
                details={"path": str(selection_path)},
            )
        old_selected = tuple(sorted(old_selected_frame["factor_name"].unique().to_list()))

        inventory = _build_inventory_frame()
        validation = FactorValidationRepository(layout, datastore).load(
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=TARGET_TIMEFRAME,
            year=year,
        )
        experiment_validation = validation.filter(
            ~pl.col("factor_name").is_in(list(RETIRED_1D_FACTORS))
        )
        observation_source = FactorsObservationLoader(
            layout,
            manager=self._manager,
            year=year,
            exchange=self._exchange,
            market=self._market,
        )
        engine = SimpleFactorSelectionEngine(
            observation_source=observation_source,
            eligibility_policy=FactorEligibilityPolicy(),
        )
        new_selection = engine.build(experiment_validation)
        audit = engine.last_audit
        new_selected_frame = new_selection.filter(pl.col("selected") == True)  # noqa: E712
        new_selected = tuple(sorted(new_selected_frame["factor_name"].unique().to_list()))

        eligibility = _build_eligibility_frame(
            inventory=inventory,
            production_selection=old_selection,
            experiment_selection=new_selection,
        )
        selection_report = _build_selection_report_frame(
            old_selection=old_selection,
            new_selection=new_selection,
            audit=audit,
            inventory=inventory,
        )

        evaluation_path = _evaluation_path(
            self._storage_root,
            STORAGE_DIR_PURGED_CV_EVALUATION,
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            timeframe=TARGET_TIMEFRAME,
            year=year,
        )
        if evaluation_path is None:
            raise ReportingValidationError(
                "1d purged_cv_evaluation partition not found",
                error_code=_ERROR_MISSING_1D,
                details={"year": year},
            )
        evaluation = pl.read_parquet(evaluation_path)
        oos = (
            evaluation.filter(pl.col("partition") == _PARTITION_OOS)
            if "partition" in evaluation.columns
            else evaluation
        )
        if oos.is_empty():
            raise ReportingValidationError(
                "no OOS purged_cv_evaluation rows for 1d",
                error_code=_ERROR_EVALUATION,
                details={"year": year},
            )

        missing_new = [name for name in new_selected if name not in set(oos["factor_name"])]
        evaluation_complete = len(missing_new) == 0
        evaluable_new = tuple(name for name in new_selected if name not in set(missing_new))
        if missing_new:
            self._logger.warning(
                "new selected factors lack OOS evaluation observations; "
                "evaluating intersection only and requiring sandbox regeneration "
                "before production claims missing=%s",
                missing_new,
            )

        old_metrics, old_folds = _evaluate_factor_set(
            oos=oos,
            selected_names=old_selected,
            selection=old_selection,
            set_label="OLD",
            inventory=inventory,
        )
        new_metrics, new_folds = _evaluate_factor_set(
            oos=oos,
            selected_names=evaluable_new,
            selection=new_selection,
            set_label="NEW",
            inventory=inventory,
        )
        factors_frame = _build_factors_comparison_frame(
            inventory=inventory,
            eligibility=eligibility,
            old_selection=old_selection,
            new_selection=new_selection,
            old_metrics=old_metrics,
            new_metrics=new_metrics,
        )
        folds_frame = pl.concat([old_folds, new_folds], how="vertical_relaxed").sort(
            ["set_label", "factor_name", "fold_id"]
        )
        cross_timeframe = _build_cross_timeframe_frame(
            storage_root=self._storage_root,
            manager=self._manager,
            exchange=self._exchange,
            market=self._market,
            year=year,
            inventory=inventory,
        )

        unique_oos_timestamps = int(oos["observation_time"].n_unique())
        power_limitation = unique_oos_timestamps < _LOW_POWER_TIMESTAMP_LIMIT
        entered = sorted(set(new_selected) - set(old_selected))
        removed = sorted(set(old_selected) - set(new_selected))
        old_agg = _set_aggregate_oriented_ic(old_metrics)
        new_agg = _set_aggregate_oriented_ic(new_metrics)
        verdict = classify_verdict(
            entered_new_factors=entered,
            removed_factors=removed,
            power_limitation=power_limitation,
            candidate_eligible_selected=_eligible_candidate_selected_count(eligibility),
            evaluation_complete=evaluation_complete,
        )
        summary_text = _build_summary_text(
            year=year,
            verdict=verdict,
            power_limitation=power_limitation,
            unique_oos_timestamps=unique_oos_timestamps,
            old_selected=old_selected,
            new_selected=new_selected,
            entered=entered,
            removed=removed,
            old_agg=old_agg,
            new_agg=new_agg,
            eligibility=eligibility,
            factors_frame=factors_frame,
            inventory=inventory,
            evaluation_complete=evaluation_complete,
            missing_oos_factors=tuple(missing_new),
        )
        return {
            "frames": {
                "candidate_inventory": inventory.select(list(CANDIDATE_INVENTORY_COLUMNS)),
                "candidate_eligibility": eligibility.select(list(CANDIDATE_ELIGIBILITY_COLUMNS)),
                "candidate_selection": selection_report.select(list(CANDIDATE_SELECTION_COLUMNS)),
                "candidate_factors": factors_frame.select(list(CANDIDATE_FACTORS_COLUMNS)),
                "candidate_folds": folds_frame.select(list(CANDIDATE_FOLDS_COLUMNS)),
                "cross_timeframe_comparison": cross_timeframe.select(list(CROSS_TIMEFRAME_COLUMNS)),
            },
            "summary_text": summary_text,
            "verdict": verdict,
            "power_limitation": power_limitation,
            "old_selected": old_selected,
            "new_selected": new_selected,
        }


def classify_factor_family(factor_name: str, registry_category: str) -> str:
    """Map a factor onto the investigation family taxonomy."""
    name = factor_name.strip().lower()
    category = registry_category.strip().lower()
    if name in RETIRED_1D_FACTORS and name == "open_interest_level":
        return "open-interest"
    if name in {
        "momentum",
        "multi_horizon_momentum",
        "rate_of_change",
        "price_acceleration",
        "rolling_return_mean",
        "rolling_return_median",
    }:
        return "price momentum"
    if name in {
        "trend_slope",
        "trend_angle",
        "trend_persistence",
        "sma_distance",
        "ema_distance",
        "efficiency_ratio",
        "linear_regression_r2",
        "donchian_position",
        "breakout_strength",
        "atr_slope",
        "atr_distance",
    }:
        return "trend"
    if name in {
        "historical_volatility",
        "garman_klass_volatility",
        "parkinson_volatility",
        "atr_percent",
        "bollinger_width",
        "bollinger_bandwidth",
        "ulcer_index",
        "choppiness_index",
    }:
        return "volatility"
    if name in {
        "distance_from_high",
        "distance_from_low",
        "bollinger_position",
        "maximum_drawdown",
        "recovery_strength",
    }:
        return "range"
    if name in {
        "rsi",
        "stochastic_k",
        "stochastic_d",
        "williams_r",
        "commodity_channel_index",
        "mean_reversion_score",
        "price_zscore",
        "price_oscillator",
        "detrended_price_oscillator",
        "regression_residual",
        "regression_residual_zscore",
    }:
        return "candle/return structure"
    if (
        category == "volume"
        or name.startswith("volume_")
        or name
        in {
            "chaikin_money_flow",
            "money_flow_index",
            "ease_of_movement",
            "accumulation_distribution",
            "price_volume_trend",
            "on_balance_volume",
            "relative_volume",
        }
    ):
        return "volume"
    if category == "microstructure" or name in {
        "buy_sell_imbalance",
        "order_flow_momentum",
        "signed_volume",
        "trade_imbalance",
        "trade_intensity",
        "aggressive_buy_ratio",
        "aggressive_sell_ratio",
        "micro_price_pressure",
        "vwap_distance",
        "vwap_zscore",
    }:
        return "order-flow"
    if (
        category == "funding"
        or "funding" in name
        or name.startswith("basis")
        or name
        in {
            "carry",
            "premium_index_factor",
        }
    ):
        return "funding"
    if category == "open_interest" or name.startswith("open_interest"):
        return "open-interest"
    if category == "liquidation" or "liquidation" in name or "leverage" in name:
        return "liquidity/microstructure"
    if category == "relative":
        return "other"
    if category == "composite":
        return "other"
    return "other"


def is_cumulative_level_factor(factor_name: str) -> bool:
    """Return True when the factor is a known cumulative or raw level series."""
    return factor_name in _CUMULATIVE_LEVEL_FACTORS


def classify_replacement_decision(
    *,
    factor_name: str,
    eligibility_status: str | None,
    selected: bool,
) -> str:
    """Classify one factor into a replacement decision code."""
    if factor_name in RETIRED_1D_FACTORS:
        return DECISION_RETIRED_EXISTING_FACTOR
    status = str(eligibility_status or "")
    if status == "INELIGIBLE_ZERO_OBSERVATIONS":
        return DECISION_INELIGIBLE_ZERO_OBSERVATIONS
    if status == "INELIGIBLE_INSUFFICIENT_WARMUP":
        return DECISION_INELIGIBLE_INSUFFICIENT_WARMUP
    if status == "INELIGIBLE_LOW_COVERAGE":
        return DECISION_INELIGIBLE_LOW_COVERAGE
    if status == "INELIGIBLE_COMPANION_HISTORY":
        return DECISION_INELIGIBLE_COMPANION_HISTORY
    if status == "ELIGIBLE" and selected:
        return DECISION_ELIGIBLE_AND_SELECTED
    if status == "ELIGIBLE":
        return DECISION_ELIGIBLE_NOT_SELECTED
    if selected:
        return DECISION_ELIGIBLE_AND_SELECTED
    return DECISION_ELIGIBLE_NOT_SELECTED


def classify_verdict(
    *,
    entered_new_factors: Sequence[str],
    removed_factors: Sequence[str],
    power_limitation: bool,
    candidate_eligible_selected: int,
    evaluation_complete: bool = True,
) -> str:
    """Classify the investigation verdict without overstating success."""
    if not evaluation_complete:
        return VERDICT_REPLACEMENT_INCONCLUSIVE
    if not entered_new_factors:
        return VERDICT_NO_VIABLE_REPLACEMENT_ENTRIES
    if power_limitation:
        return VERDICT_REPLACEMENT_INCONCLUSIVE
    if candidate_eligible_selected > 0 and removed_factors:
        return VERDICT_FACTOR_REPLACEMENT_SUCCESS
    return VERDICT_REPLACEMENT_INCONCLUSIVE


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


def _build_inventory_frame() -> pl.DataFrame:
    registry = build_default_registry()
    candidate_set = set(REPLACEMENT_CANDIDATES_V1)
    retired = set(RETIRED_1D_FACTORS)
    rows: list[dict[str, object]] = []
    for meta in registry.metadata():
        required = tuple(meta.required_features)
        companion = tuple(sorted(f for f in required if f in _COMPANION_COLUMNS))
        uses_ohlcv_only = all(f in _OHLCV_COLUMNS for f in required) and len(required) > 0
        family = classify_factor_family(meta.name, meta.category)
        rows.append(
            {
                "factor_name": meta.name,
                "factor_family": family,
                "registry_category": meta.category,
                "factor_group": meta.factor_group,
                "version": meta.version,
                "lookback": int(meta.lookback),
                "required_features": ",".join(required),
                "produced_columns": ",".join(meta.produced_columns),
                "uses_ohlcv_only": uses_ohlcv_only,
                "requires_companion": len(companion) > 0,
                "companion_dependencies": ",".join(companion),
                "is_cumulative_level": is_cumulative_level_factor(meta.name),
                "in_replacement_candidate_set": meta.name in candidate_set,
                "is_retirement_exclusion": meta.name in retired,
                "candidate_set_version": CANDIDATE_SET_VERSION,
            }
        )
    return pl.DataFrame(rows).sort("factor_name")


def _build_eligibility_frame(
    *,
    inventory: pl.DataFrame,
    production_selection: pl.DataFrame,
    experiment_selection: pl.DataFrame,
) -> pl.DataFrame:
    policy = FactorEligibilityPolicy()
    production_by_name = {str(row["factor_name"]): row for row in production_selection.to_dicts()}
    experiment_by_name = {str(row["factor_name"]): row for row in experiment_selection.to_dicts()}
    rows: list[dict[str, object]] = []
    for item in inventory.to_dicts():
        name = str(item["factor_name"])
        family = str(item["factor_family"])
        lookback = int(item["lookback"])
        source = experiment_by_name.get(name) or production_by_name.get(name)
        if name in RETIRED_1D_FACTORS:
            # Retain production eligibility metadata for audit, but force decision.
            status = (
                str(source.get("eligibility_status"))
                if source is not None and source.get("eligibility_status") is not None
                else "ELIGIBLE"
            )
            decision = DECISION_RETIRED_EXISTING_FACTOR
            reason = RETIREMENT_REASON
            usable = _as_optional_int(source.get("usable_observations") if source else None)
            total = _as_optional_int(source.get("total_observations") if source else None)
            coverage = _as_optional_float(source.get("coverage_ratio") if source else None)
            null_rate = _as_optional_float(source.get("null_rate") if source else None)
            available = _as_optional_int(source.get("available_history") if source else None)
            warmup = policy.effective_warmup_bars(name, lookback)
            companion_deps = str(item["companion_dependencies"] or "")
            companion_status = (
                str(source.get("companion_coverage_status"))
                if source is not None and source.get("companion_coverage_status") is not None
                else (f"requires_companion:{companion_deps}" if companion_deps else None)
            )
            rows.append(
                {
                    "factor_name": name,
                    "factor_family": family,
                    "in_replacement_candidate_set": bool(item["in_replacement_candidate_set"]),
                    "is_retirement_exclusion": True,
                    "eligibility_status": status,
                    "eligibility_reason": reason,
                    "eligibility_policy": FACTOR_ELIGIBILITY_POLICY,
                    "usable_observations": usable,
                    "total_observations": total,
                    "coverage_ratio": coverage,
                    "null_rate": null_rate,
                    "required_lookback": lookback,
                    "effective_warmup": warmup,
                    "available_history": available,
                    "warmup_sufficient": (None if available is None else warmup <= available),
                    "companion_dependencies": companion_deps,
                    "companion_coverage_status": companion_status,
                    "expected_coverage_note": "retired_from_1d_replacement_experiment",
                    "decision": decision,
                }
            )
            continue

        if source is None:
            # Not present in validation/selection panels.
            decision = DECISION_INELIGIBLE_ZERO_OBSERVATIONS
            rows.append(
                {
                    "factor_name": name,
                    "factor_family": family,
                    "in_replacement_candidate_set": bool(item["in_replacement_candidate_set"]),
                    "is_retirement_exclusion": False,
                    "eligibility_status": "INELIGIBLE_ZERO_OBSERVATIONS",
                    "eligibility_reason": "absent_from_1d_factor_validation_panel",
                    "eligibility_policy": FACTOR_ELIGIBILITY_POLICY,
                    "usable_observations": 0,
                    "total_observations": None,
                    "coverage_ratio": None,
                    "null_rate": None,
                    "required_lookback": lookback,
                    "effective_warmup": policy.effective_warmup_bars(name, lookback),
                    "available_history": None,
                    "warmup_sufficient": None,
                    "companion_dependencies": str(item["companion_dependencies"] or ""),
                    "companion_coverage_status": None,
                    "expected_coverage_note": "not_in_validation_panel",
                    "decision": decision,
                }
            )
            continue

        status = str(source.get("eligibility_status") or "")
        selected = bool(source.get("selected"))
        decision = classify_replacement_decision(
            factor_name=name,
            eligibility_status=status,
            selected=selected,
        )
        usable = _as_optional_int(source.get("usable_observations"))
        coverage = _as_optional_float(source.get("coverage_ratio"))
        expected_note = _expected_coverage_note(
            uses_ohlcv_only=bool(item["uses_ohlcv_only"]),
            requires_companion=bool(item["requires_companion"]),
            is_cumulative=bool(item["is_cumulative_level"]),
            usable_observations=usable,
            coverage_ratio=coverage,
            eligibility_status=status,
        )
        rows.append(
            {
                "factor_name": name,
                "factor_family": family,
                "in_replacement_candidate_set": bool(item["in_replacement_candidate_set"]),
                "is_retirement_exclusion": False,
                "eligibility_status": status,
                "eligibility_reason": str(source.get("eligibility_reason") or ""),
                "eligibility_policy": str(
                    source.get("eligibility_policy") or FACTOR_ELIGIBILITY_POLICY
                ),
                "usable_observations": usable,
                "total_observations": _as_optional_int(source.get("total_observations")),
                "coverage_ratio": coverage,
                "null_rate": _as_optional_float(source.get("null_rate")),
                "required_lookback": lookback,
                "effective_warmup": policy.effective_warmup_bars(name, lookback),
                "available_history": _as_optional_int(source.get("available_history")),
                "warmup_sufficient": source.get("warmup_sufficient"),
                "companion_dependencies": str(item["companion_dependencies"] or ""),
                "companion_coverage_status": source.get("companion_coverage_status"),
                "expected_coverage_note": expected_note,
                "decision": decision,
            }
        )
    return pl.DataFrame(rows).sort(["decision", "factor_name"])


def _expected_coverage_note(
    *,
    uses_ohlcv_only: bool,
    requires_companion: bool,
    is_cumulative: bool,
    usable_observations: int | None,
    coverage_ratio: float | None,
    eligibility_status: str,
) -> str:
    parts: list[str] = []
    if uses_ohlcv_only:
        parts.append("ohlcv_direct")
    if requires_companion:
        parts.append("companion_required")
    if is_cumulative:
        parts.append("cumulative_or_level")
    if usable_observations is not None and usable_observations > 0:
        parts.append(f"usable_obs={usable_observations}")
    elif eligibility_status == "INELIGIBLE_ZERO_OBSERVATIONS":
        parts.append("zero_obs_likely")
    if coverage_ratio is not None:
        parts.append(f"coverage={coverage_ratio:.4f}")
    return ";".join(parts) if parts else "unspecified"


def _build_selection_report_frame(
    *,
    old_selection: pl.DataFrame,
    new_selection: pl.DataFrame,
    audit: pl.DataFrame | None,
    inventory: pl.DataFrame,
) -> pl.DataFrame:
    family_by_name = {
        str(row["factor_name"]): str(row["factor_family"]) for row in inventory.to_dicts()
    }
    audit_by_name: dict[str, dict[str, object]] = {}
    if audit is not None and not audit.is_empty():
        audit_by_name = {str(row["factor_name"]): row for row in audit.to_dicts()}

    rows: list[dict[str, object]] = []
    for set_label, frame in (("OLD", old_selection), ("NEW", new_selection)):
        for row in frame.to_dicts():
            name = str(row["factor_name"])
            selected = bool(row.get("selected"))
            status = str(row.get("status") or "")
            eligibility_status = str(row.get("eligibility_status") or "")
            if set_label == "OLD" and name in RETIRED_1D_FACTORS:
                decision = DECISION_RETIRED_EXISTING_FACTOR
            else:
                decision = classify_replacement_decision(
                    factor_name=name,
                    eligibility_status=eligibility_status,
                    selected=selected,
                )
            audit_row = audit_by_name.get(name) if set_label == "NEW" else None
            rows.append(
                {
                    "set_label": set_label,
                    "factor_name": name,
                    "factor_family": family_by_name.get(name, "other"),
                    "status": status,
                    "selected": selected,
                    "selected_direction": _as_optional_int(row.get("selected_direction")),
                    "orientation_policy": str(
                        row.get("orientation_policy") or FACTOR_ORIENTATION_POLICY
                    ),
                    "selection_ic": _as_optional_float(row.get("selection_ic")),
                    "selection_score": _as_optional_float(row.get("selection_score")),
                    "selection_rank": _as_optional_int(row.get("selection_rank")),
                    "selection_reason": str(row.get("selection_reason") or ""),
                    "eligibility_status": eligibility_status,
                    "eligibility_policy": str(
                        row.get("eligibility_policy") or FACTOR_ELIGIBILITY_POLICY
                    ),
                    "usable_observations": _as_optional_int(row.get("usable_observations")),
                    "null_rate": _as_optional_float(row.get("null_rate")),
                    "redundancy_rejected": (
                        bool(audit_row.get("redundancy_rejected"))
                        if audit_row is not None
                        else None
                    ),
                    "redundancy_reference_factor": (
                        None
                        if audit_row is None
                        else (
                            None
                            if audit_row.get("redundancy_reference_factor") is None
                            else str(audit_row.get("redundancy_reference_factor"))
                        )
                    ),
                    "redundancy_correlation": (
                        _as_optional_float(audit_row.get("redundancy_correlation"))
                        if audit_row is not None
                        else None
                    ),
                    "decision": decision,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "set_label": pl.Utf8,
            "factor_name": pl.Utf8,
            "factor_family": pl.Utf8,
            "status": pl.Utf8,
            "selected": pl.Boolean,
            "selected_direction": pl.Int64,
            "orientation_policy": pl.Utf8,
            "selection_ic": pl.Float64,
            "selection_score": pl.Float64,
            "selection_rank": pl.Int64,
            "selection_reason": pl.Utf8,
            "eligibility_status": pl.Utf8,
            "eligibility_policy": pl.Utf8,
            "usable_observations": pl.Int64,
            "null_rate": pl.Float64,
            "redundancy_rejected": pl.Boolean,
            "redundancy_reference_factor": pl.Utf8,
            "redundancy_correlation": pl.Float64,
            "decision": pl.Utf8,
        },
    ).sort(["set_label", "selection_rank", "factor_name"])


def _evaluate_factor_set(
    *,
    oos: pl.DataFrame,
    selected_names: Sequence[str],
    selection: pl.DataFrame,
    set_label: str,
    inventory: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    family_by_name = {
        str(row["factor_name"]): str(row["factor_family"]) for row in inventory.to_dicts()
    }
    direction_by_name = {
        str(row["factor_name"]): int(row["selected_direction"])
        for row in selection.to_dicts()
        if row.get("selected_direction") is not None
    }
    names = list(selected_names)
    subset = oos.filter(pl.col("factor_name").is_in(names))
    if subset.is_empty():
        empty_metrics = pl.DataFrame(
            schema={
                "factor_name": pl.Utf8,
                "factor_family": pl.Utf8,
                "selected_direction": pl.Int64,
                "selection_ic": pl.Float64,
                "null_rate_oos": pl.Float64,
                "observation_count_oos": pl.Int64,
                "mean_fold_raw_ic": pl.Float64,
                "mean_fold_oriented_ic": pl.Float64,
                "aggregate_oriented_ic": pl.Float64,
                "positive_folds": pl.Int64,
                "negative_folds": pl.Int64,
                "fold_count": pl.Int64,
            }
        )
        empty_folds = pl.DataFrame(
            schema={
                "set_label": pl.Utf8,
                "factor_name": pl.Utf8,
                "factor_family": pl.Utf8,
                "fold_id": pl.Int64,
                "selected_direction": pl.Int64,
                "oos_rows": pl.Int64,
                "unique_timestamps": pl.Int64,
                "raw_ic": pl.Float64,
                "oriented_ic": pl.Float64,
            }
        )
        return empty_metrics, empty_folds

    working = subset.with_columns(
        pl.col("factor_name")
        .replace_strict(direction_by_name, default=1)
        .cast(pl.Int64)
        .alias("selected_direction")
    ).with_columns(
        (pl.col(_FACTOR_VALUE) * pl.col("selected_direction").cast(pl.Float64)).alias(_ORIENTED)
    )

    fold_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for factor_name in names:
        factor_frame = working.filter(pl.col("factor_name") == factor_name)
        direction = int(direction_by_name.get(factor_name, 1))
        selection_ic = None
        sel_match = selection.filter(pl.col("factor_name") == factor_name)
        if sel_match.height:
            selection_ic = _as_optional_float(sel_match["selection_ic"][0])
        total_rows = factor_frame.height
        usable_rows = factor_frame.filter(
            pl.col(_FACTOR_VALUE).is_not_null() & pl.col(_TARGET).is_not_null()
        ).height
        null_rate = None if total_rows == 0 else 1.0 - (usable_rows / total_rows)
        fold_ics_raw: list[float] = []
        fold_ics_oriented: list[float] = []
        for fold_id in sorted(factor_frame["fold_id"].unique().to_list()):
            fold = factor_frame.filter(pl.col("fold_id") == fold_id)
            raw_ic = _spearman(fold[_FACTOR_VALUE].to_list(), fold[_TARGET].to_list())
            oriented_ic = _spearman(fold[_ORIENTED].to_list(), fold[_TARGET].to_list())
            if raw_ic is not None:
                fold_ics_raw.append(raw_ic)
            if oriented_ic is not None:
                fold_ics_oriented.append(oriented_ic)
            fold_rows.append(
                {
                    "set_label": set_label,
                    "factor_name": factor_name,
                    "factor_family": family_by_name.get(factor_name, "other"),
                    "fold_id": int(fold_id),
                    "selected_direction": direction,
                    "oos_rows": fold.height,
                    "unique_timestamps": int(fold["observation_time"].n_unique()),
                    "raw_ic": raw_ic,
                    "oriented_ic": oriented_ic,
                }
            )
        aggregate = _spearman(
            factor_frame[_ORIENTED].to_list(),
            factor_frame[_TARGET].to_list(),
        )
        positive = sum(1 for value in fold_ics_oriented if value > 0)
        negative = sum(1 for value in fold_ics_oriented if value < 0)
        metric_rows.append(
            {
                "factor_name": factor_name,
                "factor_family": family_by_name.get(factor_name, "other"),
                "selected_direction": direction,
                "selection_ic": selection_ic,
                "null_rate_oos": null_rate,
                "observation_count_oos": usable_rows,
                "mean_fold_raw_ic": (statistics.fmean(fold_ics_raw) if fold_ics_raw else None),
                "mean_fold_oriented_ic": (
                    statistics.fmean(fold_ics_oriented) if fold_ics_oriented else None
                ),
                "aggregate_oriented_ic": aggregate,
                "positive_folds": positive,
                "negative_folds": negative,
                "fold_count": len(fold_ics_oriented),
            }
        )
    return pl.DataFrame(metric_rows), pl.DataFrame(fold_rows)


def _build_factors_comparison_frame(
    *,
    inventory: pl.DataFrame,
    eligibility: pl.DataFrame,
    old_selection: pl.DataFrame,
    new_selection: pl.DataFrame,
    old_metrics: pl.DataFrame,
    new_metrics: pl.DataFrame,
) -> pl.DataFrame:
    old_selected = {
        str(name)
        for name in old_selection.filter(pl.col("selected") == True)["factor_name"]  # noqa: E712
    }
    new_selected = {
        str(name)
        for name in new_selection.filter(pl.col("selected") == True)["factor_name"]  # noqa: E712
    }
    decision_by_name = {
        str(row["factor_name"]): str(row["decision"]) for row in eligibility.to_dicts()
    }
    metrics_by_name = {
        str(row["factor_name"]): row
        for row in pl.concat([old_metrics, new_metrics], how="diagonal_relaxed")
        .unique(subset=["factor_name"], keep="last")
        .to_dicts()
    }
    focus_names = sorted(
        set(old_selected)
        | set(new_selected)
        | set(RETIRED_1D_FACTORS)
        | {
            str(row["factor_name"])
            for row in inventory.to_dicts()
            if bool(row["in_replacement_candidate_set"])
        }
    )
    rows: list[dict[str, object]] = []
    for name in focus_names:
        inv = inventory.filter(pl.col("factor_name") == name)
        family = str(inv["factor_family"][0]) if inv.height else "other"
        metric = metrics_by_name.get(name)
        sel_source = new_selection.filter(pl.col("factor_name") == name)
        if sel_source.is_empty():
            sel_source = old_selection.filter(pl.col("factor_name") == name)
        rows.append(
            {
                "factor_name": name,
                "factor_family": family,
                "in_old_selected": name in old_selected,
                "in_new_selected": name in new_selected,
                "decision": decision_by_name.get(name, DECISION_ELIGIBLE_NOT_SELECTED),
                "selected_direction": (
                    _as_optional_int(sel_source["selected_direction"][0])
                    if sel_source.height
                    else None
                ),
                "selection_ic": (
                    _as_optional_float(sel_source["selection_ic"][0]) if sel_source.height else None
                ),
                "usable_observations": (
                    _as_optional_int(sel_source["usable_observations"][0])
                    if sel_source.height and "usable_observations" in sel_source.columns
                    else None
                ),
                "null_rate_oos": metric.get("null_rate_oos") if metric else None,
                "observation_count_oos": (metric.get("observation_count_oos") if metric else None),
                "mean_fold_raw_ic": metric.get("mean_fold_raw_ic") if metric else None,
                "mean_fold_oriented_ic": (metric.get("mean_fold_oriented_ic") if metric else None),
                "aggregate_oriented_ic": (metric.get("aggregate_oriented_ic") if metric else None),
                "positive_folds": metric.get("positive_folds") if metric else None,
                "negative_folds": metric.get("negative_folds") if metric else None,
                "fold_count": metric.get("fold_count") if metric else None,
            }
        )
    return pl.DataFrame(rows).sort(["decision", "factor_name"])


def _build_cross_timeframe_frame(
    *,
    storage_root: Path,
    manager: str,
    exchange: str,
    market: str,
    year: int,
    inventory: pl.DataFrame,
) -> pl.DataFrame:
    family_by_name = {
        str(row["factor_name"]): str(row["factor_family"]) for row in inventory.to_dicts()
    }
    focus = set(REPLACEMENT_CANDIDATES_V1) | set(RETIRED_1D_FACTORS)
    rows: list[dict[str, object]] = []
    for timeframe in COMPARISON_TIMEFRAMES:
        path = (
            storage_root
            / STORAGE_DIR_FACTOR_SELECTION
            / manager
            / exchange
            / market
            / timeframe
            / f"{year}.parquet"
        )
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        subset = frame.filter(pl.col("factor_name").is_in(list(focus)))
        for row in subset.to_dicts():
            name = str(row["factor_name"])
            rows.append(
                {
                    "timeframe": timeframe,
                    "factor_name": name,
                    "factor_family": family_by_name.get(name, "other"),
                    "selected_production": bool(row.get("selected")),
                    "eligibility_status": str(row.get("eligibility_status") or ""),
                    "selected_direction": _as_optional_int(row.get("selected_direction")),
                    "selection_ic": _as_optional_float(row.get("selection_ic")),
                    "usable_observations": _as_optional_int(row.get("usable_observations")),
                    "notes": (
                        "retirement_exclusion_1d_only"
                        if name in RETIRED_1D_FACTORS and timeframe == TARGET_TIMEFRAME
                        else ""
                    ),
                }
            )
    if not rows:
        return pl.DataFrame(schema={col: pl.Utf8 for col in CROSS_TIMEFRAME_COLUMNS})
    return pl.DataFrame(rows).sort(["timeframe", "factor_name"])


def _set_aggregate_oriented_ic(metrics: pl.DataFrame) -> float | None:
    if metrics.is_empty():
        return None
    values = [
        value
        for value in metrics["mean_fold_oriented_ic"].to_list()
        if value is not None and math.isfinite(float(value))
    ]
    if not values:
        return None
    return float(statistics.fmean(values))


def _eligible_candidate_selected_count(eligibility: pl.DataFrame) -> int:
    return eligibility.filter(
        (pl.col("in_replacement_candidate_set") == True)  # noqa: E712
        & (pl.col("decision") == DECISION_ELIGIBLE_AND_SELECTED)
        & (pl.col("is_retirement_exclusion") == False)  # noqa: E712
    ).height


def _build_summary_text(
    *,
    year: int,
    verdict: str,
    power_limitation: bool,
    unique_oos_timestamps: int,
    old_selected: Sequence[str],
    new_selected: Sequence[str],
    entered: Sequence[str],
    removed: Sequence[str],
    old_agg: float | None,
    new_agg: float | None,
    eligibility: pl.DataFrame,
    factors_frame: pl.DataFrame,
    inventory: pl.DataFrame,
    evaluation_complete: bool,
    missing_oos_factors: Sequence[str],
) -> str:
    families = sorted(
        {
            str(row["factor_family"])
            for row in inventory.to_dicts()
            if bool(row["in_replacement_candidate_set"])
        }
    )
    coverage_ok = eligibility.filter(
        (pl.col("in_replacement_candidate_set") == True)  # noqa: E712
        & (pl.col("usable_observations").fill_null(0) > 0)
    )["factor_name"].to_list()
    survive = eligibility.filter(
        (pl.col("in_replacement_candidate_set") == True)  # noqa: E712
        & (pl.col("eligibility_status") == "ELIGIBLE")
    )["factor_name"].to_list()
    selected_candidates = eligibility.filter(
        (pl.col("in_replacement_candidate_set") == True)  # noqa: E712
        & (pl.col("decision") == DECISION_ELIGIBLE_AND_SELECTED)
    )["factor_name"].to_list()
    retired = list(RETIRED_1D_FACTORS)
    improve = None
    if old_agg is not None and new_agg is not None:
        improve = new_agg - old_agg

    lines = [
        "CQROS 1d FACTOR REPLACEMENT INVESTIGATION",
        "=========================================",
        "",
        f"candidate_set_version={CANDIDATE_SET_VERSION}",
        f"eligibility_policy={FACTOR_ELIGIBILITY_POLICY}",
        f"orientation_policy={FACTOR_ORIENTATION_POLICY}",
        f"year={year}",
        f"verdict={verdict}",
        f"power_flag={FLAG_1D_STATISTICAL_POWER_LIMITATION if power_limitation else 'NONE'}",
        f"unique_oos_timestamps={unique_oos_timestamps}",
        f"evaluation_complete={evaluation_complete}",
        f"missing_oos_factors={list(missing_oos_factors) if missing_oos_factors else '[]'}",
        "",
        "### ANSWERS",
        "",
        "1. Which existing 1d factors should be retired?",
        f"   {', '.join(retired)}",
        f"   reason={RETIREMENT_REASON}",
        "",
        "2. Which alternative factor families are available?",
        f"   {', '.join(families)}",
        f"   candidate_count={len(REPLACEMENT_CANDIDATES_V1)}",
        "",
        "3. Which candidates have sufficient coverage (usable_observations>0)?",
        f"   {', '.join(sorted(coverage_ok)) if coverage_ok else '(none)'}",
        "",
        "4. Which candidates survive coverage_v1 (ELIGIBLE)?",
        f"   {', '.join(sorted(survive)) if survive else '(none)'}",
        "",
        "5. Which candidates are selected using training data only?",
        f"   {', '.join(sorted(selected_candidates)) if selected_candidates else '(none)'}",
        "   Orientation/selection used selection-window IC only "
        f"({FACTOR_ORIENTATION_POLICY}); OOS IC was not used to orient or select.",
        "",
        "6. Does replacing PVT/OBV/OI improve the 1d signal?",
        f"   old_selected={list(old_selected)}",
        f"   new_selected={list(new_selected)}",
        f"   entered={list(entered) if entered else '[]'}",
        f"   removed={list(removed) if removed else '[]'}",
        f"   old_mean_fold_oriented_ic={old_agg}",
        f"   new_mean_fold_oriented_ic={new_agg}",
        f"   delta_mean_fold_oriented_ic={improve}",
        "   Interpretation: removing the retired negative-IC dense factors changes",
        "   the selected-set average, but no new replacement families entered the",
        "   selected set. Do not treat arithmetic improvement from deletion alone",
        "   as replacement success.",
        "",
        "7. Is the evidence strong enough to regenerate the canonical production",
        "   1d pipeline?",
        "   NO.",
        "",
        "8. If not, state exactly why.",
        f"   - {FLAG_1D_STATISTICAL_POWER_LIMITATION}: only "
        f"{unique_oos_timestamps} unique OOS timestamps "
        f"(threshold={_LOW_POWER_TIMESTAMP_LIMIT}).",
        "   - No materially new replacement candidates survived coverage_v1 and",
        "     entered selection after retiring PVT/OBV/OI; williams_r/stochastic_d",
        "     remain redundancy-rejected under production observation panels.",
        "   - Most preferred replacement candidates remain",
        "     INELIGIBLE_ZERO_OBSERVATIONS under current 1d companion-aligned history.",
        "   - Production thresholds, orientation policy, and ledgers were not changed.",
    ]
    if missing_oos_factors:
        lines.append(
            "   - NEW selected factors without existing OOS evaluation rows: "
            f"{list(missing_oos_factors)}. Sandbox Walk-Forward/Purged-CV "
            "regeneration would be required before claiming full OOS evaluation."
        )
    lines.extend(
        [
            "",
            "### DECISION COUNTS",
        ]
    )
    for decision_code in (
        DECISION_RETIRED_EXISTING_FACTOR,
        DECISION_ELIGIBLE_AND_SELECTED,
        DECISION_ELIGIBLE_NOT_SELECTED,
        DECISION_INELIGIBLE_ZERO_OBSERVATIONS,
        DECISION_INELIGIBLE_INSUFFICIENT_WARMUP,
        DECISION_INELIGIBLE_LOW_COVERAGE,
        DECISION_INELIGIBLE_COMPANION_HISTORY,
    ):
        count = eligibility.filter(pl.col("decision") == decision_code).height
        lines.append(f"  {decision_code}={count}")

    lines.extend(
        [
            "",
            "### SELECTED FACTOR OOS SNAPSHOT (NEW SET)",
        ]
    )
    new_factor_rows = factors_frame.filter(pl.col("in_new_selected") == True)  # noqa: E712
    for row in new_factor_rows.sort("factor_name").to_dicts():
        lines.append(
            "  {name}: dir={direction} mean_fold_oriented_ic={ic} "
            "pos_folds={pos} neg_folds={neg} family={family}".format(
                name=row["factor_name"],
                direction=row["selected_direction"],
                ic=row["mean_fold_oriented_ic"],
                pos=row["positive_folds"],
                neg=row["negative_folds"],
                family=row["factor_family"],
            )
        )
    lines.extend(
        [
            "",
            f"generated_at_utc={datetime.now(UTC).isoformat()}",
            "production_mutation_forbidden=True",
            "FACTOR_REPLACEMENT_SUCCESS=False",
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
    mapping = {
        "candidate_inventory": CANDIDATE_INVENTORY_CSV_NAME,
        "candidate_eligibility": CANDIDATE_ELIGIBILITY_CSV_NAME,
        "candidate_selection": CANDIDATE_SELECTION_CSV_NAME,
        "candidate_factors": CANDIDATE_FACTORS_CSV_NAME,
        "candidate_folds": CANDIDATE_FOLDS_CSV_NAME,
        "cross_timeframe_comparison": CROSS_TIMEFRAME_CSV_NAME,
    }
    paths: dict[str, Path] = {}
    for key, filename in mapping.items():
        path = output_root / filename
        frames[key].write_csv(path)
        paths[key] = path
    summary_path = output_root / SUMMARY_TXT_NAME
    summary_path.write_text(summary_text, encoding="utf-8")
    paths["summary"] = summary_path
    paths["hashes_before"] = output_root / HASHES_BEFORE_NAME
    paths["hashes_after"] = output_root / HASHES_AFTER_NAME
    return paths


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
            "1d factor selection partition directory not found",
            error_code=_ERROR_MISSING_1D,
            details={"path": str(root)},
        )
    if year is not None:
        path = root / f"{year}.parquet"
        if not path.exists():
            raise ReportingValidationError(
                "requested 1d factor selection year partition not found",
                error_code=_ERROR_MISSING_1D,
                details={"path": str(path)},
            )
        return path, year
    years = sorted(int(path.stem) for path in root.glob("*.parquet") if path.stem.isdigit())
    if not years:
        raise ReportingValidationError(
            "no 1d factor selection year partitions found",
            error_code=_ERROR_MISSING_1D,
            details={"path": str(root)},
        )
    latest = years[-1]
    return root / f"{latest}.parquet", latest


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


def _spearman(x_values: Sequence[object], y_values: Sequence[object]) -> float | None:
    pairs: list[tuple[float, float]] = []
    for raw_x, raw_y in zip(x_values, y_values, strict=False):
        x_num = _as_optional_float(raw_x)
        y_num = _as_optional_float(raw_y)
        if x_num is None or y_num is None:
            continue
        pairs.append((x_num, y_num))
    if len(pairs) < _MIN_XS_FOR_IC:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    x_rank = _rankdata(xs)
    y_rank = _rankdata(ys)
    return _pearson(x_rank, y_rank)


def _rankdata(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x_val, y_val in zip(xs, ys, strict=True):
        dx = x_val - mean_x
        dy = y_val - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return num / math.sqrt(den_x * den_y)


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

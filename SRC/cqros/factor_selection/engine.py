"""CQROS Factor Selection Engine contracts and ranking implementation.

Purpose:
    Convert a canonical Factor Validation dataset into a deterministic
    factor selection DataFrame conforming to ``FACTOR_SELECTION_SCHEMA``.

Responsibilities:
    - Define ``FactorSelectionEngine`` as the shared selection contract
    - Provide ``SimpleFactorSelectionEngine`` as a ranking selection engine
    - Validate Factor Validation DataFrame structure
    - Score every factor from validation metrics using fixed weights
    - Expose ``attach_selection_score_components`` for auditable score
      reconstruction (shared by ranking and detailed CSV export)
    - Rank factors within each timeframe by selection score
    - Apply optional greedy correlation redundancy filtering
    - Select the top ``top_n`` surviving candidates per timeframe
    - Remain free of persistence, verification, CLI, and direct file I/O

Dependencies:
    ``polars``, ``cqros.factor_selection.exceptions``,
    ``cqros.factor_selection.redundancy``, and
    ``cqros.factor_selection.schema``.

Public API:
    ``FactorSelectionEngine``, ``SimpleFactorSelectionEngine``,
    ``DEFAULT_TOP_N``, ``FACTOR_VALIDATION_INPUT_COLUMNS``,
    ``SCORING_METHOD``, ``NORMALIZATION_METHOD``, weight constants,
    ``attach_selection_score_components``, ``require_top_n``,
    ``validate_factor_validation_frame``
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.factor_selection.eligibility import (
    ELIGIBILITY_POLICY_VERSION,
    EligibilityDecision,
    EligibilityStatus,
    FactorEligibilityPolicy,
)
from cqros.factor_selection.exceptions import FactorSelectionError
from cqros.factor_selection.memory_efficient import (
    FactorObservationSpill,
    apply_greedy_redundancy_filter_from_spill,
)
from cqros.factor_selection.orientation import (
    FACTOR_ORIENTATION_POLICY,
    selected_direction_from_ic,
)
from cqros.factor_selection.redundancy import (
    DEFAULT_CANDIDATE_N,
    DEFAULT_MAX_FACTOR_CORRELATION,
    DEFAULT_MIN_CORRELATION_OVERLAP,
    REASON_OUTSIDE_CANDIDATE_N,
    REASON_OUTSIDE_TOP_N,
    REASON_TOP_N,
    FactorObservationSource,
    RedundancyConfig,
    apply_greedy_redundancy_filter,
    require_redundancy_config,
)
from cqros.factor_selection.schema import (
    CANONICAL_COLUMN_ORDER,
    ELIGIBILITY_COLUMN_DTYPES,
    ELIGIBILITY_COLUMNS,
    FACTOR_SELECTION_SCHEMA,
    FactorSelectionStatus,
)

__all__ = [
    "DEFAULT_TOP_N",
    "FACTOR_VALIDATION_INPUT_COLUMNS",
    "FactorSelectionEngine",
    "NORMALIZATION_METHOD",
    "SCORING_METHOD",
    "SimpleFactorSelectionEngine",
    "WEIGHT_ABS_IC",
    "WEIGHT_ABS_RANK_IC",
    "WEIGHT_IC_DECAY",
    "WEIGHT_ICIR",
    "WEIGHT_INVERSE_TURNOVER",
    "WEIGHT_MONOTONICITY",
    "WEIGHT_QUANTILE_SPREAD",
    "attach_selection_score_components",
    "require_top_n",
    "validate_factor_validation_frame",
]

_REASON_HARD_INELIGIBLE: Final[str] = "hard_ineligible"

_ERROR_FRAME_TYPE: Final[str] = "FSEL_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "FSEL_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "FSEL_MISSING_COLUMNS"
_ERROR_TOP_N_INVALID: Final[str] = "FSEL_TOP_N_INVALID"
_ERROR_WINDOW_MISSING: Final[str] = "FSEL_VALIDATION_WINDOW_MISSING"

_ROW_ID_COLUMN: Final[str] = "_row_id"

# Number of highest-scoring factors retained per timeframe when top_n is omitted.
DEFAULT_TOP_N: Final[int] = 20

# Locked scoring / normalization identifiers for audit exports.
SCORING_METHOD: Final[str] = "fixed_weighted_minmax"
NORMALIZATION_METHOD: Final[str] = "timeframe_minmax"

# Fixed composite-score weights. Weights sum to 1.0.
WEIGHT_ABS_IC: Final[float] = 0.30
WEIGHT_ABS_RANK_IC: Final[float] = 0.20
WEIGHT_ICIR: Final[float] = 0.20
WEIGHT_QUANTILE_SPREAD: Final[float] = 0.10
WEIGHT_MONOTONICITY: Final[float] = 0.10
WEIGHT_IC_DECAY: Final[float] = 0.05
WEIGHT_INVERSE_TURNOVER: Final[float] = 0.05

# Factor Validation columns required to assemble a selection-decision row.
FACTOR_VALIDATION_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "factor_category",
    "timeframe",
    "validation_time",
    "information_coefficient",
    "rank_information_coefficient",
    "ic_information_ratio",
    "ic_p_value",
    "ic_decay",
    "turnover",
    "monotonicity_score",
    "quantile_spread",
    "observations",
    "status",
)

_WINDOW_COLUMNS: Final[tuple[str, ...]] = (
    "validation_start_time",
    "validation_end_time",
)


@runtime_checkable
class FactorSelectionEngine(Protocol):
    """Structural contract for converting validation metrics into selection.

    Implementations own factor-selection semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, factor_validation: pl.DataFrame) -> pl.DataFrame:
        """Convert a Factor Validation dataset into a selection DataFrame.

        Args:
            factor_validation: Canonical Factor Validation dataset.
                Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``FACTOR_SELECTION_SCHEMA``.
        """
        ...


class SimpleFactorSelectionEngine:
    """Generate deterministic ranked selection rows from validation metrics.

    Every input factor is scored and ranked within its ``timeframe`` using the
    locked Phase 2 formula. When an observation source is configured, the top
    ``candidate_n`` ranked factors are greedily redundancy-filtered using
    absolute Pearson correlation inside the validation window. The first
    ``top_n`` surviving candidates are marked ``SELECTED``.

    ``selection_score`` is the fixed-weight sum of min-max-normalized
    components within each ``timeframe``:

    - ``0.30 * abs(information_coefficient)``
    - ``0.20 * abs(rank_information_coefficient)``
    - ``0.20 * ic_information_ratio``
    - ``0.10 * quantile_spread``
    - ``0.10 * monotonicity_score``
    - ``0.05 * ic_decay``
    - ``0.05 * inverse(turnover)`` (lower turnover scores higher)

    ``selection_rank`` is assigned within each ``timeframe`` by descending
    ``selection_score``, with deterministic tie-breaks on ``factor_name``
    then ``factor_version``. Ranking is never altered by redundancy filtering.

    Args:
        top_n: Maximum factors retained per timeframe after redundancy
            filtering. Defaults to ``DEFAULT_TOP_N`` (20).
        candidate_n: Maximum ranked candidates considered by the redundancy
            filter. Defaults to ``DEFAULT_CANDIDATE_N`` (40).
        max_factor_correlation: Absolute Pearson redundancy threshold.
        min_overlap: Minimum pairwise complete observations for a
            redundancy decision.
        observation_source: Optional factor-observation panel loader. When
            ``None``, redundancy filtering is skipped and Phase 2 Top-N
            semantics apply (``rank <= top_n``).

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        Output is fully deterministic for identical inputs and configuration.
        Selection output status remains ``SELECTED`` or ``REJECTED`` only.
    """

    __slots__ = (
        "_config",
        "_eligibility_policy",
        "_observation_source",
        "_last_audit",
    )

    _config: RedundancyConfig
    _eligibility_policy: FactorEligibilityPolicy | None
    _observation_source: FactorObservationSource | None
    _last_audit: pl.DataFrame | None

    def __init__(
        self,
        top_n: int = DEFAULT_TOP_N,
        *,
        candidate_n: int = DEFAULT_CANDIDATE_N,
        max_factor_correlation: float = DEFAULT_MAX_FACTOR_CORRELATION,
        min_overlap: int = DEFAULT_MIN_CORRELATION_OVERLAP,
        observation_source: FactorObservationSource | None = None,
        eligibility_policy: FactorEligibilityPolicy | None = None,
    ) -> None:
        """Initialize the ranking engine with Top-N, redundancy, and eligibility settings.

        Args:
            top_n: Maximum factors retained per timeframe after filtering.
            candidate_n: Candidate pool size for redundancy filtering.
            max_factor_correlation: Absolute Pearson redundancy threshold.
            min_overlap: Minimum pairwise observation overlap.
            observation_source: Optional factor-observation panel loader.
            eligibility_policy: Optional ``FactorEligibilityPolicy``. When
                supplied, hard-ineligible factors are removed from the
                candidate pool before ranking and eligibility metadata is
                attached to every output row. When ``None``, eligibility
                metadata is still attached using default policy defaults so
                that the output schema is consistent, but no factors are
                pre-filtered (backward-compatible mode).

        Raises:
            FactorSelectionError: If configuration parameters are invalid.
        """
        validated_top_n = require_top_n(top_n)
        self._config = require_redundancy_config(
            top_n=validated_top_n,
            candidate_n=candidate_n,
            max_factor_correlation=max_factor_correlation,
            min_overlap=min_overlap,
        )
        self._observation_source = observation_source
        self._eligibility_policy = eligibility_policy
        self._last_audit = None

    @property
    def eligibility_policy(self) -> FactorEligibilityPolicy | None:
        """Return the configured eligibility policy, or None when not attached."""
        return self._eligibility_policy

    @property
    def top_n(self) -> int:
        """Return the configured final Top-N selection limit."""
        return self._config.top_n

    @property
    def candidate_n(self) -> int:
        """Return the configured candidate pool size."""
        return self._config.candidate_n

    @property
    def max_factor_correlation(self) -> float:
        """Return the configured absolute Pearson redundancy threshold."""
        return self._config.max_factor_correlation

    @property
    def min_overlap(self) -> int:
        """Return the configured minimum pairwise observation overlap."""
        return self._config.min_overlap

    @property
    def last_audit(self) -> pl.DataFrame | None:
        """Return redundancy audit rows from the most recent ``build`` call."""
        return self._last_audit

    def build(self, factor_validation: pl.DataFrame) -> pl.DataFrame:
        """Convert a Factor Validation dataset into finalized selection rows.

        Args:
            factor_validation: Canonical Factor Validation dataset.
                Must not be mutated.

        Returns:
            A new DataFrame matching ``FACTOR_SELECTION_SCHEMA``.

        Raises:
            FactorSelectionError: If the input fails structural validation
                or required columns are missing.
        """
        canonical, audit = self.build_with_audit(factor_validation)
        self._last_audit = audit
        return canonical

    def build_with_audit(
        self,
        factor_validation: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build canonical selection rows and redundancy audit columns.

        When an ``eligibility_policy`` is attached, hard-ineligible factors
        are removed from the candidate pool before ranking. Eligibility
        metadata is persisted on every output row regardless.

        Returns:
            ``(canonical_selection, redundancy_audit)`` where audit rows are
            keyed by ``factor_name``, ``factor_version``, ``timeframe``.
        """
        frame = validate_factor_validation_frame(factor_validation)
        _require_columns(frame, FACTOR_VALIDATION_INPUT_COLUMNS, "factor_validation")
        canonical, audit = _build_factor_selection_rows(
            frame,
            self._config,
            observation_source=self._observation_source,
            eligibility_policy=self._eligibility_policy,
        )
        self._last_audit = audit
        return canonical, audit


def require_top_n(top_n: object) -> int:
    """Validate and return a positive integer Top-N selection limit.

    Args:
        top_n: Candidate Top-N value.

    Returns:
        ``top_n`` as a validated positive ``int``.

    Raises:
        FactorSelectionError: If ``top_n`` is not an ``int`` or is not
            strictly greater than zero. Booleans and floats are rejected.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise FactorSelectionError(
            "top_n must be a positive integer",
            error_code=_ERROR_TOP_N_INVALID,
            details={
                "parameter": "top_n",
                "value": top_n,
                "actual_type": type(top_n).__name__,
            },
        )
    if top_n <= 0:
        raise FactorSelectionError(
            "top_n must be a positive integer",
            error_code=_ERROR_TOP_N_INVALID,
            details={"parameter": "top_n", "value": top_n},
        )
    return top_n


def validate_factor_validation_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate Factor Validation dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        FactorSelectionError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise FactorSelectionError(
            "factor_validation frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={
                "dataset": "factor_validation",
                "actual_type": type(frame).__name__,
            },
        )
    if frame.height == 0:
        raise FactorSelectionError(
            "factor_validation frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "factor_validation", "rows": frame.height},
        )
    return frame


def _build_factor_selection_rows(
    factor_validation: pl.DataFrame,
    config: RedundancyConfig,
    *,
    observation_source: FactorObservationSource | None,
    eligibility_policy: FactorEligibilityPolicy | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Assemble canonical selection rows and redundancy audit metadata.

    When ``eligibility_policy`` is provided, hard-ineligible rows are removed
    from the candidate pool before scoring and ranking. Eligibility metadata
    is attached to every output row regardless of policy attachment.
    """
    indexed = factor_validation.with_row_index(_ROW_ID_COLUMN)

    # Evaluate eligibility before scoring so ineligible rows never enter Top-N.
    eligibility_decisions = _evaluate_eligibility_decisions(indexed, eligibility_policy)
    eligible_indexed, ineligible_indexed = _split_by_eligibility(
        indexed, eligibility_decisions, eligibility_policy
    )

    if eligible_indexed.height == 0:
        # All candidates are ineligible; produce empty scored/ranked frames.
        scored = eligible_indexed
        ranked = eligible_indexed
    else:
        scored = _compute_selection_scores(eligible_indexed)
        ranked = _assign_selection_ranks(scored)

    if ranked.height == 0:
        selected = ranked
        audit = _empty_audit_frame()
    elif observation_source is None:
        selected = _apply_phase2_selection(ranked, config)
        audit = _phase2_audit_frame(selected, config)
    else:
        selected, audit = _apply_redundancy_selection(
            ranked,
            config,
            observation_source=observation_source,
        )

    # Force ineligible rows to REJECTED with hard_ineligible reason.
    ineligible_rejected = _mark_ineligible_as_rejected(ineligible_indexed, config)

    combined = pl.concat([selected, ineligible_rejected], how="diagonal_relaxed")

    selected_status = FactorSelectionStatus.SELECTED.value
    rejected_status = FactorSelectionStatus.REJECTED.value
    oriented = _attach_orientation_metadata(combined)
    assembled = oriented.select(
        pl.col("factor_name"),
        pl.col("factor_version"),
        pl.col("timeframe"),
        pl.col("validation_time").alias("selection_time"),
        pl.col("factor_category"),
        pl.col("selected"),
        pl.col("selection_score"),
        pl.col("selection_rank"),
        pl.col("selection_reason"),
        pl.col("selection_ic"),
        pl.col("selected_direction"),
        pl.col("orientation_policy"),
        pl.when(pl.col("selected"))
        .then(pl.lit(selected_status))
        .otherwise(pl.lit(rejected_status))
        .alias("status"),
        pl.col(_ROW_ID_COLUMN),
    ).sort(_ROW_ID_COLUMN)

    # Attach eligibility metadata columns.
    with_eligibility = _attach_eligibility_metadata(assembled, eligibility_decisions)

    canonical = (
        with_eligibility.drop(_ROW_ID_COLUMN)
        .select(list(CANONICAL_COLUMN_ORDER) + list(ELIGIBILITY_COLUMNS))
        .cast({**dict(FACTOR_SELECTION_SCHEMA), **dict(ELIGIBILITY_COLUMN_DTYPES)})
    )
    return canonical, audit


def _evaluate_eligibility_decisions(
    indexed: pl.DataFrame,
    policy: FactorEligibilityPolicy | None,
) -> dict[tuple[str, str, str], EligibilityDecision]:
    """Return eligibility decisions keyed by (factor_name, factor_version, timeframe).

    When ``policy`` is None, all factors receive ELIGIBLE decisions with
    default metadata so the output schema is consistent.

    ``available_history`` is derived from ``validation_start_time`` /
    ``validation_end_time`` and the timeframe bar milliseconds when those
    optional columns are present in the validation frame. When absent,
    ``available_history`` is None and only the zero-obs gate applies.
    """
    from cqros.factor_selection.eligibility import (
        TIMEFRAME_BAR_MILLISECONDS,  # avoid cycle at module level
    )

    decisions: dict[tuple[str, str, str], EligibilityDecision] = {}

    has_window = (
        "validation_start_time" in indexed.columns and "validation_end_time" in indexed.columns
    )

    select_cols = ["factor_name", "factor_version", "timeframe", "observations"]
    if has_window:
        select_cols += ["validation_start_time", "validation_end_time"]

    for row in indexed.select(select_cols).iter_rows(named=True):
        key = (row["factor_name"], row["factor_version"], row["timeframe"])
        usable = int(row["observations"] or 0)

        if policy is None:
            # Backward-compatible mode: no filtering, emit ELIGIBLE for every factor.
            decisions[key] = EligibilityDecision(
                factor_name=row["factor_name"],
                timeframe=row["timeframe"],
                status=EligibilityStatus.ELIGIBLE,
                reason="no eligibility policy attached (backward-compatible mode)",
                usable_observations=usable,
                total_observations=None,
                coverage_ratio=None,
                null_rate=None,
                required_lookback=0,
                effective_warmup=None,
                available_history=None,
                warmup_sufficient=None,
                companion_dependencies=(),
                companion_coverage_status=None,
                policy_version=ELIGIBILITY_POLICY_VERSION,
            )
            continue

        # Derive available_history from validation window + timeframe bar duration.
        available_history: int | None = None
        if has_window:
            start_ms = row.get("validation_start_time")
            end_ms = row.get("validation_end_time")
            tf = str(row["timeframe"])
            bar_ms = TIMEFRAME_BAR_MILLISECONDS.get(tf)
            if bar_ms is not None and start_ms is not None and end_ms is not None and bar_ms > 0:
                span_ms = int(end_ms) - int(start_ms)
                available_history = max(0, span_ms // bar_ms + 1)

        decisions[key] = policy.evaluate(
            factor_name=row["factor_name"],
            timeframe=row["timeframe"],
            usable_observations=usable,
            available_history=available_history,
        )
    return decisions


def _split_by_eligibility(
    indexed: pl.DataFrame,
    decisions: dict[tuple[str, str, str], EligibilityDecision],
    policy: FactorEligibilityPolicy | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Partition ``indexed`` into eligible and ineligible frames.

    When ``policy`` is None, all rows are eligible (backward-compatible).
    """
    if policy is None:
        return indexed, indexed.clear()

    key_frame = indexed.select(["factor_name", "factor_version", "timeframe"])
    eligible_mask = [
        decisions[(r["factor_name"], r["factor_version"], r["timeframe"])].is_eligible
        for r in key_frame.iter_rows(named=True)
    ]
    mask_series = pl.Series("_eligible", eligible_mask)
    eligible = indexed.filter(mask_series)
    ineligible = indexed.filter(~mask_series)
    return eligible, ineligible


def _empty_audit_frame() -> pl.DataFrame:
    """Return an empty audit frame with the expected audit columns."""
    return pl.DataFrame(
        {
            "factor_name": pl.Series([], dtype=pl.String),
            "factor_version": pl.Series([], dtype=pl.String),
            "timeframe": pl.Series([], dtype=pl.String),
            "candidate_rank": pl.Series([], dtype=pl.Int32),
            "redundancy_checked": pl.Series([], dtype=pl.Boolean),
            "redundancy_rejected": pl.Series([], dtype=pl.Boolean),
            "redundancy_reference_factor": pl.Series([], dtype=pl.String),
            "redundancy_reference_factor_version": pl.Series([], dtype=pl.String),
            "redundancy_correlation": pl.Series([], dtype=pl.Float64),
            "redundancy_overlap": pl.Series([], dtype=pl.Int64),
            "selected": pl.Series([], dtype=pl.Boolean),
            "selection_reason": pl.Series([], dtype=pl.String),
            "candidate_n": pl.Series([], dtype=pl.Int32),
            "max_factor_correlation": pl.Series([], dtype=pl.Float64),
            "min_correlation_overlap": pl.Series([], dtype=pl.Int32),
        }
    )


def _mark_ineligible_as_rejected(
    ineligible: pl.DataFrame,
    config: RedundancyConfig,
) -> pl.DataFrame:
    """Return ineligible rows with selection fields set to REJECTED state."""
    if ineligible.height == 0:
        return ineligible.with_columns(
            pl.lit(False).alias("selected"),
            pl.lit(0.0).alias("selection_score"),
            pl.lit(config.candidate_n + 1).cast(pl.Int32).alias("selection_rank"),
            pl.lit(_REASON_HARD_INELIGIBLE).alias("selection_reason"),
        )
    return ineligible.with_columns(
        pl.lit(False).alias("selected"),
        pl.lit(0.0).alias("selection_score"),
        pl.lit(config.candidate_n + 1).cast(pl.Int32).alias("selection_rank"),
        pl.lit(_REASON_HARD_INELIGIBLE).alias("selection_reason"),
    )


def _attach_eligibility_metadata(
    assembled: pl.DataFrame,
    decisions: dict[tuple[str, str, str], EligibilityDecision],
) -> pl.DataFrame:
    """Attach eligibility metadata columns to every assembled selection row."""
    keys = assembled.select(["factor_name", "factor_version", "timeframe"]).iter_rows(named=True)
    rows_data: dict[str, list[object]] = {
        "eligibility_status": [],
        "eligibility_reason": [],
        "eligibility_policy": [],
        "usable_observations": [],
        "total_observations": [],
        "coverage_ratio": [],
        "null_rate": [],
        "required_lookback": [],
        "available_history": [],
        "warmup_sufficient": [],
        "companion_dependencies": [],
        "companion_coverage_status": [],
    }
    for row in keys:
        key = (row["factor_name"], row["factor_version"], row["timeframe"])
        decision = decisions.get(key)
        if decision is None:
            # Should not happen; emit missing-metadata sentinel.
            rows_data["eligibility_status"].append(
                EligibilityStatus.INELIGIBLE_MISSING_METADATA.value
            )
            rows_data["eligibility_reason"].append("eligibility decision not found")
            rows_data["eligibility_policy"].append(ELIGIBILITY_POLICY_VERSION)
            rows_data["usable_observations"].append(0)
            rows_data["total_observations"].append(None)
            rows_data["coverage_ratio"].append(None)
            rows_data["null_rate"].append(None)
            rows_data["required_lookback"].append(0)
            rows_data["available_history"].append(None)
            rows_data["warmup_sufficient"].append(None)
            rows_data["companion_dependencies"].append("")
            rows_data["companion_coverage_status"].append(None)
        else:
            rows_data["eligibility_status"].append(decision.status.value)
            rows_data["eligibility_reason"].append(decision.reason)
            rows_data["eligibility_policy"].append(decision.policy_version)
            rows_data["usable_observations"].append(decision.usable_observations)
            rows_data["total_observations"].append(decision.total_observations)
            rows_data["coverage_ratio"].append(decision.coverage_ratio)
            rows_data["null_rate"].append(decision.null_rate)
            rows_data["required_lookback"].append(decision.required_lookback)
            rows_data["available_history"].append(decision.available_history)
            rows_data["warmup_sufficient"].append(decision.warmup_sufficient)
            rows_data["companion_dependencies"].append(
                ",".join(sorted(decision.companion_dependencies))
            )
            rows_data["companion_coverage_status"].append(decision.companion_coverage_status)

    eligibility_frame = pl.DataFrame(rows_data)
    return pl.concat([assembled, eligibility_frame], how="horizontal_extend")


def _attach_orientation_metadata(frame: pl.DataFrame) -> pl.DataFrame:
    """Persist signed selection IC and leakage-safe direction metadata.

    Direction is derived exclusively from Factor Validation
    ``information_coefficient`` available at selection time. Ranking remains
    driven by ``abs(IC)`` via ``selection_score``; orientation does not alter
    ranks.
    """
    directions = [
        selected_direction_from_ic(value)
        for value in frame.get_column("information_coefficient").to_list()
    ]
    return frame.with_columns(
        pl.col("information_coefficient").cast(pl.Float64).alias("selection_ic"),
        pl.Series("selected_direction", directions, dtype=pl.Int8),
        pl.lit(FACTOR_ORIENTATION_POLICY).alias("orientation_policy"),
    )


def _apply_redundancy_selection(
    ranked: pl.DataFrame,
    config: RedundancyConfig,
    *,
    observation_source: FactorObservationSource,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply per-timeframe redundancy filtering then finalize Top-N."""
    _require_columns(ranked, _WINDOW_COLUMNS, "factor_validation")
    pieces: list[pl.DataFrame] = []
    audits: list[pl.DataFrame] = []

    for timeframe in ranked["timeframe"].unique().sort().to_list():
        subset = ranked.filter(pl.col("timeframe") == timeframe)
        start_raw = subset["validation_start_time"].min()
        end_raw = subset["validation_end_time"].max()
        if start_raw is None or end_raw is None:
            raise FactorSelectionError(
                "validation window is missing",
                error_code=_ERROR_WINDOW_MISSING,
                details={"timeframe": timeframe},
            )
        if not isinstance(start_raw, (int, float)) or not isinstance(end_raw, (int, float)):
            raise FactorSelectionError(
                "validation window is invalid",
                error_code=_ERROR_WINDOW_MISSING,
                details={
                    "timeframe": timeframe,
                    "validation_start_time": start_raw,
                    "validation_end_time": end_raw,
                },
            )
        start_time = int(start_raw)
        end_time = int(end_raw)
        if start_time > end_time:
            raise FactorSelectionError(
                "validation window is invalid",
                error_code=_ERROR_WINDOW_MISSING,
                details={
                    "timeframe": timeframe,
                    "validation_start_time": start_time,
                    "validation_end_time": end_time,
                },
            )

        candidate_limit = min(config.candidate_n, subset.height)
        candidates = subset.sort("selection_rank").head(candidate_limit)
        factor_names = candidates["factor_name"].to_list()
        factor_versions = candidates["factor_version"].to_list()
        filtered = _apply_redundancy_filter_for_source(
            subset,
            config,
            observation_source=observation_source,
            timeframe=str(timeframe),
            factor_names=factor_names,
            factor_versions=factor_versions,
            start_time=start_time,
            end_time=end_time,
        )
        pieces.append(filtered)
        audits.append(
            filtered.select(
                [
                    "factor_name",
                    "factor_version",
                    pl.lit(str(timeframe)).alias("timeframe"),
                    "candidate_rank",
                    "redundancy_checked",
                    "redundancy_rejected",
                    "redundancy_reference_factor",
                    "redundancy_reference_factor_version",
                    "redundancy_correlation",
                    "redundancy_overlap",
                    "selected",
                    "selection_reason",
                ]
            ).with_columns(
                pl.lit(config.candidate_n).alias("candidate_n"),
                pl.lit(config.max_factor_correlation).alias("max_factor_correlation"),
                pl.lit(config.min_overlap).alias("min_correlation_overlap"),
            )
        )

    combined = pl.concat(pieces, how="diagonal_relaxed").sort(_ROW_ID_COLUMN)
    audit = pl.concat(audits, how="diagonal_relaxed")
    return combined, audit


def _apply_redundancy_filter_for_source(
    ranked_subset: pl.DataFrame,
    config: RedundancyConfig,
    *,
    observation_source: FactorObservationSource,
    timeframe: str,
    factor_names: list[str],
    factor_versions: list[str],
    start_time: int,
    end_time: int,
) -> pl.DataFrame:
    """Dispatch legacy full-panel or memory-efficient spill redundancy.

    When ``observation_source`` exposes ``spill_panel``, observations are
    spilled per factor and greedy filtering uses pairwise joins (bounded
    memory). Otherwise the legacy ``load_panel`` + wide-pivot path is used.
    Scoring, ranking, thresholds, and accept/reject rules are unchanged.
    """
    spill_panel = getattr(observation_source, "spill_panel", None)
    if callable(spill_panel):
        spill_obj = spill_panel(
            timeframe=timeframe,
            factor_names=factor_names,
            factor_versions=factor_versions,
            start_time=start_time,
            end_time=end_time,
        )
        if not isinstance(spill_obj, FactorObservationSpill):
            raise FactorSelectionError(
                "spill_panel must return FactorObservationSpill",
                error_code="FSEL_MEM_SPILL_TYPE",
                details={"actual_type": type(spill_obj).__name__},
            )
        try:
            return apply_greedy_redundancy_filter_from_spill(ranked_subset, spill_obj, config)
        except FactorSelectionError:
            raise
        except Exception as exc:
            raise FactorSelectionError(
                "memory-efficient redundancy filtering failed; not falling back to full_panel",
                error_code="FSEL_MEM_REDUNDANCY_FAILED",
                details={
                    "timeframe": timeframe,
                    "cause_type": type(exc).__name__,
                    "cause": str(exc),
                },
            ) from exc
        finally:
            spill_obj.cleanup()

    observations = observation_source.load_panel(
        timeframe=timeframe,
        factor_names=factor_names,
        factor_versions=factor_versions,
        start_time=start_time,
        end_time=end_time,
    )
    return apply_greedy_redundancy_filter(ranked_subset, observations, config)


def _phase2_audit_frame(selected: pl.DataFrame, config: RedundancyConfig) -> pl.DataFrame:
    """Build audit rows when redundancy filtering is disabled."""
    return selected.select(
        [
            "factor_name",
            "factor_version",
            "timeframe",
            pl.when(pl.col("selection_rank") <= config.candidate_n)
            .then(pl.col("selection_rank"))
            .otherwise(None)
            .cast(pl.Int32)
            .alias("candidate_rank"),
            pl.lit(False).alias("redundancy_checked"),
            pl.lit(False).alias("redundancy_rejected"),
            pl.lit(None, dtype=pl.String).alias("redundancy_reference_factor"),
            pl.lit(None, dtype=pl.String).alias("redundancy_reference_factor_version"),
            pl.lit(None, dtype=pl.Float64).alias("redundancy_correlation"),
            pl.lit(None, dtype=pl.Int64).alias("redundancy_overlap"),
            "selected",
            "selection_reason",
            pl.lit(config.candidate_n).alias("candidate_n"),
            pl.lit(config.max_factor_correlation).alias("max_factor_correlation"),
            pl.lit(config.min_overlap).alias("min_correlation_overlap"),
        ]
    )


def attach_selection_score_components(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach ranking components, normalized values, contributions, and score.

    This is the single scoring implementation used by
    ``SimpleFactorSelectionEngine.build`` and the detailed CSV audit export.
    Intermediate columns are safe to drop after ``selection_score`` is
    consumed. The caller-supplied frame is never mutated.
    """
    abs_ic = pl.col("information_coefficient").abs()
    abs_rank_ic = pl.col("rank_information_coefficient").abs()
    inverse_turnover = -pl.col("turnover")

    ic_norm = _minmax_normalize(abs_ic, "timeframe")
    rank_ic_norm = _minmax_normalize(abs_rank_ic, "timeframe")
    icir_norm = _minmax_normalize(pl.col("ic_information_ratio"), "timeframe")
    spread_norm = _minmax_normalize(pl.col("quantile_spread"), "timeframe")
    mono_norm = _minmax_normalize(pl.col("monotonicity_score"), "timeframe")
    decay_norm = _minmax_normalize(pl.col("ic_decay"), "timeframe")
    turnover_norm = _minmax_normalize(inverse_turnover, "timeframe")

    ic_contribution = WEIGHT_ABS_IC * ic_norm
    rank_ic_contribution = WEIGHT_ABS_RANK_IC * rank_ic_norm
    icir_contribution = WEIGHT_ICIR * icir_norm
    spread_contribution = WEIGHT_QUANTILE_SPREAD * spread_norm
    mono_contribution = WEIGHT_MONOTONICITY * mono_norm
    decay_contribution = WEIGHT_IC_DECAY * decay_norm
    turnover_contribution = WEIGHT_INVERSE_TURNOVER * turnover_norm

    selection_score = (
        ic_contribution
        + rank_ic_contribution
        + icir_contribution
        + spread_contribution
        + mono_contribution
        + decay_contribution
        + turnover_contribution
    )

    return frame.with_columns(
        abs_ic.alias("abs_information_coefficient"),
        abs_rank_ic.alias("abs_rank_information_coefficient"),
        inverse_turnover.alias("inverse_turnover"),
        ic_norm.alias("information_coefficient_normalized"),
        rank_ic_norm.alias("rank_information_coefficient_normalized"),
        icir_norm.alias("ic_information_ratio_normalized"),
        spread_norm.alias("quantile_spread_normalized"),
        mono_norm.alias("monotonicity_normalized"),
        decay_norm.alias("ic_decay_normalized"),
        turnover_norm.alias("turnover_normalized"),
        pl.lit(WEIGHT_ABS_IC).alias("information_coefficient_weight"),
        pl.lit(WEIGHT_ABS_RANK_IC).alias("rank_information_coefficient_weight"),
        pl.lit(WEIGHT_ICIR).alias("ic_information_ratio_weight"),
        pl.lit(WEIGHT_QUANTILE_SPREAD).alias("quantile_spread_weight"),
        pl.lit(WEIGHT_MONOTONICITY).alias("monotonicity_weight"),
        pl.lit(WEIGHT_IC_DECAY).alias("ic_decay_weight"),
        pl.lit(WEIGHT_INVERSE_TURNOVER).alias("turnover_weight"),
        ic_contribution.alias("information_coefficient_contribution"),
        rank_ic_contribution.alias("rank_information_coefficient_contribution"),
        icir_contribution.alias("ic_information_ratio_contribution"),
        spread_contribution.alias("quantile_spread_contribution"),
        mono_contribution.alias("monotonicity_contribution"),
        decay_contribution.alias("ic_decay_contribution"),
        turnover_contribution.alias("turnover_contribution"),
        selection_score.alias("selection_score"),
    )


def _compute_selection_scores(frame: pl.DataFrame) -> pl.DataFrame:
    """Compute the fixed-weight min-max-normalized selection score per timeframe."""
    return attach_selection_score_components(frame)


def _minmax_normalize(expression: pl.Expr, group_column: str) -> pl.Expr:
    """Min-max normalize ``expression`` within ``group_column`` to ``[0, 1]``."""
    minimum = expression.min().over(group_column)
    maximum = expression.max().over(group_column)
    return (
        pl.when(expression.is_null())
        .then(pl.lit(0.0))
        .when(maximum == minimum)
        .then(pl.lit(1.0))
        .otherwise((expression - minimum) / (maximum - minimum))
    )


def _assign_selection_ranks(frame: pl.DataFrame) -> pl.DataFrame:
    """Assign dense ordinal ranks within each timeframe by descending score."""
    ordered = frame.sort(
        "timeframe",
        "selection_score",
        "factor_name",
        "factor_version",
        descending=[False, True, False, False],
        maintain_order=True,
    )
    return ordered.with_columns(
        (pl.int_range(pl.len()).over("timeframe") + 1).cast(pl.Int32).alias("selection_rank")
    ).sort(_ROW_ID_COLUMN)


def _apply_phase2_selection(frame: pl.DataFrame, config: RedundancyConfig) -> pl.DataFrame:
    """Apply Top-N selection without observation-based redundancy checks.

    Ranks beyond ``candidate_n`` are marked ``outside_candidate_n``. Within the
    candidate pool, the first ``top_n`` ranks are selected.
    """
    selected = pl.col("selection_rank") <= config.top_n
    outside_candidate = pl.col("selection_rank") > config.candidate_n
    return frame.with_columns(
        selected.alias("selected"),
        pl.when(outside_candidate)
        .then(pl.lit(REASON_OUTSIDE_CANDIDATE_N))
        .when(selected)
        .then(pl.lit(REASON_TOP_N))
        .otherwise(pl.lit(REASON_OUTSIDE_TOP_N))
        .alias("selection_reason"),
    )


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FactorSelectionError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )

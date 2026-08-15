"""CQROS Factor Orthogonalization Engine contracts and implementation.

Purpose:
    Convert a Factor Combination dataset into a deterministic
    combination-unit orthogonalization DataFrame conforming to
    ``FACTOR_ORTHOGONALIZATION_SCHEMA``.

Responsibilities:
    - Define ``FactorOrthogonalizationEngine`` as the shared
      orthogonalization contract
    - Provide ``SimpleFactorOrthogonalizationEngine`` for greedy
      correlation-filter redundancy removal among combinations
    - Validate Factor Combination DataFrame structure
    - Load validation-window factor observations through an injected
      ``FactorObservationSource``
    - Emit deterministic orthogonalization decision rows with audit fields
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.factor_orthogonalization.exceptions``,
    ``cqros.factor_orthogonalization.redundancy``,
    ``cqros.factor_orthogonalization.schema``, and
    ``cqros.factor_selection.redundancy.FactorObservationSource``.

Public API:
    ``FactorOrthogonalizationEngine``, ``SimpleFactorOrthogonalizationEngine``,
    ``FACTOR_COMBINATION_INPUT_COLUMNS``, ``LineageContext``,
    ``validate_factor_combination_frame``
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol, cast, runtime_checkable

import polars as pl

from cqros.factor_orthogonalization.exceptions import FactorOrthogonalizationError
from cqros.factor_orthogonalization.redundancy import (
    DEFAULT_MAX_COMBINATION_CORRELATION,
    DEFAULT_MIN_CORRELATION_OVERLAP,
    ORTHOGONALIZATION_METHOD,
    ORTHOGONALIZATION_VERSION,
    OrthogonalizationConfig,
    apply_greedy_combination_filter,
    require_orthogonalization_config,
)
from cqros.factor_orthogonalization.schema import (
    CANONICAL_COLUMN_ORDER,
    FACTOR_ORTHOGONALIZATION_SCHEMA,
    FactorOrthogonalizationStatus,
)
from cqros.factor_selection.redundancy import FactorObservationSource

__all__ = [
    "FACTOR_COMBINATION_INPUT_COLUMNS",
    "FactorOrthogonalizationEngine",
    "LineageContext",
    "SimpleFactorOrthogonalizationEngine",
    "validate_factor_combination_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "FORTH_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "FORTH_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "FORTH_MISSING_COLUMNS"
_ERROR_OBSERVATION_SOURCE: Final[str] = "FORTH_OBSERVATION_SOURCE_REQUIRED"
_ERROR_VALIDATION_WINDOW: Final[str] = "FORTH_VALIDATION_WINDOW_INVALID"
_ERROR_TIMEFRAME: Final[str] = "FORTH_TIMEFRAME_INCONSISTENT"
_ERROR_DUPLICATE_IDS: Final[str] = "FORTH_DUPLICATE_COMBINATION_IDS"

# Factor Combination columns required to assemble an orthogonalization row.
FACTOR_COMBINATION_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "combination_id",
    "factor_names",
    "factor_versions",
    "factor_categories",
    "timeframe",
    "combination_size",
    "combination_method",
    "combination_rank",
    "combination_score",
    "stability_score",
    "confidence_score",
)


@dataclass(frozen=True, slots=True)
class LineageContext:
    """Immutable lineage and validation-window context for one build.

    Attributes:
        validation_start_time: Inclusive validation-window start (UTC ms).
        validation_end_time: Inclusive validation-window end (UTC ms).
        source_combination_version: Factor Combination partition version.
        source_fta_version: Factor Timeframe Analysis version.
        source_selection_version: Factor Selection version.
        dataset_version: Research dataset / validation dataset version.
    """

    validation_start_time: int
    validation_end_time: int
    source_combination_version: str
    source_fta_version: str
    source_selection_version: str
    dataset_version: str


@runtime_checkable
class FactorOrthogonalizationEngine(Protocol):
    """Structural contract for converting combinations into orthogonalization.

    Implementations own orthogonalization semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(
        self,
        factor_combination: pl.DataFrame,
        *,
        lineage: LineageContext,
    ) -> pl.DataFrame:
        """Convert a Factor Combination dataset into orthogonalization rows.

        Args:
            factor_combination: Canonical Factor Combination dataset.
                Must not be mutated.
            lineage: Validation window and source version provenance.

        Returns:
            A new DataFrame containing the columns required by
            ``FACTOR_ORTHOGONALIZATION_SCHEMA``.
        """
        ...


class SimpleFactorOrthogonalizationEngine:
    """Generate deterministic correlation-filter orthogonalization rows.

    Operates on Factor Combination rows (combination-unit). Candidates are
    evaluated in ascending ``combination_rank`` order using absolute Pearson
    correlation of equal-weight member-factor signals within the supplied
    validation window. Greedy acceptance rejects a candidate when correlation
    with any already-accepted combination meets the configured threshold and
    overlap requirement.

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        Output is fully deterministic for identical inputs and observation
        panels. Empty observation panels do not reject candidates.
    """

    __slots__ = ("_config", "_observation_source")

    _config: OrthogonalizationConfig
    _observation_source: FactorObservationSource

    def __init__(
        self,
        *,
        observation_source: FactorObservationSource | None,
        max_combination_correlation: float = DEFAULT_MAX_COMBINATION_CORRELATION,
        min_overlap: int = DEFAULT_MIN_CORRELATION_OVERLAP,
    ) -> None:
        """Initialize the engine with observation source and thresholds.

        Args:
            observation_source: Validation-window factor observation loader.
            max_combination_correlation: Absolute Pearson redundancy threshold.
            min_overlap: Minimum pairwise complete observations for a
                correlation check to apply.

        Raises:
            FactorOrthogonalizationError: If configuration is invalid or
                ``observation_source`` is ``None``.
        """
        if observation_source is None:
            raise FactorOrthogonalizationError(
                "observation_source is required for combination orthogonalization",
                error_code=_ERROR_OBSERVATION_SOURCE,
                details={"observation_source": None},
            )
        self._observation_source = observation_source
        self._config = require_orthogonalization_config(
            max_combination_correlation=max_combination_correlation,
            min_overlap=min_overlap,
        )

    @property
    def config(self) -> OrthogonalizationConfig:
        """Return the immutable orthogonalization configuration."""
        return self._config

    def build(
        self,
        factor_combination: pl.DataFrame,
        *,
        lineage: LineageContext,
    ) -> pl.DataFrame:
        """Convert a Factor Combination dataset into orthogonalization rows.

        Args:
            factor_combination: Canonical Factor Combination dataset.
                Must not be mutated.
            lineage: Validation window and source version provenance.

        Returns:
            A new DataFrame matching ``FACTOR_ORTHOGONALIZATION_SCHEMA``.

        Raises:
            FactorOrthogonalizationError: If the input fails structural
                validation, required columns are missing, the validation
                window is invalid, or timeframe consistency fails.
        """
        frame = validate_factor_combination_frame(factor_combination)
        _require_columns(frame, FACTOR_COMBINATION_INPUT_COLUMNS, "factor_combination")
        _require_validation_window(lineage)
        ordered = _order_combinations(frame)
        _require_unique_combination_ids(ordered)
        timeframe = _require_single_timeframe(ordered)

        factor_names, factor_versions = _collect_member_identities(ordered)
        observations = self._observation_source.load_panel(
            timeframe=timeframe,
            factor_names=factor_names,
            factor_versions=factor_versions,
            start_time=lineage.validation_start_time,
            end_time=lineage.validation_end_time,
        )
        decisions = apply_greedy_combination_filter(ordered, observations, self._config)
        return _assemble_output(
            ordered=ordered,
            decisions=decisions,
            lineage=lineage,
            config=self._config,
        )


def validate_factor_combination_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate Factor Combination dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        FactorOrthogonalizationError: If ``frame`` is not a Polars DataFrame
            or contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise FactorOrthogonalizationError(
            "factor_combination frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={
                "dataset": "factor_combination",
                "actual_type": type(frame).__name__,
            },
        )
    if frame.height == 0:
        raise FactorOrthogonalizationError(
            "factor_combination frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "factor_combination", "rows": frame.height},
        )
    return frame


def _order_combinations(frame: pl.DataFrame) -> pl.DataFrame:
    """Order combinations by ascending rank then combination_id."""
    return frame.sort(
        "combination_rank",
        "combination_id",
        descending=[False, False],
        nulls_last=True,
        maintain_order=True,
    )


def _require_unique_combination_ids(frame: pl.DataFrame) -> None:
    """Raise when duplicate combination_id values are present."""
    duplicates = (
        frame.group_by("combination_id")
        .len()
        .filter(pl.col("len") > 1)
        .select("combination_id")
        .to_series()
        .to_list()
    )
    if duplicates:
        raise FactorOrthogonalizationError(
            "factor_combination frame contains duplicate combination_id values",
            error_code=_ERROR_DUPLICATE_IDS,
            details={"duplicate_combination_ids": tuple(sorted(str(item) for item in duplicates))},
        )


def _require_single_timeframe(frame: pl.DataFrame) -> str:
    """Require exactly one timeframe value within the partition frame."""
    values = frame.select("timeframe").unique().to_series().to_list()
    if len(values) != 1:
        raise FactorOrthogonalizationError(
            "factor_combination frame must contain exactly one timeframe",
            error_code=_ERROR_TIMEFRAME,
            details={"timeframes": tuple(sorted(str(item) for item in values))},
        )
    return str(values[0])


def _require_validation_window(lineage: LineageContext) -> None:
    """Validate inclusive validation-window bounds."""
    start = lineage.validation_start_time
    end = lineage.validation_end_time
    if start > end:
        raise FactorOrthogonalizationError(
            "validation window start_time must be <= end_time",
            error_code=_ERROR_VALIDATION_WINDOW,
            details={
                "validation_start_time": start,
                "validation_end_time": end,
            },
        )


def _collect_member_identities(
    frame: pl.DataFrame,
) -> tuple[list[str], list[str]]:
    """Collect unique member factor name/version pairs for observation loading."""
    names: list[str] = []
    versions: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in frame.select("factor_names", "factor_versions").to_dicts():
        row_names = _as_string_list(row["factor_names"])
        row_versions = _as_string_list(row["factor_versions"])
        for name, version in zip(row_names, row_versions, strict=False):
            key = (name, version)
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
            versions.append(version)
    return names, versions


def _assemble_output(
    *,
    ordered: pl.DataFrame,
    decisions: Sequence[dict[str, object]],
    lineage: LineageContext,
    config: OrthogonalizationConfig,
) -> pl.DataFrame:
    """Join greedy decisions onto ordered combinations and emit schema rows."""
    decision_frame = pl.DataFrame(
        {
            "combination_id": [str(item["combination_id"]) for item in decisions],
            "redundancy_checked": [bool(item["redundancy_checked"]) for item in decisions],
            "redundancy_rejected": [bool(item["redundancy_rejected"]) for item in decisions],
            "redundancy_reference_combination_id": [
                item["redundancy_reference_combination_id"] for item in decisions
            ],
            "correlation_score": [item["correlation_score"] for item in decisions],
            "correlation_overlap": [item["correlation_overlap"] for item in decisions],
            "orthogonalization_reason": [
                str(item["orthogonalization_reason"]) for item in decisions
            ],
        }
    )
    joined = ordered.join(decision_frame, on="combination_id", how="left")
    analysis_time_ms = int(datetime.now(UTC).timestamp() * 1000)
    pass_status = FactorOrthogonalizationStatus.PASS.value
    fail_status = FactorOrthogonalizationStatus.FAIL.value

    selected_mask = pl.col("redundancy_rejected") == False  # noqa: E712
    ranked = joined.with_columns(
        pl.lit(ORTHOGONALIZATION_METHOD).alias("orthogonalization_method"),
        pl.lit(ORTHOGONALIZATION_VERSION).alias("orthogonalization_version"),
        pl.lit(analysis_time_ms).cast(pl.Int64).alias("analysis_time"),
        pl.col("combination_rank").alias("source_combination_rank"),
        pl.col("combination_score").alias("source_combination_score"),
        pl.col("stability_score").alias("source_stability_score"),
        pl.col("confidence_score").alias("source_confidence_score"),
        pl.lit(None, dtype=pl.Float64).alias("vif_score"),
        pl.col("correlation_score").alias("redundancy_score"),
        pl.lit(None, dtype=pl.Float64).alias("orthogonality_score"),
        pl.lit(None, dtype=pl.Float64).alias("information_retained"),
        pl.col("correlation_overlap").cast(pl.Int64, strict=False),
        pl.lit(config.max_combination_correlation).cast(pl.Float64).alias("correlation_threshold"),
        pl.lit(int(config.min_overlap)).cast(pl.Int64).alias("min_overlap_threshold"),
        selected_mask.alias("selected"),
        pl.lit(lineage.source_combination_version).alias("source_combination_version"),
        pl.lit(lineage.source_fta_version).alias("source_fta_version"),
        pl.lit(lineage.source_selection_version).alias("source_selection_version"),
        pl.lit(lineage.dataset_version).alias("dataset_version"),
        pl.lit(lineage.validation_start_time).cast(pl.Int64).alias("validation_start_time"),
        pl.lit(lineage.validation_end_time).cast(pl.Int64).alias("validation_end_time"),
        pl.when(selected_mask)
        .then(pl.lit(pass_status))
        .otherwise(pl.lit(fail_status))
        .alias("status"),
    ).with_columns(
        pl.when(pl.col("selected"))
        .then(pl.col("selected").cum_sum().cast(pl.Int32))
        .otherwise(pl.lit(None, dtype=pl.Int32))
        .alias("orthogonalization_rank")
    )

    return (
        ranked.sort(
            "selected",
            "orthogonalization_rank",
            "source_combination_rank",
            "combination_id",
            descending=[True, False, False, False],
            nulls_last=True,
            maintain_order=True,
        )
        .select(list(CANONICAL_COLUMN_ORDER))
        .cast(FACTOR_ORTHOGONALIZATION_SCHEMA)
    )


def _as_string_list(value: object) -> list[str]:
    """Normalize list-like member identity columns to ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in cast(Sequence[object], value)]
    return [str(value)]


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FactorOrthogonalizationError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )

"""CQROS factor orthogonalization metrics dataset verification.

Purpose:
    Perform structural and lineage validation of canonical Factor
    Orthogonalization frames against ``FACTOR_ORTHOGONALIZATION_SCHEMA``
    and the originating Factor Combination frame.

Responsibilities:
    - Reject non-DataFrame and empty inputs
    - Validate required columns, canonical order, and schema equality
    - Verify uniqueness, deterministic ordering, and source membership
    - Verify correlation-threshold and overlap reconstructability
    - Verify rejected rows are not marked selected
    - Raise ``FactorOrthogonalizationError`` on failure
    - Return the validated frame unchanged
    - Remain free of persistence, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.factor_orthogonalization.exceptions``,
    ``cqros.factor_orthogonalization.redundancy``, and
    ``cqros.factor_orthogonalization.schema``.

Public API:
    ``FactorOrthogonalizationVerifier`` and verification error-code constants.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.factor_orthogonalization.exceptions import FactorOrthogonalizationError
from cqros.factor_orthogonalization.redundancy import REASON_ACCEPTED, REASON_REDUNDANT
from cqros.factor_orthogonalization.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FACTOR_ORTHOGONALIZATION_SCHEMA,
    REQUIRED_COLUMNS,
    FactorOrthogonalizationStatus,
)

__all__ = [
    "ERROR_COLUMN_ORDER",
    "ERROR_FRAME_EMPTY",
    "ERROR_FRAME_TYPE",
    "ERROR_LINEAGE_COMBINATION_TYPE",
    "ERROR_LINEAGE_DUPLICATE",
    "ERROR_LINEAGE_MEMBERSHIP",
    "ERROR_LINEAGE_REJECTED_SELECTED",
    "ERROR_LINEAGE_THRESHOLD",
    "ERROR_LINEAGE_TIMEFRAME",
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "FactorOrthogonalizationVerifier",
]

ERROR_FRAME_TYPE: Final[str] = "FORTH-VERIFICATION-001"
ERROR_FRAME_EMPTY: Final[str] = "FORTH-VERIFICATION-002"
ERROR_REQUIRED_COLUMNS: Final[str] = "FORTH-VERIFICATION-003"
ERROR_COLUMN_ORDER: Final[str] = "FORTH-VERIFICATION-004"
ERROR_SCHEMA_MISMATCH: Final[str] = "FORTH-VERIFICATION-005"
ERROR_LINEAGE_COMBINATION_TYPE: Final[str] = "FORTH-LINEAGE-001"
ERROR_LINEAGE_MEMBERSHIP: Final[str] = "FORTH-LINEAGE-002"
ERROR_LINEAGE_DUPLICATE: Final[str] = "FORTH-LINEAGE-003"
ERROR_LINEAGE_TIMEFRAME: Final[str] = "FORTH-LINEAGE-004"
ERROR_LINEAGE_REJECTED_SELECTED: Final[str] = "FORTH-LINEAGE-005"
ERROR_LINEAGE_THRESHOLD: Final[str] = "FORTH-LINEAGE-006"

_CORR_TOLERANCE: Final[float] = 1e-9

_COMBINATION_REQUIRED: Final[tuple[str, ...]] = (
    "combination_id",
    "factor_names",
    "factor_versions",
    "timeframe",
    "combination_rank",
)


class FactorOrthogonalizationVerifier:
    """Structural and lineage verifier for Factor Orthogonalization frames."""

    __slots__ = ()

    def verify(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Verify ``frame`` structurally and return it unchanged."""
        validated = _require_dataframe(frame)
        _require_non_empty(validated)
        _require_required_columns(validated)
        _require_canonical_column_order(validated)
        _require_schema_equality(validated)
        _require_unique_combination_ids(validated)
        _require_rejected_not_selected(validated)
        return validated

    def verify_against_combination(
        self,
        orthogonalization_frame: pl.DataFrame,
        combination_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """Verify lineage of ``orthogonalization_frame`` against Combination.

        Checks source membership, timeframe consistency, version/member
        consistency, and that rejected rows are not selected. Threshold and
        overlap fields must be internally consistent with rejection rows.
        """
        validated = self.verify(orthogonalization_frame)
        combination = _require_combination_frame(combination_frame)
        _require_columns(combination, _COMBINATION_REQUIRED, "factor_combination")
        _require_source_membership(validated, combination)
        _require_timeframe_consistency(validated, combination)
        _require_threshold_consistency(validated)
        return validated


def _require_dataframe(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a Polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorOrthogonalizationError(
            "factor orthogonalization frame must be a polars DataFrame",
            error_code=ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    return frame


def _require_non_empty(frame: pl.DataFrame) -> None:
    """Raise when ``frame`` contains no rows."""
    if frame.height == 0:
        raise FactorOrthogonalizationError(
            "factor orthogonalization frame must contain at least one row",
            error_code=ERROR_FRAME_EMPTY,
            details={"rows": frame.height},
        )


def _require_required_columns(frame: pl.DataFrame) -> None:
    """Raise when any required column is absent from ``frame``."""
    missing = tuple(name for name in REQUIRED_COLUMNS if name not in frame.columns)
    if missing:
        raise FactorOrthogonalizationError(
            f"missing required columns: {list(missing)}",
            error_code=ERROR_REQUIRED_COLUMNS,
            details={
                "missing_columns": missing,
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(frame.columns),
            },
        )


def _require_canonical_column_order(frame: pl.DataFrame) -> None:
    """Raise when ``frame`` column order differs from canonical order."""
    actual_order = tuple(frame.columns)
    if actual_order != CANONICAL_COLUMN_ORDER:
        raise FactorOrthogonalizationError(
            "factor orthogonalization frame column order does not match " "canonical order",
            error_code=ERROR_COLUMN_ORDER,
            details={
                "expected_order": CANONICAL_COLUMN_ORDER,
                "actual_order": actual_order,
            },
        )


def _require_schema_equality(frame: pl.DataFrame) -> None:
    """Raise when ``frame`` schema differs from the canonical schema."""
    if frame.schema == FACTOR_ORTHOGONALIZATION_SCHEMA:
        return

    mismatched: list[dict[str, object]] = []
    for column in CANONICAL_COLUMN_ORDER:
        expected = COLUMN_DTYPES[column]
        actual = frame.schema[column]
        if actual != expected:
            mismatched.append(
                {
                    "column": column,
                    "expected": str(expected),
                    "actual": str(actual),
                }
            )

    raise FactorOrthogonalizationError(
        "factor orthogonalization schema mismatch",
        error_code=ERROR_SCHEMA_MISMATCH,
        details={
            "expected_schema": str(FACTOR_ORTHOGONALIZATION_SCHEMA),
            "actual_schema": str(frame.schema),
            "mismatched_columns": tuple(item["column"] for item in mismatched),
            "mismatches": tuple(mismatched),
        },
    )


def _require_unique_combination_ids(frame: pl.DataFrame) -> None:
    """Raise when duplicate combination_id values exist."""
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
            "orthogonalization frame contains duplicate combination_id values",
            error_code=ERROR_LINEAGE_DUPLICATE,
            details={"duplicate_combination_ids": tuple(sorted(str(item) for item in duplicates))},
        )


def _require_rejected_not_selected(frame: pl.DataFrame) -> None:
    """Raise when a redundancy-rejected row is marked selected or PASS."""
    bad = frame.filter(
        (pl.col("redundancy_rejected") == True)  # noqa: E712
        & (
            (pl.col("selected") == True)  # noqa: E712
            | (pl.col("status") == FactorOrthogonalizationStatus.PASS.value)
            | (pl.col("orthogonalization_reason") != REASON_REDUNDANT)
        )
    )
    if bad.height > 0:
        raise FactorOrthogonalizationError(
            "rejected combinations must not be selected or PASS",
            error_code=ERROR_LINEAGE_REJECTED_SELECTED,
            details={
                "combination_ids": tuple(bad["combination_id"].to_list()),
            },
        )

    bad_accepted = frame.filter(
        (pl.col("redundancy_rejected") == False)  # noqa: E712
        & (pl.col("selected") == True)  # noqa: E712
        & (pl.col("orthogonalization_reason") != REASON_ACCEPTED)
    )
    if bad_accepted.height > 0:
        raise FactorOrthogonalizationError(
            "accepted combinations must use accepted orthogonalization_reason",
            error_code=ERROR_LINEAGE_REJECTED_SELECTED,
            details={
                "combination_ids": tuple(bad_accepted["combination_id"].to_list()),
            },
        )


def _require_combination_frame(frame: object) -> pl.DataFrame:
    """Validate that the source combination frame is a non-empty DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorOrthogonalizationError(
            "factor_combination frame must be a polars DataFrame",
            error_code=ERROR_LINEAGE_COMBINATION_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    if frame.height == 0:
        raise FactorOrthogonalizationError(
            "factor_combination frame must contain at least one row",
            error_code=ERROR_LINEAGE_COMBINATION_TYPE,
            details={"rows": frame.height},
        )
    return frame


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when required columns are missing."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FactorOrthogonalizationError(
            f"{dataset} frame is missing required columns",
            error_code=ERROR_LINEAGE_COMBINATION_TYPE,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
            },
        )


def _require_source_membership(
    ortho: pl.DataFrame,
    combination: pl.DataFrame,
) -> None:
    """Every orthogonalization combination_id must exist in Combination."""
    source_ids = set(combination["combination_id"].cast(pl.String).to_list())
    ortho_ids = ortho["combination_id"].to_list()
    missing = sorted({str(item) for item in ortho_ids if str(item) not in source_ids})
    if missing:
        raise FactorOrthogonalizationError(
            "orthogonalization contains combination_id values absent from source",
            error_code=ERROR_LINEAGE_MEMBERSHIP,
            details={"missing_combination_ids": tuple(missing)},
        )

    # Member factor names/versions must match source Combination rows.
    source = {
        str(row["combination_id"]): row
        for row in combination.select(
            "combination_id",
            "factor_names",
            "factor_versions",
            "timeframe",
            "combination_rank",
        ).to_dicts()
    }
    mismatches: list[str] = []
    for row in ortho.select(
        "combination_id",
        "factor_names",
        "factor_versions",
        "source_combination_rank",
    ).to_dicts():
        source_row = source.get(str(row["combination_id"]))
        if source_row is None:
            continue
        if list(row["factor_names"]) != list(source_row["factor_names"]):
            mismatches.append(str(row["combination_id"]))
            continue
        if list(row["factor_versions"]) != list(source_row["factor_versions"]):
            mismatches.append(str(row["combination_id"]))
            continue
        if int(row["source_combination_rank"]) != int(source_row["combination_rank"]):
            mismatches.append(str(row["combination_id"]))
    if mismatches:
        raise FactorOrthogonalizationError(
            "orthogonalization member identity or rank mismatches source combination",
            error_code=ERROR_LINEAGE_MEMBERSHIP,
            details={"mismatched_combination_ids": tuple(sorted(set(mismatches)))},
        )


def _require_timeframe_consistency(
    ortho: pl.DataFrame,
    combination: pl.DataFrame,
) -> None:
    """Orthogonalization and Combination must share one timeframe."""
    ortho_tfs = set(ortho["timeframe"].to_list())
    combo_tfs = set(combination["timeframe"].to_list())
    if len(ortho_tfs) != 1 or ortho_tfs != combo_tfs:
        raise FactorOrthogonalizationError(
            "orthogonalization timeframe is inconsistent with source combination",
            error_code=ERROR_LINEAGE_TIMEFRAME,
            details={
                "orthogonalization_timeframes": tuple(sorted(str(item) for item in ortho_tfs)),
                "combination_timeframes": tuple(sorted(str(item) for item in combo_tfs)),
            },
        )


def _require_threshold_consistency(frame: pl.DataFrame) -> None:
    """Rejected rows must record correlation >= threshold and sufficient overlap."""
    rejected = frame.filter(pl.col("redundancy_rejected") == True)  # noqa: E712
    if rejected.height == 0:
        return

    bad_threshold = rejected.filter(
        pl.col("correlation_score").is_null()
        | pl.col("correlation_threshold").is_null()
        | (pl.col("correlation_score") + _CORR_TOLERANCE < pl.col("correlation_threshold"))
    )
    if bad_threshold.height > 0:
        raise FactorOrthogonalizationError(
            "rejected rows must record correlation_score >= correlation_threshold",
            error_code=ERROR_LINEAGE_THRESHOLD,
            details={
                "combination_ids": tuple(bad_threshold["combination_id"].to_list()),
            },
        )

    bad_overlap = rejected.filter(
        pl.col("correlation_overlap").is_null()
        | pl.col("min_overlap_threshold").is_null()
        | (pl.col("correlation_overlap") < pl.col("min_overlap_threshold"))
    )
    if bad_overlap.height > 0:
        raise FactorOrthogonalizationError(
            "rejected rows must record correlation_overlap >= min_overlap_threshold",
            error_code=ERROR_LINEAGE_THRESHOLD,
            details={
                "combination_ids": tuple(bad_overlap["combination_id"].to_list()),
            },
        )

    bad_reference = rejected.filter(pl.col("redundancy_reference_combination_id").is_null())
    if bad_reference.height > 0:
        raise FactorOrthogonalizationError(
            "rejected rows must record redundancy_reference_combination_id",
            error_code=ERROR_LINEAGE_THRESHOLD,
            details={
                "combination_ids": tuple(bad_reference["combination_id"].to_list()),
            },
        )

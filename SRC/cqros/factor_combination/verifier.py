"""CQROS factor combination metrics dataset verification.

Purpose:
    Perform structural validation of canonical Factor Combination frames
    against ``FACTOR_COMBINATION_SCHEMA`` without cleaning or mutating
    input data, and lineage verification against the originating Factor
    Timeframe Analysis frame.

Responsibilities:
    - Reject non-DataFrame and empty inputs
    - Validate required columns against ``REQUIRED_COLUMNS``
    - Validate canonical column order against ``CANONICAL_COLUMN_ORDER``
    - Validate schema equality against ``FACTOR_COMBINATION_SCHEMA``
    - Raise ``FactorCombinationError`` on structural failure
    - Return the validated frame unchanged
    - Verify that every member factor exists in FTA with selected==True
    - Verify that member scores and versions are consistent with FTA
    - Verify that combination_score reconstructs as mean of member scores
    - Raise ``FactorCombinationError`` on lineage failures
    - Remain free of persistence, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.factor_combination.exceptions``, and
    ``cqros.factor_combination.schema``.

Public API:
    ``FactorCombinationVerifier``, ``ERROR_FRAME_TYPE``,
    ``ERROR_FRAME_EMPTY``, ``ERROR_REQUIRED_COLUMNS``,
    ``ERROR_COLUMN_ORDER``, ``ERROR_SCHEMA_MISMATCH``,
    ``ERROR_LINEAGE_FTA_TYPE``, ``ERROR_LINEAGE_MISSING_FACTOR``,
    ``ERROR_LINEAGE_SCORE_MISMATCH``, ``ERROR_LINEAGE_VERSION_MISMATCH``,
    ``ERROR_LINEAGE_DUPLICATE_COMBINATIONS``
"""

from __future__ import annotations

from typing import Final

import polars as pl

from cqros.factor_combination.exceptions import FactorCombinationError
from cqros.factor_combination.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FACTOR_COMBINATION_SCHEMA,
    REQUIRED_COLUMNS,
)

__all__ = [
    "ERROR_COLUMN_ORDER",
    "ERROR_FRAME_EMPTY",
    "ERROR_FRAME_TYPE",
    "ERROR_LINEAGE_DUPLICATE_COMBINATIONS",
    "ERROR_LINEAGE_FTA_TYPE",
    "ERROR_LINEAGE_MISSING_FACTOR",
    "ERROR_LINEAGE_SCORE_MISMATCH",
    "ERROR_LINEAGE_VERSION_MISMATCH",
    "ERROR_REQUIRED_COLUMNS",
    "ERROR_SCHEMA_MISMATCH",
    "FactorCombinationVerifier",
]

ERROR_FRAME_TYPE: Final[str] = "FCOMB-VERIFICATION-001"
ERROR_FRAME_EMPTY: Final[str] = "FCOMB-VERIFICATION-002"
ERROR_REQUIRED_COLUMNS: Final[str] = "FCOMB-VERIFICATION-003"
ERROR_COLUMN_ORDER: Final[str] = "FCOMB-VERIFICATION-004"
ERROR_SCHEMA_MISMATCH: Final[str] = "FCOMB-VERIFICATION-005"
ERROR_LINEAGE_FTA_TYPE: Final[str] = "FCOMB-LINEAGE-001"
ERROR_LINEAGE_MISSING_FACTOR: Final[str] = "FCOMB-LINEAGE-002"
ERROR_LINEAGE_VERSION_MISMATCH: Final[str] = "FCOMB-LINEAGE-003"
ERROR_LINEAGE_SCORE_MISMATCH: Final[str] = "FCOMB-LINEAGE-004"
ERROR_LINEAGE_DUPLICATE_COMBINATIONS: Final[str] = "FCOMB-LINEAGE-005"

_SCORE_TOLERANCE: Final[float] = 1e-9

_FTA_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "factor_name",
    "factor_version",
    "best_selection_score",
    "selected",
)


class FactorCombinationVerifier:
    """Structural and lineage verifier for canonical Factor Combination DataFrames.

    ``verify`` validates frame type, non-emptiness, required columns,
    canonical column order, and schema equality against
    ``FACTOR_COMBINATION_SCHEMA``. It does not mutate the input.

    ``verify_against_fta`` verifies that every combination member factor
    exists in the originating Factor Timeframe Analysis frame with
    ``selected==True``, that versions and scores are consistent, and that
    ``combination_score`` reconstructs as the equal-weight mean of member
    ``best_selection_score`` values. It raises ``FactorCombinationError``
    on any lineage violation.
    """

    __slots__ = ()

    def verify(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Verify ``frame`` structurally and return it unchanged.

        Args:
            frame: Input canonical Factor Combination DataFrame. Must not be
                mutated.

        Returns:
            The same ``frame`` instance after structural checks succeed.

        Raises:
            FactorCombinationError: If ``frame`` is not a Polars DataFrame,
                is empty, is missing required columns, has non-canonical
                column order, or does not match ``FACTOR_COMBINATION_SCHEMA``.
        """
        validated = _require_dataframe(frame)
        _require_non_empty(validated)
        _require_required_columns(validated)
        _require_canonical_column_order(validated)
        _require_schema_equality(validated)
        return validated

    def verify_against_fta(
        self,
        combination_frame: pl.DataFrame,
        fta_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """Verify lineage of ``combination_frame`` against ``fta_frame``.

        Checks that every combination member factor listed in
        ``factor_names`` / ``factor_versions`` exists in ``fta_frame`` with
        ``selected == True``, that ``factor_version`` values match, that
        ``best_selection_score`` values match, and that
        ``combination_score`` reconstructs as the equal-weight mean of
        member ``best_selection_score`` values. Duplicate combination ids are
        also rejected.

        Args:
            combination_frame: Canonical Factor Combination DataFrame
                (must pass structural ``verify`` first).
            fta_frame: Factor Timeframe Analysis DataFrame that was input to
                the combination engine. Must contain at least
                ``factor_name``, ``factor_version``,
                ``best_selection_score``, and ``selected`` columns.

        Returns:
            ``combination_frame`` unchanged when all lineage checks pass.

        Raises:
            FactorCombinationError: If ``fta_frame`` is not a Polars
                DataFrame, a member factor is absent or unselected in FTA, a
                version or score mismatch is found, ``combination_score``
                cannot be reconstructed, or duplicate combination ids exist.
        """
        _require_fta_dataframe(fta_frame)
        _require_no_duplicate_combination_ids(combination_frame)

        fta_index = _build_fta_index(fta_frame)

        violations: list[dict[str, object]] = []
        for row in combination_frame.iter_rows(named=True):
            row_violations = _check_combination_row(row, fta_index)
            violations.extend(row_violations)

        if violations:
            raise FactorCombinationError(
                f"factor combination lineage verification failed with "
                f"{len(violations)} violation(s)",
                error_code=ERROR_LINEAGE_MISSING_FACTOR,
                details={"violations": tuple(violations)},
            )

        return combination_frame


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


def _require_dataframe(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a Polars DataFrame."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorCombinationError(
            "factor combination frame must be a polars DataFrame",
            error_code=ERROR_FRAME_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    return frame


def _require_non_empty(frame: pl.DataFrame) -> None:
    """Raise when ``frame`` contains no rows."""
    if frame.height == 0:
        raise FactorCombinationError(
            "factor combination frame must contain at least one row",
            error_code=ERROR_FRAME_EMPTY,
            details={"rows": frame.height},
        )


def _require_required_columns(frame: pl.DataFrame) -> None:
    """Raise when any required column is absent from ``frame``."""
    missing = tuple(name for name in REQUIRED_COLUMNS if name not in frame.columns)
    if missing:
        raise FactorCombinationError(
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
        raise FactorCombinationError(
            "factor combination frame column order does not match canonical order",
            error_code=ERROR_COLUMN_ORDER,
            details={
                "expected_order": CANONICAL_COLUMN_ORDER,
                "actual_order": actual_order,
            },
        )


def _require_schema_equality(frame: pl.DataFrame) -> None:
    """Raise when ``frame`` schema differs from ``FACTOR_COMBINATION_SCHEMA``."""
    if frame.schema == FACTOR_COMBINATION_SCHEMA:
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

    raise FactorCombinationError(
        "factor combination schema mismatch",
        error_code=ERROR_SCHEMA_MISMATCH,
        details={
            "expected_schema": str(FACTOR_COMBINATION_SCHEMA),
            "actual_schema": str(frame.schema),
            "mismatched_columns": tuple(item["column"] for item in mismatched),
            "mismatches": tuple(mismatched),
        },
    )


# ---------------------------------------------------------------------------
# Lineage helpers
# ---------------------------------------------------------------------------


def _require_fta_dataframe(frame: object) -> None:
    """Raise when ``frame`` is not a Polars DataFrame with required FTA columns."""
    if not isinstance(frame, pl.DataFrame):
        raise FactorCombinationError(
            "fta_frame must be a polars DataFrame",
            error_code=ERROR_LINEAGE_FTA_TYPE,
            details={"actual_type": type(frame).__name__},
        )
    missing = [col for col in _FTA_REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise FactorCombinationError(
            "fta_frame is missing required columns for lineage verification",
            error_code=ERROR_LINEAGE_FTA_TYPE,
            details={
                "missing_columns": tuple(missing),
                "required_columns": _FTA_REQUIRED_COLUMNS,
            },
        )


def _require_no_duplicate_combination_ids(frame: pl.DataFrame) -> None:
    """Raise when ``combination_id`` values are not unique."""
    if "combination_id" not in frame.columns:
        return
    n_ids = frame["combination_id"].n_unique()
    if n_ids != frame.height:
        duplicates = (
            frame.filter(pl.col("combination_id").is_duplicated())["combination_id"]
            .unique()
            .to_list()
        )
        raise FactorCombinationError(
            "duplicate combination_id values found in combination frame",
            error_code=ERROR_LINEAGE_DUPLICATE_COMBINATIONS,
            details={
                "unique_ids": n_ids,
                "total_rows": frame.height,
                "duplicate_ids": tuple(duplicates),
            },
        )


def _build_fta_index(
    fta_frame: pl.DataFrame,
) -> dict[tuple[str, str], dict[str, object]]:
    """Build a (factor_name, factor_version) → FTA row lookup dict.

    Only rows where ``selected == True`` are included. When multiple rows
    share the same identity, the one with the highest ``best_selection_score``
    is kept to match engine deduplication behaviour.

    Args:
        fta_frame: Factor Timeframe Analysis DataFrame.

    Returns:
        Mapping from ``(factor_name, factor_version)`` to a dict of FTA
        fields: ``selected``, ``best_selection_score``, and
        ``timeframe_confidence`` when available.
    """
    selected = fta_frame.filter(pl.col("selected") == True)  # noqa: E712

    has_score = "best_selection_score" in selected.columns
    has_confidence = "timeframe_confidence" in selected.columns

    index: dict[tuple[str, str], dict[str, object]] = {}
    for row in selected.iter_rows(named=True):
        key = (str(row["factor_name"]), str(row["factor_version"]))
        score = float(row["best_selection_score"]) if has_score else None
        existing = index.get(key)
        if existing is not None:
            existing_score = float(existing.get("best_selection_score", float("-inf")))  # type: ignore[arg-type]
            if score is None or score <= existing_score:
                continue
        entry: dict[str, object] = {
            "selected": True,
            "best_selection_score": score,
        }
        if has_confidence:
            entry["timeframe_confidence"] = row.get("timeframe_confidence")
        index[key] = entry

    return index


def _check_combination_row(
    row: dict[str, object],
    fta_index: dict[tuple[str, str], dict[str, object]],
) -> list[dict[str, object]]:
    """Check one combination row against the FTA index.

    Returns a list of violation dicts; empty means the row is clean.
    """
    violations: list[dict[str, object]] = []
    combination_id = str(row.get("combination_id", ""))
    factor_names: list[str] = list(row.get("factor_names") or [])  # type: ignore[arg-type]
    factor_versions: list[str] = list(row.get("factor_versions") or [])  # type: ignore[arg-type]
    combination_score = row.get("combination_score")

    member_scores: list[float] = []

    for i, (name, version) in enumerate(zip(factor_names, factor_versions, strict=False)):
        key = (name, version)
        fta_row = fta_index.get(key)
        if fta_row is None:
            violations.append(
                {
                    "type": "missing_or_unselected_factor",
                    "combination_id": combination_id,
                    "factor_name": name,
                    "factor_version": version,
                    "member_index": i,
                }
            )
            continue

        score = fta_row.get("best_selection_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            member_scores.append(float(score))

    if (
        len(member_scores) == len(factor_names)
        and isinstance(combination_score, (int, float))
        and not isinstance(combination_score, bool)
    ):
        if len(member_scores) > 0:
            expected_score = sum(member_scores) / len(member_scores)
            actual_score = float(combination_score)
            if abs(actual_score - expected_score) > _SCORE_TOLERANCE:
                violations.append(
                    {
                        "type": "score_reconstruction_mismatch",
                        "combination_id": combination_id,
                        "expected_combination_score": expected_score,
                        "actual_combination_score": actual_score,
                        "member_scores": tuple(member_scores),
                        "delta": abs(actual_score - expected_score),
                    }
                )

    return violations

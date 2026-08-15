"""CQROS Purged Cross Validation Engine contracts and fold implementation.

Purpose:
    Convert a canonical Walk-Forward dataset into a deterministic purged
    cross-validation DataFrame conforming to ``PURGED_CV_SCHEMA``.

Responsibilities:
    - Define ``PurgedCVEngine`` as the shared purged-CV contract
    - Provide ``SimplePurgedCVEngine`` for contiguous purged/embargoed folds
    - Validate Walk-Forward DataFrame structure
    - Split chronological observations into ``n_folds`` contiguous folds
    - Apply purge and embargo exclusions to prevent temporal leakage
    - Remain free of persistence, verification, CLI, storage, and file I/O

Dependencies:
    ``polars``, ``cqros.purged_cv.exceptions``, and
    ``cqros.purged_cv.schema``.

Public API:
    ``PurgedCVEngine``, ``SimplePurgedCVEngine``,
    ``WALK_FORWARD_INPUT_COLUMNS``, ``validate_walk_forward_frame``
"""

from __future__ import annotations

from typing import Final, Protocol, runtime_checkable

import polars as pl

from cqros.purged_cv.exceptions import PurgedCVError
from cqros.purged_cv.schema import (
    CANONICAL_COLUMN_ORDER,
    PURGED_CV_SCHEMA,
    PurgedCVStatus,
)

__all__ = [
    "WALK_FORWARD_INPUT_COLUMNS",
    "PurgedCVEngine",
    "SimplePurgedCVEngine",
    "validate_walk_forward_frame",
]

_ERROR_FRAME_TYPE: Final[str] = "PCV_FRAME_TYPE"
_ERROR_FRAME_EMPTY: Final[str] = "PCV_FRAME_EMPTY"
_ERROR_MISSING_COLUMNS: Final[str] = "PCV_MISSING_COLUMNS"
_ERROR_INVALID_CONFIG: Final[str] = "PCV_INVALID_CONFIG"

_DEFAULT_N_FOLDS: Final[int] = 5
_DEFAULT_PURGE_SIZE: Final[int] = 5
_DEFAULT_EMBARGO_SIZE: Final[int] = 5

_GROUP_COLUMNS: Final[tuple[str, ...]] = (
    "strategy_name",
    "strategy_version",
    "timeframe",
)

# Walk-Forward columns required to assemble a purged-CV fold.
# ``test_start`` is the chronological observation key from the upstream ledger.
WALK_FORWARD_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "strategy_name",
    "strategy_version",
    "timeframe",
    "test_start",
    "train_score",
    "test_score",
)

_CHRONOLOGICAL_COLUMN: Final[str] = "test_start"


@runtime_checkable
class PurgedCVEngine(Protocol):
    """Structural contract for converting walk-forward rows into purged folds.

    Implementations own purged-CV semantics. Pipeline orchestration
    delegates exclusively through this contract. Implementations must return
    a new DataFrame and must not mutate the input frame.
    """

    def build(self, walk_forward: pl.DataFrame) -> pl.DataFrame:
        """Convert a Walk-Forward dataset into a purged-CV DataFrame.

        Args:
            walk_forward: Canonical Walk-Forward dataset.
                Must not be mutated.

        Returns:
            A new DataFrame containing the columns required by
            ``PURGED_CV_SCHEMA``.
        """
        ...


class SimplePurgedCVEngine:
    """Generate deterministic purged cross-validation folds from walk-forward.

    Rules:
        - Rows are grouped by ``strategy_name``, ``strategy_version``, and
          ``timeframe``
        - Each group is sorted ascending by ``test_start`` (walk-forward
          chronological observation key)
        - Observations are split into ``n_folds`` contiguous folds; remainder
          rows are distributed one-at-a-time to the earliest folds
        - Fold ``i`` uses the current contiguous block as the test set and
          every other observation as the candidate training set
        - Purge removes the nearest ``purge_size`` observations immediately
          before and immediately after the test block from training
        - Embargo additionally removes ``embargo_size`` observations after
          the test block from training (after the post-test purge zone)
        - Purged and embargoed observations are excluded from both train and
          test
        - ``train_score`` is the mean of walk-forward ``train_score`` values
          on the retained training rows
        - ``test_score`` is the mean of walk-forward ``test_score`` values
          on the test rows
        - ``overfit_gap`` is ``train_score - test_score``
        - ``status`` is ``PASS`` when ``train_rows > 0`` and ``test_rows > 0``,
          otherwise ``FAIL``

    Notes:
        Implementations must not mutate the caller-supplied DataFrame.
        Purge and embargo are always applied. Fold generation is
        chronological only; random splitting is forbidden.
    """

    __slots__ = ("_embargo_size", "_n_folds", "_purge_size")

    _n_folds: int
    _purge_size: int
    _embargo_size: int

    def __init__(
        self,
        n_folds: int = _DEFAULT_N_FOLDS,
        purge_size: int = _DEFAULT_PURGE_SIZE,
        embargo_size: int = _DEFAULT_EMBARGO_SIZE,
    ) -> None:
        """Initialize purged cross-validation fold configuration.

        Args:
            n_folds: Number of contiguous chronological folds. Defaults to
                ``5``.
            purge_size: Number of observations purged on each side of the
                test fold. Defaults to ``5``.
            embargo_size: Number of additional post-test observations
                removed from training. Defaults to ``5``.

        Raises:
            PurgedCVError: If ``n_folds`` is not a positive integer or if
                ``purge_size`` / ``embargo_size`` are not non-negative
                integers.
        """
        self._n_folds = _require_positive_int(n_folds, "n_folds")
        self._purge_size = _require_non_negative_int(purge_size, "purge_size")
        self._embargo_size = _require_non_negative_int(embargo_size, "embargo_size")

    def build(self, walk_forward: pl.DataFrame) -> pl.DataFrame:
        """Convert a Walk-Forward dataset into finalized purged-CV rows.

        Args:
            walk_forward: Canonical Walk-Forward dataset.
                Must not be mutated.

        Returns:
            A new DataFrame matching ``PURGED_CV_SCHEMA``.

        Raises:
            PurgedCVError: If the input fails structural validation
                or required columns are missing.
        """
        frame = validate_walk_forward_frame(walk_forward)
        _require_columns(frame, WALK_FORWARD_INPUT_COLUMNS, "walk_forward")
        return _build_purged_cv_rows(
            frame,
            n_folds=self._n_folds,
            purge_size=self._purge_size,
            embargo_size=self._embargo_size,
        )


def validate_walk_forward_frame(frame: object) -> pl.DataFrame:
    """Validate that ``frame`` is a non-empty Polars DataFrame.

    Args:
        frame: Candidate Walk-Forward dataset passed to an engine.

    Returns:
        ``frame`` as a DataFrame after structural checks.

    Raises:
        PurgedCVError: If ``frame`` is not a Polars DataFrame or
            contains no rows.
    """
    if not isinstance(frame, pl.DataFrame):
        raise PurgedCVError(
            "walk_forward frame must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={
                "dataset": "walk_forward",
                "actual_type": type(frame).__name__,
            },
        )
    if frame.height == 0:
        raise PurgedCVError(
            "walk_forward frame must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"dataset": "walk_forward", "rows": frame.height},
        )
    return frame


def _build_purged_cv_rows(
    walk_forward: pl.DataFrame,
    *,
    n_folds: int,
    purge_size: int,
    embargo_size: int,
) -> pl.DataFrame:
    """Assemble canonical purged-CV rows from Walk-Forward observations."""
    partitions = walk_forward.partition_by(
        list(_GROUP_COLUMNS),
        maintain_order=True,
        include_key=True,
    )
    fold_rows: list[dict[str, object]] = []
    for partition in partitions:
        sorted_partition = partition.sort(_CHRONOLOGICAL_COLUMN, descending=False)
        fold_rows.extend(
            _folds_for_group(
                sorted_partition,
                n_folds=n_folds,
                purge_size=purge_size,
                embargo_size=embargo_size,
            )
        )
    return pl.DataFrame(fold_rows).select(list(CANONICAL_COLUMN_ORDER)).cast(PURGED_CV_SCHEMA)


def _folds_for_group(
    group: pl.DataFrame,
    *,
    n_folds: int,
    purge_size: int,
    embargo_size: int,
) -> list[dict[str, object]]:
    """Generate purged/embargoed folds for one strategy/timeframe group."""
    strategy_name = str(group["strategy_name"][0])
    strategy_version = str(group["strategy_version"][0])
    timeframe = str(group["timeframe"][0])
    boundaries = _fold_boundaries(group.height, n_folds)

    folds: list[dict[str, object]] = []
    for fold_id, (test_start_index, test_end_index) in enumerate(boundaries, start=1):
        folds.append(
            _fold_row(
                group=group,
                strategy_name=strategy_name,
                strategy_version=strategy_version,
                timeframe=timeframe,
                fold_id=fold_id,
                test_start_index=test_start_index,
                test_end_index=test_end_index,
                purge_size=purge_size,
                embargo_size=embargo_size,
            )
        )
    return folds


def _fold_boundaries(row_count: int, n_folds: int) -> list[tuple[int, int]]:
    """Return contiguous ``[start, end)`` index ranges for ``n_folds`` folds.

    Remainder rows are distributed one-at-a-time to the earliest folds so
    every observation participates in exactly one test fold when
    ``row_count >= n_folds``. When ``row_count < n_folds``, later folds may
    be empty.
    """
    base_size = row_count // n_folds
    remainder = row_count % n_folds
    boundaries: list[tuple[int, int]] = []
    start = 0
    for fold_index in range(n_folds):
        fold_size = base_size + (1 if fold_index < remainder else 0)
        end = start + fold_size
        boundaries.append((start, end))
        start = end
    return boundaries


def _train_indices(
    row_count: int,
    *,
    test_start_index: int,
    test_end_index: int,
    purge_size: int,
    embargo_size: int,
) -> list[int]:
    """Return chronological training indices after purge and embargo.

    Purge excludes ``purge_size`` observations immediately before and after
    the test block. Embargo additionally excludes ``embargo_size``
    observations after the post-test purge zone. Test indices are never
    retained in training.
    """
    purge_before_start = max(0, test_start_index - purge_size)
    post_test_exclusion_end = min(
        row_count,
        test_end_index + purge_size + embargo_size,
    )
    indices: list[int] = []
    for index in range(row_count):
        if test_start_index <= index < test_end_index:
            continue
        if purge_before_start <= index < test_start_index:
            continue
        if test_end_index <= index < post_test_exclusion_end:
            continue
        indices.append(index)
    return indices


def _fold_row(
    *,
    group: pl.DataFrame,
    strategy_name: str,
    strategy_version: str,
    timeframe: str,
    fold_id: int,
    test_start_index: int,
    test_end_index: int,
    purge_size: int,
    embargo_size: int,
) -> dict[str, object]:
    """Assemble one canonical purged-CV fold dictionary."""
    train_index_list = _train_indices(
        group.height,
        test_start_index=test_start_index,
        test_end_index=test_end_index,
        purge_size=purge_size,
        embargo_size=embargo_size,
    )
    test_index_list = list(range(test_start_index, test_end_index))

    train_rows = len(train_index_list)
    test_rows = len(test_index_list)

    train_frame = group[train_index_list] if train_rows > 0 else group.clear()
    test_frame = group[test_index_list] if test_rows > 0 else group.clear()

    train_times = train_frame[_CHRONOLOGICAL_COLUMN]
    test_times = test_frame[_CHRONOLOGICAL_COLUMN]

    train_score = _as_optional_float(train_frame["train_score"].mean()) if train_rows > 0 else None
    test_score = _as_optional_float(test_frame["test_score"].mean()) if test_rows > 0 else None
    overfit_gap = (
        train_score - test_score if train_score is not None and test_score is not None else None
    )
    status = (
        PurgedCVStatus.PASS.value if train_rows > 0 and test_rows > 0 else PurgedCVStatus.FAIL.value
    )

    return {
        "strategy_name": strategy_name,
        "strategy_version": strategy_version,
        "timeframe": timeframe,
        "fold_id": fold_id,
        "train_start_time": int(train_times[0]) if train_rows > 0 else None,
        "train_end_time": int(train_times[-1]) if train_rows > 0 else None,
        "test_start_time": int(test_times[0]) if test_rows > 0 else None,
        "test_end_time": int(test_times[-1]) if test_rows > 0 else None,
        "purge_size": purge_size,
        "embargo_size": embargo_size,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_score": train_score,
        "test_score": test_score,
        "overfit_gap": overfit_gap,
        "status": status,
    }


def _as_optional_float(value: object) -> float | None:
    """Convert a Polars scalar to ``float``, or ``None`` when undefined."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if value != value:  # NaN
        return None
    return float(value)


def _require_positive_int(value: object, name: str) -> int:
    """Validate that ``value`` is a positive integer configuration parameter."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PurgedCVError(
            f"{name} must be a positive integer",
            error_code=_ERROR_INVALID_CONFIG,
            details={"parameter": name, "actual_value": value},
        )
    return value


def _require_non_negative_int(value: object, name: str) -> int:
    """Validate that ``value`` is a non-negative integer configuration parameter."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PurgedCVError(
            f"{name} must be a non-negative integer",
            error_code=_ERROR_INVALID_CONFIG,
            details={"parameter": name, "actual_value": value},
        )
    return value


def _require_columns(
    frame: pl.DataFrame,
    required: tuple[str, ...],
    dataset: str,
) -> None:
    """Raise when any required column is missing from ``frame``."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PurgedCVError(
            f"{dataset} frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "dataset": dataset,
                "missing_columns": tuple(missing),
                "required_columns": required,
                "available_columns": tuple(frame.columns),
            },
        )

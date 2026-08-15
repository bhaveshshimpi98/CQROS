"""Unit tests for CQROS purged cross-validation schema."""

from __future__ import annotations

import polars as pl

from cqros.purged_cv import (
    PURGED_CV_COLUMNS,
    PURGED_CV_SCHEMA,
    PurgedCVStatus,
)
from cqros.purged_cv.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    purged_cv_status_values,
    purged_cv_statuses,
)
from cqros.purged_cv.schema import (
    PURGED_CV_SCHEMA as PURGED_CV_SCHEMA_DIRECT,
)

_FOLD_METADATA_COLUMNS: tuple[str, ...] = (
    "train_start_time",
    "train_end_time",
    "test_start_time",
    "test_end_time",
    "purge_size",
    "embargo_size",
)

_EVALUATION_COLUMNS: tuple[str, ...] = (
    "train_rows",
    "test_rows",
    "train_score",
    "test_score",
    "overfit_gap",
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical purged-CV contract."""
    assert PRIMARY_KEY_COLUMNS == (
        "strategy_name",
        "strategy_version",
        "timeframe",
        "fold_id",
    )
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == PURGED_CV_COLUMNS


def test_required_columns_are_complete() -> None:
    """REQUIRED_COLUMNS covers every canonical purged-CV column exactly once."""
    assert set(REQUIRED_COLUMNS) == set(PURGED_CV_COLUMNS)
    assert len(REQUIRED_COLUMNS) == len(PURGED_CV_COLUMNS)


def test_purged_cv_columns_contain_required_domain_columns() -> None:
    """PURGED_CV_COLUMNS enumerates identity, metadata, evaluation, and status."""
    for column in (
        "strategy_name",
        "strategy_version",
        "timeframe",
        "fold_id",
        *_FOLD_METADATA_COLUMNS,
        *_EVALUATION_COLUMNS,
        "status",
    ):
        assert column in PURGED_CV_COLUMNS


def test_canonical_column_order_has_no_duplicates() -> None:
    """Canonical purged-CV column order contains no duplicate names."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))


def test_column_dtypes_and_purged_cv_schema() -> None:
    """Purged-CV schema dtypes match COLUMN_DTYPES in canonical order."""
    assert PURGED_CV_SCHEMA is PURGED_CV_SCHEMA_DIRECT
    assert PURGED_CV_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert PURGED_CV_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["strategy_name"] == pl.String
    assert COLUMN_DTYPES["strategy_version"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["fold_id"] == pl.Int32
    assert COLUMN_DTYPES["train_start_time"] == pl.Int64
    assert COLUMN_DTYPES["train_end_time"] == pl.Int64
    assert COLUMN_DTYPES["test_start_time"] == pl.Int64
    assert COLUMN_DTYPES["test_end_time"] == pl.Int64
    assert COLUMN_DTYPES["purge_size"] == pl.Int64
    assert COLUMN_DTYPES["embargo_size"] == pl.Int64
    assert COLUMN_DTYPES["train_rows"] == pl.Int64
    assert COLUMN_DTYPES["test_rows"] == pl.Int64
    assert COLUMN_DTYPES["train_score"] == pl.Float64
    assert COLUMN_DTYPES["test_score"] == pl.Float64
    assert COLUMN_DTYPES["overfit_gap"] == pl.Float64
    assert COLUMN_DTYPES["status"] == pl.String


def test_column_dtypes_cover_every_canonical_column() -> None:
    """COLUMN_DTYPES defines an entry for every canonical purged-CV column."""
    assert set(COLUMN_DTYPES.keys()) == set(CANONICAL_COLUMN_ORDER)


def test_canonical_order_starts_with_identity_keys() -> None:
    """Canonical column order begins with primary key, then metadata, then status."""
    assert CANONICAL_COLUMN_ORDER[0] == "strategy_name"
    assert CANONICAL_COLUMN_ORDER[1] == "strategy_version"
    assert CANONICAL_COLUMN_ORDER[2] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[3] == "fold_id"
    assert CANONICAL_COLUMN_ORDER[4] == "train_start_time"
    assert CANONICAL_COLUMN_ORDER[-1] == "status"


def test_canonical_order_places_metadata_before_evaluation() -> None:
    """Fold metadata columns appear after the primary key and before evaluation."""
    metadata_slice = CANONICAL_COLUMN_ORDER[4:10]
    assert metadata_slice == _FOLD_METADATA_COLUMNS


def test_canonical_order_places_evaluation_before_status() -> None:
    """Evaluation columns appear in the declared order before status."""
    evaluation_slice = CANONICAL_COLUMN_ORDER[10:-1]
    assert evaluation_slice == _EVALUATION_COLUMNS


def test_primary_key_precedes_metadata_columns() -> None:
    """Primary key columns occupy the leading positions of the schema."""
    assert PURGED_CV_SCHEMA.names()[:4] == list(PRIMARY_KEY_COLUMNS)
    assert CANONICAL_COLUMN_ORDER.index("strategy_name") == 0
    assert CANONICAL_COLUMN_ORDER.index("strategy_version") == 1
    assert CANONICAL_COLUMN_ORDER.index("timeframe") == 2
    assert CANONICAL_COLUMN_ORDER.index("fold_id") == 3


def test_purged_cv_status_enum_members() -> None:
    """PurgedCVStatus exposes PASS and FAIL members."""
    assert PurgedCVStatus.PASS.value == "PASS"
    assert PurgedCVStatus.FAIL.value == "FAIL"
    assert len(list(PurgedCVStatus)) == 2


def test_purged_cv_statuses_helper() -> None:
    """purged_cv_statuses() returns a tuple of all status members."""
    statuses = purged_cv_statuses()
    assert statuses == (PurgedCVStatus.PASS, PurgedCVStatus.FAIL)
    assert isinstance(statuses, tuple)


def test_purged_cv_status_values_helper() -> None:
    """purged_cv_status_values() returns valid PASS and FAIL strings."""
    status_values = purged_cv_status_values()
    assert status_values == ("PASS", "FAIL")
    assert isinstance(status_values, tuple)
    assert set(status_values) == {member.value for member in PurgedCVStatus}


def test_purged_cv_schema_has_sixteen_columns() -> None:
    """Purged-CV schema defines exactly 16 canonical columns."""
    assert len(CANONICAL_COLUMN_ORDER) == 16
    assert len(PURGED_CV_SCHEMA) == 16


def test_fold_metadata_timestamp_columns_are_int64() -> None:
    """Fold window timestamp columns use Int64 dtype."""
    for column in (
        "train_start_time",
        "train_end_time",
        "test_start_time",
        "test_end_time",
    ):
        assert COLUMN_DTYPES[column] == pl.Int64


def test_purge_and_embargo_size_columns_are_int64() -> None:
    """Purge and embargo size columns use Int64 dtype."""
    assert COLUMN_DTYPES["purge_size"] == pl.Int64
    assert COLUMN_DTYPES["embargo_size"] == pl.Int64


def test_evaluation_score_columns_are_float64() -> None:
    """Train/test score and overfit gap columns use Float64 dtype."""
    assert COLUMN_DTYPES["train_score"] == pl.Float64
    assert COLUMN_DTYPES["test_score"] == pl.Float64
    assert COLUMN_DTYPES["overfit_gap"] == pl.Float64


def test_row_count_columns_are_int64() -> None:
    """Train and test row count columns use Int64 dtype."""
    assert COLUMN_DTYPES["train_rows"] == pl.Int64
    assert COLUMN_DTYPES["test_rows"] == pl.Int64

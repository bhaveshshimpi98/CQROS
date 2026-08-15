"""Unit tests for CQROS factor validation metrics schema."""

from __future__ import annotations

import polars as pl

from cqros.factor_validation import (
    FACTOR_VALIDATION_COLUMNS,
    FACTOR_VALIDATION_SCHEMA,
    FactorValidationStatus,
)
from cqros.factor_validation.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    factor_validation_status_values,
    factor_validation_statuses,
)
from cqros.factor_validation.schema import (
    FACTOR_VALIDATION_SCHEMA as FACTOR_VALIDATION_SCHEMA_DIRECT,
)

_VALIDATION_METRIC_COLUMNS: tuple[str, ...] = (
    "information_coefficient",
    "rank_information_coefficient",
    "ic_information_ratio",
    "ic_std",
    "ic_p_value",
    "ic_t_stat",
    "ic_decay",
    "turnover",
    "monotonicity_score",
    "quantile_spread",
    "observations",
    "ic_observations",
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical validation contract."""
    assert PRIMARY_KEY_COLUMNS == (
        "factor_name",
        "factor_version",
        "timeframe",
        "validation_time",
    )
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == FACTOR_VALIDATION_COLUMNS


def test_required_columns_are_complete() -> None:
    """REQUIRED_COLUMNS covers every canonical validation column exactly once."""
    assert set(REQUIRED_COLUMNS) == set(FACTOR_VALIDATION_COLUMNS)
    assert len(REQUIRED_COLUMNS) == len(FACTOR_VALIDATION_COLUMNS)


def test_factor_validation_columns_contain_required_domain_columns() -> None:
    """FACTOR_VALIDATION_COLUMNS enumerates identity, metrics, and status fields."""
    for column in (
        "factor_name",
        "factor_version",
        "timeframe",
        "validation_time",
        "factor_category",
        "dataset_version",
        "label_version",
        "validation_start_time",
        "validation_end_time",
        *_VALIDATION_METRIC_COLUMNS,
        "status",
    ):
        assert column in FACTOR_VALIDATION_COLUMNS


def test_canonical_column_order_has_no_duplicates() -> None:
    """Canonical factor validation column order contains no duplicate names."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))


def test_column_dtypes_and_factor_validation_schema() -> None:
    """Factor validation schema dtypes match COLUMN_DTYPES in canonical order."""
    assert FACTOR_VALIDATION_SCHEMA is FACTOR_VALIDATION_SCHEMA_DIRECT
    assert FACTOR_VALIDATION_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert FACTOR_VALIDATION_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["factor_name"] == pl.String
    assert COLUMN_DTYPES["factor_version"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["validation_time"] == pl.Int64
    assert COLUMN_DTYPES["factor_category"] == pl.String
    assert COLUMN_DTYPES["observations"] == pl.Int64
    assert COLUMN_DTYPES["status"] == pl.String


def test_column_dtypes_cover_every_canonical_column() -> None:
    """COLUMN_DTYPES defines an entry for every canonical validation column."""
    assert set(COLUMN_DTYPES.keys()) == set(CANONICAL_COLUMN_ORDER)


def test_canonical_order_starts_with_identity_keys() -> None:
    """Canonical column order begins with primary key, then metadata, then status."""
    assert CANONICAL_COLUMN_ORDER[0] == "factor_name"
    assert CANONICAL_COLUMN_ORDER[1] == "factor_version"
    assert CANONICAL_COLUMN_ORDER[2] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[3] == "validation_time"
    assert CANONICAL_COLUMN_ORDER[4] == "factor_category"
    assert CANONICAL_COLUMN_ORDER[-1] == "status"


def test_canonical_order_places_metrics_before_status() -> None:
    """Validation metric columns appear in the declared order before status."""
    first_metric_index = CANONICAL_COLUMN_ORDER.index("information_coefficient")
    status_index = CANONICAL_COLUMN_ORDER.index("status")
    metric_slice = CANONICAL_COLUMN_ORDER[first_metric_index:status_index]
    assert metric_slice == _VALIDATION_METRIC_COLUMNS


def test_primary_key_precedes_metric_columns() -> None:
    """Primary key columns occupy the leading positions of the schema."""
    assert FACTOR_VALIDATION_SCHEMA.names()[:4] == list(PRIMARY_KEY_COLUMNS)
    assert CANONICAL_COLUMN_ORDER.index("factor_name") == 0
    assert CANONICAL_COLUMN_ORDER.index("factor_version") == 1
    assert CANONICAL_COLUMN_ORDER.index("timeframe") == 2
    assert CANONICAL_COLUMN_ORDER.index("validation_time") == 3


def test_factor_validation_status_enum_members() -> None:
    """FactorValidationStatus exposes PASS, FAIL, and SKIPPED members."""
    assert FactorValidationStatus.PASS.value == "PASS"
    assert FactorValidationStatus.FAIL.value == "FAIL"
    assert FactorValidationStatus.SKIPPED.value == "SKIPPED"
    assert len(list(FactorValidationStatus)) == 3


def test_factor_validation_statuses_helper() -> None:
    """factor_validation_statuses() returns a tuple of all status members."""
    statuses = factor_validation_statuses()
    assert statuses == (
        FactorValidationStatus.PASS,
        FactorValidationStatus.FAIL,
        FactorValidationStatus.SKIPPED,
    )
    assert isinstance(statuses, tuple)


def test_factor_validation_status_values_helper() -> None:
    """factor_validation_status_values() returns valid PASS, FAIL, and SKIPPED strings."""
    status_values = factor_validation_status_values()
    assert status_values == ("PASS", "FAIL", "SKIPPED")
    assert isinstance(status_values, tuple)
    assert set(status_values) == {member.value for member in FactorValidationStatus}


def test_factor_validation_schema_has_twenty_two_columns() -> None:
    """Factor validation schema defines exactly 22 canonical columns."""
    assert len(CANONICAL_COLUMN_ORDER) == len(FACTOR_VALIDATION_COLUMNS)
    assert len(CANONICAL_COLUMN_ORDER) == 22
    assert len(FACTOR_VALIDATION_SCHEMA) == 22


def test_ic_information_ratio_exists_as_float64() -> None:
    """ic_information_ratio is a first-class Float64 validation metric."""
    assert "ic_information_ratio" in FACTOR_VALIDATION_COLUMNS
    assert "ic_information_ratio" in REQUIRED_COLUMNS
    assert "ic_information_ratio" in CANONICAL_COLUMN_ORDER
    assert COLUMN_DTYPES["ic_information_ratio"] == pl.Float64
    assert FACTOR_VALIDATION_SCHEMA["ic_information_ratio"] == pl.Float64


def test_ic_information_ratio_follows_rank_information_coefficient() -> None:
    """ic_information_ratio is placed immediately after rank_information_coefficient."""
    rank_index = CANONICAL_COLUMN_ORDER.index("rank_information_coefficient")
    assert CANONICAL_COLUMN_ORDER[rank_index + 1] == "ic_information_ratio"


def test_validation_metric_columns_are_float64() -> None:
    """Floating-point validation metric columns use Float64 dtype."""
    for column in (
        "information_coefficient",
        "rank_information_coefficient",
        "ic_information_ratio",
        "ic_t_stat",
        "ic_p_value",
        "ic_std",
        "ic_decay",
        "turnover",
        "monotonicity_score",
        "quantile_spread",
    ):
        assert COLUMN_DTYPES[column] == pl.Float64

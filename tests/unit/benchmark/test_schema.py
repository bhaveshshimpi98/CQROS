"""Unit tests for CQROS benchmark dataset schema."""

from __future__ import annotations

import polars as pl

from cqros.benchmark import (
    BENCHMARK_COLUMNS,
    BENCHMARK_SCHEMA,
    BenchmarkType,
)
from cqros.benchmark.schema import (
    BENCHMARK_SCHEMA as BENCHMARK_SCHEMA_DIRECT,
)
from cqros.benchmark.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    benchmark_type_values,
    benchmark_types,
)

_EXPECTED_BENCHMARK_TYPES: tuple[BenchmarkType, ...] = (
    BenchmarkType.BTC,
    BenchmarkType.ETH,
    BenchmarkType.BNB,
    BenchmarkType.EQUAL_WEIGHT,
    BenchmarkType.MARKET_CAP,
    BenchmarkType.SECTOR,
    BenchmarkType.CUSTOM,
)

_EXPECTED_BENCHMARK_TYPE_VALUES: tuple[str, ...] = (
    "BTC",
    "ETH",
    "BNB",
    "EQUAL_WEIGHT",
    "MARKET_CAP",
    "SECTOR",
    "CUSTOM",
)


def test_primary_key_and_required_columns() -> None:
    """Primary key and required columns match the canonical benchmark contract."""
    assert PRIMARY_KEY_COLUMNS == (
        "benchmark_name",
        "benchmark_version",
        "exchange",
        "market",
        "timeframe",
        "open_time",
    )
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER
    assert CANONICAL_COLUMN_ORDER == BENCHMARK_COLUMNS


def test_required_columns_are_complete() -> None:
    """REQUIRED_COLUMNS covers every canonical benchmark column exactly once."""
    assert set(REQUIRED_COLUMNS) == set(BENCHMARK_COLUMNS)
    assert len(REQUIRED_COLUMNS) == len(BENCHMARK_COLUMNS)


def test_benchmark_columns_contain_required_domain_columns() -> None:
    """BENCHMARK_COLUMNS enumerates identity, venue, timestamp, and value fields."""
    for column in (
        "benchmark_name",
        "benchmark_version",
        "benchmark_type",
        "exchange",
        "market",
        "timeframe",
        "open_time",
        "benchmark_value",
    ):
        assert column in BENCHMARK_COLUMNS


def test_canonical_column_order_has_no_duplicates() -> None:
    """Canonical benchmark column order contains no duplicate names."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))


def test_primary_key_columns_have_no_duplicates() -> None:
    """Primary key column definitions contain no duplicate names."""
    assert len(PRIMARY_KEY_COLUMNS) == len(set(PRIMARY_KEY_COLUMNS))


def test_column_dtypes_and_benchmark_schema() -> None:
    """Benchmark schema dtypes match COLUMN_DTYPES in canonical order."""
    assert BENCHMARK_SCHEMA is BENCHMARK_SCHEMA_DIRECT
    assert BENCHMARK_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert BENCHMARK_SCHEMA[column] == COLUMN_DTYPES[column]
    assert COLUMN_DTYPES["benchmark_name"] == pl.String
    assert COLUMN_DTYPES["benchmark_version"] == pl.String
    assert COLUMN_DTYPES["benchmark_type"] == pl.String
    assert COLUMN_DTYPES["exchange"] == pl.String
    assert COLUMN_DTYPES["market"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["open_time"] == pl.Int64
    assert COLUMN_DTYPES["benchmark_value"] == pl.Float64


def test_column_dtypes_cover_every_canonical_column() -> None:
    """COLUMN_DTYPES defines an entry for every canonical benchmark column."""
    assert set(COLUMN_DTYPES.keys()) == set(CANONICAL_COLUMN_ORDER)


def test_canonical_order_matches_declared_contract() -> None:
    """Canonical column order matches the declared benchmark dataset contract."""
    assert CANONICAL_COLUMN_ORDER == (
        "benchmark_name",
        "benchmark_version",
        "benchmark_type",
        "exchange",
        "market",
        "timeframe",
        "open_time",
        "benchmark_value",
    )


def test_canonical_order_starts_with_identity_keys() -> None:
    """Canonical column order begins with identity, then venue, then value."""
    assert CANONICAL_COLUMN_ORDER[0] == "benchmark_name"
    assert CANONICAL_COLUMN_ORDER[1] == "benchmark_version"
    assert CANONICAL_COLUMN_ORDER[2] == "benchmark_type"
    assert CANONICAL_COLUMN_ORDER[3] == "exchange"
    assert CANONICAL_COLUMN_ORDER[4] == "market"
    assert CANONICAL_COLUMN_ORDER[5] == "timeframe"
    assert CANONICAL_COLUMN_ORDER[6] == "open_time"
    assert CANONICAL_COLUMN_ORDER[-1] == "benchmark_value"


def test_primary_key_columns_are_subset_of_canonical_order() -> None:
    """Every primary key column appears in the canonical schema."""
    assert set(PRIMARY_KEY_COLUMNS).issubset(set(CANONICAL_COLUMN_ORDER))
    for column in PRIMARY_KEY_COLUMNS:
        assert column in BENCHMARK_SCHEMA.names()


def test_primary_key_excludes_type_and_value() -> None:
    """Primary key excludes classification and payload columns."""
    assert "benchmark_type" not in PRIMARY_KEY_COLUMNS
    assert "benchmark_value" not in PRIMARY_KEY_COLUMNS


def test_benchmark_type_enum_members() -> None:
    """BenchmarkType exposes the seven canonical benchmark classifications."""
    assert BenchmarkType.BTC.value == "BTC"
    assert BenchmarkType.ETH.value == "ETH"
    assert BenchmarkType.BNB.value == "BNB"
    assert BenchmarkType.EQUAL_WEIGHT.value == "EQUAL_WEIGHT"
    assert BenchmarkType.MARKET_CAP.value == "MARKET_CAP"
    assert BenchmarkType.SECTOR.value == "SECTOR"
    assert BenchmarkType.CUSTOM.value == "CUSTOM"
    assert len(list(BenchmarkType)) == 7


def test_benchmark_types_helper() -> None:
    """benchmark_types() returns a tuple of all type members."""
    types = benchmark_types()
    assert types == _EXPECTED_BENCHMARK_TYPES
    assert isinstance(types, tuple)
    assert len(types) == len(set(types))


def test_benchmark_type_values_helper() -> None:
    """benchmark_type_values() returns valid benchmark type strings."""
    type_values = benchmark_type_values()
    assert type_values == _EXPECTED_BENCHMARK_TYPE_VALUES
    assert isinstance(type_values, tuple)
    assert len(type_values) == len(set(type_values))
    assert set(type_values) == {member.value for member in BenchmarkType}


def test_benchmark_schema_has_eight_columns() -> None:
    """Benchmark schema defines exactly 8 canonical columns."""
    assert len(CANONICAL_COLUMN_ORDER) == 8
    assert len(BENCHMARK_SCHEMA) == 8
    assert len(PRIMARY_KEY_COLUMNS) == 6


def test_benchmark_value_exists_as_float64() -> None:
    """benchmark_value is a first-class Float64 payload column."""
    assert "benchmark_value" in BENCHMARK_COLUMNS
    assert "benchmark_value" in REQUIRED_COLUMNS
    assert "benchmark_value" in CANONICAL_COLUMN_ORDER
    assert COLUMN_DTYPES["benchmark_value"] == pl.Float64
    assert BENCHMARK_SCHEMA["benchmark_value"] == pl.Float64


def test_open_time_is_int64() -> None:
    """open_time uses Int64 dtype consistent with research dataset conventions."""
    assert COLUMN_DTYPES["open_time"] == pl.Int64
    assert BENCHMARK_SCHEMA["open_time"] == pl.Int64


def test_string_identity_columns_are_string() -> None:
    """Identity and venue classification columns use String dtype."""
    for column in (
        "benchmark_name",
        "benchmark_version",
        "benchmark_type",
        "exchange",
        "market",
        "timeframe",
    ):
        assert COLUMN_DTYPES[column] == pl.String

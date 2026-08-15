"""CQROS benchmark dataset schema.

Purpose:
    Define the canonical columnar contract for benchmark datasets produced by
    CQROS benchmark providers and consumed by benchmark-dependent factors.

Responsibilities:
    - Declare the benchmark-dataset primary key
    - Enumerate benchmark identity, venue, and value columns
    - Expose required columns, canonical column order, and expected dtypes
    - Expose the benchmark type enumeration
    - Remain free of benchmark computation, provider, and persistence logic

Dependencies:
    ``polars`` and the Python standard library.

Public API:
    ``PRIMARY_KEY_COLUMNS``, ``BENCHMARK_COLUMNS``,
    ``REQUIRED_COLUMNS``, ``CANONICAL_COLUMN_ORDER``, ``COLUMN_DTYPES``,
    ``BENCHMARK_SCHEMA``, ``BenchmarkType``, ``benchmark_types``,
    ``benchmark_type_values``

Notes:
    This module describes column presence and dtypes only; it does not
    compute benchmarks, validate frames, or persist datasets.
    ``benchmark_type`` stores ``BenchmarkType`` enum string values
    (``BTC``, ``ETH``, ``BNB``, ``EQUAL_WEIGHT``, ``MARKET_CAP``,
    ``SECTOR``, ``CUSTOM``).
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final

import polars as pl

__all__ = [
    "BENCHMARK_COLUMNS",
    "BENCHMARK_SCHEMA",
    "CANONICAL_COLUMN_ORDER",
    "COLUMN_DTYPES",
    "PRIMARY_KEY_COLUMNS",
    "REQUIRED_COLUMNS",
    "BenchmarkType",
    "benchmark_type_values",
    "benchmark_types",
]

PRIMARY_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "benchmark_name",
    "benchmark_version",
    "exchange",
    "market",
    "timeframe",
    "open_time",
)

# Benchmark identity, venue context, timestamp, and value.
BENCHMARK_COLUMNS: Final[tuple[str, ...]] = (
    "benchmark_name",
    "benchmark_version",
    "benchmark_type",
    "exchange",
    "market",
    "timeframe",
    "open_time",
    "benchmark_value",
)

CANONICAL_COLUMN_ORDER: Final[tuple[str, ...]] = BENCHMARK_COLUMNS

REQUIRED_COLUMNS: Final[tuple[str, ...]] = CANONICAL_COLUMN_ORDER

COLUMN_DTYPES: Final = MappingProxyType(
    {
        "benchmark_name": pl.String,
        "benchmark_version": pl.String,
        "benchmark_type": pl.String,
        "exchange": pl.String,
        "market": pl.String,
        "timeframe": pl.String,
        "open_time": pl.Int64,
        "benchmark_value": pl.Float64,
    }
)

BENCHMARK_SCHEMA: Final[pl.Schema] = pl.Schema(
    [(column, COLUMN_DTYPES[column]) for column in CANONICAL_COLUMN_ORDER]
)


# Prefer str+Enum over StrEnum so enum values embed directly into Polars.
class BenchmarkType(str, Enum):  # noqa: UP042
    """Canonical classification for a benchmark dataset row.

    Attributes:
        BTC: Bitcoin single-asset benchmark.
        ETH: Ethereum single-asset benchmark.
        BNB: BNB single-asset benchmark.
        EQUAL_WEIGHT: Equal-weighted multi-asset benchmark.
        MARKET_CAP: Market-capitalization-weighted benchmark.
        SECTOR: Sector or peer-group benchmark.
        CUSTOM: Researcher-defined custom benchmark.
    """

    BTC = "BTC"
    ETH = "ETH"
    BNB = "BNB"
    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    MARKET_CAP = "MARKET_CAP"
    SECTOR = "SECTOR"
    CUSTOM = "CUSTOM"


def benchmark_types() -> tuple[BenchmarkType, ...]:
    """Return an immutable copy of every ``BenchmarkType`` member.

    Returns:
        All benchmark-type members in declaration order.
    """
    return (
        BenchmarkType.BTC,
        BenchmarkType.ETH,
        BenchmarkType.BNB,
        BenchmarkType.EQUAL_WEIGHT,
        BenchmarkType.MARKET_CAP,
        BenchmarkType.SECTOR,
        BenchmarkType.CUSTOM,
    )


def benchmark_type_values() -> tuple[str, ...]:
    """Return an immutable copy of every ``BenchmarkType`` string value.

    Returns:
        All benchmark-type string values in declaration order.
    """
    return tuple(member.value for member in benchmark_types())

"""Unit tests for CQROS project-wide constants."""

from __future__ import annotations

import re

import pytest

import cqros.core.constants as constants


def test_all_exports_are_defined() -> None:
    """Every name in ``__all__`` resolves to a module attribute."""
    for name in constants.__all__:
        assert hasattr(constants, name), f"missing export: {name}"


def test_all_exports_are_unique() -> None:
    """``__all__`` must not list the same name twice."""
    assert len(constants.__all__) == len(set(constants.__all__))


def test_exported_values_are_immutable_collections() -> None:
    """Collection constants use immutable ``tuple`` or ``frozenset`` types."""
    for name in constants.__all__:
        value = getattr(constants, name)
        if isinstance(value, (list, set, dict)):
            pytest.fail(f"{name} must not use a mutable collection type")


def test_application_identity() -> None:
    """Application identity matches the CQROS configuration defaults."""
    assert constants.APP_NAME == "CQROS"
    assert constants.APP_VERSION == "1.0.0"
    assert constants.DEFAULT_TIMEZONE == "UTC"
    assert constants.DEFAULT_ENVIRONMENT == constants.ENVIRONMENT_DEVELOPMENT
    assert constants.DEFAULT_CONFIG_DIRECTORY == "configs"
    assert constants.DEFAULT_CONFIG_FILENAME == "default.toml"
    assert constants.DEFAULT_LOG_DIRECTORY == "logs"


def test_supported_environments() -> None:
    """Supported environments match documented configuration profiles."""
    expected = {
        constants.ENVIRONMENT_DEVELOPMENT,
        constants.ENVIRONMENT_TESTING,
        constants.ENVIRONMENT_PAPER,
        constants.ENVIRONMENT_PRODUCTION,
    }
    assert constants.SUPPORTED_ENVIRONMENTS == frozenset(expected)
    assert constants.DEFAULT_ENVIRONMENT in constants.SUPPORTED_ENVIRONMENTS


def test_exchange_defaults_and_allowlists() -> None:
    """Exchange constants align with the initial Binance USDT-M support."""
    assert constants.DEFAULT_EXCHANGE == constants.EXCHANGE_BINANCE
    assert constants.SUPPORTED_EXCHANGES == frozenset({constants.EXCHANGE_BINANCE})
    assert constants.DEFAULT_MARKET == constants.MARKET_USDT_PERPETUAL
    assert constants.SUPPORTED_MARKETS == frozenset({constants.MARKET_USDT_PERPETUAL})


def test_research_timeframes() -> None:
    """Default timeframes are a subset of the supported interval set."""
    assert constants.DEFAULT_TIMEFRAMES == (
        constants.TIMEFRAME_1M,
        constants.TIMEFRAME_5M,
        constants.TIMEFRAME_15M,
        constants.TIMEFRAME_1H,
        constants.TIMEFRAME_4H,
        constants.TIMEFRAME_1D,
    )
    assert set(constants.DEFAULT_TIMEFRAMES).issubset(constants.SUPPORTED_TIMEFRAMES)
    assert constants.TIMEFRAME_1S in constants.SUPPORTED_TIMEFRAMES
    assert constants.TIMEFRAME_30M in constants.SUPPORTED_TIMEFRAMES
    assert constants.TIMEFRAME_1W in constants.SUPPORTED_TIMEFRAMES


def test_research_compression_and_seed() -> None:
    """Dataset compression defaults and codecs are consistent."""
    assert constants.DEFAULT_DATASET_COMPRESSION == constants.COMPRESSION_ZSTD
    assert constants.DEFAULT_DATASET_COMPRESSION in constants.SUPPORTED_COMPRESSION_CODECS
    assert constants.COMPRESSION_NONE in constants.SUPPORTED_COMPRESSION_CODECS
    assert constants.DEFAULT_RANDOM_SEED == 42


def test_trading_quote_asset() -> None:
    """Trading quote asset defaults to USDT for USDT-M markets."""
    assert constants.DEFAULT_QUOTE_ASSET == "USDT"
    assert constants.SUPPORTED_QUOTE_ASSETS == frozenset({constants.DEFAULT_QUOTE_ASSET})


def test_storage_directories() -> None:
    """Storage directory names match configuration defaults."""
    assert constants.DEFAULT_STORAGE_ROOT == "data"
    assert constants.STORAGE_DIR_RAW == "raw"
    assert constants.STORAGE_DIR_PROCESSED == "processed"
    assert constants.STORAGE_DIR_FEATURES == "features"
    assert constants.STORAGE_DIR_LABELS == "labels"
    assert constants.STORAGE_DIR_TRAINING == "training"
    assert constants.STORAGE_DIR_SIGNALS == "signals"
    assert constants.STORAGE_DIR_PREDICTIONS == "predictions"
    assert constants.STORAGE_DIR_PORTFOLIOS == "portfolios"
    assert constants.STORAGE_DIR_RISKS == "risks"
    assert constants.STORAGE_DIR_MODELS == "models"
    assert constants.STORAGE_DIR_REPORTS == "reports"
    assert constants.STORAGE_DIR_CACHE == "cache"
    assert constants.STORAGE_DIR_METADATA == "metadata"


def test_file_formats() -> None:
    """File format names and extensions stay aligned."""
    assert constants.FILE_EXTENSION_PARQUET == f".{constants.FILE_FORMAT_PARQUET}"
    assert constants.FILE_EXTENSION_JSON == f".{constants.FILE_FORMAT_JSON}"
    assert constants.FILE_EXTENSION_YAML == f".{constants.FILE_FORMAT_YAML}"
    assert constants.FILE_EXTENSION_TOML == f".{constants.FILE_FORMAT_TOML}"
    assert constants.FILE_EXTENSION_CSV == f".{constants.FILE_FORMAT_CSV}"
    assert constants.SUPPORTED_DATASET_FORMATS == frozenset({constants.FILE_FORMAT_PARQUET})
    assert constants.FILE_FORMAT_TOML in constants.SUPPORTED_CONFIG_FORMATS
    assert constants.FILE_FORMAT_YAML in constants.SUPPORTED_CONFIG_FORMATS


def test_time_conversion_relationships() -> None:
    """Derived time conversion factors match base unit products."""
    assert constants.SECONDS_PER_HOUR == (constants.SECONDS_PER_MINUTE * constants.MINUTES_PER_HOUR)
    assert constants.SECONDS_PER_DAY == (constants.SECONDS_PER_HOUR * constants.HOURS_PER_DAY)
    assert constants.MILLISECONDS_PER_MINUTE == (
        constants.MILLISECONDS_PER_SECOND * constants.SECONDS_PER_MINUTE
    )
    assert constants.MILLISECONDS_PER_HOUR == (
        constants.MILLISECONDS_PER_MINUTE * constants.MINUTES_PER_HOUR
    )
    assert constants.MILLISECONDS_PER_DAY == (
        constants.MILLISECONDS_PER_HOUR * constants.HOURS_PER_DAY
    )
    assert constants.DAYS_PER_YEAR == 365
    assert constants.MICROSECONDS_PER_SECOND == 1_000_000
    assert constants.NANOSECONDS_PER_SECOND == 1_000_000_000


def test_validation_patterns_compile() -> None:
    """Validation regex patterns are syntactically valid."""
    re.compile(constants.SEMVER_PATTERN)
    re.compile(constants.TIMEFRAME_PATTERN)
    assert re.fullmatch(constants.SEMVER_PATTERN, constants.APP_VERSION)
    assert re.fullmatch(constants.TIMEFRAME_PATTERN, constants.TIMEFRAME_1M)
    assert re.fullmatch(constants.TIMEFRAME_PATTERN, constants.TIMEFRAME_1W)


def test_validation_missing_data_policies_and_hash() -> None:
    """Missing-data policies and hash algorithm defaults are consistent."""
    assert constants.DEFAULT_HASH_ALGORITHM == constants.HASH_ALGORITHM_SHA256
    assert constants.SUPPORTED_MISSING_DATA_POLICIES == frozenset(
        {
            constants.MISSING_DATA_POLICY_REJECT,
            constants.MISSING_DATA_POLICY_WARN,
            constants.MISSING_DATA_POLICY_IGNORE,
        }
    )


def test_default_aliases_reference_canonical_constants() -> None:
    """Default aliases reuse canonical constants instead of duplicating literals."""
    assert constants.DEFAULT_EXCHANGE is constants.EXCHANGE_BINANCE
    assert constants.DEFAULT_MARKET is constants.MARKET_USDT_PERPETUAL
    assert constants.DEFAULT_ENVIRONMENT is constants.ENVIRONMENT_DEVELOPMENT
    assert constants.DEFAULT_DATASET_COMPRESSION is constants.COMPRESSION_ZSTD
    assert constants.DEFAULT_HASH_ALGORITHM is constants.HASH_ALGORITHM_SHA256


def test_annotations_present_for_all_exports() -> None:
    """Every exported constant has an explicit module annotation."""
    annotations = constants.__annotations__
    for name in constants.__all__:
        assert name in annotations, f"{name} lacks a type annotation"

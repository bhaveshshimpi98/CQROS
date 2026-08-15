"""Project-wide CQROS constants.

This module defines immutable, project-wide constant values shared across
CQROS packages. Values are grouped by domain and use ``Final`` annotations
with immutable containers (``tuple``, ``frozenset``) where collections are
required.

Constants here describe structural project facts such as application
identity, supported exchanges, intervals, storage directory names, file
formats, and time conversion factors. Configurable business thresholds
(risk limits, promotion cutoffs, and similar policy values) do not belong
in this module.

Dependencies:
    Python standard library only (``typing.Final``).

Example:
    from cqros.core.constants import APP_NAME, DEFAULT_TIMEZONE

    assert APP_NAME == "CQROS"
    assert DEFAULT_TIMEZONE == "UTC"
"""

from __future__ import annotations

from typing import Final

__all__ = [
    # Application
    "APP_NAME",
    "APP_VERSION",
    "DEFAULT_TIMEZONE",
    "DEFAULT_ENVIRONMENT",
    "ENVIRONMENT_DEVELOPMENT",
    "ENVIRONMENT_TESTING",
    "ENVIRONMENT_PAPER",
    "ENVIRONMENT_PRODUCTION",
    "SUPPORTED_ENVIRONMENTS",
    "DEFAULT_CONFIG_DIRECTORY",
    "DEFAULT_CONFIG_FILENAME",
    "DEFAULT_LOG_DIRECTORY",
    # Exchange
    "EXCHANGE_BINANCE",
    "DEFAULT_EXCHANGE",
    "SUPPORTED_EXCHANGES",
    "MARKET_USDT_PERPETUAL",
    "DEFAULT_MARKET",
    "SUPPORTED_MARKETS",
    # Research
    "TIMEFRAME_1S",
    "TIMEFRAME_1M",
    "TIMEFRAME_5M",
    "TIMEFRAME_15M",
    "TIMEFRAME_30M",
    "TIMEFRAME_1H",
    "TIMEFRAME_4H",
    "TIMEFRAME_1D",
    "TIMEFRAME_1W",
    "DEFAULT_TIMEFRAMES",
    "SUPPORTED_TIMEFRAMES",
    "COMPRESSION_ZSTD",
    "COMPRESSION_SNAPPY",
    "COMPRESSION_GZIP",
    "COMPRESSION_LZ4",
    "COMPRESSION_BROTLI",
    "COMPRESSION_UNCOMPRESSED",
    "COMPRESSION_NONE",
    "DEFAULT_DATASET_COMPRESSION",
    "SUPPORTED_COMPRESSION_CODECS",
    "DEFAULT_RANDOM_SEED",
    # Trading
    "DEFAULT_QUOTE_ASSET",
    "SUPPORTED_QUOTE_ASSETS",
    # Storage
    "DEFAULT_STORAGE_ROOT",
    "STORAGE_DIR_RAW",
    "STORAGE_DIR_PROCESSED",
    "STORAGE_DIR_FEATURES",
    "STORAGE_DIR_LABELS",
    "STORAGE_DIR_TRAINING",
    "STORAGE_DIR_SIGNALS",
    "STORAGE_DIR_PREDICTIONS",
    "STORAGE_DIR_THRESHOLDS",
    "STORAGE_DIR_PORTFOLIOS",
    "STORAGE_DIR_RISKS",
    "STORAGE_DIR_ORDERS",
    "STORAGE_DIR_EXECUTIONS",
    "STORAGE_DIR_POSITIONS",
    "STORAGE_DIR_ACCOUNTING",
    "STORAGE_DIR_PORTFOLIO_RISK",
    "STORAGE_DIR_TRADE_MANAGEMENT",
    "STORAGE_DIR_PYRAMIDING",
    "STORAGE_DIR_EXIT_ENGINE",
    "STORAGE_DIR_BACKTESTING",
    "STORAGE_DIR_PERFORMANCE",
    "STORAGE_DIR_ANALYTICS",
    "STORAGE_DIR_REPORTING",
    "STORAGE_DIR_MONITORING",
    "STORAGE_DIR_FACTOR_VALIDATION",
    "STORAGE_DIR_FACTOR_SELECTION",
    "STORAGE_DIR_FACTOR_TIMEFRAME_ANALYSIS",
    "STORAGE_DIR_FACTOR_COMBINATION",
    "STORAGE_DIR_FACTOR_ORTHOGONALIZATION",
    "STORAGE_DIR_ALPHA",
    "STORAGE_DIR_REGIME",
    "STORAGE_DIR_MODELS",
    "STORAGE_DIR_WALK_FORWARD",
    "STORAGE_DIR_WALK_FORWARD_EVALUATION",
    "STORAGE_DIR_PURGED_CV",
    "STORAGE_DIR_PURGED_CV_EVALUATION",
    "STORAGE_DIR_FACTORS",
    "STORAGE_DIR_REPORTS",
    "STORAGE_DIR_CACHE",
    "STORAGE_DIR_METADATA",
    # File formats
    "FILE_FORMAT_PARQUET",
    "FILE_FORMAT_JSON",
    "FILE_FORMAT_YAML",
    "FILE_FORMAT_TOML",
    "FILE_FORMAT_CSV",
    "FILE_EXTENSION_PARQUET",
    "FILE_EXTENSION_JSON",
    "FILE_EXTENSION_YAML",
    "FILE_EXTENSION_YML",
    "FILE_EXTENSION_TOML",
    "FILE_EXTENSION_CSV",
    "SUPPORTED_DATASET_FORMATS",
    "SUPPORTED_CONFIG_FORMATS",
    # Time conversions
    "MILLISECONDS_PER_SECOND",
    "MICROSECONDS_PER_SECOND",
    "NANOSECONDS_PER_SECOND",
    "SECONDS_PER_MINUTE",
    "MINUTES_PER_HOUR",
    "HOURS_PER_DAY",
    "DAYS_PER_WEEK",
    "DAYS_PER_YEAR",
    "SECONDS_PER_HOUR",
    "SECONDS_PER_DAY",
    "MILLISECONDS_PER_MINUTE",
    "MILLISECONDS_PER_HOUR",
    "MILLISECONDS_PER_DAY",
    # Validation
    "SEMVER_PATTERN",
    "TIMEFRAME_PATTERN",
    "HASH_ALGORITHM_SHA256",
    "DEFAULT_HASH_ALGORITHM",
    "MISSING_DATA_POLICY_REJECT",
    "MISSING_DATA_POLICY_WARN",
    "MISSING_DATA_POLICY_IGNORE",
    "SUPPORTED_MISSING_DATA_POLICIES",
]

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

APP_NAME: Final[str] = "CQROS"
APP_VERSION: Final[str] = "1.0.0"
DEFAULT_TIMEZONE: Final[str] = "UTC"

ENVIRONMENT_DEVELOPMENT: Final[str] = "development"
ENVIRONMENT_TESTING: Final[str] = "testing"
ENVIRONMENT_PAPER: Final[str] = "paper"
ENVIRONMENT_PRODUCTION: Final[str] = "production"

DEFAULT_ENVIRONMENT: Final[str] = ENVIRONMENT_DEVELOPMENT

SUPPORTED_ENVIRONMENTS: Final[frozenset[str]] = frozenset(
    {
        ENVIRONMENT_DEVELOPMENT,
        ENVIRONMENT_TESTING,
        ENVIRONMENT_PAPER,
        ENVIRONMENT_PRODUCTION,
    }
)

DEFAULT_CONFIG_DIRECTORY: Final[str] = "configs"
DEFAULT_CONFIG_FILENAME: Final[str] = "default.toml"
DEFAULT_LOG_DIRECTORY: Final[str] = "logs"

# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------

EXCHANGE_BINANCE: Final[str] = "binance"
DEFAULT_EXCHANGE: Final[str] = EXCHANGE_BINANCE
SUPPORTED_EXCHANGES: Final[frozenset[str]] = frozenset({EXCHANGE_BINANCE})

MARKET_USDT_PERPETUAL: Final[str] = "usdt_perpetual"
DEFAULT_MARKET: Final[str] = MARKET_USDT_PERPETUAL
SUPPORTED_MARKETS: Final[frozenset[str]] = frozenset({MARKET_USDT_PERPETUAL})

# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

TIMEFRAME_1S: Final[str] = "1s"
TIMEFRAME_1M: Final[str] = "1m"
TIMEFRAME_5M: Final[str] = "5m"
TIMEFRAME_15M: Final[str] = "15m"
TIMEFRAME_30M: Final[str] = "30m"
TIMEFRAME_1H: Final[str] = "1h"
TIMEFRAME_4H: Final[str] = "4h"
TIMEFRAME_1D: Final[str] = "1d"
TIMEFRAME_1W: Final[str] = "1w"

DEFAULT_TIMEFRAMES: Final[tuple[str, ...]] = (
    TIMEFRAME_1M,
    TIMEFRAME_5M,
    TIMEFRAME_15M,
    TIMEFRAME_1H,
    TIMEFRAME_4H,
    TIMEFRAME_1D,
)

SUPPORTED_TIMEFRAMES: Final[frozenset[str]] = frozenset(
    {
        TIMEFRAME_1S,
        TIMEFRAME_1M,
        TIMEFRAME_5M,
        TIMEFRAME_15M,
        TIMEFRAME_30M,
        TIMEFRAME_1H,
        TIMEFRAME_4H,
        TIMEFRAME_1D,
        TIMEFRAME_1W,
    }
)

COMPRESSION_ZSTD: Final[str] = "zstd"
COMPRESSION_SNAPPY: Final[str] = "snappy"
COMPRESSION_GZIP: Final[str] = "gzip"
COMPRESSION_LZ4: Final[str] = "lz4"
COMPRESSION_BROTLI: Final[str] = "brotli"
COMPRESSION_UNCOMPRESSED: Final[str] = "uncompressed"
COMPRESSION_NONE: Final[str] = "none"

DEFAULT_DATASET_COMPRESSION: Final[str] = COMPRESSION_ZSTD

SUPPORTED_COMPRESSION_CODECS: Final[frozenset[str]] = frozenset(
    {
        COMPRESSION_ZSTD,
        COMPRESSION_SNAPPY,
        COMPRESSION_GZIP,
        COMPRESSION_LZ4,
        COMPRESSION_BROTLI,
        COMPRESSION_UNCOMPRESSED,
        COMPRESSION_NONE,
    }
)

DEFAULT_RANDOM_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------

DEFAULT_QUOTE_ASSET: Final[str] = "USDT"
SUPPORTED_QUOTE_ASSETS: Final[frozenset[str]] = frozenset({DEFAULT_QUOTE_ASSET})

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

DEFAULT_STORAGE_ROOT: Final[str] = "data"
STORAGE_DIR_RAW: Final[str] = "raw"
STORAGE_DIR_PROCESSED: Final[str] = "processed"
STORAGE_DIR_FEATURES: Final[str] = "features"
STORAGE_DIR_LABELS: Final[str] = "labels"
STORAGE_DIR_TRAINING: Final[str] = "training"
STORAGE_DIR_SIGNALS: Final[str] = "signals"
STORAGE_DIR_PREDICTIONS: Final[str] = "predictions"
STORAGE_DIR_THRESHOLDS: Final[str] = "thresholds"
STORAGE_DIR_PORTFOLIOS: Final[str] = "portfolios"
STORAGE_DIR_RISKS: Final[str] = "risks"
STORAGE_DIR_ORDERS: Final[str] = "orders"
STORAGE_DIR_EXECUTIONS: Final[str] = "executions"
STORAGE_DIR_POSITIONS: Final[str] = "positions"
STORAGE_DIR_ACCOUNTING: Final[str] = "accounting"
STORAGE_DIR_PORTFOLIO_RISK: Final[str] = "portfolio_risk"
STORAGE_DIR_TRADE_MANAGEMENT: Final[str] = "trade_management"
STORAGE_DIR_PYRAMIDING: Final[str] = "pyramiding"
STORAGE_DIR_EXIT_ENGINE: Final[str] = "exit_engine"
STORAGE_DIR_BACKTESTING: Final[str] = "backtesting"
STORAGE_DIR_PERFORMANCE: Final[str] = "performance"
STORAGE_DIR_ANALYTICS: Final[str] = "analytics"
STORAGE_DIR_REPORTING: Final[str] = "reporting"
STORAGE_DIR_MONITORING: Final[str] = "monitoring"
STORAGE_DIR_FACTOR_VALIDATION: Final[str] = "factor_validation"
STORAGE_DIR_FACTOR_SELECTION: Final[str] = "factor_selection"
STORAGE_DIR_FACTOR_TIMEFRAME_ANALYSIS: Final[str] = "factor_timeframe_analysis"
STORAGE_DIR_FACTOR_COMBINATION: Final[str] = "factor_combination"
STORAGE_DIR_FACTOR_ORTHOGONALIZATION: Final[str] = "factor_orthogonalization"
STORAGE_DIR_ALPHA: Final[str] = "alpha"
STORAGE_DIR_REGIME: Final[str] = "regime"
STORAGE_DIR_MODELS: Final[str] = "models"
STORAGE_DIR_WALK_FORWARD: Final[str] = "walk_forward"
STORAGE_DIR_WALK_FORWARD_EVALUATION: Final[str] = "walk_forward_evaluation"
STORAGE_DIR_PURGED_CV: Final[str] = "purged_cv"
STORAGE_DIR_PURGED_CV_EVALUATION: Final[str] = "purged_cv_evaluation"
STORAGE_DIR_FACTORS: Final[str] = "factors"
STORAGE_DIR_REPORTS: Final[str] = "reports"
STORAGE_DIR_CACHE: Final[str] = "cache"
STORAGE_DIR_METADATA: Final[str] = "metadata"

# ---------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------

FILE_FORMAT_PARQUET: Final[str] = "parquet"
FILE_FORMAT_JSON: Final[str] = "json"
FILE_FORMAT_YAML: Final[str] = "yaml"
FILE_FORMAT_TOML: Final[str] = "toml"
FILE_FORMAT_CSV: Final[str] = "csv"

FILE_EXTENSION_PARQUET: Final[str] = ".parquet"
FILE_EXTENSION_JSON: Final[str] = ".json"
FILE_EXTENSION_YAML: Final[str] = ".yaml"
FILE_EXTENSION_YML: Final[str] = ".yml"
FILE_EXTENSION_TOML: Final[str] = ".toml"
FILE_EXTENSION_CSV: Final[str] = ".csv"

SUPPORTED_DATASET_FORMATS: Final[frozenset[str]] = frozenset({FILE_FORMAT_PARQUET})
SUPPORTED_CONFIG_FORMATS: Final[frozenset[str]] = frozenset(
    {
        FILE_FORMAT_TOML,
        FILE_FORMAT_YAML,
    }
)

# ---------------------------------------------------------------------------
# Time conversions
# ---------------------------------------------------------------------------

MILLISECONDS_PER_SECOND: Final[int] = 1_000
MICROSECONDS_PER_SECOND: Final[int] = 1_000_000
NANOSECONDS_PER_SECOND: Final[int] = 1_000_000_000

SECONDS_PER_MINUTE: Final[int] = 60
MINUTES_PER_HOUR: Final[int] = 60
HOURS_PER_DAY: Final[int] = 24
DAYS_PER_WEEK: Final[int] = 7
DAYS_PER_YEAR: Final[int] = 365

SECONDS_PER_HOUR: Final[int] = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
SECONDS_PER_DAY: Final[int] = SECONDS_PER_HOUR * HOURS_PER_DAY

MILLISECONDS_PER_MINUTE: Final[int] = MILLISECONDS_PER_SECOND * SECONDS_PER_MINUTE
MILLISECONDS_PER_HOUR: Final[int] = MILLISECONDS_PER_MINUTE * MINUTES_PER_HOUR
MILLISECONDS_PER_DAY: Final[int] = MILLISECONDS_PER_HOUR * HOURS_PER_DAY

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

SEMVER_PATTERN: Final[str] = r"[0-9]+\.[0-9]+\.[0-9]+"
TIMEFRAME_PATTERN: Final[str] = r"[1-9][0-9]*[smhdwM]"

HASH_ALGORITHM_SHA256: Final[str] = "sha256"
DEFAULT_HASH_ALGORITHM: Final[str] = HASH_ALGORITHM_SHA256

MISSING_DATA_POLICY_REJECT: Final[str] = "reject"
MISSING_DATA_POLICY_WARN: Final[str] = "warn"
MISSING_DATA_POLICY_IGNORE: Final[str] = "ignore"

SUPPORTED_MISSING_DATA_POLICIES: Final[frozenset[str]] = frozenset(
    {
        MISSING_DATA_POLICY_REJECT,
        MISSING_DATA_POLICY_WARN,
        MISSING_DATA_POLICY_IGNORE,
    }
)

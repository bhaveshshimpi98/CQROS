"""CQROS configuration data models.

This module defines immutable configuration dataclasses used by the
central configuration system. Models carry defaults only; loading,
merging, environment overrides, and validation are handled elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum


class Environment(StrEnum):
    """Supported CQROS deployment environment profiles."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PAPER = "paper"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Allowed logging severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(StrEnum):
    """Supported log message serialization formats."""

    JSON = "json"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Application identity and runtime settings.

    Attributes:
        name: Application display name.
        version: Application semantic version string.
        environment: Active deployment environment profile.
        timezone: Canonical timezone for timestamps (UTC required).
        debug: Whether debug mode is enabled.
    """

    name: str = "CQROS"
    version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    timezone: str = "UTC"
    debug: bool = False


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Logging behavior settings.

    Attributes:
        level: Minimum severity level emitted by the logger.
        format: Serialization format for log records.
        console: Whether logs are written to the console.
        file: Whether logs are written to files.
        directory: Relative directory for log files.
    """

    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.JSON
    console: bool = True
    file: bool = True
    directory: str = "logs"


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Dataset and artifact path settings.

    Path values are relative directory names resolved against the
    storage root by the storage layer.

    Attributes:
        root: Root directory for all CQROS data artifacts.
        raw: Subdirectory for immutable raw market data.
        processed: Subdirectory for processed datasets.
        features: Subdirectory for feature datasets.
        models: Subdirectory for model artifacts.
        reports: Subdirectory for research and operational reports.
    """

    root: str = "data"
    raw: str = "raw"
    processed: str = "processed"
    features: str = "features"
    models: str = "models"
    reports: str = "reports"


@dataclass(frozen=True, slots=True)
class ExchangeConfig:
    """Primary exchange connectivity settings.

    Attributes:
        name: Exchange identifier (for example, ``binance``).
        market: Market segment identifier (for example, ``usdt_perpetual``).
        testnet: Whether to use the exchange testnet endpoints.
    """

    name: str = "binance"
    market: str = "usdt_perpetual"
    testnet: bool = False


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Risk limit settings applied before execution.

    Attributes:
        max_drawdown: Maximum allowed portfolio drawdown as a fraction.
        max_leverage: Maximum allowed account leverage.
        max_position_size: Maximum size of a single position as a
            fraction of portfolio equity.
        stop_loss_required: Whether every position must define a stop loss.
    """

    max_drawdown: float = 0.15
    max_leverage: float = 3.0
    max_position_size: float = 0.10
    stop_loss_required: bool = True


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Research, dataset, and experiment settings.

    Attributes:
        random_seed: Seed used for reproducible research workloads.
        parallel: Whether research workloads may execute in parallel.
        save_checkpoints: Whether training checkpoints are persisted.
        hpo_enabled: Whether hyperparameter optimization is enabled.
        hpo_max_trials: Maximum number of HPO trials per study.
        hpo_timeout_minutes: Wall-clock HPO timeout in minutes.
        dataset_versioning: Whether dataset outputs are versioned.
        dataset_compression: Compression codec for dataset artifacts.
        dataset_chunk_size: Row count per dataset write chunk.
        feature_parallel: Whether feature engineering may run in parallel.
        store_intermediate_features: Whether intermediate feature
            artifacts are persisted.
        timeframes: Default candle timeframes included in research data.
        max_symbols: Maximum number of symbols in a research universe.
        history_days: Default historical lookback window in days.
        worker_count: Default bounded worker-pool size for research and CLI
            concurrency. Caps at 32 and falls back to 8 when CPU count is
            unavailable.
    """

    random_seed: int = 42
    parallel: bool = True
    save_checkpoints: bool = True
    hpo_enabled: bool = True
    hpo_max_trials: int = 100
    hpo_timeout_minutes: int = 240
    dataset_versioning: bool = True
    dataset_compression: str = "zstd"
    dataset_chunk_size: int = 500_000
    feature_parallel: bool = True
    store_intermediate_features: bool = False
    timeframes: tuple[str, ...] = (
        "1m",
        "5m",
        "15m",
        "1h",
        "4h",
        "1d",
    )
    max_symbols: int = 200
    history_days: int = 3650
    worker_count: int = min(32, os.cpu_count() or 8)


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    """Dataset-specific historical download retention defaults.

    Retention policy belongs in configuration. Downloaders must not hardcode
    exchange retention limits; callers resolve and clamp windows using these
    values before issuing requests.

    Attributes:
        ohlcv_history_days: Default OHLCV lookback window in days.
        funding_history_days: Default funding-rate lookback window in days.
        futures_data_history_days: Maximum lookback for Binance Futures Data
            endpoints (open interest, taker volume, and long/short ratios).
        futures_data_safety_margin_days: Days subtracted from
            ``futures_data_history_days`` before Futures Data start times are
            computed, keeping requests strictly inside exchange retention.
    """

    ohlcv_history_days: int = 3650
    funding_history_days: int = 3650
    futures_data_history_days: int = 30
    futures_data_safety_margin_days: int = 1


@dataclass(frozen=True, slots=True)
class Config:
    """Root CQROS configuration object.

    Nested sections are immutable and constructed with documented
    defaults. Configuration becomes immutable after startup; this
    object is the in-memory representation of that contract.

    Attributes:
        config_version: Configuration schema version.
        app: Application identity and runtime settings.
        logging: Logging behavior settings.
        storage: Dataset and artifact path settings.
        exchange: Primary exchange connectivity settings.
        risk: Risk limit settings.
        research: Research, dataset, and experiment settings.
        download: Dataset-specific download retention defaults.
    """

    config_version: str = "1.0.0"
    app: AppConfig = field(default_factory=AppConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)

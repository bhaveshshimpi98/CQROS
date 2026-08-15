"""Unit tests for CQROS configuration models."""

from __future__ import annotations

import pytest

from cqros.config.models import (
    AppConfig,
    Config,
    DownloadConfig,
    Environment,
    ExchangeConfig,
    LogFormat,
    LoggingConfig,
    LogLevel,
    ResearchConfig,
    RiskConfig,
    StorageConfig,
)


def test_app_config_defaults() -> None:
    """AppConfig exposes documented identity defaults."""
    config = AppConfig()

    assert config.name == "CQROS"
    assert config.version == "1.0.0"
    assert config.environment is Environment.DEVELOPMENT
    assert config.timezone == "UTC"
    assert config.debug is False


def test_logging_config_defaults() -> None:
    """LoggingConfig exposes documented logging defaults."""
    config = LoggingConfig()

    assert config.level is LogLevel.INFO
    assert config.format is LogFormat.JSON
    assert config.console is True
    assert config.file is True
    assert config.directory == "logs"


def test_storage_config_defaults() -> None:
    """StorageConfig exposes documented path defaults."""
    config = StorageConfig()

    assert config.root == "data"
    assert config.raw == "raw"
    assert config.processed == "processed"
    assert config.features == "features"
    assert config.models == "models"
    assert config.reports == "reports"


def test_exchange_config_defaults() -> None:
    """ExchangeConfig exposes documented connectivity defaults."""
    config = ExchangeConfig()

    assert config.name == "binance"
    assert config.market == "usdt_perpetual"
    assert config.testnet is False


def test_risk_config_defaults() -> None:
    """RiskConfig exposes documented risk-limit defaults."""
    config = RiskConfig()

    assert config.max_drawdown == pytest.approx(0.15)
    assert config.max_leverage == pytest.approx(3.0)
    assert config.max_position_size == pytest.approx(0.10)
    assert config.stop_loss_required is True


def test_research_config_defaults() -> None:
    """ResearchConfig exposes documented research defaults."""
    import os

    config = ResearchConfig()

    assert config.random_seed == 42
    assert config.parallel is True
    assert config.save_checkpoints is True
    assert config.hpo_enabled is True
    assert config.hpo_max_trials == 100
    assert config.hpo_timeout_minutes == 240
    assert config.dataset_versioning is True
    assert config.dataset_compression == "zstd"
    assert config.dataset_chunk_size == 500_000
    assert config.feature_parallel is True
    assert config.store_intermediate_features is False
    assert config.timeframes == ("1m", "5m", "15m", "1h", "4h", "1d")
    assert config.max_symbols == 200
    assert config.history_days == 3650
    assert config.worker_count == min(32, os.cpu_count() or 8)


def test_download_config_defaults() -> None:
    """DownloadConfig exposes dataset-specific retention defaults."""
    config = DownloadConfig()

    assert config.ohlcv_history_days == 3650
    assert config.funding_history_days == 3650
    assert config.futures_data_history_days == 30
    assert config.futures_data_safety_margin_days == 1


def test_root_config_defaults_compose_nested_sections() -> None:
    """Config composes nested sections with documented defaults."""
    config = Config()

    assert config.config_version == "1.0.0"
    assert config.app == AppConfig()
    assert config.logging == LoggingConfig()
    assert config.storage == StorageConfig()
    assert config.exchange == ExchangeConfig()
    assert config.risk == RiskConfig()
    assert config.research == ResearchConfig()
    assert config.download == DownloadConfig()


def test_config_sections_are_frozen() -> None:
    """All configuration dataclasses reject attribute mutation."""
    config = Config()

    with pytest.raises(AttributeError):
        config.app.debug = True  # type: ignore[misc]

    with pytest.raises(AttributeError):
        config.logging.level = LogLevel.DEBUG  # type: ignore[misc]

    with pytest.raises(AttributeError):
        config.storage.root = "other"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        config.exchange.testnet = True  # type: ignore[misc]

    with pytest.raises(AttributeError):
        config.risk.max_leverage = 1.0  # type: ignore[misc]

    with pytest.raises(AttributeError):
        config.research.random_seed = 7  # type: ignore[misc]

    with pytest.raises(AttributeError):
        config.download.futures_data_history_days = 7  # type: ignore[misc]

    with pytest.raises(AttributeError):
        config.config_version = "2.0.0"  # type: ignore[misc]


def test_config_accepts_explicit_overrides() -> None:
    """Config accepts explicit nested overrides without mutation."""
    config = Config(
        config_version="1.1.0",
        app=AppConfig(
            environment=Environment.PRODUCTION,
            debug=False,
        ),
        logging=LoggingConfig(level=LogLevel.WARNING),
        exchange=ExchangeConfig(testnet=True),
        risk=RiskConfig(max_drawdown=0.10),
        research=ResearchConfig(random_seed=123),
    )

    assert config.config_version == "1.1.0"
    assert config.app.environment is Environment.PRODUCTION
    assert config.logging.level is LogLevel.WARNING
    assert config.exchange.testnet is True
    assert config.risk.max_drawdown == pytest.approx(0.10)
    assert config.research.random_seed == 123

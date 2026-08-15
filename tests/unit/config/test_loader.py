"""Unit tests for CQROS configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from cqros.config.loader import (
    ConfigLoader,
    ConfigLoadError,
    ConfigurationValidationError,
)
from cqros.config.models import Environment, LogFormat, LogLevel


def test_load_empty_toml_uses_defaults(tmp_path: Path) -> None:
    """An empty TOML file loads a fully defaulted Config."""
    path = tmp_path / "empty.toml"
    path.write_text("", encoding="utf-8")

    config = ConfigLoader().load(path)

    assert config.config_version == "1.0.0"
    assert config.app.name == "CQROS"
    assert config.app.environment is Environment.DEVELOPMENT
    assert config.logging.level is LogLevel.INFO
    assert config.logging.format is LogFormat.JSON
    assert config.exchange.name == "binance"
    assert config.risk.max_drawdown == 0.15
    assert config.research.timeframes == ("1m", "5m", "15m", "1h", "4h", "1d")


def test_load_partial_overrides_preserve_defaults(tmp_path: Path) -> None:
    """Partial section overrides merge with dataclass defaults."""
    path = tmp_path / "partial.toml"
    path.write_text(
        """
config_version = "2.0.0"

[app]
name = "CQROS-Research"
environment = "paper"
debug = true

[risk]
max_leverage = 2
""",
        encoding="utf-8",
    )

    config = ConfigLoader().load(path)

    assert config.config_version == "2.0.0"
    assert config.app.name == "CQROS-Research"
    assert config.app.environment is Environment.PAPER
    assert config.app.debug is True
    assert config.app.timezone == "UTC"
    assert config.risk.max_leverage == 2.0
    assert config.risk.max_drawdown == 0.15
    assert config.storage.root == "data"


def test_load_research_timeframes_become_tuple(tmp_path: Path) -> None:
    """TOML arrays for research.timeframes are converted to tuples."""
    path = tmp_path / "research.toml"
    path.write_text(
        """
[research]
timeframes = ["1m", "1h", "1d"]
max_symbols = 50
""",
        encoding="utf-8",
    )

    config = ConfigLoader().load(path)

    assert config.research.timeframes == ("1m", "1h", "1d")
    assert config.research.max_symbols == 50


def test_load_full_valid_configuration(tmp_path: Path) -> None:
    """A complete valid TOML file maps onto every configuration section."""
    path = tmp_path / "full.toml"
    path.write_text(
        """
config_version = "1.2.3"

[app]
name = "CQROS"
version = "1.2.3"
environment = "testing"
timezone = "UTC"
debug = false

[logging]
level = "DEBUG"
format = "text"
console = true
file = false
directory = "var/log"

[storage]
root = "datasets"
raw = "raw_data"
processed = "processed_data"
features = "feature_store"
models = "model_store"
reports = "report_store"

[exchange]
name = "binance"
market = "usdt_perpetual"
testnet = true

[risk]
max_drawdown = 0.2
max_leverage = 1.5
max_position_size = 0.05
stop_loss_required = true

[research]
random_seed = 7
parallel = false
save_checkpoints = false
hpo_enabled = false
hpo_max_trials = 10
hpo_timeout_minutes = 30
dataset_versioning = true
dataset_compression = "snappy"
dataset_chunk_size = 1000
feature_parallel = false
store_intermediate_features = true
timeframes = ["5m", "1h"]
max_symbols = 25
history_days = 365
""",
        encoding="utf-8",
    )

    config = ConfigLoader().load(path)

    assert config.config_version == "1.2.3"
    assert config.app.environment is Environment.TESTING
    assert config.logging.level is LogLevel.DEBUG
    assert config.logging.format is LogFormat.TEXT
    assert config.logging.file is False
    assert config.storage.root == "datasets"
    assert config.exchange.testnet is True
    assert config.risk.max_position_size == 0.05
    assert config.research.dataset_compression == "snappy"
    assert config.research.timeframes == ("5m", "1h")


def test_load_missing_file_raises_config_load_error(tmp_path: Path) -> None:
    """Missing configuration files raise ConfigLoadError."""
    missing = tmp_path / "missing.toml"

    with pytest.raises(ConfigLoadError, match="not found"):
        ConfigLoader().load(missing)


def test_load_invalid_toml_raises_config_load_error(tmp_path: Path) -> None:
    """Malformed TOML raises ConfigLoadError."""
    path = tmp_path / "bad.toml"
    path.write_text("app = [unterminated", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="Invalid TOML"):
        ConfigLoader().load(path)


def test_load_invalid_enum_raises_config_load_error(tmp_path: Path) -> None:
    """Unrecognized enum values fail during model construction."""
    path = tmp_path / "bad_enum.toml"
    path.write_text(
        """
[app]
environment = "staging"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError, match="Invalid value for field 'environment'"):
        ConfigLoader().load(path)


def test_load_non_table_section_raises_config_load_error(tmp_path: Path) -> None:
    """Scalar values where nested tables are required raise ConfigLoadError."""
    path = tmp_path / "bad_section.toml"
    path.write_text('app = "not-a-table"\n', encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="Expected a TOML table for field 'app'"):
        ConfigLoader().load(path)


def test_validation_failure_raises_configuration_validation_error(
    tmp_path: Path,
) -> None:
    """Semantic validation failures raise ConfigurationValidationError."""
    path = tmp_path / "invalid.toml"
    path.write_text(
        """
[app]
timezone = "Asia/Kolkata"

[risk]
max_drawdown = 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationValidationError) as exc_info:
        ConfigLoader().load(path)

    error = exc_info.value
    assert error.result.valid is False
    sections = {issue.section for issue in error.result.errors()}
    assert "app" in sections
    assert "risk" in sections


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    """Unknown TOML keys do not prevent loading known fields."""
    path = tmp_path / "extra.toml"
    path.write_text(
        """
future_root_key = true

[app]
name = "CQROS"
extra_app_key = 123

[unknown_section]
enabled = true
""",
        encoding="utf-8",
    )

    config = ConfigLoader().load(path)

    assert config.app.name == "CQROS"
    assert config.app.version == "1.0.0"


def test_load_accepts_string_path(tmp_path: Path) -> None:
    """load() accepts string paths in addition to Path objects."""
    path = tmp_path / "string_path.toml"
    path.write_text('config_version = "1.0.0"\n', encoding="utf-8")

    config = ConfigLoader().load(str(path))

    assert config.config_version == "1.0.0"

"""Unit tests for CQROS configuration validation."""

from __future__ import annotations

from cqros.config.models import (
    AppConfig,
    Config,
    DownloadConfig,
    Environment,
    ExchangeConfig,
    LoggingConfig,
    ResearchConfig,
    RiskConfig,
    StorageConfig,
)
from cqros.config.validator import (
    ConfigValidator,
    ValidationResult,
    ValidationSeverity,
)


def test_default_config_is_valid() -> None:
    """Default Config passes validation with no issues."""
    result = ConfigValidator().validate(Config())

    assert result.valid is True
    assert result.has_errors() is False
    assert result.has_warnings() is False
    assert result.issues == []


def test_validation_result_helpers_accumulate_issues() -> None:
    """ValidationResult helpers record errors and warnings independently."""
    result = ValidationResult()
    result.add_error("app", "name", "name missing", "")
    result.add_warning("risk", "max_leverage", "high leverage", 25.0)

    assert result.valid is False
    assert result.has_errors() is True
    assert result.has_warnings() is True
    assert len(result.errors()) == 1
    assert len(result.warnings()) == 1
    assert result.errors()[0].severity is ValidationSeverity.ERROR
    assert result.warnings()[0].severity is ValidationSeverity.WARNING


def test_collects_multiple_section_errors_without_short_circuit() -> None:
    """Validator continues after failures and collects issues across sections."""
    config = Config(
        config_version="",
        app=AppConfig(name="", timezone="Asia/Kolkata"),
        logging=LoggingConfig(console=False, file=False, directory=""),
        storage=StorageConfig(root=""),
        exchange=ExchangeConfig(name="", market=""),
        risk=RiskConfig(max_drawdown=0.0, max_leverage=0.0, max_position_size=2.0),
        research=ResearchConfig(
            random_seed=-1,
            hpo_max_trials=0,
            hpo_timeout_minutes=0,
            dataset_compression="xz",
            dataset_chunk_size=0,
            timeframes=(),
            max_symbols=0,
            history_days=0,
            worker_count=0,
        ),
        download=DownloadConfig(
            ohlcv_history_days=0,
            funding_history_days=0,
            futures_data_history_days=0,
        ),
    )

    result = ConfigValidator().validate(config)

    assert result.valid is False
    sections = {issue.section for issue in result.errors()}
    assert sections == {
        "config",
        "app",
        "logging",
        "storage",
        "exchange",
        "risk",
        "research",
        "download",
    }
    assert len(result.errors()) >= 15


def test_futures_data_safety_margin_must_leave_positive_effective_history() -> None:
    """Safety margin that exhausts Futures Data retention is rejected."""
    result = ConfigValidator().validate(
        Config(
            download=DownloadConfig(
                futures_data_history_days=30,
                futures_data_safety_margin_days=30,
            )
        ),
    )

    assert result.valid is False
    issue = next(
        issue
        for issue in result.errors()
        if issue.field == "futures_data_safety_margin_days"
        and "must be greater than 0" in issue.message
    )
    assert issue.section == "download"
    assert issue.value == 0


def test_app_timezone_must_be_utc() -> None:
    """App timezone values other than UTC are rejected."""
    result = ConfigValidator().validate(
        Config(app=AppConfig(timezone="US/Eastern")),
    )

    assert result.valid is False
    issue = next(issue for issue in result.errors() if issue.field == "timezone")
    assert issue.section == "app"
    assert issue.value == "US/Eastern"


def test_production_debug_emits_warning_not_error() -> None:
    """Production debug mode is a warning and does not invalidate config."""
    result = ConfigValidator().validate(
        Config(
            app=AppConfig(
                environment=Environment.PRODUCTION,
                debug=True,
            ),
        ),
    )

    assert result.valid is True
    assert result.has_warnings() is True
    warning = next(issue for issue in result.warnings() if issue.field == "debug")
    assert warning.section == "app"


def test_production_testnet_emits_warning() -> None:
    """Production testnet usage is reported as a warning."""
    result = ConfigValidator().validate(
        Config(
            app=AppConfig(environment=Environment.PRODUCTION),
            exchange=ExchangeConfig(testnet=True),
        ),
    )

    assert result.valid is True
    warning = next(issue for issue in result.warnings() if issue.field == "testnet")
    assert warning.section == "exchange"


def test_risk_fraction_bounds() -> None:
    """Risk fractions outside (0, 1] are errors."""
    result = ConfigValidator().validate(
        Config(risk=RiskConfig(max_drawdown=1.5, max_position_size=-0.1)),
    )

    fields = {issue.field for issue in result.errors()}
    assert "max_drawdown" in fields
    assert "max_position_size" in fields


def test_high_leverage_warning() -> None:
    """Unusually high leverage emits a warning while remaining valid."""
    result = ConfigValidator().validate(
        Config(risk=RiskConfig(max_leverage=25.0)),
    )

    assert result.valid is True
    warning = next(issue for issue in result.warnings() if issue.field == "max_leverage")
    assert warning.value == 25.0


def test_research_timeframe_and_compression_rules() -> None:
    """Invalid research timeframes and compression codecs are errors."""
    result = ConfigValidator().validate(
        Config(
            research=ResearchConfig(
                dataset_compression="lzma",
                timeframes=("1m", "bad", "15x"),
            ),
        ),
    )

    assert result.valid is False
    fields = {issue.field for issue in result.errors()}
    assert "dataset_compression" in fields
    assert "timeframes" in fields
    assert sum(1 for issue in result.errors() if issue.field == "timeframes") == 2


def test_dataset_versioning_disabled_warning() -> None:
    """Disabling dataset versioning emits a reproducibility warning."""
    result = ConfigValidator().validate(
        Config(research=ResearchConfig(dataset_versioning=False)),
    )

    assert result.valid is True
    warning = next(issue for issue in result.warnings() if issue.field == "dataset_versioning")
    assert warning.section == "research"


def test_unsupported_exchange_and_market() -> None:
    """Unsupported exchange identifiers are rejected."""
    result = ConfigValidator().validate(
        Config(exchange=ExchangeConfig(name="coinbase", market="spot")),
    )

    assert result.valid is False
    fields = {issue.field for issue in result.errors()}
    assert fields == {"name", "market"}

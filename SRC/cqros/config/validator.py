"""CQROS configuration validation.

This module validates immutable configuration dataclasses without raising
on failure. Every section is checked independently and all issues are
collected into a single ``ValidationResult``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from re import fullmatch

from cqros.config.models import (
    AppConfig,
    Config,
    Environment,
    ExchangeConfig,
)

_SEMVER_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+"
_TIMEFRAME_PATTERN = r"[1-9][0-9]*[smhdwM]"
_ALLOWED_COMPRESSION = frozenset(
    {
        "zstd",
        "snappy",
        "gzip",
        "lz4",
        "brotli",
        "uncompressed",
        "none",
    }
)
_ALLOWED_EXCHANGES = frozenset({"binance"})
_ALLOWED_MARKETS = frozenset({"usdt_perpetual"})


class ValidationSeverity(StrEnum):
    """Severity assigned to a configuration validation issue."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single configuration validation finding.

    Attributes:
        severity: Whether the issue is an error or a warning.
        section: Configuration section name (for example, ``risk``).
        field: Field name within the section.
        message: Human-readable description of the problem.
        value: The value that failed validation, when available.
    """

    severity: ValidationSeverity
    section: str
    field: str
    message: str
    value: object | None = None


@dataclass(slots=True)
class ValidationResult:
    """Accumulated configuration validation outcome.

    Attributes:
        issues: Ordered list of all validation findings.
    """

    issues: list[ValidationIssue] = field(default_factory=list[ValidationIssue])

    @property
    def valid(self) -> bool:
        """Return ``True`` when no error-severity issues were recorded."""
        return not self.has_errors()

    def add_error(
        self,
        section: str,
        field_name: str,
        message: str,
        value: object | None = None,
    ) -> None:
        """Record an error-severity validation issue.

        Args:
            section: Configuration section name.
            field_name: Field name within the section.
            message: Human-readable description of the problem.
            value: The value that failed validation, when available.
        """
        self.issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                section=section,
                field=field_name,
                message=message,
                value=value,
            )
        )

    def add_warning(
        self,
        section: str,
        field_name: str,
        message: str,
        value: object | None = None,
    ) -> None:
        """Record a warning-severity validation issue.

        Args:
            section: Configuration section name.
            field_name: Field name within the section.
            message: Human-readable description of the problem.
            value: The value that triggered the warning, when available.
        """
        self.issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                section=section,
                field=field_name,
                message=message,
                value=value,
            )
        )

    def has_errors(self) -> bool:
        """Return ``True`` when at least one error-severity issue exists."""
        return any(issue.severity is ValidationSeverity.ERROR for issue in self.issues)

    def has_warnings(self) -> bool:
        """Return ``True`` when at least one warning-severity issue exists."""
        return any(issue.severity is ValidationSeverity.WARNING for issue in self.issues)

    def errors(self) -> list[ValidationIssue]:
        """Return all error-severity issues in discovery order."""
        return [issue for issue in self.issues if issue.severity is ValidationSeverity.ERROR]

    def warnings(self) -> list[ValidationIssue]:
        """Return all warning-severity issues in discovery order."""
        return [issue for issue in self.issues if issue.severity is ValidationSeverity.WARNING]


type SectionValidator = Callable[[Config, ValidationResult], None]


class ConfigValidator:
    """Validate a complete CQROS ``Config`` object.

    Each configuration section is validated independently. Validation
    never raises for rule failures; callers inspect ``ValidationResult``.

    Section validators are registered as an ordered sequence so future
    sections can be appended without modifying existing section logic.
    """

    def __init__(self) -> None:
        """Initialize the validator with the default section pipeline."""
        self._section_validators: tuple[SectionValidator, ...] = (
            self._validate_root,
            self._validate_app,
            self._validate_logging,
            self._validate_storage,
            self._validate_exchange,
            self._validate_risk,
            self._validate_research,
            self._validate_download,
        )

    def validate(self, config: Config) -> ValidationResult:
        """Validate every configuration section and collect all issues.

        Args:
            config: Immutable root configuration to validate.

        Returns:
            Aggregated validation result. ``result.valid`` is ``True``
            when no error-severity issues were found.
        """
        result = ValidationResult()
        for validate_section in self._section_validators:
            validate_section(config, result)
        return result

    def _validate_root(self, config: Config, result: ValidationResult) -> None:
        """Validate root-level configuration fields."""
        if not _is_non_blank(config.config_version):
            result.add_error(
                "config",
                "config_version",
                "config_version must be a non-empty string",
                config.config_version,
            )
        elif not fullmatch(_SEMVER_PATTERN, config.config_version):
            result.add_error(
                "config",
                "config_version",
                "config_version must be a semantic version of the form MAJOR.MINOR.PATCH",
                config.config_version,
            )

    def _validate_app(self, config: Config, result: ValidationResult) -> None:
        """Validate application identity and runtime settings."""
        app = config.app
        section = "app"

        if not _is_non_blank(app.name):
            result.add_error(
                section,
                "name",
                "name must be a non-empty string",
                app.name,
            )

        if not _is_non_blank(app.version):
            result.add_error(
                section,
                "version",
                "version must be a non-empty string",
                app.version,
            )
        elif not fullmatch(_SEMVER_PATTERN, app.version):
            result.add_error(
                section,
                "version",
                "version must be a semantic version of the form MAJOR.MINOR.PATCH",
                app.version,
            )

        if app.timezone != "UTC":
            result.add_error(
                section,
                "timezone",
                "timezone must be UTC",
                app.timezone,
            )

        _validate_app_environment_warnings(app, result)

    def _validate_logging(self, config: Config, result: ValidationResult) -> None:
        """Validate logging behavior settings."""
        logging_config = config.logging
        section = "logging"

        if not _is_non_blank(logging_config.directory):
            result.add_error(
                section,
                "directory",
                "directory must be a non-empty string",
                logging_config.directory,
            )

        if not logging_config.console and not logging_config.file:
            result.add_error(
                section,
                "console",
                "at least one of console or file logging must be enabled",
                False,
            )

    def _validate_storage(self, config: Config, result: ValidationResult) -> None:
        """Validate dataset and artifact path settings."""
        storage = config.storage
        section = "storage"
        path_fields: tuple[tuple[str, str], ...] = (
            ("root", storage.root),
            ("raw", storage.raw),
            ("processed", storage.processed),
            ("features", storage.features),
            ("models", storage.models),
            ("reports", storage.reports),
        )
        for field_name, value in path_fields:
            if not _is_non_blank(value):
                result.add_error(
                    section,
                    field_name,
                    f"{field_name} must be a non-empty path segment",
                    value,
                )

    def _validate_exchange(self, config: Config, result: ValidationResult) -> None:
        """Validate primary exchange connectivity settings."""
        exchange = config.exchange
        section = "exchange"

        if not _is_non_blank(exchange.name):
            result.add_error(
                section,
                "name",
                "name must be a non-empty string",
                exchange.name,
            )
        elif exchange.name not in _ALLOWED_EXCHANGES:
            result.add_error(
                section,
                "name",
                f"unsupported exchange name; allowed values: {sorted(_ALLOWED_EXCHANGES)}",
                exchange.name,
            )

        if not _is_non_blank(exchange.market):
            result.add_error(
                section,
                "market",
                "market must be a non-empty string",
                exchange.market,
            )
        elif exchange.market not in _ALLOWED_MARKETS:
            result.add_error(
                section,
                "market",
                f"unsupported market; allowed values: {sorted(_ALLOWED_MARKETS)}",
                exchange.market,
            )

        _validate_exchange_environment_warnings(config.app, exchange, result)

    def _validate_risk(self, config: Config, result: ValidationResult) -> None:
        """Validate risk limit settings."""
        risk = config.risk
        section = "risk"

        if not (0.0 < risk.max_drawdown <= 1.0):
            result.add_error(
                section,
                "max_drawdown",
                "max_drawdown must be greater than 0.0 and less than or equal to 1.0",
                risk.max_drawdown,
            )

        if risk.max_leverage <= 0.0:
            result.add_error(
                section,
                "max_leverage",
                "max_leverage must be greater than 0.0",
                risk.max_leverage,
            )
        elif risk.max_leverage > 20.0:
            result.add_warning(
                section,
                "max_leverage",
                "max_leverage above 20.0 is unusually high for institutional risk controls",
                risk.max_leverage,
            )

        if not (0.0 < risk.max_position_size <= 1.0):
            result.add_error(
                section,
                "max_position_size",
                "max_position_size must be greater than 0.0 and less than or equal to 1.0",
                risk.max_position_size,
            )

        if not risk.stop_loss_required and config.app.environment is Environment.PRODUCTION:
            result.add_warning(
                section,
                "stop_loss_required",
                "stop_loss_required is disabled in production",
                risk.stop_loss_required,
            )

    def _validate_research(self, config: Config, result: ValidationResult) -> None:
        """Validate research, dataset, and experiment settings."""
        research = config.research
        section = "research"

        if research.random_seed < 0:
            result.add_error(
                section,
                "random_seed",
                "random_seed must be greater than or equal to 0",
                research.random_seed,
            )

        if research.hpo_max_trials <= 0:
            result.add_error(
                section,
                "hpo_max_trials",
                "hpo_max_trials must be greater than 0",
                research.hpo_max_trials,
            )

        if research.hpo_timeout_minutes <= 0:
            result.add_error(
                section,
                "hpo_timeout_minutes",
                "hpo_timeout_minutes must be greater than 0",
                research.hpo_timeout_minutes,
            )

        if not _is_non_blank(research.dataset_compression):
            result.add_error(
                section,
                "dataset_compression",
                "dataset_compression must be a non-empty string",
                research.dataset_compression,
            )
        elif research.dataset_compression not in _ALLOWED_COMPRESSION:
            result.add_error(
                section,
                "dataset_compression",
                "unsupported dataset_compression; "
                f"allowed values: {sorted(_ALLOWED_COMPRESSION)}",
                research.dataset_compression,
            )

        if research.dataset_chunk_size <= 0:
            result.add_error(
                section,
                "dataset_chunk_size",
                "dataset_chunk_size must be greater than 0",
                research.dataset_chunk_size,
            )

        if not research.timeframes:
            result.add_error(
                section,
                "timeframes",
                "timeframes must contain at least one timeframe",
                research.timeframes,
            )
        else:
            for timeframe in research.timeframes:
                if not fullmatch(_TIMEFRAME_PATTERN, timeframe):
                    result.add_error(
                        section,
                        "timeframes",
                        "timeframe values must match the pattern "
                        "<positive integer><s|m|h|d|w|M>",
                        timeframe,
                    )

        if research.max_symbols <= 0:
            result.add_error(
                section,
                "max_symbols",
                "max_symbols must be greater than 0",
                research.max_symbols,
            )

        if research.history_days <= 0:
            result.add_error(
                section,
                "history_days",
                "history_days must be greater than 0",
                research.history_days,
            )

        if research.worker_count <= 0:
            result.add_error(
                section,
                "worker_count",
                "worker_count must be greater than 0",
                research.worker_count,
            )

        if not research.dataset_versioning:
            result.add_warning(
                section,
                "dataset_versioning",
                "dataset_versioning is disabled; research reproducibility is reduced",
                research.dataset_versioning,
            )

    def _validate_download(self, config: Config, result: ValidationResult) -> None:
        """Validate dataset-specific download retention settings."""
        download = config.download
        section = "download"

        if download.ohlcv_history_days <= 0:
            result.add_error(
                section,
                "ohlcv_history_days",
                "ohlcv_history_days must be greater than 0",
                download.ohlcv_history_days,
            )

        if download.funding_history_days <= 0:
            result.add_error(
                section,
                "funding_history_days",
                "funding_history_days must be greater than 0",
                download.funding_history_days,
            )

        if download.futures_data_history_days <= 0:
            result.add_error(
                section,
                "futures_data_history_days",
                "futures_data_history_days must be greater than 0",
                download.futures_data_history_days,
            )

        if download.futures_data_safety_margin_days < 0:
            result.add_error(
                section,
                "futures_data_safety_margin_days",
                "futures_data_safety_margin_days must be greater than or equal to 0",
                download.futures_data_safety_margin_days,
            )

        effective_futures_history = (
            download.futures_data_history_days - download.futures_data_safety_margin_days
        )
        if (
            download.futures_data_history_days > 0
            and download.futures_data_safety_margin_days >= 0
            and effective_futures_history <= 0
        ):
            result.add_error(
                section,
                "futures_data_safety_margin_days",
                (
                    "futures_data_history_days minus futures_data_safety_margin_days "
                    "must be greater than 0"
                ),
                effective_futures_history,
            )

        if (
            download.futures_data_history_days > 0
            and download.funding_history_days > 0
            and download.futures_data_history_days > download.funding_history_days
        ):
            result.add_warning(
                section,
                "futures_data_history_days",
                "futures_data_history_days exceeds funding_history_days",
                download.futures_data_history_days,
            )


def _is_non_blank(value: str) -> bool:
    """Return ``True`` when ``value`` contains non-whitespace characters."""
    return bool(value.strip())


def _validate_app_environment_warnings(
    app: AppConfig,
    result: ValidationResult,
) -> None:
    """Record environment-dependent warnings for application settings."""
    if app.environment is Environment.PRODUCTION and app.debug:
        result.add_warning(
            "app",
            "debug",
            "debug mode is enabled in production",
            app.debug,
        )


def _validate_exchange_environment_warnings(
    app: AppConfig,
    exchange: ExchangeConfig,
    result: ValidationResult,
) -> None:
    """Record environment-dependent warnings for exchange settings."""
    if app.environment is Environment.PRODUCTION and exchange.testnet:
        result.add_warning(
            "exchange",
            "testnet",
            "testnet is enabled in production",
            exchange.testnet,
        )

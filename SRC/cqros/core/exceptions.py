"""CQROS exception hierarchy.

Purpose:
    Provide the shared, extensible exception taxonomy used across the CQROS
    platform.

Responsibilities:
    - Define the root ``CQROSError`` type that every project exception inherits
    - Carry structured diagnostic fields (``message``, ``error_code``,
      ``details``) without requiring subclass API changes
    - Expose domain category exceptions for infrastructure, configuration,
      exchange, data, research, risk, execution, and models

Dependencies:
    Python standard library only.

Public API:
    ``CQROSError`` and the domain / specialized exception types listed in
    ``__all__``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "CQROSError",
    "InfrastructureError",
    "DependencyError",
    "InternalError",
    "TimeoutError",
    "ResourceError",
    "ConfigurationError",
    "ConfigNotFoundError",
    "ConfigParseError",
    "ConfigValidationError",
    "ValidationError",
    "ExchangeError",
    "ExchangeRateLimitError",
    "ExchangeAuthenticationError",
    "ExchangeUnavailableError",
    "ExchangeTimeoutError",
    "ExchangeValidationError",
    "ExchangePermissionError",
    "ExchangeSymbolNotFoundError",
    "DataError",
    "SchemaError",
    "DuplicateDataError",
    "MissingDataError",
    "IntegrityError",
    "DataValidationError",
    "ResearchError",
    "DatasetError",
    "FeatureError",
    "TargetError",
    "LeakageError",
    "ExperimentError",
    "ModelError",
    "TrainingError",
    "InferenceError",
    "SerializationError",
    "CheckpointError",
    "ModelValidationError",
    "RiskError",
    "ExposureError",
    "VaRError",
    "CVaRError",
    "StressTestError",
    "RiskLimitError",
    "ExecutionError",
    "OrderError",
    "RoutingError",
    "AlgorithmError",
    "FillError",
]


class CQROSError(Exception):
    """Base exception for all CQROS errors.

    Every project exception must inherit from this class. Optional
    ``error_code`` and ``details`` support stable programmatic handling and
    diagnostic context without changing subclass constructors.

    Args:
        message: Human-readable description of the failure.
        error_code: Optional stable machine-readable identifier.
        details: Optional mapping of additional diagnostic context.
        recovery_suggestion: Optional guidance for recovering from the failure.

    Attributes:
        message: Human-readable description of the failure.
        error_code: Optional stable machine-readable identifier.
        details: Read-only mapping of additional diagnostic context.
        recovery_suggestion: Optional recovery guidance.
    """

    __slots__ = ("message", "error_code", "details", "recovery_suggestion")

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        details: Mapping[str, object] | None = None,
        recovery_suggestion: str | None = None,
    ) -> None:
        """Initialize the error with message and optional structured context.

        Args:
            message: Human-readable description of the failure.
            error_code: Optional stable machine-readable identifier.
            details: Optional mapping of additional diagnostic context.
            recovery_suggestion: Optional guidance for recovering from the
                failure.
        """
        self.message = message
        self.error_code = error_code
        self.details: Mapping[str, object] = MappingProxyType(
            dict(details) if details is not None else {}
        )
        self.recovery_suggestion = recovery_suggestion
        super().__init__(message)

    def __str__(self) -> str:
        """Return a human-readable representation with code and details.

        Returns:
            Formatted error string including optional code, details, and
            recovery guidance.
        """
        parts: list[str] = []
        if self.error_code is not None:
            parts.append(f"[{self.error_code}]")
        parts.append(self.message)
        if self.details:
            rendered = ", ".join(f"{key}={value!r}" for key, value in self.details.items())
            parts.append(f"({rendered})")
        if self.recovery_suggestion is not None:
            parts.append(f"| recovery: {self.recovery_suggestion}")
        return " ".join(parts)

    def __repr__(self) -> str:
        """Return an unambiguous developer-facing representation.

        Returns:
            Constructor-style representation of this exception instance.
        """
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"details={dict(self.details)!r}, "
            f"recovery_suggestion={self.recovery_suggestion!r})"
        )


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


class InfrastructureError(CQROSError):
    """Raised when platform infrastructure or runtime services fail."""

    __slots__ = ()


class DependencyError(InfrastructureError):
    """Raised when a required dependency is missing or incompatible."""

    __slots__ = ()


class InternalError(InfrastructureError):
    """Raised for unexpected internal failures that indicate a defect."""

    __slots__ = ()


class TimeoutError(InfrastructureError):
    """Raised when an operation exceeds its allowed duration.

    This CQROS-specific timeout type is distinct from the built-in
    ``TimeoutError``. Prefer importing it from this module explicitly.
    """

    __slots__ = ()


class ResourceError(InfrastructureError):
    """Raised when a required compute, memory, or system resource is unavailable."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigurationError(CQROSError):
    """Raised when configuration loading, parsing, or validation fails."""

    __slots__ = ()


class ConfigNotFoundError(ConfigurationError):
    """Raised when a required configuration file or section cannot be found."""

    __slots__ = ()


class ConfigParseError(ConfigurationError):
    """Raised when configuration content cannot be parsed."""

    __slots__ = ()


class ConfigValidationError(ConfigurationError):
    """Raised when configuration values fail semantic validation."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Cross-cutting validation
# ---------------------------------------------------------------------------


class ValidationError(CQROSError):
    """Raised when input, schema, or business-rule validation fails."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------


class ExchangeError(CQROSError):
    """Raised when exchange connectivity or exchange operations fail."""

    __slots__ = ()


class ExchangeRateLimitError(ExchangeError):
    """Raised when an exchange rate limit is exceeded."""

    __slots__ = ()


class ExchangeAuthenticationError(ExchangeError):
    """Raised when exchange authentication or credential validation fails."""

    __slots__ = ()


class ExchangeUnavailableError(ExchangeError):
    """Raised when an exchange endpoint is unavailable."""

    __slots__ = ()


class ExchangeTimeoutError(ExchangeError):
    """Raised when an exchange request exceeds its allowed duration."""

    __slots__ = ()


class ExchangeValidationError(ExchangeError):
    """Raised when an exchange rejects a request as invalid."""

    __slots__ = ()


class ExchangePermissionError(ExchangeError):
    """Raised when exchange permissions are insufficient for an operation."""

    __slots__ = ()


class ExchangeSymbolNotFoundError(ExchangeError):
    """Raised when a requested exchange symbol cannot be resolved."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class DataError(CQROSError):
    """Raised when historical data ingestion, storage, or access fails."""

    __slots__ = ()


class SchemaError(DataError):
    """Raised when a dataset schema is missing, incompatible, or invalid."""

    __slots__ = ()


class DuplicateDataError(DataError):
    """Raised when unexpected duplicate records are detected."""

    __slots__ = ()


class MissingDataError(DataError):
    """Raised when required data is missing."""

    __slots__ = ()


class IntegrityError(DataError):
    """Raised when data integrity checks fail."""

    __slots__ = ()


class DataValidationError(DataError):
    """Raised when dataset content fails data-layer validation rules."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


class ResearchError(CQROSError):
    """Raised when research workflows or research artifacts fail."""

    __slots__ = ()


class DatasetError(ResearchError):
    """Raised when research dataset construction or access fails."""

    __slots__ = ()


class FeatureError(ResearchError):
    """Raised when feature engineering or feature metadata handling fails."""

    __slots__ = ()


class TargetError(ResearchError):
    """Raised when target generation or target metadata handling fails."""

    __slots__ = ()


class LeakageError(ResearchError):
    """Raised when future leakage or look-ahead contamination is detected."""

    __slots__ = ()


class ExperimentError(ResearchError):
    """Raised when experiment tracking or experiment execution fails."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ModelError(CQROSError):
    """Raised when model training, inference, or artifact handling fails."""

    __slots__ = ()


class TrainingError(ModelError):
    """Raised when model training fails."""

    __slots__ = ()


class InferenceError(ModelError):
    """Raised when model inference fails."""

    __slots__ = ()


class SerializationError(ModelError):
    """Raised when model serialization or deserialization fails."""

    __slots__ = ()


class CheckpointError(ModelError):
    """Raised when model checkpoint creation or restoration fails."""

    __slots__ = ()


class ModelValidationError(ModelError):
    """Raised when model validation or promotion checks fail."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


class RiskError(CQROSError):
    """Raised when risk calculation or risk-policy enforcement fails."""

    __slots__ = ()


class ExposureError(RiskError):
    """Raised when exposure measurement or exposure limits fail."""

    __slots__ = ()


class VaRError(RiskError):
    """Raised when Value-at-Risk calculation fails."""

    __slots__ = ()


class CVaRError(RiskError):
    """Raised when Conditional Value-at-Risk calculation fails."""

    __slots__ = ()


class StressTestError(RiskError):
    """Raised when stress-test evaluation fails."""

    __slots__ = ()


class RiskLimitError(RiskError):
    """Raised when a configured risk limit is breached or cannot be applied."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class ExecutionError(CQROSError):
    """Raised when order execution or execution orchestration fails."""

    __slots__ = ()


class OrderError(ExecutionError):
    """Raised when order construction, validation, or submission fails."""

    __slots__ = ()


class RoutingError(ExecutionError):
    """Raised when order routing fails."""

    __slots__ = ()


class AlgorithmError(ExecutionError):
    """Raised when an execution algorithm fails."""

    __slots__ = ()


class FillError(ExecutionError):
    """Raised when fill processing or fill reconciliation fails."""

    __slots__ = ()

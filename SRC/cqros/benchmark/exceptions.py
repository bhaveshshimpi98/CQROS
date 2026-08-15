"""CQROS Benchmark Engine exception hierarchy.

Purpose:
    Provide benchmark-specific exception types used by providers, pipelines,
    repositories, and verification workflows.

Responsibilities:
    - Define the package ``BenchmarkException`` root under ``ResearchError``
    - Expose ``BenchmarkError`` for input and contract validation
    - Expose canonical benchmark error-code constants
    - Remain free of logging, validation, and business logic

Dependencies:
    ``cqros.core.exceptions.ResearchError``.

Public API:
    The exception types and error-code constants listed in ``__all__``.

Notes:
    ``CQROSError`` already supports ``error_code``, ``details``, and
    ``recovery_suggestion`` on every subclass. Callers should pass those
    keyword arguments when raising for stable programmatic handling.
"""

from __future__ import annotations

from typing import Final

from cqros.core.exceptions import ResearchError

__all__ = [
    "BENCHMARK_BUILD_FAILED",
    "BENCHMARK_CONFIGURATION_ERROR",
    "BENCHMARK_DUPLICATE_PROVIDER",
    "BENCHMARK_INVALID_INPUT",
    "BENCHMARK_INVALID_SCHEMA",
    "BENCHMARK_INVALID_TYPE",
    "BENCHMARK_NOT_FOUND",
    "BENCHMARK_STORAGE_ERROR",
    "BENCHMARK_UNKNOWN_PROVIDER",
    "BenchmarkError",
    "BenchmarkException",
]

BENCHMARK_INVALID_INPUT: Final[str] = "BENCHMARK_INVALID_INPUT"
BENCHMARK_INVALID_SCHEMA: Final[str] = "BENCHMARK_INVALID_SCHEMA"
BENCHMARK_UNKNOWN_PROVIDER: Final[str] = "BENCHMARK_UNKNOWN_PROVIDER"
BENCHMARK_DUPLICATE_PROVIDER: Final[str] = "BENCHMARK_DUPLICATE_PROVIDER"
BENCHMARK_NOT_FOUND: Final[str] = "BENCHMARK_NOT_FOUND"
BENCHMARK_STORAGE_ERROR: Final[str] = "BENCHMARK_STORAGE_ERROR"
BENCHMARK_CONFIGURATION_ERROR: Final[str] = "BENCHMARK_CONFIGURATION_ERROR"
BENCHMARK_INVALID_TYPE: Final[str] = "BENCHMARK_INVALID_TYPE"
BENCHMARK_BUILD_FAILED: Final[str] = "BENCHMARK_BUILD_FAILED"


class BenchmarkException(ResearchError):  # noqa: N818
    """Raised when benchmark workflows or artifacts fail."""

    __slots__ = ()


class BenchmarkError(BenchmarkException):
    """Raised when benchmark inputs, outputs, or contracts fail."""

    __slots__ = ()

"""CQROS adaptive download chunk sizing.

Purpose:
    Derive timeframe-aware historical download chunk durations so each
    planned request stays near the configured exchange kline limit.

Responsibilities:
    - Parse supported bar intervals into millisecond durations
    - Compute inclusive chunk window lengths from timeframe, kline limit,
      and safety factor
    - Expose injectable chunk sizing strategies for ``DownloadPlanner``
    - Remain free of networking, repository, and Binance client logic

Dependencies:
    Python standard library, ``cqros.core.constants``, ``cqros.core.exceptions``,
    and ``cqros.core.types``.

Public API:
    Duration helpers, safety / timeframe constants, and the chunk sizing
    strategies listed in ``__all__``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

from cqros.core.constants import (
    DAYS_PER_WEEK,
    MILLISECONDS_PER_DAY,
    MILLISECONDS_PER_HOUR,
    MILLISECONDS_PER_MINUTE,
    MILLISECONDS_PER_SECOND,
    TIMEFRAME_PATTERN,
)
from cqros.core.exceptions import ValidationError
from cqros.core.types import Timeframe

__all__ = [
    "DEFAULT_CHUNK_SAFETY_FACTOR",
    "DOWNLOAD_TIMEFRAMES",
    "DAYS_PER_MONTH_APPROXIMATION",
    "ChunkSizingStrategy",
    "FixedChunkSizingStrategy",
    "AdaptiveChunkSizingStrategy",
    "timeframe_duration_ms",
    "effective_kline_count",
]

DEFAULT_CHUNK_SAFETY_FACTOR: Final[float] = 0.90

# Binance monthly klines are calendar months; chunk sizing uses a fixed
# 30-day approximation so durations remain deterministic.
DAYS_PER_MONTH_APPROXIMATION: Final[int] = 30

_TIMEFRAME_RE: Final[re.Pattern[str]] = re.compile(f"^{TIMEFRAME_PATTERN}$")

_UNIT_MILLISECONDS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "s": MILLISECONDS_PER_SECOND,
        "m": MILLISECONDS_PER_MINUTE,
        "h": MILLISECONDS_PER_HOUR,
        "d": MILLISECONDS_PER_DAY,
        "w": DAYS_PER_WEEK * MILLISECONDS_PER_DAY,
        "M": DAYS_PER_MONTH_APPROXIMATION * MILLISECONDS_PER_DAY,
    }
)

DOWNLOAD_TIMEFRAMES: Final[frozenset[str]] = frozenset(
    {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    }
)


def timeframe_duration_ms(timeframe: object) -> int:
    """Convert a supported download timeframe into milliseconds.

    Args:
        timeframe: Bar interval identifier (for example ``1h`` or ``15m``).

    Returns:
        Interval duration in Unix milliseconds.

    Raises:
        ValidationError: If ``timeframe`` is missing from the download
            allowlist or cannot be parsed.
    """
    if not isinstance(timeframe, str):
        raise ValidationError(
            "timeframe must be a string bar interval",
            error_code="INGESTION-CHUNK-SIZING-001",
            details={"parameter": "timeframe", "type": type(timeframe).__name__},
        )

    if timeframe not in DOWNLOAD_TIMEFRAMES:
        raise ValidationError(
            f"unsupported download timeframe: {timeframe!r}",
            error_code="INGESTION-CHUNK-SIZING-002",
            details={
                "parameter": "timeframe",
                "value": timeframe,
                "allowed": tuple(sorted(DOWNLOAD_TIMEFRAMES)),
            },
        )

    match = _TIMEFRAME_RE.fullmatch(timeframe)
    if match is None:
        raise ValidationError(
            f"invalid timeframe format: {timeframe!r}",
            error_code="INGESTION-CHUNK-SIZING-003",
            details={"parameter": "timeframe", "value": timeframe},
        )

    amount = int(timeframe[:-1])
    unit = timeframe[-1]
    unit_ms = _UNIT_MILLISECONDS.get(unit)
    if unit_ms is None:
        raise ValidationError(
            f"unsupported timeframe unit: {unit!r}",
            error_code="INGESTION-CHUNK-SIZING-004",
            details={"parameter": "timeframe", "value": timeframe, "unit": unit},
        )
    return amount * unit_ms


def effective_kline_count(*, kline_limit: int, safety_factor: float) -> int:
    """Return the candle budget used when deriving chunk duration.

    Args:
        kline_limit: Maximum klines allowed per exchange request.
        safety_factor: Fraction of ``kline_limit`` retained as headroom.

    Returns:
        ``floor(kline_limit * safety_factor)``.

    Raises:
        ValidationError: If inputs are invalid or the effective count is
            below one candle.
    """
    _require_positive_int(kline_limit, parameter="kline_limit")
    _require_safety_factor(safety_factor)

    effective = math.floor(kline_limit * safety_factor)
    if effective < 1:
        raise ValidationError(
            "effective kline count must be at least 1",
            error_code="INGESTION-CHUNK-SIZING-005",
            details={
                "kline_limit": kline_limit,
                "safety_factor": safety_factor,
                "effective_kline_count": effective,
            },
        )
    return effective


class ChunkSizingStrategy(Protocol):
    """Strategy that resolves inclusive download chunk duration in milliseconds."""

    def chunk_size_ms(self, timeframe: Timeframe) -> int:
        """Return the inclusive chunk window length for ``timeframe``.

        Args:
            timeframe: Bar interval identifier.

        Returns:
            Positive inclusive window length in Unix milliseconds.
        """
        ...


@dataclass(frozen=True, slots=True)
class FixedChunkSizingStrategy:
    """Always return a caller-configured chunk duration.

    Attributes:
        size_ms: Inclusive window length in Unix milliseconds.
    """

    size_ms: int

    def __post_init__(self) -> None:
        """Validate fixed chunk configuration."""
        _require_positive_int(self.size_ms, parameter="size_ms")

    def chunk_size_ms(self, timeframe: Timeframe) -> int:
        """Return the configured fixed chunk size.

        Args:
            timeframe: Unused; retained for strategy interface parity.

        Returns:
            Configured inclusive window length in milliseconds.
        """
        del timeframe
        return self.size_ms


@dataclass(frozen=True, slots=True)
class AdaptiveChunkSizingStrategy:
    """Derive chunk duration from timeframe duration and kline capacity.

    Chunk length is:

    ``floor(kline_limit * safety_factor) * timeframe_duration_ms``

    so each planned window stays below the exchange kline limit.

    Attributes:
        kline_limit: Maximum klines allowed per exchange request.
        safety_factor: Fraction of ``kline_limit`` retained as headroom.
    """

    kline_limit: int
    safety_factor: float = DEFAULT_CHUNK_SAFETY_FACTOR

    def __post_init__(self) -> None:
        """Validate adaptive chunk configuration."""
        # Trigger shared validation eagerly at construction time.
        effective_kline_count(
            kline_limit=self.kline_limit,
            safety_factor=self.safety_factor,
        )

    def chunk_size_ms(self, timeframe: Timeframe) -> int:
        """Return timeframe-aware inclusive chunk duration.

        Args:
            timeframe: Bar interval identifier.

        Returns:
            Inclusive window length in Unix milliseconds.
        """
        candles = effective_kline_count(
            kline_limit=self.kline_limit,
            safety_factor=self.safety_factor,
        )
        return candles * timeframe_duration_ms(timeframe)


def _require_positive_int(value: object, *, parameter: str) -> int:
    """Require a strictly positive integer configuration value."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(
            f"{parameter} must be an int greater than 0",
            error_code="INGESTION-CHUNK-SIZING-006",
            details={"parameter": parameter, "value": value},
        )
    return value


def _require_safety_factor(value: object) -> float:
    """Require a safety factor in the open-closed interval ``(0, 1]``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            "safety_factor must be a number in the interval (0, 1]",
            error_code="INGESTION-CHUNK-SIZING-007",
            details={"parameter": "safety_factor", "type": type(value).__name__},
        )
    factor = float(value)
    if not 0.0 < factor <= 1.0:
        raise ValidationError(
            "safety_factor must be a number in the interval (0, 1]",
            error_code="INGESTION-CHUNK-SIZING-007",
            details={"parameter": "safety_factor", "value": factor},
        )
    return factor

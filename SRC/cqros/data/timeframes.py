"""CQROS canonical timeframe definitions and helpers.

Purpose:
    Provide a single, shared timeframe system used across downloading,
    storage, feature engineering, research, backtesting, and live
    execution.

Responsibilities:
    - Define the closed ``Timeframe`` enumeration of supported intervals
    - Define ``TimeframeInfo`` as the immutable canonical definition for
      each timeframe
    - Expose ordered timeframe collections and lookup helpers
    - Remain free of business logic, validation pipelines, and I/O

Dependencies:
    Python standard library and ``cqros.core.constants``.

Public API:
    The enumerations, dataclasses, collections, and helpers listed in
    ``__all__``.

Notes:
    Duration comparisons and conversions always read from
    ``TIMEFRAME_INFO``. Every supported timeframe has exactly one
    ``TimeframeInfo`` entry. Intraday means strictly shorter than one
    day; higher timeframes are one day or longer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from cqros.core.constants import (
    DAYS_PER_WEEK,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
)

__all__ = [
    "Timeframe",
    "TimeframeInfo",
    "TIMEFRAME_INFO",
    "ALL_TIMEFRAMES",
    "INTRADAY_TIMEFRAMES",
    "HIGHER_TIMEFRAMES",
    "to_minutes",
    "to_seconds",
    "display_name",
    "is_intraday",
    "is_higher_timeframe",
    "is_lower_than",
    "is_higher_than",
]


class Timeframe(StrEnum):
    """Canonical bar / candle timeframe identifiers.

    Values match the project timeframe string allowlist (for example
    ``1m``, ``1h``, ``1d``).

    Attributes:
        S1: One-second interval.
        M1: One-minute interval.
        M5: Five-minute interval.
        M15: Fifteen-minute interval.
        M30: Thirty-minute interval.
        H1: One-hour interval.
        H4: Four-hour interval.
        D1: One-day interval.
        W1: One-week interval.
    """

    S1 = "1s"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass(frozen=True, slots=True)
class TimeframeInfo:
    """Immutable canonical definition for a single timeframe.

    Attributes:
        timeframe: Canonical timeframe identifier.
        seconds: Interval duration in whole seconds.
        minutes: Interval duration in minutes (fractional when needed).
        display_name: Human-readable label for reporting and UI.
        is_intraday: Whether the interval is strictly shorter than one day.
    """

    timeframe: Timeframe
    seconds: int
    minutes: float
    display_name: str
    is_intraday: bool


def _info(
    timeframe: Timeframe,
    seconds: int,
    display_name: str,
) -> TimeframeInfo:
    """Build a ``TimeframeInfo`` from seconds and display metadata."""
    return TimeframeInfo(
        timeframe=timeframe,
        seconds=seconds,
        minutes=seconds / SECONDS_PER_MINUTE,
        display_name=display_name,
        is_intraday=seconds < SECONDS_PER_DAY,
    )


TIMEFRAME_INFO: Final[Mapping[Timeframe, TimeframeInfo]] = MappingProxyType(
    {
        Timeframe.S1: _info(Timeframe.S1, 1, "1 Second"),
        Timeframe.M1: _info(Timeframe.M1, SECONDS_PER_MINUTE, "1 Minute"),
        Timeframe.M5: _info(Timeframe.M5, 5 * SECONDS_PER_MINUTE, "5 Minutes"),
        Timeframe.M15: _info(Timeframe.M15, 15 * SECONDS_PER_MINUTE, "15 Minutes"),
        Timeframe.M30: _info(Timeframe.M30, 30 * SECONDS_PER_MINUTE, "30 Minutes"),
        Timeframe.H1: _info(Timeframe.H1, SECONDS_PER_HOUR, "1 Hour"),
        Timeframe.H4: _info(
            Timeframe.H4,
            4 * SECONDS_PER_HOUR,
            "4 Hours",
        ),
        Timeframe.D1: _info(Timeframe.D1, SECONDS_PER_DAY, "1 Day"),
        Timeframe.W1: _info(
            Timeframe.W1,
            DAYS_PER_WEEK * SECONDS_PER_DAY,
            "1 Week",
        ),
    }
)

ALL_TIMEFRAMES: Final[tuple[Timeframe, ...]] = (
    Timeframe.S1,
    Timeframe.M1,
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.M30,
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.D1,
    Timeframe.W1,
)

INTRADAY_TIMEFRAMES: Final[tuple[Timeframe, ...]] = tuple(
    timeframe for timeframe in ALL_TIMEFRAMES if TIMEFRAME_INFO[timeframe].is_intraday
)

HIGHER_TIMEFRAMES: Final[tuple[Timeframe, ...]] = tuple(
    timeframe for timeframe in ALL_TIMEFRAMES if not TIMEFRAME_INFO[timeframe].is_intraday
)


def to_minutes(timeframe: Timeframe) -> float:
    """Return the interval duration in minutes.

    Args:
        timeframe: Canonical timeframe identifier.

    Returns:
        Duration in minutes. Sub-minute intervals return a fractional value.
    """
    return TIMEFRAME_INFO[timeframe].minutes


def to_seconds(timeframe: Timeframe) -> int:
    """Return the interval duration in whole seconds.

    Args:
        timeframe: Canonical timeframe identifier.

    Returns:
        Duration in seconds.
    """
    return TIMEFRAME_INFO[timeframe].seconds


def display_name(timeframe: Timeframe) -> str:
    """Return the human-readable label for a timeframe.

    Args:
        timeframe: Canonical timeframe identifier.

    Returns:
        Display name from the canonical timeframe definition.
    """
    return TIMEFRAME_INFO[timeframe].display_name


def is_intraday(timeframe: Timeframe) -> bool:
    """Return whether a timeframe is strictly shorter than one day.

    Args:
        timeframe: Canonical timeframe identifier.

    Returns:
        ``True`` when the interval is intraday; otherwise ``False``.
    """
    return TIMEFRAME_INFO[timeframe].is_intraday


def is_higher_timeframe(timeframe: Timeframe) -> bool:
    """Return whether a timeframe is one day or longer.

    Args:
        timeframe: Canonical timeframe identifier.

    Returns:
        ``True`` when the interval is a higher timeframe; otherwise ``False``.
    """
    return not TIMEFRAME_INFO[timeframe].is_intraday


def is_lower_than(left: Timeframe, right: Timeframe) -> bool:
    """Return whether ``left`` has a shorter duration than ``right``.

    Args:
        left: Candidate lower timeframe.
        right: Reference timeframe.

    Returns:
        ``True`` when ``left`` is strictly shorter than ``right``.
    """
    return TIMEFRAME_INFO[left].seconds < TIMEFRAME_INFO[right].seconds


def is_higher_than(left: Timeframe, right: Timeframe) -> bool:
    """Return whether ``left`` has a longer duration than ``right``.

    Args:
        left: Candidate higher timeframe.
        right: Reference timeframe.

    Returns:
        ``True`` when ``left`` is strictly longer than ``right``.
    """
    return TIMEFRAME_INFO[left].seconds > TIMEFRAME_INFO[right].seconds

"""Unit tests for CQROS canonical timeframe definitions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from enum import StrEnum
from types import MappingProxyType

import pytest

from cqros.core import constants
from cqros.data.timeframes import (
    ALL_TIMEFRAMES,
    HIGHER_TIMEFRAMES,
    INTRADAY_TIMEFRAMES,
    TIMEFRAME_INFO,
    Timeframe,
    TimeframeInfo,
    display_name,
    is_higher_than,
    is_higher_timeframe,
    is_intraday,
    is_lower_than,
    to_minutes,
    to_seconds,
)

_EXPECTED_ENUM_VALUES: dict[str, str] = {
    "S1": "1s",
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d",
    "W1": "1w",
}

_EXPECTED_SECONDS: dict[Timeframe, int] = {
    Timeframe.S1: 1,
    Timeframe.M1: constants.SECONDS_PER_MINUTE,
    Timeframe.M5: 5 * constants.SECONDS_PER_MINUTE,
    Timeframe.M15: 15 * constants.SECONDS_PER_MINUTE,
    Timeframe.M30: 30 * constants.SECONDS_PER_MINUTE,
    Timeframe.H1: constants.SECONDS_PER_HOUR,
    Timeframe.H4: 4 * constants.SECONDS_PER_HOUR,
    Timeframe.D1: constants.SECONDS_PER_DAY,
    Timeframe.W1: constants.DAYS_PER_WEEK * constants.SECONDS_PER_DAY,
}

_EXPECTED_DISPLAY_NAMES: dict[Timeframe, str] = {
    Timeframe.S1: "1 Second",
    Timeframe.M1: "1 Minute",
    Timeframe.M5: "5 Minutes",
    Timeframe.M15: "15 Minutes",
    Timeframe.M30: "30 Minutes",
    Timeframe.H1: "1 Hour",
    Timeframe.H4: "4 Hours",
    Timeframe.D1: "1 Day",
    Timeframe.W1: "1 Week",
}


def test_timeframe_is_str_enum() -> None:
    """Timeframe is a serializable string enumeration."""
    assert issubclass(Timeframe, StrEnum)
    for member in Timeframe:
        assert isinstance(member.value, str)
        assert str(member) == member.value


def test_timeframe_values_match_supported_constants() -> None:
    """Enum values align with the project supported timeframe allowlist."""
    assert {member.name: member.value for member in Timeframe} == _EXPECTED_ENUM_VALUES
    assert {member.value for member in Timeframe} == set(constants.SUPPORTED_TIMEFRAMES)


def test_timeframe_info_is_frozen_dataclass() -> None:
    """TimeframeInfo is an immutable slotted dataclass."""
    assert is_dataclass(TimeframeInfo)
    info = TIMEFRAME_INFO[Timeframe.M1]
    assert hasattr(type(info), "__slots__")
    with pytest.raises(FrozenInstanceError):
        info.seconds = 0  # type: ignore[misc]


def test_timeframe_info_mapping_is_complete_and_immutable() -> None:
    """Every timeframe has exactly one mapping entry and the map is read-only."""
    assert isinstance(TIMEFRAME_INFO, MappingProxyType)
    assert set(TIMEFRAME_INFO) == set(Timeframe)
    assert len(TIMEFRAME_INFO) == len(Timeframe)
    with pytest.raises(TypeError):
        TIMEFRAME_INFO[Timeframe.M1] = TIMEFRAME_INFO[Timeframe.M1]  # type: ignore[index]


def test_all_timeframes_cover_enum_in_ascending_duration_order() -> None:
    """ALL_TIMEFRAMES lists every member ordered by increasing duration."""
    assert ALL_TIMEFRAMES == tuple(Timeframe)
    assert set(ALL_TIMEFRAMES) == set(Timeframe)
    seconds = [TIMEFRAME_INFO[timeframe].seconds for timeframe in ALL_TIMEFRAMES]
    assert seconds == sorted(seconds)


def test_intraday_and_higher_partition_all_timeframes() -> None:
    """Intraday and higher collections partition ALL_TIMEFRAMES without overlap."""
    assert set(INTRADAY_TIMEFRAMES) | set(HIGHER_TIMEFRAMES) == set(ALL_TIMEFRAMES)
    assert set(INTRADAY_TIMEFRAMES).isdisjoint(HIGHER_TIMEFRAMES)
    assert HIGHER_TIMEFRAMES == (Timeframe.D1, Timeframe.W1)
    assert Timeframe.H4 in INTRADAY_TIMEFRAMES
    assert Timeframe.S1 in INTRADAY_TIMEFRAMES


@pytest.mark.parametrize("timeframe", list(Timeframe))
def test_canonical_definition_fields(timeframe: Timeframe) -> None:
    """Each TimeframeInfo stores a consistent single canonical definition."""
    info = TIMEFRAME_INFO[timeframe]
    assert info.timeframe is timeframe
    assert info.seconds == _EXPECTED_SECONDS[timeframe]
    assert info.minutes == pytest.approx(info.seconds / constants.SECONDS_PER_MINUTE)
    assert info.display_name == _EXPECTED_DISPLAY_NAMES[timeframe]
    assert info.is_intraday is (info.seconds < constants.SECONDS_PER_DAY)


@pytest.mark.parametrize("timeframe", list(Timeframe))
def test_helper_functions_read_canonical_definition(timeframe: Timeframe) -> None:
    """Helpers expose fields from TIMEFRAME_INFO without independent logic."""
    info = TIMEFRAME_INFO[timeframe]
    assert to_seconds(timeframe) == info.seconds
    assert to_minutes(timeframe) == info.minutes
    assert display_name(timeframe) == info.display_name
    assert is_intraday(timeframe) is info.is_intraday
    assert is_higher_timeframe(timeframe) is (not info.is_intraday)


def test_duration_comparisons() -> None:
    """Lower/higher comparisons use strict duration ordering."""
    assert is_lower_than(Timeframe.M1, Timeframe.H1)
    assert is_higher_than(Timeframe.H1, Timeframe.M1)
    assert not is_lower_than(Timeframe.H1, Timeframe.M1)
    assert not is_higher_than(Timeframe.M1, Timeframe.H1)
    assert not is_lower_than(Timeframe.H1, Timeframe.H1)
    assert not is_higher_than(Timeframe.H1, Timeframe.H1)
    assert is_lower_than(Timeframe.S1, Timeframe.W1)
    assert is_higher_than(Timeframe.W1, Timeframe.D1)


def test_one_second_minutes_are_fractional() -> None:
    """Sub-minute intervals expose fractional minute durations."""
    assert to_minutes(Timeframe.S1) == pytest.approx(1 / constants.SECONDS_PER_MINUTE)
    assert to_seconds(Timeframe.S1) == 1


def test_timeframe_info_field_names() -> None:
    """TimeframeInfo exposes the documented public fields."""
    names = {field.name for field in fields(TimeframeInfo)}
    assert names == {
        "timeframe",
        "seconds",
        "minutes",
        "display_name",
        "is_intraday",
    }

"""Unit tests for the CQROS Signal enumeration."""

from __future__ import annotations

from enum import Enum

import pytest

from cqros.signals import Signal, signals, values
from cqros.signals.enums import Signal as SignalDirect
from cqros.signals.enums import signals as signals_direct
from cqros.signals.enums import values as values_direct


def test_signal_is_exported_from_package() -> None:
    """Package exports match the enums module by identity."""
    assert Signal is SignalDirect
    assert signals is signals_direct
    assert values is values_direct


def test_buy_member() -> None:
    """BUY member name and value are stable."""
    assert Signal.BUY.name == "BUY"
    assert Signal.BUY.value == "BUY"
    assert Signal.BUY == "BUY"


def test_sell_member() -> None:
    """SELL member name and value are stable."""
    assert Signal.SELL.name == "SELL"
    assert Signal.SELL.value == "SELL"
    assert Signal.SELL == "SELL"


def test_hold_member() -> None:
    """HOLD member name and value are stable."""
    assert Signal.HOLD.name == "HOLD"
    assert Signal.HOLD.value == "HOLD"
    assert Signal.HOLD == "HOLD"


def test_enum_names() -> None:
    """Member names remain the canonical uppercase identifiers."""
    assert tuple(member.name for member in Signal) == ("BUY", "SELL", "HOLD")


def test_enum_values() -> None:
    """Member values remain the canonical uppercase strings."""
    assert tuple(member.value for member in Signal) == ("BUY", "SELL", "HOLD")


def test_signal_subclasses_str_and_enum() -> None:
    """Signal subclasses both str and Enum for natural serialization."""
    assert issubclass(Signal, str)
    assert issubclass(Signal, Enum)
    for member in Signal:
        assert isinstance(member, str)
        assert isinstance(member, Signal)
        assert member == member.value


def test_signals_helper_output() -> None:
    """signals returns every member in declaration order."""
    assert signals() == (Signal.BUY, Signal.SELL, Signal.HOLD)


def test_values_helper_output() -> None:
    """values returns every string value in declaration order."""
    assert values() == ("BUY", "SELL", "HOLD")


def test_helper_outputs_are_immutable_tuples() -> None:
    """Helpers return immutable tuples."""
    signal_members = signals()
    signal_values = values()

    assert isinstance(signal_members, tuple)
    assert isinstance(signal_values, tuple)

    with pytest.raises(TypeError):
        signal_members[0] = Signal.HOLD  # type: ignore[index]

    with pytest.raises(TypeError):
        signal_values[0] = "HOLD"  # type: ignore[index]


def test_helper_independence() -> None:
    """Helpers return independent copies, not shared mutable state."""
    first_signals = signals()
    second_signals = signals()
    first_values = values()
    second_values = values()

    assert first_signals == second_signals
    assert first_signals is not second_signals
    assert first_values == second_values
    assert first_values is not second_values


def test_signal_members_and_values_are_unique() -> None:
    """Signal names and values contain no duplicates."""
    names = tuple(member.name for member in Signal)
    member_values = tuple(member.value for member in Signal)

    assert len(names) == len(set(names))
    assert len(member_values) == len(set(member_values))
    assert len(signals()) == len(set(signals()))
    assert len(values()) == len(set(values()))


def test_signal_round_trips_from_value() -> None:
    """Signal members can be reconstructed from their string values."""
    for member in Signal:
        assert Signal(member.value) is member


def test_invalid_value_raises_value_error() -> None:
    """Unknown serialized values raise ValueError."""
    with pytest.raises(ValueError):
        Signal("not_a_valid_signal")

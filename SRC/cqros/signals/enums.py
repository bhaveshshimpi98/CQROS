"""CQROS Signal enumeration.

Purpose:
    Define the canonical discrete trading-signal vocabulary used throughout
    CQROS.

Responsibilities:
    - Enumerate every supported signal as a string-backed enumeration
    - Expose helper accessors that return immutable member and value tuples
    - Remain free of validation, persistence, pipeline, and trading logic

Dependencies:
    Python standard library only (``enum.Enum``).

Public API:
    ``Signal``, ``signals``, ``values``

Notes:
    ``Signal`` subclasses both ``str`` and ``Enum`` so members serialize
    naturally into Polars DataFrames without conversion.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Signal",
    "signals",
    "values",
]


# Prefer str+Enum over StrEnum so Signal values embed directly into Polars.
class Signal(str, Enum):  # noqa: UP042
    """Canonical discrete trading signal.

    Attributes:
        BUY: Recommend acquiring or increasing long exposure.
        SELL: Recommend disposing or increasing short exposure.
        HOLD: Recommend no change in exposure.
    """

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


def signals() -> tuple[Signal, ...]:
    """Return an immutable copy of every ``Signal`` member.

    Returns:
        All signal members in declaration order.
    """
    return (Signal.BUY, Signal.SELL, Signal.HOLD)


def values() -> tuple[str, ...]:
    """Return an immutable copy of every ``Signal`` string value.

    Returns:
        All signal values in declaration order.
    """
    return (Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value)

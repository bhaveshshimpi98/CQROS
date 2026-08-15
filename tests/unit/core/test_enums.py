"""Unit tests for CQROS shared enumerations."""

from __future__ import annotations

from enum import StrEnum

import pytest

from cqros.core.enums import (
    DatasetSplit,
    DataSource,
    MarketType,
    ModelType,
    OrderSide,
    OrderStatus,
    OrderType,
    PipelineStage,
    PositionSide,
    RiskLevel,
    SignalSide,
    TimeInForce,
    TradeStatus,
)

_ENUM_TYPES: tuple[type[StrEnum], ...] = (
    MarketType,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    PositionSide,
    SignalSide,
    RiskLevel,
    ModelType,
    TradeStatus,
    DataSource,
    DatasetSplit,
    PipelineStage,
)

_EXPECTED_VALUES: dict[type[StrEnum], dict[str, str]] = {
    MarketType: {
        "SPOT": "spot",
        "PERPETUAL": "perpetual",
        "FUTURES": "futures",
        "OPTIONS": "options",
    },
    OrderSide: {
        "BUY": "buy",
        "SELL": "sell",
    },
    OrderType: {
        "MARKET": "market",
        "LIMIT": "limit",
        "STOP_MARKET": "stop_market",
        "STOP_LIMIT": "stop_limit",
        "TAKE_PROFIT": "take_profit",
        "TAKE_PROFIT_LIMIT": "take_profit_limit",
        "TRAILING_STOP": "trailing_stop",
    },
    OrderStatus: {
        "CREATED": "created",
        "VALIDATED": "validated",
        "SUBMITTED": "submitted",
        "ACCEPTED": "accepted",
        "PARTIALLY_FILLED": "partially_filled",
        "FILLED": "filled",
        "CANCELLED": "cancelled",
        "REJECTED": "rejected",
        "EXPIRED": "expired",
        "ARCHIVED": "archived",
    },
    TimeInForce: {
        "GTC": "gtc",
        "IOC": "ioc",
        "FOK": "fok",
    },
    PositionSide: {
        "LONG": "long",
        "SHORT": "short",
        "FLAT": "flat",
    },
    SignalSide: {
        "LONG": "long",
        "SHORT": "short",
        "FLAT": "flat",
    },
    RiskLevel: {
        "LOW": "low",
        "MEDIUM": "medium",
        "HIGH": "high",
        "CRITICAL": "critical",
    },
    ModelType: {
        "STATISTICAL": "statistical",
        "LINEAR": "linear",
        "TREE": "tree",
        "GRADIENT_BOOSTING": "gradient_boosting",
        "DEEP_LEARNING": "deep_learning",
        "TIME_SERIES": "time_series",
        "PROBABILISTIC": "probabilistic",
        "ENSEMBLE": "ensemble",
        "META": "meta",
    },
    TradeStatus: {
        "PENDING": "pending",
        "OPEN": "open",
        "PARTIALLY_CLOSED": "partially_closed",
        "CLOSED": "closed",
        "CANCELLED": "cancelled",
        "FAILED": "failed",
    },
    DataSource: {
        "EXCHANGE_REST": "exchange_rest",
        "EXCHANGE_WEBSOCKET": "exchange_websocket",
        "HISTORICAL_ARCHIVE": "historical_archive",
        "THIRD_PARTY": "third_party",
        "REPLAY": "replay",
        "SIMULATION": "simulation",
    },
    DatasetSplit: {
        "TRAIN": "train",
        "VALIDATION": "validation",
        "TEST": "test",
    },
    PipelineStage: {
        "INGESTION": "ingestion",
        "STORAGE": "storage",
        "VALIDATION": "validation",
        "METADATA": "metadata",
        "DATASET": "dataset",
        "FEATURES": "features",
        "TARGETS": "targets",
        "STATISTICS": "statistics",
        "REGIME": "regime",
        "TRAINING": "training",
        "EVALUATION": "evaluation",
        "ALPHA": "alpha",
        "PORTFOLIO": "portfolio",
        "RISK": "risk",
        "EXECUTION": "execution",
        "BACKTESTING": "backtesting",
        "MONITORING": "monitoring",
        "DEPLOYMENT": "deployment",
    },
}


@pytest.mark.parametrize("enum_type", _ENUM_TYPES)
def test_enum_is_str_enum(enum_type: type[StrEnum]) -> None:
    """Every shared enumeration subclasses StrEnum."""
    assert issubclass(enum_type, StrEnum)


@pytest.mark.parametrize("enum_type", _ENUM_TYPES)
def test_enum_members_match_expected_values(enum_type: type[StrEnum]) -> None:
    """Member names and string values remain stable for serialization."""
    expected = _EXPECTED_VALUES[enum_type]
    actual = {member.name: member.value for member in enum_type}

    assert actual == expected


@pytest.mark.parametrize("enum_type", _ENUM_TYPES)
def test_enum_members_are_strings(enum_type: type[StrEnum]) -> None:
    """StrEnum members behave as strings for configuration and persistence."""
    for member in enum_type:
        assert isinstance(member, str)
        assert member == member.value


@pytest.mark.parametrize("enum_type", _ENUM_TYPES)
def test_enum_round_trips_from_value(enum_type: type[StrEnum]) -> None:
    """Enumerations can be reconstructed from their serialized values."""
    for member in enum_type:
        assert enum_type(member.value) is member


@pytest.mark.parametrize("enum_type", _ENUM_TYPES)
def test_invalid_value_raises_value_error(enum_type: type[StrEnum]) -> None:
    """Unknown serialized values raise ValueError."""
    with pytest.raises(ValueError):
        enum_type("not_a_valid_member")

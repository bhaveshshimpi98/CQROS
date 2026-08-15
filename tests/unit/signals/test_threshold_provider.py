"""Unit tests for CQROS ``InMemoryThresholdProvider``."""

from __future__ import annotations

import pytest

from cqros.signals import (
    InMemoryThresholdProvider,
    RegressionThresholds,
    SignalValidationError,
    ThresholdProvider,
)
from cqros.signals.threshold_provider import (
    InMemoryThresholdProvider as InMemoryThresholdProviderDirect,
)

_GLOBAL = RegressionThresholds(buy_threshold=0.01, sell_threshold=-0.01)
_SYMBOL = RegressionThresholds(buy_threshold=0.02, sell_threshold=-0.02)
_SYMBOL_TF = RegressionThresholds(buy_threshold=0.03, sell_threshold=-0.03)
_MODEL = RegressionThresholds(buy_threshold=0.04, sell_threshold=-0.04)
_PARTITION = RegressionThresholds(buy_threshold=0.05, sell_threshold=-0.05)


def _provider(**kwargs: object) -> InMemoryThresholdProvider:
    """Build an in-memory provider with shared global defaults."""
    return InMemoryThresholdProvider(global_thresholds=_GLOBAL, **kwargs)  # type: ignore[arg-type]


def test_exported_from_package() -> None:
    """Package export matches the threshold_provider module by identity."""
    assert InMemoryThresholdProvider is InMemoryThresholdProviderDirect


def test_satisfies_threshold_provider_protocol() -> None:
    """InMemoryThresholdProvider structurally satisfies ThresholdProvider."""
    assert isinstance(_provider(), ThresholdProvider)


def test_global_thresholds() -> None:
    """Absent overrides return configured global thresholds."""
    result = _provider().get_thresholds("BTCUSDT", "1h", "alpha", "1.0.0")
    assert result == _GLOBAL
    assert result.buy_threshold == 0.01
    assert result.sell_threshold == -0.01


def test_symbol_overrides() -> None:
    """Symbol overrides beat global defaults."""
    provider = _provider(symbol_overrides={"BTCUSDT": _SYMBOL})
    assert provider.get_thresholds("BTCUSDT", "1h", "alpha", "1.0.0") == _SYMBOL
    assert provider.get_thresholds("ETHUSDT", "1h", "alpha", "1.0.0") == _GLOBAL


def test_symbol_timeframe_overrides() -> None:
    """Symbol+timeframe overrides beat symbol and global defaults."""
    provider = _provider(
        symbol_overrides={"BTCUSDT": _SYMBOL},
        symbol_timeframe_overrides={("BTCUSDT", "1h"): _SYMBOL_TF},
    )
    assert provider.get_thresholds("BTCUSDT", "1h", "alpha", "1.0.0") == _SYMBOL_TF
    assert provider.get_thresholds("BTCUSDT", "4h", "alpha", "1.0.0") == _SYMBOL


def test_model_overrides() -> None:
    """Model+version overrides beat global defaults when no symbol match."""
    provider = _provider(model_overrides={("alpha", "1.0.0"): _MODEL})
    assert provider.get_thresholds("BTCUSDT", "1h", "alpha", "1.0.0") == _MODEL
    assert provider.get_thresholds("BTCUSDT", "1h", "beta", "1.0.0") == _GLOBAL


def test_partition_overrides() -> None:
    """Full partition overrides beat every broader override tier."""
    provider = _provider(
        symbol_overrides={"BTCUSDT": _SYMBOL},
        symbol_timeframe_overrides={("BTCUSDT", "1h"): _SYMBOL_TF},
        model_overrides={("alpha", "1.0.0"): _MODEL},
        partition_overrides={("BTCUSDT", "1h", "alpha", "1.0.0"): _PARTITION},
    )
    assert provider.get_thresholds("BTCUSDT", "1h", "alpha", "1.0.0") == _PARTITION


def test_resolution_precedence() -> None:
    """Resolution order matches the documented override ladder."""
    provider = _provider(
        symbol_overrides={"BTCUSDT": _SYMBOL},
        symbol_timeframe_overrides={("BTCUSDT", "1h"): _SYMBOL_TF},
        model_overrides={("alpha", "1.0.0"): _MODEL},
        partition_overrides={("ETHUSDT", "4h", "beta", "2.0.0"): _PARTITION},
    )

    assert provider.get_thresholds("ETHUSDT", "4h", "beta", "2.0.0") == _PARTITION
    assert provider.get_thresholds("BTCUSDT", "1h", "alpha", "1.0.0") == _SYMBOL_TF
    assert provider.get_thresholds("BTCUSDT", "4h", "alpha", "1.0.0") == _SYMBOL
    assert provider.get_thresholds("SOLUSDT", "1h", "alpha", "1.0.0") == _MODEL
    assert provider.get_thresholds("SOLUSDT", "1h", "gamma", "3.0.0") == _GLOBAL


def test_symbol_beats_model_override() -> None:
    """Symbol override takes precedence over model+version override."""
    provider = _provider(
        symbol_overrides={"BTCUSDT": _SYMBOL},
        model_overrides={("alpha", "1.0.0"): _MODEL},
    )
    assert provider.get_thresholds("BTCUSDT", "1h", "alpha", "1.0.0") == _SYMBOL


def test_get_thresholds_rejects_blank_keys() -> None:
    """Blank lookup keys raise SignalValidationError."""
    provider = _provider()
    with pytest.raises(SignalValidationError) as exc_info:
        provider.get_thresholds("  ", "1h", "alpha", "1.0.0")
    assert exc_info.value.error_code == "SIGNAL-THR-001"


def test_constructor_rejects_non_thresholds() -> None:
    """Non-RegressionThresholds global defaults raise SignalValidationError."""
    with pytest.raises(SignalValidationError) as exc_info:
        InMemoryThresholdProvider(global_thresholds=(0.01, -0.01))  # type: ignore[arg-type]
    assert exc_info.value.error_code == "SIGNAL-THR-002"


def test_constructor_rejects_blank_override_keys() -> None:
    """Blank override mapping keys raise SignalValidationError."""
    with pytest.raises(SignalValidationError) as exc_info:
        InMemoryThresholdProvider(
            global_thresholds=_GLOBAL,
            symbol_overrides={"": _SYMBOL},
        )
    assert exc_info.value.error_code == "SIGNAL-THR-001"


def test_get_thresholds_is_read_only() -> None:
    """Repeated lookups return the same immutable thresholds object."""
    provider = _provider(symbol_overrides={"BTCUSDT": _SYMBOL})
    first = provider.get_thresholds("BTCUSDT", "1h", "alpha", "1.0.0")
    second = provider.get_thresholds("BTCUSDT", "1h", "alpha", "1.0.0")
    assert first is second
    assert first is _SYMBOL

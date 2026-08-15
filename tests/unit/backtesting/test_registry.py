"""Unit tests for CQROS ``BacktestingRegistry``."""

from __future__ import annotations

import pytest

from cqros.backtesting import (
    BacktestingRegistry,
    BacktestingValidationError,
    SimpleBacktestingEngine,
)


def _registry_with(*names: str) -> BacktestingRegistry:
    """Build a registry pre-populated with SimpleBacktestingEngine under each name."""
    registry = BacktestingRegistry()
    for name in names:
        registry.register(name, SimpleBacktestingEngine())
    return registry


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_empty_registry_reports_no_engines() -> None:
    """A freshly created registry lists no engine names."""
    registry = BacktestingRegistry()
    assert registry.list() == ()


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_and_get_single_engine() -> None:
    """Registering an engine and retrieving it returns the same instance."""
    registry = BacktestingRegistry()
    engine = SimpleBacktestingEngine()
    registry.register("simple", engine)
    assert registry.get("simple") is engine


def test_register_multiple_engines_preserves_insertion_order() -> None:
    """list() returns engine names in the order they were registered."""
    registry = _registry_with("alpha", "beta", "gamma")
    assert registry.list() == ("alpha", "beta", "gamma")


def test_register_rejects_blank_name() -> None:
    """Blank or whitespace-only names raise BT_REG_NAME_BLANK."""
    registry = BacktestingRegistry()
    for blank in ("", "   ", "\t"):
        with pytest.raises(BacktestingValidationError) as exc_info:
            registry.register(blank, SimpleBacktestingEngine())
        assert exc_info.value.error_code == "BT_REG_NAME_BLANK"


def test_register_rejects_non_engine_object() -> None:
    """Objects that do not implement BacktestingEngine raise BT_REG_NOT_ENGINE."""
    registry = BacktestingRegistry()
    with pytest.raises(BacktestingValidationError) as exc_info:
        registry.register("bad", "not-an-engine")  # type: ignore[arg-type]
    assert exc_info.value.error_code == "BT_REG_NOT_ENGINE"


def test_register_rejects_duplicate_name() -> None:
    """Re-registering an existing name raises BT_REG_DUPLICATE."""
    registry = _registry_with("simple")
    with pytest.raises(BacktestingValidationError) as exc_info:
        registry.register("simple", SimpleBacktestingEngine())
    assert exc_info.value.error_code == "BT_REG_DUPLICATE"


# ---------------------------------------------------------------------------
# register_many
# ---------------------------------------------------------------------------


def test_register_many_adds_all_engines() -> None:
    """register_many registers every engine in the mapping atomically."""
    registry = BacktestingRegistry()
    engines = {
        "engine-a": SimpleBacktestingEngine(),
        "engine-b": SimpleBacktestingEngine(),
    }
    registry.register_many(engines)
    assert registry.exists("engine-a")
    assert registry.exists("engine-b")
    assert registry.list() == ("engine-a", "engine-b")


def test_register_many_rejects_duplicate_within_mapping() -> None:
    """register_many raises when items() yields the same name twice."""

    class _DuplicateMapping:
        """Custom mapping whose items() iterator yields the same name twice."""

        def items(self) -> list[tuple[str, SimpleBacktestingEngine]]:
            return [("dup", SimpleBacktestingEngine()), ("dup", SimpleBacktestingEngine())]

    registry = BacktestingRegistry()
    with pytest.raises(BacktestingValidationError) as exc_info:
        registry.register_many(_DuplicateMapping())  # type: ignore[arg-type]
    assert exc_info.value.error_code == "BT_REG_DUPLICATE"
    assert registry.list() == ()


def test_register_many_rejects_already_registered_name() -> None:
    """register_many rejects a name already present in the registry."""
    registry = _registry_with("existing")
    with pytest.raises(BacktestingValidationError) as exc_info:
        registry.register_many({"existing": SimpleBacktestingEngine()})
    assert exc_info.value.error_code == "BT_REG_DUPLICATE"
    assert registry.list() == ("existing",)


def test_register_many_is_atomic_on_bad_engine() -> None:
    """register_many rolls back when any entry fails engine validation."""
    registry = BacktestingRegistry()
    with pytest.raises(BacktestingValidationError):
        registry.register_many(
            {
                "good": SimpleBacktestingEngine(),
                "bad": "not-an-engine",  # type: ignore[dict-value]
            }
        )
    assert registry.list() == ()


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def test_get_returns_registered_engine() -> None:
    """get() returns the exact instance registered under the name."""
    engine = SimpleBacktestingEngine()
    registry = BacktestingRegistry()
    registry.register("simple", engine)
    assert registry.get("simple") is engine


def test_get_rejects_unknown_name() -> None:
    """get() raises BT_REG_UNKNOWN for unregistered names."""
    registry = BacktestingRegistry()
    with pytest.raises(BacktestingValidationError) as exc_info:
        registry.get("does-not-exist")
    assert exc_info.value.error_code == "BT_REG_UNKNOWN"


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


def test_exists_returns_true_for_registered_name() -> None:
    """exists() returns True for a registered engine name."""
    registry = _registry_with("present")
    assert registry.exists("present") is True


def test_exists_returns_false_for_unregistered_name() -> None:
    """exists() returns False for an unregistered engine name."""
    registry = BacktestingRegistry()
    assert registry.exists("missing") is False


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_returns_new_tuple_each_call() -> None:
    """list() returns a new tuple reference on each call."""
    registry = _registry_with("alpha", "beta")
    first = registry.list()
    second = registry.list()
    assert first == second
    assert first is not second


def test_list_is_unaffected_by_clear() -> None:
    """list() returns empty tuple after clear()."""
    registry = _registry_with("alpha", "beta")
    registry.clear()
    assert registry.list() == ()


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_removes_all_engines() -> None:
    """clear() removes all registered engines."""
    registry = _registry_with("a", "b", "c")
    registry.clear()
    assert registry.list() == ()
    assert not registry.exists("a")
    assert not registry.exists("b")


def test_clear_allows_re_registration() -> None:
    """After clear(), the same name can be registered again."""
    registry = _registry_with("simple")
    registry.clear()
    new_engine = SimpleBacktestingEngine()
    registry.register("simple", new_engine)
    assert registry.get("simple") is new_engine

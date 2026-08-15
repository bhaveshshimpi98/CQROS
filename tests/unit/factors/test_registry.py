"""Unit tests for CQROS Factor Research Engine ``FactorRegistry``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field

import polars as pl
import pytest

from cqros.factors.base import BaseFactor
from cqros.factors.exceptions import FactorRegistrationError, UnknownFactorError
from cqros.factors.metadata import FactorMetadata
from cqros.factors.registry import FactorRegistry
from cqros.factors.schema import FactorStatus


@dataclass(frozen=True, slots=True)
class _StubFactor(BaseFactor):
    """Minimal concrete factor used only for registry unit tests."""

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged."""
        return frame


@dataclass(frozen=True, slots=True)
class _UncheckedFactor:
    """Factor-shaped stub that allows blank names for registration tests."""

    name: str
    version: str = "1.0.0"
    description: str = "stub"
    category: str = "momentum"
    required_features: tuple[str, ...] = ("returns",)
    produced_columns: tuple[str, ...] = ("score",)
    lookback: int = 20
    factor_group: str = "alpha"
    prediction_horizon: int = 1
    enabled: bool = True
    status: FactorStatus = FactorStatus.ACTIVE

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` unchanged."""
        return frame


def _factor(
    name: str,
    *,
    version: str = "1.0.0",
    description: str = "stub",
    category: str = "momentum",
    required_features: tuple[str, ...] = ("returns",),
    produced_columns: tuple[str, ...] | None = None,
    lookback: int = 20,
) -> _StubFactor:
    """Build a stub factor with the given name and optional overrides."""
    return _StubFactor(
        name=name,
        version=version,
        description=description,
        category=category,
        required_features=required_features,
        produced_columns=produced_columns if produced_columns is not None else (name,),
        lookback=lookback,
    )


# --- registration ---


def test_register_and_get() -> None:
    """register stores a factor that get can retrieve by name."""
    registry = FactorRegistry()
    factor = _factor("momentum")
    registry.register(factor)
    assert registry.get("momentum") is factor


def test_register_preserves_identity() -> None:
    """Registry stores the exact factor reference supplied by the caller."""
    registry = FactorRegistry()
    factor = _factor("value")
    registry.register(factor)
    assert registry.get("value") is factor
    assert registry.list()[0] is factor


def test_register_empty_registry_starts_empty() -> None:
    """A new registry has no registered factors."""
    registry = FactorRegistry()
    assert registry.names() == ()
    assert registry.list() == ()
    assert registry.metadata() == ()


def test_register_multiple_distinct_names() -> None:
    """Distinct factor names can be registered sequentially."""
    registry = FactorRegistry()
    momentum = _factor("momentum")
    value = _factor("value")
    quality = _factor("quality")
    registry.register(momentum)
    registry.register(value)
    registry.register(quality)
    assert registry.get("momentum") is momentum
    assert registry.get("value") is value
    assert registry.get("quality") is quality


# --- blank / duplicate registration ---


@pytest.mark.parametrize("name", ["", "   ", "\t", "\n"])
def test_register_rejects_blank_names(name: str) -> None:
    """Blank factor names raise FactorRegistrationError."""
    registry = FactorRegistry()
    with pytest.raises(FactorRegistrationError, match="non-blank"):
        registry.register(_UncheckedFactor(name=name))
    assert registry.names() == ()


def test_register_rejects_non_string_name() -> None:
    """Non-string factor names raise FactorRegistrationError."""
    registry = FactorRegistry()
    with pytest.raises(FactorRegistrationError, match="non-blank"):
        registry.register(_UncheckedFactor(name=None))  # type: ignore[arg-type]
    assert registry.names() == ()


def test_register_rejects_duplicates() -> None:
    """Duplicate factor names raise FactorRegistrationError."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    with pytest.raises(FactorRegistrationError, match="already registered"):
        registry.register(_factor("momentum", version="2.0.0"))
    assert registry.get("momentum").version == "1.0.0"


def test_duplicate_registration_error_code() -> None:
    """Duplicate registration uses a stable error code."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    with pytest.raises(FactorRegistrationError) as exc_info:
        registry.register(_factor("momentum"))
    assert exc_info.value.error_code == "FACTOR-REG-002"
    assert exc_info.value.details["name"] == "momentum"


def test_blank_registration_error_code() -> None:
    """Blank-name registration uses a stable error code."""
    registry = FactorRegistry()
    with pytest.raises(FactorRegistrationError) as exc_info:
        registry.register(_UncheckedFactor(name=""))
    assert exc_info.value.error_code == "FACTOR-REG-001"


# --- register_many / atomicity ---


def test_register_many_registers_all() -> None:
    """register_many stores every provided factor."""
    registry = FactorRegistry()
    momentum = _factor("momentum")
    value = _factor("value")
    registry.register_many((momentum, value))
    assert registry.get("momentum") is momentum
    assert registry.get("value") is value


def test_register_many_empty_iterable() -> None:
    """register_many with an empty iterable is a no-op."""
    registry = FactorRegistry()
    registry.register_many(())
    assert registry.names() == ()


def test_register_many_is_atomic_on_duplicate_existing() -> None:
    """register_many leaves the registry unchanged when a name already exists."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    with pytest.raises(FactorRegistrationError):
        registry.register_many((_factor("value"), _factor("momentum")))
    assert registry.names() == ("momentum",)
    assert not registry.exists("value")


def test_register_many_is_atomic_on_duplicate_within_batch() -> None:
    """register_many rejects duplicate names within the same batch."""
    registry = FactorRegistry()
    with pytest.raises(FactorRegistrationError):
        registry.register_many(
            (_factor("momentum"), _factor("momentum", version="2.0.0")),
        )
    assert registry.names() == ()


def test_register_many_is_atomic_on_blank_name() -> None:
    """register_many leaves the registry unchanged when a blank name appears."""
    registry = FactorRegistry()
    with pytest.raises(FactorRegistrationError):
        registry.register_many((_factor("value"), _UncheckedFactor(name="")))
    assert registry.names() == ()


def test_register_many_is_atomic_on_blank_first() -> None:
    """A blank name at the start of a batch registers nothing."""
    registry = FactorRegistry()
    with pytest.raises(FactorRegistrationError):
        registry.register_many((_UncheckedFactor(name="   "), _factor("value")))
    assert registry.names() == ()


def test_register_many_after_partial_success_then_failure() -> None:
    """Later atomic failure does not keep earlier successful register_many calls."""
    registry = FactorRegistry()
    registry.register_many((_factor("alpha"), _factor("beta")))
    with pytest.raises(FactorRegistrationError):
        registry.register_many((_factor("gamma"), _factor("alpha")))
    assert registry.names() == ("alpha", "beta")
    assert not registry.exists("gamma")


def test_register_many_accepts_list() -> None:
    """register_many accepts list inputs."""
    registry = FactorRegistry()
    factors = [_factor("momentum"), _factor("value")]
    registry.register_many(factors)
    assert registry.names() == ("momentum", "value")


# --- lookup ---


def test_get_unknown_raises() -> None:
    """get raises UnknownFactorError for missing names."""
    registry = FactorRegistry()
    with pytest.raises(UnknownFactorError, match="not registered"):
        registry.get("missing")


def test_get_unknown_error_code() -> None:
    """Unknown lookup uses a stable error code."""
    registry = FactorRegistry()
    with pytest.raises(UnknownFactorError) as exc_info:
        registry.get("missing")
    assert exc_info.value.error_code == "FACTOR-REG-003"
    assert exc_info.value.details["name"] == "missing"


def test_get_after_remove_raises() -> None:
    """get raises after a previously registered factor is removed."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    registry.remove("momentum")
    with pytest.raises(UnknownFactorError):
        registry.get("momentum")


def test_get_is_case_sensitive() -> None:
    """Factor name lookup is case-sensitive."""
    registry = FactorRegistry()
    registry.register(_factor("Momentum"))
    assert registry.exists("Momentum") is True
    assert registry.exists("momentum") is False
    with pytest.raises(UnknownFactorError):
        registry.get("momentum")


# --- exists ---


def test_exists_false_when_empty() -> None:
    """exists returns False for an empty registry."""
    assert FactorRegistry().exists("momentum") is False


def test_exists_true_when_registered() -> None:
    """exists returns True after registration."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    assert registry.exists("momentum") is True


def test_exists_false_for_other_names() -> None:
    """exists returns False for names that were never registered."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    assert registry.exists("value") is False


def test_exists_does_not_raise_for_blank() -> None:
    """exists returns False for blank names without raising."""
    registry = FactorRegistry()
    assert registry.exists("") is False
    assert registry.exists("   ") is False


# --- remove ---


def test_remove_deletes_registered_factor() -> None:
    """remove deletes a registered factor."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    registry.remove("momentum")
    assert registry.exists("momentum") is False


def test_remove_unknown_raises() -> None:
    """remove raises UnknownFactorError for missing names."""
    registry = FactorRegistry()
    with pytest.raises(UnknownFactorError):
        registry.remove("momentum")


def test_remove_unknown_error_code() -> None:
    """Unknown remove uses a stable error code."""
    registry = FactorRegistry()
    with pytest.raises(UnknownFactorError) as exc_info:
        registry.remove("momentum")
    assert exc_info.value.error_code == "FACTOR-REG-003"


def test_remove_one_leaves_others() -> None:
    """remove deletes only the requested factor."""
    registry = FactorRegistry()
    registry.register_many((_factor("momentum"), _factor("value"), _factor("quality")))
    registry.remove("value")
    assert registry.names() == ("momentum", "quality")


def test_remove_then_reregister() -> None:
    """A removed name can be registered again."""
    registry = FactorRegistry()
    first = _factor("momentum", version="1.0.0")
    second = _factor("momentum", version="2.0.0")
    registry.register(first)
    registry.remove("momentum")
    registry.register(second)
    assert registry.get("momentum") is second
    assert registry.get("momentum").version == "2.0.0"


# --- clear ---


def test_clear_removes_all() -> None:
    """clear removes all registered factors."""
    registry = FactorRegistry()
    registry.register_many((_factor("momentum"), _factor("value")))
    registry.clear()
    assert registry.names() == ()
    assert registry.list() == ()
    assert registry.metadata() == ()


def test_clear_on_empty_registry() -> None:
    """clear on an empty registry is a no-op."""
    registry = FactorRegistry()
    registry.clear()
    assert registry.names() == ()


def test_clear_allows_reregistration() -> None:
    """Factors can be registered again after clear."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    registry.clear()
    factor = _factor("momentum", version="9.0.0")
    registry.register(factor)
    assert registry.get("momentum") is factor


# --- metadata snapshots ---


def test_metadata_generation() -> None:
    """metadata projects registered factors into FactorMetadata tuples."""
    registry = FactorRegistry()
    registry.register_many(
        (
            _factor(
                "momentum",
                version="1.2.0",
                description="Cross-sectional momentum",
                category="momentum",
                required_features=("returns",),
                produced_columns=("momentum_score",),
                lookback=20,
            ),
            _factor(
                "value",
                version="2.0.0",
                description="Value score",
                category="value",
                required_features=("book_value", "market_cap"),
                produced_columns=("value_score",),
                lookback=5,
            ),
        )
    )
    metadata = registry.metadata()
    assert isinstance(metadata, tuple)
    assert len(metadata) == 2
    assert all(isinstance(item, FactorMetadata) for item in metadata)
    assert metadata[0].name == "momentum"
    assert metadata[0].version == "1.2.0"
    assert metadata[0].description == "Cross-sectional momentum"
    assert metadata[0].category == "momentum"
    assert metadata[0].required_features == ("returns",)
    assert metadata[0].produced_columns == ("momentum_score",)
    assert metadata[0].lookback == 20
    assert metadata[1].name == "value"
    assert metadata[1].version == "2.0.0"
    assert metadata[1].required_features == ("book_value", "market_cap")


def test_metadata_empty_when_no_factors() -> None:
    """metadata returns an empty tuple for an empty registry."""
    assert FactorRegistry().metadata() == ()


def test_metadata_matches_base_factor_metadata_property() -> None:
    """Projected metadata matches BaseFactor.metadata for registered factors."""
    registry = FactorRegistry()
    factor = _factor("quality", lookback=60)
    registry.register(factor)
    assert registry.metadata()[0] == factor.metadata


def test_metadata_snapshots_are_frozen() -> None:
    """Returned FactorMetadata objects are immutable."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    meta = registry.metadata()[0]
    with pytest.raises(FrozenInstanceError):
        meta.name = "other"  # type: ignore[misc]


# --- immutability ---


def test_returned_collections_are_immutable_snapshots() -> None:
    """Returned tuples are snapshots unaffected by later registry mutation."""
    registry = FactorRegistry()
    registry.register_many((_factor("momentum"), _factor("value")))
    names = registry.names()
    factors = registry.list()
    metadata = registry.metadata()
    assert isinstance(names, tuple)
    assert isinstance(factors, tuple)
    assert isinstance(metadata, tuple)
    registry.clear()
    assert names == ("momentum", "value")
    assert tuple(factor.name for factor in factors) == ("momentum", "value")
    assert tuple(item.name for item in metadata) == ("momentum", "value")
    assert registry.names() == ()


def test_register_does_not_mutate_factor() -> None:
    """Registry stores the factor reference without altering its metadata."""
    registry = FactorRegistry()
    factor = _factor("momentum", version="1.0.0")
    registry.register(factor)
    assert factor.name == "momentum"
    assert factor.version == "1.0.0"
    assert factor.required_features == ("returns",)
    assert registry.get("momentum") is factor


def test_internal_storage_is_private() -> None:
    """Internal storage attribute is private and not part of the public API."""
    registry = FactorRegistry()
    assert hasattr(registry, "_factors")
    assert not hasattr(FactorRegistry, "factors")


def test_list_returns_new_tuple_each_call() -> None:
    """list returns a new tuple instance on each call."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    first = registry.list()
    second = registry.list()
    assert first == second
    assert first is not second


def test_names_returns_new_tuple_each_call() -> None:
    """names returns a new tuple instance on each call."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    first = registry.names()
    second = registry.names()
    assert first == second
    assert first is not second


def test_metadata_returns_new_tuple_each_call() -> None:
    """metadata returns a new tuple instance on each call."""
    registry = FactorRegistry()
    registry.register(_factor("momentum"))
    first = registry.metadata()
    second = registry.metadata()
    assert first == second
    assert first is not second


# --- alphabetical ordering ---


def test_list_and_names_are_alphabetical() -> None:
    """list and names return factors sorted alphabetically by name."""
    registry = FactorRegistry()
    registry.register_many((_factor("zeta"), _factor("alpha"), _factor("mu")))
    assert registry.names() == ("alpha", "mu", "zeta")
    assert tuple(factor.name for factor in registry.list()) == ("alpha", "mu", "zeta")


def test_metadata_is_alphabetical() -> None:
    """metadata follows alphabetical factor-name order."""
    registry = FactorRegistry()
    registry.register_many((_factor("zeta"), _factor("alpha"), _factor("mu")))
    assert tuple(item.name for item in registry.metadata()) == ("alpha", "mu", "zeta")


def test_ordering_stable_after_remove() -> None:
    """Alphabetical order is preserved after removals."""
    registry = FactorRegistry()
    registry.register_many(
        (_factor("delta"), _factor("alpha"), _factor("charlie"), _factor("bravo")),
    )
    registry.remove("charlie")
    assert registry.names() == ("alpha", "bravo", "delta")


def test_ordering_after_register_many() -> None:
    """register_many results are exposed in alphabetical order."""
    registry = FactorRegistry()
    registry.register_many((_factor("m"), _factor("a"), _factor("z")))
    assert registry.names() == ("a", "m", "z")


# --- package export / non-execution ---


def test_package_exports_factor_registry() -> None:
    """FactorRegistry is exported from the factors package."""
    import cqros.factors as factors_package

    assert "FactorRegistry" in factors_package.__all__
    assert factors_package.FactorRegistry is FactorRegistry


def test_registry_does_not_execute_compute() -> None:
    """Registration and listing never invoke factor compute."""

    @dataclass(frozen=True, slots=True)
    class _TrackingFactor(BaseFactor):
        calls: list[str] = field(default_factory=list)

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            self.calls.append("compute")
            return frame

    calls: list[str] = []
    factor = _TrackingFactor(
        name="tracked",
        version="1.0.0",
        description="tracks compute calls",
        category="test",
        required_features=(),
        produced_columns=("tracked",),
        lookback=0,
        calls=calls,
    )
    registry = FactorRegistry()
    registry.register(factor)
    _ = registry.get("tracked")
    _ = registry.list()
    _ = registry.names()
    _ = registry.metadata()
    _ = registry.by_category("test")
    _ = registry.categories()
    _ = registry.metadata_for("tracked")
    assert calls == []


# --- category / metadata lookup ---


def test_by_category_returns_matching_factors_alphabetically() -> None:
    """by_category returns only matching factors in alphabetical name order."""
    registry = FactorRegistry()
    registry.register_many(
        (
            _factor("zeta", category="price"),
            _factor("alpha", category="price"),
            _factor("mu", category="volume"),
        )
    )
    price = registry.by_category("price")
    assert tuple(factor.name for factor in price) == ("alpha", "zeta")
    assert all(factor.category == "price" for factor in price)


def test_by_category_empty_when_no_match() -> None:
    """by_category returns an empty tuple when no factors match."""
    registry = FactorRegistry()
    registry.register(_factor("momentum", category="price"))
    assert registry.by_category("funding") == ()


def test_by_category_is_case_sensitive() -> None:
    """by_category matching is case-sensitive."""
    registry = FactorRegistry()
    registry.register(_factor("momentum", category="price"))
    assert registry.by_category("Price") == ()
    assert len(registry.by_category("price")) == 1


def test_categories_returns_unique_sorted_categories() -> None:
    """categories returns unique category names in alphabetical order."""
    registry = FactorRegistry()
    registry.register_many(
        (
            _factor("a", category="volume"),
            _factor("b", category="price"),
            _factor("c", category="volume"),
            _factor("d", category="funding"),
        )
    )
    assert registry.categories() == ("funding", "price", "volume")


def test_categories_empty_when_no_factors() -> None:
    """categories returns an empty tuple for an empty registry."""
    assert FactorRegistry().categories() == ()


def test_metadata_for_returns_single_factor_metadata() -> None:
    """metadata_for projects one registered factor into FactorMetadata."""
    registry = FactorRegistry()
    factor = _factor(
        "momentum",
        version="1.2.0",
        description="Cross-sectional momentum",
        category="price",
        required_features=("returns",),
        produced_columns=("momentum_score",),
        lookback=20,
    )
    registry.register(factor)
    metadata = registry.metadata_for("momentum")
    assert isinstance(metadata, FactorMetadata)
    assert metadata == factor.metadata


def test_metadata_for_unknown_raises() -> None:
    """metadata_for raises UnknownFactorError for missing names."""
    registry = FactorRegistry()
    with pytest.raises(UnknownFactorError, match="not registered"):
        registry.metadata_for("missing")


def test_metadata_for_unknown_error_code() -> None:
    """Unknown metadata_for lookup uses a stable error code."""
    registry = FactorRegistry()
    with pytest.raises(UnknownFactorError) as exc_info:
        registry.metadata_for("missing")
    assert exc_info.value.error_code == "FACTOR-REG-003"
    assert exc_info.value.details["name"] == "missing"


# --- duplicate produced columns ---


def test_register_rejects_duplicate_produced_columns() -> None:
    """Duplicate produced columns across factors raise FactorRegistrationError."""
    registry = FactorRegistry()
    registry.register(_factor("momentum", produced_columns=("signal",)))
    with pytest.raises(FactorRegistrationError, match="produced column already registered"):
        registry.register(_factor("value", produced_columns=("signal",)))
    assert registry.names() == ("momentum",)


def test_duplicate_produced_column_error_code() -> None:
    """Duplicate produced-column registration uses a stable error code."""
    registry = FactorRegistry()
    registry.register(_factor("momentum", produced_columns=("signal",)))
    with pytest.raises(FactorRegistrationError) as exc_info:
        registry.register(_factor("value", produced_columns=("signal",)))
    assert exc_info.value.error_code == "FACTOR-REG-004"
    assert exc_info.value.details["column"] == "signal"
    assert exc_info.value.details["name"] == "value"
    assert exc_info.value.details["owner"] == "momentum"


def test_register_rejects_duplicate_produced_columns_within_factor() -> None:
    """A factor claiming the same produced column twice is rejected."""
    registry = FactorRegistry()
    with pytest.raises(FactorRegistrationError, match="produced column already registered"):
        registry.register(_factor("momentum", produced_columns=("signal", "signal")))
    assert registry.names() == ()


def test_register_many_rejects_duplicate_produced_columns_atomically() -> None:
    """register_many leaves the registry unchanged on produced-column conflicts."""
    registry = FactorRegistry()
    with pytest.raises(FactorRegistrationError, match="produced column already registered"):
        registry.register_many(
            (
                _factor("momentum", produced_columns=("signal",)),
                _factor("value", produced_columns=("signal",)),
            )
        )
    assert registry.names() == ()


def test_remove_releases_produced_columns() -> None:
    """Removing a factor frees its produced columns for re-registration."""
    registry = FactorRegistry()
    registry.register(_factor("momentum", produced_columns=("signal",)))
    registry.remove("momentum")
    replacement = _factor("value", produced_columns=("signal",))
    registry.register(replacement)
    assert registry.get("value") is replacement


def test_clear_releases_produced_columns() -> None:
    """clear frees all produced-column ownership."""
    registry = FactorRegistry()
    registry.register(_factor("momentum", produced_columns=("signal",)))
    registry.clear()
    replacement = _factor("value", produced_columns=("signal",))
    registry.register(replacement)
    assert registry.get("value") is replacement

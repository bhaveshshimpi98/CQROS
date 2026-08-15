"""Unit tests for CQROS forward-return target generation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import polars as pl
import pytest

from cqros.core.exceptions import TargetError as CoreTargetError
from cqros.research.exceptions import TargetDefinitionError, TargetError
from cqros.research.target import ForwardReturnTarget, TargetDefinition


def _definition(**overrides: object) -> TargetDefinition:
    """Build a target definition with optional field overrides."""
    values: dict[str, object] = {
        "name": "forward_return_1",
        "horizon": 1,
        "price_column": "close",
        "output_column": "forward_return",
    }
    values.update(overrides)
    return TargetDefinition(**values)  # type: ignore[arg-type]


def _target(**overrides: object) -> ForwardReturnTarget:
    """Build a forward-return target from a definition."""
    return ForwardReturnTarget(_definition(**overrides))


# --- metadata ---


def test_target_definition_is_frozen_dataclass() -> None:
    """TargetDefinition is an immutable slotted dataclass."""
    definition = _definition()
    assert is_dataclass(definition)
    with pytest.raises(FrozenInstanceError):
        definition.name = "other"  # type: ignore[misc]


def test_target_definition_metadata_defaults() -> None:
    """Default price and output columns match the research contract."""
    definition = TargetDefinition(name="fwd", horizon=1)
    assert definition.price_column == "close"
    assert definition.output_column == "forward_return"


def test_forward_return_target_exposes_definition_metadata() -> None:
    """ForwardReturnTarget exposes definition metadata through properties."""
    target = _target(
        name="fwd_5",
        horizon=5,
        price_column="close",
        output_column="fwd_ret_5",
    )
    assert target.name == "fwd_5"
    assert target.horizon == 5
    assert target.price_column == "close"
    assert target.output_column == "fwd_ret_5"
    assert target.definition.name == "fwd_5"
    assert target.definition.horizon == 5


def test_target_error_is_core_target_error() -> None:
    """Research TargetError is the shared core TargetError type."""
    assert TargetError is CoreTargetError


# --- calculation ---


def test_forward_return_calculation_horizon_1() -> None:
    """Horizon-1 forward returns match (close.shift(-1) / close) - 1."""
    frame = pl.DataFrame({"close": [100.0, 110.0, 99.0, 108.9]})
    result = _target(horizon=1).transform(frame)
    values = result.get_column("forward_return").to_list()
    assert values[0] == pytest.approx((110.0 / 100.0) - 1.0)
    assert values[1] == pytest.approx((99.0 / 110.0) - 1.0)
    assert values[2] == pytest.approx((108.9 / 99.0) - 1.0)
    assert values[3] is None


def test_forward_return_calculation_horizon_5() -> None:
    """Horizon-5 forward returns use a five-row look-ahead."""
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 120.0]
    frame = pl.DataFrame({"close": closes})
    result = _target(horizon=5).transform(frame)
    values = result.get_column("forward_return").to_list()
    assert values[0] == pytest.approx((110.0 / 100.0) - 1.0)
    assert values[1] == pytest.approx((120.0 / 101.0) - 1.0)
    assert values[2] is None
    assert values[3] is None
    assert values[4] is None
    assert values[5] is None
    assert values[6] is None


def test_custom_price_and_output_columns() -> None:
    """Custom price and output column names are honored."""
    frame = pl.DataFrame({"px": [10.0, 12.0, 15.0]})
    target = _target(price_column="px", output_column="ret_fwd")
    result = target.transform(frame)
    assert "ret_fwd" in result.columns
    assert "forward_return" not in result.columns
    values = result.get_column("ret_fwd").to_list()
    assert values[0] == pytest.approx(0.2)
    assert values[1] == pytest.approx(0.25)
    assert values[2] is None


def test_preserves_existing_columns() -> None:
    """Existing non-price columns are preserved alongside the target."""
    frame = pl.DataFrame({"close": [1.0, 2.0], "volume": [10, 20]})
    result = _target().transform(frame)
    assert result.columns == ["close", "volume", "forward_return"]


def test_zero_return_when_price_unchanged() -> None:
    """Unchanged future price yields a zero forward return."""
    frame = pl.DataFrame({"close": [50.0, 50.0]})
    result = _target(horizon=1).transform(frame)
    assert result.get_column("forward_return")[0] == pytest.approx(0.0)
    assert result.get_column("forward_return")[1] is None


# --- null tail ---


def test_null_tail_matches_horizon() -> None:
    """The final horizon rows of the target column are null."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    values = _target(horizon=3).transform(frame).get_column("forward_return").to_list()
    assert values[0] is not None
    assert values[1] is not None
    assert values[2] is not None
    assert values[3] is None
    assert values[4] is None
    assert values[5] is None


def test_null_tail_never_filled() -> None:
    """Null tail values remain null and are never forward-filled."""
    frame = pl.DataFrame({"close": [10.0, 20.0, 30.0]})
    values = _target(horizon=2).transform(frame).get_column("forward_return").to_list()
    assert values[0] == pytest.approx(2.0)
    assert values[1] is None
    assert values[2] is None


def test_single_row_is_entirely_null_for_horizon_1() -> None:
    """A single-row frame yields a single null forward return."""
    frame = pl.DataFrame({"close": [42.0]})
    result = _target(horizon=1).transform(frame)
    assert result.height == 1
    assert result.get_column("forward_return")[0] is None


def test_empty_dataframe() -> None:
    """An empty frame with the price column yields an empty target column."""
    frame = pl.DataFrame({"close": pl.Series("close", [], dtype=pl.Float64)})
    result = _target().transform(frame)
    assert result.height == 0
    assert "forward_return" in result.columns


# --- missing column ---


def test_missing_price_column_raises_target_error() -> None:
    """Missing price column raises TargetError."""
    frame = pl.DataFrame({"open": [1.0, 2.0]})
    with pytest.raises(TargetError, match="required column missing: close") as exc_info:
        _target().transform(frame)
    error = exc_info.value
    assert error.error_code == "RESEARCH-TARGET-005"
    assert error.details["required_column"] == "close"
    assert error.details["available_columns"] == ("open",)


def test_missing_custom_price_column_raises() -> None:
    """Missing custom price column raises TargetError with that column name."""
    frame = pl.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(TargetError, match="required column missing: px"):
        _target(price_column="px").transform(frame)


# --- invalid definition ---


@pytest.mark.parametrize("horizon", [0, -1, -10, True, False, 1.5, "1", None])
def test_invalid_horizon_raises(horizon: object) -> None:
    """Invalid horizon values raise TargetDefinitionError."""
    with pytest.raises(TargetDefinitionError, match="horizon must be an integer"):
        _definition(horizon=horizon)


@pytest.mark.parametrize("horizon", [1, 2, 5, 20])
def test_valid_horizon_accepted(horizon: int) -> None:
    """Positive integer horizons are accepted."""
    assert _definition(horizon=horizon).horizon == horizon


@pytest.mark.parametrize("output_column", ["", "   ", "\t"])
def test_blank_output_column_raises(output_column: str) -> None:
    """Blank output column raises TargetDefinitionError."""
    with pytest.raises(TargetDefinitionError, match="output_column must be a non-blank"):
        _definition(output_column=output_column)


@pytest.mark.parametrize("name", ["", "   "])
def test_blank_name_raises(name: str) -> None:
    """Blank name raises TargetDefinitionError."""
    with pytest.raises(TargetDefinitionError, match="name must be a non-blank"):
        _definition(name=name)


@pytest.mark.parametrize("price_column", ["", "   "])
def test_blank_price_column_raises(price_column: str) -> None:
    """Blank price column raises TargetDefinitionError."""
    with pytest.raises(TargetDefinitionError, match="price_column must be a non-blank"):
        _definition(price_column=price_column)


def test_invalid_horizon_error_code() -> None:
    """Invalid horizon uses a stable error code."""
    with pytest.raises(TargetDefinitionError) as exc_info:
        _definition(horizon=0)
    assert exc_info.value.error_code == "RESEARCH-TARGET-002"


def test_blank_output_column_error_code() -> None:
    """Blank output column uses a stable error code."""
    with pytest.raises(TargetDefinitionError) as exc_info:
        _definition(output_column="")
    assert exc_info.value.error_code == "RESEARCH-TARGET-004"


# --- immutability ---


def test_input_immutability() -> None:
    """transform does not mutate the caller-supplied DataFrame."""
    frame = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    original_columns = list(frame.columns)
    original_values = frame.get_column("close").to_list()
    result = _target().transform(frame)
    assert list(frame.columns) == original_columns
    assert frame.get_column("close").to_list() == original_values
    assert "forward_return" not in frame.columns
    assert "forward_return" in result.columns
    assert result is not frame


def test_definition_immutability_via_target() -> None:
    """Definition exposed by ForwardReturnTarget cannot be mutated."""
    target = _target()
    with pytest.raises(FrozenInstanceError):
        target.definition.horizon = 9  # type: ignore[misc]


def test_package_exports() -> None:
    """Public target symbols are exported from the research package."""
    import cqros.research as research_package

    assert "ForwardReturnTarget" in research_package.__all__
    assert "TargetDefinition" in research_package.__all__
    assert "TargetError" in research_package.__all__
    assert "TargetDefinitionError" in research_package.__all__
    assert research_package.ForwardReturnTarget is ForwardReturnTarget

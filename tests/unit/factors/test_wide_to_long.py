"""Unit tests for CQROS ``WideToLongFactorTransformer``."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.factors.default_registry import build_default_registry
from cqros.factors.exceptions import FactorValidationError
from cqros.factors.metadata import FactorMetadata
from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FACTOR_SCHEMA,
    FactorStatus,
)
from cqros.factors.wide_to_long import WideToLongFactorTransformer

_PRODUCTION_FACTOR_COUNT = 111
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"


def _metadata(
    name: str,
    *,
    version: str = "1.0.0",
    category: str = "price",
    lookback: int = 20,
    factor_group: str = "alpha",
    prediction_horizon: int = 1,
    enabled: bool = True,
    status: FactorStatus = FactorStatus.ACTIVE,
) -> FactorMetadata:
    """Build ``FactorMetadata`` for transformer unit tests."""
    return FactorMetadata(
        name=name,
        version=version,
        description=f"{name} stub",
        category=category,
        required_features=("close",),
        produced_columns=(name,),
        lookback=lookback,
        factor_group=factor_group,
        prediction_horizon=prediction_horizon,
        enabled=enabled,
        status=status,
    )


def _wide_frame(
    factor_values: Mapping[str, list[float]],
    *,
    row_count: int | None = None,
) -> pl.DataFrame:
    """Build a synthetic wide factor matrix."""
    height = row_count if row_count is not None else len(next(iter(factor_values.values())))
    data: dict[str, object] = {
        "symbol": [_SYMBOL] * height,
        "timeframe": [_TIMEFRAME] * height,
        "open_time": list(range(height)),
    }
    data.update(factor_values)
    return pl.DataFrame(data)


def _metadata_map(*names: str, **overrides: FactorMetadata) -> dict[str, FactorMetadata]:
    """Build a metadata map for the given factor names."""
    mapping = {name: _metadata(name) for name in names}
    mapping.update(overrides)
    return mapping


def _production_metadata_map() -> dict[str, FactorMetadata]:
    """Build metadata for the full production catalog (test helper only)."""
    mapping: dict[str, FactorMetadata] = {}
    for factor in build_default_registry().list():
        mapping[factor.name] = FactorMetadata(
            name=factor.name,
            version=factor.version,
            description=factor.description,
            category=factor.category,
            required_features=tuple(factor.required_features),
            produced_columns=tuple(factor.produced_columns),
            lookback=factor.lookback,
            factor_group=factor.factor_group,
            prediction_horizon=factor.prediction_horizon,
            enabled=factor.enabled,
            status=factor.status,
        )
    return mapping


class TestWideToLongFactorTransformer:
    """Unit tests for wide-to-long conversion and metadata enrichment."""

    def test_single_factor(self) -> None:
        """One factor column expands to one long row per bar."""
        frame = _wide_frame({"momentum": [0.1, 0.2]})
        metadata = _metadata_map("momentum")
        result = WideToLongFactorTransformer().transform(frame, metadata)

        assert result.height == 2
        assert result.columns == list(CANONICAL_COLUMN_ORDER)
        assert result.schema == FACTOR_SCHEMA
        assert result.get_column("factor_name").to_list() == ["momentum", "momentum"]
        assert result.get_column("factor_value").to_list() == pytest.approx([0.1, 0.2])
        assert result.get_column("factor_version").to_list() == ["1.0.0", "1.0.0"]
        assert result.get_column("factor_category").to_list() == ["price", "price"]
        assert result.get_column("factor_group").to_list() == ["alpha", "alpha"]
        assert result.get_column("lookback").to_list() == [20, 20]
        assert result.get_column("prediction_horizon").to_list() == [1, 1]
        assert result.get_column("enabled").to_list() == [True, True]
        assert result.get_column("status").to_list() == ["ACTIVE", "ACTIVE"]

    def test_multiple_factors(self) -> None:
        """Multiple factor columns expand correctly with distinct metadata."""
        frame = _wide_frame(
            {
                "momentum": [0.1],
                "rsi": [55.0],
            }
        )
        metadata = {
            "momentum": _metadata("momentum", version="1.0.0", category="price", lookback=20),
            "rsi": _metadata(
                "rsi",
                version="2.0.0",
                category="oscillator",
                lookback=14,
                factor_group="signal",
                prediction_horizon=3,
                enabled=False,
                status=FactorStatus.DEPRECATED,
            ),
        }
        result = WideToLongFactorTransformer().transform(frame, metadata)

        assert result.height == 2
        by_name = {row["factor_name"]: row for row in result.to_dicts()}
        assert by_name["momentum"]["factor_value"] == pytest.approx(0.1)
        assert by_name["momentum"]["factor_version"] == "1.0.0"
        assert by_name["momentum"]["factor_category"] == "price"
        assert by_name["momentum"]["lookback"] == 20
        assert by_name["rsi"]["factor_value"] == pytest.approx(55.0)
        assert by_name["rsi"]["factor_version"] == "2.0.0"
        assert by_name["rsi"]["factor_category"] == "oscillator"
        assert by_name["rsi"]["factor_group"] == "signal"
        assert by_name["rsi"]["lookback"] == 14
        assert by_name["rsi"]["prediction_horizon"] == 3
        assert by_name["rsi"]["enabled"] is False
        assert by_name["rsi"]["status"] == "DEPRECATED"

    def test_production_sized_dataframe_row_expansion(self) -> None:
        """2 bars × 111 factors expands to 222 canonical rows."""
        metadata = _production_metadata_map()
        assert len(metadata) == _PRODUCTION_FACTOR_COUNT
        names = tuple(metadata.keys())
        frame = _wide_frame(
            {name: [float(index), float(index + 1000)] for index, name in enumerate(names)},
            row_count=2,
        )
        result = WideToLongFactorTransformer().transform(frame, metadata)

        assert result.height == 2 * _PRODUCTION_FACTOR_COUNT
        assert result.schema == FACTOR_SCHEMA
        assert set(result.get_column("factor_name").to_list()) == set(names)

        first_bar = result.filter(pl.col("open_time") == 0)
        assert first_bar.height == _PRODUCTION_FACTOR_COUNT
        for index, name in enumerate(names):
            row = first_bar.filter(pl.col("factor_name") == name).to_dicts()[0]
            assert row["factor_value"] == pytest.approx(float(index))
            assert row["factor_version"] == metadata[name].version
            assert row["factor_category"] == metadata[name].category
            assert row["factor_group"] == metadata[name].factor_group
            assert row["lookback"] == metadata[name].lookback
            assert row["prediction_horizon"] == metadata[name].prediction_horizon
            assert row["enabled"] is metadata[name].enabled
            assert row["status"] == metadata[name].status.value

    def test_empty_dataframe(self) -> None:
        """Empty wide frames yield an empty ``FACTOR_SCHEMA`` frame."""
        frame = _wide_frame({"momentum": [], "rsi": []}, row_count=0)
        metadata = _metadata_map("momentum", "rsi")
        result = WideToLongFactorTransformer().transform(frame, metadata)

        assert result.height == 0
        assert result.schema == FACTOR_SCHEMA
        assert result.columns == list(CANONICAL_COLUMN_ORDER)

    def test_empty_dataframe_without_factor_columns(self) -> None:
        """Primary-key-only empty frames return an empty schema-conformant frame."""
        frame = pl.DataFrame(
            {
                "symbol": pl.Series([], dtype=pl.String),
                "timeframe": pl.Series([], dtype=pl.String),
                "open_time": pl.Series([], dtype=pl.Int64),
            }
        )
        result = WideToLongFactorTransformer().transform(frame, {})
        assert result.height == 0
        assert result.schema == FACTOR_SCHEMA

    def test_missing_primary_keys(self) -> None:
        """Missing primary-key columns fail fast."""
        frame = pl.DataFrame({"symbol": ["BTCUSDT"], "momentum": [0.1]})
        with pytest.raises(FactorValidationError, match="primary key") as exc_info:
            WideToLongFactorTransformer().transform(frame, _metadata_map("momentum"))
        assert exc_info.value.error_code == "FACTOR-W2L-001"

    def test_missing_metadata(self) -> None:
        """Unknown factor columns without metadata fail fast."""
        frame = _wide_frame({"momentum": [0.1], "rsi": [50.0]})
        with pytest.raises(FactorValidationError, match="metadata is missing") as exc_info:
            WideToLongFactorTransformer().transform(frame, _metadata_map("momentum"))
        assert exc_info.value.error_code == "FACTOR-W2L-003"
        missing = exc_info.value.details["missing_factor_names"]
        assert isinstance(missing, tuple)
        assert "rsi" in missing

    def test_metadata_name_mismatch(self) -> None:
        """Metadata map key must equal ``FactorMetadata.name``."""
        frame = _wide_frame({"momentum": [0.1]})
        mismatched = {
            "momentum": _metadata("not_momentum"),
        }
        with pytest.raises(FactorValidationError, match="does not match") as exc_info:
            WideToLongFactorTransformer().transform(frame, mismatched)
        assert exc_info.value.error_code == "FACTOR-W2L-004"

    def test_schema_validation_rejects_non_numeric_factor_value(self) -> None:
        """Non-castable ``factor_value`` values fail ``FACTOR_SCHEMA`` validation."""
        frame = pl.DataFrame(
            {
                "symbol": [_SYMBOL],
                "timeframe": [_TIMEFRAME],
                "open_time": [0],
                "momentum": ["not-a-number"],
            }
        )
        with pytest.raises(FactorValidationError, match="FACTOR_SCHEMA") as exc_info:
            WideToLongFactorTransformer().transform(frame, _metadata_map("momentum"))
        assert exc_info.value.error_code == "FACTOR-W2L-005"

    def test_schema_dtypes_match_column_dtypes(self) -> None:
        """Output dtypes match the canonical column dtype contract."""
        frame = _wide_frame({"momentum": [0.1]})
        result = WideToLongFactorTransformer().transform(frame, _metadata_map("momentum"))
        for column, dtype in COLUMN_DTYPES.items():
            assert result.schema[column] == dtype

    def test_correct_metadata_enrichment_is_not_fabricated(self) -> None:
        """Enrichment uses only injected metadata values."""
        frame = _wide_frame({"momentum": [1.5]})
        metadata = {
            "momentum": _metadata(
                "momentum",
                version="9.9.9",
                category="custom",
                lookback=7,
                factor_group="research",
                prediction_horizon=5,
                enabled=False,
                status=FactorStatus.DEPRECATED,
            )
        }
        result = WideToLongFactorTransformer().transform(frame, metadata)
        row = result.to_dicts()[0]
        assert row["factor_version"] == "9.9.9"
        assert row["factor_category"] == "custom"
        assert row["factor_group"] == "research"
        assert row["lookback"] == 7
        assert row["prediction_horizon"] == 5
        assert row["enabled"] is False
        assert row["status"] == "DEPRECATED"
        assert row["factor_value"] == pytest.approx(1.5)

    def test_preserves_primary_keys(self) -> None:
        """Primary keys are preserved across the expansion."""
        frame = pl.DataFrame(
            {
                "symbol": ["BTCUSDT", "ETHUSDT"],
                "timeframe": ["1h", "4h"],
                "open_time": [10, 20],
                "momentum": [0.1, 0.2],
            }
        )
        result = WideToLongFactorTransformer().transform(frame, _metadata_map("momentum"))
        expected = pl.DataFrame(
            {
                "symbol": ["BTCUSDT", "ETHUSDT"],
                "timeframe": ["1h", "4h"],
                "open_time": [10, 20],
                "factor_name": ["momentum", "momentum"],
                "factor_value": [0.1, 0.2],
            }
        )
        assert_frame_equal(
            result.select(["symbol", "timeframe", "open_time", "factor_name", "factor_value"]),
            expected,
            check_dtypes=False,
        )

    def test_rejects_non_dataframe_input(self) -> None:
        """Non-DataFrame inputs fail with a typed validation error."""
        with pytest.raises(FactorValidationError, match="polars DataFrame") as exc_info:
            WideToLongFactorTransformer().transform(
                {"momentum": [0.1]},  # type: ignore[arg-type]
                _metadata_map("momentum"),
            )
        assert exc_info.value.error_code == "FACTOR-W2L-006"

    def test_extra_metadata_keys_are_ignored(self) -> None:
        """Metadata entries for absent factor columns are ignored."""
        frame = _wide_frame({"momentum": [0.1]})
        metadata = _metadata_map("momentum", "rsi")
        result = WideToLongFactorTransformer().transform(frame, metadata)
        assert result.height == 1
        assert result.get_column("factor_name").to_list() == ["momentum"]

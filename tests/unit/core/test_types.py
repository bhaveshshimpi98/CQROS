"""Unit tests for CQROS shared type aliases, type variables, and protocols."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import get_args, get_origin

import cqros.core.constants as constants
import cqros.core.types as types


def _alias_value(alias: object) -> object:
    """Return the underlying value of a PEP 695 type alias when present."""
    return getattr(alias, "__value__", alias)


def test_all_exports_are_defined() -> None:
    """Every name in ``__all__`` resolves to a module attribute."""
    for name in types.__all__:
        assert hasattr(types, name), f"missing export: {name}"


def test_all_exports_are_unique() -> None:
    """``__all__`` must not list the same name twice."""
    assert len(types.__all__) == len(set(types.__all__))


def test_type_variables_are_distinct() -> None:
    """Generic type variables remain independently usable."""
    assert types.T is not types.KT
    assert types.KT is not types.VT
    assert types.T is not types.VT


def test_domain_aliases_accept_string_values() -> None:
    """Open domain aliases are compatible with plain strings."""
    symbol: types.Symbol = "BTCUSDT"
    asset: types.Asset = "BTC"
    exchange: types.Exchange = "binance"
    market: types.Market = "usdt_perpetual"
    timeframe: types.Timeframe = "1m"

    assert symbol == "BTCUSDT"
    assert asset == "BTC"
    assert exchange == constants.EXCHANGE_BINANCE
    assert market == constants.MARKET_USDT_PERPETUAL
    assert timeframe == constants.TIMEFRAME_1M


def test_quantity_aliases_accept_floats() -> None:
    """Market quantity aliases are compatible with float values."""
    price: types.Price = 42000.5
    quantity: types.Quantity = 0.01
    volume: types.Volume = 1250.0
    percentage: types.Percentage = 0.15
    leverage: types.Leverage = 3.0

    assert price == 42000.5
    assert quantity == 0.01
    assert volume == 1250.0
    assert percentage == 0.15
    assert leverage == 3.0


def test_timestamp_and_id_aliases() -> None:
    """Timestamp and identifier aliases accept timezone-aware and string values."""
    timestamp: types.Timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    unix_ts: types.UnixTimestamp = 1_700_000_000
    unix_ts_ms: types.UnixTimestampMs = 1_700_000_000_000
    entity_id: types.Id = "exp_001"

    assert timestamp.tzinfo is UTC
    assert unix_ts == 1_700_000_000
    assert unix_ts_ms == 1_700_000_000_000
    assert entity_id == "exp_001"


def test_feature_aliases() -> None:
    """Feature vector and matrix aliases accept nested float sequences."""
    value: types.FeatureValue = 1.25
    vector: types.FeatureVector = (1.0, 2.0, 3.0)
    matrix: types.FeatureMatrix = (
        (1.0, 2.0),
        (3.0, 4.0),
    )

    assert value == 1.25
    assert list(vector) == [1.0, 2.0, 3.0]
    assert len(matrix) == 2


def test_metadata_and_filepath_aliases() -> None:
    """Metadata and file path aliases accept mapping and path-like values."""
    metadata: types.Metadata = {
        "version": "1.0.0",
        "seed": 42,
        "enabled": True,
        "tags": ["alpha", "beta"],
    }
    path_obj: types.FilePath = Path("data") / "raw"
    path_str: types.FilePath = "data/raw"

    assert metadata["version"] == "1.0.0"
    assert isinstance(path_obj, Path)
    assert path_str == "data/raw"


def test_environment_literal_matches_constants() -> None:
    """Environment literals align with supported environment constants."""
    values = set(get_args(_alias_value(types.EnvironmentName)))
    assert values == set(constants.SUPPORTED_ENVIRONMENTS)


def test_supported_timeframe_literal_matches_constants() -> None:
    """Supported timeframe literals align with timeframe constants."""
    values = set(get_args(_alias_value(types.SupportedTimeframe)))
    assert values == set(constants.SUPPORTED_TIMEFRAMES)


def test_compression_literal_matches_constants() -> None:
    """Compression codec literals align with compression constants."""
    values = set(get_args(_alias_value(types.CompressionCodec)))
    assert values == set(constants.SUPPORTED_COMPRESSION_CODECS)


def test_file_format_literal_covers_known_formats() -> None:
    """File format literals cover the project file-format constants."""
    values = set(get_args(_alias_value(types.FileFormat)))
    assert constants.FILE_FORMAT_PARQUET in values
    assert constants.FILE_FORMAT_JSON in values
    assert constants.FILE_FORMAT_YAML in values
    assert constants.FILE_FORMAT_TOML in values
    assert constants.FILE_FORMAT_CSV in values


def test_hash_and_missing_policy_literals_match_constants() -> None:
    """Hash and missing-data policy literals align with constants."""
    assert get_args(_alias_value(types.HashAlgorithm)) == (constants.HASH_ALGORITHM_SHA256,)
    assert set(get_args(_alias_value(types.MissingDataPolicy))) == set(
        constants.SUPPORTED_MISSING_DATA_POLICIES
    )


def test_protocols_are_structural() -> None:
    """Protocol implementations satisfy structural typing contracts."""

    class _Sample:
        def to_dict(self) -> dict[str, types.JSONValue]:
            return {"ok": True}

        def validate(self) -> None:
            return None

        def serialize(self) -> bytes:
            return b"{}"

    sample = _Sample()
    as_dict: types.SupportsToDict = sample
    as_validate: types.SupportsValidate = sample
    as_serialize: types.SupportsSerialize = sample

    assert as_dict.to_dict() == {"ok": True}
    assert as_validate.validate() is None
    assert as_serialize.serialize() == b"{}"


def test_json_value_origin_is_union() -> None:
    """``JSONValue`` remains a union suitable for nested metadata trees."""
    origin = get_origin(_alias_value(types.JSONValue))
    assert origin is not None

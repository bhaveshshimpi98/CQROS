"""Unit tests for the CQROS merged feature dataset schema."""

from __future__ import annotations

import polars as pl

from cqros.features import (
    ATRFeature,
    BuyPressureFeature,
    BuySellRatioFeature,
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    CrowdingScoreFeature,
    DeltaVolumeFeature,
    DollarVolumeFeature,
    FEATURE_COLUMNS,
    FEATURE_NAMES,
    FeatureRegistry,
    FlowImbalanceFeature,
    FundingChangeFeature,
    FundingMomentumFeature,
    FundingRollingMeanFeature,
    FundingZScoreFeature,
    LogReturnsFeature,
    MERGED_FEATURE_SCHEMA,
    OIChangeFeature,
    OIMomentumFeature,
    OIPercentChangeFeature,
    OIRollingMeanFeature,
    OIZScoreFeature,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    RatioChangeFeature,
    RatioMomentumFeature,
    RatioZScoreFeature,
    ReturnsFeature,
    RollingMaxFeature,
    RollingMeanFeature,
    RollingMinFeature,
    RollingStdFeature,
    SellPressureFeature,
)
from cqros.features.schema import (
    CANONICAL_COLUMN_ORDER as CANONICAL_COLUMN_ORDER_DIRECT,
)

_IMPLEMENTED_FEATURES = (
    ReturnsFeature(),
    LogReturnsFeature(),
    RollingMeanFeature(),
    RollingStdFeature(),
    RollingMaxFeature(),
    RollingMinFeature(),
    ATRFeature(),
    DollarVolumeFeature(),
    FundingChangeFeature(),
    FundingRollingMeanFeature(),
    FundingZScoreFeature(),
    FundingMomentumFeature(),
    OIChangeFeature(),
    OIPercentChangeFeature(),
    OIRollingMeanFeature(),
    OIZScoreFeature(),
    OIMomentumFeature(),
    BuyPressureFeature(),
    SellPressureFeature(),
    BuySellRatioFeature(),
    DeltaVolumeFeature(),
    FlowImbalanceFeature(),
    RatioChangeFeature(),
    RatioMomentumFeature(),
    RatioZScoreFeature(),
    CrowdingScoreFeature(),
)


def test_merged_feature_schema_is_exported_from_package() -> None:
    """Package export matches the schema module constant."""
    assert CANONICAL_COLUMN_ORDER is CANONICAL_COLUMN_ORDER_DIRECT


def test_primary_key_columns() -> None:
    """Primary key is symbol, timeframe, open_time."""
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")


def test_canonical_column_order() -> None:
    """Canonical order is primary key followed by feature columns."""
    assert CANONICAL_COLUMN_ORDER[:3] == PRIMARY_KEY_COLUMNS
    assert CANONICAL_COLUMN_ORDER[3:] == FEATURE_COLUMNS
    assert CANONICAL_COLUMN_ORDER == (
        "symbol",
        "timeframe",
        "open_time",
        "returns",
        "log_returns",
        "rolling_mean",
        "rolling_std",
        "rolling_max",
        "rolling_min",
        "atr",
        "dollar_volume",
        "funding_change",
        "funding_rolling_mean",
        "funding_zscore",
        "funding_momentum",
        "oi_change",
        "oi_percent_change",
        "oi_rolling_mean",
        "oi_zscore",
        "oi_momentum",
        "buy_pressure",
        "sell_pressure",
        "buy_sell_ratio",
        "delta_volume",
        "flow_imbalance",
        "ratio_change",
        "ratio_momentum",
        "ratio_zscore",
        "crowding_score",
    )


def test_required_columns_match_canonical_order() -> None:
    """Required columns expose the full merged schema contract."""
    assert REQUIRED_COLUMNS == CANONICAL_COLUMN_ORDER


def test_column_names_are_unique() -> None:
    """Canonical column names contain no duplicates."""
    assert len(CANONICAL_COLUMN_ORDER) == len(set(CANONICAL_COLUMN_ORDER))
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_feature_names_are_unique() -> None:
    """Feature names contain no duplicates."""
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert FEATURE_NAMES == FEATURE_COLUMNS


def test_feature_column_count_matches_implemented_catalog() -> None:
    """Schema enumerates exactly the 26 currently implemented features."""
    assert len(FEATURE_COLUMNS) == 26
    assert len(_IMPLEMENTED_FEATURES) == 26


def test_column_dtypes_cover_canonical_order() -> None:
    """Expected dtypes are defined for every canonical column."""
    assert tuple(COLUMN_DTYPES) == CANONICAL_COLUMN_ORDER
    assert COLUMN_DTYPES["symbol"] == pl.String
    assert COLUMN_DTYPES["timeframe"] == pl.String
    assert COLUMN_DTYPES["open_time"] == pl.Int64
    for column in FEATURE_COLUMNS:
        assert COLUMN_DTYPES[column] == pl.Float64


def test_merged_feature_schema_matches_canonical_order_and_dtypes() -> None:
    """Polars schema preserves canonical order and expected dtypes."""
    assert MERGED_FEATURE_SCHEMA.names() == list(CANONICAL_COLUMN_ORDER)
    for column in CANONICAL_COLUMN_ORDER:
        assert MERGED_FEATURE_SCHEMA[column] == COLUMN_DTYPES[column]


def test_schema_contains_every_registered_feature_output() -> None:
    """Every registered feature produced column appears in the schema."""
    registry = FeatureRegistry()
    registry.register_many(_IMPLEMENTED_FEATURES)

    produced: list[str] = []
    for feature in registry.list():
        produced.extend(feature.produced_columns)

    assert len(produced) == len(set(produced))
    assert set(produced) == set(FEATURE_COLUMNS)
    assert set(FEATURE_COLUMNS).issubset(set(CANONICAL_COLUMN_ORDER))
    for column in produced:
        assert column in MERGED_FEATURE_SCHEMA


def test_registered_feature_names_match_schema_feature_names() -> None:
    """Registered feature names align with FEATURE_NAMES without duplicates."""
    registry = FeatureRegistry()
    registry.register_many(_IMPLEMENTED_FEATURES)

    names = registry.names()
    assert len(names) == len(set(names))
    assert set(names) == set(FEATURE_NAMES)

"""Unit tests for the CQROS ML ``DatasetScaler`` implementations."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.ml.dataset import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    MERGED_TRAINING_SCHEMA,
    PRIMARY_KEY_COLUMNS,
    DatasetScaler,
    DatasetScalerError,
    IdentityScaler,
    MinMaxScaler,
    StandardScaler,
)
from cqros.ml.dataset.scaler import IdentityScaler as IdentityScalerDirect

_START = 1_700_000_000_000
_INTERVAL = 3_600_000


def _feature_values(row_count: int, *, value: float = 1.0) -> dict[str, list[float]]:
    """Build increasing float values for every feature column."""
    values: dict[str, list[float]] = {}
    for column_index, column in enumerate(FEATURE_COLUMNS):
        values[column] = [
            value + float(column_index) + float(row_index) for row_index in range(row_count)
        ]
    return values


def _label_values(row_count: int) -> dict[str, list[float] | list[int]]:
    """Build default values for every label column."""
    values: dict[str, list[float] | list[int]] = {}
    for column in LABEL_COLUMNS:
        if column.startswith("direction_"):
            values[column] = [1 if index % 2 == 0 else 0 for index in range(row_count)]
        else:
            values[column] = [0.01 * float(index + 1) for index in range(row_count)]
    return values


def _training_frame(
    *,
    row_count: int = 4,
    symbol: str = "BTCUSDT",
    feature_overrides: dict[str, list[float]] | None = None,
) -> pl.DataFrame:
    """Build a canonical ML dataset for scaler tests."""
    open_times = [_START + index * _INTERVAL for index in range(row_count)]
    data: dict[str, object] = {
        "symbol": [symbol] * row_count,
        "timeframe": ["1h"] * row_count,
        "open_time": open_times,
    }
    data.update(_feature_values(row_count))
    data.update(_label_values(row_count))
    if feature_overrides is not None:
        data.update(feature_overrides)
    frame = pl.DataFrame(data, schema=COLUMN_DTYPES)
    return frame.select(list(CANONICAL_COLUMN_ORDER))


def test_scalers_are_exported_from_package() -> None:
    """Package exports match the scaler module classes."""
    assert IdentityScaler is IdentityScalerDirect
    assert issubclass(IdentityScaler, DatasetScaler)
    assert issubclass(StandardScaler, DatasetScaler)
    assert issubclass(MinMaxScaler, DatasetScaler)


def test_identity_scaler_leaves_frame_unchanged() -> None:
    """IdentityScaler returns an equal copy of the input frame."""
    frame = _training_frame()
    scaler = IdentityScaler()

    result = scaler.fit_transform(frame)

    assert_frame_equal(result, frame)
    assert result.columns == list(CANONICAL_COLUMN_ORDER)


def test_standard_scaling() -> None:
    """StandardScaler centers features to roughly zero mean and unit variance."""
    frame = _training_frame(row_count=5)
    scaler = StandardScaler()

    scaled = scaler.fit_transform(frame)

    for column in FEATURE_COLUMNS:
        series = scaled.get_column(column)
        assert series.mean() == pytest.approx(0.0, abs=1e-9)
        assert series.std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_minmax_scaling() -> None:
    """MinMaxScaler maps each feature into the unit interval."""
    frame = _training_frame(row_count=5)
    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(frame)

    for column in FEATURE_COLUMNS:
        series = scaled.get_column(column)
        assert series.min() == pytest.approx(0.0)
        assert series.max() == pytest.approx(1.0)


def test_fit_transform_workflow() -> None:
    """fit_transform matches sequential fit then transform."""
    frame = _training_frame()
    scaler = StandardScaler()

    combined = scaler.fit_transform(frame)
    sequential = StandardScaler().fit(frame).transform(frame)

    assert_frame_equal(combined, sequential)


def test_fit_transform_reuses_training_parameters() -> None:
    """Validation/test transforms reuse parameters fitted on train only."""
    train = _training_frame(row_count=4, symbol="BTCUSDT")
    test = _training_frame(row_count=2, symbol="ETHUSDT")
    scaler = StandardScaler().fit(train)

    scaled_test = scaler.transform(test)
    train_returns = train.get_column("returns").cast(pl.Float64)
    test_returns = test.get_column("returns").cast(pl.Float64).to_list()
    train_mean = train_returns.mean()
    train_std = train_returns.std(ddof=0)
    assert train_mean is not None
    assert train_std is not None
    mean = float(str(train_mean))
    std = float(str(train_std))
    expected = [(float(str(value)) - mean) / std for value in test_returns]
    assert scaled_test.get_column("returns").to_list() == pytest.approx(expected)


def test_inverse_transform_round_trip() -> None:
    """inverse_transform restores original feature values after scaling."""
    frame = _training_frame(row_count=5)

    for scaler in (StandardScaler(), MinMaxScaler(), IdentityScaler()):
        scaled = scaler.fit_transform(frame)
        restored = scaler.inverse_transform(scaled)
        assert_frame_equal(restored, frame)


def test_constant_feature_columns_standard_scaler() -> None:
    """Zero-variance features scale to 0.0 under StandardScaler."""
    constants = {column: [3.0, 3.0, 3.0] for column in FEATURE_COLUMNS}
    frame = _training_frame(row_count=3, feature_overrides=constants)
    scaler = StandardScaler()

    scaled = scaler.fit_transform(frame)
    restored = scaler.inverse_transform(scaled)

    for column in FEATURE_COLUMNS:
        assert scaled.get_column(column).to_list() == [0.0, 0.0, 0.0]
    assert_frame_equal(restored.select(list(FEATURE_COLUMNS)), frame.select(list(FEATURE_COLUMNS)))


def test_constant_feature_columns_minmax_scaler() -> None:
    """Constant features scale to 0.0 under MinMaxScaler when max equals min."""
    constants = {column: [2.5, 2.5, 2.5] for column in FEATURE_COLUMNS}
    frame = _training_frame(row_count=3, feature_overrides=constants)
    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(frame)
    restored = scaler.inverse_transform(scaled)

    for column in FEATURE_COLUMNS:
        assert scaled.get_column(column).to_list() == [0.0, 0.0, 0.0]
    assert_frame_equal(restored.select(list(FEATURE_COLUMNS)), frame.select(list(FEATURE_COLUMNS)))


def test_transform_before_fit_raises() -> None:
    """transform and inverse_transform require a prior fit."""
    frame = _training_frame()
    scaler = StandardScaler()

    with pytest.raises(DatasetScalerError, match="must be fitted"):
        scaler.transform(frame)
    with pytest.raises(DatasetScalerError, match="must be fitted"):
        scaler.inverse_transform(frame)


def test_empty_datasets_raise() -> None:
    """Empty frames are rejected by fit and transform."""
    empty = pl.DataFrame(schema=MERGED_TRAINING_SCHEMA)
    scaler = StandardScaler()

    with pytest.raises(DatasetScalerError, match="at least one row"):
        scaler.fit(empty)

    scaler.fit(_training_frame())
    with pytest.raises(DatasetScalerError, match="at least one row"):
        scaler.transform(empty)


def test_missing_feature_columns_raise() -> None:
    """Frames missing feature columns are rejected."""
    frame = _training_frame().drop(FEATURE_COLUMNS[0])
    with pytest.raises(DatasetScalerError, match="missing required feature columns"):
        StandardScaler().fit(frame)


def test_labels_remain_unchanged() -> None:
    """Label columns are preserved exactly through scaling."""
    frame = _training_frame()
    original_labels = frame.select(list(LABEL_COLUMNS))

    scaled = StandardScaler().fit_transform(frame)

    assert_frame_equal(scaled.select(list(LABEL_COLUMNS)), original_labels)


def test_primary_keys_remain_unchanged() -> None:
    """Primary-key columns are preserved exactly through scaling."""
    frame = _training_frame()
    original_keys = frame.select(list(PRIMARY_KEY_COLUMNS))

    scaled = MinMaxScaler().fit_transform(frame)

    assert_frame_equal(scaled.select(list(PRIMARY_KEY_COLUMNS)), original_keys)


def test_canonical_column_order_preserved() -> None:
    """Scaled output follows CANONICAL_COLUMN_ORDER."""
    disordered = _training_frame().select([*CANONICAL_COLUMN_ORDER[1:], CANONICAL_COLUMN_ORDER[0]])
    scaled = IdentityScaler().fit_transform(disordered)

    assert scaled.columns == list(CANONICAL_COLUMN_ORDER)


def test_input_immutability() -> None:
    """Caller-supplied frames are never mutated."""
    frame = _training_frame()
    snapshot = frame.clone()

    StandardScaler().fit_transform(frame)

    assert_frame_equal(frame, snapshot)


def test_output_independence() -> None:
    """Mutating a scaled frame does not affect the original or other outputs."""
    frame = _training_frame()
    snapshot = frame.clone()
    scaler = StandardScaler().fit(frame)
    train_scaled = scaler.transform(frame)
    train_snapshot = train_scaled.clone()

    mutated = train_scaled.with_columns(pl.lit(999.0).alias("returns"))

    assert_frame_equal(frame, snapshot)
    assert_frame_equal(train_scaled, train_snapshot)
    assert mutated.get_column("returns").to_list() == [999.0] * frame.height
    mutated_mean = mutated.get_column("returns").mean()
    assert mutated_mean is not None
    assert not math.isclose(float(str(mutated_mean)), 0.0)

"""Unit tests for the CQROS ML ``DatasetSplitter``."""

from __future__ import annotations

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
    DatasetSplitter,
    DatasetSplitterError,
)
from cqros.ml.dataset.splitter import DatasetSplitter as DatasetSplitterDirect

_START = 1_700_000_000_000
_INTERVAL = 3_600_000


def _feature_values(row_count: int, *, value: float = 0.01) -> dict[str, list[float]]:
    """Build default float values for every feature column."""
    values: dict[str, list[float]] = {}
    for column in FEATURE_COLUMNS:
        values[column] = [value + float(index) for index in range(row_count)]
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


def _training_frame(*, row_count: int = 10, symbol: str = "BTCUSDT") -> pl.DataFrame:
    """Build a chronologically ordered canonical ML dataset."""
    open_times = [_START + index * _INTERVAL for index in range(row_count)]
    data: dict[str, object] = {
        "symbol": [symbol] * row_count,
        "timeframe": ["1h"] * row_count,
        "open_time": open_times,
    }
    data.update(_feature_values(row_count))
    data.update(_label_values(row_count))
    frame = pl.DataFrame(data, schema=COLUMN_DTYPES)
    return frame.select(list(CANONICAL_COLUMN_ORDER))


def test_dataset_splitter_is_exported_from_package() -> None:
    """Package export matches the splitter module class."""
    assert DatasetSplitter is DatasetSplitterDirect


def test_split_80_10_10() -> None:
    """Default-style 80/10/10 ratios partition ten rows as 8/1/1."""
    frame = _training_frame(row_count=10)
    splitter = DatasetSplitter()

    train, validation, test = splitter.split(
        frame,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )

    assert train.height == 8
    assert validation.height == 1
    assert test.height == 1
    assert train.get_column("open_time").to_list() == [
        _START + index * _INTERVAL for index in range(8)
    ]
    assert validation.get_column("open_time").to_list() == [_START + 8 * _INTERVAL]
    assert test.get_column("open_time").to_list() == [_START + 9 * _INTERVAL]


def test_split_custom_ratios() -> None:
    """Custom ratios are applied chronologically with remainder on test."""
    frame = _training_frame(row_count=10)
    splitter = DatasetSplitter()

    train, validation, test = splitter.split(
        frame,
        train_ratio=0.5,
        validation_ratio=0.2,
        test_ratio=0.3,
    )

    assert train.height == 5
    assert validation.height == 2
    assert test.height == 3


def test_split_preserves_chronological_ordering() -> None:
    """Each split keeps ascending open_time order and never shuffles."""
    frame = _training_frame(row_count=20)
    train, validation, test = DatasetSplitter().split(
        frame,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
    )

    for part in (train, validation, test):
        times = part.get_column("open_time").to_list()
        assert times == sorted(times)

    assert (
        train.get_column("open_time").to_list()[-1]
        < validation.get_column("open_time").to_list()[0]
    )
    assert (
        validation.get_column("open_time").to_list()[-1] < test.get_column("open_time").to_list()[0]
    )


def test_split_has_no_overlap() -> None:
    """No primary-key row appears in more than one split."""
    frame = _training_frame(row_count=15)
    train, validation, test = DatasetSplitter().split(
        frame,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
    )

    train_keys = set(train.select(list(PRIMARY_KEY_COLUMNS)).iter_rows())
    validation_keys = set(validation.select(list(PRIMARY_KEY_COLUMNS)).iter_rows())
    test_keys = set(test.select(list(PRIMARY_KEY_COLUMNS)).iter_rows())

    assert train_keys.isdisjoint(validation_keys)
    assert train_keys.isdisjoint(test_keys)
    assert validation_keys.isdisjoint(test_keys)


def test_split_covers_all_rows() -> None:
    """Train, validation, and test together cover every input row once."""
    frame = _training_frame(row_count=17)
    train, validation, test = DatasetSplitter().split(
        frame,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )

    combined = pl.concat([train, validation, test], how="vertical")
    assert combined.height == frame.height
    assert_frame_equal(
        combined.sort(list(PRIMARY_KEY_COLUMNS)),
        frame.sort(list(PRIMARY_KEY_COLUMNS)),
    )


def test_split_preserves_dtypes_and_canonical_column_order() -> None:
    """Every split uses MERGED_TRAINING_SCHEMA column order and dtypes."""
    disordered = _training_frame(row_count=10).select(
        [*CANONICAL_COLUMN_ORDER[1:], CANONICAL_COLUMN_ORDER[0]]
    )
    train, validation, test = DatasetSplitter().split(
        disordered,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )

    for part in (train, validation, test):
        assert part.columns == list(CANONICAL_COLUMN_ORDER)
        assert part.schema == MERGED_TRAINING_SCHEMA


def test_split_tiny_dataset() -> None:
    """Tiny datasets remain chronological even when some splits are empty."""
    frame = _training_frame(row_count=2)
    train, validation, test = DatasetSplitter().split(
        frame,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )

    assert train.height == 1
    assert validation.height == 0
    assert test.height == 1
    assert train.get_column("open_time").to_list() == [_START]
    assert test.get_column("open_time").to_list() == [_START + _INTERVAL]
    assert validation.schema == MERGED_TRAINING_SCHEMA


def test_split_single_row_dataset() -> None:
    """A single-row dataset assigns the remainder to the test split."""
    frame = _training_frame(row_count=1)
    train, validation, test = DatasetSplitter().split(
        frame,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )

    assert train.height == 0
    assert validation.height == 0
    assert test.height == 1
    assert_frame_equal(test, frame)


def test_split_rejects_empty_dataset() -> None:
    """Empty input frames raise DatasetSplitterError."""
    empty = pl.DataFrame(schema=MERGED_TRAINING_SCHEMA)
    with pytest.raises(DatasetSplitterError, match="at least one row"):
        DatasetSplitter().split(
            empty,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
        )


def test_split_rejects_invalid_ratios() -> None:
    """Negative, >1, non-numeric, and non-summing ratios are rejected."""
    frame = _training_frame(row_count=10)
    splitter = DatasetSplitter()

    with pytest.raises(DatasetSplitterError, match="train_ratio"):
        splitter.split(
            frame,
            train_ratio=-0.1,
            validation_ratio=0.5,
            test_ratio=0.6,
        )
    with pytest.raises(DatasetSplitterError, match="validation_ratio"):
        splitter.split(
            frame,
            train_ratio=0.2,
            validation_ratio=1.5,
            test_ratio=-0.7,
        )
    with pytest.raises(DatasetSplitterError, match="test_ratio"):
        splitter.split(
            frame,
            train_ratio=0.5,
            validation_ratio=0.5,
            test_ratio=True,  # type: ignore[arg-type]
        )
    with pytest.raises(DatasetSplitterError, match="sum to 1.0"):
        splitter.split(
            frame,
            train_ratio=0.5,
            validation_ratio=0.3,
            test_ratio=0.3,
        )
    with pytest.raises(DatasetSplitterError, match="sum to 1.0"):
        splitter.split(
            frame,
            train_ratio=0.0,
            validation_ratio=0.0,
            test_ratio=0.0,
        )


def test_split_rejects_missing_required_columns() -> None:
    """Frames missing required schema columns are rejected."""
    frame = _training_frame(row_count=5).drop("returns")
    with pytest.raises(DatasetSplitterError, match="missing required columns"):
        DatasetSplitter().split(
            frame,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
        )


def test_split_does_not_mutate_input() -> None:
    """The caller-supplied frame remains unchanged after splitting."""
    frame = _training_frame(row_count=10)
    snapshot = frame.clone()

    DatasetSplitter().split(
        frame,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )

    assert_frame_equal(frame, snapshot)


def test_split_outputs_are_independent() -> None:
    """Mutating one split does not affect the input or other splits."""
    frame = _training_frame(row_count=10)
    snapshot = frame.clone()
    train, validation, test = DatasetSplitter().split(
        frame,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
    )
    validation_snapshot = validation.clone()
    test_snapshot = test.clone()

    mutated_train = train.with_columns(pl.lit("MUTATED").alias("symbol"))

    assert_frame_equal(frame, snapshot)
    assert_frame_equal(validation, validation_snapshot)
    assert_frame_equal(test, test_snapshot)
    assert mutated_train.get_column("symbol").to_list() == ["MUTATED"] * train.height

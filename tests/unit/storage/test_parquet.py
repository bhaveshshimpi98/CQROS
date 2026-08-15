"""Unit tests for CQROS Parquet storage backend."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.exceptions import ValidationError
from cqros.storage import (
    DEFAULT_PARQUET_COMPRESSION,
    DatasetNotFoundError,
    IDataStore,
    ParquetStore,
    StorageError,
    StorageSerializationError,
)
from cqros.storage.exceptions import CorruptedDatasetError
from cqros.storage.parquet import ParquetStore as ParquetStoreDirect


@pytest.fixture
def store() -> ParquetStore:
    """Return a ParquetStore with default ZSTD compression."""
    return ParquetStore()


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    """Return a deterministic sample DataFrame with mixed dtypes."""
    return pl.DataFrame(
        {
            "timestamp_ms": [1_700_000_000_000, 1_700_000_060_000],
            "open": [100.0, 101.5],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "is_valid": [True, False],
        }
    )


def test_parquet_store_is_exported_from_package() -> None:
    """Package export matches the parquet module class."""
    assert ParquetStore is ParquetStoreDirect


def test_parquet_store_satisfies_idata_store_protocol(store: ParquetStore) -> None:
    """ParquetStore structurally satisfies IDataStore."""
    _: IDataStore = store


def test_default_compression_is_zstd(store: ParquetStore) -> None:
    """Default compression codec is ZSTD."""
    assert store.compression == "zstd"
    assert DEFAULT_PARQUET_COMPRESSION == "zstd"


def test_invalid_compression_raises_validation_error() -> None:
    """Unsupported compression codecs fail fast at construction."""
    with pytest.raises(ValidationError) as exc_info:
        ParquetStore(compression="invalid")  # type: ignore[arg-type]

    assert exc_info.value.error_code == "STORAGE-PARQUET-001"


def test_write_and_read_round_trip(
    store: ParquetStore,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """Written frames round-trip with schema and values preserved."""
    path = tmp_path / "raw" / "ohlcv" / "2026.parquet"

    store.write(path, sample_frame)
    loaded = store.read(path)

    assert_frame_equal(loaded, sample_frame)
    assert loaded.schema == sample_frame.schema


def test_write_is_atomic_and_creates_parent_directories(
    store: ParquetStore,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """Write creates parents and leaves no temporary sibling artifacts."""
    path = tmp_path / "nested" / "dir" / "dataset.parquet"

    store.write(path, sample_frame)

    assert path.is_file()
    leftovers = list(path.parent.glob(".*.tmp"))
    assert leftovers == []


def test_write_accepts_string_path(
    store: ParquetStore,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """String paths are accepted for write and read."""
    path = tmp_path / "string_path.parquet"

    store.write(str(path), sample_frame)
    loaded = store.read(str(path))

    assert_frame_equal(loaded, sample_frame)


def test_scan_returns_lazy_frame(
    store: ParquetStore,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """Scan returns a LazyFrame that materializes to the stored data."""
    path = tmp_path / "scan.parquet"
    store.write(path, sample_frame)

    lazy = store.scan(path)

    assert isinstance(lazy, pl.LazyFrame)
    assert_frame_equal(lazy.collect(), sample_frame)


def test_exists_true_for_file_false_for_missing_and_directory(
    store: ParquetStore,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """exists reports only regular files."""
    path = tmp_path / "exists.parquet"
    missing = tmp_path / "missing.parquet"
    directory = tmp_path / "directory"
    directory.mkdir()

    store.write(path, sample_frame)

    assert store.exists(path) is True
    assert store.exists(missing) is False
    assert store.exists(directory) is False


def test_delete_removes_file(
    store: ParquetStore,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """delete removes an existing dataset file."""
    path = tmp_path / "delete.parquet"
    store.write(path, sample_frame)

    store.delete(path)

    assert path.exists() is False
    assert store.exists(path) is False


def test_delete_missing_raises_dataset_not_found(
    store: ParquetStore,
    tmp_path: Path,
) -> None:
    """delete raises DatasetNotFoundError when the file is absent."""
    path = tmp_path / "missing.parquet"

    with pytest.raises(DatasetNotFoundError) as exc_info:
        store.delete(path)

    assert exc_info.value.error_code == "STORAGE-PARQUET-002"
    assert isinstance(exc_info.value, StorageError)


def test_schema_preserves_column_types(
    store: ParquetStore,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """schema returns the stored column names and dtypes."""
    path = tmp_path / "schema.parquet"
    store.write(path, sample_frame)

    schema = store.schema(path)

    assert schema == sample_frame.schema
    assert schema["timestamp_ms"] == pl.Int64
    assert schema["open"] == pl.Float64
    assert schema["symbol"] == pl.String
    assert schema["is_valid"] == pl.Boolean


def test_row_count_matches_frame_height(
    store: ParquetStore,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """row_count returns the number of persisted rows."""
    path = tmp_path / "rows.parquet"
    store.write(path, sample_frame)

    assert store.row_count(path) == sample_frame.height


def test_read_missing_raises_dataset_not_found(
    store: ParquetStore,
    tmp_path: Path,
) -> None:
    """read raises DatasetNotFoundError for a missing path."""
    with pytest.raises(DatasetNotFoundError):
        store.read(tmp_path / "absent.parquet")


def test_scan_missing_raises_dataset_not_found(
    store: ParquetStore,
    tmp_path: Path,
) -> None:
    """scan raises DatasetNotFoundError for a missing path."""
    with pytest.raises(DatasetNotFoundError):
        store.scan(tmp_path / "absent.parquet")


def test_schema_missing_raises_dataset_not_found(
    store: ParquetStore,
    tmp_path: Path,
) -> None:
    """schema raises DatasetNotFoundError for a missing path."""
    with pytest.raises(DatasetNotFoundError):
        store.schema(tmp_path / "absent.parquet")


def test_row_count_missing_raises_dataset_not_found(
    store: ParquetStore,
    tmp_path: Path,
) -> None:
    """row_count raises DatasetNotFoundError for a missing path."""
    with pytest.raises(DatasetNotFoundError):
        store.row_count(tmp_path / "absent.parquet")


def test_read_corrupted_parquet_raises_storage_exception(
    store: ParquetStore,
    tmp_path: Path,
) -> None:
    """Corrupt Parquet content is translated into a CQROS storage exception."""
    path = tmp_path / "corrupt.parquet"
    path.write_bytes(b"not-a-parquet-file")

    with pytest.raises((CorruptedDatasetError, StorageSerializationError)) as exc_info:
        store.read(path)

    assert isinstance(exc_info.value, StorageError)
    assert exc_info.value.error_code in {
        "STORAGE-PARQUET-003",
        "STORAGE-PARQUET-004",
    }


def test_write_failure_cleans_up_temporary_file(
    store: ParquetStore,
    sample_frame: pl.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed writes remove temporary artifacts and raise StorageError."""
    path = tmp_path / "fail.parquet"
    store.write(path, sample_frame)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", _boom)

    with pytest.raises(StorageError) as exc_info:
        store.write(path, sample_frame)

    assert exc_info.value.error_code == "STORAGE-PARQUET-005"
    assert list(path.parent.glob(".*.tmp")) == []
    # Original file remains untouched after failed overwrite attempt.
    assert_frame_equal(store.read(path), sample_frame)


def test_none_compression_maps_to_uncompressed(
    sample_frame: pl.DataFrame,
    tmp_path: Path,
) -> None:
    """Codec alias ``none`` is accepted and persists successfully."""
    store = ParquetStore(compression="none")
    path = tmp_path / "none.parquet"

    store.write(path, sample_frame)

    assert_frame_equal(store.read(path), sample_frame)

"""Unit tests for CQROS market-dataset partition manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from cqros.core.constants import HASH_ALGORITHM_SHA256
from cqros.core.exceptions import IntegrityError, MissingDataError, ValidationError
from cqros.ingestion import (
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ManifestRepository,
    PartitionMetadata,
)
from cqros.ingestion.manifest import (
    DatasetManifest as DatasetManifestDirect,
)
from cqros.ingestion.manifest import (
    ManifestRepository as ManifestRepositoryDirect,
)
from cqros.ingestion.manifest import (
    PartitionMetadata as PartitionMetadataDirect,
)

_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1m"
_DATASET_TYPE = "ohlcv"
_CREATED_AT = "2026-01-01T00:00:00+00:00"
_UPDATED_AT = "2026-07-26T00:00:00+00:00"


def _partition(
    *,
    year: int = 2024,
    filename: str | None = None,
    row_count: int = 10,
    checksum: str = "abc",
    size_bytes: int = 4,
    updated_at: str = _UPDATED_AT,
) -> PartitionMetadata:
    """Build a partition metadata fixture."""
    return PartitionMetadata(
        year=year,
        filename=filename if filename is not None else f"{year}.parquet",
        row_count=row_count,
        start_time_ms=1_704_067_200_000,
        end_time_ms=1_735_689_540_000,
        checksum=checksum,
        size_bytes=size_bytes,
        updated_at=updated_at,
    )


def _manifest(
    *,
    partitions: tuple[PartitionMetadata, ...] = (),
    created_at: str = _CREATED_AT,
    updated_at: str = _UPDATED_AT,
) -> DatasetManifest:
    """Build a dataset manifest fixture."""
    return DatasetManifest(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        dataset_type=_DATASET_TYPE,
        created_at=created_at,
        updated_at=updated_at,
        partitions=partitions,
    )


def _write_partition_file(directory: Path, content: bytes, year: int = 2024) -> Path:
    """Write a fake partition file and return its path."""
    path = directory / f"{year}.parquet"
    path.write_bytes(content)
    return path


def test_partition_metadata_is_frozen_dataclass() -> None:
    """PartitionMetadata is an immutable slotted dataclass."""
    partition = _partition()
    assert is_dataclass(partition)
    with pytest.raises(FrozenInstanceError):
        partition.year = 2025  # type: ignore[misc]


def test_dataset_manifest_is_frozen_dataclass() -> None:
    """DatasetManifest is an immutable slotted dataclass."""
    manifest = _manifest()
    assert is_dataclass(manifest)
    with pytest.raises(FrozenInstanceError):
        manifest.symbol = "ETHUSDT"  # type: ignore[misc]


def test_manifest_types_are_exported_from_package() -> None:
    """Package exports match the manifest module classes and constants."""
    assert PartitionMetadata is PartitionMetadataDirect
    assert DatasetManifest is DatasetManifestDirect
    assert ManifestRepository is ManifestRepositoryDirect
    assert DEFAULT_MANIFEST_FILENAME == "manifest.json"
    assert DEFAULT_MANIFEST_SCHEMA_VERSION == "1.0.0"


def test_partition_metadata_round_trip_dict() -> None:
    """PartitionMetadata serializes to and from a mapping."""
    partition = _partition()
    restored = PartitionMetadata.from_dict(partition.to_dict())
    assert restored == partition
    assert restored.checksum_algorithm == HASH_ALGORITHM_SHA256


def test_dataset_manifest_round_trip_dict_sorts_partitions() -> None:
    """DatasetManifest round-trips and orders partitions by year."""
    manifest = _manifest(partitions=(_partition(year=2025), _partition(year=2024)))
    restored = DatasetManifest.from_dict(manifest.to_dict())
    assert restored.years == (2024, 2025)
    assert restored.total_rows == 20
    assert restored.partition_for_year(2024) is not None
    assert restored.partition_for_year(2030) is None


def test_dataset_manifest_with_partition_upserts_by_year() -> None:
    """with_partition replaces an existing year and keeps sort order."""
    original = _manifest(partitions=(_partition(year=2024, row_count=1),))
    updated = original.with_partition(_partition(year=2025, row_count=2))
    replaced = updated.with_partition(_partition(year=2024, row_count=9))
    assert replaced.years == (2024, 2025)
    year_2024 = replaced.partition_for_year(2024)
    assert year_2024 is not None
    assert year_2024.row_count == 9
    assert replaced.total_rows == 11


def test_dataset_manifest_from_dict_rejects_invalid_partitions() -> None:
    """from_dict raises ValidationError for non-mapping partition entries."""
    payload = _manifest().to_dict()
    payload["partitions"] = ["not-a-mapping"]
    with pytest.raises(ValidationError, match="partition entries must be mappings"):
        DatasetManifest.from_dict(payload)


def test_dataset_manifest_from_dict_rejects_missing_fields() -> None:
    """from_dict raises ValidationError when required fields are absent."""
    with pytest.raises(ValidationError, match="missing required field"):
        DatasetManifest.from_dict({"exchange": _EXCHANGE})


def test_manifest_repository_save_load_round_trip(tmp_path: Path) -> None:
    """save and load round-trip a manifest beside partition files."""
    repository = ManifestRepository(tmp_path)
    manifest = _manifest(partitions=(_partition(year=2024), _partition(year=2023)))

    assert repository.exists() is False
    repository.save(manifest)
    assert repository.exists() is True
    assert repository.path == tmp_path / DEFAULT_MANIFEST_FILENAME

    loaded = repository.load()
    assert loaded.years == (2023, 2024)
    assert loaded.exchange == _EXCHANGE
    assert loaded.dataset_type == _DATASET_TYPE

    raw = json.loads(repository.path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == DEFAULT_MANIFEST_SCHEMA_VERSION
    assert [item["year"] for item in raw["partitions"]] == [2023, 2024]


def test_manifest_repository_load_missing_raises(tmp_path: Path) -> None:
    """load raises MissingDataError when the manifest file is absent."""
    repository = ManifestRepository(tmp_path)
    with pytest.raises(MissingDataError, match="not found"):
        repository.load()


def test_manifest_repository_load_invalid_json_raises(tmp_path: Path) -> None:
    """load raises ValidationError for corrupt JSON."""
    path = tmp_path / DEFAULT_MANIFEST_FILENAME
    path.write_text("{not-json", encoding="utf-8")
    repository = ManifestRepository(tmp_path)
    with pytest.raises(ValidationError, match="not valid JSON"):
        repository.load()


def test_manifest_repository_update_creates_when_absent(tmp_path: Path) -> None:
    """update creates the manifest when none exists yet."""
    repository = ManifestRepository(tmp_path)
    manifest = _manifest(partitions=(_partition(year=2024),))
    merged = repository.update(manifest)
    assert merged.years == (2024,)
    assert repository.load() == merged


def test_manifest_repository_update_merges_partitions(tmp_path: Path) -> None:
    """update upserts partitions by year and preserves created_at."""
    repository = ManifestRepository(tmp_path)
    repository.save(
        _manifest(
            partitions=(_partition(year=2024, row_count=1),),
            created_at=_CREATED_AT,
            updated_at=_CREATED_AT,
        )
    )

    merged = repository.update(
        _manifest(
            partitions=(
                _partition(year=2024, row_count=99),
                _partition(year=2025, row_count=5),
            ),
            created_at="2099-01-01T00:00:00+00:00",
            updated_at=_UPDATED_AT,
        )
    )

    assert merged.created_at == _CREATED_AT
    assert merged.updated_at == _UPDATED_AT
    assert merged.years == (2024, 2025)
    year_2024 = merged.partition_for_year(2024)
    assert year_2024 is not None
    assert year_2024.row_count == 99
    assert merged.total_rows == 104


def test_manifest_repository_verify_success(tmp_path: Path) -> None:
    """verify returns True when partition files match recorded metadata."""
    content = b"partition-bytes"
    partition_path = _write_partition_file(tmp_path, content, year=2024)
    checksum = hashlib.sha256(content).hexdigest()
    partition = _partition(
        year=2024,
        checksum=checksum,
        size_bytes=partition_path.stat().st_size,
    )
    repository = ManifestRepository(tmp_path)
    repository.save(_manifest(partitions=(partition,)))

    assert repository.verify() is True
    assert repository.verify(manifest=_manifest(partitions=(partition,))) is True


def test_manifest_repository_verify_missing_file_raises(tmp_path: Path) -> None:
    """verify raises IntegrityError when a listed partition file is absent."""
    repository = ManifestRepository(tmp_path)
    repository.save(_manifest(partitions=(_partition(year=2024),)))
    with pytest.raises(IntegrityError, match="Partition file missing"):
        repository.verify()


def test_manifest_repository_verify_size_mismatch_raises(tmp_path: Path) -> None:
    """verify raises IntegrityError when on-disk size differs."""
    _write_partition_file(tmp_path, b"abcd", year=2024)
    partition = _partition(year=2024, checksum="x", size_bytes=999)
    repository = ManifestRepository(tmp_path)
    repository.save(_manifest(partitions=(partition,)))
    with pytest.raises(IntegrityError, match="size mismatch"):
        repository.verify()


def test_manifest_repository_verify_checksum_mismatch_raises(tmp_path: Path) -> None:
    """verify raises IntegrityError when the content checksum differs."""
    content = b"abcd"
    path = _write_partition_file(tmp_path, content, year=2024)
    partition = _partition(
        year=2024,
        checksum="0" * 64,
        size_bytes=path.stat().st_size,
    )
    repository = ManifestRepository(tmp_path)
    repository.save(_manifest(partitions=(partition,)))
    with pytest.raises(IntegrityError, match="checksum mismatch"):
        repository.verify()


def test_manifest_repository_verify_unsupported_algorithm_raises(
    tmp_path: Path,
) -> None:
    """verify raises IntegrityError for unsupported checksum algorithms."""
    content = b"abcd"
    path = _write_partition_file(tmp_path, content, year=2024)
    partition = PartitionMetadata(
        year=2024,
        filename="2024.parquet",
        row_count=1,
        start_time_ms=0,
        end_time_ms=1,
        checksum=hashlib.sha256(content).hexdigest(),
        size_bytes=path.stat().st_size,
        updated_at=_UPDATED_AT,
        checksum_algorithm="md5",
    )
    repository = ManifestRepository(tmp_path)
    with pytest.raises(IntegrityError, match="Unsupported partition checksum"):
        repository.verify(manifest=_manifest(partitions=(partition,)))


def test_manifest_repository_custom_filename(tmp_path: Path) -> None:
    """Repositories honor a custom manifest filename."""
    repository = ManifestRepository(tmp_path, filename="dataset.manifest.json")
    repository.save(_manifest())
    assert (tmp_path / "dataset.manifest.json").is_file()
    assert repository.exists() is True
    assert repository.load().symbol == _SYMBOL


def test_manifest_repository_save_is_deterministic(tmp_path: Path) -> None:
    """Repeated saves produce identical JSON for identical manifests."""
    repository = ManifestRepository(tmp_path)
    manifest = _manifest(partitions=(_partition(year=2025), _partition(year=2024)))
    repository.save(manifest)
    first = repository.path.read_text(encoding="utf-8")
    repository.save(manifest)
    second = repository.path.read_text(encoding="utf-8")
    assert first == second
    assert '"year": 2024' in first

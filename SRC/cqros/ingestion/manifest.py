"""CQROS market-dataset partition manifests.

Purpose:
    Persist and retrieve lightweight JSON manifests that describe yearly
    Parquet partitions for a single market dataset.

Responsibilities:
    - Represent immutable ``PartitionMetadata`` and ``DatasetManifest`` values
    - Load, save, update, and verify manifests beside yearly partition files
    - Remain free of downloader, validation, and market-domain business logic

Dependencies:
    Python standard library and ``cqros.core`` constants, exceptions, and
    type aliases.

Public API:
    ``PartitionMetadata``, ``DatasetManifest``, ``ManifestRepository``,
    ``DEFAULT_MANIFEST_FILENAME``, and ``DEFAULT_MANIFEST_SCHEMA_VERSION``.

Notes:
    One manifest is stored per market dataset directory (the directory that
    contains ``{year}.parquet`` partitions). The repository is path-oriented
    so alternate exchanges and storage roots can reuse the same contract
    without embedding venue-specific or backend-specific behavior.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, cast
from uuid import uuid4

from cqros.core.constants import (
    DEFAULT_HASH_ALGORITHM,
    FILE_EXTENSION_JSON,
    HASH_ALGORITHM_SHA256,
)
from cqros.core.exceptions import (
    IntegrityError,
    MissingDataError,
    ValidationError,
)
from cqros.core.types import (
    Exchange,
    FilePath,
    JSONValue,
    Market,
    Symbol,
    Timeframe,
    UnixTimestampMs,
)

__all__ = [
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_MANIFEST_SCHEMA_VERSION",
    "PartitionMetadata",
    "DatasetManifest",
    "ManifestRepository",
]

DEFAULT_MANIFEST_FILENAME: Final[str] = f"manifest{FILE_EXTENSION_JSON}"
DEFAULT_MANIFEST_SCHEMA_VERSION: Final[str] = "1.0.0"

_HASH_CHUNK_SIZE_BYTES: Final[int] = 1024 * 1024
_JSON_INDENT: Final[int] = 2
_JSON_ENCODING: Final[str] = "utf-8"

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PartitionMetadata:
    """Integrity and coverage metadata for one yearly dataset partition.

    Attributes:
        year: Calendar year of the partition.
        filename: Partition filename relative to the dataset directory
            (for example ``2024.parquet``).
        row_count: Number of rows stored in the partition.
        start_time_ms: Inclusive earliest record timestamp (Unix ms, UTC).
        end_time_ms: Inclusive latest record timestamp (Unix ms, UTC).
        checksum: Hex digest of the partition file contents.
        size_bytes: Partition file size in bytes.
        updated_at: UTC timestamp when this partition record was last
            written, as an ISO-8601 string.
        checksum_algorithm: Hash algorithm used for ``checksum``. Defaults
            to SHA-256.
    """

    year: int
    filename: str
    row_count: int
    start_time_ms: UnixTimestampMs
    end_time_ms: UnixTimestampMs
    checksum: str
    size_bytes: int
    updated_at: str
    checksum_algorithm: str = HASH_ALGORITHM_SHA256

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize this partition record to a JSON-compatible mapping.

        Returns:
            Mapping suitable for JSON encoding.
        """
        return {
            "year": self.year,
            "filename": self.filename,
            "row_count": self.row_count,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "updated_at": self.updated_at,
            "checksum_algorithm": self.checksum_algorithm,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PartitionMetadata:
        """Deserialize a partition record from a mapping.

        Args:
            payload: Mapping produced by ``to_dict`` or equivalent JSON.

        Returns:
            Immutable ``PartitionMetadata`` instance.

        Raises:
            ValidationError: If required fields are missing or mistyped.
        """
        return cls(
            year=_require_int(payload, "year"),
            filename=_require_str(payload, "filename"),
            row_count=_require_int(payload, "row_count"),
            start_time_ms=_require_int(payload, "start_time_ms"),
            end_time_ms=_require_int(payload, "end_time_ms"),
            checksum=_require_str(payload, "checksum"),
            size_bytes=_require_int(payload, "size_bytes"),
            updated_at=_require_str(payload, "updated_at"),
            checksum_algorithm=_optional_str(
                payload,
                "checksum_algorithm",
                default=HASH_ALGORITHM_SHA256,
            ),
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Manifest describing all yearly partitions of one market dataset.

    A market dataset is identified by exchange, market, symbol, timeframe,
    and dataset type. Partition records are stored as an immutable tuple
    ordered by ascending year when constructed through repository writes.

    Attributes:
        exchange: Exchange identifier (for example ``binance``).
        market: Market segment (for example ``perpetual``).
        symbol: Tradeable symbol (for example ``BTCUSDT``).
        timeframe: Bar or sampling interval (for example ``1m``).
        dataset_type: Dataset kind (for example ``ohlcv`` or ``funding``).
        created_at: UTC creation timestamp as an ISO-8601 string.
        updated_at: UTC last-update timestamp as an ISO-8601 string.
        partitions: Immutable sequence of partition metadata records.
        schema_version: Manifest schema version string.
    """

    exchange: Exchange
    market: Market
    symbol: Symbol
    timeframe: Timeframe
    dataset_type: str
    created_at: str
    updated_at: str
    partitions: tuple[PartitionMetadata, ...] = ()
    schema_version: str = DEFAULT_MANIFEST_SCHEMA_VERSION

    @property
    def total_rows(self) -> int:
        """Return the sum of ``row_count`` across all partitions."""
        return sum(partition.row_count for partition in self.partitions)

    @property
    def years(self) -> tuple[int, ...]:
        """Return partition years in ascending order."""
        return tuple(partition.year for partition in self.partitions)

    def partition_for_year(self, year: int) -> PartitionMetadata | None:
        """Return the partition record for ``year``, if present.

        Args:
            year: Calendar year to look up.

        Returns:
            Matching ``PartitionMetadata``, or ``None`` when absent.
        """
        for partition in self.partitions:
            if partition.year == year:
                return partition
        return None

    def with_partition(self, partition: PartitionMetadata) -> DatasetManifest:
        """Return a copy with ``partition`` upserted by year.

        Args:
            partition: Partition metadata to insert or replace.

        Returns:
            New manifest with partitions sorted by ascending year.
        """
        by_year = {item.year: item for item in self.partitions}
        by_year[partition.year] = partition
        return replace(
            self,
            partitions=_sorted_partitions(tuple(by_year.values())),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize this manifest to a JSON-compatible mapping.

        Returns:
            Mapping suitable for JSON encoding.
        """
        return {
            "schema_version": self.schema_version,
            "exchange": self.exchange,
            "market": self.market,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "dataset_type": self.dataset_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "partitions": [partition.to_dict() for partition in self.partitions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DatasetManifest:
        """Deserialize a dataset manifest from a mapping.

        Args:
            payload: Mapping produced by ``to_dict`` or equivalent JSON.

        Returns:
            Immutable ``DatasetManifest`` instance.

        Raises:
            ValidationError: If required fields are missing or mistyped.
        """
        raw_partitions = payload.get("partitions", ())
        if not isinstance(raw_partitions, Sequence) or isinstance(
            raw_partitions,
            (str, bytes, bytearray),
        ):
            raise ValidationError(
                "Manifest field 'partitions' must be a sequence",
                details={"field": "partitions"},
            )

        partition_items = cast(Sequence[object], raw_partitions)
        partitions: list[PartitionMetadata] = []
        for index, item in enumerate(partition_items):
            if not isinstance(item, Mapping):
                raise ValidationError(
                    "Manifest partition entries must be mappings",
                    details={"field": "partitions", "index": index},
                )
            partitions.append(PartitionMetadata.from_dict(cast(Mapping[str, object], item)))

        return cls(
            exchange=_require_str(payload, "exchange"),
            market=_require_str(payload, "market"),
            symbol=_require_str(payload, "symbol"),
            timeframe=_require_str(payload, "timeframe"),
            dataset_type=_require_str(payload, "dataset_type"),
            created_at=_require_str(payload, "created_at"),
            updated_at=_require_str(payload, "updated_at"),
            partitions=_sorted_partitions(partitions),
            schema_version=_optional_str(
                payload,
                "schema_version",
                default=DEFAULT_MANIFEST_SCHEMA_VERSION,
            ),
        )


class ManifestRepository:
    """Filesystem repository for a single market-dataset manifest.

    Manifests are stored as JSON beside yearly Parquet partitions under
    ``dataset_dir``. Callers supply the dataset directory; this repository
    never composes exchange-specific storage layouts.

    Args:
        dataset_dir: Directory containing yearly partition files and the
            manifest JSON file.
        filename: Manifest filename within ``dataset_dir``. Defaults to
            ``manifest.json``.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_dataset_dir", "_logger", "_manifest_path")

    _dataset_dir: Path
    _manifest_path: Path
    _logger: logging.Logger

    def __init__(
        self,
        dataset_dir: FilePath,
        *,
        filename: str = DEFAULT_MANIFEST_FILENAME,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the repository for a dataset directory.

        Args:
            dataset_dir: Directory that holds partition files and the
                manifest.
            filename: Manifest filename within ``dataset_dir``.
            logger: Optional logger instance.
        """
        self._dataset_dir = Path(dataset_dir)
        self._manifest_path = self._dataset_dir / filename
        self._logger = logger if logger is not None else _logger

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory managed by this repository."""
        return self._dataset_dir

    @property
    def path(self) -> Path:
        """Return the absolute-or-relative path of the manifest file."""
        return self._manifest_path

    def exists(self) -> bool:
        """Return whether the manifest file exists.

        Returns:
            ``True`` when a regular file exists at the manifest path.
        """
        return self._manifest_path.is_file()

    def load(self) -> DatasetManifest:
        """Load and deserialize the dataset manifest.

        Returns:
            Immutable ``DatasetManifest`` parsed from JSON.

        Raises:
            MissingDataError: If the manifest file does not exist.
            ValidationError: If the JSON payload is malformed or incomplete.
        """
        if not self.exists():
            raise MissingDataError(
                f"Dataset manifest not found: {self._manifest_path}",
                details={"path": str(self._manifest_path)},
            )

        self._logger.debug(
            "Loading dataset manifest",
            extra={"path": str(self._manifest_path)},
        )

        try:
            text = self._manifest_path.read_text(encoding=_JSON_ENCODING)
            payload = json.loads(text)
        except OSError as exc:
            raise MissingDataError(
                f"Failed to read dataset manifest: {self._manifest_path}",
                details={"path": str(self._manifest_path)},
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"Dataset manifest is not valid JSON: {self._manifest_path}",
                details={
                    "path": str(self._manifest_path),
                    "error": str(exc),
                },
            ) from exc

        if not isinstance(payload, dict):
            raise ValidationError(
                "Dataset manifest root must be a JSON object",
                details={"path": str(self._manifest_path)},
            )

        manifest = DatasetManifest.from_dict(cast(dict[str, object], payload))
        self._logger.info(
            "Loaded dataset manifest",
            extra={
                "path": str(self._manifest_path),
                "partitions": len(manifest.partitions),
                "total_rows": manifest.total_rows,
            },
        )
        return manifest

    def save(self, manifest: DatasetManifest) -> None:
        """Atomically persist ``manifest`` as JSON beside the partitions.

        Parent directories are created when missing. The payload is written to
        a temporary sibling file and renamed into place so readers never
        observe a partial manifest.

        Args:
            manifest: Manifest to persist. Partitions are stored sorted by
                ascending year.

        Raises:
            OSError: If the filesystem write or rename fails.
        """
        ordered = replace(
            manifest,
            partitions=_sorted_partitions(manifest.partitions),
        )
        payload = ordered.to_dict()

        self._logger.debug(
            "Saving dataset manifest",
            extra={
                "path": str(self._manifest_path),
                "partitions": len(ordered.partitions),
            },
        )

        self._atomic_write(payload)

        self._logger.info(
            "Saved dataset manifest",
            extra={
                "path": str(self._manifest_path),
                "partitions": len(ordered.partitions),
                "total_rows": ordered.total_rows,
            },
        )

    def update(self, manifest: DatasetManifest) -> DatasetManifest:
        """Merge ``manifest`` into the stored manifest and persist the result.

        When no manifest exists yet, ``manifest`` is saved as-is. Otherwise
        partition records are upserted by year (incoming values win),
        dataset-level fields are taken from ``manifest``, and ``created_at``
        from the existing manifest is preserved.

        Args:
            manifest: Manifest providing identity fields, timestamps, and
                partition records to merge.

        Returns:
            The merged manifest that was written to disk.
        """
        if not self.exists():
            self.save(manifest)
            return replace(
                manifest,
                partitions=_sorted_partitions(manifest.partitions),
            )

        existing = self.load()
        by_year = {partition.year: partition for partition in existing.partitions}
        for partition in manifest.partitions:
            by_year[partition.year] = partition

        merged = DatasetManifest(
            exchange=manifest.exchange,
            market=manifest.market,
            symbol=manifest.symbol,
            timeframe=manifest.timeframe,
            dataset_type=manifest.dataset_type,
            created_at=existing.created_at,
            updated_at=manifest.updated_at,
            partitions=_sorted_partitions(tuple(by_year.values())),
            schema_version=manifest.schema_version,
        )
        self.save(merged)

        self._logger.info(
            "Updated dataset manifest",
            extra={
                "path": str(self._manifest_path),
                "partitions": len(merged.partitions),
                "total_rows": merged.total_rows,
            },
        )
        return merged

    def verify(self, *, manifest: DatasetManifest | None = None) -> bool:
        """Verify partition files against manifest integrity metadata.

        For each listed partition, confirms that the file exists beside the
        manifest, that ``size_bytes`` matches the on-disk size, and that the
        content checksum matches when the recorded algorithm is SHA-256.

        Args:
            manifest: Manifest to verify. When omitted, the stored manifest
                is loaded first.

        Returns:
            ``True`` when every listed partition passes integrity checks.

        Raises:
            MissingDataError: If the stored manifest is required but absent.
            ValidationError: If the stored manifest cannot be parsed.
            IntegrityError: If a partition file is missing, the size differs,
                the checksum algorithm is unsupported, or the checksum does
                not match.
        """
        resolved = manifest if manifest is not None else self.load()

        self._logger.debug(
            "Verifying dataset manifest",
            extra={
                "path": str(self._manifest_path),
                "partitions": len(resolved.partitions),
            },
        )

        for partition in resolved.partitions:
            partition_path = self._dataset_dir / partition.filename
            if not partition_path.is_file():
                raise IntegrityError(
                    f"Partition file missing for year {partition.year}: " f"{partition_path}",
                    details={
                        "path": str(partition_path),
                        "year": partition.year,
                        "filename": partition.filename,
                    },
                )

            actual_size = partition_path.stat().st_size
            if actual_size != partition.size_bytes:
                raise IntegrityError(
                    f"Partition size mismatch for year {partition.year}: "
                    f"expected {partition.size_bytes}, found {actual_size}",
                    details={
                        "path": str(partition_path),
                        "year": partition.year,
                        "expected_size_bytes": partition.size_bytes,
                        "actual_size_bytes": actual_size,
                    },
                )

            algorithm = partition.checksum_algorithm
            if algorithm not in {HASH_ALGORITHM_SHA256, DEFAULT_HASH_ALGORITHM}:
                raise IntegrityError(
                    f"Unsupported partition checksum algorithm: {algorithm}",
                    details={
                        "path": str(partition_path),
                        "year": partition.year,
                        "checksum_algorithm": algorithm,
                    },
                )

            actual_checksum = _sha256_file(partition_path)
            if actual_checksum != partition.checksum:
                raise IntegrityError(
                    f"Partition checksum mismatch for year {partition.year}",
                    details={
                        "path": str(partition_path),
                        "year": partition.year,
                        "expected_checksum": partition.checksum,
                        "actual_checksum": actual_checksum,
                        "checksum_algorithm": algorithm,
                    },
                )

        self._logger.info(
            "Verified dataset manifest",
            extra={
                "path": str(self._manifest_path),
                "partitions": len(resolved.partitions),
            },
        )
        return True

    def _atomic_write(self, payload: Mapping[str, JSONValue]) -> None:
        """Write JSON to the manifest path via a temporary sibling file.

        Args:
            payload: JSON-serializable mapping to persist.

        Raises:
            OSError: If directory creation, writing, or renaming fails.
        """
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._manifest_path.with_name(f".{self._manifest_path.name}.{uuid4().hex}.tmp")
        try:
            text = json.dumps(
                payload,
                indent=_JSON_INDENT,
                sort_keys=True,
                ensure_ascii=False,
            )
            temporary.write_text(f"{text}\n", encoding=_JSON_ENCODING)
            os.replace(temporary, self._manifest_path)
        except Exception:
            _cleanup_temporary(temporary)
            raise


def _sorted_partitions(
    partitions: Iterable[PartitionMetadata],
) -> tuple[PartitionMetadata, ...]:
    """Return partitions sorted by ascending year.

    Args:
        partitions: Partition records to order.

    Returns:
        Immutable tuple ordered by ``year``.
    """
    return tuple(sorted(partitions, key=lambda item: item.year))


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _cleanup_temporary(path: Path) -> None:
    """Best-effort removal of a temporary manifest file.

    Args:
        path: Temporary file path to delete when present.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        _logger.warning(
            "Failed to remove temporary manifest file",
            extra={"path": str(path)},
        )


def _require_str(payload: Mapping[str, object], field: str) -> str:
    """Return a required string field from ``payload``.

    Args:
        payload: Source mapping.
        field: Field name to read.

    Returns:
        Non-empty string value when present and typed correctly. Empty
        strings are accepted as valid values.

    Raises:
        ValidationError: If the field is missing or not a ``str``.
    """
    if field not in payload:
        raise ValidationError(
            f"Manifest missing required field '{field}'",
            details={"field": field},
        )
    value = payload[field]
    if not isinstance(value, str):
        raise ValidationError(
            f"Manifest field '{field}' must be a string",
            details={"field": field, "type": type(value).__name__},
        )
    return value


def _require_int(payload: Mapping[str, object], field: str) -> int:
    """Return a required integer field from ``payload``.

    Args:
        payload: Source mapping.
        field: Field name to read.

    Returns:
        Integer value. Booleans are rejected.

    Raises:
        ValidationError: If the field is missing or not an ``int``.
    """
    if field not in payload:
        raise ValidationError(
            f"Manifest missing required field '{field}'",
            details={"field": field},
        )
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            f"Manifest field '{field}' must be an integer",
            details={"field": field, "type": type(value).__name__},
        )
    return value


def _optional_str(
    payload: Mapping[str, object],
    field: str,
    *,
    default: str,
) -> str:
    """Return an optional string field, falling back to ``default``.

    Args:
        payload: Source mapping.
        field: Field name to read.
        default: Value used when the field is absent.

    Returns:
        String value or ``default``.

    Raises:
        ValidationError: If the field is present but not a ``str``.
    """
    if field not in payload:
        return default
    value = payload[field]
    if not isinstance(value, str):
        raise ValidationError(
            f"Manifest field '{field}' must be a string",
            details={"field": field, "type": type(value).__name__},
        )
    return value

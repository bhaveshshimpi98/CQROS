"""CQROS regression threshold repository.

Purpose:
    Provide a path-free facade for persisting and retrieving production-approved
    regression signal thresholds used by adaptive signal generation.

Responsibilities:
    - Resolve storage locations via ``StorageLayout.threshold_path``
    - Persist, load, check existence, and discover threshold partitions
    - Validate canonical threshold schema, finiteness, ordering, and uniqueness
    - Delegate read and write operations to an injected ``IDataStore``
    - Keep filesystem paths out of the public API
    - Remain free of threshold estimation, calibration, and signal generation

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.storage`` layout/interfaces/exceptions,
    and ``cqros.storage.threshold_schema``.

Public API:
    ``ThresholdPartitionRef``, ``ThresholdRepository``

Partition layout::

    thresholds/{model_name}/{model_version}/{symbol}/{timeframe}/thresholds.parquet
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import FILE_EXTENSION_PARQUET, STORAGE_DIR_THRESHOLDS
from cqros.core.types import Symbol, Timeframe
from cqros.storage.exceptions import (
    CorruptedDatasetError,
    DatasetNotFoundError,
    StorageError,
)
from cqros.storage.interfaces import IDataStore
from cqros.storage.layout import StorageLayout
from cqros.storage.threshold_schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    THRESHOLD_SCHEMA,
)

__all__ = [
    "ThresholdPartitionRef",
    "ThresholdRepository",
]

_logger = logging.getLogger(__name__)

_THRESHOLDS_FILENAME: Final[str] = f"thresholds{FILE_EXTENSION_PARQUET}"

_ERROR_FRAME_TYPE: Final[str] = "STORAGE-THR-001"
_ERROR_FRAME_EMPTY: Final[str] = "STORAGE-THR-002"
_ERROR_MISSING_COLUMNS: Final[str] = "STORAGE-THR-003"
_ERROR_DTYPE_MISMATCH: Final[str] = "STORAGE-THR-004"
_ERROR_THRESHOLD_ORDER: Final[str] = "STORAGE-THR-005"
_ERROR_THRESHOLD_FINITE: Final[str] = "STORAGE-THR-006"
_ERROR_DUPLICATE_KEYS: Final[str] = "STORAGE-THR-007"
_ERROR_PARTITION_MISMATCH: Final[str] = "STORAGE-THR-008"
_ERROR_NAME_BLANK: Final[str] = "STORAGE-THR-009"


@dataclass(frozen=True, slots=True)
class ThresholdPartitionRef:
    """Identity of one discovered regression threshold partition.

    Attributes:
        model_name: Stable model identifier.
        model_version: Model version identifier.
        symbol: Tradeable symbol.
        timeframe: Threshold bar interval.
    """

    model_name: str
    model_version: str
    symbol: Symbol
    timeframe: Timeframe


class ThresholdRepository:
    """Repository facade for canonical regression threshold datasets.

    Callers identify partitions by model name, model version, symbol, and
    timeframe. Paths are composed privately via ``StorageLayout.threshold_path``
    and never returned. Persistence is delegated to the injected ``IDataStore``.
    Frames are validated against ``THRESHOLD_SCHEMA`` before save and after
    load.

    Args:
        layout: Canonical path composer for the data lake.
        datastore: Storage backend implementing ``IDataStore``.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = ("_datastore", "_layout", "_logger")

    _layout: StorageLayout
    _datastore: IDataStore
    _logger: logging.Logger

    def __init__(
        self,
        layout: StorageLayout,
        datastore: IDataStore,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the repository with injected layout and datastore.

        Args:
            layout: Canonical path composer for the data lake.
            datastore: Storage backend used for all persistence operations.
            logger: Optional logger instance.
        """
        self._layout = layout
        self._datastore = datastore
        self._logger = logger if logger is not None else _logger

    def save(
        self,
        dataframe: pl.DataFrame,
        *,
        model_name: str,
        model_version: str,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> None:
        """Persist a validated threshold partition.

        Args:
            dataframe: Threshold frame to store.
            model_name: Stable model identifier.
            model_version: Model version identifier.
            symbol: Tradeable symbol.
            timeframe: Threshold bar interval.

        Raises:
            StorageError: If the frame fails schema, finiteness, ordering,
                uniqueness, or partition-identity validation.
        """
        validated_model_name = _require_name(model_name, parameter="model_name")
        validated_model_version = _require_name(model_version, parameter="model_version")
        validated_symbol = _require_name(symbol, parameter="symbol")
        validated_timeframe = _require_name(timeframe, parameter="timeframe")
        finalized = _finalize_threshold_frame(
            dataframe,
            model_name=validated_model_name,
            model_version=validated_model_version,
            symbol=validated_symbol,
            timeframe=validated_timeframe,
        )
        path = self._layout.threshold_path(
            validated_model_name,
            validated_model_version,
            validated_symbol,
            validated_timeframe,
        )
        self._logger.debug(
            "Saving threshold dataset",
            extra=_threshold_log_extra(
                model_name=validated_model_name,
                model_version=validated_model_version,
                symbol=validated_symbol,
                timeframe=validated_timeframe,
                rows=finalized.height,
                columns=finalized.width,
            ),
        )
        self._datastore.write(path, finalized)
        self._logger.info(
            "Saved threshold dataset",
            extra=_threshold_log_extra(
                model_name=validated_model_name,
                model_version=validated_model_version,
                symbol=validated_symbol,
                timeframe=validated_timeframe,
                rows=finalized.height,
                columns=finalized.width,
            ),
        )

    def load(
        self,
        *,
        model_name: str,
        model_version: str,
        symbol: Symbol,
        timeframe: Timeframe,
        profile: str | None = None,
    ) -> pl.DataFrame:
        """Load and validate a threshold partition.

        Args:
            model_name: Stable model identifier.
            model_version: Model version identifier.
            symbol: Tradeable symbol.
            timeframe: Threshold bar interval.
            profile: Optional profile filter. When set, only matching rows are
                returned.

        Returns:
            Canonical threshold DataFrame ordered to ``CANONICAL_COLUMN_ORDER``.

        Raises:
            DatasetNotFoundError: If the partition does not exist.
            CorruptedDatasetError: If stored contents fail validation.
            StorageError: If identity arguments are blank.
        """
        validated_model_name = _require_name(model_name, parameter="model_name")
        validated_model_version = _require_name(model_version, parameter="model_version")
        validated_symbol = _require_name(symbol, parameter="symbol")
        validated_timeframe = _require_name(timeframe, parameter="timeframe")
        path = self._layout.threshold_path(
            validated_model_name,
            validated_model_version,
            validated_symbol,
            validated_timeframe,
        )
        try:
            frame = self._datastore.read(path)
        except DatasetNotFoundError:
            raise
        except Exception as exc:
            raise DatasetNotFoundError(
                "threshold dataset not found",
                error_code="STORAGE-THR-404",
                details={
                    "model_name": validated_model_name,
                    "model_version": validated_model_version,
                    "symbol": validated_symbol,
                    "timeframe": validated_timeframe,
                    "cause_type": type(exc).__name__,
                    "cause": str(exc),
                },
            ) from exc

        try:
            finalized = _finalize_threshold_frame(
                frame,
                model_name=validated_model_name,
                model_version=validated_model_version,
                symbol=validated_symbol,
                timeframe=validated_timeframe,
            )
        except StorageError as exc:
            raise CorruptedDatasetError(
                "stored threshold dataset failed validation",
                error_code="STORAGE-THR-CORRUPT",
                details={
                    "model_name": validated_model_name,
                    "model_version": validated_model_version,
                    "symbol": validated_symbol,
                    "timeframe": validated_timeframe,
                    "cause_code": exc.error_code,
                    "cause": str(exc),
                },
            ) from exc

        if profile is not None:
            validated_profile = _require_name(profile, parameter="profile")
            finalized = finalized.filter(pl.col("profile") == validated_profile)

        self._logger.debug(
            "Loaded threshold dataset",
            extra=_threshold_log_extra(
                model_name=validated_model_name,
                model_version=validated_model_version,
                symbol=validated_symbol,
                timeframe=validated_timeframe,
                rows=finalized.height,
                columns=finalized.width,
                profile=profile,
            ),
        )
        return finalized

    def exists(
        self,
        *,
        model_name: str,
        model_version: str,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> bool:
        """Return whether a threshold partition exists.

        Args:
            model_name: Stable model identifier.
            model_version: Model version identifier.
            symbol: Tradeable symbol.
            timeframe: Threshold bar interval.

        Returns:
            ``True`` when the partition exists; otherwise ``False``.
        """
        path = self._layout.threshold_path(
            _require_name(model_name, parameter="model_name"),
            _require_name(model_version, parameter="model_version"),
            _require_name(symbol, parameter="symbol"),
            _require_name(timeframe, parameter="timeframe"),
        )
        present = self._datastore.exists(path)
        self._logger.debug(
            "Threshold dataset exists" if present else "Threshold dataset does not exist",
            extra=_threshold_log_extra(
                model_name=model_name,
                model_version=model_version,
                symbol=symbol,
                timeframe=timeframe,
            ),
        )
        return present

    def discover(
        self,
        *,
        symbols: Sequence[Symbol] | None = None,
        timeframes: Sequence[Timeframe] | None = None,
        model_names: Sequence[str] | None = None,
        model_versions: Sequence[str] | None = None,
        profiles: Sequence[str] | None = None,
    ) -> tuple[ThresholdPartitionRef, ...]:
        """Discover threshold partitions matching optional filters.

        Args:
            symbols: Optional symbol allowlist.
            timeframes: Optional timeframe allowlist.
            model_names: Optional model-name allowlist.
            model_versions: Optional model-version allowlist.
            profiles: Optional profile allowlist. When set, partitions are
                retained only when their stored frame contains at least one
                matching profile row.

        Returns:
            Deterministically ordered partition references.
        """
        symbol_filter = set(symbols) if symbols is not None else None
        timeframe_filter = set(timeframes) if timeframes is not None else None
        model_name_filter = set(model_names) if model_names is not None else None
        model_version_filter = set(model_versions) if model_versions is not None else None
        profile_filter = set(profiles) if profiles is not None else None

        items: list[ThresholdPartitionRef] = []
        for model_name in self._discover_model_names():
            if model_name_filter is not None and model_name not in model_name_filter:
                continue
            for model_version in self._discover_model_versions(model_name=model_name):
                if model_version_filter is not None and model_version not in model_version_filter:
                    continue
                for symbol in self._discover_symbols(
                    model_name=model_name,
                    model_version=model_version,
                ):
                    if symbol_filter is not None and symbol not in symbol_filter:
                        continue
                    for timeframe in self._discover_timeframes(
                        model_name=model_name,
                        model_version=model_version,
                        symbol=symbol,
                    ):
                        if timeframe_filter is not None and timeframe not in timeframe_filter:
                            continue
                        if not self.exists(
                            model_name=model_name,
                            model_version=model_version,
                            symbol=symbol,
                            timeframe=timeframe,
                        ):
                            continue
                        if profile_filter is not None:
                            frame = self.load(
                                model_name=model_name,
                                model_version=model_version,
                                symbol=symbol,
                                timeframe=timeframe,
                            )
                            present_profiles = set(frame.get_column("profile").to_list())
                            if present_profiles.isdisjoint(profile_filter):
                                continue
                        items.append(
                            ThresholdPartitionRef(
                                model_name=model_name,
                                model_version=model_version,
                                symbol=symbol,
                                timeframe=timeframe,
                            )
                        )

        return tuple(
            sorted(
                items,
                key=lambda item: (
                    item.model_name,
                    item.model_version,
                    item.symbol,
                    item.timeframe,
                ),
            )
        )

    def _thresholds_root(self) -> Path:
        """Return the thresholds tier root directory."""
        return self._layout.root / STORAGE_DIR_THRESHOLDS

    def _discover_model_names(self) -> tuple[str, ...]:
        """Return sorted model names under the thresholds tier."""
        base = self._thresholds_root()
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def _discover_model_versions(self, *, model_name: str) -> tuple[str, ...]:
        """Return sorted model versions for ``model_name``."""
        base = self._thresholds_root() / model_name
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def _discover_symbols(
        self,
        *,
        model_name: str,
        model_version: str,
    ) -> tuple[Symbol, ...]:
        """Return sorted symbols for one model identity."""
        base = self._thresholds_root() / model_name / model_version
        if not base.is_dir():
            return ()
        return tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))

    def _discover_timeframes(
        self,
        *,
        model_name: str,
        model_version: str,
        symbol: Symbol,
    ) -> tuple[Timeframe, ...]:
        """Return sorted timeframes that contain ``thresholds.parquet``."""
        base = self._thresholds_root() / model_name / model_version / symbol
        if not base.is_dir():
            return ()
        return tuple(
            sorted(
                path.name
                for path in base.iterdir()
                if path.is_dir() and (path / _THRESHOLDS_FILENAME).exists()
            )
        )


def _finalize_threshold_frame(
    dataframe: object,
    *,
    model_name: str,
    model_version: str,
    symbol: str,
    timeframe: str,
) -> pl.DataFrame:
    """Validate, order, and cast a threshold frame to the canonical schema.

    Args:
        dataframe: Candidate threshold dataset.
        model_name: Expected model name for all rows.
        model_version: Expected model version for all rows.
        symbol: Expected symbol for all rows.
        timeframe: Expected timeframe for all rows.

    Returns:
        Canonical threshold DataFrame.

    Raises:
        StorageError: If validation fails.
    """
    frame = _require_threshold_frame(dataframe)
    _require_partition_identity(
        frame,
        model_name=model_name,
        model_version=model_version,
        symbol=symbol,
        timeframe=timeframe,
    )
    _require_finite_thresholds(frame)
    _require_threshold_order(frame)
    _require_unique_primary_keys(frame)
    ordered = frame.select(list(CANONICAL_COLUMN_ORDER))
    return ordered.cast(THRESHOLD_SCHEMA)


def _require_threshold_frame(dataframe: object) -> pl.DataFrame:
    """Validate structural shape of a threshold DataFrame."""
    if not isinstance(dataframe, pl.DataFrame):
        raise StorageError(
            "thresholds must be a polars DataFrame",
            error_code=_ERROR_FRAME_TYPE,
            details={"actual_type": type(dataframe).__name__},
        )
    if dataframe.height == 0:
        raise StorageError(
            "thresholds must contain at least one row",
            error_code=_ERROR_FRAME_EMPTY,
            details={"rows": dataframe.height},
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise StorageError(
            "threshold frame is missing required columns",
            error_code=_ERROR_MISSING_COLUMNS,
            details={
                "missing_columns": tuple(missing),
                "required_columns": REQUIRED_COLUMNS,
                "available_columns": tuple(dataframe.columns),
            },
        )

    mismatched: list[dict[str, object]] = []
    for column in REQUIRED_COLUMNS:
        expected = COLUMN_DTYPES[column]
        actual = dataframe.schema[column]
        if actual != expected:
            mismatched.append(
                {
                    "column": column,
                    "expected": str(expected),
                    "actual": str(actual),
                }
            )
    if mismatched:
        raise StorageError(
            "threshold frame dtype mismatch",
            error_code=_ERROR_DTYPE_MISMATCH,
            details={
                "mismatched_columns": tuple(item["column"] for item in mismatched),
                "mismatches": tuple(mismatched),
            },
        )
    return dataframe


def _require_partition_identity(
    frame: pl.DataFrame,
    *,
    model_name: str,
    model_version: str,
    symbol: str,
    timeframe: str,
) -> None:
    """Require every row to match the destination partition identity."""
    mismatches: list[str] = []
    checks = (
        ("symbol", symbol),
        ("timeframe", timeframe),
        ("model_name", model_name),
        ("model_version", model_version),
    )
    for column, expected in checks:
        values = set(frame.get_column(column).to_list())
        if values != {expected}:
            mismatches.append(column)
    if mismatches:
        raise StorageError(
            "threshold frame partition identity mismatch",
            error_code=_ERROR_PARTITION_MISMATCH,
            details={
                "mismatched_columns": tuple(mismatches),
                "expected": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "model_name": model_name,
                    "model_version": model_version,
                },
            },
        )


def _require_finite_thresholds(frame: pl.DataFrame) -> None:
    """Reject null, NaN, or non-finite buy/sell threshold values."""
    invalid_count = int(
        frame.select(
            (
                pl.col("buy_threshold").is_null()
                | pl.col("buy_threshold").is_nan()
                | pl.col("buy_threshold").is_infinite()
                | pl.col("sell_threshold").is_null()
                | pl.col("sell_threshold").is_nan()
                | pl.col("sell_threshold").is_infinite()
            )
            .sum()
            .alias("invalid_count")
        ).item()
    )
    if invalid_count > 0:
        raise StorageError(
            "threshold values must be finite and non-null",
            error_code=_ERROR_THRESHOLD_FINITE,
            details={
                "invalid_rows": invalid_count,
                "row_count": frame.height,
            },
        )


def _require_threshold_order(frame: pl.DataFrame) -> None:
    """Require ``buy_threshold > sell_threshold`` for every row."""
    invalid_count = int(
        frame.select(
            (pl.col("buy_threshold") <= pl.col("sell_threshold")).sum().alias("invalid_count")
        ).item()
    )
    if invalid_count > 0:
        raise StorageError(
            "buy_threshold must be greater than sell_threshold",
            error_code=_ERROR_THRESHOLD_ORDER,
            details={
                "invalid_rows": invalid_count,
                "row_count": frame.height,
            },
        )


def _require_unique_primary_keys(frame: pl.DataFrame) -> None:
    """Reject duplicate primary-key combinations."""
    unique_keys = frame.select(list(PRIMARY_KEY_COLUMNS)).n_unique()
    if unique_keys != frame.height:
        raise StorageError(
            "threshold frame contains duplicate primary keys",
            error_code=_ERROR_DUPLICATE_KEYS,
            details={
                "primary_key_columns": PRIMARY_KEY_COLUMNS,
                "row_count": frame.height,
                "unique_key_count": unique_keys,
            },
        )


def _require_name(value: object, *, parameter: str) -> str:
    """Require a non-blank string identity argument."""
    if not isinstance(value, str) or value.strip() == "":
        raise StorageError(
            f"{parameter} must be a non-blank string",
            error_code=_ERROR_NAME_BLANK,
            details={"parameter": parameter, "value": value},
        )
    return value


def _threshold_log_extra(
    *,
    model_name: str,
    model_version: str,
    symbol: str,
    timeframe: str,
    rows: int | None = None,
    columns: int | None = None,
    profile: str | None = None,
) -> dict[str, object]:
    """Build structured logging extras for threshold repository operations."""
    extra: dict[str, object] = {
        "model_name": model_name,
        "model_version": model_version,
        "symbol": symbol,
        "timeframe": timeframe,
    }
    if rows is not None:
        extra["rows"] = rows
    if columns is not None:
        extra["columns"] = columns
    if profile is not None:
        extra["profile"] = profile
    return extra

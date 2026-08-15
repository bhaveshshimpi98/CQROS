"""CQROS automatic repair for corrupted market datasets.

Purpose:
    Detect missing partitions, invalid manifests, and corrupted OHLCV
    datasets, then download only the damaged ranges and rewrite affected
    year partitions while refreshing dataset manifests.

Responsibilities:
    - Detect missing year partitions within a requested coverage window
    - Detect invalid or integrity-failing manifests
    - Detect corrupted datasets through ``MarketDataValidator``
    - Download only damaged timestamp ranges via ``HistoricalDownloader``
    - Rewrite solely the affected year partitions
    - Update the dataset manifest for rewritten partitions
    - Produce a detailed immutable ``RepairReport``

Dependencies:
    ``polars``, ``cqros.core``, ``cqros.data.timeframes``,
    ``cqros.ingestion.downloader``, ``cqros.ingestion.manifest``,
    ``cqros.ingestion.validator``, and ``cqros.storage``.

Public API:
    ``RepairSeverity``, ``RepairIssue``, ``RepairReport``, and
    ``DatasetRepairEngine``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    FILE_EXTENSION_PARQUET,
    HASH_ALGORITHM_SHA256,
    MARKET_USDT_PERPETUAL,
    MILLISECONDS_PER_SECOND,
)
from cqros.core.exceptions import (
    DataValidationError,
    IntegrityError,
    MissingDataError,
    ValidationError,
)
from cqros.core.types import (
    Exchange,
    Market,
    Symbol,
    Timeframe,
    UnixTimestampMs,
)
from cqros.data.timeframes import Timeframe as CanonicalTimeframe
from cqros.data.timeframes import to_seconds
from cqros.ingestion.downloader import HistoricalDownloader
from cqros.ingestion.manifest import (
    DatasetManifest,
    ManifestRepository,
    PartitionMetadata,
)
from cqros.ingestion.validator import MarketDataValidator, ValidationReport
from cqros.storage.exceptions import CorruptedDatasetError, DatasetNotFoundError
from cqros.storage.repository import MarketDataRepository

__all__ = [
    "RepairSeverity",
    "RepairIssue",
    "RepairReport",
    "DatasetRepairEngine",
]

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL
_DATASET_TYPE: Final[str] = "ohlcv"
_HASH_CHUNK_SIZE_BYTES: Final[int] = 1024 * 1024

# Binance USDⓈ-M Futures launched in 2019; partitions cannot precede this year.
_EARLIEST_PARTITION_YEAR: Final[int] = 2019

_OHLCV_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "symbol": pl.String,
        "timeframe": pl.String,
        "open_time": pl.Int64,
        "close_time": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "quote_volume": pl.Float64,
        "trade_count": pl.Int64,
    }
)

_GAP_ONLY_CHECKS: Final[frozenset[str]] = frozenset({"missing_timestamps"})

_logger = logging.getLogger(__name__)


class RepairSeverity(StrEnum):
    """Severity assigned to a dataset repair finding."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class RepairIssue:
    """A single finding discovered during dataset repair inspection.

    Attributes:
        severity: Whether the finding is an error, warning, or informational
            note.
        check: Stable identifier of the failed or reported check.
        message: Human-readable description of the finding.
        symbol: Tradeable symbol associated with the finding, when applicable.
        timeframe: Bar interval associated with the finding, when applicable.
        year: Calendar year of the affected partition, when applicable.
        start_time_ms: Inclusive start of an affected range in Unix ms UTC.
        end_time_ms: Inclusive end of an affected range in Unix ms UTC.
        count: Number of affected rows, bars, or partitions when aggregated.
        value: Representative diagnostic value, when available.
    """

    severity: RepairSeverity
    check: str
    message: str
    symbol: Symbol | None = None
    timeframe: Timeframe | None = None
    year: int | None = None
    start_time_ms: UnixTimestampMs | None = None
    end_time_ms: UnixTimestampMs | None = None
    count: int | None = None
    value: object | None = None


@dataclass(frozen=True, slots=True)
class RepairReport:
    """Immutable outcome of a single-symbol dataset repair.

    Attributes:
        exchange: Exchange identifier for the repaired dataset.
        market: Market segment for the repaired dataset.
        symbol: Tradeable symbol that was inspected and repaired.
        timeframe: Bar interval that was inspected and repaired.
        start_time_ms: Inclusive coverage window start (Unix ms, UTC).
        end_time_ms: Inclusive coverage window end (Unix ms, UTC).
        issues: Ordered findings discovered before and during repair.
        repaired_ranges: Inclusive timestamp ranges that were downloaded.
        rewritten_years: Calendar years whose partitions were rewritten.
        downloaded_rows: Total rows fetched from the exchange for repair.
        manifest_updated: Whether the dataset manifest was rewritten.
    """

    exchange: Exchange
    market: Market
    symbol: Symbol
    timeframe: Timeframe
    start_time_ms: UnixTimestampMs
    end_time_ms: UnixTimestampMs
    issues: tuple[RepairIssue, ...]
    repaired_ranges: tuple[tuple[UnixTimestampMs, UnixTimestampMs], ...]
    rewritten_years: tuple[int, ...]
    downloaded_rows: int
    manifest_updated: bool

    @property
    def has_errors(self) -> bool:
        """Return ``True`` when at least one error-severity issue exists."""
        return any(issue.severity is RepairSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Return ``True`` when at least one warning-severity issue exists."""
        return any(issue.severity is RepairSeverity.WARNING for issue in self.issues)

    def errors(self) -> tuple[RepairIssue, ...]:
        """Return all error-severity issues in discovery order."""
        return tuple(issue for issue in self.issues if issue.severity is RepairSeverity.ERROR)

    def warnings(self) -> tuple[RepairIssue, ...]:
        """Return all warning-severity issues in discovery order."""
        return tuple(issue for issue in self.issues if issue.severity is RepairSeverity.WARNING)


class DatasetRepairEngine:
    """Detect and repair corrupted stored Binance OHLCV datasets.

    Inspects year partitions and manifests within a caller-supplied coverage
    window, downloads only damaged ranges, rewrites affected partitions, and
    refreshes partition metadata in the dataset manifest.

    The downloader's ``BinanceClient`` session must already be open before
    calling repair methods that fetch data.

    Args:
        repository: Market-data repository used to load and rewrite partitions.
        downloader: Historical downloader used to fetch damaged ranges without
            persisting them directly (rewrite is owned by this class).
        validator: Validator applied to stored and downloaded OHLCV frames.
        manifest_repository: Manifest repository providing the manifest
            filename and logger configuration. Per-symbol dataset directories are
            resolved from the market-data repository layout.
        logger: Optional logger instance. Defaults to the module logger.
    """

    __slots__ = (
        "_repository",
        "_downloader",
        "_validator",
        "_manifest_repository",
        "_logger",
    )

    _repository: MarketDataRepository
    _downloader: HistoricalDownloader
    _validator: MarketDataValidator
    _manifest_repository: ManifestRepository
    _logger: logging.Logger

    def __init__(
        self,
        repository: MarketDataRepository,
        downloader: HistoricalDownloader,
        validator: MarketDataValidator,
        manifest_repository: ManifestRepository,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the repair engine with injected collaborators.

        Args:
            repository: Repository used to read and rewrite OHLCV partitions.
            downloader: Downloader used to fetch damaged kline ranges.
            validator: Market-data validator for stored and downloaded frames.
            manifest_repository: Manifest repository used as a template for
                per-dataset manifest access.
            logger: Optional logger instance.
        """
        self._repository = repository
        self._downloader = downloader
        self._validator = validator
        self._manifest_repository = manifest_repository
        self._logger = logger if logger is not None else _logger

    async def repair_symbol(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> RepairReport:
        """Inspect and repair OHLCV storage for a single symbol.

        Args:
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval identifier (for example ``1m``).
            start_time: Inclusive coverage window start as UTC Unix
                milliseconds.
            end_time: Inclusive coverage window end as UTC Unix milliseconds.

        Returns:
            Immutable report describing findings, downloaded ranges, rewritten
            partitions, and whether the manifest was updated.

        Raises:
            ValidationError: If timestamps or timeframe are invalid, or if
                ``start_time`` is greater than ``end_time``.
            DataValidationError: If a downloaded repair frame fails validation.
            ExchangeError: Propagated from transport failures during fetch.
        """
        resolved_start = _require_unix_ms(start_time, parameter="start_time")
        resolved_end = _require_unix_ms(end_time, parameter="end_time")
        if resolved_start > resolved_end:
            raise ValidationError(
                "start_time must be less than or equal to end_time",
                error_code="INGESTION-REPAIR-001",
                details={
                    "parameter": "start_time",
                    "start_time": resolved_start,
                    "end_time": resolved_end,
                },
            )

        interval_ms = _resolve_interval_ms(timeframe)
        issues: list[RepairIssue] = []
        download_ranges: list[tuple[UnixTimestampMs, UnixTimestampMs]] = []

        self._logger.info(
            "Starting symbol dataset repair",
            extra={
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": resolved_start,
                "end_time": resolved_end,
            },
        )

        years = _years_in_range(resolved_start, resolved_end)
        manifest_repo = self._manifest_for(symbol=symbol, timeframe=timeframe)
        manifest_state = self._inspect_manifest(
            manifest_repo,
            symbol=symbol,
            timeframe=timeframe,
            issues=issues,
        )

        for year in years:
            year_start, year_end = _year_bounds_ms(year)
            expected_start = max(resolved_start, year_start)
            expected_end = min(resolved_end, year_end)
            if expected_start > expected_end:
                continue

            self._inspect_partition(
                symbol=symbol,
                timeframe=timeframe,
                year=year,
                expected_start=expected_start,
                expected_end=expected_end,
                interval_ms=interval_ms,
                manifest=manifest_state,
                issues=issues,
                download_ranges=download_ranges,
            )

        merged_ranges = _merge_ranges(download_ranges)
        downloaded_rows = 0
        rewritten_years: tuple[int, ...] = ()

        if merged_ranges:
            downloaded = await self._download_ranges(
                symbol=symbol,
                timeframe=timeframe,
                ranges=merged_ranges,
            )
            downloaded_rows = downloaded.height
            if downloaded_rows > 0:
                report = self._validator.validate(downloaded, timeframe)
                _require_valid_download(report, symbol=symbol, timeframe=timeframe)
                rewritten_years = self._merge_and_rewrite(
                    downloaded,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                issues.append(
                    RepairIssue(
                        severity=RepairSeverity.INFO,
                        check="partitions_rewritten",
                        message=(
                            f"rewrote {len(rewritten_years)} year partition(s) "
                            f"after downloading {downloaded_rows} row(s)"
                        ),
                        symbol=symbol,
                        timeframe=timeframe,
                        count=len(rewritten_years),
                        value=list(rewritten_years),
                    )
                )

        manifest_updated = self._refresh_manifest(
            manifest_repo,
            symbol=symbol,
            timeframe=timeframe,
            start_time=resolved_start,
            end_time=resolved_end,
            issues=issues,
        )

        result = RepairReport(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            start_time_ms=resolved_start,
            end_time_ms=resolved_end,
            issues=tuple(issues),
            repaired_ranges=tuple(merged_ranges),
            rewritten_years=rewritten_years,
            downloaded_rows=downloaded_rows,
            manifest_updated=manifest_updated,
        )
        self._logger.info(
            "Completed symbol dataset repair",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "issue_count": len(issues),
                "error_count": len(result.errors()),
                "repaired_range_count": len(merged_ranges),
                "rewritten_years": list(rewritten_years),
                "downloaded_rows": downloaded_rows,
                "manifest_updated": manifest_updated,
            },
        )
        return result

    async def repair_universe(
        self,
        symbols: Sequence[Symbol],
        *,
        timeframe: Timeframe,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
    ) -> tuple[RepairReport, ...]:
        """Inspect and repair OHLCV storage for multiple symbols.

        Symbols are processed sequentially so rate-limit pressure remains
        predictable. Each symbol produces an independent ``RepairReport``.

        Args:
            symbols: Ordered symbols to repair.
            timeframe: Bar interval identifier (for example ``1m``).
            start_time: Inclusive coverage window start as UTC Unix
                milliseconds.
            end_time: Inclusive coverage window end as UTC Unix milliseconds.

        Returns:
            Immutable tuple of per-symbol repair reports in input order.

        Raises:
            ValidationError: If timestamps or timeframe are invalid.
            DataValidationError: If a downloaded repair frame fails validation.
            ExchangeError: Propagated from transport failures during fetch.
        """
        resolved_start = _require_unix_ms(start_time, parameter="start_time")
        resolved_end = _require_unix_ms(end_time, parameter="end_time")
        self._logger.info(
            "Starting universe dataset repair",
            extra={
                "exchange": _EXCHANGE,
                "market": _MARKET,
                "symbol_count": len(symbols),
                "timeframe": timeframe,
                "start_time": resolved_start,
                "end_time": resolved_end,
            },
        )
        reports: list[RepairReport] = []
        for symbol in symbols:
            report = await self.repair_symbol(
                symbol=symbol,
                timeframe=timeframe,
                start_time=resolved_start,
                end_time=resolved_end,
            )
            reports.append(report)
        self._logger.info(
            "Completed universe dataset repair",
            extra={
                "symbol_count": len(symbols),
                "timeframe": timeframe,
                "start_time": resolved_start,
                "end_time": resolved_end,
            },
        )
        return tuple(reports)

    def _manifest_for(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> ManifestRepository:
        """Return a manifest repository bound to the symbol dataset directory.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.

        Returns:
            Manifest repository for the resolved OHLCV dataset directory.
        """
        dataset_dir = self._dataset_dir(symbol=symbol, timeframe=timeframe)
        if self._manifest_repository.dataset_dir.resolve() == dataset_dir.resolve():
            return self._manifest_repository
        return ManifestRepository(
            dataset_dir,
            filename=self._manifest_repository.path.name,
            logger=self._logger,
        )

    def _dataset_dir(self, *, symbol: Symbol, timeframe: Timeframe) -> Path:
        """Resolve the OHLCV dataset directory for a symbol and timeframe.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.

        Returns:
            Directory that holds yearly ``{year}.parquet`` partitions.
        """
        layout = self._repository._layout  # pyright: ignore[reportPrivateUsage]
        sample = layout.raw_ohlcv_path(
            _EXCHANGE,
            _MARKET,
            symbol,
            timeframe,
            year=_EARLIEST_PARTITION_YEAR,
        )
        return sample.parent

    def _inspect_manifest(
        self,
        manifest_repo: ManifestRepository,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        issues: list[RepairIssue],
    ) -> DatasetManifest | None:
        """Load and verify the dataset manifest, recording issues.

        Args:
            manifest_repo: Manifest repository for the dataset directory.
            symbol: Tradeable symbol for issue context.
            timeframe: Bar interval for issue context.
            issues: Mutable issue accumulator.

        Returns:
            Loaded manifest when parseable; otherwise ``None``.
        """
        if not manifest_repo.exists():
            issues.append(
                RepairIssue(
                    severity=RepairSeverity.WARNING,
                    check="missing_manifest",
                    message="dataset manifest is missing and will be rebuilt",
                    symbol=symbol,
                    timeframe=timeframe,
                )
            )
            return None

        try:
            manifest = manifest_repo.load()
        except (MissingDataError, ValidationError) as exc:
            issues.append(
                RepairIssue(
                    severity=RepairSeverity.ERROR,
                    check="invalid_manifest",
                    message=f"dataset manifest is invalid: {exc.message}",
                    symbol=symbol,
                    timeframe=timeframe,
                    value=exc.error_code,
                )
            )
            return None

        try:
            manifest_repo.verify(manifest=manifest)
        except IntegrityError as exc:
            issues.append(
                RepairIssue(
                    severity=RepairSeverity.ERROR,
                    check="manifest_integrity",
                    message=f"manifest integrity check failed: {exc.message}",
                    symbol=symbol,
                    timeframe=timeframe,
                    value=exc.details,
                )
            )
        except (MissingDataError, ValidationError) as exc:
            issues.append(
                RepairIssue(
                    severity=RepairSeverity.ERROR,
                    check="invalid_manifest",
                    message=f"manifest verification could not complete: {exc.message}",
                    symbol=symbol,
                    timeframe=timeframe,
                    value=exc.error_code,
                )
            )

        return manifest

    def _inspect_partition(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
        expected_start: UnixTimestampMs,
        expected_end: UnixTimestampMs,
        interval_ms: int,
        manifest: DatasetManifest | None,
        issues: list[RepairIssue],
        download_ranges: list[tuple[UnixTimestampMs, UnixTimestampMs]],
    ) -> None:
        """Inspect one year partition and queue damaged ranges for download.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.
            year: Calendar year of the partition.
            expected_start: Inclusive expected coverage start for the year.
            expected_end: Inclusive expected coverage end for the year.
            interval_ms: Timeframe duration in milliseconds.
            manifest: Loaded manifest when available.
            issues: Mutable issue accumulator.
            download_ranges: Mutable list of inclusive ranges to download.
        """
        if manifest is not None and manifest.partition_for_year(year) is None:
            issues.append(
                RepairIssue(
                    severity=RepairSeverity.WARNING,
                    check="manifest_partition_missing",
                    message=(
                        f"manifest does not list year {year}; metadata will be "
                        "refreshed after repair"
                    ),
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                )
            )

        frame: pl.DataFrame | None
        try:
            frame = self._repository.load_ohlcv(
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
        except DatasetNotFoundError:
            issues.append(
                RepairIssue(
                    severity=RepairSeverity.ERROR,
                    check="missing_partition",
                    message=f"year partition {year} is missing",
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                    start_time_ms=expected_start,
                    end_time_ms=expected_end,
                )
            )
            download_ranges.append((expected_start, expected_end))
            return
        except CorruptedDatasetError as exc:
            issues.append(
                RepairIssue(
                    severity=RepairSeverity.ERROR,
                    check="corrupted_dataset",
                    message=f"year partition {year} could not be read: {exc.message}",
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                    start_time_ms=expected_start,
                    end_time_ms=expected_end,
                    value=exc.error_code,
                )
            )
            download_ranges.append((expected_start, expected_end))
            return

        report = self._validator.validate(frame, timeframe)
        if not report.is_valid:
            error_checks = tuple(sorted({issue.check for issue in report.errors()}))
            issues.append(
                RepairIssue(
                    severity=RepairSeverity.ERROR,
                    check="corrupted_dataset",
                    message=(
                        f"year partition {year} failed validation "
                        f"({len(report.errors())} error(s))"
                    ),
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                    start_time_ms=expected_start,
                    end_time_ms=expected_end,
                    count=len(report.errors()),
                    value=list(error_checks),
                )
            )
            if set(error_checks) <= _GAP_ONLY_CHECKS:
                open_times = _open_times(frame)
                download_ranges.extend(
                    _gap_ranges(
                        open_times,
                        interval_ms=interval_ms,
                        expected_start=expected_start,
                        expected_end=expected_end,
                    )
                )
            else:
                download_ranges.append((expected_start, expected_end))
            return

        open_times = _open_times(frame)
        gaps = _gap_ranges(
            open_times,
            interval_ms=interval_ms,
            expected_start=expected_start,
            expected_end=expected_end,
        )
        if gaps:
            issues.append(
                RepairIssue(
                    severity=RepairSeverity.ERROR,
                    check="coverage_gap",
                    message=(
                        f"year partition {year} is missing coverage for "
                        f"{len(gaps)} range(s) within the requested window"
                    ),
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                    start_time_ms=expected_start,
                    end_time_ms=expected_end,
                    count=len(gaps),
                    value=[list(item) for item in gaps],
                )
            )
            download_ranges.extend(gaps)

    async def _download_ranges(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        ranges: Sequence[tuple[UnixTimestampMs, UnixTimestampMs]],
    ) -> pl.DataFrame:
        """Download and concatenate OHLCV frames for damaged ranges.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.
            ranges: Inclusive timestamp ranges to fetch.

        Returns:
            Combined OHLCV frame for all ranges. Empty when nothing is
            returned by the exchange.
        """
        frames: list[pl.DataFrame] = []
        for range_start, range_end in ranges:
            self._logger.info(
                "Downloading damaged OHLCV range",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_time": range_start,
                    "end_time": range_end,
                },
            )
            frame = await self._downloader.fetch_symbol(
                symbol=symbol,
                timeframe=timeframe,
                start_time=range_start,
                end_time=range_end,
            )
            if frame.height > 0:
                frames.append(frame)

        if not frames:
            return pl.DataFrame(schema=_OHLCV_SCHEMA)
        return pl.concat(frames, how="vertical")

    def _merge_and_rewrite(
        self,
        downloaded: pl.DataFrame,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> tuple[int, ...]:
        """Merge downloaded rows into existing partitions and rewrite them.

        Args:
            downloaded: Validated OHLCV rows to merge.
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.

        Returns:
            Sorted tuple of rewritten calendar years.
        """
        year_expr = pl.from_epoch(pl.col("open_time"), time_unit="ms").dt.year()
        year_frame = downloaded.with_columns(  # pyright: ignore[reportUnknownMemberType]
            year_expr.alias("_year")
        )
        years = year_frame.get_column("_year").unique().sort().to_list()
        rewritten: list[int] = []

        for year in years:
            year_int = int(year)
            new_partition = year_frame.filter(  # pyright: ignore[reportUnknownMemberType]
                pl.col("_year") == year
            ).drop("_year")
            existing = self._load_partition_or_empty(
                symbol=symbol,
                timeframe=timeframe,
                year=year_int,
            )
            merged = _merge_ohlcv(existing, new_partition)
            self._repository.save_ohlcv(
                merged,
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=timeframe,
                year=year_int,
            )
            rewritten.append(year_int)
            self._logger.debug(
                "Rewrote OHLCV year partition after repair merge",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "year": year_int,
                    "rows": merged.height,
                },
            )

        return tuple(rewritten)

    def _load_partition_or_empty(
        self,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> pl.DataFrame:
        """Load a year partition or return an empty OHLCV frame.

        Args:
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.
            year: Calendar year of the partition.

        Returns:
            Existing partition rows, or an empty frame with the canonical
            OHLCV schema when the partition is absent or unreadable.
        """
        try:
            return self._repository.load_ohlcv(
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=symbol,
                timeframe=timeframe,
                year=year,
            )
        except (DatasetNotFoundError, CorruptedDatasetError):
            return pl.DataFrame(schema=_OHLCV_SCHEMA)

    def _refresh_manifest(
        self,
        manifest_repo: ManifestRepository,
        *,
        symbol: Symbol,
        timeframe: Timeframe,
        start_time: UnixTimestampMs,
        end_time: UnixTimestampMs,
        issues: list[RepairIssue],
    ) -> bool:
        """Rebuild partition metadata for years in the coverage window.

        Args:
            manifest_repo: Manifest repository for the dataset directory.
            symbol: Tradeable symbol.
            timeframe: Bar interval identifier.
            start_time: Inclusive coverage window start.
            end_time: Inclusive coverage window end.
            issues: Mutable issue accumulator.

        Returns:
            ``True`` when the manifest was written; otherwise ``False``.
        """
        now = _utc_now_iso()
        partitions: list[PartitionMetadata] = []
        for year in _years_in_range(start_time, end_time):
            try:
                frame = self._repository.load_ohlcv(
                    exchange=_EXCHANGE,
                    market=_MARKET,
                    symbol=symbol,
                    timeframe=timeframe,
                    year=year,
                )
            except (DatasetNotFoundError, CorruptedDatasetError):
                continue
            if frame.height == 0:
                continue
            partitions.append(
                self._partition_metadata(
                    frame,
                    year=year,
                    dataset_dir=manifest_repo.dataset_dir,
                    updated_at=now,
                )
            )

        if not partitions and not manifest_repo.exists():
            return False

        created_at = now
        overwrite_corrupt = False
        if manifest_repo.exists():
            try:
                existing = manifest_repo.load()
                created_at = existing.created_at
            except (MissingDataError, ValidationError):
                created_at = now
                overwrite_corrupt = True

        manifest = DatasetManifest(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=timeframe,
            dataset_type=_DATASET_TYPE,
            created_at=created_at,
            updated_at=now,
            partitions=tuple(partitions),
        )
        if overwrite_corrupt:
            # Corrupt JSON cannot be merged through update(); replace it.
            manifest_repo.save(manifest)
        else:
            manifest_repo.update(manifest)
        issues.append(
            RepairIssue(
                severity=RepairSeverity.INFO,
                check="manifest_updated",
                message=f"updated dataset manifest with {len(partitions)} partition(s)",
                symbol=symbol,
                timeframe=timeframe,
                count=len(partitions),
            )
        )
        return True

    def _partition_metadata(
        self,
        frame: pl.DataFrame,
        *,
        year: int,
        dataset_dir: Path,
        updated_at: str,
    ) -> PartitionMetadata:
        """Build partition metadata from a frame and optional on-disk file.

        Args:
            frame: Partition rows used for coverage statistics.
            year: Calendar year of the partition.
            dataset_dir: Directory containing the partition file.
            updated_at: ISO-8601 UTC timestamp for the metadata record.

        Returns:
            Immutable ``PartitionMetadata`` for the partition.
        """
        open_times = frame.get_column("open_time")
        start_time_ms = int(
            open_times.min()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportArgumentType]
        )
        end_time_ms = int(
            open_times.max()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportArgumentType]
        )
        filename = f"{year}{FILE_EXTENSION_PARQUET}"
        partition_path = dataset_dir / filename
        checksum, size_bytes = _file_integrity(partition_path)
        return PartitionMetadata(
            year=year,
            filename=filename,
            row_count=frame.height,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            checksum=checksum,
            size_bytes=size_bytes,
            updated_at=updated_at,
            checksum_algorithm=HASH_ALGORITHM_SHA256,
        )


def _resolve_interval_ms(timeframe: Timeframe) -> int:
    """Resolve a timeframe string to its duration in milliseconds.

    Args:
        timeframe: Candidate timeframe identifier.

    Returns:
        Interval length in Unix milliseconds.

    Raises:
        ValidationError: If ``timeframe`` is unsupported.
    """
    try:
        canonical = CanonicalTimeframe(timeframe)
    except ValueError as exc:
        raise ValidationError(
            f"unsupported timeframe: {timeframe!r}",
            error_code="INGESTION-REPAIR-002",
            details={
                "parameter": "timeframe",
                "value": timeframe,
                "allowed": sorted(item.value for item in CanonicalTimeframe),
            },
        ) from exc
    return to_seconds(canonical) * MILLISECONDS_PER_SECOND


def _require_unix_ms(value: object, *, parameter: str) -> UnixTimestampMs:
    """Validate a Unix-millisecond timestamp parameter.

    Args:
        value: Candidate timestamp value.
        parameter: Parameter name for error context.

    Returns:
        The validated integer timestamp.

    Raises:
        ValidationError: If ``value`` is not an ``int``.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            f"{parameter} must be an int Unix timestamp in milliseconds",
            error_code="INGESTION-REPAIR-003",
            details={"parameter": parameter, "type": type(value).__name__},
        )
    return value


def _require_valid_download(
    report: ValidationReport,
    *,
    symbol: Symbol,
    timeframe: Timeframe,
) -> None:
    """Raise when a downloaded repair frame failed validation.

    Args:
        report: Validation outcome for the downloaded frame.
        symbol: Tradeable symbol for error context.
        timeframe: Bar interval for error context.

    Raises:
        DataValidationError: If the report contains error-severity issues.
    """
    if report.is_valid:
        return

    errors = report.errors()
    raise DataValidationError(
        "Downloaded OHLCV failed validation before repair rewrite",
        error_code="INGESTION-REPAIR-004",
        details={
            "symbol": symbol,
            "timeframe": timeframe,
            "row_count": report.row_count,
            "error_count": len(errors),
            "checks": [issue.check for issue in errors],
        },
        recovery_suggestion=(
            "Inspect validation errors, correct the exchange payload or "
            "timeframe contract, and retry the dataset repair."
        ),
    )


def _years_in_range(
    start_time_ms: UnixTimestampMs,
    end_time_ms: UnixTimestampMs,
) -> tuple[int, ...]:
    """Return calendar years spanned by an inclusive millisecond range."""
    start_year = datetime.fromtimestamp(
        start_time_ms / MILLISECONDS_PER_SECOND,
        tz=UTC,
    ).year
    end_year = datetime.fromtimestamp(
        end_time_ms / MILLISECONDS_PER_SECOND,
        tz=UTC,
    ).year
    return tuple(range(start_year, end_year + 1))


def _year_bounds_ms(year: int) -> tuple[UnixTimestampMs, UnixTimestampMs]:
    """Return inclusive Unix-ms bounds for a UTC calendar year."""
    start = datetime(year, 1, 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC)
    start_ms = int(start.timestamp() * MILLISECONDS_PER_SECOND)
    end_ms = int(end.timestamp() * MILLISECONDS_PER_SECOND) - 1
    return start_ms, end_ms


def _open_times(frame: pl.DataFrame) -> list[int]:
    """Return sorted unique ``open_time`` values from a frame."""
    if frame.height == 0 or "open_time" not in frame.columns:
        return []
    values = frame.get_column("open_time").drop_nulls().unique().sort().to_list()
    return [int(cast(int, value)) for value in values]


def _gap_ranges(
    open_times: Sequence[int],
    *,
    interval_ms: int,
    expected_start: UnixTimestampMs,
    expected_end: UnixTimestampMs,
) -> list[tuple[UnixTimestampMs, UnixTimestampMs]]:
    """Compute inclusive download ranges that fill missing coverage.

    Args:
        open_times: Sorted unique open times already present.
        interval_ms: Timeframe duration in milliseconds.
        expected_start: Inclusive expected coverage start.
        expected_end: Inclusive expected coverage end.

    Returns:
        Inclusive ``(start, end)`` ranges that should be downloaded.
    """
    if not open_times:
        return [(expected_start, expected_end)]

    ranges: list[tuple[UnixTimestampMs, UnixTimestampMs]] = []
    first = open_times[0]
    last = open_times[-1]

    if first > expected_start:
        ranges.append((expected_start, first - 1))

    for previous, current in zip(open_times, open_times[1:], strict=False):
        if current - previous > interval_ms:
            gap_start = previous + interval_ms
            gap_end = current - 1
            if gap_start <= gap_end:
                ranges.append((gap_start, gap_end))

    next_open = last + interval_ms
    if next_open <= expected_end:
        ranges.append((next_open, expected_end))

    return ranges


def _merge_ranges(
    ranges: Sequence[tuple[UnixTimestampMs, UnixTimestampMs]],
) -> list[tuple[UnixTimestampMs, UnixTimestampMs]]:
    """Merge overlapping or adjacent inclusive timestamp ranges."""
    if not ranges:
        return []

    ordered = sorted(ranges, key=lambda item: (item[0], item[1]))
    merged: list[tuple[UnixTimestampMs, UnixTimestampMs]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _merge_ohlcv(existing: pl.DataFrame, new_rows: pl.DataFrame) -> pl.DataFrame:
    """Concatenate, deduplicate by ``open_time``, and sort chronologically.

    When both frames contain the same ``open_time``, the newer row wins so
    exchange corrections in the download replace previously stored values.

    Args:
        existing: Rows already stored for the affected year partition.
        new_rows: Newly downloaded rows for the same year.

    Returns:
        Deduplicated frame sorted by ascending ``open_time``.
    """
    if existing.height == 0:
        base = new_rows
    elif new_rows.height == 0:
        base = existing
    else:
        base = pl.concat([existing, new_rows], how="vertical")

    return base.unique(  # pyright: ignore[reportUnknownMemberType]
        subset=["open_time"], keep="last"
    ).sort(  # pyright: ignore[reportUnknownMemberType]
        "open_time"
    )


def _file_integrity(path: Path) -> tuple[str, int]:
    """Return SHA-256 checksum and size for a partition file when present.

    Args:
        path: Partition file path.

    Returns:
        ``(checksum, size_bytes)``. Empty checksum and zero size when the
        file is absent (for example under an in-memory datastore).
    """
    if not path.is_file():
        return "", 0

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()

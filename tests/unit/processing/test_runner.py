"""Unit tests for CQROS ProcessingRunner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, is_dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.types import FilePath
from cqros.processing.cleaning import (
    CleaningReport,
    FundingCleaner,
    LongShortCleaner,
    OHLCVCleaner,
    OpenInterestCleaner,
    TakerVolumeCleaner,
)
from cqros.processing.exceptions import ProcessingError, ProcessingValidationError
from cqros.processing.pipeline import ProcessingPipeline
from cqros.processing.runner import (
    ProcessingRunner,
    ProcessingSummary,
    ProcessingTaskResult,
)
from cqros.storage import (
    DatasetNotFoundError,
    MarketDataRepository,
    ProcessedMarketDataRepository,
    StorageLayout,
)

_EXCHANGE = "binance"
_MARKET = "usdt_perpetual"
_SYMBOL = "BTCUSDT"
_OTHER = "ETHUSDT"
_TIMEFRAME = "1m"
_YEAR = 2024


class _InMemoryDataStore:
    """Minimal ``IDataStore`` stub used by both repositories."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}

    def write(self, path: FilePath, dataframe: pl.DataFrame) -> None:
        self.frames[Path(path)] = dataframe.clone()

    def read(self, path: FilePath) -> pl.DataFrame:
        target = Path(path)
        try:
            return self.frames[target]
        except KeyError as exc:
            raise DatasetNotFoundError(
                "Dataset not found",
                error_code="STORAGE-TEST-001",
                details={"path": str(target)},
            ) from exc

    def scan(self, path: FilePath) -> pl.LazyFrame:
        return self.read(path).lazy()

    def exists(self, path: FilePath) -> bool:
        return Path(path) in self.frames

    def delete(self, path: FilePath) -> None:
        del self.frames[Path(path)]

    def schema(self, path: FilePath) -> pl.Schema:
        return self.read(path).schema

    def row_count(self, path: FilePath) -> int:
        return self.read(path).height


@dataclass(frozen=True, slots=True)
class _PassthroughStep:
    """Identity processing step used for runner unit tests."""

    name: str = "passthrough"
    version: str = "1.0.0"
    description: str = "Return frame unchanged"

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.clone()


@dataclass(frozen=True, slots=True)
class _FailingStep:
    """Processing step that always fails."""

    name: str = "failing"
    version: str = "1.0.0"
    description: str = "Always fail"

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        raise ProcessingValidationError(
            "forced failure",
            error_code="PROCESSING-TEST-FAIL",
            details={"rows": frame.height},
        )


@dataclass(frozen=True, slots=True)
class _AddMarkerStep:
    """Append a marker column for verifying pipeline execution."""

    name: str = "add_marker"
    version: str = "1.0.0"
    description: str = "Add marker column"

    def process(self, frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.lit(True).alias("processed"))


@dataclass(frozen=True, slots=True)
class _PassthroughCleaner:
    """Cleaner that returns the input frame unchanged with a zero report."""

    def clean(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
        report = CleaningReport(
            rows_before=frame.height,
            rows_after=frame.height,
            duplicates_removed=0,
            null_rows_removed=0,
            invalid_price_rows_removed=0,
            invalid_volume_rows_removed=0,
            invalid_trade_count_rows_removed=0,
            invalid_timestamp_rows_removed=0,
            warnings=(),
        )
        return frame.clone(), report


@dataclass(frozen=True, slots=True)
class _MarkerCleaner:
    """Cleaner that requires the pipeline marker and appends a cleaned flag."""

    def clean(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
        if "processed" not in frame.columns:
            raise ProcessingValidationError(
                "pipeline marker missing before cleaning",
                error_code="PROCESSING-TEST-ORDER",
                details={"columns": tuple(frame.columns)},
            )
        cleaned = frame.with_columns(pl.lit(True).alias("cleaned"))
        report = CleaningReport(
            rows_before=frame.height,
            rows_after=cleaned.height,
            duplicates_removed=0,
            null_rows_removed=0,
            invalid_price_rows_removed=0,
            invalid_volume_rows_removed=0,
            invalid_trade_count_rows_removed=0,
            invalid_timestamp_rows_removed=0,
            warnings=(),
        )
        return cleaned, report


@dataclass(frozen=True, slots=True)
class _DropRowCleaner:
    """Cleaner that drops the last row and reports the removal."""

    def clean(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
        cleaned = frame.head(max(frame.height - 1, 0))
        report = CleaningReport(
            rows_before=frame.height,
            rows_after=cleaned.height,
            duplicates_removed=1 if frame.height > cleaned.height else 0,
            null_rows_removed=0,
            invalid_price_rows_removed=0,
            invalid_volume_rows_removed=0,
            invalid_trade_count_rows_removed=0,
            invalid_timestamp_rows_removed=0,
            warnings=(
                ("Removed 1 duplicate timestamp row(s).",) if frame.height > cleaned.height else ()
            ),
        )
        return cleaned, report


def _frame(*, marker: int = 1) -> pl.DataFrame:
    """Build a minimal partition frame."""
    return pl.DataFrame({"value": [float(marker), float(marker + 1)]})


def _zero_report(*, rows: int) -> CleaningReport:
    """Build a zero-removal CleaningReport."""
    return CleaningReport(
        rows_before=rows,
        rows_after=rows,
        duplicates_removed=0,
        null_rows_removed=0,
        invalid_price_rows_removed=0,
        invalid_volume_rows_removed=0,
        invalid_trade_count_rows_removed=0,
        invalid_timestamp_rows_removed=0,
        warnings=(),
    )


def _make_cleaner_mock(*, label: str) -> MagicMock:
    """Return a cleaner mock that passes frames through with a labeled report."""

    def _clean(frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
        report = _zero_report(rows=frame.height)
        cleaned = frame.with_columns(pl.lit(label).alias("cleaner"))
        return cleaned, report

    mock = MagicMock()
    mock.clean.side_effect = _clean
    return mock


def _runner(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
    pipeline: Any,
    *,
    ohlcv_cleaner: Any | None = None,
    funding_cleaner: Any | None = None,
    open_interest_cleaner: Any | None = None,
    taker_volume_cleaner: Any | None = None,
    long_short_cleaner: Any | None = None,
) -> ProcessingRunner:
    """Build a runner with passthrough cleaners unless overrides are supplied."""
    passthrough = _PassthroughCleaner()
    return ProcessingRunner(
        raw_repository,
        processed_repository,
        pipeline,
        ohlcv_cleaner=passthrough if ohlcv_cleaner is None else ohlcv_cleaner,
        funding_cleaner=passthrough if funding_cleaner is None else funding_cleaner,
        open_interest_cleaner=(
            passthrough if open_interest_cleaner is None else open_interest_cleaner
        ),
        taker_volume_cleaner=(
            passthrough if taker_volume_cleaner is None else taker_volume_cleaner
        ),
        long_short_cleaner=(passthrough if long_short_cleaner is None else long_short_cleaner),
    )


@pytest.fixture
def layout(tmp_path: Path) -> StorageLayout:
    """Return a temporary storage layout."""
    return StorageLayout(tmp_path)


@pytest.fixture
def datastore() -> _InMemoryDataStore:
    """Return a shared in-memory datastore."""
    return _InMemoryDataStore()


@pytest.fixture
def raw_repository(
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> MarketDataRepository:
    """Return a raw repository on the shared datastore."""
    return MarketDataRepository(layout, datastore)


@pytest.fixture
def processed_repository(
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> ProcessedMarketDataRepository:
    """Return a processed repository on the shared datastore."""
    return ProcessedMarketDataRepository(layout, datastore)


def _seed_raw_ohlcv(
    raw_repository: MarketDataRepository,
    *,
    symbol: str,
    year: int,
    marker: int = 1,
) -> pl.DataFrame:
    """Persist a raw OHLCV partition and return the stored frame."""
    frame = _frame(marker=marker)
    raw_repository.save_ohlcv(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=symbol,
        timeframe=_TIMEFRAME,
        year=year,
    )
    return frame


def test_processing_summary_and_task_result_are_immutable() -> None:
    """Summary and task result models are frozen slotted dataclasses."""
    result = ProcessingTaskResult(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
        status="succeeded",
        rows_loaded=2,
        rows_saved=2,
        cleaning_report=_zero_report(rows=2),
    )
    summary = ProcessingSummary(
        dataset="ohlcv",
        exchange=_EXCHANGE,
        market=_MARKET,
        results=(result,),
    )
    assert is_dataclass(result)
    assert is_dataclass(summary)
    assert ProcessingTaskResult.__slots__ == (
        "symbol",
        "timeframe",
        "year",
        "status",
        "rows_loaded",
        "rows_saved",
        "cleaning_report",
        "error_type",
        "error_message",
        "error_code",
    )
    with pytest.raises(FrozenInstanceError):
        result.status = "failed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        summary.dataset = "funding"  # type: ignore[misc]


def test_process_ohlcv_loads_runs_pipeline_and_saves(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
    layout: StorageLayout,
    datastore: _InMemoryDataStore,
) -> None:
    """Successful OHLCV processing persists the pipeline output."""
    original = _seed_raw_ohlcv(raw_repository, symbol=_SYMBOL, year=_YEAR)
    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_AddMarkerStep(),)),
    )

    summary = runner.process_ohlcv(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )

    assert summary.total_count == 1
    assert summary.succeeded_count == 1
    assert summary.failed_count == 0
    assert summary.results[0].status == "succeeded"
    assert summary.results[0].rows_loaded == original.height
    assert summary.results[0].rows_saved == original.height
    assert summary.results[0].cleaning_report == _zero_report(rows=original.height)

    loaded = processed_repository.load_ohlcv(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert "processed" in loaded.columns
    assert loaded.get_column("processed").to_list() == [True, True]
    assert (
        layout.processed_ohlcv_path(
            _EXCHANGE,
            _MARKET,
            _SYMBOL,
            _TIMEFRAME,
            _YEAR,
        )
        in datastore.frames
    )


def test_runner_continues_after_partition_failure(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """A failed partition does not stop remaining assets from processing."""
    _seed_raw_ohlcv(raw_repository, symbol=_SYMBOL, year=_YEAR, marker=1)
    _seed_raw_ohlcv(raw_repository, symbol=_OTHER, year=_YEAR, marker=10)

    class _ConditionalPipeline:
        def run(self, frame: pl.DataFrame) -> pl.DataFrame:
            if frame.get_column("value").to_list()[0] == 1.0:
                raise RuntimeError("boom")
            return frame.with_columns(pl.lit(True).alias("processed"))

    runner = _runner(
        raw_repository,
        processed_repository,
        _ConditionalPipeline(),
    )
    summary = runner.process_ohlcv(
        symbols=(_SYMBOL, _OTHER),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )

    assert summary.total_count == 2
    assert summary.succeeded_count == 1
    assert summary.failed_count == 1
    assert summary.failed[0].symbol == _SYMBOL
    assert summary.failed[0].error_type == "RuntimeError"
    assert summary.failed[0].cleaning_report is None
    assert summary.succeeded[0].symbol == _OTHER
    assert summary.succeeded[0].cleaning_report is not None

    loaded = processed_repository.load_ohlcv(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_OTHER,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert "processed" in loaded.columns


def test_runner_records_missing_raw_partition_as_failure(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """Missing raw partitions are captured without aborting the run."""
    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
    )
    summary = runner.process_ohlcv(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    assert summary.failed_count == 1
    assert summary.results[0].error_type == "DatasetNotFoundError"
    assert summary.results[0].cleaning_report is None


def test_runner_records_pipeline_validation_failure(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """Processing validation failures are recorded with error codes."""
    _seed_raw_ohlcv(raw_repository, symbol=_SYMBOL, year=_YEAR)
    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_FailingStep(),)),
    )
    summary = runner.process_ohlcv(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    assert summary.failed_count == 1
    assert summary.results[0].error_code == "PROCESSING-TEST-FAIL"
    assert summary.results[0].error_type == "ProcessingValidationError"
    assert summary.results[0].cleaning_report is None


def test_runner_rejects_empty_symbols(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """Empty symbol lists raise ProcessingError before execution."""
    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
    )
    with pytest.raises(ProcessingError, match="symbols") as exc:
        runner.process_ohlcv(
            symbols=(),
            timeframes=(_TIMEFRAME,),
            years=(_YEAR,),
        )
    assert exc.value.error_code == "PROCESSING-RUNNER-001"


def test_runner_rejects_empty_years(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """Empty year lists raise ProcessingError before execution."""
    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
    )
    with pytest.raises(ProcessingError, match="years") as exc:
        runner.process_ohlcv(
            symbols=(_SYMBOL,),
            timeframes=(_TIMEFRAME,),
            years=(),
        )
    assert exc.value.error_code == "PROCESSING-RUNNER-003"


def test_process_funding_round_trip(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """Funding processing uses the injected pipeline and processed repository."""
    frame = _frame()
    raw_repository.save_funding(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
    )
    summary = runner.process_funding(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    assert summary.dataset == "funding"
    assert summary.succeeded_count == 1
    assert summary.results[0].cleaning_report == _zero_report(rows=frame.height)
    loaded = processed_repository.load_funding(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert_frame_equal(loaded, frame)


def test_package_exports_runner_symbols() -> None:
    """Processing runner types are exported from the processing package."""
    import cqros.processing as processing_package

    for name in ("ProcessingRunner", "ProcessingSummary", "ProcessingTaskResult"):
        assert name in processing_package.__all__
        assert getattr(processing_package, name).__name__ == name


# --- Cleaner orchestration ---


def test_default_cleaners_are_constructed(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """Omitted cleaners default to the concrete cleaner classes."""
    runner = ProcessingRunner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
    )
    assert isinstance(runner._ohlcv_cleaner, OHLCVCleaner)
    assert isinstance(runner._funding_cleaner, FundingCleaner)
    assert isinstance(runner._open_interest_cleaner, OpenInterestCleaner)
    assert isinstance(runner._taker_volume_cleaner, TakerVolumeCleaner)
    assert isinstance(runner._long_short_cleaner, LongShortCleaner)


def test_dependency_injection_uses_supplied_cleaners(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """Constructor-injected cleaners are retained by the runner."""
    ohlcv = _PassthroughCleaner()
    funding = _PassthroughCleaner()
    open_interest = _PassthroughCleaner()
    taker_volume = _PassthroughCleaner()
    long_short = _PassthroughCleaner()
    runner = ProcessingRunner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
        ohlcv_cleaner=ohlcv,
        funding_cleaner=funding,
        open_interest_cleaner=open_interest,
        taker_volume_cleaner=taker_volume,
        long_short_cleaner=long_short,
    )
    assert runner._ohlcv_cleaner is ohlcv
    assert runner._funding_cleaner is funding
    assert runner._open_interest_cleaner is open_interest
    assert runner._taker_volume_cleaner is taker_volume
    assert runner._long_short_cleaner is long_short


@pytest.mark.parametrize(
    ("method_name", "save_raw", "load_processed", "cleaner_kwarg"),
    [
        (
            "process_ohlcv",
            "save_ohlcv",
            "load_ohlcv",
            "ohlcv_cleaner",
        ),
        (
            "process_funding",
            "save_funding",
            "load_funding",
            "funding_cleaner",
        ),
        (
            "process_open_interest",
            "save_open_interest",
            "load_open_interest",
            "open_interest_cleaner",
        ),
        (
            "process_taker_volume",
            "save_taker_volume",
            "load_taker_volume",
            "taker_volume_cleaner",
        ),
        (
            "process_global_long_short_account_ratio",
            "save_global_long_short_account_ratio",
            "load_global_long_short_account_ratio",
            "long_short_cleaner",
        ),
        (
            "process_top_long_short_account_ratio",
            "save_top_long_short_account_ratio",
            "load_top_long_short_account_ratio",
            "long_short_cleaner",
        ),
        (
            "process_top_long_short_position_ratio",
            "save_top_long_short_position_ratio",
            "load_top_long_short_position_ratio",
            "long_short_cleaner",
        ),
    ],
)
def test_correct_cleaner_selected_and_invoked_once(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
    method_name: str,
    save_raw: str,
    load_processed: str,
    cleaner_kwarg: str,
) -> None:
    """Each dataset method invokes only its mapped cleaner exactly once."""
    frame = _frame()
    getattr(raw_repository, save_raw)(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    cleaners = {
        "ohlcv_cleaner": _make_cleaner_mock(label="ohlcv"),
        "funding_cleaner": _make_cleaner_mock(label="funding"),
        "open_interest_cleaner": _make_cleaner_mock(label="open_interest"),
        "taker_volume_cleaner": _make_cleaner_mock(label="taker_volume"),
        "long_short_cleaner": _make_cleaner_mock(label="long_short"),
    }
    runner = ProcessingRunner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
        **cleaners,
    )
    summary = getattr(runner, method_name)(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    assert summary.succeeded_count == 1
    assert cleaners[cleaner_kwarg].clean.call_count == 1
    for name, mock in cleaners.items():
        if name != cleaner_kwarg:
            mock.clean.assert_not_called()

    loaded = getattr(processed_repository, load_processed)(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert (
        loaded.get_column("cleaner").to_list()
        == [
            {
                "ohlcv_cleaner": "ohlcv",
                "funding_cleaner": "funding",
                "open_interest_cleaner": "open_interest",
                "taker_volume_cleaner": "taker_volume",
                "long_short_cleaner": "long_short",
            }[cleaner_kwarg]
        ]
        * frame.height
    )


def test_cleaned_frame_is_saved_not_pipeline_only_output(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """The cleaned frame, not the pre-clean pipeline output, is persisted."""
    original = _seed_raw_ohlcv(raw_repository, symbol=_SYMBOL, year=_YEAR)
    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
        ohlcv_cleaner=_DropRowCleaner(),
    )
    summary = runner.process_ohlcv(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    assert summary.succeeded_count == 1
    assert summary.results[0].rows_loaded == original.height
    assert summary.results[0].rows_saved == original.height - 1
    assert summary.results[0].cleaning_report is not None
    assert summary.results[0].cleaning_report.duplicates_removed == 1

    loaded = processed_repository.load_ohlcv(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert loaded.height == original.height - 1


def test_cleaning_report_propagated_on_success(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """Successful tasks expose the cleaner CleaningReport on the result."""
    _seed_raw_ohlcv(raw_repository, symbol=_SYMBOL, year=_YEAR)
    expected = CleaningReport(
        rows_before=2,
        rows_after=1,
        duplicates_removed=1,
        null_rows_removed=0,
        invalid_price_rows_removed=0,
        invalid_volume_rows_removed=0,
        invalid_trade_count_rows_removed=0,
        invalid_timestamp_rows_removed=0,
        warnings=("Removed 1 duplicate timestamp row(s).",),
    )
    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
        ohlcv_cleaner=_DropRowCleaner(),
    )
    summary = runner.process_ohlcv(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    assert summary.results[0].cleaning_report == expected


def test_pipeline_runs_before_cleaner(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """Pipeline output is passed to the cleaner before persistence."""
    _seed_raw_ohlcv(raw_repository, symbol=_SYMBOL, year=_YEAR)
    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_AddMarkerStep(),)),
        ohlcv_cleaner=_MarkerCleaner(),
    )
    summary = runner.process_ohlcv(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    assert summary.succeeded_count == 1
    loaded = processed_repository.load_ohlcv(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert loaded.get_column("processed").to_list() == [True, True]
    assert loaded.get_column("cleaned").to_list() == [True, True]


def test_cleaner_failure_is_isolated(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """A cleaner failure marks the partition failed and continues the run."""
    _seed_raw_ohlcv(raw_repository, symbol=_SYMBOL, year=_YEAR, marker=1)
    _seed_raw_ohlcv(raw_repository, symbol=_OTHER, year=_YEAR, marker=10)

    class _ConditionalCleaner:
        def clean(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, CleaningReport]:
            if frame.get_column("value").to_list()[0] == 1.0:
                raise ProcessingValidationError(
                    "cleaner boom",
                    error_code="PROCESSING-TEST-CLEANER-BOOM",
                    details={},
                )
            return _PassthroughCleaner().clean(frame)

    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
        ohlcv_cleaner=_ConditionalCleaner(),
    )
    summary = runner.process_ohlcv(
        symbols=(_SYMBOL, _OTHER),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    assert summary.total_count == 2
    assert summary.failed_count == 1
    assert summary.succeeded_count == 1
    assert summary.failed[0].symbol == _SYMBOL
    assert summary.failed[0].error_code == "PROCESSING-TEST-CLEANER-BOOM"
    assert summary.failed[0].cleaning_report is None
    assert summary.succeeded[0].symbol == _OTHER
    assert summary.succeeded[0].cleaning_report is not None

    loaded = processed_repository.load_ohlcv(
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_OTHER,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    assert loaded.height == 2


def test_summary_remains_immutable_with_cleaning_reports(
    raw_repository: MarketDataRepository,
    processed_repository: ProcessedMarketDataRepository,
) -> None:
    """ProcessingSummary remains frozen after successful cleaning."""
    _seed_raw_ohlcv(raw_repository, symbol=_SYMBOL, year=_YEAR)
    runner = _runner(
        raw_repository,
        processed_repository,
        ProcessingPipeline((_PassthroughStep(),)),
    )
    summary = runner.process_ohlcv(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
        exchange=_EXCHANGE,
        market=_MARKET,
    )
    with pytest.raises(FrozenInstanceError):
        summary.results = ()  # type: ignore[misc]
    report = summary.results[0].cleaning_report
    assert report is not None
    with pytest.raises(FrozenInstanceError):
        report.rows_after = 0  # type: ignore[misc]

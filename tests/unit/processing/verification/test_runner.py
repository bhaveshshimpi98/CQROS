"""Unit tests for CQROS VerificationRunner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.types import FilePath
from cqros.processing.exceptions import ProcessingError
from cqros.processing.verification import (
    VerificationReport,
    VerificationRunner,
    VerificationSummary,
    VerificationTaskResult,
)
from cqros.storage import (
    DatasetNotFoundError,
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
    """Minimal ``IDataStore`` stub used by the processed repository."""

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


class _PassthroughVerifier:
    """Verifier that always returns a deterministic passing report."""

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        return VerificationReport(
            rows_checked=frame.height,
            duplicate_timestamp_rows=0,
            null_rows=0,
            nan_rows=0,
            invalid_timestamp_rows=0,
            invalid_numeric_rows=0,
            warnings=(),
            passed=True,
        )


class _FailingVerifier:
    """Verifier that always raises."""

    def verify(self, frame: pl.DataFrame) -> VerificationReport:
        raise RuntimeError(f"forced failure rows={frame.height}")


def _ohlcv_frame() -> pl.DataFrame:
    """Return a minimal OHLCV-shaped frame for repository storage."""
    return pl.DataFrame(
        {
            "timestamp": [1_700_000_000_000, 1_700_000_060_000],
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [10.0, 20.0],
        }
    )


@pytest.fixture
def repository(tmp_path: Path) -> ProcessedMarketDataRepository:
    """Return a processed repository backed by an in-memory datastore."""
    return ProcessedMarketDataRepository(StorageLayout(tmp_path), _InMemoryDataStore())


def test_task_result_and_summary_are_frozen() -> None:
    """Runner result types are immutable dataclasses."""
    assert is_dataclass(VerificationTaskResult)
    assert is_dataclass(VerificationSummary)
    result = VerificationTaskResult(
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
        status="succeeded",
    )
    with pytest.raises(FrozenInstanceError):
        result.status = "failed"  # type: ignore[misc]


def test_verify_ohlcv_success(repository: ProcessedMarketDataRepository) -> None:
    """Successful OHLCV verification returns a passed report."""
    frame = _ohlcv_frame()
    repository.save_ohlcv(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    runner = VerificationRunner(repository, ohlcv_verifier=_PassthroughVerifier())

    summary = runner.verify_ohlcv(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
    )

    assert summary.dataset == "ohlcv"
    assert summary.succeeded_count == 1
    assert summary.failed_count == 0
    assert summary.results[0].report is not None
    assert summary.results[0].report.rows_checked == 2
    assert summary.results[0].report.passed is True


def test_verify_partition_failure_continues(
    repository: ProcessedMarketDataRepository,
) -> None:
    """A partition failure is captured and remaining partitions continue."""
    frame = _ohlcv_frame()
    repository.save_ohlcv(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    repository.save_ohlcv(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_OTHER,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )

    class _SelectiveVerifier:
        def verify(self, frame: pl.DataFrame) -> VerificationReport:
            if frame.height == 2 and "open" in frame.columns:
                # Fail only the first symbol by raising on first call.
                if not getattr(self, "_seen", False):
                    self._seen = True
                    raise RuntimeError("first failure")
            return VerificationReport(
                rows_checked=frame.height,
                duplicate_timestamp_rows=0,
                null_rows=0,
                nan_rows=0,
                invalid_timestamp_rows=0,
                invalid_numeric_rows=0,
                warnings=(),
                passed=True,
            )

    runner = VerificationRunner(repository, ohlcv_verifier=_SelectiveVerifier())
    summary = runner.verify_ohlcv(
        symbols=(_SYMBOL, _OTHER),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
    )

    assert summary.total_count == 2
    assert summary.failed_count == 1
    assert summary.succeeded_count == 1
    assert summary.failed[0].error_type == "RuntimeError"


def test_verify_missing_partition_is_failed(
    repository: ProcessedMarketDataRepository,
) -> None:
    """Missing partitions become failed task results."""
    runner = VerificationRunner(repository, ohlcv_verifier=_PassthroughVerifier())
    summary = runner.verify_ohlcv(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
    )
    assert summary.failed_count == 1
    assert summary.results[0].error_type == "DatasetNotFoundError"


def test_verify_rejects_empty_inputs(repository: ProcessedMarketDataRepository) -> None:
    """Empty symbols, timeframes, or years raise ProcessingError."""
    runner = VerificationRunner(repository, ohlcv_verifier=_PassthroughVerifier())
    with pytest.raises(ProcessingError, match="symbols"):
        runner.verify_ohlcv(symbols=(), timeframes=(_TIMEFRAME,), years=(_YEAR,))
    with pytest.raises(ProcessingError, match="timeframes"):
        runner.verify_ohlcv(symbols=(_SYMBOL,), timeframes=(), years=(_YEAR,))
    with pytest.raises(ProcessingError, match="years"):
        runner.verify_ohlcv(symbols=(_SYMBOL,), timeframes=(_TIMEFRAME,), years=())


def test_verify_funding_uses_injected_verifier(
    repository: ProcessedMarketDataRepository,
) -> None:
    """Funding verification routes through the funding verifier."""
    frame = pl.DataFrame({"funding_time": [1], "funding_rate": [0.01]})
    repository.save_funding(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe="8h",
        year=_YEAR,
    )
    verifier = _PassthroughVerifier()
    runner = VerificationRunner(repository, funding_verifier=verifier)
    summary = runner.verify_funding(
        symbols=(_SYMBOL,),
        timeframes=("8h",),
        years=(_YEAR,),
    )
    assert summary.succeeded_count == 1
    assert summary.results[0].report is not None
    assert_frame_equal(
        repository.load_funding(
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=_SYMBOL,
            timeframe="8h",
            year=_YEAR,
        ),
        frame,
    )


def test_verify_long_short_methods(repository: ProcessedMarketDataRepository) -> None:
    """All long/short runner methods succeed with the shared verifier."""
    frame = pl.DataFrame({"timestamp": [1], "long_short_ratio": [1.0]})
    repository.save_global_long_short_account_ratio(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    repository.save_top_long_short_account_ratio(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    repository.save_top_long_short_position_ratio(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    runner = VerificationRunner(repository, long_short_verifier=_PassthroughVerifier())

    for method in (
        runner.verify_global_long_short_account_ratio,
        runner.verify_top_long_short_account_ratio,
        runner.verify_top_long_short_position_ratio,
    ):
        summary = method(
            symbols=(_SYMBOL,),
            timeframes=(_TIMEFRAME,),
            years=(_YEAR,),
        )
        assert summary.succeeded_count == 1


def test_failing_verifier_is_isolated(repository: ProcessedMarketDataRepository) -> None:
    """Verifier exceptions become failed task results."""
    frame = _ohlcv_frame()
    repository.save_ohlcv(
        frame,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL,
        timeframe=_TIMEFRAME,
        year=_YEAR,
    )
    runner = VerificationRunner(repository, ohlcv_verifier=_FailingVerifier())
    summary = runner.verify_ohlcv(
        symbols=(_SYMBOL,),
        timeframes=(_TIMEFRAME,),
        years=(_YEAR,),
    )
    assert summary.failed_count == 1
    assert summary.results[0].error_message is not None
    assert "forced failure" in summary.results[0].error_message

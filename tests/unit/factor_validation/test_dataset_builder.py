"""Unit tests for CQROS ``ValidationDatasetBuilder``."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.types import FilePath
from cqros.factor_validation import (
    FactorValidationError,
    ValidationDatasetBuilder,
)
from cqros.factor_validation.validation_dataset_schema import (
    CANONICAL_COLUMN_ORDER,
    COLUMN_DTYPES,
    PRIMARY_KEY_COLUMNS,
    VALIDATION_DATASET_SCHEMA,
    VALIDATION_LABEL_COLUMNS,
)
from cqros.factors import FactorsRepository, FactorStatus
from cqros.factors.schema import CANONICAL_COLUMN_ORDER as FACTOR_CANONICAL_COLUMN_ORDER
from cqros.factors.schema import COLUMN_DTYPES as FACTOR_COLUMN_DTYPES
from cqros.labels.schema import CANONICAL_COLUMN_ORDER as LABEL_CANONICAL_COLUMN_ORDER
from cqros.labels.schema import COLUMN_DTYPES as LABEL_COLUMN_DTYPES
from cqros.labels.schema import LABEL_COLUMNS
from cqros.storage import LabelRepository, StorageLayout
from cqros.storage.exceptions import DatasetNotFoundError

_MANAGER = "default"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1h"
_YEAR = 2026
_OPEN_TIMES = (1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000)
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_FACTOR_GROUP = "alpha"

_PANEL_KWARGS = {
    "manager": _MANAGER,
    "exchange": _EXCHANGE,
    "market": _MARKET,
    "timeframe": _TIMEFRAME,
    "year": _YEAR,
}
_SYMBOL_PARTITION_KWARGS = {
    **_PANEL_KWARGS,
    "symbol": _SYMBOL,
}


class _InMemoryDataStore:
    """Minimal in-memory ``IDataStore`` for builder partition tests."""

    def __init__(self) -> None:
        self.frames: dict[Path, pl.DataFrame] = {}

    def write(self, path: FilePath, frame: pl.DataFrame) -> None:
        self.frames[Path(path)] = frame.clone()

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


def _factors_frame(
    *,
    open_times: tuple[int, ...] = _OPEN_TIMES,
    factor_name: str = _FACTOR_NAME,
    duplicate_observation: bool = False,
) -> pl.DataFrame:
    """Build a canonical Factors frame for builder tests."""
    rows = list(open_times)
    if duplicate_observation:
        rows = [*rows, rows[0]]
    data = {
        "symbol": [_SYMBOL] * len(rows),
        "timeframe": [_TIMEFRAME] * len(rows),
        "open_time": list(rows),
        "factor_name": [factor_name] * len(rows),
        "factor_version": [_FACTOR_VERSION] * len(rows),
        "factor_category": [_FACTOR_CATEGORY] * len(rows),
        "factor_group": [_FACTOR_GROUP] * len(rows),
        "factor_value": [0.1 * float(index + 1) for index in range(len(rows))],
        "lookback": [20] * len(rows),
        "prediction_horizon": [1] * len(rows),
        "enabled": [True] * len(rows),
        "status": [FactorStatus.ACTIVE.value] * len(rows),
    }
    return pl.DataFrame(data, schema=dict(FACTOR_COLUMN_DTYPES)).select(
        list(FACTOR_CANONICAL_COLUMN_ORDER)
    )


def _labels_frame(
    *,
    open_times: tuple[int, ...] = _OPEN_TIMES,
    include_all_label_columns: bool = True,
) -> pl.DataFrame:
    """Build a Labels frame containing required validation label columns."""
    rows = len(open_times)
    data: dict[str, list[object]] = {
        "symbol": [_SYMBOL] * rows,
        "timeframe": [_TIMEFRAME] * rows,
        "open_time": list(open_times),
    }
    if include_all_label_columns:
        for index, column in enumerate(LABEL_COLUMNS):
            if column.startswith("direction_"):
                data[column] = [1 if offset % 2 == 0 else 0 for offset in range(rows)]
            else:
                data[column] = [0.01 * float(index + offset + 1) for offset in range(rows)]
        return pl.DataFrame(data, schema=dict(LABEL_COLUMN_DTYPES)).select(
            list(LABEL_CANONICAL_COLUMN_ORDER)
        )

    data["future_return_1"] = [0.01 * float(offset + 1) for offset in range(rows)]
    return pl.DataFrame(data)


def _builder(
    *,
    factors: pl.DataFrame | None = None,
    labels: pl.DataFrame | None = None,
    persist: bool = True,
) -> ValidationDatasetBuilder:
    """Compose a builder backed by optional in-memory partitions."""
    layout = StorageLayout(Path("unused-root"))
    datastore = _InMemoryDataStore()
    factors_repository = FactorsRepository(layout, datastore)  # type: ignore[arg-type]
    label_repository = LabelRepository(layout, datastore)  # type: ignore[arg-type]
    if persist:
        if factors is not None:
            factors_repository.save(factors, **_SYMBOL_PARTITION_KWARGS)
        if labels is not None:
            label_repository.save(
                labels,
                exchange=_EXCHANGE,
                market=_MARKET,
                symbol=_SYMBOL,
                timeframe=_TIMEFRAME,
                year=_YEAR,
            )
    return ValidationDatasetBuilder(factors_repository, label_repository)


def test_assemble_successful_join_preserves_factor_columns_and_appends_label() -> None:
    """Matching Factors and Labels produce the validation dataset schema."""
    factors = _factors_frame()
    labels = _labels_frame()
    builder = _builder(persist=False)

    result = builder.assemble(factors, labels)

    assert result.height == factors.height
    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == VALIDATION_DATASET_SCHEMA
    for column in FACTOR_CANONICAL_COLUMN_ORDER:
        assert column in result.columns
    assert VALIDATION_LABEL_COLUMNS[0] in result.columns
    assert result.get_column("future_return_1").to_list() == [
        0.01,
        0.02,
        0.03,
    ]


def test_assemble_column_ordering_matches_canonical_order() -> None:
    """Assembled output follows validation-dataset canonical column order."""
    factors = _factors_frame().select(list(reversed(FACTOR_CANONICAL_COLUMN_ORDER)))
    labels = _labels_frame()
    builder = _builder(persist=False)

    result = builder.assemble(factors, labels)

    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == VALIDATION_DATASET_SCHEMA


def test_assemble_schema_dtypes_match_validation_dataset_schema() -> None:
    """Assembled columns are cast to VALIDATION_DATASET_SCHEMA dtypes."""
    factors = _factors_frame().with_columns(pl.col("lookback").cast(pl.Int64))
    labels = _labels_frame().with_columns(pl.col("future_return_1").cast(pl.Float32))
    builder = _builder(persist=False)

    result = builder.assemble(factors, labels)

    assert result.schema == VALIDATION_DATASET_SCHEMA
    for column in CANONICAL_COLUMN_ORDER:
        assert result.schema[column] == COLUMN_DTYPES[column]


def test_assemble_inner_join_drops_unmatched_factor_rows() -> None:
    """Factor rows without matching Labels keys are excluded."""
    factors = _factors_frame(open_times=_OPEN_TIMES)
    labels = _labels_frame(open_times=_OPEN_TIMES[1:])
    builder = _builder(persist=False)

    result = builder.assemble(factors, labels)

    assert result.get_column("open_time").to_list() == list(_OPEN_TIMES[1:])
    assert result.height == 2


def test_assemble_rejects_empty_join() -> None:
    """Disjoint Factors/Labels keys raise FVAL_VDB_EMPTY_JOIN."""
    factors = _factors_frame(open_times=_OPEN_TIMES[:2])
    labels = _labels_frame(open_times=(9_000_000_000_000, 9_000_003_600_000))
    builder = _builder(persist=False)

    with pytest.raises(FactorValidationError) as exc_info:
        builder.assemble(factors, labels)
    assert exc_info.value.error_code == "FVAL_VDB_EMPTY_JOIN"


def test_assemble_rejects_missing_labels_columns() -> None:
    """Labels missing future_return_1 raise FVAL_VDB_MISSING_COLUMNS."""
    factors = _factors_frame()
    labels = _labels_frame().drop("future_return_1")
    builder = _builder(persist=False)

    with pytest.raises(FactorValidationError) as exc_info:
        builder.assemble(factors, labels)
    assert exc_info.value.error_code == "FVAL_VDB_MISSING_COLUMNS"
    assert exc_info.value.details["side"] == "labels"


def test_assemble_rejects_duplicate_label_join_keys() -> None:
    """Duplicate Labels primary keys raise FVAL_VDB_DUPLICATE_KEYS."""
    factors = _factors_frame(open_times=_OPEN_TIMES[:1])
    labels = pl.concat([_labels_frame(open_times=_OPEN_TIMES[:1])] * 2)
    builder = _builder(persist=False)

    with pytest.raises(FactorValidationError) as exc_info:
        builder.assemble(factors, labels)
    assert exc_info.value.error_code == "FVAL_VDB_DUPLICATE_KEYS"
    assert exc_info.value.details["side"] == "labels"


def test_assemble_rejects_duplicate_factor_observation_keys() -> None:
    """Duplicate Factors observation keys raise FVAL_VDB_DUPLICATE_KEYS."""
    factors = _factors_frame(open_times=_OPEN_TIMES[:1], duplicate_observation=True)
    labels = _labels_frame(open_times=_OPEN_TIMES[:1])
    builder = _builder(persist=False)

    with pytest.raises(FactorValidationError) as exc_info:
        builder.assemble(factors, labels)
    assert exc_info.value.error_code == "FVAL_VDB_DUPLICATE_KEYS"
    assert exc_info.value.details["side"] == "factors"


def test_assemble_rejects_missing_primary_keys() -> None:
    """Missing join-key columns raise FVAL_VDB_MISSING_PRIMARY_KEY."""
    factors = _factors_frame().drop("symbol")
    labels = _labels_frame()
    builder = _builder(persist=False)

    with pytest.raises(FactorValidationError) as exc_info:
        builder.assemble(factors, labels)
    assert exc_info.value.error_code == "FVAL_VDB_MISSING_PRIMARY_KEY"
    assert exc_info.value.details["side"] == "factors"
    assert "symbol" in exc_info.value.details["missing_columns"]


def test_assemble_does_not_mutate_inputs() -> None:
    """Assemble must not mutate caller-supplied Factors or Labels frames."""
    factors = _factors_frame()
    labels = _labels_frame()
    before_factors = factors.clone()
    before_labels = labels.clone()
    builder = _builder(persist=False)

    builder.assemble(factors, labels)

    assert_frame_equal(factors, before_factors)
    assert_frame_equal(labels, before_labels)


def test_build_loads_repositories_and_assembles_dataset() -> None:
    """build() loads Factors and Labels panel partitions then assembles them."""
    factors = _factors_frame()
    labels = _labels_frame()
    builder = _builder(factors=factors, labels=labels)

    result = builder.build(**_PANEL_KWARGS, symbols=(_SYMBOL,))

    assert tuple(result.columns) == CANONICAL_COLUMN_ORDER
    assert result.schema == VALIDATION_DATASET_SCHEMA
    assert result.height == factors.height
    assert PRIMARY_KEY_COLUMNS == ("symbol", "timeframe", "open_time")


def test_build_concatenates_multi_symbol_panel() -> None:
    """build() concatenates multiple symbol partitions into one panel."""
    layout = StorageLayout(Path("unused-root"))
    datastore = _InMemoryDataStore()
    factors_repository = FactorsRepository(layout, datastore)  # type: ignore[arg-type]
    label_repository = LabelRepository(layout, datastore)  # type: ignore[arg-type]

    for symbol, factor_value in (("BTCUSDT", 0.1), ("ETHUSDT", 0.2)):
        factors = _factors_frame().with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit(factor_value).alias("factor_value"),
        )
        labels = _labels_frame().with_columns(pl.lit(symbol).alias("symbol"))
        factors_repository.save(
            factors,
            manager=_MANAGER,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )
        label_repository.save(
            labels,
            exchange=_EXCHANGE,
            market=_MARKET,
            symbol=symbol,
            timeframe=_TIMEFRAME,
            year=_YEAR,
        )

    builder = ValidationDatasetBuilder(factors_repository, label_repository)
    result = builder.build(**_PANEL_KWARGS, symbols=("BTCUSDT", "ETHUSDT"))

    assert result.height == len(_OPEN_TIMES) * 2
    assert sorted(result.get_column("symbol").unique().to_list()) == ["BTCUSDT", "ETHUSDT"]


def test_build_propagates_missing_labels_partition() -> None:
    """Missing Labels for every Factors symbol raises FVAL_VDB_EMPTY_PANEL."""
    builder = _builder(factors=_factors_frame(), labels=None)

    with pytest.raises(FactorValidationError) as exc_info:
        builder.build(**_PANEL_KWARGS, symbols=(_SYMBOL,))
    assert exc_info.value.error_code == "FVAL_VDB_EMPTY_PANEL"

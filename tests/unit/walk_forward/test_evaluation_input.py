"""Unit tests for CQROS Walk-Forward evaluation-input adapter."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cqros.alpha.schema import CANONICAL_COLUMN_ORDER as ALPHA_COLUMNS
from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.types import FilePath
from cqros.factor_selection.schema import (
    CANONICAL_COLUMN_ORDER as FACTOR_SELECTION_COLUMNS,
)
from cqros.factor_selection.schema import (
    COLUMN_DTYPES as FACTOR_SELECTION_DTYPES,
)
from cqros.factor_selection.schema import FactorSelectionStatus
from cqros.factors import FactorsRepository, FactorStatus
from cqros.factors.schema import (
    CANONICAL_COLUMN_ORDER as FACTOR_CANONICAL_COLUMN_ORDER,
)
from cqros.factors.schema import (
    COLUMN_DTYPES as FACTOR_COLUMN_DTYPES,
)
from cqros.labels.schema import (
    CANONICAL_COLUMN_ORDER as LABEL_CANONICAL_COLUMN_ORDER,
)
from cqros.labels.schema import (
    COLUMN_DTYPES as LABEL_COLUMN_DTYPES,
)
from cqros.labels.schema import LABEL_COLUMNS
from cqros.predictions.schema import CANONICAL_COLUMN_ORDER as PREDICTION_COLUMNS
from cqros.regime.schema import CANONICAL_COLUMN_ORDER as REGIME_COLUMNS
from cqros.signals.schema import CANONICAL_COLUMN_ORDER as SIGNAL_COLUMNS
from cqros.storage import LabelRepository, StorageLayout
from cqros.storage.exceptions import DatasetNotFoundError
from cqros.walk_forward import (
    OBSERVATION_JOIN_KEYS,
    TARGET_COLUMN,
    WALK_FORWARD_EVALUATION_COLUMNS,
    WalkForwardError,
    WalkForwardInputBuilder,
    assemble_walk_forward_input,
)
from cqros.walk_forward.engine import SimpleWalkForwardEngine

_MANAGER = "default"
_EXCHANGE = EXCHANGE_BINANCE
_MARKET = MARKET_USDT_PERPETUAL
_SYMBOL_BTC = "BTCUSDT"
_SYMBOL_ETH = "ETHUSDT"
_TIMEFRAME_1H = "1h"
_TIMEFRAME_4H = "4h"
_YEAR = 2026
_OPEN_TIMES = (1_700_000_000_000, 1_700_003_600_000, 1_700_007_200_000)
_FACTOR_NAME = "momentum"
_FACTOR_VERSION = "1.0.0"
_FACTOR_CATEGORY = "price"
_FACTOR_GROUP = "alpha"


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


def _factor_selection_frame(
    *,
    factor_names: list[str] | None = None,
    timeframes: list[str] | None = None,
    selected: list[bool] | None = None,
    selection_ics: list[float] | None = None,
    selected_directions: list[int] | None = None,
) -> pl.DataFrame:
    """Build a canonical Factor Selection frame for adapter tests."""
    from cqros.factor_selection.orientation import FACTOR_ORIENTATION_POLICY

    factor_names = factor_names if factor_names is not None else [_FACTOR_NAME]
    count = len(factor_names)
    timeframes = timeframes if timeframes is not None else [_TIMEFRAME_1H] * count
    selected = selected if selected is not None else [True] * count
    selection_ics = selection_ics if selection_ics is not None else [0.08] * count
    selected_directions = (
        selected_directions
        if selected_directions is not None
        else [1 if value >= 0 else -1 for value in selection_ics]
    )
    return pl.DataFrame(
        {
            "factor_name": factor_names,
            "factor_version": [_FACTOR_VERSION] * count,
            "timeframe": timeframes,
            "selection_time": [_OPEN_TIMES[0]] * count,
            "factor_category": [_FACTOR_CATEGORY] * count,
            "selected": selected,
            "selection_score": [0.12] * count,
            "selection_rank": list(range(1, count + 1)),
            "selection_reason": ["v1_default_selection"] * count,
            "selection_ic": selection_ics,
            "selected_direction": selected_directions,
            "orientation_policy": [FACTOR_ORIENTATION_POLICY] * count,
            "status": [
                (
                    FactorSelectionStatus.SELECTED.value
                    if flag
                    else FactorSelectionStatus.REJECTED.value
                )
                for flag in selected
            ],
        },
        schema=dict(FACTOR_SELECTION_DTYPES),
    ).select(list(FACTOR_SELECTION_COLUMNS))


def _factors_frame(
    *,
    symbol: str = _SYMBOL_BTC,
    timeframe: str = _TIMEFRAME_1H,
    open_times: tuple[int, ...] = _OPEN_TIMES,
    factor_name: str = _FACTOR_NAME,
) -> pl.DataFrame:
    """Build a canonical Factors frame for adapter tests."""
    rows = len(open_times)
    data = {
        "symbol": [symbol] * rows,
        "timeframe": [timeframe] * rows,
        "open_time": list(open_times),
        "factor_name": [factor_name] * rows,
        "factor_version": [_FACTOR_VERSION] * rows,
        "factor_category": [_FACTOR_CATEGORY] * rows,
        "factor_group": [_FACTOR_GROUP] * rows,
        "factor_value": [0.1 * float(index + 1) for index in range(rows)],
        "lookback": [20] * rows,
        "prediction_horizon": [1] * rows,
        "enabled": [True] * rows,
        "status": [FactorStatus.ACTIVE.value] * rows,
    }
    return pl.DataFrame(data, schema=dict(FACTOR_COLUMN_DTYPES)).select(
        list(FACTOR_CANONICAL_COLUMN_ORDER)
    )


def _labels_frame(
    *,
    symbol: str = _SYMBOL_BTC,
    timeframe: str = _TIMEFRAME_1H,
    open_times: tuple[int, ...] = _OPEN_TIMES,
    future_returns: list[float] | None = None,
) -> pl.DataFrame:
    """Build a Labels frame containing ``future_return_1``."""
    rows = len(open_times)
    returns = (
        future_returns
        if future_returns is not None
        else [0.01 * float(offset + 1) for offset in range(rows)]
    )
    data: dict[str, object] = {
        "symbol": [symbol] * rows,
        "timeframe": [timeframe] * rows,
        "open_time": list(open_times),
        "future_return_1": returns,
        "future_return_5": [0.05] * rows,
        "future_return_10": [0.10] * rows,
        "future_return_20": [0.20] * rows,
        "direction_1": [1] * rows,
        "direction_5": [1] * rows,
        "direction_10": [0] * rows,
        "direction_20": [0] * rows,
    }
    return pl.DataFrame(data, schema=dict(LABEL_COLUMN_DTYPES)).select(
        list(LABEL_CANONICAL_COLUMN_ORDER)
    )


def test_successful_enrichment_produces_walk_forward_engine_input() -> None:
    """Selected Factor Selection rows plus Labels yield valid engine input."""
    selection = _factor_selection_frame()
    factors = _factors_frame()
    labels = _labels_frame()

    result = assemble_walk_forward_input(selection, factors, labels)

    assert tuple(result.columns) == WALK_FORWARD_EVALUATION_COLUMNS
    assert TARGET_COLUMN in result.columns
    assert result.height == factors.height
    assert result.get_column(TARGET_COLUMN).to_list() == [0.01, 0.02, 0.03]
    assert result.get_column("selection_time").to_list() == list(_OPEN_TIMES)
    assert result.get_column("selected").to_list() == [True, True, True]

    engine_output = SimpleWalkForwardEngine(
        train_window=2,
        test_window=1,
        step_size=1,
    ).build(result)
    assert engine_output.height >= 1


def test_join_uses_symbol_timeframe_open_time_alignment() -> None:
    """Matching rows align exclusively on observation identity keys."""
    assert OBSERVATION_JOIN_KEYS == ("symbol", "timeframe", "open_time")
    selection = _factor_selection_frame()
    factors = _factors_frame(open_times=_OPEN_TIMES)
    labels = _labels_frame(
        open_times=_OPEN_TIMES,
        future_returns=[0.11, 0.22, 0.33],
    )

    result = assemble_walk_forward_input(selection, factors, labels)

    matched = result.select(["symbol", "timeframe", "open_time", TARGET_COLUMN]).sort(["open_time"])
    expected = pl.DataFrame(
        {
            "symbol": [_SYMBOL_BTC] * 3,
            "timeframe": [_TIMEFRAME_1H] * 3,
            "open_time": list(_OPEN_TIMES),
            TARGET_COLUMN: [0.11, 0.22, 0.33],
        }
    )
    assert_frame_equal(matched, expected)


def test_no_cross_symbol_contamination() -> None:
    """BTCUSDT observations never receive ETHUSDT target values."""
    selection = _factor_selection_frame()
    factors = pl.concat(
        [
            _factors_frame(symbol=_SYMBOL_BTC, open_times=_OPEN_TIMES[:1]),
            _factors_frame(symbol=_SYMBOL_ETH, open_times=_OPEN_TIMES[:1]),
        ]
    )
    labels = pl.concat(
        [
            _labels_frame(
                symbol=_SYMBOL_BTC,
                open_times=_OPEN_TIMES[:1],
                future_returns=[0.42],
            ),
            _labels_frame(
                symbol=_SYMBOL_ETH,
                open_times=_OPEN_TIMES[:1],
                future_returns=[0.99],
            ),
        ]
    )

    result = assemble_walk_forward_input(selection, factors, labels)

    btc = result.filter(pl.col("symbol") == _SYMBOL_BTC)
    eth = result.filter(pl.col("symbol") == _SYMBOL_ETH)
    assert btc.get_column(TARGET_COLUMN).to_list() == [0.42]
    assert eth.get_column(TARGET_COLUMN).to_list() == [0.99]


def test_no_cross_timeframe_contamination() -> None:
    """1h observations never receive 4h target values."""
    selection = _factor_selection_frame(
        factor_names=[_FACTOR_NAME, _FACTOR_NAME],
        timeframes=[_TIMEFRAME_1H, _TIMEFRAME_4H],
        selected=[True, True],
    )
    factors = pl.concat(
        [
            _factors_frame(timeframe=_TIMEFRAME_1H, open_times=_OPEN_TIMES[:1]),
            _factors_frame(timeframe=_TIMEFRAME_4H, open_times=_OPEN_TIMES[:1]),
        ]
    )
    labels = pl.concat(
        [
            _labels_frame(
                timeframe=_TIMEFRAME_1H,
                open_times=_OPEN_TIMES[:1],
                future_returns=[0.15],
            ),
            _labels_frame(
                timeframe=_TIMEFRAME_4H,
                open_times=_OPEN_TIMES[:1],
                future_returns=[0.75],
            ),
        ]
    )

    result = assemble_walk_forward_input(selection, factors, labels)

    one_hour = result.filter(pl.col("timeframe") == _TIMEFRAME_1H)
    four_hour = result.filter(pl.col("timeframe") == _TIMEFRAME_4H)
    assert one_hour.get_column(TARGET_COLUMN).to_list() == [0.15]
    assert four_hour.get_column(TARGET_COLUMN).to_list() == [0.75]


def test_missing_target_rows_are_dropped_not_fabricated() -> None:
    """Unmatched Factors rows are excluded; returns are never invented."""
    selection = _factor_selection_frame()
    factors = _factors_frame(open_times=_OPEN_TIMES)
    labels = _labels_frame(open_times=_OPEN_TIMES[1:], future_returns=[0.02, 0.03])

    result = assemble_walk_forward_input(selection, factors, labels)

    assert result.get_column("open_time").to_list() == list(_OPEN_TIMES[1:])
    assert result.get_column(TARGET_COLUMN).to_list() == [0.02, 0.03]


def test_missing_target_column_raises() -> None:
    """Labels without ``future_return_1`` fail instead of fabricating values."""
    selection = _factor_selection_frame()
    factors = _factors_frame(open_times=_OPEN_TIMES[:1])
    labels = _labels_frame(open_times=_OPEN_TIMES[:1]).drop(TARGET_COLUMN)

    with pytest.raises(WalkForwardError) as exc_info:
        assemble_walk_forward_input(selection, factors, labels)
    assert exc_info.value.error_code == "WF_EVAL_MISSING_COLUMNS"
    missing_columns = exc_info.value.details["missing_columns"]
    assert isinstance(missing_columns, tuple)
    assert TARGET_COLUMN in missing_columns


def test_empty_factors_labels_join_raises() -> None:
    """Completely unmatched Factors/Labels keys raise WF_EVAL_EMPTY_JOIN."""
    selection = _factor_selection_frame()
    factors = _factors_frame(open_times=_OPEN_TIMES[:1])
    labels = _labels_frame(open_times=(9_999_999_999_999,))

    with pytest.raises(WalkForwardError) as exc_info:
        assemble_walk_forward_input(selection, factors, labels)
    assert exc_info.value.error_code == "WF_EVAL_EMPTY_JOIN"


def test_duplicate_label_keys_raise() -> None:
    """Duplicate ``(symbol, timeframe, open_time)`` Labels keys are rejected."""
    selection = _factor_selection_frame()
    factors = _factors_frame(open_times=_OPEN_TIMES[:1])
    labels = pl.concat(
        [
            _labels_frame(open_times=_OPEN_TIMES[:1], future_returns=[0.01]),
            _labels_frame(open_times=_OPEN_TIMES[:1], future_returns=[0.99]),
        ]
    )

    with pytest.raises(WalkForwardError) as exc_info:
        assemble_walk_forward_input(selection, factors, labels)
    assert exc_info.value.error_code == "WF_EVAL_DUPLICATE_KEYS"
    assert exc_info.value.details["side"] == "labels"


def test_duplicate_selection_join_keys_raise() -> None:
    """Duplicate Factor Selection identity keys are rejected before join."""
    selection = pl.concat(
        [
            _factor_selection_frame(),
            _factor_selection_frame(),
        ]
    )
    factors = _factors_frame(open_times=_OPEN_TIMES[:1])
    labels = _labels_frame(open_times=_OPEN_TIMES[:1])

    with pytest.raises(WalkForwardError) as exc_info:
        assemble_walk_forward_input(selection, factors, labels)
    assert exc_info.value.error_code == "WF_EVAL_DUPLICATE_KEYS"
    assert exc_info.value.details["side"] == "factor_selection"


def test_leakage_boundary_future_return_not_in_downstream_contracts() -> None:
    """``future_return_1`` must not appear in FS/Alpha/Regime/Predictions/Signals."""
    selection = _factor_selection_frame()
    assert TARGET_COLUMN not in selection.columns
    assert TARGET_COLUMN not in FACTOR_SELECTION_COLUMNS
    assert TARGET_COLUMN not in ALPHA_COLUMNS
    assert TARGET_COLUMN not in REGIME_COLUMNS
    assert TARGET_COLUMN not in PREDICTION_COLUMNS
    assert TARGET_COLUMN not in SIGNAL_COLUMNS

    # Adapter output is evaluation-only and does not rewrite Factor Selection.
    result = assemble_walk_forward_input(
        selection,
        _factors_frame(open_times=_OPEN_TIMES[:1]),
        _labels_frame(open_times=_OPEN_TIMES[:1]),
    )
    assert TARGET_COLUMN in result.columns
    assert TARGET_COLUMN not in selection.columns
    assert_frame_equal(selection, _factor_selection_frame())


def test_assemble_is_deterministic() -> None:
    """Identical inputs produce identical evaluation frames."""
    selection = _factor_selection_frame()
    factors = pl.concat(
        [
            _factors_frame(symbol=_SYMBOL_ETH, open_times=_OPEN_TIMES),
            _factors_frame(symbol=_SYMBOL_BTC, open_times=_OPEN_TIMES),
        ]
    )
    labels = pl.concat(
        [
            _labels_frame(symbol=_SYMBOL_ETH, open_times=_OPEN_TIMES),
            _labels_frame(symbol=_SYMBOL_BTC, open_times=_OPEN_TIMES),
        ]
    )

    first = assemble_walk_forward_input(selection, factors, labels)
    second = assemble_walk_forward_input(selection, factors, labels)
    assert_frame_equal(first, second)


def test_inputs_are_immutable() -> None:
    """assemble_walk_forward_input must not mutate caller frames."""
    selection = _factor_selection_frame()
    factors = _factors_frame(open_times=_OPEN_TIMES[:2])
    labels = _labels_frame(open_times=_OPEN_TIMES[:2])
    selection_before = selection.clone()
    factors_before = factors.clone()
    labels_before = labels.clone()

    assemble_walk_forward_input(selection, factors, labels)

    assert_frame_equal(selection, selection_before)
    assert_frame_equal(factors, factors_before)
    assert_frame_equal(labels, labels_before)


def test_builder_loads_partitions_and_assembles(tmp_path: Path) -> None:
    """WalkForwardInputBuilder loads Factors/Labels and assembles evaluation input."""
    layout = StorageLayout(tmp_path)
    datastore = _InMemoryDataStore()
    factors_repository = FactorsRepository(layout, datastore)  # type: ignore[arg-type]
    label_repository = LabelRepository(layout, datastore)  # type: ignore[arg-type]
    factors_repository.save(
        _factors_frame(),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL_BTC,
        timeframe=_TIMEFRAME_1H,
        year=_YEAR,
    )
    label_repository.save(
        _labels_frame(),
        exchange=_EXCHANGE,
        market=_MARKET,
        symbol=_SYMBOL_BTC,
        timeframe=_TIMEFRAME_1H,
        year=_YEAR,
    )
    builder = WalkForwardInputBuilder(factors_repository, label_repository)

    result = builder.build(
        _factor_selection_frame(),
        manager=_MANAGER,
        exchange=_EXCHANGE,
        market=_MARKET,
        timeframe=_TIMEFRAME_1H,
        year=_YEAR,
        symbols=(_SYMBOL_BTC,),
    )

    assert result.height == 3
    assert TARGET_COLUMN in result.columns
    assert LABEL_COLUMNS[0] == TARGET_COLUMN

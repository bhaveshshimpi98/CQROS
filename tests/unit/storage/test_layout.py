"""Unit tests for CQROS storage layout path composition."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from cqros.core.constants import (
    FILE_EXTENSION_PARQUET,
    STORAGE_DIR_ACCOUNTING,
    STORAGE_DIR_ANALYTICS,
    STORAGE_DIR_BACKTESTING,
    STORAGE_DIR_CACHE,
    STORAGE_DIR_EXECUTIONS,
    STORAGE_DIR_EXIT_ENGINE,
    STORAGE_DIR_FACTOR_SELECTION,
    STORAGE_DIR_FACTOR_VALIDATION,
    STORAGE_DIR_FACTORS,
    STORAGE_DIR_FEATURES,
    STORAGE_DIR_LABELS,
    STORAGE_DIR_MODELS,
    STORAGE_DIR_MONITORING,
    STORAGE_DIR_ORDERS,
    STORAGE_DIR_PERFORMANCE,
    STORAGE_DIR_PORTFOLIO_RISK,
    STORAGE_DIR_PORTFOLIOS,
    STORAGE_DIR_POSITIONS,
    STORAGE_DIR_PREDICTIONS,
    STORAGE_DIR_PROCESSED,
    STORAGE_DIR_PURGED_CV,
    STORAGE_DIR_PURGED_CV_EVALUATION,
    STORAGE_DIR_PYRAMIDING,
    STORAGE_DIR_RAW,
    STORAGE_DIR_REPORTING,
    STORAGE_DIR_REPORTS,
    STORAGE_DIR_RISKS,
    STORAGE_DIR_SIGNALS,
    STORAGE_DIR_THRESHOLDS,
    STORAGE_DIR_TRADE_MANAGEMENT,
    STORAGE_DIR_TRAINING,
    STORAGE_DIR_WALK_FORWARD,
    STORAGE_DIR_WALK_FORWARD_EVALUATION,
)
from cqros.storage import StorageLayout
from cqros.storage.layout import StorageLayout as StorageLayoutDirect

_EXCHANGE = "binance"
_MARKET = "perpetual"
_SYMBOL = "BTCUSDT"
_TIMEFRAME = "1m"
_YEAR = 2026


@pytest.fixture
def layout() -> StorageLayout:
    """Return a layout rooted at a relative data directory."""
    return StorageLayout(Path("data"))


def test_storage_layout_is_frozen_dataclass() -> None:
    """StorageLayout is an immutable slotted dataclass."""
    layout = StorageLayout(Path("data"))
    assert is_dataclass(layout)
    assert isinstance(layout.root, Path)
    with pytest.raises(FrozenInstanceError):
        layout.root = Path("other")  # type: ignore[misc]


def test_storage_layout_normalizes_string_root() -> None:
    """String roots are coerced to pathlib.Path without filesystem access."""
    layout = StorageLayout("data")
    assert layout.root == Path("data")
    assert isinstance(layout.root, Path)


def test_storage_layout_is_exported_from_package() -> None:
    """Package export matches the layout module class."""
    assert StorageLayout is StorageLayoutDirect


def test_raw_ohlcv_path(layout: StorageLayout) -> None:
    """Raw OHLCV paths follow raw/ohlcv/exchange/market/symbol/timeframe."""
    path = layout.raw_ohlcv_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        STORAGE_DIR_RAW,
        "ohlcv",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )
    assert isinstance(path, Path)


def test_raw_funding_path(layout: StorageLayout) -> None:
    """Raw funding paths follow raw/funding/exchange/market/symbol/timeframe."""
    path = layout.raw_funding_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        STORAGE_DIR_RAW,
        "funding",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_raw_open_interest_path(layout: StorageLayout) -> None:
    """Raw open-interest paths use the open_interest dataset segment."""
    path = layout.raw_open_interest_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_RAW,
        "open_interest",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_raw_taker_volume_path(layout: StorageLayout) -> None:
    """Raw taker-volume paths use the taker_volume dataset segment."""
    path = layout.raw_taker_volume_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_RAW,
        "taker_volume",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_raw_global_long_short_account_ratio_path(layout: StorageLayout) -> None:
    """Global long/short account-ratio paths use a dedicated dataset segment."""
    path = layout.raw_global_long_short_account_ratio_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_RAW,
        "global_long_short_account_ratio",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_raw_top_long_short_account_ratio_path(layout: StorageLayout) -> None:
    """Top-trader account-ratio paths use a dedicated dataset segment."""
    path = layout.raw_top_long_short_account_ratio_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_RAW,
        "top_long_short_account_ratio",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_raw_top_long_short_position_ratio_path(layout: StorageLayout) -> None:
    """Top-trader position-ratio paths use a dedicated dataset segment."""
    path = layout.raw_top_long_short_position_ratio_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_RAW,
        "top_long_short_position_ratio",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_raw_liquidation_path(layout: StorageLayout) -> None:
    """Raw liquidation paths use the liquidation dataset segment."""
    path = layout.raw_liquidation_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_RAW,
        "liquidation",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_processed_ohlcv_path(layout: StorageLayout) -> None:
    """Processed OHLCV paths follow processed/ohlcv/... partitioning."""
    path = layout.processed_ohlcv_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        STORAGE_DIR_PROCESSED,
        "ohlcv",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_processed_funding_path(layout: StorageLayout) -> None:
    """Processed funding paths use the processed funding dataset segment."""
    path = layout.processed_funding_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_PROCESSED,
        "funding",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_processed_open_interest_path(layout: StorageLayout) -> None:
    """Processed open-interest paths use the processed hierarchy."""
    path = layout.processed_open_interest_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_PROCESSED,
        "open_interest",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_processed_taker_volume_path(layout: StorageLayout) -> None:
    """Processed taker-volume paths use the processed hierarchy."""
    path = layout.processed_taker_volume_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_PROCESSED,
        "taker_volume",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_processed_long_short_ratio_paths(layout: StorageLayout) -> None:
    """Processed long/short ratio datasets use dedicated processed segments."""
    global_path = layout.processed_global_long_short_account_ratio_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    top_account = layout.processed_top_long_short_account_ratio_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    top_position = layout.processed_top_long_short_position_ratio_path(
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert global_path.parts[-6] == "global_long_short_account_ratio"
    assert top_account.parts[-6] == "top_long_short_account_ratio"
    assert top_position.parts[-6] == "top_long_short_position_ratio"
    assert all(
        STORAGE_DIR_PROCESSED in path.parts for path in (global_path, top_account, top_position)
    )


def test_feature_path(layout: StorageLayout) -> None:
    """Feature paths follow features/exchange/market/symbol/timeframe."""
    path = layout.feature_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        STORAGE_DIR_FEATURES,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_label_path(layout: StorageLayout) -> None:
    """Label paths follow labels/exchange/market/symbol/timeframe."""
    path = layout.label_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        STORAGE_DIR_LABELS,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_training_path(layout: StorageLayout) -> None:
    """Training paths follow training/exchange/market/symbol/timeframe."""
    path = layout.training_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        STORAGE_DIR_TRAINING,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_signal_path(layout: StorageLayout) -> None:
    """Signal paths follow signals/exchange/market/symbol/timeframe."""
    path = layout.signal_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        STORAGE_DIR_SIGNALS,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_portfolio_path(layout: StorageLayout) -> None:
    """Portfolio paths follow portfolios/exchange/market/symbol/timeframe."""
    path = layout.portfolio_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        STORAGE_DIR_PORTFOLIOS,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_risk_path(layout: StorageLayout) -> None:
    """Risk paths follow risks/policy/exchange/market/symbol/timeframe."""
    path = layout.risk_path(
        "fixed_risk",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_RISKS,
        "fixed_risk",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_order_path(layout: StorageLayout) -> None:
    """Order paths follow orders/manager/exchange/market/symbol/timeframe."""
    path = layout.order_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_ORDERS,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_execution_path(layout: StorageLayout) -> None:
    """Execution paths follow executions/manager/exchange/market/symbol/timeframe."""
    path = layout.execution_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_EXECUTIONS,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_position_path(layout: StorageLayout) -> None:
    """Position paths follow positions/manager/exchange/market/symbol/timeframe."""
    path = layout.position_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_POSITIONS,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_accounting_path(layout: StorageLayout) -> None:
    """Accounting paths follow accounting/manager/exchange/market/symbol/timeframe."""
    path = layout.accounting_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_ACCOUNTING,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_portfolio_risk_path(layout: StorageLayout) -> None:
    """Portfolio-risk paths follow portfolio_risk/manager/exchange/market/symbol/timeframe."""
    path = layout.portfolio_risk_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_PORTFOLIO_RISK,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_trade_management_path(layout: StorageLayout) -> None:
    """Trade-management paths follow trade_management/manager/exchange/market/symbol/timeframe."""
    path = layout.trade_management_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_TRADE_MANAGEMENT,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_pyramiding_path(layout: StorageLayout) -> None:
    """Pyramiding paths follow pyramiding/manager/exchange/market/symbol/timeframe."""
    path = layout.pyramiding_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_PYRAMIDING,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_exit_engine_path(layout: StorageLayout) -> None:
    """Exit-engine paths follow exit_engine/manager/exchange/market/symbol/timeframe."""
    path = layout.exit_engine_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_EXIT_ENGINE,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_backtesting_path(layout: StorageLayout) -> None:
    """Backtesting paths follow backtesting/manager/exchange/market/symbol/timeframe."""
    path = layout.backtesting_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_BACKTESTING,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_performance_path(layout: StorageLayout) -> None:
    """Performance paths follow performance/manager/exchange/market/symbol/timeframe."""
    path = layout.performance_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_PERFORMANCE,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_analytics_path(layout: StorageLayout) -> None:
    """Analytics paths follow analytics/manager/exchange/market/symbol/timeframe."""
    path = layout.analytics_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_ANALYTICS,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_reporting_path(layout: StorageLayout) -> None:
    """Reporting paths follow reporting/manager/exchange/market/symbol/timeframe."""
    path = layout.reporting_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_REPORTING,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_monitoring_path(layout: StorageLayout) -> None:
    """Monitoring paths follow monitoring/manager/exchange/market/symbol/timeframe."""
    path = layout.monitoring_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_MONITORING,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_factor_validation_path(layout: StorageLayout) -> None:
    """Factor validation paths follow factor_validation/manager/exchange/market/timeframe."""
    path = layout.factor_validation_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_FACTOR_VALIDATION,
        "simple",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_factor_selection_path(layout: StorageLayout) -> None:
    """Factor selection paths follow factor_selection/manager/exchange/market/timeframe."""
    path = layout.factor_selection_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_FACTOR_SELECTION,
        "simple",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_walk_forward_path(layout: StorageLayout) -> None:
    """Walk-forward paths follow walk_forward/manager/exchange/market/timeframe."""
    path = layout.walk_forward_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_WALK_FORWARD,
        "simple",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_walk_forward_evaluation_path(layout: StorageLayout) -> None:
    """Evaluation paths mirror walk_forward under walk_forward_evaluation."""
    path = layout.walk_forward_evaluation_path(
        "default",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_WALK_FORWARD_EVALUATION,
        "default",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_purged_cv_path(layout: StorageLayout) -> None:
    """Purged-CV paths follow purged_cv/manager/exchange/market/timeframe."""
    path = layout.purged_cv_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_PURGED_CV,
        "simple",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_purged_cv_evaluation_path(layout: StorageLayout) -> None:
    """Evaluation paths mirror purged_cv under purged_cv_evaluation."""
    path = layout.purged_cv_evaluation_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_PURGED_CV_EVALUATION,
        "simple",
        _EXCHANGE,
        _MARKET,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_factors_path(layout: StorageLayout) -> None:
    """Factors paths follow factors/manager/exchange/market/symbol/timeframe."""
    path = layout.factors_path(
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_FACTORS,
        "simple",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_prediction_path(layout: StorageLayout) -> None:
    """Prediction paths follow predictions/framework/model/version/.../year."""
    path = layout.prediction_path(
        "lightgbm",
        "alpha-lgbm",
        "1.0.0",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert path == Path(
        "data",
        STORAGE_DIR_PREDICTIONS,
        "lightgbm",
        "alpha-lgbm",
        "1.0.0",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_threshold_path(layout: StorageLayout) -> None:
    """Threshold paths follow thresholds/model/version/symbol/timeframe."""
    path = layout.threshold_path("alpha-lgbm", "1.0.0", _SYMBOL, _TIMEFRAME)
    assert path == Path(
        "data",
        STORAGE_DIR_THRESHOLDS,
        "alpha-lgbm",
        "1.0.0",
        _SYMBOL,
        _TIMEFRAME,
        f"thresholds{FILE_EXTENSION_PARQUET}",
    )


def test_model_path(layout: StorageLayout) -> None:
    """Model paths follow models/framework/model_name/version."""
    path = layout.model_path("lightgbm", "alpha-lgbm", "1.0.0")
    assert path == Path(
        "data",
        STORAGE_DIR_MODELS,
        "lightgbm",
        "alpha-lgbm",
        "1.0.0",
    )


def test_experiment_path(layout: StorageLayout) -> None:
    """Experiment paths follow experiments/exchange/market/symbol/timeframe."""
    path = layout.experiment_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        "experiments",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_report_path(layout: StorageLayout) -> None:
    """Report paths follow reports/exchange/market/symbol/timeframe."""
    path = layout.report_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        STORAGE_DIR_REPORTS,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_cache_path(layout: StorageLayout) -> None:
    """Cache paths follow cache/exchange/market/symbol/timeframe."""
    path = layout.cache_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert path == Path(
        "data",
        STORAGE_DIR_CACHE,
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        f"{_YEAR}{FILE_EXTENSION_PARQUET}",
    )


def test_layout_is_extensible_across_exchanges_and_markets() -> None:
    """Different exchanges and markets produce distinct path prefixes."""
    layout = StorageLayout(Path("lake"))
    binance = layout.raw_ohlcv_path("binance", "spot", "ETHUSDT", "1h", 2025)
    okx = layout.raw_ohlcv_path("okx", "futures", "ETHUSDT", "1h", 2025)
    assert binance == Path(
        "lake",
        STORAGE_DIR_RAW,
        "ohlcv",
        "binance",
        "spot",
        "ETHUSDT",
        "1h",
        f"2025{FILE_EXTENSION_PARQUET}",
    )
    assert okx == Path(
        "lake",
        STORAGE_DIR_RAW,
        "ohlcv",
        "okx",
        "futures",
        "ETHUSDT",
        "1h",
        f"2025{FILE_EXTENSION_PARQUET}",
    )
    assert binance != okx


def test_path_composition_is_deterministic(layout: StorageLayout) -> None:
    """Identical inputs always compose to the same path object value."""
    left = layout.feature_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    right = layout.feature_path(_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    assert left == right
    assert str(left) == str(right)


def test_all_methods_return_pathlib_path(layout: StorageLayout) -> None:
    """Every public path method returns pathlib.Path only."""
    partition_args = (_EXCHANGE, _MARKET, _SYMBOL, _TIMEFRAME, _YEAR)
    partition_methods = (
        layout.raw_ohlcv_path,
        layout.raw_funding_path,
        layout.raw_open_interest_path,
        layout.raw_taker_volume_path,
        layout.raw_global_long_short_account_ratio_path,
        layout.raw_top_long_short_account_ratio_path,
        layout.raw_top_long_short_position_ratio_path,
        layout.raw_liquidation_path,
        layout.processed_ohlcv_path,
        layout.processed_funding_path,
        layout.processed_open_interest_path,
        layout.processed_taker_volume_path,
        layout.processed_global_long_short_account_ratio_path,
        layout.processed_top_long_short_account_ratio_path,
        layout.processed_top_long_short_position_ratio_path,
        layout.feature_path,
        layout.label_path,
        layout.training_path,
        layout.signal_path,
        layout.experiment_path,
        layout.report_path,
        layout.cache_path,
    )
    for method in partition_methods:
        result = method(*partition_args)
        assert isinstance(result, Path)
        assert result.suffix == FILE_EXTENSION_PARQUET

    model_directory = layout.model_path("lightgbm", "alpha-lgbm", "1.0.0")
    assert isinstance(model_directory, Path)
    assert model_directory.name == "1.0.0"
    assert model_directory.suffix != FILE_EXTENSION_PARQUET

    prediction = layout.prediction_path(
        "lightgbm",
        "alpha-lgbm",
        "1.0.0",
        _EXCHANGE,
        _MARKET,
        _SYMBOL,
        _TIMEFRAME,
        _YEAR,
    )
    assert isinstance(prediction, Path)
    assert prediction.suffix == FILE_EXTENSION_PARQUET
    assert STORAGE_DIR_PREDICTIONS in prediction.parts

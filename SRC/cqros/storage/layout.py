"""CQROS canonical data-lake path layout.

Purpose:
    Compose deterministic ``pathlib.Path`` locations for research artifacts
    stored in the CQROS structured data lake.

Responsibilities:
    - Own the canonical directory hierarchy under a storage root
    - Compose paths for raw market datasets and derived artifacts
    - Remain free of file I/O, directory creation, validation, and
      business logic

Dependencies:
    Python standard library and ``cqros.core`` constants and type aliases.

Public API:
    ``StorageLayout``

Notes:
    Paths follow::

        {root}/…/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

    Directory segment names are plain identifiers (not Hive ``key=value``
    partitions) so additional exchanges and asset classes can be added
    without changing the layout contract. This module never touches the
    filesystem; callers are responsible for creating directories and
    reading or writing files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cqros.core.constants import (
    FILE_EXTENSION_PARQUET,
    STORAGE_DIR_ACCOUNTING,
    STORAGE_DIR_ALPHA,
    STORAGE_DIR_ANALYTICS,
    STORAGE_DIR_BACKTESTING,
    STORAGE_DIR_CACHE,
    STORAGE_DIR_EXECUTIONS,
    STORAGE_DIR_EXIT_ENGINE,
    STORAGE_DIR_FACTOR_COMBINATION,
    STORAGE_DIR_FACTOR_ORTHOGONALIZATION,
    STORAGE_DIR_FACTOR_SELECTION,
    STORAGE_DIR_FACTOR_TIMEFRAME_ANALYSIS,
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
    STORAGE_DIR_REGIME,
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
from cqros.core.types import Exchange, FilePath, Market, Symbol, Timeframe

__all__ = [
    "StorageLayout",
]

_DATASET_OHLCV: Final[str] = "ohlcv"
_DATASET_FUNDING: Final[str] = "funding"
_DATASET_OPEN_INTEREST: Final[str] = "open_interest"
_DATASET_TAKER_VOLUME: Final[str] = "taker_volume"
_DATASET_GLOBAL_LONG_SHORT_ACCOUNT_RATIO: Final[str] = "global_long_short_account_ratio"
_DATASET_TOP_LONG_SHORT_ACCOUNT_RATIO: Final[str] = "top_long_short_account_ratio"
_DATASET_TOP_LONG_SHORT_POSITION_RATIO: Final[str] = "top_long_short_position_ratio"
_DATASET_LIQUIDATION: Final[str] = "liquidation"
_STORAGE_DIR_EXPERIMENTS: Final[str] = "experiments"


@dataclass(frozen=True, slots=True)
class StorageLayout:
    """Immutable path composer for the CQROS data lake.

    Attributes:
        root: Storage root directory against which all artifact paths are
            resolved. Accepts ``str`` or ``pathlib.Path``. May be absolute
            or relative; no existence check is performed.
    """

    root: Path

    def __init__(self, root: FilePath) -> None:
        """Initialize a layout for the given storage root.

        Args:
            root: Storage root as a string or ``pathlib.Path``.
        """
        object.__setattr__(self, "root", Path(root))

    def raw_ohlcv_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a raw OHLCV year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval (for example ``1m``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the raw OHLCV hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_RAW,
            _DATASET_OHLCV,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def raw_funding_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a raw funding-rate year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the raw funding hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_RAW,
            _DATASET_FUNDING,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def raw_open_interest_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a raw open-interest year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the raw open-interest hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_RAW,
            _DATASET_OPEN_INTEREST,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def raw_taker_volume_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a raw taker-volume year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the raw taker-volume hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_RAW,
            _DATASET_TAKER_VOLUME,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def raw_global_long_short_account_ratio_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a raw global long/short account-ratio partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the raw global long/short
            account-ratio hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_RAW,
            _DATASET_GLOBAL_LONG_SHORT_ACCOUNT_RATIO,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def raw_top_long_short_account_ratio_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a raw top-trader long/short account-ratio partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the raw top-trader long/short
            account-ratio hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_RAW,
            _DATASET_TOP_LONG_SHORT_ACCOUNT_RATIO,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def raw_top_long_short_position_ratio_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a raw top-trader long/short position-ratio partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the raw top-trader long/short
            position-ratio hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_RAW,
            _DATASET_TOP_LONG_SHORT_POSITION_RATIO,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def raw_liquidation_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a raw liquidation year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the raw liquidation hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_RAW,
            _DATASET_LIQUIDATION,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def processed_ohlcv_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a processed OHLCV year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Bar interval (for example ``1m``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the processed OHLCV hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PROCESSED,
            _DATASET_OHLCV,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def processed_funding_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a processed funding-rate year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the processed funding hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PROCESSED,
            _DATASET_FUNDING,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def processed_open_interest_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a processed open-interest year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the processed open-interest
            hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PROCESSED,
            _DATASET_OPEN_INTEREST,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def processed_taker_volume_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a processed taker-volume year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the processed taker-volume
            hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PROCESSED,
            _DATASET_TAKER_VOLUME,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def processed_global_long_short_account_ratio_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a processed global long/short account-ratio partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the processed global long/short
            account-ratio hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PROCESSED,
            _DATASET_GLOBAL_LONG_SHORT_ACCOUNT_RATIO,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def processed_top_long_short_account_ratio_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a processed top-trader long/short account-ratio partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the processed top-trader long/short
            account-ratio hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PROCESSED,
            _DATASET_TOP_LONG_SHORT_ACCOUNT_RATIO,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def processed_top_long_short_position_ratio_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a processed top-trader long/short position-ratio partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Sampling interval used for the stored series.
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the processed top-trader long/short
            position-ratio hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PROCESSED,
            _DATASET_TOP_LONG_SHORT_POSITION_RATIO,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def feature_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a feature dataset year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Feature bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the features hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_FEATURES,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def label_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a label dataset year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Label bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the labels hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_LABELS,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def training_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a training dataset year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Training bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the training hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_TRAINING,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def signal_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical signal dataset year partition.

        Layout::

            signals/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Signal bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the signals hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_SIGNALS,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def prediction_path(
        self,
        framework: str,
        model_name: str,
        version: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a prediction dataset year partition.

        Args:
            framework: Machine-learning framework identifier (for example
                ``lightgbm``).
            model_name: Stable model identifier (for example ``alpha-lgbm``).
            version: Model version identifier (for example ``1.0.0``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Prediction bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the predictions hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PREDICTIONS,
            framework,
            model_name,
            version,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def threshold_path(
        self,
        model_name: str,
        model_version: str,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> Path:
        """Return the path for a regression threshold partition.

        Layout::

            thresholds/{model_name}/{model_version}/{symbol}/{timeframe}/thresholds.parquet

        Args:
            model_name: Stable model identifier (for example ``alpha-lgbm``).
            model_version: Model version identifier (for example ``1.0.0``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Threshold bar interval (for example ``1h``).

        Returns:
            Path to ``thresholds.parquet`` under the thresholds hierarchy.
        """
        return self.root.joinpath(
            STORAGE_DIR_THRESHOLDS,
            model_name,
            model_version,
            symbol,
            timeframe,
            f"thresholds{FILE_EXTENSION_PARQUET}",
        )

    def portfolio_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical portfolio dataset year partition.

        Layout::

            portfolios/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Portfolio bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the portfolios hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PORTFOLIOS,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def risk_path(
        self,
        policy: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical risk-decision dataset year partition.

        Layout::

            risks/{policy}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            policy: Risk policy identifier (for example ``fixed_risk``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Risk-decision bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the risks hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_RISKS,
            policy,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def order_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical OMS order dataset year partition.

        Layout::

            orders/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Order bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the orders hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_ORDERS,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def execution_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical execution trade year partition.

        Layout::

            executions/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the executions hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_EXECUTIONS,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def position_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical position year partition.

        Layout::

            positions/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the positions hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_POSITIONS,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def accounting_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical accounting year partition.

        Layout::

            accounting/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the accounting hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_ACCOUNTING,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def portfolio_risk_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical portfolio-risk year partition.

        Layout::

            portfolio_risk/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the portfolio-risk hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PORTFOLIO_RISK,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def trade_management_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical trade-management year partition.

        Layout::

            trade_management/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the trade-management hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_TRADE_MANAGEMENT,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def pyramiding_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical pyramiding year partition.

        Layout::

            pyramiding/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the pyramiding hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PYRAMIDING,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def exit_engine_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical exit-engine year partition.

        Layout::

            exit_engine/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the exit-engine hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_EXIT_ENGINE,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def backtesting_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical backtesting year partition.

        Layout::

            backtesting/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the backtesting hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_BACKTESTING,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def performance_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical performance year partition.

        Layout::

            performance/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the performance hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PERFORMANCE,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def analytics_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical analytics year partition.

        Layout::

            analytics/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the analytics hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_ANALYTICS,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def reporting_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical reporting year partition.

        Layout::

            reporting/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the reporting hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_REPORTING,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def monitoring_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical monitoring year partition.

        Layout::

            monitoring/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the monitoring hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_MONITORING,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def factor_validation_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical factor validation year partition.

        Layout::

            factor_validation/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet

        Cross-sectional factor validation panels are keyed by manager,
        exchange, market, timeframe, and year. Symbol is not part of the
        partition path because validation metrics are computed across the
        full symbol universe.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the factor validation hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_FACTOR_VALIDATION,
            manager,
            exchange,
            market,
            timeframe,
            year=year,
        )

    def factor_selection_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical factor selection year partition.

        Layout::

            factor_selection/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet

        Cross-sectional factor selection panels are keyed by manager,
        exchange, market, timeframe, and year. Symbol is not part of the
        partition path because selection metrics are computed across the
        full symbol universe from factor validation panels.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the factor selection hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_FACTOR_SELECTION,
            manager,
            exchange,
            market,
            timeframe,
            year=year,
        )

    def factor_timeframe_analysis_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        year: int,
    ) -> Path:
        """Return the path for a canonical factor timeframe analysis year partition.

        Layout::

            factor_timeframe_analysis/{manager}/{exchange}/{market}/{year}.parquet

        Cross-sectional FTA panels are keyed by manager, exchange, market, and
        year. Symbol and source timeframe are not part of the partition path
        because FTA resolves best timeframe across Factor Selection panels.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the factor timeframe analysis hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_FACTOR_TIMEFRAME_ANALYSIS,
            manager,
            exchange,
            market,
            year=year,
        )

    def factor_combination_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical factor combination year partition.

        Layout::

            factor_combination/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet

        Cross-sectional combination panels are keyed by manager, exchange,
        market, timeframe, and year. Symbol is not part of the partition path.
        ``timeframe`` is the FTA-resolved combination timeframe
        (``best_timeframe`` when members agree).

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the factor combination hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_FACTOR_COMBINATION,
            manager,
            exchange,
            market,
            timeframe,
            year=year,
        )

    def factor_orthogonalization_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical factor orthogonalization year partition.

        Layout::

            factor_orthogonalization/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet

        Cross-sectional orthogonalization decision ledgers are keyed by manager,
        exchange, market, timeframe, and year. Symbol is not part of the
        partition path. ``timeframe`` matches the Factor Combination partition
        being orthogonalized.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the factor orthogonalization hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_FACTOR_ORTHOGONALIZATION,
            manager,
            exchange,
            market,
            timeframe,
            year=year,
        )

    def alpha_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical alpha year partition.

        Layout::

            alpha/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the alpha hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_ALPHA,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def regime_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical regime year partition.

        Layout::

            regime/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the regime hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_REGIME,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def models_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical models year partition.

        Layout::

            models/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the models hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_MODELS,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def walk_forward_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical walk-forward year partition.

        Layout::

            walk_forward/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet

        Cross-sectional walk-forward panels are keyed by manager, exchange,
        market, timeframe, and year. Symbol is not part of the partition path
        because evaluation metrics are computed from factor selection panels.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the walk-forward hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_WALK_FORWARD,
            manager,
            exchange,
            market,
            timeframe,
            year=year,
        )

    def walk_forward_evaluation_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a walk-forward evaluation-result year partition.

        Layout::

            walk_forward_evaluation/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet

        Evaluation results are a separate artifact from the walk-forward ledger.
        Partition identity mirrors ``walk_forward_path`` so discovery stays
        aligned without mutating ledger parquet files.

        Args:
            manager: Order manager identifier (for example ``default``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the walk-forward-evaluation hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_WALK_FORWARD_EVALUATION,
            manager,
            exchange,
            market,
            timeframe,
            year=year,
        )

    def purged_cv_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical purged-CV year partition.

        Layout::

            purged_cv/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet

        Cross-sectional purged-CV panels are keyed by manager, exchange,
        market, timeframe, and year. Symbol is not part of the partition path
        because evaluation metrics are computed from walk-forward panels.

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the purged-CV hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PURGED_CV,
            manager,
            exchange,
            market,
            timeframe,
            year=year,
        )

    def purged_cv_evaluation_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a purged-CV evaluation-result year partition.

        Layout::

            purged_cv_evaluation/{manager}/{exchange}/{market}/{timeframe}/{year}.parquet

        Evaluation results are a separate artifact from the purged-CV ledger.
        Partition identity mirrors ``purged_cv_path`` so discovery stays
        aligned without mutating purged-CV parquet files.

        Args:
            manager: Order manager identifier (for example ``default``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the purged-CV-evaluation hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_PURGED_CV_EVALUATION,
            manager,
            exchange,
            market,
            timeframe,
            year=year,
        )

    def factors_path(
        self,
        manager: str,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a canonical factors year partition.

        Layout::

            factors/{manager}/{exchange}/{market}/{symbol}/{timeframe}/{year}.parquet

        Args:
            manager: Order manager identifier (for example ``simple``).
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``usdt_perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Trade bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the factors hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_FACTORS,
            manager,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def model_path(
        self,
        framework: str,
        model_name: str,
        version: str,
    ) -> Path:
        """Return the directory path for a versioned model artifact.

        Args:
            framework: Machine-learning framework identifier (for example
                ``lightgbm``).
            model_name: Stable model identifier (for example ``alpha-lgbm``).
            version: Model version identifier (for example ``1.0.0``).

        Returns:
            Path to ``models/{framework}/{model_name}/{version}/``.
        """
        return self.root.joinpath(
            STORAGE_DIR_MODELS,
            framework,
            model_name,
            version,
        )

    def experiment_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for an experiment artifact year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Experiment bar interval (for example ``1h``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the experiments hierarchy.
        """
        return self._year_partition_path(
            _STORAGE_DIR_EXPERIMENTS,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def report_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a report artifact year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Report bar interval (for example ``1d``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the reports hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_REPORTS,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def cache_path(
        self,
        exchange: Exchange,
        market: Market,
        symbol: Symbol,
        timeframe: Timeframe,
        year: int,
    ) -> Path:
        """Return the path for a cache artifact year partition.

        Args:
            exchange: Exchange identifier (for example ``binance``).
            market: Market segment (for example ``perpetual``).
            symbol: Tradeable symbol (for example ``BTCUSDT``).
            timeframe: Cache entry bar interval (for example ``1m``).
            year: Calendar year of the partition.

        Returns:
            Path to ``{year}.parquet`` under the cache hierarchy.
        """
        return self._year_partition_path(
            STORAGE_DIR_CACHE,
            exchange,
            market,
            symbol,
            timeframe,
            year=year,
        )

    def _year_partition_path(
        self,
        *directories: str,
        year: int,
    ) -> Path:
        """Compose ``root / … / {year}.parquet`` from directory segments.

        Args:
            *directories: Ordered path segments under ``root``.
            year: Calendar year used as the parquet file stem.

        Returns:
            Composed path ending in ``{year}.parquet``.
        """
        return self.root.joinpath(*directories, f"{year}{FILE_EXTENSION_PARQUET}")

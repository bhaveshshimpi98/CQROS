"""CQROS Factor Selection factor-observation panel loader.

Purpose:
    Load validation-window factor observations from the canonical Factors
    lake for redundancy filtering without inventing a second storage system.

Responsibilities:
    - Resolve Factors year partitions through ``StorageLayout``
    - Scan symbol partitions once per timeframe
    - Filter by factor identity and validation window
    - Return long-format panels for ``FactorObservationSource``

Dependencies:
    ``polars``, ``pathlib``, ``cqros.core.constants``, and
    ``cqros.storage.layout``.

Public API:
    ``FactorsObservationLoader``
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import polars as pl

from cqros.core.constants import (
    EXCHANGE_BINANCE,
    FILE_EXTENSION_PARQUET,
    MARKET_USDT_PERPETUAL,
    STORAGE_DIR_FACTORS,
)
from cqros.core.types import Exchange, Market
from cqros.storage.layout import StorageLayout

__all__ = ["FactorsObservationLoader"]

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL


class FactorsObservationLoader:
    """Load Factors panels for one manager/exchange/market/year scope.

    Implements the ``FactorObservationSource`` contract used by
    ``SimpleFactorSelectionEngine`` redundancy filtering.
    """

    __slots__ = ("_exchange", "_layout", "_manager", "_market", "_year")

    def __init__(
        self,
        layout: StorageLayout,
        *,
        manager: str,
        year: int,
        exchange: Exchange = _EXCHANGE,
        market: Market = _MARKET,
    ) -> None:
        """Initialize the loader for one Factors year scope.

        Args:
            layout: Canonical storage layout.
            manager: Order manager identifier.
            year: Calendar year partition.
            exchange: Exchange identifier.
            market: Market segment.
        """
        self._layout = layout
        self._manager = manager
        self._year = year
        self._exchange = exchange
        self._market = market

    def load_panel(
        self,
        *,
        timeframe: str,
        factor_names: Sequence[str],
        factor_versions: Sequence[str],
        start_time: int,
        end_time: int,
    ) -> pl.DataFrame:
        """Load long-format observations for requested factors in-window."""
        paths = self._partition_paths(timeframe)
        if len(paths) == 0 or len(factor_names) == 0:
            return pl.DataFrame(
                schema={
                    "symbol": pl.String,
                    "open_time": pl.Int64,
                    "factor_name": pl.String,
                    "factor_version": pl.String,
                    "factor_value": pl.Float64,
                }
            )

        name_set = set(factor_names)
        version_set = set(factor_versions)
        return (
            pl.scan_parquet([str(path) for path in paths])
            .filter(pl.col("factor_name").is_in(list(name_set)))
            .filter(pl.col("factor_version").is_in(list(version_set)))
            .filter(pl.col("open_time") >= start_time)
            .filter(pl.col("open_time") <= end_time)
            .select(
                [
                    "symbol",
                    "open_time",
                    "factor_name",
                    "factor_version",
                    "factor_value",
                ]
            )
            .collect()
        )

    def _partition_paths(self, timeframe: str) -> list[Path]:
        """Return existing Factors parquet paths for ``timeframe``/year."""
        base = (
            self._layout.root / STORAGE_DIR_FACTORS / self._manager / self._exchange / self._market
        )
        if not base.is_dir():
            return []
        paths: list[Path] = []
        for symbol_dir in sorted(base.iterdir()):
            if not symbol_dir.is_dir():
                continue
            path = symbol_dir / timeframe / f"{self._year}{FILE_EXTENSION_PARQUET}"
            if path.is_file():
                paths.append(path)
        return paths

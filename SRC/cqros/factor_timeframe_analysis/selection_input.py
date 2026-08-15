"""CQROS Factor Timeframe Analysis — Factor Selection input adapter.

Purpose:
    Load and concatenate Factor Selection partitions for a given manager,
    exchange, market, and year to provide FTA with its cross-timeframe input
    dataset.

Responsibilities:
    - Discover available timeframe partitions for a year via
      ``FactorSelectionRepository``
    - Load each partition and concatenate into a single FTA-ready frame
    - Sort the result deterministically by factor_name, factor_version,
      timeframe
    - Raise ``FactorTimeframeAnalysisError`` when no partitions are found or
      the concatenated result is empty

Dependencies:
    ``polars``, ``cqros.core``,
    ``cqros.factor_selection.repository``,
    ``cqros.factor_timeframe_analysis.exceptions``.

Public API:
    ``load_factor_selection_for_analysis``, ``discover_selection_timeframes``

Notes:
    This module is FTA-owned. It must not alter Factor Selection semantics or
    defaults. All loading is delegated to the injected
    ``FactorSelectionRepository`` instance.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

import polars as pl

from cqros.core.constants import EXCHANGE_BINANCE, MARKET_USDT_PERPETUAL
from cqros.core.types import Exchange, Market, Timeframe
from cqros.factor_selection.repository import FactorSelectionRepository
from cqros.factor_timeframe_analysis.exceptions import FactorTimeframeAnalysisError

__all__ = [
    "discover_selection_timeframes",
    "load_factor_selection_for_analysis",
]

_logger = logging.getLogger(__name__)

_EXCHANGE: Final[Exchange] = EXCHANGE_BINANCE
_MARKET: Final[Market] = MARKET_USDT_PERPETUAL

_ERROR_SELECTION_MISSING: Final[str] = "FTA_SELECTION_MISSING"
_ERROR_SELECTION_EMPTY: Final[str] = "FTA_SELECTION_EMPTY"

_SORT_COLUMNS: Final[tuple[str, ...]] = ("factor_name", "factor_version", "timeframe")


def discover_selection_timeframes(
    repository: FactorSelectionRepository,
    *,
    manager: str,
    exchange: Exchange = _EXCHANGE,
    market: Market = _MARKET,
    year: int,
) -> tuple[Timeframe, ...]:
    """Return sorted timeframes for which a Factor Selection partition exists for ``year``.

    Iterates all timeframes present under manager/exchange/market and returns
    those with a partition file for the requested year.

    Args:
        repository: Factor Selection repository used for discovery.
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.
        year: Calendar year to search.

    Returns:
        Sorted timeframe identifiers for which ``year`` partitions exist.
        Returns an empty tuple when no matching partitions are found.
    """
    available: list[Timeframe] = []
    for timeframe in repository.discover_timeframes(
        manager=manager,
        exchange=exchange,
        market=market,
    ):
        years = repository.list_years(
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
        )
        if year in years:
            available.append(timeframe)
    return tuple(sorted(available))


def load_factor_selection_for_analysis(
    repository: FactorSelectionRepository,
    *,
    manager: str,
    exchange: Exchange = _EXCHANGE,
    market: Market = _MARKET,
    year: int,
    timeframes: Sequence[Timeframe] | None = None,
) -> pl.DataFrame:
    """Load and concatenate Factor Selection partitions for FTA input.

    All available timeframe partitions for the given manager/exchange/market/year
    are loaded through the injected ``FactorSelectionRepository`` and
    concatenated. The result is sorted deterministically by factor_name,
    factor_version, timeframe.

    Args:
        repository: Factor Selection repository providing per-timeframe
            partitions.
        manager: Order manager identifier.
        exchange: Exchange identifier.
        market: Market segment.
        year: Calendar year for which to load Factor Selection data.
        timeframes: Optional allowlist of timeframes. ``None`` uses every
            available timeframe for the year.

    Returns:
        Concatenated Factor Selection frame, sorted by factor_name,
        factor_version, timeframe. Schema is the native
        ``FACTOR_SELECTION_SCHEMA`` as returned by the repository.

    Raises:
        FactorTimeframeAnalysisError: If no partitions are found for the year
            (error code ``FTA_SELECTION_MISSING``) or the concatenated result
            is empty (error code ``FTA_SELECTION_EMPTY``).
    """
    available = discover_selection_timeframes(
        repository,
        manager=manager,
        exchange=exchange,
        market=market,
        year=year,
    )
    if timeframes is not None:
        filter_set = frozenset(timeframes)
        available = tuple(tf for tf in available if tf in filter_set)

    if not available:
        raise FactorTimeframeAnalysisError(
            "no Factor Selection partitions found for the requested year",
            error_code=_ERROR_SELECTION_MISSING,
            details={
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "year": year,
                "timeframes_filter": list(timeframes) if timeframes is not None else None,
            },
        )

    frames: list[pl.DataFrame] = []
    for timeframe in available:
        _logger.debug(
            "Loading Factor Selection partition for FTA",
            extra={
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "year": year,
                "timeframe": timeframe,
            },
        )
        frame = repository.load(
            manager=manager,
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            year=year,
        )
        frames.append(frame)

    concatenated = pl.concat(frames, how="vertical_relaxed")
    if concatenated.height == 0:
        raise FactorTimeframeAnalysisError(
            "concatenated Factor Selection frame is empty",
            error_code=_ERROR_SELECTION_EMPTY,
            details={
                "manager": manager,
                "exchange": exchange,
                "market": market,
                "year": year,
                "timeframes": list(available),
                "partition_count": len(frames),
            },
        )

    result = concatenated.sort(list(_SORT_COLUMNS))
    _logger.info(
        "Loaded Factor Selection data for FTA",
        extra={
            "manager": manager,
            "exchange": exchange,
            "market": market,
            "year": year,
            "timeframes": list(available),
            "rows": result.height,
        },
    )
    return result

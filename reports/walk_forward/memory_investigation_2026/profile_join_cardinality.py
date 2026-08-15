"""Bounded, read-only profiling of Walk Forward join cardinality.

Each symbol is loaded and joined independently, then released. This reproduces
the canonical inner-join predicates without constructing a timeframe panel,
running fold generation, or writing under ``data/``.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
EVIDENCE = Path(__file__).resolve().parent
TIMEFRAMES = ("1d", "4h", "1h", "15m", "5m")
YEAR = 2026

FACTOR_READ_COLUMNS = (
    "symbol",
    "timeframe",
    "open_time",
    "factor_name",
    "factor_version",
)
LABEL_READ_COLUMNS = ("symbol", "timeframe", "open_time", "future_return_1")
SELECTION_COLUMNS = (
    "factor_name",
    "factor_version",
    "timeframe",
    "selected",
)


def _factor_paths(timeframe: str) -> list[Path]:
    root = DATA / "factors" / "default" / "binance" / "usdt_perpetual"
    return sorted(root.glob(f"*/{timeframe}/{YEAR}.parquet"))


def _label_path(symbol: str, timeframe: str) -> Path:
    return (
        DATA
        / "labels"
        / "binance"
        / "usdt_perpetual"
        / symbol
        / timeframe
        / f"{YEAR}.parquet"
    )


def _selection(timeframe: str) -> pl.DataFrame:
    path = (
        DATA
        / "factor_selection"
        / "default"
        / "binance"
        / "usdt_perpetual"
        / timeframe
        / f"{YEAR}.parquet"
    )
    return pl.read_parquet(path, columns=list(SELECTION_COLUMNS))


def _profile_symbol(
    factor_path: Path,
    selection: pl.DataFrame,
    timeframe: str,
) -> dict[str, object]:
    symbol = factor_path.parents[1].name
    label_path = _label_path(symbol, timeframe)
    factors = pl.read_parquet(factor_path, columns=list(FACTOR_READ_COLUMNS))
    labels = pl.read_parquet(label_path, columns=list(LABEL_READ_COLUMNS))
    observations = factors.join(
        labels,
        on=["symbol", "timeframe", "open_time"],
        how="inner",
    )
    enriched = observations.join(
        selection,
        on=["factor_name", "factor_version", "timeframe"],
        how="inner",
    ).with_columns(pl.col("open_time").alias("selection_time"))
    engine_frame = enriched.select(
        ["timeframe", "selection_time", "selected", "future_return_1"]
    )
    return {
        "symbol": symbol,
        "factor_rows": factors.height,
        "label_rows": labels.height,
        "factor_label_rows": observations.height,
        "selection_join_rows": enriched.height,
        "selected_rows": int(enriched["selected"].sum()),
        "engine_columns_estimated_bytes": engine_frame.estimated_size(),
    }


def _profile_timeframe(timeframe: str) -> dict[str, object]:
    started = time.perf_counter()
    selection = _selection(timeframe)
    symbols = [
        _profile_symbol(path, selection, timeframe)
        for path in _factor_paths(timeframe)
        if _label_path(path.parents[1].name, timeframe).is_file()
    ]
    return {
        "timeframe": timeframe,
        "year": YEAR,
        "selection_rows": selection.height,
        "selection_selected_rows": int(selection["selected"].sum()),
        "symbols": len(symbols),
        "factor_rows": sum(int(item["factor_rows"]) for item in symbols),
        "label_rows": sum(int(item["label_rows"]) for item in symbols),
        "factor_label_rows": sum(int(item["factor_label_rows"]) for item in symbols),
        "selection_join_rows": sum(int(item["selection_join_rows"]) for item in symbols),
        "selected_rows": sum(int(item["selected_rows"]) for item in symbols),
        "engine_columns_estimated_bytes": sum(
            int(item["engine_columns_estimated_bytes"]) for item in symbols
        ),
        "duration_seconds": time.perf_counter() - started,
        "per_symbol": symbols,
    }


def main() -> int:
    payload = {
        "captured_at_utc": datetime.now(tz=UTC).isoformat(),
        "profiles": {timeframe: _profile_timeframe(timeframe) for timeframe in TIMEFRAMES},
    }
    output = EVIDENCE / "join_cardinality.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

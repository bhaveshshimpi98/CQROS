"""Inspect BTCUSDT 1d factor coverage and incomplete 2026 partitions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FACTORS = ROOT / "data/factors/default/binance/usdt_perpetual"
TFS = ("5m", "15m", "1h", "4h", "1d")
NAMES = (
    "price_volume_trend",
    "on_balance_volume",
    "open_interest_level",
    "rsi",
    "money_flow_index",
    "stochastic_k",
    "ease_of_movement",
    "rate_of_change",
)


def main() -> None:
    path = FACTORS / "BTCUSDT/1d/2026.parquet"
    df = pl.read_parquet(path)
    print("BTCUSDT 1d coverage:")
    for name in NAMES:
        sub = df.filter(pl.col("factor_name") == name)
        non_null = sub.filter(pl.col("factor_value").is_not_null())
        print(f"  {name}: non_null={non_null.height}/{sub.height}")
    print(
        "unique_ts",
        df["open_time"].n_unique(),
        "n_factors",
        df["factor_name"].n_unique(),
    )

    symbols = sorted(path.name for path in FACTORS.iterdir() if path.is_dir())
    present = {tf: 0 for tf in TFS}
    missing = {tf: 0 for tf in TFS}
    for symbol in symbols:
        for tf in TFS:
            if (FACTORS / symbol / tf / "2026.parquet").exists():
                present[tf] += 1
            else:
                missing[tf] += 1
    print("symbols", len(symbols))
    print("present", present)
    print("missing", missing)

    cutoff = datetime(2026, 8, 12, 7, 50).timestamp()
    new = 0
    old = 0
    for file_path in FACTORS.rglob("2026.parquet"):
        if file_path.stat().st_mtime >= cutoff:
            new += 1
        else:
            old += 1
    print("new_since_0750", new, "old", old)


if __name__ == "__main__":
    main()

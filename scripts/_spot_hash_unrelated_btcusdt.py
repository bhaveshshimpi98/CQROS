"""Lightweight unrelated BTCUSDT spot hashes for immutability checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "reports"
    / "factor_stability"
    / "input_partition_downstream_regen"
    / "hashes_before_unrelated_spot.txt"
)

TIERS = (
    "processed/ohlcv",
    "processed/funding",
    "processed/open_interest",
    "processed/taker_volume",
    "features",
    "labels",
)
TFS = ("5m", "15m", "1h", "4h", "1d")


def main() -> None:
    paths: list[Path] = []
    for tf in TFS:
        for tier in TIERS:
            candidate = (
                ROOT
                / "data"
                / tier
                / "binance"
                / "usdt_perpetual"
                / "BTCUSDT"
                / tf
                / "2026.parquet"
            )
            if candidate.exists():
                paths.append(candidate)
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}"
        for path in sorted(set(paths))
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"spot hashes: {len(lines)}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()

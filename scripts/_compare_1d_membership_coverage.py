"""Compare before/after 1d selection membership and BTCUSDT coverage."""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "factor_stability" / "input_partition_downstream_regen"
BEFORE = OUT / "before"
AFTER = OUT / "after"


def compare_selection() -> None:
    before = pl.read_csv(BEFORE / "1d_selected_factors_before.csv")
    after_path = (
        ROOT / "data/factor_selection/default/binance/usdt_perpetual/1d/2026.parquet"
    )
    after = pl.read_parquet(after_path)
    b = set(before.filter(pl.col("selected") == True)["factor_name"].to_list())  # noqa: E712
    a_sel = after.filter(pl.col("selected") == True)  # noqa: E712
    a = set(a_sel["factor_name"].to_list())
    entered = sorted(a - b)
    removed = sorted(b - a)
    retained = sorted(a & b)
    lines = [
        f"before_selected={sorted(b)}",
        f"after_selected={sorted(a)}",
        f"entered={entered}",
        f"removed={removed}",
        f"retained={retained}",
        "",
        "AFTER DETAILS:",
    ]
    detail_cols = [
        c
        for c in [
            "factor_name",
            "selected",
            "selected_direction",
            "eligibility_status",
            "usable_observations",
            "selection_ic",
            "selection_rank",
            "orientation_policy",
        ]
        if c in after.columns
    ]
    lines.append(a_sel.select(detail_cols).sort("selection_rank").write_csv(None))
    AFTER.mkdir(parents=True, exist_ok=True)
    (OUT / "1d_membership_comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def compare_coverage() -> None:
    b = pl.read_csv(BEFORE / "btc_1d_factor_coverage.csv")
    a = pl.read_csv(AFTER / "btc_1d_factor_coverage.csv")
    joined = b.join(a, on="factor_name", suffix="_after")
    joined.write_csv(OUT / "btc_1d_coverage_comparison.csv")
    print(joined.write_csv(None))


if __name__ == "__main__":
    compare_selection()
    if (AFTER / "btc_1d_factor_coverage.csv").exists():
        compare_coverage()

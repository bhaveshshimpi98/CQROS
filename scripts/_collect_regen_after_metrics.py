"""Collect after-state metrics for input-partition downstream regeneration."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/factor_stability/input_partition_downstream_regen"
AFTER = OUT / "after"
BEFORE = OUT / "before"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    AFTER.mkdir(parents=True, exist_ok=True)

    # BTC coverage after
    df = pl.read_parquet(
        ROOT / "data/factors/default/binance/usdt_perpetual/BTCUSDT/1d/2026.parquet"
    )
    names = [
        "price_volume_trend",
        "on_balance_volume",
        "open_interest_level",
        "rsi",
        "money_flow_index",
        "stochastic_k",
        "ease_of_movement",
        "rate_of_change",
    ]
    rows = []
    for name in names:
        sub = df.filter(pl.col("factor_name") == name)
        nn = sub.filter(pl.col("factor_value").is_not_null())
        rows.append(
            {
                "factor_name": name,
                "non_null": nn.height,
                "rows": sub.height,
                "null_rate": 1.0 - (nn.height / sub.height if sub.height else 0.0),
                "unique_ts": nn["open_time"].n_unique() if nn.height else 0,
                "first": nn["open_time"].min() if nn.height else None,
                "last": nn["open_time"].max() if nn.height else None,
            }
        )
    cov = pl.DataFrame(rows)
    cov.write_csv(AFTER / "btc_1d_factor_coverage.csv")
    print(cov)

    sel = pl.read_parquet(
        ROOT / "data/factor_selection/default/binance/usdt_perpetual/1d/2026.parquet"
    )
    # PVT/OBV eligibility even if not selected
    focus = sel.filter(
        pl.col("factor_name").is_in(
            [
                "price_volume_trend",
                "on_balance_volume",
                "open_interest_level",
                "rsi",
                "money_flow_index",
                "stochastic_k",
                "ease_of_movement",
                "rate_of_change",
            ]
        )
    ).select(
        [
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
            if c in sel.columns
        ]
    )
    focus.write_csv(AFTER / "1d_focus_factor_selection_status.csv")
    print(focus)

    folds = pl.read_csv(OUT / "pcv_eval/purged_cv_evaluation_folds.csv")
    print("FOLD ICS")
    print(
        folds.select(
            [
                "fold_id",
                "raw_oos_ic",
                "oriented_oos_ic",
                "oos_return_mean",
                "oos_positive_rate",
                "purge_valid",
                "embargo_valid",
                "train_test_disjoint",
                "fold_order_valid",
                "timestamp_valid",
            ]
        )
    )

    fac = pl.read_csv(OUT / "pcv_eval/purged_cv_evaluation_factors.csv")
    fac_mean = (
        fac.group_by("factor_name")
        .agg(
            pl.col("oriented_oos_ic").mean().alias("mean_oriented_oos_ic"),
            pl.col("raw_oos_ic").mean().alias("mean_raw_oos_ic"),
            pl.col("selected_direction").first(),
            pl.col("selection_ic").first(),
        )
        .sort("mean_oriented_oos_ic", descending=True)
    )
    fac_mean.write_csv(AFTER / "1d_pcv_factor_mean_ic.csv")
    print(fac_mean)

    # unique OOS timestamps from WF eval if available
    wf_eval_path = OUT / "wf_eval/walk_forward_evaluation_global.csv"
    if wf_eval_path.exists():
        print("WF GLOBAL")
        print(pl.read_csv(wf_eval_path))

    # hashes of critical 1d artifacts
    crit = [
        ROOT / "data/factors/default/binance/usdt_perpetual/BTCUSDT/1d/2026.parquet",
        ROOT / "data/factor_validation/default/binance/usdt_perpetual/1d/2026.parquet",
        ROOT / "data/factor_selection/default/binance/usdt_perpetual/1d/2026.parquet",
        ROOT / "data/walk_forward/default/binance/usdt_perpetual/1d/2026.parquet",
        ROOT / "data/purged_cv/default/binance/usdt_perpetual/1d/2026.parquet",
        ROOT
        / "data/purged_cv_evaluation/default/binance/usdt_perpetual/1d/2026.parquet",
    ]
    lines = [f"{sha(p)}  {p.relative_to(ROOT).as_posix()}" for p in crit if p.exists()]
    (OUT / "hashes_after_1d_critical.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("critical hashes written", len(lines))


if __name__ == "__main__":
    main()

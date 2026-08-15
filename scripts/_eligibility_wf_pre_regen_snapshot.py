"""Snapshot pre-regeneration Walk-Forward baselines for eligibility regeneration."""

from __future__ import annotations

from pathlib import Path

import polars as pl

REPORTS = Path("reports/walk_forward")
BEFORE = REPORTS / "eligibility_regeneration_hashes_before.txt"
TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")


def main() -> None:
    """Append selected-count baselines and dump old 1d membership."""
    old_eval = pl.read_csv(REPORTS / "walk_forward_evaluation_all.csv")
    orient = pl.read_csv(REPORTS / "factor_orientation_summary.csv")

    lines: list[str] = [
        "=== SELECTED FACTOR COUNTS BEFORE REGENERATION ===",
        "FS counts = current Factor Selection artifacts (post eligibility).",
        "WF_eval_old_selected = prior Walk-Forward evaluation unique_selected_factors.",
        "",
    ]
    for timeframe in TIMEFRAMES:
        fs = pl.read_parquet(
            f"data/factor_selection/default/binance/usdt_perpetual/{timeframe}/2026.parquet"
        )
        new_sel = int(fs.filter(pl.col("selected")).height)
        old_row = old_eval.filter(pl.col("timeframe") == timeframe)
        old_n = int(old_row["unique_selected_factors"][0]) if old_row.height else -1
        wf = pl.read_parquet(
            f"data/walk_forward/default/binance/usdt_perpetual/{timeframe}/2026.parquet"
        )
        folds = int(wf["fold_id"].n_unique())
        lines.append(
            f"{timeframe}: FS_new_selected={new_sel} "
            f"WF_eval_old_selected={old_n} WF_rows={wf.height} WF_folds={folds}"
        )

    ev = pl.read_parquet(
        "data/walk_forward_evaluation/default/binance/usdt_perpetual/1d/2026.parquet",
        columns=[
            "factor_name",
            "selected",
            "selected_direction",
            "factor_value",
            "partition",
        ],
    )
    old_sel = (
        ev.filter(pl.col("selected"))
        .select(["factor_name", "selected_direction"])
        .unique()
        .sort("factor_name")
    )
    old_sel.write_csv(REPORTS / "eligibility_regen_old_1d_selected_factors.csv")
    lines.append("")
    lines.append(f"OLD_1d_selected_factor_count={old_sel.height}")
    for row in old_sel.iter_rows(named=True):
        lines.append(
            f"  {row['factor_name']} selected_direction={row['selected_direction']}"
        )

    partition_values = ev["partition"].unique().to_list()
    lines.append("")
    lines.append(f"1d_eval_partition_values={partition_values}")
    for partition in sorted(str(value) for value in partition_values):
        sub = ev.filter((pl.col("selected")) & (pl.col("partition") == partition))
        null_by_factor = (
            sub.group_by("factor_name")
            .agg(
                [
                    pl.len().alias("n"),
                    pl.col("factor_value").null_count().alias("nulls"),
                    (pl.col("factor_value").null_count() / pl.len()).alias("null_rate"),
                ]
            )
            .sort("null_rate", descending=True)
        )
        lines.append(f"--- partition={partition} selected null rates ---")
        for row in null_by_factor.iter_rows(named=True):
            lines.append(
                f"  {row['factor_name']}: n={row['n']} nulls={row['nulls']} "
                f"null_rate={row['null_rate']:.6f}"
            )

    extra = "\n".join(lines)
    extra += "\n\n=== OLD ORIENTATION / OOS IC ===\n"
    extra += orient.write_csv(file=None)
    extra += "\n=== OLD EVAL SUMMARY ===\n"
    extra += old_eval.write_csv(file=None)
    BEFORE.write_text(BEFORE.read_text(encoding="utf-8") + "\n" + extra, encoding="utf-8")
    print(extra)


if __name__ == "__main__":
    main()

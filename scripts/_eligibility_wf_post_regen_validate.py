"""Complete post-regeneration validation, orientation rebuild, hashes, diagnostics."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import polars as pl

from cqros.reporting.factor_orientation_diagnostic import (
    FactorOrientationReporter,
    build_factor_orientation_details,
    build_orientation_summary,
)

REPORTS = Path("reports/walk_forward")
TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
FS_ROOT = Path("data/factor_selection/default/binance/usdt_perpetual")
WF_ROOT = Path("data/walk_forward/default/binance/usdt_perpetual")
PCV_ROOT = Path("data/purged_cv/default/binance/usdt_perpetual")
EVAL_ROOT = Path("data/walk_forward_evaluation/default/binance/usdt_perpetual")

# Known 100%-NULL OOS factors from pre-regen snapshot
NULL_100_PCT = {
    "atr_slope",
    "bollinger_width",
    "atr_distance",
    "aggressive_sell_ratio",
    "aggressive_buy_ratio",
    "atr_percent",
    "bollinger_bandwidth",
    "breakout_strength",
    "bollinger_position",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_fs(timeframe: str) -> pl.DataFrame:
    return pl.read_parquet(FS_ROOT / timeframe / "2026.parquet").filter(pl.col("selected"))


def rebuild_orientation() -> pl.DataFrame:
    factors = pl.read_csv(REPORTS / "walk_forward_evaluation_factors.csv")
    # Keep selected-factor metric rows only; factors CSV is already selected-oriented.
    summary = build_orientation_summary(factors)
    details = build_factor_orientation_details(factors)
    FactorOrientationReporter(REPORTS).write_reports(summary=summary, factor_details=details)
    return summary


def write_after_hashes() -> None:
    lines: list[str] = []
    for label, root in (
        ("factor_selection", Path("data/factor_selection")),
        ("walk_forward", Path("data/walk_forward")),
        ("purged_cv", Path("data/purged_cv")),
    ):
        lines.append(f"=== {label} ===")
        for path in sorted(root.rglob("*.parquet")):
            # skip analysis helpers if any
            if "analysis" in path.parts:
                continue
            digest = sha256(path)
            lines.append(f"{digest}  {path.as_posix()}  bytes={path.stat().st_size}")
            if label == "factor_selection":
                frame = pl.read_parquet(path)
                n_sel = int(frame.filter(pl.col("selected")).height)
                lines.append(f"  selected_count={n_sel}")
            if label == "walk_forward":
                frame = pl.read_parquet(path)
                lines.append(f"  rows={frame.height}")
        lines.append("")
    (REPORTS / "eligibility_regeneration_hashes_after.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def compare_hashes() -> str:
    before = REPORTS / "eligibility_regeneration_hashes_before.txt"
    after = REPORTS / "eligibility_regeneration_hashes_after.txt"

    def parse(path: Path) -> dict[str, str]:
        mapping: dict[str, str] = {}
        current = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("=== "):
                current = line.strip("= ").strip()
                continue
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) == 64:
                mapping[f"{current}|{parts[1]}"] = parts[0]
        return mapping

    b = parse(before)
    a = parse(after)
    changed: list[str] = []
    unchanged: list[str] = []
    for key in sorted(set(b) | set(a)):
        if key not in b:
            changed.append(f"ADDED {key}")
        elif key not in a:
            changed.append(f"REMOVED {key}")
        elif b[key] != a[key]:
            changed.append(f"CHANGED {key}\n  before={b[key]}\n  after ={a[key]}")
        else:
            unchanged.append(key)
    text = ["=== IMMUTABILITY DIFF ===", "CHANGED:"]
    text.extend(changed if changed else ["  (none)"])
    text.append("")
    text.append("UNCHANGED:")
    text.extend(f"  {item}" for item in unchanged)
    out = "\n".join(text) + "\n"
    (REPORTS / "eligibility_regen_immutability_diff.txt").write_text(out, encoding="utf-8")
    return out


def membership_and_eligibility() -> str:
    old_1d = pl.read_csv(REPORTS / "eligibility_regen_old_1d_selected_factors.csv")
    new_1d = selected_fs("1d").select(
        [
            "factor_name",
            "selected_direction",
            "eligibility_status",
            "eligibility_policy",
            "usable_observations",
            "coverage_ratio",
            "null_rate",
        ]
    )
    old_names = set(old_1d["factor_name"].to_list())
    new_names = set(new_1d["factor_name"].to_list())
    removed = sorted(old_names - new_names)
    retained = sorted(old_names & new_names)
    newly = sorted(new_names - old_names)

    rows: list[dict[str, object]] = []
    for timeframe in TIMEFRAMES:
        fs = selected_fs(timeframe)
        for row in fs.iter_rows(named=True):
            rows.append(
                {
                    "timeframe": timeframe,
                    "factor": row["factor_name"],
                    "old_selected": (row["factor_name"] in old_names)
                    if timeframe == "1d"
                    else True,
                    "new_selected": True,
                    "eligibility_status": row["eligibility_status"],
                    "selected_direction": int(row["selected_direction"]),
                }
            )
        if timeframe == "1d":
            for name in removed:
                old_dir = int(
                    old_1d.filter(pl.col("factor_name") == name)["selected_direction"][0]
                )
                rows.append(
                    {
                        "timeframe": "1d",
                        "factor": name,
                        "old_selected": True,
                        "new_selected": False,
                        "eligibility_status": None,
                        "selected_direction": old_dir,
                    }
                )
    pl.DataFrame(rows).sort(["timeframe", "factor"]).write_csv(
        REPORTS / "eligibility_regen_factor_membership_comparison.csv"
    )

    lines: list[str] = []
    lines.append("=== ELIGIBILITY / MEMBERSHIP ===")
    for timeframe in TIMEFRAMES:
        fs = selected_fs(timeframe)
        assert fs.filter(pl.col("eligibility_status") != "ELIGIBLE").height == 0
        assert fs.filter(pl.col("eligibility_policy") != "coverage_v1").height == 0
        assert fs.filter(~pl.col("selected_direction").is_in([-1, 1])).height == 0
        lines.append(
            f"{timeframe}: selected={fs.height} all ELIGIBLE/coverage_v1/dir±1=TRUE"
        )

    fs_all = pl.read_parquet(FS_ROOT / "1d" / "2026.parquet")
    zero = fs_all.filter(pl.col("eligibility_status") == "INELIGIBLE_ZERO_OBSERVATIONS")
    lines.append(
        f"1d zero-obs ineligible={zero.height} selected_among_them="
        f"{zero.filter(pl.col('selected')).height}"
    )
    lines.append(f"1d removed ({len(removed)}): {removed}")
    lines.append(f"1d retained ({len(retained)}): {retained}")
    lines.append(f"1d newly ({len(newly)}): {newly}")
    lines.append(
        f"100pct_null_factors_still_selected={sorted(NULL_100_PCT & new_names)}"
    )

    # Orientation inheritance FS -> evaluation
    lines.append("")
    lines.append("=== ORIENTATION FS → EVAL ===")
    for timeframe in TIMEFRAMES:
        fs = selected_fs(timeframe).select(["factor_name", "selected_direction"])
        ev = pl.read_parquet(
            EVAL_ROOT / timeframe / "2026.parquet",
            columns=["factor_name", "selected", "selected_direction"],
        )
        ev_sel = (
            ev.filter(pl.col("selected"))
            .select(["factor_name", "selected_direction"])
            .unique()
            .sort("factor_name")
        )
        joined = fs.join(ev_sel, on="factor_name", how="full", suffix="_eval")
        mismatch = joined.filter(
            (pl.col("selected_direction") != pl.col("selected_direction_eval"))
            | pl.col("selected_direction_eval").is_null()
            | pl.col("selected_direction").is_null()
        )
        lines.append(
            f"{timeframe}: fs={fs.height} eval={ev_sel.height} mismatches={mismatch.height}"
        )
        if mismatch.height:
            lines.append(str(mismatch))

    # Observability of remaining 1d factors in NEW eval
    ev1d = pl.read_parquet(
        EVAL_ROOT / "1d" / "2026.parquet",
        columns=["factor_name", "selected", "factor_value", "observation_time", "partition"],
    )
    oos = ev1d.filter(pl.col("selected"))
    null_rates = (
        oos.group_by("factor_name")
        .agg(
            [
                pl.len().alias("n"),
                pl.col("factor_value").null_count().alias("nulls"),
                (pl.col("factor_value").null_count() / pl.len()).alias("null_rate"),
                pl.col("observation_time").n_unique().alias("unique_times"),
            ]
        )
        .sort("null_rate", descending=True)
    )
    lines.append("")
    lines.append("=== 1d NEW SELECTED OOS OBSERVABILITY ===")
    lines.append(null_rates.write_csv(file=None))
    lines.append(
        f"unique_oos_observation_times={int(oos['observation_time'].n_unique())}"
    )
    lines.append(f"total_selected_oos_rows={oos.height}")

    text = "\n".join(lines) + "\n"
    (REPORTS / "eligibility_regen_validation_notes.txt").write_text(text, encoding="utf-8")
    return text


def oos_comparison(orient: pl.DataFrame) -> str:
    old_path = REPORTS / "eligibility_regen_old_orientation_summary.csv"
    if not old_path.exists():
        # Recreate from pre-regeneration orientation snapshot.
        old = pl.DataFrame(
            {
                "timeframe": ["15m", "1d", "1h", "4h", "5m"],
                "year": [2026] * 5,
                "selected_factor_count": [20, 20, 20, 20, 20],
                "mean_raw_oos_ic": [
                    -0.015067647132757522,
                    -0.03655311804461355,
                    -0.015936999668253425,
                    -0.02102082081113424,
                    -0.01879780093815259,
                ],
                "mean_oriented_oos_ic": [
                    0.026073300018509403,
                    -0.024028022306263642,
                    0.022734764852865758,
                    0.039545157148639175,
                    0.028567175596210945,
                ],
            }
        )
        old.write_csv(old_path)
    else:
        old = pl.read_csv(old_path)

    old_eval = pl.DataFrame(
        {
            "timeframe": ["15m", "1d", "1h", "4h", "5m"],
            "oos_return_mean": [
                -0.00005019664506621768,
                0.0036794820359822428,
                0.00007633194416215746,
                0.0001803071355621306,
                -0.000022388120226203497,
            ],
            "oos_positive_rate": [
                0.4541504683911467,
                0.48013492743569863,
                0.46582416061345566,
                0.4622947106127534,
                0.43372129289577854,
            ],
            "unique_selected_factors": [20, 20, 20, 20, 20],
            "folds": [18398, 1167, 46558, 14715, 23767],
            "oos_rows": [1159074, 73521, 2933154, 927045, 1497321],
            "train_rows": [4636296, 294084, 11732616, 3708180, 5989284],
        }
    )
    new_eval = pl.read_csv(REPORTS / "walk_forward_evaluation_all.csv")
    lines = ["=== OOS OLD vs NEW ==="]
    for timeframe in TIMEFRAMES:
        o_old = old.filter(pl.col("timeframe") == timeframe)
        o_new = orient.filter(pl.col("timeframe") == timeframe)
        e_old = old_eval.filter(pl.col("timeframe") == timeframe)
        e_new = new_eval.filter(pl.col("timeframe") == timeframe)
        lines.append(
            f"{timeframe}\n"
            f"  selected: {int(e_old['unique_selected_factors'][0])} -> "
            f"{int(e_new['unique_selected_factors'][0])}\n"
            f"  folds: {int(e_old['folds'][0])} -> {int(e_new['folds'][0])}\n"
            f"  train_rows: {int(e_old['train_rows'][0])} -> {int(e_new['train_rows'][0])}\n"
            f"  oos_rows: {int(e_old['oos_rows'][0])} -> {int(e_new['oos_rows'][0])}\n"
            f"  oos_return_mean: {float(e_old['oos_return_mean'][0]):.8g} -> "
            f"{float(e_new['oos_return_mean'][0]):.8g}\n"
            f"  oos_positive_rate: {float(e_old['oos_positive_rate'][0]):.8g} -> "
            f"{float(e_new['oos_positive_rate'][0]):.8g}\n"
            f"  raw_oos_ic: {float(o_old['mean_raw_oos_ic'][0]):.8g} -> "
            f"{float(o_new['mean_raw_oos_ic'][0]):.8g}\n"
            f"  oriented_oos_ic: {float(o_old['mean_oriented_oos_ic'][0]):.8g} -> "
            f"{float(o_new['mean_oriented_oos_ic'][0]):.8g}"
        )
    text = "\n".join(lines) + "\n"
    (REPORTS / "eligibility_regen_oos_comparison.txt").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    print("Rebuilding orientation reports...")
    orient = rebuild_orientation()
    print(orient)
    print(membership_and_eligibility())
    print(oos_comparison(orient))
    print("Writing after hashes...")
    write_after_hashes()
    print(compare_hashes())


if __name__ == "__main__":
    main()

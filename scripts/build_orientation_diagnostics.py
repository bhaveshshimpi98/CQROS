"""Generate orientation diagnostic summaries and immutability hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from cqros.reporting.factor_orientation_diagnostic import (
    FactorOrientationReporter,
    build_factor_orientation_details,
    build_orientation_summary,
)


def _load_selected_factor_selection() -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for path in sorted(Path("data/factor_selection").rglob("*.parquet")):
        timeframe = path.parts[-2]
        year = int(path.stem)
        frame = pl.read_parquet(path).filter(pl.col("selected"))
        if frame.height == 0:
            continue
        parts.append(
            frame.select(
                [
                    pl.lit(timeframe).alias("timeframe"),
                    pl.lit(year, dtype=pl.Int32).alias("year"),
                    "factor_name",
                    "factor_version",
                    "selection_ic",
                    "selected_direction",
                    "orientation_policy",
                    pl.lit(None, dtype=pl.Float64).alias("raw_oos_ic"),
                    pl.lit(None, dtype=pl.Float64).alias("oriented_oos_ic"),
                ]
            )
        )
    return pl.concat(parts) if parts else pl.DataFrame()


def _factor_level_detail_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse fold/observation metrics to one row per selected factor."""
    finite_raw = pl.when(pl.col("raw_oos_ic").is_nan()).then(None).otherwise(pl.col("raw_oos_ic"))
    finite_oriented = (
        pl.when(pl.col("oriented_oos_ic").is_nan()).then(None).otherwise(pl.col("oriented_oos_ic"))
    )
    keys = ["timeframe", "year", "factor_name", "factor_version"]
    return (
        frame.sort(keys + (["fold_id"] if "fold_id" in frame.columns else []))
        .group_by(keys, maintain_order=True)
        .agg(
            [
                pl.col("selection_ic").first().alias("selection_ic"),
                pl.col("selected_direction").first().alias("selected_direction"),
                pl.col("orientation_policy").first().alias("orientation_policy"),
                finite_raw.mean().alias("raw_oos_ic"),
                finite_oriented.mean().alias("oriented_oos_ic"),
                pl.lit(None, dtype=pl.Int32).alias("fold_id"),
            ]
        )
    )


def _summarize_eval(path: Path, out_root: Path) -> pl.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        print(f"missing_or_empty={path}")
        return None
    frame = pl.read_csv(path)
    needed = [
        "timeframe",
        "year",
        "selection_ic",
        "selected_direction",
        "raw_oos_ic",
        "oriented_oos_ic",
    ]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        print(f"{path} missing_columns={missing}")
        return None
    summary = build_orientation_summary(frame)
    detail_source = (
        _factor_level_detail_frame(frame)
        if {"factor_name", "factor_version"}.issubset(frame.columns)
        else frame
    )
    details = build_factor_orientation_details(
        detail_source,
        manager="default",
        exchange="binance",
        market="usdt_perpetual",
    )
    FactorOrientationReporter(out_root).write_reports(
        summary=summary,
        factor_details=details,
    )
    print(f"wrote_orientation_summary={out_root}")
    print(summary.write_csv(None))
    return summary


def _write_hashes(path: Path) -> int:
    roots = [
        Path("data/walk_forward"),
        Path("data/walk_forward_evaluation"),
        Path("data/purged_cv"),
        Path("data/purged_cv_evaluation"),
        Path("data/factor_selection"),
    ]
    lines: list[str] = []
    for root in roots:
        for file_path in sorted(root.rglob("*.parquet")):
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {file_path.as_posix()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    selected = _load_selected_factor_selection()
    fs_summary = build_orientation_summary(selected)
    FactorOrientationReporter(Path("reports/factor_selection")).write_reports(
        summary=fs_summary,
        factor_details=build_factor_orientation_details(
            selected,
            manager="default",
            exchange="binance",
            market="usdt_perpetual",
        ),
    )
    print("FS_SUMMARY")
    print(fs_summary.write_csv(None))

    _summarize_eval(
        Path("reports/walk_forward/walk_forward_evaluation_factors.csv"),
        Path("reports/walk_forward"),
    )
    _summarize_eval(
        Path("reports/purged_cv/purged_cv_evaluation_factors.csv"),
        Path("reports/purged_cv"),
    )
    count = _write_hashes(Path("reports/orientation_hashes_after.txt"))
    print(f"after_hash_count={count}")


if __name__ == "__main__":
    main()

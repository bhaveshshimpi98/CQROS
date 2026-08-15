"""Helpers for factor-input-partitioning downstream regeneration evidence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "factor_stability" / "input_partition_downstream_regen"
BEFORE = OUT / "before"
AFTER = OUT / "after"

TIERS = (
    "data/factors/default/binance/usdt_perpetual",
    "data/factor_validation/default/binance/usdt_perpetual",
    "data/factor_selection/default/binance/usdt_perpetual",
    "data/walk_forward/default/binance/usdt_perpetual",
    "data/walk_forward_evaluation/default/binance/usdt_perpetual",
    "data/purged_cv/default/binance/usdt_perpetual",
    "data/purged_cv_evaluation/default/binance/usdt_perpetual",
)

FOCUS_TFS = ("5m", "15m", "1h", "4h", "1d")
KEY_FACTORS = (
    "price_volume_trend",
    "on_balance_volume",
    "open_interest_level",
    "rsi",
    "money_flow_index",
    "stochastic_k",
    "ease_of_movement",
    "rate_of_change",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_hash_manifest(path: Path, files: list[Path]) -> None:
    lines: list[str] = []
    for file_path in sorted(files, key=lambda p: str(p).replace("\\", "/")):
        rel = file_path.resolve().relative_to(ROOT).as_posix()
        lines.append(f"{_sha256(file_path)}  {rel}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote {len(lines)} hashes -> {path.relative_to(ROOT).as_posix()}")


def collect_all_parquets() -> list[Path]:
    files: list[Path] = []
    for tier in TIERS:
        root = ROOT / tier
        if root.exists():
            files.extend(root.rglob("*.parquet"))
    return files


def collect_focus_2026() -> list[Path]:
    files: list[Path] = []
    for tf in FOCUS_TFS:
        for tier in TIERS:
            root = ROOT / tier
            if not root.exists():
                continue
            if "factors" in tier or "factor_validation" in tier:
                files.extend(
                    p
                    for p in root.rglob("2026.parquet")
                    if p.parent.name == tf
                )
            else:
                candidate = root / tf / "2026.parquet"
                if candidate.exists():
                    files.append(candidate)
    return files


def snapshot_btcusdt_1d_factors(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    factor_path = (
        ROOT
        / "data/factors/default/binance/usdt_perpetual/BTCUSDT/1d/2026.parquet"
    )
    df = pl.read_parquet(factor_path)
    rows: list[dict[str, object]] = []
    if "factor_name" in df.columns:
        value_col = "value" if "value" in df.columns else None
        for name in KEY_FACTORS:
            sub = df.filter(pl.col("factor_name") == name)
            if value_col is None:
                non_null = sub
            else:
                non_null = sub.filter(pl.col(value_col).is_not_null())
            first_ts = None
            last_ts = None
            if "open_time" in non_null.columns and non_null.height:
                first_ts = non_null["open_time"].min()
                last_ts = non_null["open_time"].max()
            rows.append(
                {
                    "factor_name": name,
                    "rows": sub.height,
                    "non_null": non_null.height,
                    "null_rate": (
                        1.0 - (non_null.height / sub.height) if sub.height else None
                    ),
                    "first_valid": first_ts,
                    "last_valid": last_ts,
                    "unique_open_times": (
                        non_null["open_time"].n_unique()
                        if "open_time" in non_null.columns and non_null.height
                        else 0
                    ),
                }
            )
        # also overall unique timestamps in partition
        unique_all = df["open_time"].n_unique() if "open_time" in df.columns else None
        meta = pl.DataFrame(
            {
                "metric": ["partition_rows", "unique_open_times", "n_factors"],
                "value": [
                    str(df.height),
                    str(unique_all),
                    str(df["factor_name"].n_unique()),
                ],
            }
        )
        meta.write_csv(dest / "btc_1d_partition_meta.csv")
    else:
        for name in KEY_FACTORS:
            if name not in df.columns:
                continue
            non_null = df.filter(pl.col(name).is_not_null())
            rows.append(
                {
                    "factor_name": name,
                    "rows": df.height,
                    "non_null": non_null.height,
                    "null_rate": 1.0 - (non_null.height / df.height),
                    "first_valid": (
                        non_null["open_time"].min()
                        if "open_time" in non_null.columns and non_null.height
                        else None
                    ),
                    "last_valid": (
                        non_null["open_time"].max()
                        if "open_time" in non_null.columns and non_null.height
                        else None
                    ),
                    "unique_open_times": (
                        non_null["open_time"].n_unique()
                        if "open_time" in non_null.columns and non_null.height
                        else 0
                    ),
                }
            )
    coverage = pl.DataFrame(rows)
    coverage.write_csv(dest / "btc_1d_factor_coverage.csv")
    print(coverage.write_csv(None))


def snapshot_selection(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data/factor_selection/default/binance/usdt_perpetual/1d/2026.parquet"
    sel = pl.read_parquet(path)
    sel.write_parquet(dest / "1d_factor_selection.parquet")
    sel.write_csv(dest / "1d_factor_selection.csv")
    cols = [c for c in sel.columns if c in {
        "factor_name",
        "selected",
        "selected_direction",
        "eligible",
        "eligibility_status",
        "orientation",
        "signed_ic",
        "coverage",
        "rank",
        "is_selected",
    }]
    print("selection columns:", sel.columns)
    if cols:
        print(sel.select(cols))
    else:
        print(sel.head(20))


def snapshot_validation(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # panel-level validation may be under a timeframe folder, not per-symbol
    root = ROOT / "data/factor_validation/default/binance/usdt_perpetual"
    # Discover 1d 2026 partitions (may be panel artifacts)
    files = [p for p in root.rglob("2026.parquet") if p.parent.name == "1d"]
    print(f"validation 1d files: {len(files)}")
    # Also copy reports summary if present
    report = ROOT / "reports/factor_validation/default/validation_summary.csv"
    if report.exists():
        pl.read_csv(report).write_csv(dest / "validation_summary_copy.csv")


def snapshot_wf_pcv(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, rel in (
        ("wf", "data/walk_forward/default/binance/usdt_perpetual/1d/2026.parquet"),
        ("pcv", "data/purged_cv/default/binance/usdt_perpetual/1d/2026.parquet"),
        (
            "pcv_eval",
            "data/purged_cv_evaluation/default/binance/usdt_perpetual/1d/2026.parquet",
        ),
    ):
        path = ROOT / rel
        if not path.exists():
            print(f"missing {rel}")
            continue
        df = pl.read_parquet(path)
        df.write_parquet(dest / f"1d_{name}.parquet")
        df.write_csv(dest / f"1d_{name}.csv")
        print(f"{name}: rows={df.height} cols={df.columns}")


def cmd_before(_: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    BEFORE.mkdir(parents=True, exist_ok=True)
    write_hash_manifest(OUT / "hashes_before_all.txt", collect_all_parquets())
    write_hash_manifest(OUT / "hashes_before_2026.txt", collect_focus_2026())
    snapshot_btcusdt_1d_factors(BEFORE)
    snapshot_selection(BEFORE)
    snapshot_validation(BEFORE)
    snapshot_wf_pcv(BEFORE)


def cmd_hash(args: argparse.Namespace) -> None:
    label = args.label
    write_hash_manifest(OUT / f"hashes_{label}_all.txt", collect_all_parquets())
    write_hash_manifest(OUT / f"hashes_{label}_2026.txt", collect_focus_2026())


def cmd_after_snapshot(_: argparse.Namespace) -> None:
    AFTER.mkdir(parents=True, exist_ok=True)
    snapshot_btcusdt_1d_factors(AFTER)
    snapshot_selection(AFTER)
    snapshot_validation(AFTER)
    snapshot_wf_pcv(AFTER)


def cmd_compare_hashes(args: argparse.Namespace) -> None:
    before = OUT / args.before
    after = OUT / args.after

    def parse(path: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            digest, _, rel = line.partition("  ")
            result[rel.strip()] = digest.strip()
        return result

    b = parse(before)
    a = parse(after)
    changed = sorted(k for k in b.keys() & a.keys() if b[k] != a[k])
    added = sorted(a.keys() - b.keys())
    removed = sorted(b.keys() - a.keys())
    unchanged = sorted(k for k in b.keys() & a.keys() if b[k] == a[k])
    report = OUT / args.report
    lines = [
        f"BEFORE={before.name} AFTER={after.name}",
        f"changed={len(changed)} unchanged={len(unchanged)} added={len(added)} removed={len(removed)}",
        "",
        "CHANGED:",
    ]
    lines.extend(f"  {p}" for p in changed)
    lines.append("")
    lines.append("ADDED:")
    lines.extend(f"  {p}" for p in added)
    lines.append("")
    lines.append("REMOVED:")
    lines.extend(f"  {p}" for p in removed)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"... wrote {report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_before = sub.add_parser("before")
    p_before.set_defaults(func=cmd_before)

    p_hash = sub.add_parser("hash")
    p_hash.add_argument("--label", required=True)
    p_hash.set_defaults(func=cmd_hash)

    p_after = sub.add_parser("after-snapshot")
    p_after.set_defaults(func=cmd_after_snapshot)

    p_cmp = sub.add_parser("compare-hashes")
    p_cmp.add_argument("--before", required=True)
    p_cmp.add_argument("--after", required=True)
    p_cmp.add_argument("--report", required=True)
    p_cmp.set_defaults(func=cmd_compare_hashes)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

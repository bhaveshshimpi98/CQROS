"""Post-regeneration hashes, immutability checks, and Walk Forward structural validation.

Evidence script for the controlled 2026 Walk Forward regeneration.
Does not modify production artifacts except reading them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
STORAGE = ROOT / "data"
MANAGER = "default"
EXCHANGE = "binance"
MARKET = "usdt_perpetual"
YEAR = 2026
TIMEFRAMES = ("1d", "4h", "1h", "15m", "5m")
EVIDENCE = Path(__file__).resolve().parent

TIERS = ("factor_selection", "walk_forward", "factor_validation", "purged_cv")

MUST_NOT_CHANGE = ("factor_selection", "factor_validation", "purged_cv")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def panel_path(tier: str, timeframe: str) -> Path:
    return STORAGE / tier / MANAGER / EXCHANGE / MARKET / timeframe / f"{YEAR}.parquet"


def file_record(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "exists": False,
        }
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "exists": True,
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "sha256": sha256_file(path),
    }


def dump_wf_summary(path: Path, timeframe: str) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from cqros.walk_forward.schema import (  # noqa: E402
        CANONICAL_COLUMN_ORDER,
        PRIMARY_KEY_COLUMNS,
    )
    from cqros.walk_forward.verifier import WalkForwardVerifier  # noqa: E402

    frame = pl.read_parquet(path)
    columns = list(frame.columns)
    missing = [col for col in CANONICAL_COLUMN_ORDER if col not in columns]
    unexpected = [col for col in columns if col not in CANONICAL_COLUMN_ORDER]
    dup_keys = 0
    if all(col in columns for col in PRIMARY_KEY_COLUMNS):
        dup_keys = frame.group_by(list(PRIMARY_KEY_COLUMNS)).len().filter(pl.col("len") > 1).height
    status_counts: dict[str, int] = {}
    if "status" in columns:
        for row in frame.group_by("status").len().sort("status").iter_rows(named=True):
            status_counts[str(row["status"])] = int(row["len"])
    timeframe_mismatch = 0
    if "timeframe" in columns:
        timeframe_mismatch = frame.filter(pl.col("timeframe") != timeframe).height

    now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    future_ts = 0
    for col in ("train_start", "train_end", "test_start", "test_end"):
        if col in columns:
            future_ts += int(frame.filter(pl.col(col) > now_ms).height)

    window_order_bad = 0
    if all(col in columns for col in ("train_start", "train_end", "test_start", "test_end")):
        window_order_bad = int(
            frame.filter(
                (pl.col("train_end") < pl.col("train_start"))
                | (pl.col("test_end") < pl.col("test_start"))
                | (pl.col("test_start") < pl.col("train_start"))
            ).height
        )

    fold_ids = frame["fold_id"].to_list() if "fold_id" in columns else []
    sorted_by_fold = fold_ids == sorted(fold_ids) if fold_ids else True

    report = WalkForwardVerifier().verify(frame)

    summary_dir = EVIDENCE / "wf_after"
    summary_dir.mkdir(parents=True, exist_ok=True)
    slim_cols = [col for col in CANONICAL_COLUMN_ORDER if col in columns]
    # Do not dump tens of thousands of fold rows into JSON; CSV is enough.
    frame.select(slim_cols).write_csv(summary_dir / f"{timeframe}_{YEAR}.csv")

    return {
        "timeframe": timeframe,
        "rows": frame.height,
        "columns": columns,
        "missing_required_columns": missing,
        "unexpected_columns": unexpected,
        "duplicate_primary_keys": dup_keys,
        "timeframe_mismatch_rows": timeframe_mismatch,
        "future_timestamp_rows": future_ts,
        "invalid_window_order_rows": window_order_bad,
        "status_counts": status_counts,
        "fold_id_min": min(fold_ids) if fold_ids else None,
        "fold_id_max": max(fold_ids) if fold_ids else None,
        "fold_count": len(fold_ids),
        "selected_factors_unique": (
            sorted(set(int(v) for v in frame["selected_factors"].to_list()))
            if "selected_factors" in columns
            else []
        ),
        "train_window_rows_unique": (
            sorted(set(int(v) for v in frame["train_rows"].to_list()))
            if "train_rows" in columns
            else []
        ),
        "test_window_rows_unique": (
            sorted(set(int(v) for v in frame["test_rows"].to_list()))
            if "test_rows" in columns
            else []
        ),
        "model_versions": (
            sorted(set(frame["model_version"].drop_nulls().to_list()))
            if "model_version" in columns
            else []
        ),
        "strategy_names": (
            sorted(set(frame["strategy_name"].drop_nulls().to_list()))
            if "strategy_name" in columns
            else []
        ),
        "strategy_versions": (
            sorted(set(frame["strategy_version"].drop_nulls().to_list()))
            if "strategy_version" in columns
            else []
        ),
        "has_factor_identity_columns": any(
            col in columns for col in ("factor_name", "factor_version")
        ),
        "sorted_by_fold_id": sorted_by_fold,
        "verifier_passed": bool(report.passed),
        "verifier_warnings": list(report.warnings),
        "verifier_row_count": int(report.rows_checked),
        "ok": (
            frame.height > 0
            and len(missing) == 0
            and len(unexpected) == 0
            and dup_keys == 0
            and timeframe_mismatch == 0
            and future_ts == 0
            and bool(report.passed)
        ),
    }


def selected_identities(timeframe: str) -> list[dict[str, object]]:
    path = panel_path("factor_selection", timeframe)
    frame = pl.read_parquet(path)
    selected = frame.filter(pl.col("selected") == True).sort(  # noqa: E712
        ["selection_rank", "factor_name", "factor_version"]
    )
    return [
        {
            "factor_name": row["factor_name"],
            "factor_version": row["factor_version"],
            "selected_direction": row["selected_direction"],
            "selection_rank": row["selection_rank"],
        }
        for row in selected.iter_rows(named=True)
    ]


def rss_peaks(rss_path: Path) -> dict[str, dict[str, float]]:
    if not rss_path.exists():
        return {}
    peaks: dict[str, dict[str, float]] = {}
    for line in rss_path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            ws = float(parts[3])
            priv = float(parts[4])
        except ValueError:
            continue
        tf = parts[5].strip()
        current = peaks.setdefault(tf, {"working_set_mb": 0.0, "private_mb": 0.0})
        current["working_set_mb"] = max(current["working_set_mb"], ws)
        current["private_mb"] = max(current["private_mb"], priv)
    return peaks


def main() -> int:
    timestamp = datetime.now(tz=UTC).isoformat()
    baseline = json.loads((EVIDENCE / "baseline.json").read_text(encoding="utf-8"))
    baseline_hashes = baseline["hashes"]

    hashes: dict[str, dict[str, object]] = {}
    for tier in TIERS:
        hashes[tier] = {}
        for timeframe in TIMEFRAMES:
            hashes[tier][timeframe] = file_record(panel_path(tier, timeframe))

    immutability: dict[str, dict[str, object]] = {}
    unexpected_changes: list[str] = []
    for tier in MUST_NOT_CHANGE:
        immutability[tier] = {}
        for timeframe in TIMEFRAMES:
            before = baseline_hashes[tier][timeframe]
            after = hashes[tier][timeframe]
            match = before.get("sha256") == after.get("sha256") and before.get("bytes") == after.get(
                "bytes"
            )
            immutability[tier][timeframe] = {
                "before_sha256": before.get("sha256"),
                "after_sha256": after.get("sha256"),
                "unchanged": match,
            }
            if not match:
                unexpected_changes.append(f"{tier}/{timeframe}")

    wf_summaries: dict[str, object] = {}
    wf_ok = True
    for timeframe in TIMEFRAMES:
        record = hashes["walk_forward"][timeframe]
        if not record.get("exists"):
            wf_summaries[timeframe] = {"exists": False, "ok": False}
            wf_ok = False
            continue
        summary = dump_wf_summary(panel_path("walk_forward", timeframe), timeframe)
        before = baseline_hashes["walk_forward"][timeframe]
        summary["before_sha256"] = before.get("sha256")
        summary["after_sha256"] = record.get("sha256")
        summary["before_bytes"] = before.get("bytes")
        summary["after_bytes"] = record.get("bytes")
        summary["hash_changed"] = before.get("sha256") != record.get("sha256")
        summary["mtime_utc"] = record.get("mtime_utc")
        if not summary["ok"]:
            wf_ok = False
        wf_summaries[timeframe] = summary

    membership_before: dict[str, list[dict[str, object]]] = {}
    for timeframe in TIMEFRAMES:
        fs = baseline["factor_selection_verification"][timeframe]
        membership_before[timeframe] = fs.get("selected_factors", [])

    membership_after = {tf: selected_identities(tf) for tf in TIMEFRAMES}
    membership_cmp: dict[str, object] = {}
    lines = [
        "timeframe,factor_name,factor_version,direction_before,direction_after,in_before,in_after"
    ]
    for timeframe in TIMEFRAMES:
        before_map = {
            (row["factor_name"], row["factor_version"]): row for row in membership_before[timeframe]
        }
        after_map = {
            (row["factor_name"], row["factor_version"]): row for row in membership_after[timeframe]
        }
        keys = sorted(set(before_map) | set(after_map))
        entered = []
        removed = []
        direction_changes = []
        for key in keys:
            b = before_map.get(key)
            a = after_map.get(key)
            lines.append(
                ",".join(
                    [
                        timeframe,
                        key[0],
                        str(key[1]),
                        "" if b is None else str(b.get("selected_direction")),
                        "" if a is None else str(a.get("selected_direction")),
                        str(b is not None).lower(),
                        str(a is not None).lower(),
                    ]
                )
            )
            if b is None and a is not None:
                entered.append(key[0])
            elif b is not None and a is None:
                removed.append(key[0])
            elif b is not None and a is not None:
                if b.get("selected_direction") != a.get("selected_direction"):
                    direction_changes.append(
                        {
                            "factor_name": key[0],
                            "before": b.get("selected_direction"),
                            "after": a.get("selected_direction"),
                        }
                    )
        membership_cmp[timeframe] = {
            "entered": entered,
            "removed": removed,
            "direction_changes": direction_changes,
            "note": (
                "Factor Selection membership is the WF input. Canonical WF parquet "
                "does not store factor identities; this comparison is FS lineage, "
                "not WF columns. FS must be unchanged during this task, so entered/"
                "removed should be empty versus the pre-WF-regeneration FS snapshot."
            ),
        }
    (EVIDENCE / "membership_before_after.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hash_lines = ["timeframe,tier,exists,bytes,mtime_utc,sha256,vs_baseline"]
    for tier in TIERS:
        for timeframe in TIMEFRAMES:
            record = hashes[tier][timeframe]
            before = baseline_hashes[tier][timeframe]
            changed = "CHANGED" if record.get("sha256") != before.get("sha256") else "UNCHANGED"
            hash_lines.append(
                ",".join(
                    [
                        timeframe,
                        tier,
                        str(bool(record.get("exists"))).lower(),
                        str(record.get("bytes", "")),
                        str(record.get("mtime_utc", "")),
                        str(record.get("sha256", "")),
                        changed,
                    ]
                )
            )
    (EVIDENCE / "hashes_after.csv").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    peaks = rss_peaks(EVIDENCE / "logs" / "rss_sampler.csv")
    run_meta = {}
    meta_path = EVIDENCE / "run_meta.json"
    if meta_path.exists():
        run_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    comparison = []
    for timeframe in TIMEFRAMES:
        before_s = baseline["walk_forward_summaries"][timeframe]
        after_s = wf_summaries[timeframe]
        comparison.append(
            {
                "timeframe": timeframe,
                "baseline_sha": before_s.get("current_sha256"),
                "after_sha": after_s.get("after_sha256") if isinstance(after_s, dict) else None,
                "bytes_before": baseline_hashes["walk_forward"][timeframe].get("bytes"),
                "bytes_after": hashes["walk_forward"][timeframe].get("bytes"),
                "rows_before": before_s.get("rows"),
                "rows_after": after_s.get("rows") if isinstance(after_s, dict) else None,
                "hash_changed": (
                    before_s.get("current_sha256") != hashes["walk_forward"][timeframe].get("sha256")
                ),
                "schema_diff": {
                    "missing_after": after_s.get("missing_required_columns")
                    if isinstance(after_s, dict)
                    else None,
                    "unexpected_after": after_s.get("unexpected_columns")
                    if isinstance(after_s, dict)
                    else None,
                    "model_versions_before": before_s.get("model_versions"),
                    "model_versions_after": after_s.get("model_versions")
                    if isinstance(after_s, dict)
                    else None,
                    "train_rows_before": before_s.get("train_window_rows_unique"),
                    "train_rows_after": after_s.get("train_window_rows_unique")
                    if isinstance(after_s, dict)
                    else None,
                    "test_rows_before": before_s.get("test_window_rows_unique"),
                    "test_rows_after": after_s.get("test_window_rows_unique")
                    if isinstance(after_s, dict)
                    else None,
                },
                "fs_membership_delta": membership_cmp[timeframe],
            }
        )

    stop_required = (not wf_ok) or len(unexpected_changes) > 0
    payload = {
        "timestamp_utc": timestamp,
        "hashes": hashes,
        "immutability": immutability,
        "unexpected_non_wf_changes": unexpected_changes,
        "walk_forward_summaries": wf_summaries,
        "comparison": comparison,
        "factor_selection_membership_after": membership_after,
        "rss_peaks": peaks,
        "run_meta": run_meta,
        "wf_ok": wf_ok,
        "stop_required": stop_required,
    }
    (EVIDENCE / "after.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stop_required": stop_required,
                "wf_ok": wf_ok,
                "unexpected_non_wf_changes": unexpected_changes,
            },
            indent=2,
        )
    )
    return 1 if stop_required else 0


if __name__ == "__main__":
    raise SystemExit(main())

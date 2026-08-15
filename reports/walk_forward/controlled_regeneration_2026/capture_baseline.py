"""Capture pre-regeneration hashes, Factor Selection inputs, and Walk Forward panels.

Evidence script for the controlled 2026 Walk Forward regeneration.
Does not modify production artifacts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
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

# Post Factor Selection regeneration hashes (authoritative upstream for this task).
CURRENT_FS_EXPECTED = {
    "1d": "B6CE50C27CAE6601FC0CED1CB650475BF471C337211D5C89D98B1FEB8432A2BB",
    "4h": "BAE924AA39D5DDA300030052B5C69A55DA94DC230A30931ECFDE911BD1592918",
    "1h": "89F17D02C7A4B6314A785DC8783836F88B233F9F04A83AD76D94E3AFEEC10022",
    "15m": "898A40F41AF52452DB3D170C667DD6A2678CB7B9317C7A2D48CC7518DBE12788",
    "5m": "1C80C6679DAA8133CED64C9D7978F02E4E9742CEBEECECB4DA4289F569F97FB0",
}

# Walk Forward hashes recorded as unchanged during Factor Selection regeneration.
PRIOR_WF_EXPECTED = {
    "1d": "87E74C6C2AF48B78B6E3D55A97967DBEB101CB48343A1C78198E01F63D228087",
    "4h": "888C4613FA4D3DD8D3345AF9F134D13DCAC2EF35641C073234EAB68E19C01C8F",
    "1h": "735E0D62872F436A2BFA9A6A84E9EC5CBC4602703C5C7B43A4CA2462F2476373",
    "15m": "97A8BA631707E8E74FBD13BC1577D5AE9EB9156A3F223B6D1D7F4D3882A73806",
    "5m": "937C16B8A98BF5C0C7F95CEA80A9A8B2EAA6EE2828423298951057995730E22D",
}

PRIOR_FV_EXPECTED = {
    "1d": "B7935E021B31BAD5BE9017577FCD49243A1E022480A47F44A5B6D5D2C4058137",
    "4h": "E49A86299CD989A8B1F5B91ABF90E92647993250A71351391EA2F9752F481EED",
    "1h": "315EF539F46E94A7B28AB902D98E88D4B79EB7BD666687E5FCED6FF1B7C3CC30",
    "15m": "57FEF604E3F24DA6FBEEC836D5ADCB2979452DAA906A5C60639DA89CCAC38CB2",
    "5m": "7F81C68E92FD51058DA13F54CDBA3E8F2981357F69D7F4E75A7FFB9474DA213D",
}

PRIOR_PCV_EXPECTED = {
    "1d": "C517DC596ABE5F397FF16B0713CCD8781B0DF8B23569BA1C1934F357496A15FB",
    "4h": "2185AB07C048E97C64155BE3004E7DB7C1FFA3101FA39FCA58DFBC6EE46DD1D4",
    "1h": "249A8119215FA02CBEAFE2D7E946412CD7441F9E456339A5613EB59D702F0767",
    "15m": "7D6EF28BD26F4A3CAE4E15A3CDBB3DEE5F89CD1625CACFDF61E14723ADE7AE4E",
    "5m": "15F3745A9CEAFE9C62076B38641C72C752A199D7765B3C7E9EB59FC41665FA26",
}

TIERS = {
    "factor_selection": STORAGE / "factor_selection",
    "walk_forward": STORAGE / "walk_forward",
    "factor_validation": STORAGE / "factor_validation",
    "purged_cv": STORAGE / "purged_cv",
}

CONFIG_FILES = (
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
    ROOT / "src" / "cqros" / "walk_forward" / "engine.py",
    ROOT / "src" / "cqros" / "walk_forward" / "evaluation_input.py",
    ROOT / "src" / "cqros" / "walk_forward" / "pipeline.py",
    ROOT / "src" / "cqros" / "walk_forward" / "schema.py",
    ROOT / "src" / "cqros" / "cli" / "generate_walk_forward.py",
)


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


def git_capture() -> dict[str, object]:
    def run(args: list[str]) -> dict[str, object]:
        try:
            completed = subprocess.run(
                args,
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }

    status = run(["git", "status", "--porcelain"])
    log = run(["git", "log", "-1", "--oneline"])
    rev = run(["git", "rev-parse", "HEAD"])
    (EVIDENCE / "git_status_before.txt").write_text(
        (status.get("stdout") or status.get("stderr") or "") + "\n",
        encoding="utf-8",
    )
    (EVIDENCE / "git_log_before.txt").write_text(
        (log.get("stdout") or log.get("stderr") or "") + "\n",
        encoding="utf-8",
    )
    return {
        "status": status,
        "log": log,
        "rev_parse": rev,
        "commits_exist": bool(rev.get("ok")) and bool(rev.get("stdout")),
        "working_tree_clean": bool(status.get("ok")) and status.get("stdout") == "",
    }


def dump_fs_verification(path: Path, timeframe: str) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from cqros.factor_selection.schema import (  # noqa: E402
        CANONICAL_COLUMN_ORDER,
        PRIMARY_KEY_COLUMNS,
        FactorSelectionStatus,
    )

    frame = pl.read_parquet(path)
    columns = list(frame.columns)
    missing_required = [col for col in CANONICAL_COLUMN_ORDER if col not in columns]
    identity_cols = [col for col in ("factor_name", "factor_version", "timeframe") if col in columns]
    dup_identities = 0
    if identity_cols:
        dup_identities = frame.group_by(identity_cols).len().filter(pl.col("len") > 1).height

    selected_true = frame.filter(pl.col("selected") == True) if "selected" in columns else None  # noqa: E712
    selected_status = (
        frame.filter(pl.col("status") == FactorSelectionStatus.SELECTED.value)
        if "status" in columns
        else None
    )
    selected_mismatch = 0
    if selected_true is not None and "status" in columns:
        selected_mismatch = selected_true.filter(
            pl.col("status") != FactorSelectionStatus.SELECTED.value
        ).height
        rejected_but_selected = frame.filter(
            (pl.col("selected") == False)  # noqa: E712
            & (pl.col("status") == FactorSelectionStatus.SELECTED.value)
        ).height
        selected_mismatch += rejected_but_selected

    timeframe_mismatch = 0
    if "timeframe" in columns:
        timeframe_mismatch = frame.filter(pl.col("timeframe") != timeframe).height

    selected_factors = []
    if selected_true is not None:
        order_cols = [col for col in ("selection_rank", "factor_name", "factor_version") if col in columns]
        ordered_selected = selected_true.sort(order_cols) if order_cols else selected_true
        selected_factors = [
            {
                "factor_name": row["factor_name"],
                "factor_version": row.get("factor_version"),
                "selected_direction": row.get("selected_direction"),
                "selection_rank": row.get("selection_rank"),
                "selection_score": row.get("selection_score"),
                "selection_ic": row.get("selection_ic"),
                "selection_reason": row.get("selection_reason"),
                "status": row.get("status"),
            }
            for row in ordered_selected.iter_rows(named=True)
        ]

    membership_dir = EVIDENCE / "membership_before"
    membership_dir.mkdir(parents=True, exist_ok=True)
    export_cols = [
        col
        for col in (
            "factor_name",
            "factor_version",
            "timeframe",
            "selected",
            "selection_rank",
            "selection_score",
            "selection_reason",
            "selection_ic",
            "selected_direction",
            "orientation_policy",
            "status",
        )
        if col in columns
    ]
    frame.sort(["factor_name", "factor_version"]).select(export_cols).write_csv(
        membership_dir / f"{timeframe}_{YEAR}.csv"
    )

    return {
        "timeframe": timeframe,
        "rows": frame.height,
        "columns": columns,
        "missing_required_columns": missing_required,
        "primary_key_columns": list(PRIMARY_KEY_COLUMNS),
        "duplicate_identities": dup_identities,
        "timeframe_mismatch_rows": timeframe_mismatch,
        "selected_count": selected_true.height if selected_true is not None else None,
        "selected_status_count": selected_status.height if selected_status is not None else None,
        "selected_status_mismatch": selected_mismatch,
        "orientation_policies": (
            sorted(set(frame["orientation_policy"].drop_nulls().to_list()))
            if "orientation_policy" in columns
            else []
        ),
        "year_inferred_from_path": YEAR,
        "manager": MANAGER,
        "exchange": EXCHANGE,
        "market": MARKET,
        "readable": True,
        "selected_factors": selected_factors,
        "ok": (
            len(missing_required) == 0
            and dup_identities == 0
            and timeframe_mismatch == 0
            and selected_mismatch == 0
            and frame.height > 0
        ),
    }


def dump_wf_summary(path: Path, timeframe: str) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from cqros.walk_forward.schema import (  # noqa: E402
        CANONICAL_COLUMN_ORDER,
        PRIMARY_KEY_COLUMNS,
    )

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

    fold_ids = frame["fold_id"].to_list() if "fold_id" in columns else []
    model_versions = (
        sorted(set(frame["model_version"].drop_nulls().to_list()))
        if "model_version" in columns
        else []
    )
    strategy_names = (
        sorted(set(frame["strategy_name"].drop_nulls().to_list()))
        if "strategy_name" in columns
        else []
    )
    strategy_versions = (
        sorted(set(frame["strategy_version"].drop_nulls().to_list()))
        if "strategy_version" in columns
        else []
    )
    has_factor_identity = any(
        col in columns for col in ("factor_name", "factor_version", "selected_factors_list")
    )

    summary_dir = EVIDENCE / "wf_before"
    summary_dir.mkdir(parents=True, exist_ok=True)
    frame.write_csv(summary_dir / f"{timeframe}_{YEAR}.csv")

    return {
        "timeframe": timeframe,
        "rows": frame.height,
        "columns": columns,
        "missing_required_columns": missing,
        "unexpected_columns": unexpected,
        "duplicate_primary_keys": dup_keys,
        "timeframe_mismatch_rows": timeframe_mismatch,
        "status_counts": status_counts,
        "fold_ids": fold_ids,
        "selected_factors_unique": (
            sorted(set(frame["selected_factors"].to_list()))
            if "selected_factors" in columns
            else []
        ),
        "train_window_rows_unique": (
            sorted(set(frame["train_rows"].to_list())) if "train_rows" in columns else []
        ),
        "test_window_rows_unique": (
            sorted(set(frame["test_rows"].to_list())) if "test_rows" in columns else []
        ),
        "model_versions": model_versions,
        "strategy_names": strategy_names,
        "strategy_versions": strategy_versions,
        "has_factor_identity_columns": has_factor_identity,
        "train_start_min": int(frame["train_start"].min()) if "train_start" in columns else None,
        "train_end_max": int(frame["train_end"].max()) if "train_end" in columns else None,
        "test_start_min": int(frame["test_start"].min()) if "test_start" in columns else None,
        "test_end_max": int(frame["test_end"].max()) if "test_end" in columns else None,
        "sorted_by_fold_id": fold_ids == sorted(fold_ids) if fold_ids else True,
    }


def count_lake_files(tier: str, timeframe: str) -> dict[str, object]:
    root = STORAGE / tier / MANAGER / EXCHANGE / MARKET
    if not root.exists():
        return {"symbol_files": 0, "total_bytes": 0, "exists": False}
    files = list(root.glob(f"*/{timeframe}/{YEAR}.parquet"))
    return {
        "symbol_files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "exists": len(files) > 0,
    }


def count_label_files(timeframe: str) -> dict[str, object]:
    root = STORAGE / "labels" / EXCHANGE / MARKET
    if not root.exists():
        return {"symbol_files": 0, "total_bytes": 0, "exists": False}
    files = list(root.glob(f"*/{timeframe}/{YEAR}.parquet"))
    return {
        "symbol_files": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "exists": len(files) > 0,
    }


def main() -> int:
    timestamp = datetime.now(tz=UTC).isoformat()
    git_info = git_capture()

    hashes: dict[str, dict[str, object]] = {}
    for tier in TIERS:
        hashes[tier] = {}
        for timeframe in TIMEFRAMES:
            hashes[tier][timeframe] = file_record(panel_path(tier, timeframe))

    config_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): file_record(path) for path in CONFIG_FILES}

    fs_verification: dict[str, object] = {}
    fs_ok = True
    for timeframe in TIMEFRAMES:
        record = hashes["factor_selection"][timeframe]
        if not record.get("exists"):
            fs_verification[timeframe] = {"exists": False, "ok": False}
            fs_ok = False
            continue
        summary = dump_fs_verification(panel_path("factor_selection", timeframe), timeframe)
        expected = CURRENT_FS_EXPECTED[timeframe]
        current = str(record["sha256"])
        summary["matches_current_fs_expected"] = current == expected
        summary["expected_sha256"] = expected
        summary["current_sha256"] = current
        if not summary["ok"] or current != expected:
            fs_ok = False
        fs_verification[timeframe] = summary

    wf_summaries: dict[str, object] = {}
    for timeframe in TIMEFRAMES:
        record = hashes["walk_forward"][timeframe]
        if not record.get("exists"):
            wf_summaries[timeframe] = {"exists": False}
            continue
        summary = dump_wf_summary(panel_path("walk_forward", timeframe), timeframe)
        expected = PRIOR_WF_EXPECTED[timeframe]
        summary["matches_prior_wf_expected"] = str(record["sha256"]) == expected
        summary["expected_sha256"] = expected
        summary["current_sha256"] = str(record["sha256"])
        wf_summaries[timeframe] = summary

    factors_lake = {tf: count_lake_files("factors", tf) for tf in TIMEFRAMES}
    labels_lake = {tf: count_label_files(tf) for tf in TIMEFRAMES}

    sys.path.insert(0, str(ROOT / "src"))
    from cqros.cli.generate_walk_forward import (  # noqa: E402
        _DEFAULT_ENGINE,
        _DEFAULT_WORKER_COUNT,
        build_parser,
    )
    from cqros.walk_forward.engine import (  # noqa: E402
        _DEFAULT_MODEL_VERSION,
        _DEFAULT_STEP_SIZE,
        _DEFAULT_STRATEGY_NAME,
        _DEFAULT_STRATEGY_VERSION,
        _DEFAULT_TEST_WINDOW,
        _DEFAULT_TRAIN_WINDOW,
        SimpleWalkForwardEngine,
    )

    parser = build_parser()
    cli_flags = sorted(action.dest for action in parser._actions if action.dest != "help")

    semantics = {
        "cli_module": "cqros.cli.generate_walk_forward",
        "cli_flags": cli_flags,
        "default_engine": _DEFAULT_ENGINE,
        "default_workers_from_research_config": _DEFAULT_WORKER_COUNT,
        "execution_mode_flag_exists": "execution_mode" in cli_flags,
        "memory_efficient_flag_exists": False,
        "factor_batch_size_flag_exists": "factor_batch_size" in cli_flags,
        "overwrite_required_because_partitions_exist": True,
        "engine_class": SimpleWalkForwardEngine.__name__,
        "train_window": _DEFAULT_TRAIN_WINDOW,
        "test_window": _DEFAULT_TEST_WINDOW,
        "step_size": _DEFAULT_STEP_SIZE,
        "strategy_name": _DEFAULT_STRATEGY_NAME,
        "strategy_version": _DEFAULT_STRATEGY_VERSION,
        "model_version": _DEFAULT_MODEL_VERSION,
        "window_style": "rolling_fixed_length",
        "purge_embargo_inside_wf": False,
        "canonical_discovery_sort": "(manager, timeframe) lexicographic, years sorted ascending",
        "canonical_all_tf_order": ["15m", "1d", "1h", "4h", "5m"],
        "controlled_execution_order": ["1d", "4h", "1h", "15m", "5m"],
        "inputs": [
            "Factor Selection panels",
            "Factors lake",
            "Labels future_return_1",
        ],
        "not_inputs": [
            "Factor Validation metrics",
            "Purged CV",
            "predictions",
            "signals",
            "alpha",
            "regime",
        ],
        "outputs": ["data/walk_forward/default/binance/usdt_perpetual/{tf}/2026.parquet"],
        "artifacts_cli_can_write": ["walk_forward parquet partitions only"],
        "python_version": sys.version,
        "polars_version": pl.__version__,
    }

    hash_vs_expected = {
        "factor_selection": {
            tf: {
                "current": hashes["factor_selection"][tf].get("sha256"),
                "expected": CURRENT_FS_EXPECTED[tf],
                "match": hashes["factor_selection"][tf].get("sha256") == CURRENT_FS_EXPECTED[tf],
            }
            for tf in TIMEFRAMES
        },
        "walk_forward": {
            tf: {
                "current": hashes["walk_forward"][tf].get("sha256"),
                "expected": PRIOR_WF_EXPECTED[tf],
                "match": hashes["walk_forward"][tf].get("sha256") == PRIOR_WF_EXPECTED[tf],
            }
            for tf in TIMEFRAMES
        },
        "factor_validation": {
            tf: {
                "current": hashes["factor_validation"][tf].get("sha256"),
                "expected": PRIOR_FV_EXPECTED[tf],
                "match": hashes["factor_validation"][tf].get("sha256") == PRIOR_FV_EXPECTED[tf],
            }
            for tf in TIMEFRAMES
        },
        "purged_cv": {
            tf: {
                "current": hashes["purged_cv"][tf].get("sha256"),
                "expected": PRIOR_PCV_EXPECTED[tf],
                "match": hashes["purged_cv"][tf].get("sha256") == PRIOR_PCV_EXPECTED[tf],
            }
            for tf in TIMEFRAMES
        },
    }

    missing: list[str] = []
    for timeframe in TIMEFRAMES:
        for tier in TIERS:
            if not hashes[tier][timeframe].get("exists"):
                missing.append(f"{tier}/{timeframe}/{YEAR}.parquet")
        if not factors_lake[timeframe]["exists"]:
            missing.append(f"factors lake {timeframe}/{YEAR}")
        if not labels_lake[timeframe]["exists"]:
            missing.append(f"labels lake {timeframe}/{YEAR}")
        if not hash_vs_expected["factor_selection"][timeframe]["match"]:
            missing.append(f"factor_selection/{timeframe} hash mismatch vs current FS regeneration")

    stop_required = (not fs_ok) or len(missing) > 0

    payload = {
        "timestamp_utc": timestamp,
        "host": "Windows",
        "git": git_info,
        "semantics": semantics,
        "hashes": hashes,
        "config_hashes": config_hashes,
        "hash_vs_expected": hash_vs_expected,
        "factor_selection_verification": fs_verification,
        "walk_forward_summaries": wf_summaries,
        "factors_lake": factors_lake,
        "labels_lake": labels_lake,
        "missing_or_ambiguous_inputs": missing,
        "stop_required": stop_required,
    }

    out_json = EVIDENCE / "baseline.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    hash_lines = [
        "timeframe,tier,exists,bytes,mtime_utc,sha256",
    ]
    for tier in TIERS:
        for timeframe in TIMEFRAMES:
            record = hashes[tier][timeframe]
            hash_lines.append(
                ",".join(
                    [
                        timeframe,
                        tier,
                        str(bool(record.get("exists"))).lower(),
                        str(record.get("bytes", "")),
                        str(record.get("mtime_utc", "")),
                        str(record.get("sha256", "")),
                    ]
                )
            )
    (EVIDENCE / "hashes_before.csv").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    print(json.dumps({"stop_required": stop_required, "missing": missing, "fs_ok": fs_ok}, indent=2))
    print(f"Wrote {out_json}")
    return 1 if stop_required else 0


if __name__ == "__main__":
    raise SystemExit(main())

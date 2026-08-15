"""Read-only Walk Forward input and artifact profiler.

This investigation script never invokes Walk Forward and never writes below
``data/``. It records canonical artifact hashes plus file-by-file Parquet
metadata and eager-size estimates while releasing each partition immediately.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import time
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
EVIDENCE = Path(__file__).resolve().parent
TIMEFRAMES = ("1d", "4h", "1h", "15m", "5m")
YEAR = 2026

FACTOR_COLUMNS = (
    "symbol",
    "timeframe",
    "open_time",
    "factor_name",
    "factor_version",
    "factor_value",
)
LABEL_COLUMNS = ("symbol", "timeframe", "open_time", "future_return_1")

ARTIFACT_TIERS = (
    "factor_selection",
    "factor_validation",
    "walk_forward",
    "purged_cv",
)


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def _memory_bytes() -> dict[str, int]:
    counters = _ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = kernel32.K32GetProcessMemoryInfo
    get_process_memory_info.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCountersEx),
        wintypes.DWORD,
    )
    get_process_memory_info.restype = wintypes.BOOL
    handle = get_current_process()
    ok = get_process_memory_info(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "working_set": int(counters.WorkingSetSize),
        "private": int(counters.PrivateUsage),
        "peak_working_set": int(counters.PeakWorkingSetSize),
        "peak_private": int(counters.PeakPagefileUsage),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _artifact_path(tier: str, timeframe: str) -> Path:
    return DATA / tier / "default" / "binance" / "usdt_perpetual" / timeframe / f"{YEAR}.parquet"


def capture_artifact_hashes() -> dict[str, object]:
    records: dict[str, object] = {}
    for tier in ARTIFACT_TIERS:
        records[tier] = {}
        for timeframe in TIMEFRAMES:
            path = _artifact_path(tier, timeframe)
            records[tier][timeframe] = {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return records


def _lake_paths(tier: str, timeframe: str) -> list[Path]:
    if tier == "factors":
        root = DATA / "factors" / "default" / "binance" / "usdt_perpetual"
    else:
        root = DATA / "labels" / "binance" / "usdt_perpetual"
    return sorted(root.glob(f"*/{timeframe}/{YEAR}.parquet"))


def _profile_file(path: Path, projected_columns: tuple[str, ...]) -> dict[str, object]:
    schema = pl.read_parquet_schema(path)
    missing = [column for column in projected_columns if column not in schema]
    if missing:
        raise RuntimeError(f"{path} missing projected columns: {missing}")

    before = _memory_bytes()
    started = time.perf_counter()
    frame = pl.read_parquet(path)
    full_loaded = _memory_bytes()
    full_estimated = frame.estimated_size()
    rows = frame.height
    columns = list(frame.columns)
    del frame

    projected = pl.read_parquet(path, columns=list(projected_columns))
    projected_loaded = _memory_bytes()
    projected_estimated = projected.estimated_size()
    del projected

    return {
        "path": path.relative_to(ROOT).as_posix(),
        "symbol": path.parents[1].name,
        "bytes_on_disk": path.stat().st_size,
        "rows": rows,
        "column_count": len(columns),
        "columns": columns,
        "schema": {name: str(dtype) for name, dtype in schema.items()},
        "full_estimated_bytes": full_estimated,
        "projected_columns": list(projected_columns),
        "projected_estimated_bytes": projected_estimated,
        "duration_seconds": time.perf_counter() - started,
        "memory_before": before,
        "memory_after_full_load": full_loaded,
        "memory_after_projected_load": projected_loaded,
    }


def profile_lake(tier: str, timeframe: str) -> dict[str, object]:
    projected_columns = FACTOR_COLUMNS if tier == "factors" else LABEL_COLUMNS
    files = [_profile_file(path, projected_columns) for path in _lake_paths(tier, timeframe)]
    column_sets = sorted({tuple(record["columns"]) for record in files})
    return {
        "tier": tier,
        "timeframe": timeframe,
        "year": YEAR,
        "file_count": len(files),
        "symbol_count": len({record["symbol"] for record in files}),
        "rows": sum(int(record["rows"]) for record in files),
        "bytes_on_disk": sum(int(record["bytes_on_disk"]) for record in files),
        "full_estimated_bytes": sum(int(record["full_estimated_bytes"]) for record in files),
        "projected_estimated_bytes": sum(
            int(record["projected_estimated_bytes"]) for record in files
        ),
        "column_sets": [list(columns) for columns in column_sets],
        "files": files,
    }


def main() -> int:
    started = time.perf_counter()
    payload = {
        "captured_at_utc": datetime.now(tz=UTC).isoformat(),
        "pid": os.getpid(),
        "python": os.sys.version,
        "polars": pl.__version__,
        "artifact_hashes_before": capture_artifact_hashes(),
        "process_memory_start": _memory_bytes(),
        "lakes": {
            timeframe: {
                "factors": profile_lake("factors", timeframe),
                "labels": profile_lake("labels", timeframe),
            }
            for timeframe in TIMEFRAMES
        },
        "process_memory_end": _memory_bytes(),
        "duration_seconds": time.perf_counter() - started,
    }
    output = EVIDENCE / "input_profile.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

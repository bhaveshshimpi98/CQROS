"""Verify protected production artifacts against investigation-start hashes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    profile = json.loads((EVIDENCE / "input_profile_clean.json").read_text(encoding="utf-8"))
    before = profile["artifact_hashes_before"]
    after: dict[str, dict[str, object]] = {}
    unchanged = True
    for tier, partitions in before.items():
        after[tier] = {}
        for timeframe, record in partitions.items():
            path = ROOT / record["path"]
            current = _sha256(path)
            match = current == record["sha256"]
            unchanged = unchanged and match
            after[tier][timeframe] = {
                "path": record["path"],
                "bytes": path.stat().st_size,
                "sha256": current,
                "before_sha256": record["sha256"],
                "unchanged": match,
            }
    payload = {
        "verified_at_utc": datetime.now(tz=UTC).isoformat(),
        "all_unchanged": unchanged,
        "artifacts": after,
    }
    output = EVIDENCE / "artifact_hashes_after.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())

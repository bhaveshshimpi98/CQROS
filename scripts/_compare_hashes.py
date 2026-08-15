"""Compare before/after hashes for the eligibility policy regeneration."""
from __future__ import annotations

from pathlib import Path

report_dir = Path("reports/factor_stability/factor_selection_policy_regeneration")
before_path = report_dir / "hashes_before.txt"
after_path = report_dir / "hashes_after.txt"


def _parse(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        hash_val, _, file_path = line.partition("  ")
        result[file_path.strip()] = hash_val.strip()
    return result


before = _parse(before_path)
after = _parse(after_path)

all_files = sorted(set(before) | set(after))

changed: list[str] = []
unchanged: list[str] = []
added: list[str] = []
removed: list[str] = []

for f in all_files:
    if f not in before:
        added.append(f)
    elif f not in after:
        removed.append(f)
    elif before[f] != after[f]:
        changed.append(f)
    else:
        unchanged.append(f)

lines: list[str] = []
lines.append("FACTOR SELECTION POLICY REGENERATION — HASH COMPARISON REPORT")
lines.append("=" * 70)
lines.append(f"Before files: {len(before)}  After files: {len(after)}")
lines.append("")

lines.append(f"CHANGED ({len(changed)}) — expected: factor_selection artifacts")
for f in changed:
    tier = f.split("\\")[0] if "\\" in f else f.split("/")[0]
    lines.append(f"  CHANGED  {f}")

lines.append("")
lines.append(f"UNCHANGED ({len(unchanged)}) — walk_forward and purged_cv must be here")
for f in unchanged:
    lines.append(f"  OK       {f}")

if added:
    lines.append(f"\nADDED ({len(added)})")
    for f in added:
        lines.append(f"  ADDED    {f}")

if removed:
    lines.append(f"\nREMOVED ({len(removed)})")
    for f in removed:
        lines.append(f"  REMOVED  {f}")

lines.append("")

# Classify by artifact tier
def _has_prefix(path: str, *prefixes: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return any(normalized.find(p.lower().replace("\\", "/")) >= 0 for p in prefixes)


factor_selection_changed = [f for f in changed if _has_prefix(f, "data/factor_selection")]
wf_changed = [f for f in changed if _has_prefix(f, "data/walk_forward")]
pcv_changed = [f for f in changed if _has_prefix(f, "data/purged_cv")]

lines.append("ARTIFACT CLASS SUMMARY")
lines.append(f"  Factor Selection: {'CHANGED as expected' if factor_selection_changed else 'UNCHANGED (unexpected)'} ({len(factor_selection_changed)} files changed)")
lines.append(f"  Walk-Forward:     {'UNCHANGED' if not wf_changed else f'CHANGED (VIOLATION) {wf_changed}'}")
lines.append(f"  Purged-CV:        {'UNCHANGED' if not pcv_changed else f'CHANGED (VIOLATION) {pcv_changed}'}")

report_text = "\n".join(lines)
print(report_text)

out = report_dir / "hash_comparison.txt"
out.write_text(report_text, encoding="utf-8")
print(f"\nReport written to {out}")

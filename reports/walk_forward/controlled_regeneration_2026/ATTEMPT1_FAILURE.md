# Attempt 1 — 4h memory stop

**Verdict for this attempt:** FAIL (host memory stop during `4h`)  
**Retry:** not performed (not classified as wrapper/transient)

---

## What ran

Sequential canonical CLI, workers=1, overwrite, engine default `simple`.

1. `1d` 2026 — succeeded, exit 0, 145.917 s, `OK 1d 2026 rows=70961`
2. `4h` 2026 — stopped at 71.627 s, wrapper exit 99 (`memory_stop_4h`)
3. `1h` / `15m` / `5m` — **not started**

Wrapper: `reports/walk_forward/controlled_regeneration_2026/run_regeneration.ps1`  
`$ErrorActionPreference` was **not** `Stop`. Stdout/stderr redirected to files. Native exit from `$LASTEXITCODE`.

---

## 4h failure evidence

| Item | Evidence |
|------|----------|
| Progress | `progress.log`: `MEMORY_STOP tf=4h pid=10860 private_mb=3464.47 threshold_mb=3200` |
| RSS | `logs/rss_sampler.csv` / `memory.csv`: python pid 10860, WS 1594.25 MB, Private **3464.47 MB** at `2026-08-14T11:10:56.5186411Z` |
| Stdout | `logs/generate_walk_forward_4h_stdout.log` empty (killed before progress print) |
| Stderr | `logs/generate_walk_forward_4h_stderr.log` UTF-16; 248× `Loaded factors dataset` and 248× `Loaded label dataset`; **no** `Assembling walk-forward evaluation input` |
| OOM allocator string | **not present** (`memory allocation of … failed` not observed) |
| Output parquet | **not overwritten**. SHA-256 identical to baseline; mtime still `2026-08-11T10:38:14Z`; readable 53,721 rows |

`4h` Factors lake has 283 symbol files. Stop occurred after 248/283 loads, before concat/join/engine/save. Remaining loads plus `pl.concat` would have increased memory further.

Host RAM is ≈3.7 GB. 3464 MB python private is approaching system exhaustion. The wrapper host-protection threshold (3200 MB private, `generate_walk_forward` processes only) stopped the child. This is not a research-policy change and not a new Walk Forward algorithm.

---

## Why this is not a same-command retry

Task rules allow retry only for infrastructure/wrapper/transient failure with no semantic/data issue.

This is a **genuine memory issue** on the canonical full-panel load path:

- Walk Forward CLI has **no** `--execution-mode` / spill / batch flag
- `WalkForwardInputBuilder` concatenates every Factors+Labels symbol panel in memory
- `4h` on-disk Factors are 147.8 MB compressed vs `1d` 27.6 MB; `1d` already peaked at 3071 MB private

Retrying the same command would re-enter the same load path. Changing the engine, reducing symbols, or inventing memory-efficient Walk Forward is **out of scope**.

---

## Partial artifact state (reconciled)

| Artifact | State |
|----------|--------|
| WF `1d` | Regenerated (new SHA-256). Readable. Verifier PASS. |
| WF `4h` | Baseline intact |
| WF `1h` `15m` `5m` | Baseline intact |
| Factor Selection (5) | Unchanged |
| Factor Validation (5) | Unchanged |
| Purged CV (5) | Unchanged |

No silent fallback. No downstream CLI invoked.

# Controlled Production Walk Forward Regeneration — 2026

**Date:** 2026-08-14  
**Host:** Windows / PowerShell, ≈3.7 GB RAM, Python 3.13.5, uv, Polars 1.43.0  
**Repository:** `D:/bss/CQROS`  
**Scope:** Controlled Walk Forward artifact regeneration from **current** Factor Selection only. No research-policy, formula, threshold, label, feature, market-data, Factor Validation, Factor Selection, Purged CV, evaluation, stability, prediction, signal, alpha, or regime changes.

Evidence directory: `reports/walk_forward/controlled_regeneration_2026/`

---

## 1. VERDICT

**FAIL**

`1d` 2026 Walk Forward completed on the canonical CLI. `4h` 2026 was stopped during Factors/Labels panel load when python private memory reached **3464.47 MB** (host-protection threshold 3200 MB on a ≈3.7 GB host). `1h`, `15m`, and `5m` were **not started**.

This is a genuine memory issue on the **existing** full-panel Walk Forward path. Walk Forward has no `--execution-mode` / spill / batch mechanism. No production code was patched. No optimization was invented. No same-command retry was performed.

Factor Selection, Factor Validation, and Purged CV SHA-256 values are identical to the pre-regeneration baseline. The `4h` Walk Forward parquet was **not** overwritten (baseline intact, readable).

Do not treat the completed `1d` partition as a five-timeframe PASS.

---

## 2. Exact command executed

Per-timeframe invocations (existing CLI flags; engine default `simple` omitted):

```text
uv run python -m cqros.cli.generate_walk_forward
  --manager default
  --timeframes <TF>
  --years 2026
  --overwrite
  --workers 1
  --storage-root data
  --verbose
```

Planned order: `1d → 4h → 1h → 15m → 5m` (controlled `--timeframes` selection).  
Canonical all-at-once discovery order would have been `15m → 1d → 1h → 4h → 5m` (lexicographic `(manager, timeframe)`). The algorithm was not modified.

Wrapper: `reports/walk_forward/controlled_regeneration_2026/run_regeneration.ps1`  
`$ErrorActionPreference` was **not** `Stop`. Stdout/stderr redirected to files. Child status from `$LASTEXITCODE`. `PYTHONUNBUFFERED=1` is wrapper logging only.

| TF | Started UTC | Ended UTC | Duration | Exit | Result |
|----|-------------|-----------|---------:|-----:|--------|
| 1d | 2026-08-14T11:07:17.405Z | 2026-08-14T11:09:43.323Z | 145.917 s | 0 | succeeded (`OK 1d 2026 rows=70961`) |
| 4h | 2026-08-14T11:09:45.814Z | 2026-08-14T11:10:57.442Z | 71.627 s | 99 | MEMORY_STOP |
| 1h | — | — | — | — | not started |
| 15m | — | — | — | — | not started |
| 5m | — | — | — | — | not started |

Overall: started `2026-08-14T11:07:15.443Z`, ended `2026-08-14T11:10:57.591Z`, **222.153 s**, exit 99. Source: `run_meta.json`, `progress.log`.

---

## 3. Baseline Factor Selection hashes

Captured `2026-08-14T10:59:50.331138+00:00` in `BASELINE.md` / `baseline.json`. Matched the Factor Selection regeneration report (authoritative upstream).

| TF | Bytes | SHA-256 |
|----|------:|---------|
| 1d | 11,915 | `B6CE50C27CAE6601FC0CED1CB650475BF471C337211D5C89D98B1FEB8432A2BB` |
| 4h | 11,827 | `BAE924AA39D5DDA300030052B5C69A55DA94DC230A30931ECFDE911BD1592918` |
| 1h | 11,816 | `89F17D02C7A4B6314A785DC8783836F88B233F9F04A83AD76D94E3AFEEC10022` |
| 15m | 11,813 | `898A40F41AF52452DB3D170C667DD6A2678CB7B9317C7A2D48CC7518DBE12788` |
| 5m | 11,812 | `1C80C6679DAA8133CED64C9D7978F02E4E9742CEBEECECB4DA4289F569F97FB0` |

All five: 73 rows, 20 selected, 0 duplicate identities, selected↔status consistent, `orientation_policy=signed_ic_v1`.

---

## 4. Baseline Walk Forward hashes

| TF | Bytes | SHA-256 | Rows |
|----|------:|---------|-----:|
| 1d | 185,255 | `87E74C6C2AF48B78B6E3D55A97967DBEB101CB48343A1C78198E01F63D228087` | 70,961 |
| 4h | 141,704 | `888C4613FA4D3DD8D3345AF9F134D13DCAC2EF35641C073234EAB68E19C01C8F` | 53,721 |
| 1h | 470,948 | `735E0D62872F436A2BFA9A6A84E9EC5CBC4602703C5C7B43A4CA2462F2476373` | 169,950 |
| 15m | 226,656 | `97A8BA631707E8E74FBD13BC1577D5AE9EB9156A3F223B6D1D7F4D3882A73806` | 67,164 |
| 5m | 390,938 | `937C16B8A98BF5C0C7F95CEA80A9A8B2EAA6EE2828423298951057995730E22D` | 86,762 |

---

## 5. Baseline Factor Validation hashes

| TF | Bytes | SHA-256 |
|----|------:|---------|
| 1d | 13,281 | `B7935E021B31BAD5BE9017577FCD49243A1E022480A47F44A5B6D5D2C4058137` |
| 4h | 14,365 | `E49A86299CD989A8B1F5B91ABF90E92647993250A71351391EA2F9752F481EED` |
| 1h | 14,444 | `315EF539F46E94A7B28AB902D98E88D4B79EB7BD666687E5FCED6FF1B7C3CC30` |
| 15m | 14,425 | `57FEF604E3F24DA6FBEEC836D5ADCB2979452DAA906A5C60639DA89CCAC38CB2` |
| 5m | 14,455 | `7F81C68E92FD51058DA13F54CDBA3E8F2981357F69D7F4E75A7FFB9474DA213D` |

---

## 6. Baseline Purged CV hashes

| TF | Bytes | SHA-256 |
|----|------:|---------|
| 1d | 5,996 | `C517DC596ABE5F397FF16B0713CCD8781B0DF8B23569BA1C1934F357496A15FB` |
| 4h | 5,989 | `2185AB07C048E97C64155BE3004E7DB7C1FFA3101FA39FCA58DFBC6EE46DD1D4` |
| 1h | 5,984 | `249A8119215FA02CBEAFE2D7E946412CD7441F9E456339A5613EB59D702F0767` |
| 15m | 5,998 | `7D6EF28BD26F4A3CAE4E15A3CDBB3DEE5F89CD1625CACFDF61E14723ADE7AE4E` |
| 5m | 5,995 | `15F3745A9CEAFE9C62076B38641C72C752A199D7765B3C7E9EB59FC41665FA26` |

---

## 7. Post-regeneration Walk Forward hashes

| TF | Bytes | SHA-256 | vs baseline |
|----|------:|---------|-------------|
| 1d | 185,255 | `B2391EE4ECBCD89E145015A488789415C410F6A5C050A818192EAEB7DFE58469` | **CHANGED** (regenerated) |
| 4h | 141,704 | `888C4613FA4D3DD8D3345AF9F134D13DCAC2EF35641C073234EAB68E19C01C8F` | UNCHANGED |
| 1h | 470,948 | `735E0D62872F436A2BFA9A6A84E9EC5CBC4602703C5C7B43A4CA2462F2476373` | UNCHANGED |
| 15m | 226,656 | `97A8BA631707E8E74FBD13BC1577D5AE9EB9156A3F223B6D1D7F4D3882A73806` | UNCHANGED |
| 5m | 390,938 | `937C16B8A98BF5C0C7F95CEA80A9A8B2EAA6EE2828423298951057995730E22D` | UNCHANGED |

---

## 8. Post-regeneration Factor Selection hashes

All five **UNCHANGED** (exact match to section 3).

---

## 9. Post-regeneration Factor Validation hashes

All five **UNCHANGED** (exact match to section 5).

---

## 10. Post-regeneration Purged CV hashes

All five **UNCHANGED** (exact match to section 6).

---

## 11. Per-timeframe runtime

| TF | Runtime | Notes |
|----|--------:|-------|
| 1d | 145.917 s wrapper / 137.325 s CLI summary | completed |
| 4h | 71.627 s | stopped during load |
| 1h | — | not started |
| 15m | — | not started |
| 5m | — | not started |

---

## 12. Per-timeframe memory peak

Python process whose command line contains `generate_walk_forward` (`memory.csv` / `logs/rss_sampler.csv`).

| TF | Peak WS (MB) | Peak Private (MB) | Timestamp UTC |
|----|-------------:|------------------:|---------------|
| 1d | 1472.34 | **3071.00** | 2026-08-14T11:08:02.915Z |
| 4h | 1594.25 | **3464.47** | 2026-08-14T11:10:56.519Z |
| 1h | — | — | not started |
| 15m | — | — | not started |
| 5m | — | — | not started |

`4h` stop threshold: 3200 MB private. No `memory allocation of … failed` string. `1d` completed just under the threshold.

---

## 13. Row counts

| TF | Rows before | Rows after | Status |
|----|------------:|-----------:|--------|
| 1d | 70,961 | 70,961 | regenerated; all PASS |
| 4h | 53,721 | 53,721 | baseline intact |
| 1h | 169,950 | 169,950 | baseline intact |
| 15m | 67,164 | 67,164 | baseline intact |
| 5m | 86,762 | 86,762 | baseline intact |

---

## 14. Structural validation

Existing contract: `WALK_FORWARD_SCHEMA` + `WalkForwardVerifier`.

**1d (regenerated):** `WalkForwardVerifier().verify` **passed**; `rows_checked=70961`; warnings empty; duplicate/null/NaN/invalid-timestamp counters 0; required columns only; `timeframe=1d`; `fold_id` 1…70961 sorted; `train_rows=252`; `test_rows=63`; `strategy_name=default_strategy`; `strategy_version=v1`; `model_version=v1`. CLI stdout: `OK 1d 2026 rows=70961`.

**4h / 1h / 15m / 5m:** not regenerated. Baseline files remain readable Parquet with the baseline schema. They are **not** accepted as regenerated outputs.

Canonical WF parquet has **no factor-identity columns**. `selected_factors` is a per-fold observation count in the evaluation frame (values 67–71 on `1d`), not the Factor Selection selected-set size of 20.

---

## 15. Factor Selection lineage comparison

Walk Forward reads current Factor Selection via `FactorSelectionRepository.load` then `WalkForwardInputBuilder` (Factors + Labels `future_return_1`).

`1d` Factor Selection SHA-256 was **unchanged** vs the current FS regeneration (already byte-identical). `1d` WF fold windows, `train_rows`, `test_rows`, `selected_factors`, and `train_start`/`train_end`/`test_start`/`test_end` match the baseline CSV exactly.

`1d` aggregate score columns (existing mapping: `train_score` = mean train score, `test_score` = mean test score, `overfit_gap` = walk-forward stability) differ:

| Field | Baseline unique value | Regenerated unique value |
|-------|----------------------:|-------------------------:|
| train_score | −0.0011972452989580306 | −0.0011972452989580304 (abs diff 2.17e-19) |
| test_score | −4.131648701133616e13 | −8.164794395582175e13 |
| overfit_gap | −0.025868255253005238 | −0.029794230955554148 |

Because `1d` Factor Selection membership did not change, this hash/score delta is **not** a selected-factor identity change. It is consistent with numerical sensitivity of the existing PASS-fold Sharpe/stability aggregates. No profitability or OOS claim is made.

`4h`/`1h`/`5m` Factor Selection **did** change in the prior FS task; those WF partitions were **not** regenerated here, so membership-driven WF updates for those timeframes remain pending.

---

## 16. Leakage audit

Static audit of `cqros.cli.generate_walk_forward` and `cqros.walk_forward` (engine, evaluation_input, pipeline, repository, schema): **no** imports of `cqros.purged_cv`, `cqros.alpha`, `cqros.regime`, predictions, or signals.

Existing inputs only:

- Factor Selection decisions (including `selected`, `selected_direction`, `orientation_policy=signed_ic_v1`)
- Factors observations
- Labels `future_return_1` (evaluation-only; not written back into Factor Selection)

Orientation is not recomputed from OOS labels (`evaluation_input.require_orientation_metadata`). Train/test folds are rolling 252/63/63 with train indices strictly before test on complete folds. No purge/embargo inside WF (that is Purged CV, not invoked).

No new information path was introduced. Production source was not modified.

---

## 17. Immutability verification

| Tier | Changed? |
|------|----------|
| Factor Selection (5) | **No** |
| Factor Validation (5) | **No** |
| Purged CV (5) | **No** |
| Walk Forward `1d` | **Yes** — approved overwrite; completed |
| Walk Forward `4h` `1h` `15m` `5m` | **No** |
| Factors / labels / market data | **No** (not regenerated) |
| Research policy / windows / engine | **No** |

`after.json` `unexpected_non_wf_changes=[]`.

---

## 18. Test results

Pre-execution:

```text
uv run pytest tests/unit/walk_forward/test_engine.py tests/unit/walk_forward/test_repository.py tests/unit/walk_forward/test_pipeline.py tests/unit/walk_forward/test_evaluation_input.py tests/unit/walk_forward/test_verifier.py tests/unit/walk_forward/test_schema.py tests/unit/walk_forward/test_registry.py tests/unit/cli/test_generate_walk_forward.py tests/unit/cli/test_verify_walk_forward.py -q --tb=short
```

**160 passed** in 202.45 s. Recorded in `tests_before.txt`.

Not run (downstream / out of scope): `generate_purged_cv`, `verify_purged_cv`, `evaluate_purged_cv`, `evaluate_walk_forward`, `diagnose_factor_stability`, `tests/unit/walk_forward/test_evaluation.py`.

---

## 19. Retry / failure evidence

See `ATTEMPT1_FAILURE.md`.

- Failure class: **memory / host exhaustion on canonical full-panel load**, not wrapper stderr, not OOM allocator string, not data corruption.
- `4h` loaded 248/283 Factors symbol files and 248 Labels files; never reached `assemble_walk_forward_input`; never wrote parquet.
- Same-command retry **not** performed (not transient; would repeat the same unbounded concat).
- Production code **not** patched.
- PowerShell UTF-16 stderr files are a wrapper encoding detail; `1d` stdout summary is UTF-8 and shows native success.

---

## 20. Remaining gaps

1. `4h`, `1h`, `15m`, `5m` Walk Forward partitions are **not** regenerated from current Factor Selection.
2. Walk Forward CLI/engine has **no** existing memory-efficient path. `1h` Factors lake is 434.8 MB on disk (larger than `4h` 147.8 MB). Completing those partitions on this host requires a **separate** investigation; it is not part of this regeneration task.
3. `1d` was overwritten. Fold geometry matches baseline; aggregate `test_score` / `overfit_gap` differ without an FS membership change. Not used as a five-TF success signal.
4. PowerShell `2>` logs are UTF-16. Counts were decoded explicitly.
5. Git has no usable commits (`safe.directory` / dubious ownership); git config was not modified.
6. Downstream Purged CV was **not** run (correct stop).

---

## 21. Explicit STOP statement

Controlled Walk Forward regeneration **did not complete** for all five 2026 partitions.

Do **not** run:

- `generate_purged_cv`
- `verify_purged_cv`
- `evaluate_purged_cv`
- `diagnose_factor_stability`
- prediction or signal generation

Do **not** invent a Walk Forward memory-efficient mode in this task.

Wait for explicit approval before any next step (memory investigation, host with more RAM, or a separately scoped implementation).

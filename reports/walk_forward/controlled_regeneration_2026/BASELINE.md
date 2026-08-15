# Controlled Walk Forward Regeneration — Investigation and Baseline

**Captured:** 2026-08-14T10:59:50.331138+00:00  
**Host:** Windows / PowerShell, Python 3.13.5, uv, Polars 1.43.0  
**Repository:** `D:/bss/CQROS`  
**Evidence:** `reports/walk_forward/controlled_regeneration_2026/`  
**stop_required (pre-execution):** `false`

This document records the investigation and pre-overwrite hashes. Production artifacts were not modified during capture.

---

## 1. Walk Forward CLI

| Item | Value |
|------|--------|
| Module | `cqros.cli.generate_walk_forward` |
| Entry | `uv run python -m cqros.cli.generate_walk_forward` |
| Required | `--manager` |
| Defaults | `--engine simple`, `--workers` = `ResearchConfig.worker_count` (4 on this host), `--storage-root data` |
| Filters | `--timeframes`, `--years` (omit = discover all existing Factor Selection partitions) |
| Overwrite | `--overwrite` required; existing partitions are skipped otherwise |
| Logging | `--verbose`, `--debug` |
| **Not present** | `--execution-mode`, `--factor-batch-size`, memory-efficient / spill flags |

Walk Forward has **no** bounded-memory execution mode. The existing canonical path loads Factors + Labels for every symbol in the panel, concatenates, joins, then runs `SimpleWalkForwardEngine`. This task does **not** invent an optimization.

Default engine is `simple` (`SimpleWalkForwardEngine`). This regeneration will omit `--engine` so the existing default is used.

---

## 2. Semantics locked (unchanged)

| Setting | Existing value |
|---------|----------------|
| Engine | `simple` / `SimpleWalkForwardEngine` |
| Train window | 252 rows |
| Test window | 63 rows |
| Step size | 63 rows |
| Window style | rolling, fixed length; train strictly before test on complete folds |
| Purge / embargo inside WF | **none** (PCV is a later stage; not invoked) |
| Strategy / version / model | `default_strategy` / `v1` / `v1` |
| Target | Labels `future_return_1` (evaluation-only; not written into Factor Selection) |
| Orientation | inherited from Factor Selection (`signed_ic_v1`); never recomputed from OOS labels |
| Transaction costs / execution | not part of WF generation schema |
| Random seed | not used by `SimpleWalkForwardEngine` |
| Artifact schema | `WALK_FORWARD_SCHEMA` (`cqros.walk_forward.schema`) |

---

## 3. Inputs and outputs

**Reads (never mutated by this CLI):**

- Factor Selection: `data/factor_selection/default/binance/usdt_perpetual/{tf}/2026.parquet`
- Factors lake: `data/factors/default/binance/usdt_perpetual/{symbol}/{tf}/2026.parquet`
- Labels: `data/labels/binance/usdt_perpetual/{symbol}/{tf}/2026.parquet` (`future_return_1`)

**Does not read:** Factor Validation metrics, Purged CV, predictions, signals, alpha, regime, stability diagnostics.

**Writes (only these production files):**

- `data/walk_forward/default/binance/usdt_perpetual/{tf}/2026.parquet`

Static import audit: `cqros.cli.generate_walk_forward` and `cqros.walk_forward` do not import `cqros.purged_cv`, `cqros.alpha`, `cqros.regime`, predictions, or signals.

---

## 4. Execution order

Canonical discovery sort is `(manager, timeframe)` lexicographic, years ascending. An all-timeframe invocation would run:

`15m → 1d → 1h → 4h → 5m`

This regeneration uses existing `--timeframes` / `--years` filters to run **one partition at a time** in the controlled order:

`1d → 4h → 1h → 15m → 5m`

with `--workers 1`. If any timeframe fails, remaining timeframes are not started.

---

## 5. Canonical command (per timeframe)

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

`<TF>` ∈ `{1d, 4h, 1h, 15m, 5m}`. No production-code modification is required.

---

## 6. Factor Selection input verification

All five current Factor Selection partitions exist, are readable, match the post-FS-regeneration SHA-256 values, have 73 rows / 20 selected, 0 duplicate identities, 0 selected/status mismatches, timeframe column exact, orientation `signed_ic_v1`.

| TF | Bytes | SHA-256 | Selected | OK |
|----|------:|---------|----------:|:--:|
| 1d | 11,915 | `B6CE50C27CAE6601FC0CED1CB650475BF471C337211D5C89D98B1FEB8432A2BB` | 20 | yes |
| 4h | 11,827 | `BAE924AA39D5DDA300030052B5C69A55DA94DC230A30931ECFDE911BD1592918` | 20 | yes |
| 1h | 11,816 | `89F17D02C7A4B6314A785DC8783836F88B233F9F04A83AD76D94E3AFEEC10022` | 20 | yes |
| 15m | 11,813 | `898A40F41AF52452DB3D170C667DD6A2678CB7B9317C7A2D48CC7518DBE12788` | 20 | yes |
| 5m | 11,812 | `1C80C6679DAA8133CED64C9D7978F02E4E9742CEBEECECB4DA4289F569F97FB0` | 20 | yes |

These **must remain byte-for-byte unchanged** during this task.

---

## 7. Baseline hashes — Walk Forward (being replaced)

| TF | Bytes | SHA-256 | Rows | PASS | mtime UTC |
|----|------:|---------|-----:|-----:|-----------|
| 1d | 185,255 | `87E74C6C2AF48B78B6E3D55A97967DBEB101CB48343A1C78198E01F63D228087` | 70,961 | 70,961 | 2026-08-12T14:41:06Z |
| 4h | 141,704 | `888C4613FA4D3DD8D3345AF9F134D13DCAC2EF35641C073234EAB68E19C01C8F` | 53,721 | 53,721 | 2026-08-11T10:38:14Z |
| 1h | 470,948 | `735E0D62872F436A2BFA9A6A84E9EC5CBC4602703C5C7B43A4CA2462F2476373` | 169,950 | 169,950 | 2026-08-11T10:34:44Z |
| 15m | 226,656 | `97A8BA631707E8E74FBD13BC1577D5AE9EB9156A3F223B6D1D7F4D3882A73806` | 67,164 | 67,164 | 2026-08-11T10:27:42Z |
| 5m | 390,938 | `937C16B8A98BF5C0C7F95CEA80A9A8B2EAA6EE2828423298951057995730E22D` | 86,762 | 86,762 | 2026-08-11T10:24:40Z |

Schema: required columns present; no unexpected columns; 0 duplicate primary keys; `train_rows=252`, `test_rows=63`; `strategy_name=default_strategy`; `strategy_version=v1`; `model_version=v1`. Canonical parquet has **no factor-identity columns**; `selected_factors` is a per-fold observation count, not a Factor Selection membership list.

---

## 8. Baseline hashes — Factor Validation (must not change)

| TF | Bytes | SHA-256 |
|----|------:|---------|
| 1d | 13,281 | `B7935E021B31BAD5BE9017577FCD49243A1E022480A47F44A5B6D5D2C4058137` |
| 4h | 14,365 | `E49A86299CD989A8B1F5B91ABF90E92647993250A71351391EA2F9752F481EED` |
| 1h | 14,444 | `315EF539F46E94A7B28AB902D98E88D4B79EB7BD666687E5FCED6FF1B7C3CC30` |
| 15m | 14,425 | `57FEF604E3F24DA6FBEEC836D5ADCB2979452DAA906A5C60639DA89CCAC38CB2` |
| 5m | 14,455 | `7F81C68E92FD51058DA13F54CDBA3E8F2981357F69D7F4E75A7FFB9474DA213D` |

---

## 9. Baseline hashes — Purged CV (must not change)

| TF | Bytes | SHA-256 |
|----|------:|---------|
| 1d | 5,996 | `C517DC596ABE5F397FF16B0713CCD8781B0DF8B23569BA1C1934F357496A15FB` |
| 4h | 5,989 | `2185AB07C048E97C64155BE3004E7DB7C1FFA3101FA39FCA58DFBC6EE46DD1D4` |
| 1h | 5,984 | `249A8119215FA02CBEAFE2D7E946412CD7441F9E456339A5613EB59D702F0767` |
| 15m | 5,998 | `7D6EF28BD26F4A3CAE4E15A3CDBB3DEE5F89CD1625CACFDF61E14723ADE7AE4E` |
| 5m | 5,995 | `15F3745A9CEAFE9C62076B38641C72C752A199D7765B3C7E9EB59FC41665FA26` |

---

## 10. Observation lakes (read-only inputs)

| TF | Factors files | Factors bytes | Labels files | Labels bytes |
|----|--------------:|--------------:|-------------:|-------------:|
| 1d | 323 | 27,620,837 | 323 | 3,465,774 |
| 4h | 283 | 147,783,306 | 283 | 12,863,350 |
| 1h | 219 | 434,758,759 | 219 | 34,492,523 |
| 15m | 21 | 170,379,588 | 21 | 12,768,735 |
| 5m | 9 | 211,036,526 | 9 | 14,066,209 |

`1h` is the largest Factors panel on disk. There is no WF spill/memory-efficient path. If regeneration fails from memory, this task stops and reports; it does not patch the engine.

---

## 11. Source / lock hashes (not modified by this task)

| File | SHA-256 |
|------|---------|
| `pyproject.toml` | `5F270470DA80CB264C693221989C6775427FDA04E49F6B7E3E935FDAD0CC0963` |
| `uv.lock` | `EBA5805D5CBEBBC4B9F40A14C915D0EB405547393337F1FB4F54E28605CC783C` |
| `src/cqros/walk_forward/engine.py` | `D012617C96F75B712F7AD1D5D973FB8EEC979B0CE8B98AE9B6BC5C0F7F26898A` |
| `src/cqros/walk_forward/evaluation_input.py` | `2DD9D24E723A8A867C70404064965041F6BB4E296D281C74099C9704F8070060` |
| `src/cqros/walk_forward/pipeline.py` | `C680B885203F667A597DBABF12612D8E79AD540CC38862A5BE4A6FF6C0D4F761` |
| `src/cqros/walk_forward/schema.py` | `CC8E2B81C1223BA2CC9EF9C4D42F94279B518B513CFE5C25407029529C42175D` |
| `src/cqros/cli/generate_walk_forward.py` | `71DDAF731376B18758AA82BE0704159C7B606BE1584D6314662F9584E4A08030` |

Git: no commits required. `git status`/`rev-parse` failed with dubious-ownership (safe.directory not set; git config was not modified).

---

## 12. Tests (pre-execution)

Command:

```text
uv run pytest tests/unit/walk_forward/test_engine.py tests/unit/walk_forward/test_repository.py tests/unit/walk_forward/test_pipeline.py tests/unit/walk_forward/test_evaluation_input.py tests/unit/walk_forward/test_verifier.py tests/unit/walk_forward/test_schema.py tests/unit/walk_forward/test_registry.py tests/unit/cli/test_generate_walk_forward.py tests/unit/cli/test_verify_walk_forward.py -q --tb=short
```

Result: **160 passed** in 202.45s.

Not run (downstream / out of scope): `evaluate_walk_forward`, `generate_purged_cv`, `verify_purged_cv`, `evaluate_purged_cv`, stability diagnostics, `tests/unit/walk_forward/test_evaluation.py` (evaluation artifacts, not generation).

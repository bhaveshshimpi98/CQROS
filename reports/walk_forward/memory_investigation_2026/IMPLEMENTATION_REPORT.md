# Controlled Memory-Efficient Walk Forward Implementation — 2026

**Date:** 2026-08-14  
**Repository:** `D:/bss/CQROS`  
**Verdict:** PASS — controlled-fixture equivalence and bounded-retention
behavior demonstrated; production regeneration was not run.

## 1. Implementation verdict

The opt-in `memory_efficient` execution mode is implemented. The unchanged
`full_panel` path remains the default reference implementation. Controlled
multi-symbol, multi-factor, repeated-timestamp fixtures produced exact frame
equality and byte-identical Parquet output.

## 2. Files changed

- `SRC/cqros/walk_forward/memory_efficient.py` (new)
- `SRC/cqros/walk_forward/evaluation_input.py`
- `SRC/cqros/walk_forward/engine.py`
- `SRC/cqros/walk_forward/__init__.py`
- `SRC/cqros/cli/generate_walk_forward.py`
- `tests/unit/walk_forward/test_memory_efficient.py` (new)
- `tests/unit/cli/test_generate_walk_forward.py`
- This report

No Factor Selection, Factor Validation, Purged CV, Factors, Labels, features,
market-data, policy, scoring, or downstream evaluation implementation changed.

## 3. Architecture implemented

The implementation follows the investigation design:

1. Load and project the partition's validated Factor Selection decisions.
2. Discover the sorted Factors/Labels symbol intersection.
3. Read projected Factors and Labels columns for one symbol.
4. Apply canonical duplicate checks and both canonical inner joins.
5. Write a sorted immutable symbol shard.
6. Perform a bounded deterministic k-way merge over symbol shards.
7. Assign row ordinals after canonical ordering.
8. Process configured row-index folds from bounded batches.
9. Spill raw fold rows.
10. Apply canonical PASS-only aggregate formulas in a second pass.
11. Return canonical schema/order for existing repository validation and
    atomic persistence.
12. Remove the unique run directory in `finally`.

No full Factors lake, Labels lake, or joined evaluation panel is accumulated.

## 4. CLI/API changes

Added CLI options:

- `--execution-mode {full_panel,memory_efficient}`
- `--spill-parent PATH`
- `--memory-budget-mb INT` (default `256`)

`full_panel` remains the default. `memory_efficient` requires `--engine simple`
and `--workers 1`; unsupported combinations fail with
`CLI-GENERATE-WALK-FORWARD-008`.

The execution mode is included in structured logging and the generation
summary. Public APIs expose `MemoryEfficientExecutionConfig`,
`MemoryEfficientWalkForwardExecutor`, execution-mode constants, and the exact
frame-equivalence assertion.

## 5. Canonical path preservation

The original `WalkForwardInputBuilder` full-panel loading path, pipeline,
registry dispatch, formulas, repository, schema, and default CLI behavior
remain active under `full_panel`. Memory-efficient dispatch is a separate
explicit branch and never falls back to full-panel execution.

## 6. Spill schema

Symbol shards contain exactly `WALK_FORWARD_EVALUATION_COLUMNS`:

`symbol, timeframe, open_time, factor_name, factor_version, factor_value,
selected, selection_time, selection_ic, selected_direction,
orientation_policy, future_return_1`.

Merged-run parts add `row_ordinal: UInt64`. Raw-fold parts use the canonical
16-column Walk Forward schema.

## 7. Spill partitioning

Every partition execution creates a unique directory beneath the configured
spill parent. It contains:

- `symbol_shards/{symbol-order-index}.parquet`
- `sorted_ordinal_run/part-{part-index}.parquet`
- `raw_folds/part-{part-index}.parquet`

Names and part order are deterministic within the run. The UUID only isolates
concurrent/stale runs and is not part of logical ordering or artifact hashing.

## 8. Deterministic ordering

Symbols are sorted explicitly. Each shard and the k-way merge use:

`(timeframe, selection_time, symbol, factor_name, factor_version)`.

Part paths are retained in ordered tuples; filesystem enumeration order is
never used. Batch collection uses `maintain_order=True`, and CLI mode enforces
one worker.

## 9. Row ordinal strategy

`row_ordinal` is assigned sequentially only after the strict canonical merge
key has ordered every admitted row. It is temporary implementation metadata,
does not replace timestamp/symbol/factor identity, and is absent from canonical
output.

## 10. Fold processing strategy

The executor reads merged parts sequentially and retains at most the configured
`train_window + test_window` row window. Fold starts advance by the engine's
configured `step_size`; no calendar or timestamp window substitutes for row
geometry. The canonical engine fold primitive computes boundaries, selected
training observations, test metrics, status, and constants.

## 11. Two-pass aggregation strategy

Pass 1 writes raw per-fold train mean, test Sharpe, win rate, status, geometry,
and identity fields in fold order. Pass 2 reads raw folds in explicit part
order and applies the same Polars PASS-only mean/sample-standard-deviation and
stability formulas as the canonical engine. No sampled or running-moment
approximation is used.

## 12. Failure atomicity

The memory executor writes temporary spill only and returns a fully validated
canonical frame. Production persistence remains through
`WalkForwardRepository` and `ParquetStore`, whose sibling temporary write is
atomically promoted with `os.replace`. A controlled memory-path failure test
confirmed an existing output remained byte-for-byte unchanged.

## 13. Cleanup behavior

The unique run directory is removed in `finally`, covering success,
validation errors, ordinary exceptions, and `KeyboardInterrupt`. Cleanup is
scoped to that run directory. Success and duplicate-key failure cleanup were
tested.

## 14. Memory-bound mechanism

Retained input state is bounded by:

- one projected Factors frame, one projected Labels frame, and one symbol join;
- one configured batch per symbol-shard merge cursor;
- an 8,192-row merged-output buffer;
- one `train_window + test_window` fold deque;
- an 8,192-row raw-fold buffer; and
- the final canonical fold ledger required by the repository API.

The default cursor budget is 256 MiB. A deterministic synthetic test runs
multiple symbols under a 1 MiB configured cursor budget, verifies exact
equivalence, and repeats execution deterministically. No OS-specific private
memory threshold is asserted in unit tests.

## 15. Tests added/changed

Coverage includes:

- exact canonical/memory-efficient frame equality;
- schema, dtype, row, timestamp, symbol, factor, fold, boundary, strategy,
  model, score, and aggregate equality through exact frame comparison;
- byte-identical Parquet SHA-256;
- repeated-timestamp tie ordering;
- cross-symbol and cross-factor behavior;
- selected and rejected factor row geometry;
- timestamp alignment;
- deterministic repeated execution;
- success and failure spill cleanup;
- structurally bounded symbol/batch retention;
- CLI default and opt-in mode parsing;
- unsupported worker combination rejection;
- no silent full-panel fallback; and
- existing-output preservation after controlled failure.

Existing engine, evaluation-input, pipeline, repository, verifier, schema,
registry, and CLI suites were retained unchanged except for additive CLI tests.

## 16. Test results

Relevant suite command:

```text
uv run pytest tests/unit/walk_forward/test_engine.py
  tests/unit/walk_forward/test_evaluation_input.py
  tests/unit/walk_forward/test_memory_efficient.py
  tests/unit/walk_forward/test_pipeline.py
  tests/unit/walk_forward/test_repository.py
  tests/unit/walk_forward/test_verifier.py
  tests/unit/walk_forward/test_schema.py
  tests/unit/walk_forward/test_registry.py
  tests/unit/cli/test_generate_walk_forward.py -q --tb=short
```

Result: **150 passed, 0 failed**, pytest duration **168.24 s**.

Final changed-test recheck:

```text
uv run pytest tests/unit/walk_forward/test_memory_efficient.py
  tests/unit/cli/test_generate_walk_forward.py -q --tb=short
```

Result: **25 passed, 0 failed**, duration **39.00 s**.

Quality gates:

- Ruff: all checks passed.
- Black: 7 files unchanged under `--check`.
- Pyright: 0 errors, 0 warnings.

No production generation, evaluation, Purged CV, stability, prediction, or
signal command was run.

## 17. Exact-equivalence results

Controlled fixtures used three symbols, two factors (one selected and one
rejected), repeated timestamps, tied cross-sections, mixed returns, and
multiple folds. `assert_frame_equal(..., check_exact=True)` passed for the
complete canonical output. Writing both outputs through the same
`ParquetStore` produced equal SHA-256 digests. No floating-point tolerance was
introduced.

Production-partition equivalence and observed private-memory measurement are
reserved for the separately authorized controlled regeneration task.

## 18. Protected artifact hashes before/after

Read-only SHA-256 capture before implementation and verification after all
tests both reported `all_unchanged: true`.

### Factor Selection — UNCHANGED × 5

- 1d `B6CE50C27CAE6601FC0CED1CB650475BF471C337211D5C89D98B1FEB8432A2BB`
- 4h `BAE924AA39D5DDA300030052B5C69A55DA94DC230A30931ECFDE911BD1592918`
- 1h `89F17D02C7A4B6314A785DC8783836F88B233F9F04A83AD76D94E3AFEEC10022`
- 15m `898A40F41AF52452DB3D170C667DD6A2678CB7B9317C7A2D48CC7518DBE12788`
- 5m `1C80C6679DAA8133CED64C9D7978F02E4E9742CEBEECECB4DA4289F569F97FB0`

### Factor Validation — UNCHANGED × 5

- 1d `B7935E021B31BAD5BE9017577FCD49243A1E022480A47F44A5B6D5D2C4058137`
- 4h `E49A86299CD989A8B1F5B91ABF90E92647993250A71351391EA2F9752F481EED`
- 1h `315EF539F46E94A7B28AB902D98E88D4B79EB7BD666687E5FCED6FF1B7C3CC30`
- 15m `57FEF604E3F24DA6FBEEC836D5ADCB2979452DAA906A5C60639DA89CCAC38CB2`
- 5m `7F81C68E92FD51058DA13F54CDBA3E8F2981357F69D7F4E75A7FFB9474DA213D`

### Purged CV — UNCHANGED × 5

- 1d `C517DC596ABE5F397FF16B0713CCD8781B0DF8B23569BA1C1934F357496A15FB`
- 4h `2185AB07C048E97C64155BE3004E7DB7C1FFA3101FA39FCA58DFBC6EE46DD1D4`
- 1h `249A8119215FA02CBEAFE2D7E946412CD7441F9E456339A5613EB59D702F0767`
- 15m `7D6EF28BD26F4A3CAE4E15A3CDBB3DEE5F89CD1625CACFDF61E14723ADE7AE4E`
- 5m `15F3745A9CEAFE9C62076B38641C72C752A199D7765B3C7E9EB59FC41665FA26`

### Walk Forward — UNCHANGED × 5

- 1d `B2391EE4ECBCD89E145015A488789415C410F6A5C050A818192EAEB7DFE58469`
- 4h `888C4613FA4D3DD8D3345AF9F134D13DCAC2EF35641C073234EAB68E19C01C8F`
- 1h `735E0D62872F436A2BFA9A6A84E9EC5CBC4602703C5C7B43A4CA2462F2476373`
- 15m `97A8BA631707E8E74FBD13BC1577D5AE9EB9156A3F223B6D1D7F4D3882A73806`
- 5m `937C16B8A98BF5C0C7F95CEA80A9A8B2EAA6EE2828423298951057995730E22D`

## 19. Known limitations

- Per the task prohibition, the 0.5–1.0 GiB target was not measured against a
  production partition in this implementation task.
- Production 1d/4h/1h/15m/5m canonical-vs-memory-efficient output comparison
  was not run; only controlled fixtures were executed.
- Final output is materialized because the existing repository contract accepts
  a canonical `pl.DataFrame`. Input-panel retention and fold working state are
  bounded; final-ledger size remains proportional to fold count.
- Run-scoped spill cleanup is best effort if the operating system terminates
  the process without allowing Python `finally` execution.

## 20. Explicit STOP

STOP.

Implementation, controlled equivalence tests, bounded-retention tests, quality
checks, cleanup/failure tests, and protected-artifact verification are
complete. No production Walk Forward artifact was regenerated or overwritten.
No Purged CV, evaluation, stability diagnostics, prediction, or signal
generation ran.

The next task requires separate explicit approval:
**CONTROLLED PRODUCTION WALK FORWARD REGENERATION — 2026** using
`--execution-mode memory_efficient --workers 1`.

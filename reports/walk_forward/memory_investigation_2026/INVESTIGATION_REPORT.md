# Controlled Walk Forward Memory Investigation — 2026

**Date:** 2026-08-14  
**Repository:** `D:/bss/CQROS`  
**Scope:** Read-only engineering and memory investigation. No production source,
policy, configuration, or artifact was modified. Walk Forward, Purged CV,
evaluation, stability, prediction, signal, alpha, and regime pipelines were not run.

## 1. Executive Summary

The `4h` failure is a retained eager-load failure in
`WalkForwardInputBuilder.build`, not a concatenation, join, sort, fold, model,
or repository-save failure. The builder eagerly reads every canonical Factors
column and every Labels column for each symbol and retains the resulting
DataFrames in `factors_parts` and `labels_parts`. At the protection stop, 248
complete symbol pairs were retained. The next code operation would have been
the 249th Factors load; the asynchronous process kill prevents identification
of a more precise machine instruction.

Measured at that point:

- 4h Factors loaded: 22,859,877 rows, 129,498,626 compressed bytes,
  approximately 1,715,388,757 Polars `estimated_size` bytes.
- 4h Labels loaded: 306,731 rows, 11,269,426 compressed bytes,
  approximately 16,557,066 `estimated_size` bytes.
- Combined logical column-buffer estimate: approximately 1.732 GB before
  allocator reservation, decode/cast transients, runtime memory, concatenation,
  or joins.
- Observed process private memory: 3,464.47 MB.

Factors are the dominant object set. Labels are a small contributor. No
selected-factor pre-filter exists. No timestamp conversion occurs. Neither
`pl.concat`, either join, evaluation-input construction, fold construction, nor
save was reached for 4h.

A bounded-memory implementation is theoretically possible without changing
canonical research semantics, but exact equivalence has two implementation
gates: equal-timestamp row ordering must be locked and tested, and final
floating-point/Parquet byte equivalence must be demonstrated against the
legacy path rather than assumed.

**Investigation verdict: PASS WITH CONSTRAINTS.**

## 2. Exact failure location

The exact code stage is the symbol-load loop in
`src/cqros/walk_forward/evaluation_input.py`,
`WalkForwardInputBuilder.build`, lines 226–247.

The last completed logged operation was:

1. `LabelRepository.load` for symbol 248.
2. `ParquetStore.read`.
3. `pl.read_parquet`.
4. Return and append to `labels_parts`.

The next intended operation was `FactorsRepository.load` for symbol 249. The
wrapper terminated the process asynchronously when private memory exceeded its
threshold, so it is not defensible to claim that a particular instruction
inside the 249th read had begun.

Evidence:

- 248 `Loaded factors dataset` records.
- 248 `Loaded label dataset` records.
- 497 `Read Parquet dataset` records: one Factor Selection plus 248 pairs.
- Zero `Assembling walk-forward evaluation input` records.
- No output write.

The operation corresponding to the observed point is therefore:

```text
WalkForwardInputBuilder.build
  → for symbol in panel_symbols
    → FactorsRepository.load
    → append full eager Factors frame to factors_parts
    → LabelRepository.load
    → append full eager Labels frame to labels_parts
```

## 3. Complete Walk Forward dataflow

```text
cqros.cli.generate_walk_forward.main
  → build_options
  → StorageLayout / ParquetStore composition
  → FactorSelectionRepository(...)
  → WalkForwardRepository(...)
  → WalkForwardInputBuilder(FactorsRepository(...), LabelRepository(...))
  → build_walk_forward_pipeline
    → build_registry
    → build_default_engine
      → SimpleWalkForwardEngine
    → WalkForwardPipeline
  → discover_work
    → FactorSelectionRepository.discover_partitions
    → _group_partitions
  → run_generation
    → _group_work_by_timeframe
    → _run_worker_pool
      → _generate_timeframe_work
        → asyncio.to_thread(_generate_partition)
          → WalkForwardRepository.exists
          → FactorSelectionRepository.exists
          → FactorSelectionRepository.load
            → ParquetStore.read
              → pl.read_parquet
            → _require_factor_selection_schema
              → select
              → cast
          → WalkForwardInputBuilder.build
            → _resolve_panel_symbols
              → _discover_factor_symbols
                → FactorsRepository.discover_partitions
              → FactorsRepository.exists
              → LabelRepository.exists
            → for each sorted symbol
              → FactorsRepository.load
                → ParquetStore.read
                  → pl.read_parquet
                → _require_factor_schema
                  → select
                  → cast
              → LabelRepository.load
                → ParquetStore.read
                  → pl.read_parquet
            → pl.concat(factors_parts, how="vertical")
            → pl.concat(labels_parts, how="vertical")
            → assemble_walk_forward_input
              → _require_columns / require_orientation_metadata
              → _require_unique_keys(labels)
              → _require_unique_keys(factors)
              → _selection_decision_frame
              → _require_unique_keys(selection)
              → labels.select
              → factors.join(labels, how="inner")
              → observations.join(selection, how="inner")
              → open_time.alias("selection_time")
              → select(WALK_FORWARD_EVALUATION_COLUMNS)
              → sort(timeframe, selection_time, symbol,
                     factor_name, factor_version)
          → WalkForwardPipeline.run
            → WalkForwardEngineRegistry.get
            → validate_factor_selection_frame
            → SimpleWalkForwardEngine.build
              → _build_walk_forward_rows
                → with_columns(strategy constants)
                → partition_by(strategy_name, strategy_version, timeframe)
                → sort(selection_time)
                → _folds_for_group
                  → _fold_row
                    → slice train/test rows
                    → _evaluate_test_window
                      → filter(selected)
                      → drop_nulls(future_return_1)
                      → mean / sample std / wins
                  → _apply_aggregate_metrics
                    → _compute_aggregate_metrics
              → pl.DataFrame(fold_rows)
              → select(CANONICAL_COLUMN_ORDER)
              → cast(WALK_FORWARD_SCHEMA)
            → _finalize
              → primary-key n_unique
              → select
              → cast
          → WalkForwardRepository.save
            → _require_walk_forward_schema
              → select
              → cast
            → ParquetStore.write
              → DataFrame.write_parquet(temp)
              → os.replace(temp, canonical path)
```

The 4h process stopped in the per-symbol load loop. Everything after that loop
in the graph was not reached.

## 4. Complete materialization inventory

The following inventory is limited to the production generation path. The
separate downstream `walk_forward/evaluation.py` module is not called by
`generate_walk_forward` and was not run.

| Operation | Location | 4h input/result | Full materialization and implication |
|---|---|---:|---|
| `pl.read_parquet` | `ParquetStore.read` | FS 73 rows | Eagerly reads the complete FS file. |
| `select().cast()` | `FactorSelectionRepository._require_factor_selection_schema` | 13 canonical plus 12 eligibility columns | Creates a validated/cast frame; small here. |
| `pl.read_parquet` | each `FactorsRepository.load` | 26,077,936 rows × 12 columns total | Reads every column in every symbol file; all frames retained. Dominant failure source. |
| `select().cast()` | `_require_factor_schema` | 12 columns per symbol | Can transiently coexist with the just-read frame; validated result remains retained. |
| `pl.read_parquet` | each `LabelRepository.load` | 349,898 rows × 11 columns total | Reads all horizons/directions although only four columns are used. |
| `pl.concat` | `WalkForwardInputBuilder.build` | 26,077,936 × 12 Factors; 349,898 × 11 Labels | Eager DataFrames. Not reached for 4h. May retain many chunks even where buffers are reused. |
| `select(keys).n_unique()` | `_require_unique_keys` | all Factors/Labels rows | Builds key projections and hash state. Not reached. |
| `labels.select(4 cols)` | `assemble_walk_forward_input` | 349,898 × 4 | Late projection; all 11 label columns have already been loaded and concatenated. |
| inner `join` | Factors × Labels | 25,542,554 rows, approximately 13 columns | Materializes matched long observations. Not reached. |
| inner `join` | observations × FS | 25,542,554 rows, approximately 17 columns before final projection | Broadcasts static decisions; does not filter to selected rows. Not reached. |
| `with_columns/select/sort` | `assemble_walk_forward_input` | 25,542,554 × 12 final evaluation columns | Full global sorted evaluation frame. Not reached. |
| `with_columns` | `_build_walk_forward_rows` | evaluation frame + two constant string columns | Eager annotated frame. |
| `partition_by` | `_build_walk_forward_rows` | one strategy/timeframe group in this CLI | Materializes a partition list. |
| `sort(selection_time)` | `_build_walk_forward_rows` | entire partition | Full-frame ordering before row-index folds. |
| `slice/filter/drop_nulls` | `_fold_row`, `_evaluate_test_window` | 252 train and 63 test rows per complete fold | Per-fold views/filtered Series; repeated over overlapping windows. |
| Python `fold_rows` list | `_build_walk_forward_rows` | current 4h estimate: 405,433 dicts | Large later-stage Python object set, but not the observed failure. |
| `pl.DataFrame(fold_rows)` | `_build_walk_forward_rows` | current 4h estimate: 405,433 × 16 | Materializes output ledger. |
| `pl.DataFrame(...).filter()` | `_compute_aggregate_metrics` | one row per fold, four columns | Global PASS-fold aggregation frame. |
| copied dict list | `_apply_aggregate_metrics` | one new dict per fold | Temporarily duplicates fold dictionaries. |
| `select().cast()` | pipeline/repository finalization | output × 16 | Final eager frames before write. |
| `write_parquet` | `ParquetStore.write` | output × 16 | Serializes to a temporary sibling then atomically replaces output. |

There is no `collect`, streaming `collect`, `scan_parquet`, `pivot`,
`to_numpy`, `to_pandas`, `rows`, or `iter_rows` in the reachable production
generation path.

Filtering inventory:

- There is no selected-factor filtering before panel construction.
- Inner joins remove unmatched observation/factor identities.
- Null `future_return_1` rows remain in the evaluation frame and are removed
  only from each selected test-return Series.
- Timestamp “normalization” is only `open_time` copied to `selection_time`;
  there is no timezone or unit conversion.

## 5. Actual required columns

### A. Factor Selection

| Column | Required by | Reason |
|---|---|---|
| `factor_name` | selection join | Factor identity. |
| `factor_version` | selection join | Factor-version identity. |
| `timeframe` | selection join and engine grouping | Partition/factor identity. |
| `selected` | engine | Train selected-observation count and test-return filter. |
| `selection_ic` | orientation audit/evaluation frame | Retained for audit; not used by `SimpleWalkForwardEngine` scoring. |
| `selected_direction` | orientation validation/evaluation frame | Must be ±1; not applied to returns by the current engine. |
| `orientation_policy` | orientation validation/evaluation frame | Must equal the canonical policy. |

The FS artifact’s own `selection_time`, category, score, rank, reason, status,
and eligibility metadata are not consumed by Walk Forward after repository
validation. Evaluation `selection_time` is derived from Factors `open_time`.

### B. Factors observations

| Column | Required by | Reason |
|---|---|---|
| `symbol` | label join and deterministic order | Observation identity and cross-symbol tie order. |
| `timeframe` | both joins/order | Observation/selection identity. |
| `open_time` | label join/order | Observation timestamp; becomes `selection_time`. |
| `factor_name` | uniqueness, FS join, order | Factor identity and deterministic tie order. |
| `factor_version` | FS join/order | Factor-version identity. |
| `factor_value` | current adapter contract/audit frame | Required and retained by the adapter, but never read by `SimpleWalkForwardEngine`. |

No factor computation occurs in Walk Forward. `factor_category`,
`factor_group`, `lookback`, `prediction_horizon`, `enabled`, and `status` can
be projected away after schema validation.

### C. Labels

| Column | Required by | Reason |
|---|---|---|
| `symbol` | observation join | Identity. |
| `timeframe` | observation join | Identity. |
| `open_time` | observation join | Identity/alignment. |
| `future_return_1` | test evaluation | Mean, sample volatility, Sharpe, and win rate. |

`future_return_5/10/20` and all `direction_*` columns are unused.

### D. Static configuration

| Value | Required by | Reason |
|---|---|---|
| manager/exchange/market/timeframe/year | repositories and paths | Exact input/output partition identity. |
| engine name | registry | Resolves `SimpleWalkForwardEngine`. |
| train/test/step = 252/63/63 | engine | Canonical row-index fold geometry. |
| strategy name/version = `default_strategy`/`v1` | engine/output | Canonical group and primary key. |
| model version = `v1` | output | Canonical lineage field. |
| storage root | layout | Canonical data root. |
| workers/overwrite | CLI | Scheduling/write policy; not score mathematics. |

## 6. Factors data-size analysis

`estimated_size` is Polars logical buffer size, not peak process memory. It
excludes allocator reservations, decode buffers, duplicate cast/join/sort
frames, and Python objects.

| TF | Symbols/files | Rows | Columns | Disk bytes | Full eager estimate | 6-column projection estimate |
|---|---:|---:|---:|---:|---:|---:|
| 1d | 323 | 4,963,416 | 12 | 27,620,837 | 372,515,647 | 241,894,493 |
| 4h | 283 | 26,077,936 | 12 | 147,783,306 | 1,955,130,544 | 1,268,843,056 |
| 1h | 219 | 78,631,074 | 12 | 434,758,759 | 5,890,967,914 | 3,821,651,009 |
| 15m | 21 | 31,750,474 | 12 | 170,379,588 | 2,420,601,140 | 1,585,030,852 |
| 5m | 9 | 40,820,140 | 12 | 211,036,526 | 3,040,811,050 | 1,966,556,210 |

The 4h full eager estimate is 13.2 times its compressed disk size. The 1h
full eager estimate is approximately 5.89 GB before joins and therefore cannot
complete through the current path on this host.

## 7. Labels data-size analysis

| TF | Symbols/files | Rows | Columns | Disk bytes | Full eager estimate | 4-column projection estimate |
|---|---:|---:|---:|---:|---:|---:|
| 1d | 323 | 61,244 | 11 | 3,465,774 | 3,306,959 | 1,592,127 |
| 4h | 283 | 349,898 | 11 | 12,863,350 | 18,863,934 | 9,066,790 |
| 1h | 219 | 1,065,202 | 11 | 34,492,523 | 57,371,946 | 27,546,290 |
| 15m | 21 | 434,518 | 11 | 12,768,735 | 23,981,250 | 11,814,746 |
| 5m | 9 | 559,000 | 11 | 14,066,209 | 29,875,444 | 14,223,444 |

Labels are not the primary failure source. Projection still removes roughly
half their logical size and should occur before materialization.

Bounded per-symbol join profiling produced these current evaluation
cardinalities:

| TF | Factor–Label–Selection rows | Selected rows | Four engine columns estimate | Expected complete folds |
|---|---:|---:|---:|---:|
| 1d | 4,470,812 | 1,224,880 | 81,033,653 | 70,961 |
| 4h | 25,542,554 | 6,997,960 | 462,958,918 | 405,433 |
| 1h | 77,759,746 | 21,304,040 | 1,409,395,451 | 1,234,277 |
| 15m | 31,719,814 | 8,690,360 | 606,641,455 | 503,485 |
| 5m | 40,807,000 | 11,180,000 | 739,626,877 | 647,726 |

The existing 4h/1h/15m/5m Walk Forward files are older baselines and their row
counts do not describe the output cardinality expected from current inputs.

## 8. Memory-growth analysis

The 4h growth chain is:

1. ZSTD-compressed Parquet is decoded into strings, offsets, validity bitmaps,
   integers, booleans, and floating buffers.
2. `FactorsRepository.load` reads all 12 columns.
3. `_require_factor_schema` selects and casts all 12 columns. The read frame,
   validated frame, decode buffers, and allocator pages may coexist
   transiently.
4. The validated frame is appended to `factors_parts` and cannot be released.
5. Labels repeat the process and are appended to `labels_parts`.
6. Polars/native allocators retain or reserve pages; process private bytes need
   not return to the OS between symbol loads.

At 248 symbols, retained Factors account for approximately 1.715 GB logical
buffers versus 16.6 MB for Labels. The approximately 1.73 GB logical set plus
decode/cast transients, chunk metadata, allocator reservation, Python/runtime
memory, and fragmentation explains the observed approximately 3.46 GB private
commit. This is a measured materialization chain, not a generic “Polars
overhead” assertion.

## 9. Largest memory-consuming stage/object

The largest observed stage is the aggregate `factors_parts` retained object
set in `WalkForwardInputBuilder.build`.

- At failure: approximately 1.715 GB logical Factors buffers.
- Largest individual measured 4h Factors partition: approximately 7.76 MB
  logical, so no single symbol file caused the failure.
- Labels at the same point: approximately 16.6 MB logical.
- Concatenated Factors, joined observations, sorted evaluation input, and
  fold-output objects had not yet been created.

For a successful larger partition, the later full global sort and engine
partition would also be major objects. They are prospective risks, not the
cause of this 4h stop.

## 10. Partition independence analysis

Each `(timeframe, year)` is independently loaded, evaluated, and saved.

- No cross-timeframe read exists.
- No cross-year read exists.
- Folds reset at each year boundary; prior-year rows are not carried in.
- Multiple years for a timeframe are processed sequentially by
  `_generate_timeframe_work`.
- Engine objects are stateless apart from immutable window configuration.
- Output primary keys and files are partition-local.

Processing one `(timeframe, year)` at a time preserves current semantics.

## 11. Fold dependency analysis

Folds are not fully independent:

- Fold boundaries depend on the globally ordered row sequence.
- Training windows overlap because the step is 63 and training length is 252.
- Test windows are adjacent under the default step/test sizes.
- Every PASS fold contributes to global means and sample standard deviation.
- Aggregate values overwrite score fields on every emitted fold.

Raw fold metrics can be computed one fold at a time, but final rows require a
second pass after group aggregates are known.

## 12. Cross-symbol dependency analysis

Factors–Labels joins and duplicate validation can be performed symbol-wise
because `symbol` is part of every observation key. Fold evaluation cannot be
performed independently by symbol:

- The canonical frame is globally ordered by timestamp, symbol, and factor.
- A 252/63 row window can cross symbols.
- Selected test returns are pooled across every symbol represented in its 63
  rows.

Symbol-wise loading is therefore valid only as a bounded assembly stage
followed by a deterministic global merge.

## 13. Cross-factor dependency analysis

Factor identity joins can be done per row/factor identity, but fold evaluation
cannot be split into independent factor runs:

- Both selected and rejected factor-observation rows contribute to row
  positions and fold boundaries.
- `selected_factors` is a selected-observation count, not a unique factor
  count.
- Selected returns from multiple factors are pooled in each test slice.

Filtering to the 20 selected factors before fold creation would change row
count, boundaries, timestamps, train/test membership, and scores.

## 14. Exact semantic constraints

A future path must preserve exactly:

1. The same authoritative FS file and validated orientation metadata.
2. Inner joins and duplicate rejection on the current key sets.
3. Null-target behavior.
4. The strict assembly order
   `(timeframe, selection_time, symbol, factor_name, factor_version)`.
5. The engine’s subsequent `selection_time` sort behavior.
6. Long-row, not unique-timestamp, 252/63/63 folds.
7. Train selected-observation counts.
8. Test selected-return sequence and Polars mean/std reduction order.
9. PASS/FAIL rules and PASS-only aggregate overwrite.
10. Schema, dtypes, row order, constants, and repository write behavior.

The assembly sort key is effectively unique after factor and label uniqueness
checks. The remaining risk is the engine’s sort on `selection_time` alone:
many rows share a timestamp and the code does not explicitly request stable
tie handling. A future equivalence test must lock the observed tie order.

## 15. Design A analysis — Projection before materialization

Feasible and necessary, but insufficient alone.

- Factors six-column projection saves approximately 35.1% for 4h
  (1.955 GB to 1.269 GB logical).
- Labels four-column projection saves approximately 51.9% for 4h
  (18.9 MB to 9.1 MB logical).
- Projection currently occurs only after full eager repository loads.

The future loader should validate the stored schema first and project during
the Parquet read. For strict current adapter behavior, `factor_value` should be
read/validated in the per-symbol stage even though the engine does not use it.
Projection alone still leaves concat/join/sort peaks and is not a complete fix.

## 16. Design B analysis — Symbol-wise processing

Feasible for loading, schema validation, duplicate checks, joins, and spill.
Not feasible for independent symbol evaluation.

Exact design:

1. Process symbols in the same sorted discovery order.
2. Read one Factors and one Labels partition.
3. Apply current schema checks.
4. Check symbol-local observation uniqueness.
5. Apply both current inner joins.
6. Write an immutable symbol shard.
7. Release all symbol frames.
8. Globally merge shards by the canonical sort key before fold assignment.

No score may be emitted per symbol.

## 17. Design C analysis — Timestamp-window processing

Calendar/timestamp windows are not equivalent because current folds are based
on row indices and can split a timestamp’s cross-section.

Incremental processing is feasible only after converting the globally sorted
stream to canonical row ordinals. A reader can maintain the 315 rows needed
for one complete train/test window and advance by 63 rows. This reads each
spill row once; overlapping windows require retained carryover, not repeated
source reads. A timestamp boundary must never replace a row boundary.

## 18. Design D analysis — Factor-wise processing

Independent factor evaluation is not equivalent. It changes global row
positions and pooled test statistics. Factor identity may be used to organize
temporary input, but all factors—selected and rejected—must be merged into the
single canonical row stream before folds.

## 19. Design E analysis — Lazy join

Lazy projection and per-symbol lazy joins can reduce decode and intermediate
memory. Laziness alone does not solve:

- the global sort,
- row-index fold generation,
- global fold aggregates, or
- final eager repository API.

A single streaming `collect` is not an equivalence proof and is not guaranteed
to bound a global sort. Lazy operations should feed explicit sorted spill
runs, not defer one full-panel materialization to the end.

## 20. Design F analysis — Spill-to-disk

Feasible and the recommended core.

Proposed first-stage spill schema is exactly
`WALK_FORWARD_EVALUATION_COLUMNS`:

```text
symbol, timeframe, open_time, factor_name, factor_version, factor_value,
selected, selection_time, selection_ic, selected_direction,
orientation_policy, future_return_1
```

Partition key:

```text
temporary-run/{manager}/{exchange}/{market}/{timeframe}/{year}/symbol
```

Sort order:

```text
timeframe, selection_time, symbol, factor_name, factor_version
```

After an external merge assigns a canonical row ordinal, fold processing needs
only `timeframe`, `selection_time`, `selected`, and `future_return_1`; identity
columns remain available in spill for audit and ordering.

Duplicate semantics:

- Labels uniqueness checked per symbol is globally sufficient because symbol
  is in the key.
- Factors uniqueness checked per symbol is globally sufficient for the same
  reason.
- FS uniqueness remains a complete small-panel check.
- No `.unique()` or deduplication may replace current rejection behavior.

Spill files are temporary, versioned by run identity, never canonical
artifacts, and removed after success or best-effort on failure. Canonical
output remains untouched until the existing atomic save succeeds.

## 21. Design G analysis — Chunked model input

There is no fitted model matrix in `SimpleWalkForwardEngine`; the effective
input is a row window containing `selected` and `future_return_1`.

Per-fold construction is feasible:

1. Read canonical row ranges from sorted spill.
2. Compute `selected_factors` from the exact 252-row train slice.
3. Compute test metrics with the same Polars operations on the exact 63-row
   test slice.
4. Spill raw fold rows in fold order.
5. Build the same four-column PASS aggregation frame, preserving fold order.
6. Apply aggregate values in a second sequential pass.

This preserves existing aggregate calculations. Computing running
floating-point moments with a different algorithm is not acceptable because
it can change bits.

## 22. Recommended bounded-memory architecture

Execution mode:

- Add explicit `full_panel` and `memory_efficient` modes.
- Keep `full_panel` as the reference implementation.
- Do not silently switch modes.

Memory bound:

- Configured external-sort budget, proposed default 256 MB.
- One Factors symbol partition, one Labels symbol partition, one join result,
  and one spill write at a time.
- One 315-row fold ring plus bounded fold-output buffers.
- Target process peak: no more than 1.0 GiB on the current data; enforce with
  stress tests rather than policy assumptions.

Pipeline:

```text
FS load/validate once
  → sorted symbol discovery
  → per-symbol eager schema validation + projected inner joins
  → immutable per-symbol Parquet shards
  → bounded external merge by canonical five-column key
  → fold-ready Parquet runs with canonical row ordinal
  → sequential 252/63/63 fold pass
  → raw fold spill
  → identical Polars PASS-fold aggregate reduction
  → second pass applying aggregate fields
  → canonical DataFrame select/cast/rechunk
  → existing WalkForwardRepository atomic save
  → cleanup temporary run
```

Failure behavior:

- Fail immediately on missing columns, invalid orientation, duplicate keys,
  empty joins, sort/order violations, or spill I/O errors.
- Never fall back to altered semantics.
- Never partially replace the canonical output.
- Preserve spill evidence on an explicitly configured diagnostic failure;
  otherwise clean it best-effort.

## 23. Why the architecture is semantically equivalent

The architecture changes physical execution only:

- The same rows are admitted by the same two inner joins.
- Per-symbol joins are distributive because the observation join key includes
  symbol.
- The external merge reconstructs the same strict total assembly order.
- Row ordinals reproduce the same fold start/end indices.
- Every fold uses the same train/test row sequence.
- Per-fold metrics use the same Polars operations.
- Global metrics use the same ordered fold columns and formulas.
- Final schema/order/cast and repository writer remain unchanged.

Equivalence is conditional on passing the tie-order and byte-hash tests in
section 28.

## 24. What cannot be streamed

The following require bounded materialization or two passes:

- Canonical global ordering cannot be replaced by independent symbol/factor
  order.
- A timestamp’s rows cannot be split/merged differently.
- PASS-fold aggregate means and sample standard deviation require all raw fold
  scores in canonical order.
- Final score fields cannot be emitted until those aggregates are known.
- Exact Parquet bytes cannot be promised from arbitrary incremental writers;
  the final writer/chunk layout must match the reference.

The full Factors/Labels panel does not need to be resident simultaneously.

## 25. Expected peak-memory behavior

The current 4h path reached 3,464.47 MB private before loading all symbols.
The bounded profiling run processed every timeframe symbol-by-symbol and
peaked below approximately 0.79 GB private despite the 1h lake having a
5.89 GB summed eager estimate.

Expected implementation target:

- 4h: approximately 0.5–1.0 GiB peak, a 70–85% reduction from the observed
  stop.
- 1h/15m/5m: bounded by the largest single projected symbol, sort budget, and
  final ledger frame rather than total lake size.

This is an engineering target, not a measured implementation result. A future
stress test must enforce it.

## 26. Expected runtime/I/O tradeoff

The bounded design increases:

- Parquet writes for symbol shards and sorted runs,
- sequential reads for fold and aggregate passes,
- external-sort merge work,
- cleanup I/O.

It decreases:

- allocator pressure,
- paging risk,
- repeated large hash/sort allocations,
- failure/retry cost.

The design should prefer sequential I/O and deterministic run sizes.
Increasing workers would multiply memory and is inappropriate on this host.

## 27. Required implementation files

Separate implementation approval would be required for:

- `src/cqros/walk_forward/memory_efficient.py` — execution config, spill
  lifecycle, sorted-run merge, fold executor.
- `src/cqros/walk_forward/evaluation_input.py` — mode dispatch and reusable
  per-symbol join/validation primitives.
- `src/cqros/walk_forward/engine.py` — expose/reuse exact fold metric and
  aggregate operations without changing formulas.
- `src/cqros/walk_forward/__init__.py` — public execution configuration API.
- `src/cqros/cli/generate_walk_forward.py` — explicit mode, spill parent,
  memory/run-size options and dependency wiring.
- Walk Forward documentation/CLI usage describing physical execution only.

No Factor Selection, Factor Validation, Purged CV, label, factor-generation,
market-data, policy, or downstream evaluation file needs modification.

## 28. Required tests for a future implementation

Required before promotion:

1. Full-panel versus memory-efficient `assert_frame_equal` on multi-symbol,
   multi-factor, repeated-timestamp data.
2. Exact output SHA-256 comparison using the same repository writer.
3. Current 1d 2026 reference comparison, including all floating bits and bytes.
4. Equal-timestamp tie-order regression.
5. Folds that split a timestamp cross-section.
6. Cross-symbol and cross-factor pooled test-return regression.
7. Selected and rejected factors both preserving row geometry.
8. Duplicate Factors, Labels, and FS key rejection parity.
9. Missing join and null-target parity.
10. Orientation metadata rejection parity.
11. Insufficient-history sentinel parity.
12. PASS/FAIL and aggregate overwrite parity.
13. External-sort run-size invariance.
14. Spill cleanup on success and every failure point.
15. Atomic-output preservation on failure.
16. Determinism across repeated runs.
17. Peak private-memory test below the configured budget.
18. CLI mode/config validation and no silent fallback.
19. Unit, integration, regression, type, lint, formatting, and coverage gates.

Investigation test command:

```text
uv run pytest tests/unit/walk_forward/test_engine.py \
  tests/unit/walk_forward/test_evaluation_input.py -q --tb=short
```

Result: exit code 0, all selected tests passed. Measured test body duration:
91.810 seconds. The earlier controlled-regeneration suite recorded 160 passed
in 202.45 seconds.

## 29. Production artifact verification

Investigation-start hashes were captured before profiling and checked again
after all read-only diagnostics.

Factor Selection:

| TF | SHA-256 |
|---|---|
| 1d | `B6CE50C27CAE6601FC0CED1CB650475BF471C337211D5C89D98B1FEB8432A2BB` |
| 4h | `BAE924AA39D5DDA300030052B5C69A55DA94DC230A30931ECFDE911BD1592918` |
| 1h | `89F17D02C7A4B6314A785DC8783836F88B233F9F04A83AD76D94E3AFEEC10022` |
| 15m | `898A40F41AF52452DB3D170C667DD6A2678CB7B9317C7A2D48CC7518DBE12788` |
| 5m | `1C80C6679DAA8133CED64C9D7978F02E4E9742CEBEECECB4DA4289F569F97FB0` |

Factor Validation:

| TF | SHA-256 |
|---|---|
| 1d | `B7935E021B31BAD5BE9017577FCD49243A1E022480A47F44A5B6D5D2C4058137` |
| 4h | `E49A86299CD989A8B1F5B91ABF90E92647993250A71351391EA2F9752F481EED` |
| 1h | `315EF539F46E94A7B28AB902D98E88D4B79EB7BD666687E5FCED6FF1B7C3CC30` |
| 15m | `57FEF604E3F24DA6FBEEC836D5ADCB2979452DAA906A5C60639DA89CCAC38CB2` |
| 5m | `7F81C68E92FD51058DA13F54CDBA3E8F2981357F69D7F4E75A7FFB9474DA213D` |

Walk Forward:

| TF | Bytes | Investigation-start SHA-256 |
|---|---:|---|
| 1d | 185,255 | `B2391EE4ECBCD89E145015A488789415C410F6A5C050A818192EAEB7DFE58469` |
| 4h | 141,704 | `888C4613FA4D3DD8D3345AF9F134D13DCAC2EF35641C073234EAB68E19C01C8F` |
| 1h | 470,948 | `735E0D62872F436A2BFA9A6A84E9EC5CBC4602703C5C7B43A4CA2462F2476373` |
| 15m | 226,656 | `97A8BA631707E8E74FBD13BC1577D5AE9EB9156A3F223B6D1D7F4D3882A73806` |
| 5m | 390,938 | `937C16B8A98BF5C0C7F95CEA80A9A8B2EAA6EE2828423298951057995730E22D` |

The task-supplied `1d` value
`87E74C6C...D228087` is the pre-controlled-regeneration baseline. The actual
file already had `B2391E...58469` when this investigation began, consistent
with the authoritative report that 1d regeneration succeeded. This
investigation did not cause that difference.

Purged CV:

| TF | SHA-256 |
|---|---|
| 1d | `C517DC596ABE5F397FF16B0713CCD8781B0DF8B23569BA1C1934F357496A15FB` |
| 4h | `2185AB07C048E97C64155BE3004E7DB7C1FFA3101FA39FCA58DFBC6EE46DD1D4` |
| 1h | `249A8119215FA02CBEAFE2D7E946412CD7441F9E456339A5613EB59D702F0767` |
| 15m | `7D6EF28BD26F4A3CAE4E15A3CDBB3DEE5F89CD1625CACFDF61E14723ADE7AE4E` |
| 5m | `15F3745A9CEAFE9C62076B38641C72C752A199D7765B3C7E9EB59FC41665FA26` |

Evidence files:

- `input_profile_clean.json`
- `join_cardinality.json`
- `artifact_hashes_after.json`
- `profile_inputs.py`
- `profile_join_cardinality.py`
- `verify_artifact_hashes.py`

## 30. Explicit STOP statement

STOP. No fix was implemented. No production source, test, policy,
configuration, Factor Selection, Factor Validation, Factors, Labels, market
data, Walk Forward, Purged CV, evaluation, stability, prediction, signal,
alpha, or regime artifact was regenerated or modified.

The next authorized task, if approved, is a separate implementation and
equivalence-validation task for the bounded-memory architecture above.

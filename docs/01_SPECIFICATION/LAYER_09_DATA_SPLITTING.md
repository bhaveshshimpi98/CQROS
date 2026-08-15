# Layer 09 – Data Splitting Specification

**Layer ID:** L09

**Layer Name:** Data Splitting

**Version:** 1.0.0

**Status:** Draft

**Dependencies**

- Layer 00 – Foundation
- Layer 05 – Metadata & Lineage
- Layer 06 – Dataset Builder
- Layer 07 – Feature Engineering
- Layer 08 – Target Engineering

**Required By**

- Layer 10 – Model Training
- Layer 11 – Feature Selection
- Layer 12 – Model Evaluation
- Layer 21 – Experiment Tracking

---

# 1. Purpose

The Data Splitting layer prepares research-ready datasets for machine
learning by creating statistically valid train, validation, and test
partitions while preventing information leakage.

Every split produced by CQROS must be deterministic,
reproducible, versioned, and fully auditable.

---

# 2. Responsibilities

This layer owns

- Dataset partitioning
- Train/Validation/Test splits
- Walk-forward validation
- Rolling windows
- Expanding windows
- Time-aware cross validation
- Purged cross validation
- Embargo periods
- Split versioning
- Split metadata
- Leakage prevention

---

# 3. Out of Scope

Layer 09 never performs

- Feature computation
- Target generation
- Model training
- Hyperparameter tuning
- Portfolio optimization
- Trading

---

# 4. Splitting Pipeline

```
Research Dataset

↓

Feature Matrix

↓

Target Matrix

↓

Split Strategy

↓

Leakage Validation

↓

Train

Validation

Test

↓

Metadata

↓

Published Split
```

---

# 5. Split Types

Supported split strategies

- Holdout
- Chronological
- Rolling Window
- Expanding Window
- Walk Forward
- Time Series Cross Validation
- Purged K-Fold
- Purged Walk Forward
- Nested Cross Validation

---

# 6. Package Structure

```
src/cqros/splitting/

interfaces.py

models.py

service.py

engine.py

factory.py

registry.py

metadata.py

validators.py

config.py

exceptions.py

strategies/

walkforward/

rolling/

expanding/

purged/

embargo/

cross_validation/

tests/
```

---

# 7. Public Interfaces

```
ISplitStrategy

ISplittingEngine

ISplitRegistry

ISplitValidator

ISplitPublisher
```

---

# 8. Holdout Split

Support configurable

Training %

Validation %

Testing %

Chronological ordering is mandatory.

Random shuffling is forbidden by default.

---

# 9. Walk-Forward Validation

Support

Multiple folds

Rolling retraining

Fixed horizon

Variable horizon

Gap between windows

Overlapping windows

Non-overlapping windows

---

# 10. Rolling Window

Support

Fixed-size training window

Fixed-size testing window

Sliding windows

Configurable overlap

Window metadata

---

# 11. Expanding Window

Support

Growing training window

Fixed testing window

Continuous retraining

Historical accumulation

---

# 12. Purged Cross Validation

Remove observations that could introduce leakage.

Support

Purge period

Embargo period

Overlap removal

Future dependency elimination

Required for financial time-series.

---

# 13. Embargo

Support configurable embargo

Time-based

Row-based

Event-based

Dynamic embargo

Embargo metadata is stored.

---

# 14. Leakage Prevention

Mandatory checks

Future timestamps

Future features

Future labels

Window overlap

Target contamination

Cross-fold contamination

Improper normalization

Leakage detection blocks publication.

---

# 15. Split Metadata

Every split records

Split ID

Version

Strategy

Parameters

Dataset version

Feature version

Target version

Creation timestamp

Random seed (if applicable)

Window definitions

Embargo settings

Checksum

---

# 16. Configuration

Configuration includes

Split strategy

Training ratio

Validation ratio

Testing ratio

Window size

Embargo

Purge

Fold count

Random seed

Chronological enforcement

Publishing options

---

# 17. Validation

Validate

Chronological ordering

Dataset completeness

Class balance

Regression distribution

Fold consistency

Window boundaries

Leakage

Metadata completeness

Checksum integrity

---

# 18. Error Handling

Exceptions

SplitError

SplitValidationError

LeakageError

WindowError

EmbargoError

PurgeError

ConfigurationError

PublishingError

---

# 19. Logging

Log

Split generation

Strategy selected

Fold generation

Leakage detection

Execution duration

Validation

Publishing

Warnings

---

# 20. Security

Support

Immutable split artifacts

Checksums

Audit trail

Version history

Future

Digital signatures

Access control

---

# 21. Performance

Support

Million-row datasets

Parallel fold generation

Memory-efficient slicing

Streaming datasets

Incremental updates

Large multi-symbol datasets

---

# 22. Thread Safety

Split engine

Concurrent-safe

Registry

Read-safe

Configuration

Immutable

Validators

Stateless

---

# 23. Monitoring

Expose

Splits generated

Leakage detections

Execution time

Fold generation time

Validation failures

Registry size

Memory usage

---

# 24. Dependency Rules

Allowed

```
Data Splitting

↓

Foundation

↓

Metadata

↓

Dataset Builder

↓

Feature Engineering

↓

Target Engineering
```

Forbidden

```
Data Splitting

↓

Model Training

↓

Portfolio

↓

Execution
```

---

# 25. Testing

Coverage

100%

Tests

Holdout

Rolling window

Expanding window

Walk-forward

Purged CV

Embargo

Leakage detection

Metadata

Performance

Concurrency

Regression tests

---

# 26. Deliverables

```
splitting/

interfaces.py

models.py

service.py

engine.py

factory.py

registry.py

metadata.py

validators.py

config.py

exceptions.py

strategies/

walkforward/

rolling/

expanding/

purged/

embargo/

cross_validation/

tests/
```

---

# 27. Acceptance Criteria

✓ Holdout splitting operational

✓ Walk-forward validation operational

✓ Rolling windows verified

✓ Expanding windows verified

✓ Purged CV implemented

✓ Embargo enforcement operational

✓ Leakage detection verified

✓ Metadata captured

✓ Versioning operational

✓ Performance targets achieved

✓ Unit tests pass

✓ Integration tests pass

✓ Documentation complete

---

# 28. Future Extensions

Future enhancements

- Adaptive window sizing
- Regime-aware splitting
- Distributed fold generation
- Online validation
- Cross-market synchronized splits
- GPU-accelerated slicing
- Nested walk-forward optimization
- Probabilistic sampling strategies
- Multi-objective validation workflows

---

# 29. Summary

The Data Splitting layer prepares feature and target datasets for
machine learning by generating deterministic, leakage-free, and
time-aware train, validation, and test partitions.

It implements institutional-grade financial validation techniques,
including walk-forward validation, rolling and expanding windows,
purged cross-validation, and embargo periods, ensuring statistically
sound model development and reproducible research across the CQROS
platform.
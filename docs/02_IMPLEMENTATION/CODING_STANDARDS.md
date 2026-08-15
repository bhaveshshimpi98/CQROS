# CQROS Coding Standards

Version: 1.0.0

Status: Active

Applies To

- All source code
- Tests
- Scripts
- Documentation examples
- Configuration
- CI/CD workflows

---

# 1. Purpose

This document defines the engineering standards for CQROS.

Every contribution must follow these rules to ensure the codebase remains

- Consistent
- Maintainable
- Testable
- Extensible
- Type-safe
- Production-ready

---

# 2. Core Principles

CQROS follows

- SOLID
- Clean Architecture
- Domain Driven Design
- Dependency Injection
- Composition over inheritance
- Explicit is better than implicit
- Immutable data where possible
- Fail fast
- Test-first mindset

---

# 3. Python Version

Required

Python 3.13+

Never support older versions unless explicitly approved.

---

# 4. Type Hints

Required

Every public function must be typed.

Every class attribute must be typed.

Avoid

```
Any
```

unless absolutely necessary.

Example

Good

```python
def calculate_return(prices: list[float]) -> float:
    ...
```

Bad

```python
def calculate_return(prices):
    ...
```

---

# 5. Imports

Order

1 Standard Library

2 Third-party

3 CQROS packages

Example

```python
from pathlib import Path

import numpy as np
import pandas as pd

from cqros.storage import DatasetStore
```

Use

isort

Never use wildcard imports.

Bad

```python
from numpy import *
```

---

# 6. Naming Conventions

Packages

snake_case

Modules

snake_case

Classes

PascalCase

Interfaces

Prefix with I

Example

```
IDataStore
IStrategy
```

Functions

snake_case

Variables

snake_case

Constants

UPPER_CASE

Private members

Prefix

```
_
```

---

# 7. File Organization

Preferred order

```python
Imports

Constants

Type aliases

Dataclasses

Enums

Exceptions

Interfaces

Implementation

Helper functions
```

---

# 8. Class Design

One responsibility per class.

Keep constructors lightweight.

Inject dependencies.

Avoid global state.

Avoid singleton abuse.

Prefer composition.

---

# 9. Function Design

Functions should

- Do one thing
- Be deterministic
- Be side-effect aware
- Be easy to test

Target length

20–40 lines

Maximum

80 lines

Split larger functions.

---

# 10. Error Handling

Raise explicit exceptions.

Good

```python
raise ValidationError(...)
```

Bad

```python
raise Exception(...)
```

Never silently ignore exceptions.

Bad

```python
except:
    pass
```

---

# 11. Logging

Use the CQROS logging framework.

Log

- Startup
- Shutdown
- Errors
- Warnings
- Important state changes

Never log

- Passwords
- Tokens
- Secrets
- API keys

---

# 12. Configuration

Never hardcode

- Paths
- Credentials
- URLs
- Limits

Read from configuration.

---

# 13. Immutability

Prefer

```python
@dataclass(frozen=True)
```

for value objects.

Avoid mutable shared state.

---

# 14. Dependency Injection

Never instantiate dependencies inside business logic.

Bad

```python
class Trainer:

    def __init__(self):

        self.store = DatasetStore()
```

Good

```python
class Trainer:

    def __init__(

        self,

        store: IDatasetStore,

    ):

        self.store = store
```

---

# 15. Interfaces

Every service must depend on interfaces.

Example

```
IDataStore

IModelRegistry

IStrategy
```

Avoid depending directly on implementations.

---

# 16. Dataclasses

Prefer dataclasses for immutable models.

Example

```python
@dataclass(frozen=True)

class Dataset:

    id: str

    rows: int
```

---

# 17. Enums

Replace magic strings with enums.

Bad

```python
status = "running"
```

Good

```python
class JobStatus(Enum):

    RUNNING = "running"
```

---

# 18. Testing

Every module requires

Unit tests

Integration tests

Regression tests (when applicable)

Public APIs must be tested.

---

# 19. Test Naming

Pattern

```
test_<behavior>()
```

Examples

```
test_load_dataset()

test_create_order()

test_validate_schema()
```

---

# 20. Documentation

Every public class

Docstring

Every public function

Docstring

Every module

Purpose description

---

# 21. Comments

Explain

WHY

not

WHAT

Bad

```python
# Increment i

i += 1
```

Good

```python
# Retry because exchange timestamps may arrive out of order.
```

---

# 22. Formatting

Use

Black

Maximum line length

88

No manual alignment.

---

# 23. Linting

Required

Ruff

No warnings allowed.

---

# 24. Static Analysis

Required

Pyright

No type errors.

---

# 25. Complexity

Maximum cyclomatic complexity

10

Split complex logic into smaller functions.

---

# 26. Performance

Measure before optimizing.

Avoid premature optimization.

Profile critical paths.

---

# 27. Security

Never commit

- Secrets
- API keys
- Passwords
- Tokens

Always validate external input.

Escape user-controlled output where applicable.

---

# 28. Git

Commit messages

```
feat:

fix:

docs:

test:

refactor:

perf:

build:

ci:

chore:
```

Example

```
feat(storage): add parquet dataset registry
```

---

# 29. Pull Requests

Each PR should

- Solve one problem
- Include tests
- Update documentation
- Pass CI

Avoid mixing unrelated changes.

---

# 30. Definition of Done

Code is complete only if

✓ Tests pass

✓ Ruff passes

✓ Black passes

✓ isort passes

✓ Pyright passes

✓ Documentation updated

✓ Interfaces implemented

✓ Reviewed

✓ CI passes

---

# 31. Project Structure

```
src/

tests/

docs/

scripts/

configs/

examples/
```

Never place production code outside

```
src/
```

---

# 32. Architecture Rules

Dependencies must always point inward.

Presentation

↓

Application

↓

Domain

↓

Infrastructure

Never violate dependency direction.

---

# 33. Continuous Integration

Every push must execute

- Ruff
- Black
- isort
- Pyright
- pytest
- Coverage

Merge only after all checks succeed.

---

# 34. Future Standards

Future additions may include

- Security scanning
- Mutation testing
- Benchmark validation
- Contract testing
- API compatibility checks
- Supply-chain verification

---

# 35. Summary

These coding standards establish a consistent engineering
foundation for CQROS.

Following these standards ensures the project remains
maintainable, scalable, type-safe, testable, and suitable
for long-term institutional-grade development.
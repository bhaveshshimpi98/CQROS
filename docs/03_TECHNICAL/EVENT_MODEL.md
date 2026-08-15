# CQROS Event Model

Version: 1.0.0

Status: Active

---

# 1. Purpose

This document defines the event-driven architecture of CQROS.

Every subsystem communicates through strongly typed
domain events.

The event model provides

- Loose coupling
- Scalability
- Deterministic workflows
- Auditability
- Replay capability

---

# 2. Event Architecture

```
Publisher

↓

Event Bus

↓

Subscribers

↓

Event Handlers

↓

State Updates
```

Business services never communicate directly when an
event can be used.

---

# 3. Event Categories

Infrastructure

- Startup
- Shutdown
- Configuration
- Health

Market

- MarketDataReceived
- CandleClosed
- TradeReceived
- OrderBookUpdated
- FundingUpdated
- OpenInterestUpdated

Research

- DatasetCreated
- DatasetValidated
- DatasetVersioned
- FeatureGenerationStarted
- FeaturesGenerated
- TargetGenerated

Machine Learning

- TrainingStarted
- TrainingCompleted
- EvaluationCompleted
- ModelRegistered
- ModelPromoted
- ModelArchived

Trading

- SignalGenerated
- SignalRejected
- RiskApproved
- RiskRejected
- OrderSubmitted
- OrderAccepted
- OrderFilled
- OrderCancelled
- PositionOpened
- PositionClosed
- PortfolioUpdated

Operations

- DeploymentStarted
- DeploymentCompleted
- HealthCheckFailed
- AlertCreated
- IncidentOpened
- IncidentResolved

---

# 4. Base Event Schema

Every event contains

```
event_id

event_type

event_version

timestamp

correlation_id

causation_id

source

payload
```

---

# 5. Event Metadata

Fields

- event_id
- correlation_id
- causation_id
- producer
- environment
- schema_version
- created_at

Metadata must remain immutable.

---

# 6. Event Lifecycle

```
Create

↓

Validate

↓

Publish

↓

Dispatch

↓

Process

↓

Acknowledge

↓

Archive
```

---

# 7. Market Events

## MarketDataReceived

Published by

Exchange Client

Subscribers

- Storage
- Feature Engine
- Live Trading

Payload

- symbol
- timestamp
- market_data

---

## CandleClosed

Published by

Candle Aggregator

Subscribers

- Feature Engine
- Strategy Engine
- Dataset Builder

---

## OrderBookUpdated

Published by

Order Book Stream

Subscribers

- Liquidity Features
- Execution Engine

---

## FundingUpdated

Subscribers

- Feature Engineering
- Strategy Engine

---

## OpenInterestUpdated

Subscribers

- Feature Engineering
- Regime Detection

---

# 8. Research Events

DatasetCreated

↓

DatasetValidated

↓

FeaturesGenerated

↓

TargetsGenerated

↓

DatasetPublished

---

# 9. Machine Learning Events

TrainingStarted

↓

TrainingCompleted

↓

EvaluationCompleted

↓

ModelRegistered

↓

ModelPromoted

---

# 10. Trading Events

SignalGenerated

↓

RiskApproved

↓

OrderSubmitted

↓

OrderAccepted

↓

OrderFilled

↓

PositionOpened

↓

PortfolioUpdated

---

# 11. Risk Events

RiskApproved

Published by

Risk Engine

Subscribers

Execution Engine

---

RiskRejected

Published by

Risk Engine

Subscribers

Strategy Engine

Logging

Alerts

---

# 12. Portfolio Events

PositionOpened

PositionClosed

PortfolioUpdated

CashChanged

ExposureChanged

---

# 13. Deployment Events

DeploymentStarted

DeploymentCompleted

RollbackStarted

RollbackCompleted

---

# 14. Monitoring Events

HealthCheckFailed

MetricThresholdExceeded

AlertCreated

AlertResolved

IncidentOpened

IncidentClosed

---

# 15. Event Ordering

Ordering is guaranteed

per aggregate.

Global ordering is not required.

---

# 16. Delivery Guarantees

Internal events

At-least-once delivery

Handlers must therefore be idempotent.

---

# 17. Idempotency

Every event includes

```
event_id
```

Duplicate processing must be ignored safely.

---

# 18. Event Versioning

Schema changes require

```
event_version
```

Older consumers continue to process supported versions.

---

# 19. Event Validation

Before publication verify

- Required fields
- Payload schema
- Event version
- Timestamp
- Correlation ID

---

# 20. Error Handling

Handler failure

↓

Retry

↓

Dead Letter Queue

↓

Alert

↓

Manual Investigation

---

# 21. Dead Letter Queue

Failed events move to

```
DLQ
```

Metadata retained

- reason
- retries
- timestamp
- handler

---

# 22. Event Replay

Stored events support

- debugging
- recovery
- rebuilding projections
- research reproducibility

Replay never modifies original events.

---

# 23. Correlation

Long-running workflows share

```
correlation_id
```

Example

TrainingStarted

↓

FeaturesGenerated

↓

TrainingCompleted

↓

ModelRegistered

All share one correlation ID.

---

# 24. Event Naming

Pattern

```
<Aggregate><PastTenseVerb>
```

Examples

DatasetCreated

OrderFilled

ModelRegistered

SignalGenerated

PortfolioUpdated

---

# 25. Publisher Rules

Publishers

- own the event
- validate payload
- never mutate events
- publish once

---

# 26. Subscriber Rules

Subscribers

- remain idempotent
- avoid side effects
- fail gracefully
- acknowledge completion

---

# 27. Performance Targets

Publish latency

<5 ms

Dispatch latency

<10 ms

Processing latency

Application specific

---

# 28. Security

Events must never contain

- passwords
- API keys
- private secrets
- authentication tokens

Sensitive identifiers should be masked where appropriate.

---

# 29. Testing

Every event requires

- schema validation
- serialization test
- replay test
- idempotency test
- handler test

---

# 30. Summary

CQROS uses an event-driven architecture to decouple
services while maintaining deterministic workflows,
high observability, and reliable processing.

Strongly typed events, versioned schemas, correlation
identifiers, replay capability, and idempotent handlers
provide a robust communication model suitable for
institutional-grade quantitative research and trading.
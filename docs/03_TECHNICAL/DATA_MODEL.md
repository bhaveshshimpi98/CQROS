# CQROS Data Model

Version: 1.0.0

Status: Active

---

# 1. Purpose

This document defines the canonical domain model for CQROS.

Every object exchanged between layers must conform to
these models.

The data model ensures

- Consistency
- Type safety
- Versioning
- Auditability
- Reproducibility

---

# 2. Design Principles

The CQROS data model follows

- Immutable value objects
- Strong typing
- Unique identifiers
- Explicit timestamps
- Versioned entities
- Composition over inheritance

---

# 3. Entity Categories

Infrastructure

- Configuration
- Service
- Metadata

Market

- Exchange
- Asset
- Symbol
- Candle
- Trade
- OrderBook

Research

- Dataset
- Feature
- FeatureSet
- Target
- Experiment

Machine Learning

- Model
- Prediction
- Evaluation

Trading

- Signal
- Order
- Fill
- Position
- Portfolio

Operations

- Deployment
- AuditEvent
- Metric
- Alert

---

# 4. Asset

Represents a tradeable asset.

Fields

- asset_id
- symbol
- base_asset
- quote_asset
- exchange
- contract_type
- tick_size
- step_size
- status

Relationships

Exchange

↓

Asset

---

# 5. Exchange

Represents a trading venue.

Fields

- exchange_id
- name
- timezone
- api_version
- supported_markets

---

# 6. Candle

Represents one OHLCV record.

Fields

- symbol
- timeframe
- open_time
- close_time
- open
- high
- low
- close
- volume
- quote_volume
- trade_count

Primary Key

(symbol, timeframe, open_time)

---

# 7. Trade

Represents a market trade.

Fields

- trade_id
- symbol
- timestamp
- side
- price
- quantity
- buyer_maker

---

# 8. Order Book Snapshot

Fields

- symbol
- timestamp
- bids
- asks
- checksum

---

# 9. Funding Rate

Fields

- symbol
- funding_time
- funding_rate
- mark_price

---

# 10. Open Interest

Fields

- symbol
- timestamp
- open_interest
- value

---

# 11. Dataset

Research dataset.

Fields

- dataset_id
- version
- created_at
- rows
- columns
- checksum
- lineage

Relationships

Dataset

↓

FeatureSet

↓

Target

---

# 12. Feature

Represents one engineered feature.

Fields

- feature_id
- name
- dtype
- source
- version
- description

---

# 13. Feature Set

Fields

- feature_set_id
- dataset_id
- version
- features
- created_at

---

# 14. Target

Fields

- target_id
- horizon
- target_type
- definition
- dataset_id

---

# 15. Experiment

Fields

- experiment_id
- name
- author
- parameters
- metrics
- artifacts
- created_at

---

# 16. Model

Fields

- model_id
- version
- algorithm
- training_dataset
- feature_set
- metrics
- created_at

---

# 17. Prediction

Fields

- prediction_id
- model_id
- timestamp
- symbol
- probability
- prediction
- confidence

---

# 18. Evaluation

Fields

- evaluation_id
- model_id
- metrics
- validation_split
- timestamp

---

# 19. Signal

Represents a trade recommendation.

Fields

- signal_id
- timestamp
- symbol
- direction
- confidence
- entry_price
- stop_loss
- take_profit
- strategy
- regime

---

# 20. Order

Represents a broker order.

Fields

- order_id
- broker_order_id
- symbol
- side
- type
- quantity
- price
- status
- created_at

---

# 21. Fill

Represents an execution.

Fields

- fill_id
- order_id
- quantity
- price
- commission
- timestamp

---

# 22. Position

Fields

- position_id
- symbol
- quantity
- average_price
- unrealized_pnl
- realized_pnl

---

# 23. Portfolio

Fields

- portfolio_id
- equity
- cash
- margin
- positions
- exposure
- leverage

---

# 24. Risk Snapshot

Fields

- snapshot_id
- timestamp
- portfolio_value
- drawdown
- exposure
- value_at_risk
- margin_usage

---

# 25. Deployment

Fields

- deployment_id
- version
- environment
- git_commit
- build_time
- deployed_at

---

# 26. Audit Event

Fields

- event_id
- timestamp
- actor
- action
- resource
- checksum

---

# 27. Metric

Fields

- metric_id
- timestamp
- name
- value
- labels

---

# 28. Alert

Fields

- alert_id
- severity
- source
- message
- created_at
- resolved_at

---

# 29. Relationships

```
Exchange

↓

Asset

↓

Market Data

↓

Dataset

↓

Feature Set

↓

Model

↓

Prediction

↓

Signal

↓

Order

↓

Fill

↓

Position

↓

Portfolio
```

---

# 30. Entity Rules

Every entity must

- Have a unique identifier
- Include timestamps where applicable
- Be serializable
- Be versioned when mutable
- Support validation

---

# 31. Serialization

Supported formats

- JSON
- Parquet
- Arrow
- MessagePack (optional)

---

# 32. Validation

Every entity validates

- Required fields
- Types
- Value ranges
- Referential integrity
- Version compatibility

---

# 33. Versioning

Version

- Datasets
- Features
- Models
- Experiments
- Deployments

Historical versions remain immutable.

---

# 34. Summary

The CQROS data model provides a unified, strongly typed,
versioned representation of every domain entity used
throughout the platform. It establishes the contracts
between layers and serves as the foundation for reliable,
auditable, and reproducible research and trading workflows.